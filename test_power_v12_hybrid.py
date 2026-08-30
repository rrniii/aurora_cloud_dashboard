from __future__ import annotations

import unittest
import json
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr

from ecmwf_forecast_provider import ForecastProviderResult
from generate_power_soc_forecast import (
    _apply_candidate_load_residual,
    _archive_row_from_forecast,
    _filter_active_forecast_contract,
    apply_forecast_identity,
    generate,
)
from generate_power_soc_v12_candidate import run_candidate
from power_load_dynamics import ControlledLoadProfile
from power_v12_hybrid import (
    build_campaign_evidence,
    campaign_score_surfaces,
    evaluation_contract_from_forecast,
    fit_bounded_load_residual,
)


CONFIG_PATH = Path(__file__).with_name("config") / "power_solar_physical_candidate_v1.json"


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _archive_with_load_history() -> xr.Dataset:
    issues = pd.date_range("2026-06-01", periods=4, freq="1D")
    steps = np.arange(16, dtype=np.int32)
    valid = np.asarray(
        [issue + pd.to_timedelta(np.arange(1, 17), unit="h") for issue in issues],
        dtype="datetime64[ns]",
    )
    return xr.Dataset(
        {
            "ForecastLoadWatts": (("issue_time", "forecast_step"), np.full((4, 16), 100.0)),
            "ForecastValidTime": (("issue_time", "forecast_step"), valid),
            "ForecastLeadHours": (("issue_time", "forecast_step"), np.tile(np.arange(1, 17), (4, 1))),
            "ECMWFCycleTime": (("issue_time",), issues.to_numpy(dtype="datetime64[ns]")),
            "LoadMode": (("issue_time",), np.asarray(["DC-Only"] * 4, dtype="U32")),
            "ForecastModelContractID": (("issue_time",), np.asarray(["v10"] * 4, dtype="U32")),
            "ForecastSystemVersion": (("issue_time",), np.asarray(["v10-control"] * 4, dtype="U32")),
        },
        coords={"issue_time": issues.to_numpy(dtype="datetime64[ns]"), "forecast_step": steps},
    )


class HybridCandidateTests(unittest.TestCase):
    def test_load_residual_cannot_erase_observed_dc_core_floor(self) -> None:
        times = pd.date_range("2026-06-01", periods=3, freq="1h")
        profile = ControlledLoadProfile(
            np.asarray([280.0, 300.0, 320.0]),
            np.asarray([300.0, 320.0, 340.0]),
            np.asarray([340.0, 360.0, 380.0]),
            np.zeros(3, dtype=np.int8),
            "test",
        )
        residual = {
            "status": "active",
            "contract_id": "test-contract",
            "p10_correction_w": pd.Series([-500.0] * 3, index=times),
            "p50_correction_w": pd.Series([-500.0] * 3, index=times),
            "p90_correction_w": pd.Series([-500.0] * 3, index=times),
            "training_samples": 48,
            "training_cycles": 3,
            "training_days": 3,
            "bound_w": 500.0,
            "selection": "test",
        }
        adjusted, diagnostics = _apply_candidate_load_residual(
            profile,
            times,
            residual,
            physical_floor_w=244.0,
            physical_floor_source="dc_only_state_registry",
        )
        np.testing.assert_array_equal(adjusted.p10_w, np.full(3, 244.0))
        np.testing.assert_array_equal(adjusted.p50_w, np.full(3, 244.0))
        np.testing.assert_array_equal(adjusted.p90_w, np.full(3, 244.0))
        self.assertEqual(diagnostics["load_residual_physical_floor_w"], 244.0)
        self.assertEqual(diagnostics["load_residual_physical_floor_source"], "dc_only_state_registry")

    def test_load_residual_excludes_future_issue_and_observation_rows(self) -> None:
        archive = _archive_with_load_history()
        times = pd.date_range("2026-06-01", periods=5 * 24, freq="1h")
        load = np.full(len(times), 200.0)
        # A radically different fourth issue/future observation must not leak
        # into a 4 June issue-time fit.
        load[times >= pd.Timestamp("2026-06-04")] = 0.0
        power = xr.Dataset(
            {
                "ACOutputWatts": (("time",), load),
                "DCInverterWatts": (("time",), np.zeros(len(times))),
            },
            coords={"time": times},
        )
        fit = fit_bounded_load_residual(
            archive,
            power,
            issue_time="2026-06-04T00:00:00",
            forecast_times=pd.date_range("2026-06-04", periods=3, freq="3h"),
            load_mode="DC-Only",
            control_forecast_model_contract_id="v10",
            control_forecast_system_version="v10-control",
        )
        self.assertEqual(fit.status, "active")
        self.assertEqual(fit.training_samples, 48)
        self.assertAlmostEqual(float(fit.p50_correction_w.iloc[0]), 50.0, delta=1.0)

    def test_load_residual_fails_closed_for_another_control_contract(self) -> None:
        archive = _archive_with_load_history()
        times = pd.date_range("2026-06-01", periods=5 * 24, freq="1h")
        power = xr.Dataset(
            {
                "ACOutputWatts": (("time",), np.full(len(times), 200.0)),
                "DCInverterWatts": (("time",), np.zeros(len(times))),
            },
            coords={"time": times},
        )
        fit = fit_bounded_load_residual(
            archive,
            power,
            issue_time="2026-06-04T00:00:00",
            forecast_times=pd.date_range("2026-06-04", periods=3, freq="3h"),
            load_mode="DC-Only",
            control_forecast_model_contract_id="different-control",
            control_forecast_system_version="v10-control",
        )
        self.assertTrue(fit.status.startswith("insufficient_issue_time_evidence"))
        self.assertTrue(np.allclose(fit.p50_correction_w.values, 0.0))

    def test_v12_identity_round_trips_to_archive_rows(self) -> None:
        issue = pd.Timestamp("2026-06-21T12:00:00")
        forecast = xr.Dataset(
            {"BatterySOCForecast": (("time",), [70.0, 69.0])},
            coords={"time": [issue, issue + pd.Timedelta(hours=3)]},
            attrs={
                "initial_soc_time": issue.isoformat(),
                "forecast_model_contract_id": "base-contract",
                "forecast_model_name": "candidate",
                "forecast_model_version": "12",
                "input_snapshot_id": "sha256:test",
            },
        )
        apply_forecast_identity(
            forecast,
            {
                "forecast_system_version": "power-v12-hybrid-candidate",
                "feature_set_version": "test-features",
                "feature_set_digest": "feature-digest",
                "training_cutoff_utc": issue.isoformat(),
                "forecast_code_revision": "test-revision",
                "source_cycle_set_id": "cycle-test",
                "source_manifest_digest": "manifest-test",
                "degraded_mode_code": "none",
                "candidate_lane": "D_hybrid",
                "local_feature_contract_id": "issue-features-test",
                "baseline_control_contract_id": "baseline-control",
                "baseline_control_system_version": "v10-control",
                "source_availability_code": "ecmwf_control=available",
            },
        )
        archived = _archive_row_from_forecast(forecast)
        self.assertTrue(str(archived["ForecastIdentityID"].values[0]).startswith("forecast-identity-v1-"))
        self.assertEqual(str(archived["ForecastSystemVersion"].values[0]), "power-v12-hybrid-candidate")
        self.assertEqual(str(archived["CandidateLane"].values[0]), "D_hybrid")
        self.assertEqual(str(archived["LocalFeatureContractID"].values[0]), "issue-features-test")
        self.assertEqual(str(archived["BaselineControlContractID"].values[0]), "baseline-control")

    def test_active_evaluation_filter_keeps_the_complete_system_version(self) -> None:
        archive = xr.Dataset(
            {
                "ForecastModelContractID": (("issue_time",), ["shared-contract", "shared-contract"]),
                "ForecastSystemVersion": (("issue_time",), ["v10", "v12"]),
            },
            coords={"issue_time": pd.date_range("2026-06-01", periods=2, freq="1D")},
        )
        table = pd.DataFrame(
            {
                "forecast_model_contract_id": ["shared-contract", "shared-contract"],
                "forecast_system_version": ["v10", "v12"],
                "error": [99.0, 1.0],
            }
        )
        filtered = _filter_active_forecast_contract(table, archive)
        self.assertEqual(filtered["forecast_system_version"].tolist(), ["v12"])
        self.assertEqual(filtered["error"].tolist(), [1.0])

    def test_v12_runner_writes_isolated_immutable_lanes(self) -> None:
        power_times = pd.date_range("2026-06-21T00:00:00", periods=25, freq="1h")
        power = xr.Dataset(
            {
                "BatterySOC": (("time",), np.linspace(72.0, 70.0, len(power_times))),
                "SolarWatts_East": (("time",), np.zeros(len(power_times))),
                "SolarWatts_South": (("time",), np.zeros(len(power_times))),
                "SolarWatts_West": (("time",), np.zeros(len(power_times))),
                "ACOutputWatts": (("time",), np.full(len(power_times), 20.0)),
                "DCInverterWatts": (("time",), np.full(len(power_times), 10.0)),
            },
            coords={"time": power_times},
        )
        forecast_times = pd.date_range(power_times[-1], periods=6, freq="3h")
        solar = xr.Dataset(
            {"ssrd": (("time",), np.cumsum(np.asarray([0.0, 150.0, 350.0, 550.0, 300.0, 75.0]) * 3.0 * 3600.0))},
            coords={"time": forecast_times},
        )
        provider = ForecastProviderResult(
            solar,
            {"effective_provider": "legacy", "fallback_reason": "", "requested_provider": "legacy"},
        )
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            power_path = root / "source" / "power.zarr"
            input_path = root / "source" / "forcing.grib2"
            baseline_path = root / "baseline" / "power_soc_forecast.zarr"
            archive_path = root / "baseline" / "power_soc_forecast_archive.zarr"
            power.to_zarr(power_path, mode="w", consolidated=True)
            input_path.write_bytes(b"v12 synthetic forcing")
            with patch("generate_power_soc_forecast.open_provider_solar_forecast", return_value=provider):
                generate(
                    power_zarr=power_path,
                    pdu_zarr=root / "source" / "missing-pdu.zarr",
                    output_zarr=baseline_path,
                    input_forecast=input_path,
                    state_path=root / "baseline" / "state.json",
                    archive_zarr=archive_path,
                    skill_zarr=None,
                    hindcast_zarr=None,
                    latitude=64.829694,
                    longitude=-23.248139,
                    horizon_hours=15,
                    provider="legacy",
                    max_power_age_minutes=None,
                    archive_forecast=True,
                    state_override={"solar_calibration_factor_w_per_wm2": 8.0},
                )
                baseline_digest = _tree_digest(baseline_path)
                archive_digest = _tree_digest(archive_path)
                results = run_candidate(
                    baseline_forecast_zarr=baseline_path,
                    baseline_archive_zarr=archive_path,
                    candidate_root=root / "candidate",
                    power_zarr=power_path,
                    pdu_zarr=root / "source" / "missing-pdu.zarr",
                    physical_config=CONFIG_PATH,
                )
                repeated = run_candidate(
                    baseline_forecast_zarr=baseline_path,
                    baseline_archive_zarr=archive_path,
                    candidate_root=root / "candidate",
                    power_zarr=power_path,
                    pdu_zarr=root / "source" / "missing-pdu.zarr",
                    physical_config=CONFIG_PATH,
                )
            self.assertEqual(set(results), {"B_physical_solar", "C_load_residual", "D_physical_solar_load_residual"})
            self.assertEqual(results, repeated)
            self.assertEqual(_tree_digest(baseline_path), baseline_digest)
            self.assertEqual(_tree_digest(archive_path), archive_digest)
            status = json.loads((root / "candidate" / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "complete")
            acceptance = json.loads((root / "candidate" / "acceptance_record.json").read_text(encoding="utf-8"))
            review = json.loads((root / "candidate" / "review_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(acceptance["status"], "not_accepted")
            self.assertEqual(review["status"], "pending_campaign_review")
            self.assertEqual(status["promotion_gates"]["status"], "not_eligible")
            with xr.open_zarr(results["C_load_residual"], chunks={}) as candidate:
                np.testing.assert_allclose(
                    candidate["ForecastSolarWatts"].values,
                    xr.open_zarr(baseline_path, chunks={})["ForecastSolarWatts"].values,
                    rtol=0.0,
                    atol=0.0,
                    equal_nan=True,
                )
                self.assertEqual(candidate.attrs["solar_forcing_mode"], "baseline_legacy_trace_replayed")
            with xr.open_zarr(results["D_physical_solar_load_residual"], chunks={}) as candidate:
                self.assertEqual(candidate.attrs["forecast_system_version"], "power-v12-hybrid-candidate")
                self.assertEqual(candidate.attrs["power_input_history_days"], "21")
                pair_id = candidate.attrs["evaluation_pair_id"]
                signature = candidate.attrs["publication_signature"]
            manifest_path = (
                root
                / "candidate"
                / "lanes"
                / "D_physical_solar_load_residual"
                / "pairs"
                / pair_id
                / signature
                / "pair_manifest.json"
            )
            self.assertTrue(manifest_path.exists())
            self.assertEqual(json.loads(manifest_path.read_text(encoding="utf-8"))["pair_status"], "complete")
            source_manifest = next((root / "candidate" / "source_manifests").glob("*.json"))
            manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
            self.assertIn("issue_time_features", manifest)
            self.assertIn("public_model_ablations", manifest)

    def test_campaign_evidence_excludes_incompatible_candidate_contracts(self) -> None:
        issue = pd.Timestamp("2026-06-01T00:00:00")
        times = pd.date_range(issue, periods=2, freq="3h")

        def forecast(contract: str, *, identity: str) -> xr.Dataset:
            return xr.Dataset(
                {
                    "BatterySOCForecast": (("time",), [80.0, 79.0]),
                    "ForecastLoadWatts": (("time",), [100.0, 100.0]),
                    "ForecastSolarWatts": (("time",), [0.0, 0.0]),
                    "ECMWFSolarIrradiance": (("time",), [0.0, 0.0]),
                },
                coords={"time": times},
                attrs={
                    "initial_soc_time": issue.isoformat(),
                    "forecast_model_contract_id": contract,
                    "forecast_system_version": "power-v12-hybrid-candidate",
                    "feature_set_version": "features-v4",
                    "feature_set_digest": "digest-v4",
                    "forecast_code_revision": "revision-v4",
                    "candidate_lane": "D_physical_solar_load_residual",
                    "baseline_control_contract_id": "baseline-control",
                    "baseline_control_system_version": "v10-control",
                    "local_feature_contract_id": "issue-features-v1",
                    "forecast_identity_id": identity,
                    "source_cycle_set_id": "cycle",
                    "source_availability_code": "ecmwf_control=available",
                },
            )

        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "pairs"
            active = forecast("active-contract", identity="active")
            incompatible = forecast("old-contract", identity="old")
            for pair, candidate in (("pair-active", active), ("pair-old", incompatible)):
                bundle = root / pair / "signature"
                bundle.mkdir(parents=True)
                active.to_zarr(bundle / "baseline_forecast.zarr", mode="w", consolidated=True)
                candidate.to_zarr(bundle / "candidate_forecast.zarr", mode="w", consolidated=True)
                (bundle / "pair_manifest.json").write_text(
                    json.dumps({"pair_status": "complete", "evaluation_pair_id": pair}),
                    encoding="utf-8",
                )
            power = xr.Dataset(
                {
                    "BatterySOC": (("time",), [80.0, 79.0]),
                    "ACOutputWatts": (("time",), [100.0, 100.0]),
                },
                coords={"time": times},
            )
            evidence = build_campaign_evidence(
                root,
                power,
                lane="D_physical_solar_load_residual",
                evaluation_contract=evaluation_contract_from_forecast(active),
            )
        self.assertEqual(evidence.sizes["record"], 2)
        self.assertEqual(evidence.attrs["incompatible_pair_count"], 1)

    def test_campaign_surface_reports_mpp_active_physical_solar_skill(self) -> None:
        issue = pd.Timestamp("2026-06-01T00:00:00")
        evidence = xr.Dataset(
            {
                "IssueTime": (("record",), [issue.to_datetime64(), issue.to_datetime64()]),
                "ValidTime": (("record",), [
                    issue.to_datetime64(),
                    (issue + pd.Timedelta(hours=3)).to_datetime64(),
                ]),
                "LeadHours": (("record",), [0.0, 3.0]),
                "SOCAuthoringAnchor": (("record",), [80.0, 80.0]),
                "CandidateSOC": (("record",), [80.0, 79.0]),
                "BaselineSOC": (("record",), [80.0, 79.0]),
                "ObservedSOC": (("record",), [80.0, 79.0]),
                "CandidateLoadWatts": (("record",), [100.0, 100.0]),
                "BaselineLoadWatts": (("record",), [100.0, 100.0]),
                "ObservedLoadWatts": (("record",), [100.0, 100.0]),
                "CandidateSolarWatts": (("record",), [900.0, 900.0]),
                "BaselineSolarWatts": (("record",), [600.0, 600.0]),
                "ObservedSolarWatts": (("record",), [900.0, 900.0]),
                "SolarEvaluationAvailable": (("record",), [True, True]),
                "EvaluationAvailable": (("record",), [True, True]),
                "LoadMode": (("record",), ["DC-Only", "DC-Only"]),
                "CloudRegime": (("record",), ["dark", "dark"]),
                "SourceAvailability": (("record",), ["ecmwf", "ecmwf"]),
                "DegradedModeCode": (("record",), ["none", "none"]),
            },
            coords={"record": [0, 1]},
        )

        summary = campaign_score_surfaces(evidence)

        solar = summary["campaign_evidence"]["solar"]
        self.assertEqual(solar["status"], "evidence")
        self.assertEqual(solar["candidate_mae"], 0.0)
        self.assertEqual(solar["baseline_mae"], 300.0)
