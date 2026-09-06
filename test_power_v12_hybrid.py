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
from generate_power_soc_v12_candidate import (
    POWER_HISTORY_FIELDS,
    _campaign_observation_start,
    _embedded_site_irradiance,
    run_candidate,
)
from power_issue_time_features import site_extract_sha256
from power_load_dynamics import ControlledLoadProfile
from power_solar_model import load_physical_solar_config
from power_v12_ensemble import (
    _validate_member_soc_physics,
    build_candidate_memberwise_ensemble,
    build_campaign_ensemble_evidence,
    campaign_ensemble_score_surfaces,
    ensemble_evaluation_contract_from_forecast,
    ensemble_promotion_gate,
    write_immutable_ensemble_pair_bundle,
)
from power_v12_hybrid import (
    PAIR_ARTIFACT_DIGEST_ALGORITHM,
    _clearness_cloud_regime,
    build_campaign_evidence,
    campaign_score_surfaces,
    evaluation_contract_from_forecast,
    fit_bounded_load_residual,
    immutable_artifact_record,
    promotion_gate_review,
)


CONFIG_PATH = Path(__file__).with_name("config") / "power_solar_physical_candidate_v1.json"


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_integrity_pair_manifest(
    bundle: Path,
    payload: dict[str, object],
    artifacts: dict[str, str],
) -> None:
    manifest = {
        **payload,
        "artifact_digest_algorithm": PAIR_ARTIFACT_DIGEST_ALGORITHM,
        "artifact_checksums": {
            logical_name: immutable_artifact_record(
                bundle / relative_path,
                relative_path=relative_path,
            )
            for logical_name, relative_path in artifacts.items()
        },
    }
    (bundle / "pair_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )


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
    def test_candidate_history_retains_mpp_modes_for_physical_solar_evidence(self) -> None:
        self.assertTrue(
            {
                "SolarMPPMode_East",
                "SolarMPPMode_South",
                "SolarMPPMode_West",
            }.issubset(POWER_HISTORY_FIELDS)
        )

    def test_embedded_site_forcing_requires_exact_grid_and_source_provenance(self) -> None:
        times = pd.date_range("2026-09-05T12:00:00", periods=3, freq="3h")
        baseline = xr.Dataset(
            {"ECMWFSolarIrradiance": (("time",), [np.nan, 120.0, 250.0])},
            coords={"time": times},
            attrs={
                "initial_soc_time": times[0].isoformat(),
                "ecmwf_cycle_time": "2026-09-05T00:00:00",
                "source_cycle_set_id": "ecmwf:legacy:cycle-test",
                "source_manifest_digest": f"sha256:{'a' * 64}",
            },
        )

        selected, digest, provenance = _embedded_site_irradiance(baseline)
        np.testing.assert_allclose(selected.values, [np.nan, 120.0, 250.0], equal_nan=True)
        self.assertRegex(digest, r"^[0-9a-f]{64}$")
        self.assertEqual(provenance["site_irradiance_sha256"], f"sha256:{digest}")
        self.assertEqual(
            provenance["source_manifest_digest"], baseline.attrs["source_manifest_digest"]
        )

        missing_field = baseline.drop_vars("ECMWFSolarIrradiance")
        with self.assertRaisesRegex(ValueError, "lacks embedded"):
            _embedded_site_irradiance(missing_field)
        missing_provenance = baseline.copy(deep=True)
        missing_provenance.attrs.pop("source_manifest_digest")
        with self.assertRaisesRegex(ValueError, "source-manifest provenance"):
            _embedded_site_irradiance(missing_provenance)
        wrong_grid = baseline.assign_coords(time=[times[0], times[0], times[2]])
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            _embedded_site_irradiance(wrong_grid)

    def test_campaign_observation_window_starts_at_first_candidate_issue(self) -> None:
        issue = pd.Timestamp("2026-09-05T12:00:00")
        training_start = issue - pd.Timedelta(days=21)
        first_issue = issue - pd.Timedelta(days=35)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifests = root / "source_manifests"
            manifests.mkdir()
            (manifests / "sha256-one.json").write_text(
                json.dumps({"initial_soc_time": first_issue.isoformat()}),
                encoding="utf-8",
            )
            (manifests / "sha256-future.json").write_text(
                json.dumps({"initial_soc_time": (issue + pd.Timedelta(days=1)).isoformat()}),
                encoding="utf-8",
            )

            selected = _campaign_observation_start(
                root,
                issue_time=issue,
                training_start=training_start,
            )

        self.assertEqual(selected, first_issue)

    def test_cloud_regime_uses_daytime_clearness_index_and_fails_closed(self) -> None:
        self.assertEqual(_clearness_cloud_regime(0.0, np.nan), "dark")
        self.assertEqual(_clearness_cloud_regime(300.0, np.nan), "unknown")
        self.assertEqual(_clearness_cloud_regime(300.0, 0.30), "cloudy")
        self.assertEqual(_clearness_cloud_regime(300.0, 0.50), "transitional")
        self.assertEqual(_clearness_cloud_regime(300.0, 0.70), "clear")

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

    def test_load_residual_contract_is_stable_across_issue_operating_modes(self) -> None:
        archive = _archive_with_load_history()
        times = pd.date_range("2026-06-01", periods=5 * 24, freq="1h")
        power = xr.Dataset(
            {
                "ACOutputWatts": (("time",), np.full(len(times), 200.0)),
                "DCInverterWatts": (("time",), np.zeros(len(times))),
            },
            coords={"time": times},
        )
        common = {
            "archive": archive,
            "power": power,
            "issue_time": "2026-06-04T00:00:00",
            "forecast_times": pd.date_range("2026-06-04", periods=3, freq="3h"),
            "control_forecast_model_contract_id": "v10",
            "control_forecast_system_version": "v10-control",
        }

        dc_only = fit_bounded_load_residual(load_mode="DC-Only", **common)
        cl61 = fit_bounded_load_residual(load_mode="CL61", **common)

        self.assertEqual(dc_only.contract_id, cl61.contract_id)
        self.assertNotEqual(dc_only.status, cl61.status)

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
                "initial_soc_pct": "70",
                "adaptive_calibration_state_id": "adaptive-state-test",
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
        self.assertEqual(
            str(archived["AdaptiveCalibrationStateID"].values[0]),
            "adaptive-state-test",
        )
        self.assertEqual(str(archived["ObservationCutoffUTC"].values[0]), issue.isoformat())
        self.assertEqual(str(archived["SOCAuthoringAnchorPct"].values[0]), "70")

    def test_active_evaluation_filter_keeps_the_complete_system_version(self) -> None:
        semantic_archive = {
            "ForecastSystemVersion": ["v10", "v12"],
            "FeatureSetVersion": ["features-v1", "features-v2"],
            "FeatureSetDigest": ["digest-v1", "digest-v2"],
            "ForecastCodeRevision": ["revision-v1", "revision-v2"],
            "CandidateLane": ["baseline", "D_hybrid"],
            "LocalFeatureContractID": ["local-v1", "local-v2"],
            "BaselineControlContractID": ["control-v9", "control-v10"],
            "BaselineControlSystemVersion": ["power-v9", "power-v10"],
        }
        archive = xr.Dataset(
            {
                "ForecastModelContractID": (("issue_time",), ["shared-contract", "shared-contract"]),
                **{
                    name: (("issue_time",), values)
                    for name, values in semantic_archive.items()
                },
            },
            coords={"issue_time": pd.date_range("2026-06-01", periods=2, freq="1D")},
        )
        table = pd.DataFrame(
            {
                "forecast_model_contract_id": ["shared-contract", "shared-contract"],
                "forecast_system_version": semantic_archive["ForecastSystemVersion"],
                "feature_set_version": semantic_archive["FeatureSetVersion"],
                "feature_set_digest": semantic_archive["FeatureSetDigest"],
                "forecast_code_revision": semantic_archive["ForecastCodeRevision"],
                "candidate_lane": semantic_archive["CandidateLane"],
                "local_feature_contract_id": semantic_archive["LocalFeatureContractID"],
                "baseline_control_contract_id": semantic_archive[
                    "BaselineControlContractID"
                ],
                "baseline_control_system_version": semantic_archive[
                    "BaselineControlSystemVersion"
                ],
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
            baseline_issue_path = root / "baseline" / "issues" / "forecast.zarr"
            archive_path = root / "baseline" / "power_soc_forecast_archive.zarr"
            baseline_ensemble_path = root / "baseline" / "power_soc_ensemble_forecast.zarr"
            public_source_root = root / "public-source-inputs"
            power.to_zarr(power_path, mode="w", consolidated=True)
            input_path.write_bytes(b"v12 synthetic forcing")
            with patch(
                "generate_power_soc_forecast.open_provider_solar_forecast",
                return_value=provider,
            ) as provider_open:
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
                    issue_snapshot_zarr=baseline_issue_path,
                )
                with xr.open_zarr(baseline_issue_path, chunks={}) as opened:
                    baseline_forecast = opened.load()
                member_count = 3
                member_grid = np.arange(1, member_count + 1, dtype=np.int16)
                solar_irradiance = np.asarray(
                    baseline_forecast["ECMWFSolarIrradiance"].values, dtype=np.float64
                )
                solar_watts = np.asarray(
                    baseline_forecast["ForecastSolarWatts"].values, dtype=np.float64
                )
                load_watts = np.asarray(
                    baseline_forecast["ForecastLoadWatts"].values, dtype=np.float64
                )
                soc = np.asarray(
                    baseline_forecast["BatterySOCForecast"].values, dtype=np.float64
                )
                baseline_ensemble = xr.Dataset(
                    {
                        "BatterySOCForecastEnsemble": (("member", "time"), np.tile(soc, (member_count, 1))),
                        "ECMWFSolarIrradianceEnsemble": (("member", "time"), np.tile(solar_irradiance, (member_count, 1))),
                        "ForecastSolarWattsEnsemble": (("member", "time"), np.tile(solar_watts, (member_count, 1))),
                        "ForecastLoadWattsEnsemble": (("member", "time"), np.tile(load_watts, (member_count, 1))),
                        "ForecastLoadPhaseCodeEnsemble": (("member", "time"), np.zeros((member_count, len(soc)), dtype=np.int8)),
                        "BatteryUsableCapacityKWhEnsemble": (("member",), np.asarray([25.0, 26.0, 27.0])),
                        "BatteryChargeEfficiencyEnsemble": (("member",), np.asarray([0.90, 0.92, 0.94])),
                        "BatteryDischargeEfficiencyEnsemble": (("member",), np.asarray([0.90, 0.92, 0.94])),
                    },
                    coords={"member": member_grid, "time": baseline_forecast["time"].values},
                    attrs={
                        "initial_soc_time": baseline_forecast.attrs["initial_soc_time"],
                        "initial_soc_pct": baseline_forecast.attrs["initial_soc_pct"],
                        "ecmwf_cycle_time": baseline_forecast.attrs["ecmwf_cycle_time"],
                        "forecast_horizon_hours": baseline_forecast.attrs["forecast_horizon_hours"],
                    },
                )
                baseline_ensemble.to_zarr(baseline_ensemble_path, mode="w", consolidated=True)
                public_source_root.mkdir()
                public_ghi = np.nan_to_num(solar_irradiance, nan=0.0, posinf=0.0, neginf=0.0)
                public_extract = public_source_root / "ifs.zarr"
                xr.Dataset(
                    {"ghi_w_m2": (("time",), public_ghi)},
                    coords={"time": baseline_forecast["time"].values},
                ).to_zarr(public_extract, mode="w", consolidated=True)
                (public_source_root / "ifs.json").write_text(
                    json.dumps(
                        {
                            "schema_version": "site-extracted-public-source-v2",
                            "source": "IFS",
                            "site_extract_only": True,
                            "global_grid_retained": False,
                            "source_cycle_time_utc": baseline_forecast.attrs["ecmwf_cycle_time"],
                            "delivery_time_utc": baseline_forecast.attrs["initial_soc_time"],
                            "site_extract_path": "ifs.zarr",
                            "site_extract_format": "zarr",
                            "irradiance_variable": "ghi_w_m2",
                            "site_latitude": 64.829694,
                            "site_longitude": -23.248139,
                            "site_extract_sha256": site_extract_sha256(public_extract),
                        }
                    ),
                    encoding="utf-8",
                )
                baseline_digest = _tree_digest(baseline_issue_path)
                archive_digest = _tree_digest(archive_path)
                input_path.unlink()
                results = run_candidate(
                    baseline_issue_zarr=baseline_issue_path,
                    baseline_archive_zarr=archive_path,
                    baseline_ensemble_zarr=baseline_ensemble_path,
                    candidate_root=root / "candidate",
                    power_zarr=power_path,
                    pdu_zarr=root / "source" / "missing-pdu.zarr",
                    physical_config=CONFIG_PATH,
                    public_source_manifest_root=public_source_root,
                )
                repeated = run_candidate(
                    baseline_issue_zarr=baseline_issue_path,
                    baseline_archive_zarr=archive_path,
                    baseline_ensemble_zarr=baseline_ensemble_path,
                    candidate_root=root / "candidate",
                    power_zarr=power_path,
                    pdu_zarr=root / "source" / "missing-pdu.zarr",
                    physical_config=CONFIG_PATH,
                    public_source_manifest_root=public_source_root,
                )
                self.assertEqual(provider_open.call_count, 1)
            self.assertEqual(set(results), {"B_physical_solar", "C_load_residual", "D_physical_solar_load_residual"})
            self.assertEqual(results, repeated)
            self.assertFalse(input_path.exists())
            self.assertEqual(_tree_digest(baseline_issue_path), baseline_digest)
            self.assertEqual(_tree_digest(archive_path), archive_digest)
            status = json.loads((root / "candidate" / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "complete")
            self.assertEqual(status["memberwise_ensemble_input_status"], "available")
            self.assertEqual(status["public_model_ablation_results"]["ifs"]["status"], "complete")
            acceptance = json.loads((root / "candidate" / "acceptance_record.json").read_text(encoding="utf-8"))
            review = json.loads((root / "candidate" / "review_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(acceptance["status"], "not_accepted")
            self.assertEqual(review["status"], "pending_campaign_review")
            self.assertEqual(status["promotion_gates"]["status"], "not_eligible")
            with xr.open_zarr(results["C_load_residual"], chunks={}) as candidate:
                np.testing.assert_allclose(
                    candidate["ForecastSolarWatts"].values,
                    xr.open_zarr(baseline_issue_path, chunks={})["ForecastSolarWatts"].values,
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
            pair_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(pair_manifest["pair_status"], "complete")
            self.assertEqual(
                pair_manifest["artifact_digest_algorithm"],
                PAIR_ARTIFACT_DIGEST_ALGORITHM,
            )
            self.assertEqual(
                set(pair_manifest["artifact_checksums"]), {"baseline", "candidate"}
            )
            ensemble_status = status["lanes"]["D_physical_solar_load_residual"]["memberwise_ensemble"]
            self.assertEqual(ensemble_status["status"], "complete", ensemble_status)
            source_manifest = next((root / "candidate" / "source_manifests").glob("*.json"))
            manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
            self.assertIn("issue_time_features", manifest)
            self.assertIn("public_model_ablations", manifest)
            self.assertEqual(manifest["solar_input_representation"], "embedded_site_irradiance")
            self.assertFalse(manifest["global_grid_retained"])
            self.assertRegex(manifest["site_irradiance_sha256"], r"^sha256:[0-9a-f]{64}$")
            self.assertRegex(
                manifest["baseline_issue_content_digest"], r"^sha256:[0-9a-f]{64}$"
            )

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
                _write_integrity_pair_manifest(
                    bundle,
                    {"pair_status": "complete", "evaluation_pair_id": pair},
                    {
                        "baseline": "baseline_forecast.zarr",
                        "candidate": "candidate_forecast.zarr",
                    },
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

    def test_campaign_evidence_retains_reanchors_but_scores_one_per_cycle_valid(self) -> None:
        issue = pd.Timestamp("2026-06-01T00:00:00")
        times = pd.date_range(issue, periods=2, freq="3h")
        attrs = {
            "initial_soc_time": issue.isoformat(),
            "ecmwf_cycle_time": issue.isoformat(),
            "forecast_model_contract_id": "candidate-contract",
            "forecast_system_version": "power-v12-hybrid-candidate",
            "feature_set_version": "features-v4",
            "feature_set_digest": "digest-v4",
            "forecast_code_revision": "revision-v4",
            "candidate_lane": "D_physical_solar_load_residual",
            "baseline_control_contract_id": "baseline-control",
            "baseline_control_system_version": "v10-control",
            "local_feature_contract_id": "issue-features-v1",
            "forecast_identity_id": "identity",
            "source_cycle_set_id": "same-source-cycle",
            "source_availability_code": "ecmwf_control=available",
            "initial_soc_pct": "80",
        }
        product = xr.Dataset(
            {
                "BatterySOCForecast": (("time",), [80.0, 79.0]),
                "ForecastLoadWatts": (("time",), [100.0, 100.0]),
                "ForecastSolarWatts": (("time",), [0.0, 0.0]),
                "ECMWFSolarIrradiance": (("time",), [0.0, 0.0]),
            },
            coords={"time": times},
            attrs=attrs,
        )
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "pairs"
            for pair in ("retry-one", "retry-two"):
                bundle = root / pair / "signature"
                bundle.mkdir(parents=True)
                product.to_zarr(bundle / "baseline_forecast.zarr", mode="w", consolidated=True)
                product.to_zarr(bundle / "candidate_forecast.zarr", mode="w", consolidated=True)
                _write_integrity_pair_manifest(
                    bundle,
                    {"pair_status": "complete", "evaluation_pair_id": pair},
                    {
                        "baseline": "baseline_forecast.zarr",
                        "candidate": "candidate_forecast.zarr",
                    },
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
                evaluation_contract=evaluation_contract_from_forecast(product),
            )

        self.assertEqual(evidence.sizes["record"], 4)
        self.assertEqual(evidence.attrs["duplicate_cycle_valid_rows_discarded"], 0)
        self.assertEqual(evidence.attrs["duplicate_cycle_valid_rows_retained"], 2)
        self.assertEqual(
            int(evidence["IndependentEvaluationSample"].sum().item()),
            2,
        )
        np.testing.assert_array_equal(evidence["SOCAnchorTime"], evidence["IssueTime"])
        summary = campaign_score_surfaces(evidence)["campaign_evidence"]["lead_buckets"]["0_6h"]
        self.assertEqual(summary["cycles"], 1)
        self.assertEqual(summary["samples"], 2)

    def test_campaign_evidence_rejects_changed_pair_artifact_bytes(self) -> None:
        issue = pd.Timestamp("2026-06-01T00:00:00")
        times = pd.date_range(issue, periods=2, freq="3h")
        product = xr.Dataset(
            {
                "BatterySOCForecast": (("time",), [80.0, 79.0]),
                "ForecastLoadWatts": (("time",), [100.0, 100.0]),
                "ForecastSolarWatts": (("time",), [0.0, 0.0]),
                "ECMWFSolarIrradiance": (("time",), [0.0, 0.0]),
            },
            coords={"time": times},
            attrs={
                "initial_soc_time": issue.isoformat(),
                "forecast_model_contract_id": "candidate-contract",
                "forecast_system_version": "power-v12-hybrid-candidate",
                "feature_set_version": "features-v4",
                "feature_set_digest": "digest-v4",
                "forecast_code_revision": "revision-v4",
                "candidate_lane": "D_physical_solar_load_residual",
                "baseline_control_contract_id": "baseline-control",
                "baseline_control_system_version": "v10-control",
                "local_feature_contract_id": "issue-features-v1",
                "source_cycle_set_id": "cycle-one",
                "initial_soc_pct": "80",
            },
        )
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "pairs"
            bundle = root / "pair" / "signature"
            bundle.mkdir(parents=True)
            product.to_zarr(bundle / "baseline_forecast.zarr", mode="w", consolidated=True)
            product.to_zarr(bundle / "candidate_forecast.zarr", mode="w", consolidated=True)
            _write_integrity_pair_manifest(
                bundle,
                {"pair_status": "complete", "evaluation_pair_id": "pair"},
                {
                    "baseline": "baseline_forecast.zarr",
                    "candidate": "candidate_forecast.zarr",
                },
            )
            changed_file = next(
                path
                for path in (bundle / "candidate_forecast.zarr").rglob("*")
                if path.is_file()
            )
            changed_file.write_bytes(changed_file.read_bytes() + b"changed")
            power = xr.Dataset(
                {"BatterySOC": (("time",), [80.0, 79.0])},
                coords={"time": times},
            )
            evidence = build_campaign_evidence(
                root,
                power,
                lane="D_physical_solar_load_residual",
                evaluation_contract=evaluation_contract_from_forecast(product),
            )

        self.assertEqual(evidence.sizes["record"], 0)
        self.assertEqual(evidence.attrs["evidence_status"], "no_complete_pair_bundles")

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

    def test_promotion_review_requires_campaign_evidence_but_never_auto_promotes(self) -> None:
        issues = pd.date_range("2026-06-01T00:00:00", periods=30, freq="8h")
        leads = np.asarray([3.0, 12.0, 30.0, 60.0])
        issue_values = np.repeat(issues.to_numpy(dtype="datetime64[ns]"), len(leads))
        lead_values = np.tile(leads, len(issues))
        valid_values = np.asarray(
            [
                issue + pd.Timedelta(hours=float(lead))
                for issue in issues
                for lead in leads
            ],
            dtype="datetime64[ns]",
        )
        count = len(issue_values)
        source_cycles = np.repeat(
            np.asarray([f"cycle-{index}" for index in range(len(issues))]),
            len(leads),
        )
        load_modes = np.repeat(
            np.asarray(
                ["DC-Only" if index % 2 == 0 else "CL61" for index in range(len(issues))]
            ),
            len(leads),
        )
        cloud_regimes = np.repeat(
            np.asarray(
                ["clear" if index % 2 == 0 else "cloudy" for index in range(len(issues))]
            ),
            len(leads),
        )
        evidence = xr.Dataset(
            {
                "IssueTime": (("record",), issue_values),
                "ValidTime": (("record",), valid_values),
                "LeadHours": (("record",), lead_values),
                "SOCAuthoringAnchor": (("record",), np.full(count, 80.0)),
                "CandidateSOC": (("record",), np.full(count, 76.0)),
                "BaselineSOC": (("record",), np.full(count, 77.0)),
                "ObservedSOC": (("record",), np.full(count, 75.0)),
                "CandidateLoadWatts": (("record",), np.full(count, 105.0)),
                "BaselineLoadWatts": (("record",), np.full(count, 110.0)),
                "ObservedLoadWatts": (("record",), np.full(count, 100.0)),
                "CandidateSolarWatts": (("record",), np.full(count, 900.0)),
                "BaselineSolarWatts": (("record",), np.full(count, 800.0)),
                "ObservedSolarWatts": (("record",), np.full(count, 900.0)),
                "SolarEvaluationAvailable": (("record",), np.ones(count, dtype=bool)),
                "EvaluationAvailable": (("record",), np.ones(count, dtype=bool)),
                "SourceCycleSetID": (("record",), source_cycles),
                "LoadMode": (("record",), load_modes),
                "CloudRegime": (("record",), cloud_regimes),
            },
            coords={"record": np.arange(count)},
        )

        review = promotion_gate_review(evidence)

        self.assertEqual(review["independent_evidence"]["0_6h"]["status"], "eligible")
        self.assertEqual(review["soc"]["status"], "pass")
        self.assertEqual(review["long_lead_soc"]["24_48h"]["status"], "pass")
        self.assertEqual(review["long_lead_soc"]["48_96h"]["status"], "pass")
        self.assertEqual(review["solar"]["status"], "pass")
        self.assertEqual(review["load"]["status"], "pass")
        self.assertEqual(review["quantitative_gates"], "pass")
        self.assertEqual(review["ensemble"], "blocked_memberwise_candidate_not_generated")
        self.assertEqual(review["status"], "not_eligible")

        missing_diversity = promotion_gate_review(
            evidence.drop_vars(["CloudRegime", "LoadMode"])
        )
        self.assertEqual(missing_diversity["evidence"], "insufficient_evidence")
        self.assertEqual(
            missing_diversity["independent_evidence"]["0_6h"]["status"],
            "insufficient_diversity",
        )

    def test_member_physics_direction_includes_battery_parasitic_load(self) -> None:
        soc = np.asarray([[50.0, 49.0]])
        solar = np.asarray([[np.nan, 100.0]])
        load = np.asarray([[np.nan, 100.0]])
        unavailable_flow = np.full((1, 2), np.nan)

        _validate_member_soc_physics(
            soc,
            solar,
            load,
            unavailable_flow,
            unavailable_flow,
            np.asarray([10.0]),
        )
        with self.assertRaisesRegex(ValueError, "falls without net discharging"):
            _validate_member_soc_physics(
                soc,
                solar,
                load,
                unavailable_flow,
                unavailable_flow,
                np.asarray([0.0]),
            )

    def test_memberwise_candidate_ensemble_preserves_pair_and_replays_legacy_solar(self) -> None:
        issue = pd.Timestamp("2026-06-21T06:00:00")
        times = pd.date_range(issue, periods=5, freq="3h")
        common_attrs = {
            "initial_soc_time": issue.isoformat(),
            "initial_soc_pct": "70",
            "ecmwf_cycle_time": issue.floor("3h").isoformat(),
            "forecast_model_contract_id": "candidate-contract",
            "forecast_system_version": "power-v12-hybrid-candidate",
            "feature_set_version": "issue-safe-v4",
            "feature_set_digest": "features",
            "forecast_code_revision": "test",
            "baseline_control_contract_id": "baseline-contract",
            "baseline_control_system_version": "v10",
            "local_feature_contract_id": "local-v1",
            "source_cycle_set_id": "ecmwf:test",
            "candidate_lane": "B_physical_solar",
            "publication_signature": "candidate-publication",
            "forecast_identity_id": "candidate-identity",
            "battery_usable_capacity_kwh": "26",
            "battery_charge_efficiency": "0.92",
            "battery_discharge_efficiency": "0.92",
            "battery_max_charge_w": "3000",
            "battery_max_discharge_w": "3000",
        }
        baseline = xr.Dataset(
            {
                "BatterySOCForecast": (("time",), np.asarray([70.0, 70.0, 70.0, 70.0, 70.0])),
                "ForecastLoadWatts": (("time",), np.full(len(times), 180.0)),
                "ForecastSolarWatts": (("time",), np.asarray([0.0, 100.0, 500.0, 300.0, 0.0])),
            },
            coords={"time": times},
            attrs={**common_attrs, "forecast_model_contract_id": "baseline-contract"},
        )
        candidate = baseline.copy(deep=True)
        candidate.attrs.update(common_attrs)
        candidate["ForecastSolarWatts"] = (("time",), np.asarray([0.0, 400.0, 1200.0, 700.0, 0.0]))
        members = np.asarray([1, 2, 3], dtype=np.int16)
        irradiance = np.asarray(
            [
                [0.0, 120.0, 500.0, 300.0, 0.0],
                [0.0, 100.0, 450.0, 250.0, 0.0],
                [0.0, 80.0, 400.0, 220.0, 0.0],
            ]
        )
        baseline_ensemble = xr.Dataset(
            {
                "BatterySOCForecastEnsemble": (("member", "time"), np.full((3, len(times)), 70.0)),
                "ECMWFSolarIrradianceEnsemble": (("member", "time"), irradiance),
                "ForecastSolarWattsEnsemble": (("member", "time"), irradiance * 2.0),
                "ForecastLoadWattsEnsemble": (("member", "time"), np.asarray([[170.0] * 5, [180.0] * 5, [190.0] * 5])),
                "ForecastLoadPhaseCodeEnsemble": (("member", "time"), np.zeros((3, len(times)), dtype=np.int8)),
                "BatteryUsableCapacityKWhEnsemble": (("member",), np.asarray([25.0, 26.0, 27.0])),
                "BatteryChargeEfficiencyEnsemble": (("member",), np.asarray([0.90, 0.92, 0.94])),
                "BatteryDischargeEfficiencyEnsemble": (("member",), np.asarray([0.90, 0.92, 0.94])),
            },
            coords={"member": members, "time": times},
            attrs={
                "initial_soc_time": issue.isoformat(),
                "initial_soc_pct": "70",
                "ecmwf_cycle_time": issue.floor("3h").isoformat(),
                "forecast_horizon_hours": "12",
            },
        )
        config = load_physical_solar_config(CONFIG_PATH)
        physical = build_candidate_memberwise_ensemble(
            baseline,
            candidate,
            baseline_ensemble,
            lane="B_physical_solar",
            physical_config=config,
            latitude=64.829694,
            longitude=-23.248139,
        )
        self.assertEqual(physical.attrs["solar_forcing_mode"], "memberwise_physical_available_pv")
        self.assertEqual(physical.attrs["candidate_lane"], "B_physical_solar")
        self.assertFalse(
            np.allclose(
                physical["ForecastSolarWattsEnsemble"].values,
                baseline_ensemble["ForecastSolarWattsEnsemble"].values,
                equal_nan=True,
            )
        )
        candidate.attrs["candidate_lane"] = "C_load_residual"
        legacy = build_candidate_memberwise_ensemble(
            baseline,
            candidate,
            baseline_ensemble,
            lane="C_load_residual",
            physical_config=config,
            latitude=64.829694,
            longitude=-23.248139,
        )
        np.testing.assert_allclose(
            legacy["ForecastSolarWattsEnsemble"].values,
            baseline_ensemble["ForecastSolarWattsEnsemble"].values,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            legacy["ForecastLoadWattsEnsemble"].values,
            baseline_ensemble["ForecastLoadWattsEnsemble"].values,
            rtol=0.0,
            atol=0.0,
        )
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = write_immutable_ensemble_pair_bundle(
                root,
                deterministic_pair_id="pair-1",
                baseline_ensemble=baseline_ensemble,
                candidate_ensemble=physical,
                manifest_extra={"baseline_publication_signature": "baseline-publication"},
            )
            self.assertTrue((bundle / "pair_manifest.json").exists())
            ensemble_manifest = json.loads(
                (bundle / "pair_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                ensemble_manifest["artifact_digest_algorithm"],
                PAIR_ARTIFACT_DIGEST_ALGORITHM,
            )
            self.assertEqual(
                set(ensemble_manifest["artifact_checksums"]),
                {"baseline", "candidate"},
            )
            power = xr.Dataset(
                {"BatterySOC": (("time",), physical["BatterySOCForecastP50"].values)},
                coords={"time": times},
            )
            evidence = build_campaign_ensemble_evidence(
                root / "ensemble_pairs",
                power,
                lane="B_physical_solar",
                evaluation_contract=ensemble_evaluation_contract_from_forecast(physical),
            )
            changed_file = next(
                path
                for path in (bundle / "candidate_ensemble.zarr").rglob("*")
                if path.is_file()
            )
            changed_file.write_bytes(changed_file.read_bytes() + b"changed")
            rejected = build_campaign_ensemble_evidence(
                root / "ensemble_pairs",
                power,
                lane="B_physical_solar",
                evaluation_contract=ensemble_evaluation_contract_from_forecast(physical),
            )
        self.assertEqual(evidence.sizes["record"], len(times))
        self.assertEqual(rejected.sizes["record"], 0)

    def test_memberwise_ensemble_gate_requires_calibrated_spread_and_scores_brier(self) -> None:
        issues = pd.date_range("2026-06-01T00:00:00", periods=30, freq="8h")
        leads = np.asarray([3.0, 12.0, 30.0, 60.0])
        issue_values = np.repeat(issues.to_numpy(dtype="datetime64[ns]"), len(leads))
        lead_values = np.tile(leads, len(issues))
        valid_values = np.asarray(
            [issue + pd.Timedelta(hours=float(lead)) for issue in issues for lead in leads],
            dtype="datetime64[ns]",
        )
        count = len(issue_values)
        observed = np.where(np.arange(count) % 4 == 0, 35.0, 60.0)
        candidate_members = np.column_stack((observed - 1.0, observed, observed + 1.0))
        # Deliberately leave one fifth of observations outside P10-P90: this
        # produces the required calibrated 80% nominal interval coverage.
        outside = np.arange(count) % 5 == 0
        candidate_members[outside] += 4.0
        baseline_members = np.column_stack((observed + 10.0, observed + 11.0, observed + 12.0))
        evidence = xr.Dataset(
            {
                "IssueTime": (("record",), issue_values),
                "ValidTime": (("record",), valid_values),
                "LeadHours": (("record",), lead_values),
                "SOCAuthoringAnchor": (("record",), np.full(count, 80.0)),
                "ObservedSOC": (("record",), observed),
                "EvaluationAvailable": (("record",), np.ones(count, dtype=bool)),
                "CandidateSOCMembers": (("record", "member"), candidate_members),
                "BaselineSOCMembers": (("record", "member"), baseline_members),
            },
            coords={"record": np.arange(count), "member": np.asarray([1, 2, 3])},
        )
        gate = ensemble_promotion_gate(evidence)
        self.assertEqual(gate["status"], "pass")
        self.assertEqual(gate["crps_status"], "pass")
        self.assertEqual(gate["coverage_status"], "pass")
        self.assertEqual(gate["reserve_events"], "pass")

    def test_ensemble_surface_counts_source_cycles_not_reissued_anchors(self) -> None:
        issues = pd.date_range("2026-06-01", periods=4, freq="1h")
        members = np.asarray([[60.0, 61.0, 62.0]] * 4)
        evidence = xr.Dataset(
            {
                "IssueTime": (("record",), issues.to_numpy(dtype="datetime64[ns]")),
                "ValidTime": (("record",), (issues + pd.Timedelta(hours=3)).to_numpy(dtype="datetime64[ns]")),
                "LeadHours": (("record",), np.full(4, 3.0)),
                "SourceCycleSetID": (("record",), np.asarray(["cycle-a", "cycle-a", "cycle-b", "cycle-b"])),
                "SOCAuthoringAnchor": (("record",), np.full(4, 65.0)),
                "ObservedSOC": (("record",), np.full(4, 61.0)),
                "EvaluationAvailable": (("record",), np.ones(4, dtype=bool)),
                "IndependentEvaluationSample": (
                    ("record",), np.asarray([False, True, False, True])
                ),
                "CandidateSOCMembers": (("record", "member"), members),
                "BaselineSOCMembers": (("record", "member"), members + 2.0),
            },
            coords={"record": np.arange(4), "member": np.arange(3)},
        )

        overall = campaign_ensemble_score_surfaces(evidence)["campaign_evidence"]["overall"]

        self.assertEqual(overall["cycles"], 2)
        self.assertEqual(overall["samples"], 2)
