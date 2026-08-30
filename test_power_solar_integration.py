from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr

from append_new_power_to_zarr import OPTIONAL_SCHEMA_EXPANSION_VARS
from ecmwf_forecast_provider import ForecastProviderResult
from generate_power_soc_forecast import (
    PHYSICAL_SOLAR_MODEL_NAME,
    POWER_SOC_FORECAST_ARCHIVE_ZARR_PATH,
    _archive_row_from_forecast,
    build_forecast_dataset,
    build_forecast_skill_dataset,
    evaluate_forecast_archive,
    evaluate_independent_forecast_archive,
    evaluate_previous_forecast,
    forecast_publication_signature,
    generate,
    integrate_soc_from_available_solar,
    integrate_soc_forecast,
)
from generate_power_soc_physical_candidate import run_candidate
from power_battery_model import BatteryModel
from power_solar_model import load_physical_solar_config


CONFIG_PATH = Path(__file__).with_name("config") / "power_solar_physical_candidate_v1.json"


def synthetic_inputs() -> tuple[xr.Dataset, xr.Dataset]:
    power_times = pd.date_range("2026-06-21T09:00:00", periods=25, freq="1h")
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
    interval_ghi = np.asarray([0.0, 150.0, 350.0, 550.0, 300.0, 75.0])
    accumulated = np.cumsum(interval_ghi * 3.0 * 3600.0)
    solar = xr.Dataset(
        {"ssrd": (("time",), accumulated)},
        coords={"time": forecast_times},
    )
    return power, solar


class PhysicalSolarIntegrationTests(unittest.TestCase):
    def test_power_ingest_allows_mpp_mode_schema_expansion(self) -> None:
        self.assertTrue(
            {"SolarMPPMode_East", "SolarMPPMode_South", "SolarMPPMode_West"}.issubset(
                OPTIONAL_SCHEMA_EXPANSION_VARS
            )
        )

    def test_candidate_publication_signature_tracks_pair_and_input_snapshot(self) -> None:
        forecast = xr.Dataset(
            attrs={
                "initial_soc_time": "2026-06-21T12:00:00",
                "initial_soc_pct": "70",
                "forecast_load_w": "100",
                "forecast_model_contract_id": "forecast-model-test",
                "evaluation_pair_id": "pair-a",
                "input_snapshot_id": "sha256:a",
            }
        )
        first = forecast_publication_signature(forecast)
        forecast.attrs["evaluation_pair_id"] = "pair-b"
        second = forecast_publication_signature(forecast)
        forecast.attrs["evaluation_pair_id"] = "pair-a"
        forecast.attrs["input_snapshot_id"] = "sha256:b"
        third = forecast_publication_signature(forecast)

        self.assertNotEqual(first, second)
        self.assertNotEqual(first, third)

    def test_available_pv_reports_battery_curtailment_at_full_soc(self) -> None:
        times = pd.date_range("2026-06-21T12:00:00", periods=2, freq="1h")
        result = integrate_soc_from_available_solar(
            initial_soc=100.0,
            available_solar_w=pd.Series([1000.0, 1000.0], index=times),
            load_w=100.0,
            battery_model=BatteryModel(
                usable_capacity_kwh=26.0,
                charge_efficiency=0.92,
                discharge_efficiency=0.92,
                max_charge_w=3000.0,
                max_discharge_w=3000.0,
            ),
        )

        self.assertEqual(float(result["BatterySOCForecast"].iloc[-1]), 100.0)
        self.assertEqual(float(result["ForecastPVAvailableWatts"].iloc[-1]), 1000.0)
        self.assertEqual(float(result["ForecastPVDeliveredWatts"].iloc[-1]), 100.0)
        self.assertEqual(float(result["ForecastPVCurtailedWatts"].iloc[-1]), 900.0)

    def test_substep_battery_integration_preserves_charge_discharge_order(self) -> None:
        initial_time = pd.Timestamp("2026-06-21T00:00:00")
        times = pd.date_range("2026-06-21T01:00:00", periods=3, freq="1h")
        battery = BatteryModel(
            usable_capacity_kwh=10.0,
            charge_efficiency=0.92,
            discharge_efficiency=0.92,
            max_charge_w=3000.0,
            max_discharge_w=3000.0,
        )
        ordered = integrate_soc_from_available_solar(
            initial_soc=50.0,
            initial_time=initial_time,
            available_solar_w=pd.Series([300.0, 0.0, 0.0], index=times),
            load_w=100.0,
            battery_model=battery,
        )
        collapsed_mean = integrate_soc_from_available_solar(
            initial_soc=50.0,
            initial_time=initial_time,
            available_solar_w=pd.Series([100.0, 100.0, 100.0], index=times),
            load_w=100.0,
            battery_model=battery,
        )

        self.assertAlmostEqual(float(collapsed_mean["BatterySOCForecast"].iloc[-1]), 50.0)
        self.assertLess(
            float(ordered["BatterySOCForecast"].iloc[-1]),
            float(collapsed_mean["BatterySOCForecast"].iloc[-1]) - 0.3,
        )

    def test_substeps_preserve_right_endpoint_load_interval_convention(self) -> None:
        initial_time = pd.Timestamp("2026-06-21T00:00:00")
        endpoint = initial_time + pd.Timedelta(hours=3)
        coarse_times = pd.DatetimeIndex([initial_time, endpoint])
        load = pd.Series([900.0, 100.0], index=coarse_times)
        battery = BatteryModel(
            usable_capacity_kwh=10.0,
            charge_efficiency=1.0,
            discharge_efficiency=1.0,
            max_charge_w=20_000.0,
            max_discharge_w=20_000.0,
        )
        baseline = integrate_soc_forecast(
            initial_soc=80.0,
            initial_time=initial_time,
            irradiance=pd.Series([0.0, 0.0], index=coarse_times),
            solar_factor=1.0,
            load_w=load,
            battery_model=battery,
        )
        substep_times = pd.date_range(
            initial_time + pd.Timedelta(minutes=10),
            endpoint,
            freq="10min",
        )
        physical = integrate_soc_from_available_solar(
            initial_soc=80.0,
            initial_time=initial_time,
            available_solar_w=pd.Series(0.0, index=substep_times),
            load_w=load,
            battery_model=battery,
        )

        self.assertAlmostEqual(
            float(physical["BatterySOCForecast"].iloc[-1]),
            float(baseline["BatterySOCForecast"].iloc[-1]),
        )
        np.testing.assert_allclose(physical["ForecastLoadWatts"].iloc[1:], 100.0)

    def test_build_candidate_preserves_compatibility_and_physical_diagnostics(self) -> None:
        power, solar = synthetic_inputs()
        forecast = build_forecast_dataset(
            power,
            solar,
            horizon_hours=15,
            capacity_kwh=26.0,
            solar_model=PHYSICAL_SOLAR_MODEL_NAME,
            physical_solar_config=load_physical_solar_config(CONFIG_PATH),
            latitude=64.829694,
            longitude=-23.248139,
        )

        self.assertEqual(forecast.attrs["solar_model_name"], PHYSICAL_SOLAR_MODEL_NAME)
        self.assertEqual(forecast.attrs["forecast_model_version"], "11")
        self.assertEqual(forecast.attrs["forecast_model_status"], "candidate")
        self.assertEqual(
            forecast.attrs["solar_residual_calibration"],
            "disabled_until_uncurtailed_mppt_samples_exist",
        )
        self.assertIn("provisional_physical_configuration", forecast.attrs["solar_degradation_codes"])
        for name in (
            "ForecastSolarWatts",
            "ECMWFSolarIrradiance",
            "ForecastPVAvailableWatts",
            "ForecastPVDeliveredWatts",
            "ForecastPVCurtailedWatts",
            "ForecastPVAvailableWattsEast",
            "ForecastPVAvailableWattsSouth",
            "ForecastPVAvailableWattsWest",
            "ForecastPlaneOfArrayIrradianceSouth",
            "ForecastEffectivePlaneOfArrayIrradianceSouth",
            "ForecastEffectiveGlobalHorizontalIrradiance",
            "ECMWFSourceIntervalHours",
        ):
            self.assertIn(name, forecast)
        components = sum(
            forecast[f"ForecastPVAvailableWatts{name}"]
            for name in ("East", "South", "West")
        )
        finite = np.isfinite(forecast["ForecastPVAvailableWatts"].values)
        np.testing.assert_allclose(
            forecast["ForecastPVAvailableWatts"].values[finite],
            components.values[finite],
            atol=1.0e-5,
        )
        np.testing.assert_allclose(
            forecast["ForecastSolarWatts"].values[finite],
            forecast["ForecastPVAvailableWatts"].values[finite],
        )

    def test_physical_candidate_requires_explicit_configuration(self) -> None:
        power, solar = synthetic_inputs()
        with self.assertRaisesRegex(ValueError, "requires an explicit PhysicalSolarConfig"):
            build_forecast_dataset(
                power,
                solar,
                horizon_hours=15,
                solar_model=PHYSICAL_SOLAR_MODEL_NAME,
            )

    def test_physical_candidate_refuses_baseline_artifact_paths(self) -> None:
        with self.assertRaisesRegex(ValueError, "refuses baseline artifact paths"):
            generate(solar_model=PHYSICAL_SOLAR_MODEL_NAME)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "latest forecast->verification archive"):
                generate(
                    output_zarr=POWER_SOC_FORECAST_ARCHIVE_ZARR_PATH,
                    state_path=root / "state.json",
                    archive_zarr=root / "archive.zarr",
                    skill_zarr=root / "skill.zarr",
                    hindcast_zarr=root / "hindcast.zarr",
                    solar_model=PHYSICAL_SOLAR_MODEL_NAME,
                )
        with self.assertRaisesRegex(ValueError, "cannot be inside a Zarr store"):
            run_candidate(
                baseline_forecast_zarr=Path("missing-baseline.zarr"),
                candidate_root=Path("outer.zarr") / "candidate",
            )

    def test_generate_writes_a_complete_isolated_candidate(self) -> None:
        power, solar = synthetic_inputs()
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            power_path = root / "source" / "power.zarr"
            input_path = root / "source" / "ecmwf.nc"
            candidate_root = root / "candidate"
            power.to_zarr(power_path, mode="w", consolidated=True)
            provider_result = ForecastProviderResult(
                solar,
                {
                    "effective_provider": "legacy",
                    "fallback_reason": "",
                    "requested_provider": "legacy",
                },
            )
            with patch(
                "generate_power_soc_forecast.open_provider_solar_forecast",
                return_value=provider_result,
            ):
                result = generate(
                    power_zarr=power_path,
                    pdu_zarr=root / "source" / "missing-pdu.zarr",
                    output_zarr=candidate_root / "power_soc_forecast.zarr",
                    input_forecast=input_path,
                    state_path=candidate_root / "state.json",
                    archive_zarr=candidate_root / "archive.zarr",
                    skill_zarr=candidate_root / "skill.zarr",
                    hindcast_zarr=candidate_root / "hindcast.zarr",
                    horizon_hours=15,
                    provider="legacy",
                    max_power_age_minutes=None,
                    archive_forecast=True,
                    solar_model=PHYSICAL_SOLAR_MODEL_NAME,
                    physical_solar_config_path=CONFIG_PATH,
                    evaluation_pair_id="pair-integration-test",
                    input_snapshot_id="sha256:integration-test",
                )

            self.assertEqual(result, candidate_root / "power_soc_forecast.zarr")
            self.assertTrue((candidate_root / "archive.zarr").exists())
            self.assertTrue((candidate_root / "skill.zarr").exists())
            self.assertTrue((candidate_root / "hindcast.zarr").exists())
            self.assertTrue((candidate_root / "state.json").exists())
            with xr.open_zarr(result, chunks={}) as written:
                self.assertEqual(written.attrs["solar_model_name"], PHYSICAL_SOLAR_MODEL_NAME)
                self.assertEqual(written.attrs["evaluation_pair_id"], "pair-integration-test")
            with xr.open_zarr(candidate_root / "archive.zarr", chunks={}) as archive:
                self.assertEqual(
                    str(archive["SolarModelName"].values[-1]),
                    PHYSICAL_SOLAR_MODEL_NAME,
                )

    def test_changed_forcing_checksum_fails_before_candidate_publication(self) -> None:
        power, solar = synthetic_inputs()
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            power_path = root / "source" / "power.zarr"
            input_path = root / "source" / "ecmwf.grib2"
            candidate_root = root / "candidate"
            power.to_zarr(power_path, mode="w", consolidated=True)
            input_path.write_bytes(b"synthetic forcing")
            provider_result = ForecastProviderResult(
                solar,
                {
                    "effective_provider": "legacy",
                    "fallback_reason": "",
                    "requested_provider": "legacy",
                },
            )
            with patch(
                "generate_power_soc_forecast.open_provider_solar_forecast",
                return_value=provider_result,
            ):
                with self.assertRaisesRegex(ValueError, "checksum changed"):
                    generate(
                        power_zarr=power_path,
                        pdu_zarr=root / "source" / "missing-pdu.zarr",
                        output_zarr=candidate_root / "power_soc_forecast.zarr",
                        input_forecast=input_path,
                        state_path=candidate_root / "state.json",
                        archive_zarr=candidate_root / "archive.zarr",
                        skill_zarr=candidate_root / "skill.zarr",
                        hindcast_zarr=candidate_root / "hindcast.zarr",
                        horizon_hours=15,
                        provider="legacy",
                        max_power_age_minutes=None,
                        archive_forecast=True,
                        solar_model=PHYSICAL_SOLAR_MODEL_NAME,
                        physical_solar_config_path=CONFIG_PATH,
                        evaluation_pair_id="pair-checksum-test",
                        input_snapshot_id=f"sha256:{'0' * 64}",
                        expected_input_sha256="0" * 64,
                    )

            self.assertFalse((candidate_root / "power_soc_forecast.zarr").exists())
            self.assertFalse((candidate_root / "archive.zarr").exists())
            self.assertFalse((candidate_root / "state.json").exists())

    def test_runner_builds_and_preserves_an_exact_baseline_pair(self) -> None:
        power, solar = synthetic_inputs()
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            power_path = source / "power.zarr"
            input_path = source / "ecmwf.grib2"
            baseline_path = root / "baseline" / "power_soc_forecast.zarr"
            power.to_zarr(power_path, mode="w", consolidated=True)
            input_path.write_bytes(b"paired synthetic ECMWF fixture")
            provider_result = ForecastProviderResult(
                solar,
                {
                    "effective_provider": "legacy",
                    "fallback_reason": "",
                    "requested_provider": "legacy",
                },
            )
            with patch(
                "generate_power_soc_forecast.open_provider_solar_forecast",
                return_value=provider_result,
            ):
                generate(
                    power_zarr=power_path,
                    pdu_zarr=source / "missing-pdu.zarr",
                    output_zarr=baseline_path,
                    input_forecast=input_path,
                    state_path=root / "baseline" / "state.json",
                    archive_zarr=root / "baseline" / "archive.zarr",
                    skill_zarr=None,
                    hindcast_zarr=None,
                    latitude=64.829694,
                    longitude=-23.248139,
                    horizon_hours=15,
                    provider="legacy",
                    max_power_age_minutes=None,
                    archive_forecast=True,
                )
                candidate_path = run_candidate(
                    baseline_forecast_zarr=baseline_path,
                    candidate_root=root / "candidate",
                    power_zarr=power_path,
                    pdu_zarr=source / "missing-pdu.zarr",
                    physical_config=CONFIG_PATH,
                )

            with xr.open_zarr(candidate_path, chunks={}) as candidate:
                pair_id = str(candidate.attrs["evaluation_pair_id"])
                candidate_signature = str(candidate.attrs["publication_signature"])
                candidate_baseline_signature = candidate.attrs["baseline_publication_signature"]
            with xr.open_zarr(baseline_path, chunks={}) as baseline:
                self.assertEqual(
                    candidate_baseline_signature,
                    baseline.attrs["publication_signature"],
                )
            pair_root = root / "candidate" / "pairs" / pair_id / candidate_signature
            self.assertTrue((pair_root / "baseline_forecast.zarr").exists())
            self.assertTrue((pair_root / "candidate_forecast.zarr").exists())
            self.assertTrue((pair_root / "pair_manifest.json").exists())
            manifest = json.loads((pair_root / "pair_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["pair_status"], "complete")

    def test_archive_preserves_candidate_model_identity(self) -> None:
        issue = pd.Timestamp("2026-06-21T12:00:00")
        forecast = xr.Dataset(
            {"ForecastSolarWatts": (("time",), [0.0, 100.0])},
            coords={"time": [issue, issue + pd.Timedelta(hours=3)]},
            attrs={
                "initial_soc_time": issue.isoformat(),
                "forecast_model_name": "aps_soc_energy_balance_v11_candidate",
                "forecast_model_version": "11",
                "forecast_model_contract_id": "forecast-model-v1-test",
                "solar_model_name": PHYSICAL_SOLAR_MODEL_NAME,
                "solar_model_version": "1",
                "solar_model_contract_id": "solar-physical-v1-test",
                "solar_feature_set_version": "test-features",
                "solar_model_status": "candidate",
                "evaluation_pair_id": "pair-test",
                "input_snapshot_id": "sha256:test",
            },
        )

        archived = _archive_row_from_forecast(forecast)

        self.assertEqual(str(archived["SolarModelName"].values[0]), PHYSICAL_SOLAR_MODEL_NAME)
        self.assertEqual(str(archived["ForecastModelVersion"].values[0]), "11")
        self.assertEqual(str(archived["EvaluationPairID"].values[0]), "pair-test")

    def test_candidate_skill_does_not_score_censored_available_pv(self) -> None:
        issue = pd.Timestamp("2026-06-21T12:00:00")
        forecast = xr.Dataset(
            {
                "BatterySOCForecast": (("time",), [70.0, 69.0]),
                "ForecastSolarWatts": (("time",), [0.0, 1000.0]),
                "ForecastLoadWatts": (("time",), [100.0, 100.0]),
            },
            coords={"time": [issue, issue + pd.Timedelta(hours=3)]},
            attrs={
                "initial_soc_time": issue.isoformat(),
                "ecmwf_cycle_time": issue.isoformat(),
                "solar_model_name": PHYSICAL_SOLAR_MODEL_NAME,
            },
        )
        archive = _archive_row_from_forecast(forecast)
        observed_times = pd.date_range(issue, issue + pd.Timedelta(hours=3), freq="10min")
        power = xr.Dataset(
            {
                "BatterySOC": (("time",), np.linspace(70.0, 69.0, len(observed_times))),
                "SolarWatts_East": (("time",), np.full(len(observed_times), 100.0)),
                "ACOutputWatts": (("time",), np.full(len(observed_times), 90.0)),
                "DCInverterWatts": (("time",), np.full(len(observed_times), 10.0)),
            },
            coords={"time": observed_times},
        )

        skill = build_forecast_skill_dataset(
            archive,
            power,
            window_hours=6,
            retention_days=1,
            freq="1h",
        )

        self.assertEqual(
            skill.attrs["solar_verification_status"],
            "excluded_available_power_observations_are_censored",
        )
        self.assertFalse(np.isfinite(skill["ForecastSolarMAE24h"].values).any())

        frame = pd.DataFrame(
            {name: power[name].values for name in power.data_vars},
            index=observed_times,
        )
        previous_metrics = evaluate_previous_forecast(forecast, frame)
        archive_metrics = evaluate_forecast_archive(archive, frame)
        independent_metrics = evaluate_independent_forecast_archive(archive, frame)
        for metrics in (previous_metrics, archive_metrics, independent_metrics):
            self.assertNotIn("solar_mae_w", metrics)
            self.assertNotIn("solar_sample_count", metrics)
            self.assertEqual(
                metrics["solar_verification_status"],
                "excluded_available_power_observations_are_censored",
            )

    def test_candidate_skill_scores_available_pv_only_when_all_arrays_are_mpp_active(self) -> None:
        issue = pd.Timestamp("2026-06-21T12:00:00")
        forecast = xr.Dataset(
            {
                "BatterySOCForecast": (("time",), [70.0, 69.0]),
                "ForecastSolarWatts": (("time",), [900.0, 900.0]),
                "ForecastLoadWatts": (("time",), [100.0, 100.0]),
            },
            coords={"time": [issue, issue + pd.Timedelta(hours=3)]},
            attrs={
                "initial_soc_time": issue.isoformat(),
                "ecmwf_cycle_time": issue.isoformat(),
                "solar_model_name": PHYSICAL_SOLAR_MODEL_NAME,
                "solar_power_semantics": "available_dc_before_battery_acceptance",
            },
        )
        archive = _archive_row_from_forecast(forecast)
        observed_times = pd.date_range(issue, issue + pd.Timedelta(hours=3), freq="10min")
        power = xr.Dataset(
            {
                "BatterySOC": (("time",), np.linspace(70.0, 69.0, len(observed_times))),
                "SolarWatts_East": (("time",), np.full(len(observed_times), 300.0)),
                "SolarWatts_South": (("time",), np.full(len(observed_times), 300.0)),
                "SolarWatts_West": (("time",), np.full(len(observed_times), 300.0)),
                "SolarMPPMode_East": (("time",), np.full(len(observed_times), 2.0)),
                "SolarMPPMode_South": (("time",), np.full(len(observed_times), 2.0)),
                "SolarMPPMode_West": (("time",), np.full(len(observed_times), 2.0)),
                "ACOutputWatts": (("time",), np.full(len(observed_times), 90.0)),
                "DCInverterWatts": (("time",), np.full(len(observed_times), 10.0)),
            },
            coords={"time": observed_times},
        )

        skill = build_forecast_skill_dataset(
            archive,
            power,
            window_hours=6,
            retention_days=1,
            freq="1h",
        )

        self.assertEqual(skill.attrs["solar_verification_status"], "eligible_available_power_mpp_active")
        self.assertTrue(np.isfinite(skill["ForecastSolarMAE24h"].values).any())

        frame = pd.DataFrame(
            {name: power[name].values for name in power.data_vars},
            index=observed_times,
        )
        previous_metrics = evaluate_previous_forecast(forecast, frame)
        archive_metrics = evaluate_forecast_archive(archive, frame)
        independent_metrics = evaluate_independent_forecast_archive(archive, frame)
        for metrics in (previous_metrics, archive_metrics, independent_metrics):
            self.assertEqual(metrics["solar_verification_status"], "eligible_available_power_mpp_active")
            self.assertEqual(metrics["solar_mae_w"], 0.0)


if __name__ == "__main__":
    unittest.main()
