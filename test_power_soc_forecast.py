from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr

from generate_power_soc_forecast import (
    _apply_soc_bias_corrections,
    _active_forecast_contract_id,
    _atomic_write_archive,
    _extend_irradiance_with_diurnal_persistence,
    _filter_active_forecast_contract,
    _load_mode_signature,
    _mode_learning_status,
    _pdu_active_kits,
    _repair_dc_only_registry,
    _resolve_load_mode,
    append_forecast_archive,
    adaptive_calibration_state_id,
    build_forecast_dataset,
    build_forecast_skill_dataset,
    build_historical_load_forecast,
    build_soc_hindcast_dataset,
    calibrated_solar_factor_profile,
    evaluate_forecast_archive,
    evaluate_independent_forecast_archive,
    forecast_publication_signature,
    legacy_solar_model_contract_id,
    resolve_ecmwf_cycle_hour,
    solar_calibration_contract_id,
    solar_irradiance_from_ssrd,
    validate_power_input_freshness,
    validate_soc_physical_consistency,
    write_immutable_issue_snapshot,
    ISSUE_SNAPSHOT_DIGEST_ALGORITHM,
    ISSUE_SNAPSHOT_DIGEST_MARKER,
)
from generate_power_soc_ensemble import (
    ENSEMBLE_POWER_MIN_HISTORY_DAYS,
    _ensemble_refresh_reasons,
    _power_at_or_before_cutoff,
    append_ensemble_archive,
    apply_operational_soc_threshold,
    build_ensemble_dataset,
    build_ensemble_skill_dataset,
    generate as generate_ensemble_forecast,
)
from grouped_timeseries import (
    PDU_WATT_FIELDS,
    SUMMARY_LAYOUTS,
    _active_panels,
    _trace_plot_values,
    build_power_display_summary_dataset,
    build_power_forecast_info,
    build_power_verification_guidance,
    build_summary_plotly,
    merge_operating_scenarios_into_display_summary,
    prepare_summary_dataset,
)
from power_soc_thresholds import (
    MINIMUM_OPERATIONAL_SOC_PCT,
    MINIMUM_OPERATIONAL_SOC_REFERENCE_LABEL,
    SOC_BELOW_THRESHOLD_BRIER_FIELD,
    SOC_BELOW_THRESHOLD_PROBABILITY_FIELD,
)
from power_load_dynamics import PHASE_CODES, PHASE_STARTUP


class PowerSocForecastTests(unittest.TestCase):
    def test_browser_power_briefing_documents_system_and_operating_scenarios(self) -> None:
        from app import _browser_power_briefing_markup

        markup = _browser_power_briefing_markup(
            xr.Dataset(
                attrs={
                    "operating_planning_status": "ready",
                    "operating_current_mode_label": "DC-Only",
                    "operating_optimization_horizon_hours": "96",
                }
            )
        )

        self.assertIn("System as-is", markup)
        self.assertIn("latest confirmed finite instrument state", markup)
        self.assertIn("states are never blended", markup)
        self.assertNotIn("recent measured whole-station load", markup)
        self.assertIn("Instrument scenarios", markup)
        self.assertIn("DC-Only", markup)
        self.assertIn("CL61 + HATPRO + Radar", markup)
        self.assertIn("HATPRO + Radar", markup)
        self.assertIn("40% operational minimum", markup)

    def test_forecast_info_documents_finite_state_load_uncertainty(self) -> None:
        ensemble = build_power_forecast_info("soc_ecmwf_forecast")
        scenarios = build_power_forecast_info("operating_plan_scenarios")

        self.assertIsNotNone(ensemble)
        self.assertIsNotNone(scenarios)
        assert ensemble is not None
        assert scenarios is not None
        ensemble_text = str(ensemble["implementation"])
        scenario_text = str(scenarios["implementation"])
        self.assertIn("latest confirmed finite instrument state", ensemble_text)
        self.assertIn("only recurrent startup or fan phases", ensemble_text)
        self.assertIn("states are never blended", scenario_text)
        self.assertNotIn("recent measured whole-station load", ensemble_text)
        self.assertNotIn("load residuals", ensemble_text)

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.tmp_archive_path = Path(self._tmp.name) / "power_soc_forecast_archive.zarr"
        self.tmp_ensemble_archive_path = Path(self._tmp.name) / "power_soc_ensemble_archive.zarr"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_archive_writer_coerces_mixed_object_metadata_to_text(self) -> None:
        archive = xr.Dataset(
            {
                "LoadMode": (("issue_time",), np.asarray(["dc_only", 109], dtype=object)),
                "BatterySOCForecast": (
                    ("issue_time", "forecast_step"),
                    np.asarray([[90.0], [89.0]], dtype=np.float32),
                ),
            },
            coords={
                "issue_time": np.asarray(
                    ["2026-07-25T00:00:00", "2026-07-25T03:00:00"],
                    dtype="datetime64[ns]",
                ),
                "forecast_step": np.asarray([0], dtype=np.int32),
            },
        )

        _atomic_write_archive(archive, self.tmp_archive_path)
        written = xr.open_zarr(self.tmp_archive_path, consolidated=True).load()

        self.assertEqual(written["LoadMode"].values.tolist(), ["dc_only", "109"])

    def test_issue_snapshot_is_idempotent_and_rejects_conflicting_content(self) -> None:
        path = Path(self._tmp.name) / "issues" / "cycle" / "forecast.zarr"
        issue = xr.Dataset(
            {"BatterySOCForecast": (("time",), [70.0])},
            coords={"time": [np.datetime64("2026-07-25T00:00:00")]},
            attrs={
                "forecast_verification_eligible": "true",
                "forecast_refresh_kind": "ecmwf_cycle",
                "publication_signature": "signature-one",
            },
        )

        self.assertEqual(write_immutable_issue_snapshot(issue, path), path)
        self.assertEqual(write_immutable_issue_snapshot(issue, path), path)
        marker = json.loads(
            (path / ISSUE_SNAPSHOT_DIGEST_MARKER).read_text(encoding="utf-8")
        )
        self.assertEqual(marker["digestAlgorithm"], ISSUE_SNAPSHOT_DIGEST_ALGORITHM)
        self.assertRegex(marker["contentDigest"], r"^sha256:[0-9a-f]{64}$")
        same_signature_conflict = issue.copy(deep=True)
        same_signature_conflict["BatterySOCForecast"][:] = 71.0
        with self.assertRaisesRegex(ValueError, "different forecast"):
            write_immutable_issue_snapshot(same_signature_conflict, path)
        conflicting = issue.copy(deep=True)
        conflicting.attrs["publication_signature"] = "signature-two"
        with self.assertRaisesRegex(ValueError, "different forecast"):
            write_immutable_issue_snapshot(conflicting, path)
        persisted_file = next(
            candidate
            for candidate in path.rglob("*")
            if candidate.is_file() and candidate.name != ISSUE_SNAPSHOT_DIGEST_MARKER
        )
        persisted_file.write_bytes(persisted_file.read_bytes() + b"corrupt")
        with self.assertRaisesRegex(ValueError, "content digest"):
            write_immutable_issue_snapshot(issue, path)

    def test_ssrd_accumulation_converts_to_irradiance(self) -> None:
        times = pd.date_range("2026-07-10T00:00:00", periods=4, freq="3h")
        ds = xr.Dataset(
            {"ssrd": (("time",), [0.0, 3 * 3600 * 100.0, 3 * 3600 * 250.0, 3 * 3600 * 400.0])},
            coords={"time": times},
        )

        irradiance = solar_irradiance_from_ssrd(ds)

        self.assertEqual(list(irradiance.index), list(times[1:]))
        np.testing.assert_allclose(irradiance.to_numpy(), [100.0, 150.0, 150.0])

    def test_solar_factor_profile_uses_defaults_when_archive_has_no_verification_rows(self) -> None:
        observed_times = pd.date_range("2026-07-22T06:00:00", periods=3, freq="10min")
        frame = pd.DataFrame({"SolarWatts_East": [10.0, 20.0, 30.0]}, index=observed_times)
        forecast_times = pd.date_range("2026-07-22T08:00:00", periods=4, freq="6h")

        profile, corrections = calibrated_solar_factor_profile(
            2.5,
            xr.Dataset(),
            frame,
            forecast_times,
            issue_time=pd.Timestamp("2026-07-22T08:00:00"),
        )

        np.testing.assert_allclose(profile.to_numpy(), np.full(4, 2.5))
        self.assertTrue(all(value == 1.0 for value in corrections.values()))

    def test_auto_long_cycle_selects_the_latest_likely_complete_cycle(self) -> None:
        self.assertEqual(resolve_ecmwf_cycle_hour("auto", now=datetime(2026, 7, 16, 9, tzinfo=timezone.utc)), 0)
        self.assertEqual(resolve_ecmwf_cycle_hour("auto", now=datetime(2026, 7, 16, 21, tzinfo=timezone.utc)), 12)
        self.assertEqual(resolve_ecmwf_cycle_hour("auto", now=datetime(2026, 7, 16, 3, tzinfo=timezone.utc)), 12)

    def test_stale_power_input_is_rejected_before_forecast_publication(self) -> None:
        latest = pd.Timestamp("2026-07-16T12:00:00")
        power = xr.Dataset(
            {"BatterySOC": (("time",), [65.0])},
            coords={"time": [latest]},
        )

        anchor_time, anchor_soc = validate_power_input_freshness(
            power,
            max_age_minutes=20,
            now=latest + pd.Timedelta(minutes=19),
        )
        self.assertEqual(anchor_time, latest)
        self.assertEqual(anchor_soc, 65.0)

        with self.assertRaisesRegex(ValueError, "stale SOC/load input"):
            validate_power_input_freshness(
                power,
                max_age_minutes=20,
                now=latest + pd.Timedelta(minutes=21),
            )

    def test_long_forecast_tail_repeats_diurnal_shape_instead_of_flatlining(self) -> None:
        times = pd.date_range("2026-07-15T00:00:00", periods=17, freq="3h")
        values = np.tile([0.0, 20.0, 100.0, 300.0, 500.0, 250.0, 50.0, 0.0], 3)[: len(times)]
        source = pd.Series(values, index=times)

        extended, hours = _extend_irradiance_with_diurnal_persistence(
            source,
            times[-1] + pd.Timedelta(hours=6),
        )

        self.assertEqual(hours, 6.0)
        self.assertEqual(float(extended.iloc[-2]), float(source.loc[times[-1] - pd.Timedelta(hours=21)]))
        self.assertEqual(float(extended.iloc[-1]), float(source.loc[times[-1] - pd.Timedelta(hours=18)]))
        self.assertNotEqual(float(extended.iloc[-2]), float(extended.iloc[-1]))

    def test_soc_bias_correction_preserves_actual_initial_anchor(self) -> None:
        issue = pd.Timestamp("2026-07-16T06:00:00")
        times = pd.DatetimeIndex([issue, issue + pd.Timedelta(hours=3), issue + pd.Timedelta(hours=6)])
        forecast = pd.DataFrame(
            {
                "BatterySOCForecast": [63.0, 62.0, 61.0],
                "ForecastSolarWatts": [0.0, 0.0, 0.0],
                "ForecastLoadWatts": [100.0, 100.0, 100.0],
            },
            index=times,
        )

        corrected = _apply_soc_bias_corrections(
            forecast,
            {"0_6h": 5.0, "6_24h": -2.0},
            issue_time=issue,
        )

        self.assertEqual(float(corrected["BatterySOCForecast"].iloc[0]), 63.0)
        # The residual is continuous from the observed anchor and cannot
        # manufacture an SOC rise while the physical path is discharging.
        self.assertEqual(float(corrected["BatterySOCForecast"].iloc[1]), 63.0)
        self.assertEqual(float(corrected["BatterySOCForecast"].iloc[2]), 63.0)
        self.assertEqual(float(corrected["ForecastSOCBiasCorrectionPctPoints"].iloc[0]), 0.0)
        self.assertTrue(np.all(np.diff(corrected["BatterySOCForecast"].values) <= 0.0))

    def test_soc_bias_correction_is_continuous_across_lead_bucket_boundaries(self) -> None:
        issue = pd.Timestamp("2026-07-16T00:00:00")
        times = pd.date_range(issue, periods=10, freq="3h")
        # A negative residual may attenuate charging continuously, but cannot
        # reverse it or create a bucket-boundary jump.
        forecast = pd.DataFrame(
            {
                "BatterySOCForecast": np.linspace(50.0, 59.0, len(times)),
                "ForecastSolarWatts": np.full(len(times), 500.0),
                "ForecastLoadWatts": np.full(len(times), 100.0),
            },
            index=times,
        )

        corrected = _apply_soc_bias_corrections(
            forecast,
            {"0_6h": -2.0, "6_24h": -4.0, "24_48h": -5.0},
            issue_time=issue,
        )

        residual = corrected["ForecastSOCBiasCorrectionPctPoints"].to_numpy()
        self.assertAlmostEqual(float(residual[0]), 0.0)
        self.assertAlmostEqual(float(residual[2]), -2.0)
        self.assertAlmostEqual(float(residual[8]), -4.0)
        # No bucket boundary is represented by an instantaneous same-time jump.
        self.assertLessEqual(float(np.max(np.abs(np.diff(residual)))), 1.0)
        self.assertTrue(np.all(np.diff(corrected["BatterySOCForecast"].values) >= 0.0))
        self.assertTrue(
            np.all(
                np.diff(corrected["BatterySOCForecast"].values)
                <= np.diff(forecast["BatterySOCForecast"].values) + 1.0e-9
            )
        )

    def test_soc_residual_preserves_zero_charge_and_discharge_energy_directions(self) -> None:
        issue = pd.Timestamp("2026-07-16T00:00:00")
        times = pd.date_range(issue, periods=3, freq="3h")

        def corrected(raw, charge, discharge, correction):
            frame = pd.DataFrame(
                {
                    "BatterySOCForecast": raw,
                    "ForecastBatteryChargeInputWatts": charge,
                    "ForecastBatteryDischargeOutputWatts": discharge,
                    "ForecastSolarWatts": np.zeros(3),
                    "ForecastLoadWatts": np.zeros(3),
                },
                index=times,
            )
            return frame, _apply_soc_bias_corrections(
                frame, {"0_6h": correction}, issue_time=issue
            )

        physical, zero_flow = corrected([50.0, 50.0, 50.0], [0.0] * 3, [0.0] * 3, 5.0)
        np.testing.assert_allclose(zero_flow["BatterySOCForecast"], physical["BatterySOCForecast"])

        physical, charging = corrected(
            [50.0, 52.0, 54.0], [0.0, 200.0, 200.0], [0.0] * 3, -3.0
        )
        self.assertTrue(np.all(np.diff(charging["BatterySOCForecast"]) >= 0.0))
        self.assertTrue(
            np.all(np.diff(charging["BatterySOCForecast"]) <= np.diff(physical["BatterySOCForecast"]))
        )

        physical, discharging = corrected(
            [50.0, 48.0, 46.0], [0.0] * 3, [0.0, 200.0, 200.0], 3.0
        )
        self.assertTrue(np.all(np.diff(discharging["BatterySOCForecast"]) <= 0.0))
        self.assertTrue(
            np.all(
                np.abs(np.diff(discharging["BatterySOCForecast"]))
                <= np.abs(np.diff(physical["BatterySOCForecast"]))
            )
        )
        validate_soc_physical_consistency(charging)
        validate_soc_physical_consistency(discharging)

    def test_soc_validator_uses_net_battery_flow_when_charge_and_discharge_coexist(self) -> None:
        times = pd.date_range("2026-07-16T00:00:00", periods=2, freq="1h")
        forecast = pd.DataFrame(
            {
                "BatterySOCForecast": [50.0, 51.0],
                # Aggregated intervals can contain both flows; the net flow is
                # charging even though discharge is non-zero and solar<load.
                "ForecastBatteryChargeInputWatts": [0.0, 200.0],
                "ForecastBatteryDischargeOutputWatts": [0.0, 100.0],
                "ForecastSolarWatts": [0.0, 0.0],
                "ForecastLoadWatts": [100.0, 100.0],
            },
            index=times,
        )

        validate_soc_physical_consistency(forecast)

        forecast.loc[times[-1], "ForecastBatteryChargeInputWatts"] = 50.0
        with self.assertRaisesRegex(ValueError, "rises without a net-charging"):
            validate_soc_physical_consistency(forecast)

        forecast["BatterySOCForecast"] = [50.0, 49.0]
        forecast.loc[times[-1], "ForecastBatteryChargeInputWatts"] = 200.0
        with self.assertRaisesRegex(ValueError, "falls without a net-discharging"):
            validate_soc_physical_consistency(forecast)

    def test_24h_forecast_panel_is_future_model_output_not_observed_soc(self) -> None:
        panel = next(panel for panel in SUMMARY_LAYOUTS["power"] if panel.key == "soc_24h_forecast")

        self.assertEqual(panel.label, "SOC Next 24 h Forecast")
        self.assertEqual([trace.var for trace in panel.traces], ["SystemAsIsDecisionSOCP50"])
        trace = panel.traces[0]
        self.assertEqual(trace.display_horizon_hours, 24.0)

        times = pd.date_range("2026-07-16T06:00:00", periods=33, freq="3h")
        values = np.linspace(60.0, 40.0, len(times))
        rendered_times, _ = _trace_plot_values(times, values, max_time_samples=100, trace=trace)

        self.assertEqual(rendered_times.min(), times.min())
        self.assertLessEqual(rendered_times.max(), times.min() + pd.Timedelta(hours=24))

    def test_mobile_summary_maps_compact_cumulative_energy_fields(self) -> None:
        times = pd.date_range("2026-07-16T00:00:00", periods=3, freq="1h")
        display = xr.Dataset(
            {
                "BatterySOC": (("time",), [70.0, 69.0, 68.0]),
                "PowerDisplaySolarYield_East": (("time",), [0.0, 0.1, 0.2]),
                "PowerDisplaySolarYield_South": (("time",), [0.0, 0.2, 0.4]),
                "PowerDisplaySolarYield_West": (("time",), [0.0, 0.1, 0.3]),
                "PowerDisplayCumulativePowerGeneratedTotal": (("time",), [0.0, 0.4, 0.9]),
                "PowerDisplayCumulativePowerUtilised": (("time",), [0.0, 0.05, 0.1]),
            },
            coords={"time": times},
        )

        prepared = prepare_summary_dataset(display, "power")
        panel = next(panel for panel in SUMMARY_LAYOUTS["power"] if panel.key == "cumulative_power")

        self.assertEqual(panel.label, "Cumulative Energy & State of Charge")
        for field_name in (
            "SolarYield_East",
            "SolarYield_South",
            "SolarYield_West",
            "CumulativePowerGeneratedTotal",
            "CumulativePowerUtilised",
        ):
            self.assertIn(field_name, prepared)
            self.assertTrue(np.isfinite(prepared[field_name].values).any())

    def test_build_forecast_dataset_integrates_soc(self) -> None:
        power_times = pd.date_range("2026-07-09T00:00:00", periods=25, freq="1h")
        power = xr.Dataset(
            {
                "BatterySOC": (("time",), np.linspace(72.0, 70.0, len(power_times))),
                "SolarWatts_East": (("time",), np.full(len(power_times), 50.0)),
                "SolarWatts_South": (("time",), np.full(len(power_times), 50.0)),
                "SolarWatts_West": (("time",), np.full(len(power_times), 50.0)),
                "ACOutputWatts": (("time",), np.full(len(power_times), 20.0)),
                "DCInverterWatts": (("time",), np.full(len(power_times), 10.0)),
            },
            coords={"time": power_times},
        )
        forecast_times = pd.date_range(power_times[-1], periods=5, freq="3h")
        solar = xr.Dataset(
            {"ssrd": (("time",), np.arange(len(forecast_times), dtype=float) * 3 * 3600 * 200.0)},
            coords={"time": forecast_times},
        )

        forecast = build_forecast_dataset(power, solar, horizon_hours=12, capacity_kwh=26.0)

        self.assertIn("BatterySOCForecast", forecast)
        self.assertIn("ECMWFSolarIrradiance", forecast)
        self.assertEqual(pd.Timestamp(forecast["time"].values[0]), power_times[-1])
        self.assertAlmostEqual(float(forecast["BatterySOCForecast"].values[0]), 70.0)
        self.assertGreaterEqual(float(forecast["BatterySOCForecast"].min()), 0.0)
        self.assertLessEqual(float(forecast["BatterySOCForecast"].max()), 100.0)
        self.assertEqual(forecast.attrs["forecast_horizon_hours"], "12")
        self.assertEqual(forecast.attrs["load_model"], "finite_controlled_state_phases_v10")
        self.assertEqual(forecast.attrs["load_model_version"], "10")
        self.assertIn("battery_charge_efficiency", forecast.attrs)
        self.assertIn("battery_discharge_efficiency", forecast.attrs)
        self.assertEqual(
            forecast.attrs["load_anchor_method"],
            "learned_exact_state_phase:steady",
        )
        self.assertEqual(forecast.attrs["load_state_contract"], "finite_operating_state_phases_v2")
        self.assertEqual(
            forecast.attrs["load_state_hold_policy"],
            "hold_confirmed_state_allow_detected_phase_or_explicit_schedule_transition",
        )
        self.assertIn("ForecastLoadP10Watts", forecast)
        self.assertIn("ForecastLoadP50Watts", forecast)
        self.assertIn("ForecastLoadP90Watts", forecast)
        self.assertIn("ForecastLoadPhaseCode", forecast)
        self.assertTrue(forecast.attrs["solar_calibration_contract_id"].startswith("solar-calibration-v1-"))
        self.assertEqual(float(forecast.attrs["minimum_operational_soc_pct"]), 40.0)
        self.assertEqual(forecast.attrs["scenario_loads_w"], "100,200,300,400,500,600")
        self.assertEqual(forecast.attrs["scenario_solar_mode"], "ecmwf")
        self.assertIn("ForecastLoadMAERecent", forecast)
        self.assertIn("ForecastSOCMAE_0_6h", forecast)
        self.assertIn("ForecastSkillSampleCount", forecast)
        for load_w in (100, 200, 300, 400, 500, 600):
            self.assertIn(f"BatterySOCForecast_Load{load_w}W", forecast)
            self.assertAlmostEqual(float(forecast[f"BatterySOCForecast_Load{load_w}W"].values[0]), 70.0)

    def test_forecast_publication_signature_ignores_generation_time(self) -> None:
        forecast = xr.Dataset(
            {"BatterySOCForecast": (("time",), [80.0, 79.0])},
            coords={"time": pd.date_range("2026-07-20", periods=2, freq="1h")},
            attrs={
                "initial_soc_time": "2026-07-20T00:04:00",
                "initial_soc_pct": "80",
                "forecast_load_w": "500",
                "load_mode": "DC + CL61",
                "solar_calibration_contract_id": "solar-a",
                "battery_usable_capacity_kwh": "26",
                "battery_charge_efficiency": "0.92",
                "battery_discharge_efficiency": "0.92",
                "load_model_version": "7",
                "generated_at_utc": "2026-07-20T00:05:00+00:00",
            },
        )
        first = forecast_publication_signature(forecast)
        forecast.attrs["generated_at_utc"] = "2099-01-01T00:00:00+00:00"

        self.assertEqual(forecast_publication_signature(forecast), first)
        forecast.attrs["forecast_load_w"] = "550"
        self.assertNotEqual(forecast_publication_signature(forecast), first)

    def test_forecast_publication_signature_tracks_load_phase_dynamics(self) -> None:
        forecast = xr.Dataset(
            attrs={
                "initial_soc_time": "2026-07-20T00:00:00",
                "initial_soc_pct": "80",
                "forecast_load_w": "275",
                "load_mode_signature": "dc_cl61",
                "load_current_phase": "fan_low",
                "load_state_dynamics": '{"phase_profiles":{"fan_low":{"p50_w":275}}}',
                "load_model_version": "10",
            }
        )
        first = forecast_publication_signature(forecast)

        forecast.attrs["load_current_phase"] = "fan_high"
        self.assertNotEqual(forecast_publication_signature(forecast), first)
        forecast.attrs["load_current_phase"] = "fan_low"
        forecast.attrs["load_state_dynamics"] = '{"phase_profiles":{"fan_low":{"p50_w":290}}}'
        self.assertNotEqual(forecast_publication_signature(forecast), first)

    def test_forecast_publication_signature_tracks_battery_power_limits(self) -> None:
        forecast = xr.Dataset(
            attrs={
                "initial_soc_time": "2026-07-20T00:00:00",
                "initial_soc_pct": "80",
                "forecast_load_w": "700",
                "load_mode_signature": "dc_cl61",
                "ecmwf_cycle_time": "2026-07-20T00:00:00",
                "solar_calibration_contract_id": "solar-v1",
                "battery_usable_capacity_kwh": "26",
                "battery_charge_efficiency": "0.92",
                "battery_discharge_efficiency": "0.92",
                "battery_parasitic_load_w": "0",
                "battery_max_charge_w": "1800",
                "battery_max_discharge_w": "1200",
                "load_model_version": "7",
            }
        )
        first = forecast_publication_signature(forecast)

        forecast.attrs["battery_max_discharge_w"] = "1500"

        self.assertNotEqual(forecast_publication_signature(forecast), first)

    def test_solar_calibration_contract_is_stable_and_input_sensitive(self) -> None:
        first = solar_calibration_contract_id(2.0, {"0_6h": 1.0, "6_24h": 0.9})
        reordered = solar_calibration_contract_id(2.0, {"6_24h": 0.9, "0_6h": 1.0})

        self.assertEqual(first, reordered)
        self.assertNotEqual(first, solar_calibration_contract_id(2.1, {"0_6h": 1.0, "6_24h": 0.9}))

    def test_adaptive_calibration_does_not_fragment_the_solar_model_contract(self) -> None:
        model_contract = legacy_solar_model_contract_id()
        first = adaptive_calibration_state_id(
            {
                "solar_calibration_contract_id": solar_calibration_contract_id(
                    2.0, {"0_6h": 1.0}
                ),
                "load_bias_correction_w": "0",
            }
        )
        second = adaptive_calibration_state_id(
            {
                "solar_calibration_contract_id": solar_calibration_contract_id(
                    2.2, {"0_6h": 0.9}
                ),
                "load_bias_correction_w": "0",
            }
        )

        self.assertEqual(model_contract, legacy_solar_model_contract_id())
        self.assertNotEqual(first, second)

    def test_adaptive_calibration_identity_tracks_load_state_not_model_contract(self) -> None:
        base = {
            "solar_calibration_contract_id": "solar-state-one",
            "load_residual_model_contract_id": "load-residual-contract-one",
            "load_residual_state_digest": "sha256:state-one",
            "load_mode_signature": "DC-Only+CL61",
            "load_state_dynamics_signature": "phase-profile-one",
            "load_mode_registry": '{"DC-Only":{"learned_level_w":220}}',
            "forecast_load_w": "315",
            "forecast_load_p10_w": "300",
            "forecast_load_p50_w": "315",
            "forecast_load_p90_w": "340",
        }
        first = adaptive_calibration_state_id(base)

        changed = dict(base)
        changed["load_residual_state_digest"] = "sha256:state-two"
        self.assertNotEqual(first, adaptive_calibration_state_id(changed))

        changed = dict(base)
        changed["load_state_dynamics_signature"] = "phase-profile-two"
        self.assertNotEqual(first, adaptive_calibration_state_id(changed))

    def test_physical_soc_validation_accounts_for_battery_parasitic_load(self) -> None:
        forecast = pd.DataFrame(
            {
                "BatterySOCForecast": [50.0, 49.9],
                "ForecastSolarWatts": [100.0, 100.0],
                "ForecastLoadWatts": [100.0, 100.0],
            },
            index=pd.date_range("2026-07-01", periods=2, freq="1h"),
        )

        validate_soc_physical_consistency(forecast, parasitic_load_w=25.0)
        with self.assertRaisesRegex(ValueError, "falls without a net-discharging"):
            validate_soc_physical_consistency(forecast, parasitic_load_w=0.0)

    def test_first_nonempty_forecast_contract_excludes_blank_legacy_rows(self) -> None:
        semantic_archive = {
            "ForecastSystemVersion": ["", "system-v1"],
            "FeatureSetVersion": ["", "features-v1"],
            "FeatureSetDigest": ["", "digest-v1"],
            "ForecastCodeRevision": ["", "revision-v1"],
            "CandidateLane": ["", "baseline"],
            "LocalFeatureContractID": ["", "local-v1"],
            "BaselineControlContractID": ["", "control-v1"],
            "BaselineControlSystemVersion": ["", "power-v10"],
        }
        archive = xr.Dataset(
            {
                "ForecastModelContractID": (("issue_time",), ["", "contract-v1"]),
                **{
                    name: (("issue_time",), values)
                    for name, values in semantic_archive.items()
                },
            },
            coords={"issue_time": pd.date_range("2026-07-01", periods=2, freq="1h")},
        )
        rows = pd.DataFrame(
            {
                "forecast_model_contract_id": ["", "contract-v1"],
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
                "value": [99.0, 1.0],
            }
        )

        self.assertEqual(_active_forecast_contract_id(archive), "contract-v1")
        filtered = _filter_active_forecast_contract(rows, archive)

        self.assertEqual(filtered["value"].tolist(), [1.0])

    def test_deterministic_evidence_filters_the_complete_semantic_identity(self) -> None:
        issue = pd.Timestamp("2026-07-10T00:00:00")
        semantic = {
            "forecast_model_contract_id": "contract-v12",
            "forecast_system_version": "power-v12",
            "feature_set_version": "features-v4",
            "forecast_code_revision": "revision-v12",
            "candidate_lane": "D_physical_solar_load_residual",
            "local_feature_contract_id": "local-v1",
            "baseline_control_contract_id": "baseline-v10",
            "baseline_control_system_version": "power-v10",
            "forecast_refresh_kind": "ecmwf_cycle",
            "forecast_verification_eligible": "true",
            "independent_cycle": "true",
            "load_model_version": "10",
        }
        old = xr.Dataset(
            {"BatterySOCForecast": (("time",), [60.0, 10.0])},
            coords={"time": [issue, issue + pd.Timedelta(hours=6)]},
            attrs={
                **semantic,
                "initial_soc_time": issue.isoformat(),
                "ecmwf_cycle_time": issue.isoformat(),
                "source_cycle_set_id": "cycle-old",
                "feature_set_digest": "digest-old",
            },
        )
        new_issue = issue + pd.Timedelta(hours=3)
        new = xr.Dataset(
            {"BatterySOCForecast": (("time",), [60.0, 58.0])},
            coords={"time": [new_issue, new_issue + pd.Timedelta(hours=6)]},
            attrs={
                **semantic,
                "initial_soc_time": new_issue.isoformat(),
                "ecmwf_cycle_time": new_issue.isoformat(),
                "source_cycle_set_id": "cycle-new",
                "feature_set_digest": "digest-new",
            },
        )
        append_forecast_archive(old, self.tmp_archive_path)
        archive = append_forecast_archive(new, self.tmp_archive_path)
        frame = pd.DataFrame(
            {"BatterySOC": [60.0, 60.0, 60.0, 58.0]},
            index=[
                issue,
                new_issue,
                issue + pd.Timedelta(hours=6),
                new_issue + pd.Timedelta(hours=6),
            ],
        )
        frame.index.name = "time"

        metrics = evaluate_independent_forecast_archive(archive, frame)
        hindcast = build_soc_hindcast_dataset(archive, frame.to_xarray(), retention_days=1)

        self.assertAlmostEqual(float(metrics["soc_mae"]), 0.0)
        self.assertEqual(int(metrics["soc_independent_cycles"]), 1)
        self.assertEqual(hindcast.attrs["forecast_model_contract_id"], "contract-v12")
        self.assertTrue(
            np.isnan(
                float(
                    hindcast["BatterySOCHindcast_6h"].sel(
                        time=(issue + pd.Timedelta(hours=6)).to_datetime64()
                    )
                )
            )
        )

    def test_load_scenarios_decline_with_higher_loads(self) -> None:
        power_times = pd.date_range("2026-07-09T00:00:00", periods=25, freq="1h")
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
            {"ssrd": (("time",), np.zeros(len(forecast_times), dtype=float))},
            coords={"time": forecast_times},
        )

        forecast = build_forecast_dataset(power, solar, horizon_hours=15, capacity_kwh=26.0)

        previous = None
        for load_w in (100, 200, 300, 400, 500, 600):
            values = forecast[f"BatterySOCForecast_Load{load_w}W"].values
            if previous is not None:
                self.assertTrue(np.all(values <= previous + 1e-6))
            previous = values

    def test_build_forecast_dataset_supports_96h_horizon(self) -> None:
        power_times = pd.date_range("2026-07-09T00:00:00", periods=25, freq="1h")
        power = xr.Dataset(
            {
                "BatterySOC": (("time",), np.linspace(72.0, 70.0, len(power_times))),
                "SolarWatts_East": (("time",), np.full(len(power_times), 50.0)),
                "SolarWatts_South": (("time",), np.full(len(power_times), 50.0)),
                "SolarWatts_West": (("time",), np.full(len(power_times), 50.0)),
                "ACOutputWatts": (("time",), np.full(len(power_times), 20.0)),
                "DCInverterWatts": (("time",), np.full(len(power_times), 10.0)),
            },
            coords={"time": power_times},
        )
        forecast_times = pd.date_range(power_times[-1], periods=34, freq="3h")
        solar = xr.Dataset(
            {"ssrd": (("time",), np.arange(len(forecast_times), dtype=float) * 3 * 3600 * 200.0)},
            coords={"time": forecast_times},
        )

        forecast = build_forecast_dataset(power, solar, horizon_hours=96, capacity_kwh=26.0)

        self.assertEqual(forecast.attrs["forecast_horizon_hours"], "96")
        self.assertEqual(pd.Timestamp(forecast["time"].values[0]), power_times[-1])
        self.assertGreaterEqual((pd.Timestamp(forecast["time"].values[-1]) - power_times[-1]) / pd.Timedelta(hours=1), 95.0)

    def test_historical_load_forecast_persists_current_regime_without_clock_aliasing(self) -> None:
        times = pd.date_range("2026-07-01T00:00:00", periods=10 * 24 * 4, freq="15min")
        total_load = np.full(len(times), 9.0)
        total_load[(times >= times[3 * 24 * 4]) & (times < times[7 * 24 * 4])] = 650.0
        frame = pd.DataFrame(
            {
                "ACOutputWatts": total_load * 0.75,
                "DCInverterWatts": total_load * 0.25,
            },
            index=times,
        )
        forecast_times = pd.date_range(times[-1], periods=33, freq="3h")

        load = build_historical_load_forecast(frame, forecast_times, end=times[-1], calibration_days=10)

        np.testing.assert_allclose(load.to_numpy(), 9.0)
        self.assertEqual(load.attrs["load_model"], "finite_controlled_state_phases_v10")
        self.assertEqual(load.attrs["load_model_version"], 10)
        self.assertEqual(load.attrs["load_mode"], "DC-Only")
        self.assertEqual(load.attrs["load_regime"], "DC-Only")
        self.assertGreater(float(load.attrs["load_regime_threshold_w"]), 9.0)
        self.assertLess(float(load.attrs["load_regime_threshold_w"]), 650.0)

    def test_dc_only_mode_uses_solar_battery_power_balance(self) -> None:
        times = pd.date_range("2026-07-15T00:00:00", periods=24 * 4, freq="15min")
        solar_total = np.zeros(len(times))
        solar_total[24:72] = np.sin(np.linspace(0.0, np.pi, 48)) * 600.0
        frame = pd.DataFrame(
            {
                "SolarWatts_East": solar_total * 0.25,
                "SolarWatts_South": solar_total * 0.45,
                "SolarWatts_West": solar_total * 0.30,
                "BatteryWatts": solar_total - 220.0,
                "ACOutputWatts": np.zeros(len(times)),
                "DCInverterWatts": np.full(len(times), 9.0),
            },
            index=times,
        )
        forecast_times = pd.date_range(times[-1], periods=9, freq="3h")

        load = build_historical_load_forecast(frame, forecast_times, end=times[-1], calibration_days=1)

        np.testing.assert_allclose(load.to_numpy(), 220.0)
        self.assertEqual(load.attrs["load_mode"], "DC-Only")
        self.assertEqual(load.attrs["load_measurement"], "solar_generation_minus_battery_power")
        self.assertEqual(load.attrs["load_balance_measurement"], "solar_generation_minus_battery_power")

    def test_active_pdu_kit_names_the_learned_mode(self) -> None:
        power_times = pd.date_range("2026-07-15T00:00:00", periods=25, freq="1h")
        power = xr.Dataset(
            {
                "BatterySOC": (("time",), np.linspace(72.0, 70.0, len(power_times))),
                "SolarWatts_East": (("time",), np.full(len(power_times), 100.0)),
                "SolarWatts_South": (("time",), np.full(len(power_times), 100.0)),
                "SolarWatts_West": (("time",), np.full(len(power_times), 100.0)),
                "BatteryWatts": (("time",), np.full(len(power_times), -200.0)),
                "ACOutputWatts": (("time",), np.full(len(power_times), 300.0)),
                "DCInverterWatts": (("time",), np.full(len(power_times), 9.0)),
            },
            coords={"time": power_times},
        )
        pdu_times = pd.date_range(power_times[-1] - pd.Timedelta(minutes=30), periods=3, freq="15min")
        pdu = xr.Dataset(
            {
                "PDUOutlet6State": (("time",), np.ones(len(pdu_times))),
                "PDUOutlet6Watts": (("time",), np.full(len(pdu_times), 300.0)),
            },
            coords={"time": pdu_times},
        )
        forecast_times = pd.date_range(power_times[-1], periods=5, freq="3h")
        solar = xr.Dataset(
            {"ssrd": (("time",), np.arange(len(forecast_times), dtype=float) * 3 * 3600 * 200.0)},
            coords={"time": forecast_times},
        )

        forecast = build_forecast_dataset(power, solar, pdu=pdu, horizon_hours=12, capacity_kwh=26.0)

        self.assertEqual(forecast.attrs["load_mode"], "DC-Only + Radar")
        self.assertEqual(forecast.attrs["load_mode_source"], "pdu_signature")
        self.assertEqual(forecast.attrs["load_mode_signature"], "PDUOutlet6Watts>=5W")
        self.assertEqual(forecast.attrs["load_mode_learning_ready"], "true")
        self.assertAlmostEqual(float(forecast["ForecastLoadWatts"].median()), 500.0)
        registry = forecast.attrs["load_mode_registry"]
        self.assertIn("DC-Only + Radar", registry)

    def test_current_state_uses_dc_baseline_plus_pdu_not_conflicting_aggregate_load(self) -> None:
        end = pd.Timestamp("2026-07-17T00:00:00")
        power_times = pd.date_range(end - pd.Timedelta(hours=48), end, freq="15min")
        battery_watts = np.full(len(power_times), -220.0)
        battery_watts[-5:] = -900.0
        power = xr.Dataset(
            {
                "BatterySOC": (("time",), np.linspace(82.0, 80.0, len(power_times))),
                "SolarWatts_East": (("time",), np.zeros(len(power_times))),
                "SolarWatts_South": (("time",), np.zeros(len(power_times))),
                "SolarWatts_West": (("time",), np.zeros(len(power_times))),
                "BatteryWatts": (("time",), battery_watts),
                "ACOutputWatts": (("time",), np.zeros(len(power_times))),
                "DCInverterWatts": (("time",), np.full(len(power_times), 9.0)),
            },
            coords={"time": power_times},
        )
        pdu_times = pd.date_range(end - pd.Timedelta(minutes=30), end, freq="15min")
        pdu = xr.Dataset(
            {
                "PDUOutlet6State": (("time",), np.ones(len(pdu_times))),
                "PDUOutlet6Watts": (("time",), np.full(len(pdu_times), 300.0)),
            },
            coords={"time": pdu_times},
        )
        forecast_times = pd.date_range(end, periods=5, freq="3h")
        solar = xr.Dataset(
            {"ssrd": (("time",), np.arange(len(forecast_times), dtype=float) * 3 * 3600 * 200.0)},
            coords={"time": forecast_times},
        )

        forecast = build_forecast_dataset(power, solar, pdu=pdu, horizon_hours=12, capacity_kwh=26.0)

        self.assertEqual(forecast.attrs["load_mode"], "DC-Only + Radar")
        self.assertEqual(forecast.attrs["load_anchor_method"], "fresh_pdu_components_for_current_state")
        self.assertAlmostEqual(float(forecast.attrs["load_component_estimate_w"]), 520.0, places=3)
        self.assertAlmostEqual(float(forecast.attrs["forecast_load_p50_w"]), 520.0, places=3)
        np.testing.assert_allclose(forecast["ForecastLoadWatts"].values, 520.0)

    def test_powered_cl61_pdu_signature_precedes_smoothed_ac_state(self) -> None:
        end = pd.Timestamp("2026-07-16T12:00:00")
        times = pd.date_range(end - pd.Timedelta(minutes=30), end, freq="5min")
        frame = pd.DataFrame({"ACOutputWatts": [0.0] * (len(times) - 1) + [175.0]}, index=times)
        pdu_times = pd.date_range(end - pd.Timedelta(minutes=10), end, freq="5min")
        pdu = xr.Dataset(
            {
                "PDUOutlet5State": (("time",), np.ones(len(pdu_times))),
                "PDUOutlet5Watts": (("time",), np.full(len(pdu_times), 223.0)),
            },
            coords={"time": pdu_times},
        )

        mode, source, active_kits, pdu_time, active_watts = _resolve_load_mode(
            frame,
            pdu,
            end=end,
            observed_level_w=450.0,
            raw_registry={},
            previous_mode="DC-Only",
        )

        self.assertEqual(mode, "Ceilometer-on-AC")
        self.assertEqual(source, "pdu_ac_signature")
        self.assertEqual(active_kits, ["CL61"])
        self.assertEqual(pdu_time, end)
        self.assertAlmostEqual(active_watts, 223.0)

        signature = _load_mode_signature(mode, source, active_kits)
        self.assertEqual(signature, "PDUOutlet5Watts>=5W+ACOutputWatts>25W")

    def test_pdu_relay_state_without_power_is_not_an_active_kit(self) -> None:
        end = pd.Timestamp("2026-07-16T12:00:00")
        times = pd.date_range(end - pd.Timedelta(minutes=30), end, freq="15min")
        pdu = xr.Dataset(
            {
                "PDUOutlet5State": (("time",), np.ones(len(times))),
                "PDUOutlet5Watts": (("time",), np.zeros(len(times))),
            },
            coords={"time": times},
        )

        active_kits, pdu_time, active_watts = _pdu_active_kits(pdu, end=end)

        self.assertEqual(active_kits, [])
        self.assertEqual(pdu_time, end)
        self.assertTrue(np.isnan(active_watts))

    def test_mode_learning_waits_for_a_stable_run(self) -> None:
        ready, reason = _mode_learning_status(
            {
                "load_mode_state": "ac-active",
                "load_regime_run_hours": 0.25,
                "load_regime_sample_count": 4,
            },
            "DC-Only + CL61",
        )

        self.assertFalse(ready)
        self.assertEqual(reason, "waiting_for_stable_duration")

    def test_dc_only_registry_rejects_transition_level_outlier(self) -> None:
        end = pd.Timestamp("2026-07-16T12:00:00")
        times = pd.date_range(end - pd.Timedelta(hours=48), end, freq="15min")
        frame = pd.DataFrame(
            {
                "SolarWatts_East": np.zeros(len(times)),
                "SolarWatts_South": np.zeros(len(times)),
                "SolarWatts_West": np.zeros(len(times)),
                "BatteryWatts": np.full(len(times), -220.0),
                "ACOutputWatts": np.zeros(len(times)),
            },
            index=times,
        )
        frame.loc[frame.index >= end - pd.Timedelta(minutes=30), "BatteryWatts"] = -450.0
        frame.loc[frame.index >= end - pd.Timedelta(minutes=30), "ACOutputWatts"] = 175.0
        raw_registry = {
            "DC-Only": {
                "observations": [{"time": end.isoformat(), "level_w": 450.0}],
                "learned_level_w": 450.0,
            }
        }

        registry, clean_level = _repair_dc_only_registry(raw_registry, frame, end=end)

        self.assertAlmostEqual(float(clean_level), 220.0)
        self.assertAlmostEqual(float(registry["DC-Only"]["learned_level_w"]), 220.0)
        self.assertEqual(registry["DC-Only"]["observation_count"], 1)
        self.assertGreater(registry["DC-Only"]["clean_dark_sample_count"], 100)

    def test_retired_hourly_model_bias_does_not_distort_regime_forecast(self) -> None:
        power_times = pd.date_range("2026-07-09T00:00:00", periods=25, freq="1h")
        power = xr.Dataset(
            {
                "BatterySOC": (("time",), np.linspace(72.0, 70.0, len(power_times))),
                "SolarWatts_East": (("time",), np.full(len(power_times), 50.0)),
                "SolarWatts_South": (("time",), np.full(len(power_times), 50.0)),
                "SolarWatts_West": (("time",), np.full(len(power_times), 50.0)),
                "ACOutputWatts": (("time",), np.full(len(power_times), 200.0)),
                "DCInverterWatts": (("time",), np.full(len(power_times), 50.0)),
            },
            coords={"time": power_times},
        )
        forecast_times = pd.date_range(power_times[-1], periods=5, freq="3h")
        solar = xr.Dataset(
            {"ssrd": (("time",), np.arange(len(forecast_times), dtype=float) * 3 * 3600 * 200.0)},
            coords={"time": forecast_times},
        )
        previous = xr.Dataset(
            {"ForecastLoadWatts": (("time",), np.full(len(power_times), 500.0))},
            coords={"time": power_times},
        )

        forecast = build_forecast_dataset(power, solar, previous_forecast=previous, horizon_hours=12, capacity_kwh=26.0)

        self.assertEqual(float(forecast.attrs["load_bias_correction_w"]), 0.0)
        self.assertAlmostEqual(float(forecast["ForecastLoadWatts"].median()), 250.0)
        self.assertEqual(forecast.attrs["load_model"], "finite_controlled_state_phases_v10")
        self.assertIn("ForecastLoadBiasRecent", forecast)

    def test_stale_negative_load_bias_cannot_zero_ac_dc_load_forecast(self) -> None:
        power_times = pd.date_range("2026-07-09T00:00:00", periods=48, freq="1h")
        power = xr.Dataset(
            {
                "BatterySOC": (("time",), np.linspace(72.0, 70.0, len(power_times))),
                "SolarWatts_East": (("time",), np.full(len(power_times), 25.0)),
                "SolarWatts_South": (("time",), np.full(len(power_times), 25.0)),
                "SolarWatts_West": (("time",), np.full(len(power_times), 25.0)),
                "ACOutputWatts": (("time",), np.full(len(power_times), 220.0)),
                "DCInverterWatts": (("time",), np.full(len(power_times), 180.0)),
            },
            coords={"time": power_times},
        )
        forecast_times = pd.date_range(power_times[-1], periods=5, freq="3h")
        solar = xr.Dataset(
            {"ssrd": (("time",), np.arange(len(forecast_times), dtype=float) * 3 * 3600 * 100.0)},
            coords={"time": forecast_times},
        )

        forecast = build_forecast_dataset(
            power,
            solar,
            state={"load_bias_correction_w": -2000.0},
            horizon_hours=12,
            capacity_kwh=26.0,
        )

        self.assertGreater(float(forecast["ForecastLoadWatts"].median()), 0.0)
        self.assertGreaterEqual(float(forecast["ForecastLoadWatts"].median()), 300.0)
        self.assertGreater(float(forecast.attrs["forecast_load_w"]), 0.0)

    def test_archive_skill_scores_lead_buckets(self) -> None:
        issue_time = pd.Timestamp("2026-07-10T00:00:00")
        times = [
            issue_time,
            issue_time + pd.Timedelta(hours=3),
            issue_time + pd.Timedelta(hours=12),
            issue_time + pd.Timedelta(hours=18),
            issue_time + pd.Timedelta(hours=30),
            issue_time + pd.Timedelta(hours=42),
            issue_time + pd.Timedelta(hours=60),
            issue_time + pd.Timedelta(hours=84),
        ]
        forecast = xr.Dataset(
            {
                "BatterySOCForecast": (("time",), [50.0, 48.0, 45.0, 44.0, 40.0, 38.0, 35.0, 30.0]),
                "ForecastSolarWatts": (("time",), [0.0, 10.0, 20.0, 22.0, 30.0, 35.0, 40.0, 45.0]),
                "ForecastLoadWatts": (("time",), np.full(8, 100.0)),
            },
            coords={"time": times},
            attrs={"initial_soc_time": issue_time.isoformat()},
        )
        archive = append_forecast_archive(forecast, self.tmp_archive_path)
        power = xr.Dataset(
            {
                "BatterySOC": (("time",), [49.0, 46.0, 42.0, 41.0, 38.0, 35.0, 30.0, 25.0]),
                "SolarWatts_East": (("time",), [0.0, 12.0, 18.0, 21.0, 31.0, 34.0, 39.0, 44.0]),
                "ACOutputWatts": (("time",), np.full(8, 95.0)),
                "DCInverterWatts": (("time",), np.full(8, 5.0)),
            },
            coords={"time": forecast["time"].values},
        )

        metrics = evaluate_forecast_archive(archive, pd.DataFrame({name: power[name].values for name in power.data_vars}, index=pd.DatetimeIndex(power["time"].values)))

        self.assertIn("soc_mae_0_6h", metrics)
        self.assertIn("soc_mae_6_24h", metrics)
        self.assertIn("soc_mae_24_48h", metrics)
        self.assertIn("soc_mae_48_96h", metrics)

    def test_archive_preserves_load_phase_uncertainty(self) -> None:
        issue_time = pd.Timestamp("2026-07-10T00:00:00")
        times = pd.date_range(issue_time, periods=3, freq="3h")
        forecast = xr.Dataset(
            {
                "ForecastLoadWatts": (("time",), [470.0, 275.0, 450.0]),
                "ForecastLoadP10Watts": (("time",), [450.0, 260.0, 425.0]),
                "ForecastLoadP50Watts": (("time",), [470.0, 275.0, 450.0]),
                "ForecastLoadP90Watts": (("time",), [500.0, 290.0, 480.0]),
                "ForecastLoadPhaseCode": (("time",), [1, 2, 3]),
            },
            coords={"time": times},
            attrs={"initial_soc_time": issue_time.isoformat(), "load_model_version": "10"},
        )

        archive = append_forecast_archive(forecast, self.tmp_archive_path)

        for name in (
            "ForecastLoadP10Watts",
            "ForecastLoadP50Watts",
            "ForecastLoadP90Watts",
            "ForecastLoadPhaseCode",
        ):
            self.assertIn(name, archive)
        np.testing.assert_array_equal(archive["ForecastLoadPhaseCode"].values[0], [1.0, 2.0, 3.0])

    def test_forecast_skill_dataset_is_past_facing(self) -> None:
        issue_time = pd.Timestamp("2026-07-10T00:00:00")
        forecast_times = pd.date_range(issue_time, periods=5, freq="3h")
        forecast = xr.Dataset(
            {
                "BatterySOCForecast": (("time",), [50.0, 49.0, 48.0, 47.0, 46.0]),
                "ForecastSolarWatts": (("time",), [0.0, 10.0, 20.0, 10.0, 0.0]),
                "ForecastLoadWatts": (("time",), np.full(5, 120.0)),
            },
            coords={"time": forecast_times},
            attrs={"initial_soc_time": issue_time.isoformat()},
        )
        archive = append_forecast_archive(forecast, self.tmp_archive_path)
        power = xr.Dataset(
            {
                "BatterySOC": (("time",), [50.0, 48.5, 47.5, 46.0, 45.0]),
                "SolarWatts_East": (("time",), [0.0, 11.0, 18.0, 9.0, 0.0]),
                "ACOutputWatts": (("time",), np.full(5, 100.0)),
                "DCInverterWatts": (("time",), np.full(5, 10.0)),
            },
            coords={"time": forecast_times},
        )

        skill = build_forecast_skill_dataset(archive, power, window_hours=24, retention_days=1, freq="3h")

        self.assertIn("ForecastSOCMAE_0_6h_Verified", skill)
        self.assertIn("ForecastLoadMAE24h", skill)
        self.assertIn("ForecastSolarMAE24h", skill)
        self.assertIn("ForecastSOCSkill_0_6h", skill)
        self.assertLessEqual(pd.Timestamp(skill["time"].values[-1]), forecast_times[-1])
        self.assertTrue(np.isfinite(skill["ForecastSOCMAE_0_6h_Verified"].values).any())

    def test_load_verification_only_scores_current_model_version(self) -> None:
        old_issue = pd.Timestamp("2026-07-10T00:00:00")
        old_times = pd.DatetimeIndex([old_issue, old_issue + pd.Timedelta(hours=3)])
        old_forecast = xr.Dataset(
            {"ForecastLoadWatts": (("time",), [500.0, 500.0])},
            coords={"time": old_times},
            attrs={
                "initial_soc_time": old_issue.isoformat(),
                "ecmwf_cycle_time": old_issue.isoformat(),
                "load_model_version": "0",
            },
        )
        append_forecast_archive(old_forecast, self.tmp_archive_path)

        new_issue = pd.Timestamp("2026-07-10T06:00:00")
        new_times = pd.DatetimeIndex([new_issue, new_issue + pd.Timedelta(hours=3)])
        new_forecast = xr.Dataset(
            {"ForecastLoadWatts": (("time",), [110.0, 110.0])},
            coords={"time": new_times},
            attrs={
                "initial_soc_time": new_issue.isoformat(),
                "ecmwf_cycle_time": new_issue.isoformat(),
                "load_model_version": "10",
            },
        )
        archive = append_forecast_archive(new_forecast, self.tmp_archive_path)
        power_times = pd.date_range(old_issue, new_times[-1], freq="15min")
        power = xr.Dataset(
            {
                "ACOutputWatts": (("time",), np.full(len(power_times), 90.0)),
                "DCInverterWatts": (("time",), np.full(len(power_times), 10.0)),
            },
            coords={"time": power_times},
        )
        frame = pd.DataFrame(
            {name: power[name].values for name in power.data_vars},
            index=pd.DatetimeIndex(power["time"].values),
        )

        metrics = evaluate_forecast_archive(archive, frame)
        skill = build_forecast_skill_dataset(archive, power, window_hours=24, retention_days=1, freq="1h")

        self.assertAlmostEqual(float(metrics["load_mae_w"]), 10.0)
        finite_mae = skill["ForecastLoadMAE24h"].dropna("time")
        self.assertTrue(len(finite_mae))
        self.assertAlmostEqual(float(finite_mae.values[-1]), 10.0)
        self.assertEqual(skill.attrs["load_model_version"], "10")

    def test_hindcast_selects_fixed_lead_forecasts(self) -> None:
        issue = pd.Timestamp("2026-07-10T00:00:00")
        forecast_times = pd.DatetimeIndex([issue, issue + pd.Timedelta(hours=6), issue + pd.Timedelta(hours=24)])
        forecast = xr.Dataset(
            {
                "BatterySOCForecast": (("time",), [60.0, 58.0, 50.0]),
                "ForecastSolarWatts": (("time",), [0.0, 10.0, 0.0]),
                "ForecastLoadWatts": (("time",), [100.0, 100.0, 100.0]),
            },
            coords={"time": forecast_times},
            attrs={"initial_soc_time": issue.isoformat(), "ecmwf_cycle_time": issue.isoformat()},
        )
        archive = append_forecast_archive(forecast, self.tmp_archive_path)
        power_times = pd.date_range(issue, issue + pd.Timedelta(hours=24), freq="15min")
        power = xr.Dataset(
            {"BatterySOC": (("time",), np.linspace(60.0, 49.0, len(power_times)))},
            coords={"time": power_times},
        )

        hindcast = build_soc_hindcast_dataset(archive, power, retention_days=2)

        self.assertIn("BatterySOCObservedHindcast", hindcast)
        self.assertIn("BatterySOCHindcast_6h", hindcast)
        self.assertIn("BatterySOCHindcast_24h", hindcast)
        self.assertAlmostEqual(float(hindcast["BatterySOCHindcast_6h"].max(skipna=True)), 58.0)

    def test_skill_counts_independent_ecmwf_cycles(self) -> None:
        issue = pd.Timestamp("2026-07-10T00:00:00")
        archive = None
        for minute in (0, 15):
            run_issue = issue + pd.Timedelta(minutes=minute)
            forecast_times = pd.DatetimeIndex([run_issue, issue + pd.Timedelta(hours=3)])
            forecast = xr.Dataset(
                {
                    "BatterySOCForecast": (("time",), [60.0, 58.0]),
                    "ForecastSolarWatts": (("time",), [0.0, 10.0]),
                    "ForecastLoadWatts": (("time",), [100.0, 100.0]),
                },
                coords={"time": forecast_times},
                attrs={"initial_soc_time": run_issue.isoformat(), "ecmwf_cycle_time": issue.isoformat()},
            )
            archive = append_forecast_archive(forecast, self.tmp_archive_path)
        power_times = pd.date_range(issue, issue + pd.Timedelta(hours=3), freq="15min")
        power = xr.Dataset(
            {
                "BatterySOC": (("time",), np.linspace(60.0, 57.0, len(power_times))),
                "ACOutputWatts": (("time",), np.full(len(power_times), 90.0)),
                "DCInverterWatts": (("time",), np.full(len(power_times), 10.0)),
            },
            coords={"time": power_times},
        )

        skill = build_forecast_skill_dataset(archive, power, window_hours=24, retention_days=1, freq="1h")

        finite = skill["ForecastIndependentCycles"].dropna("time")
        self.assertTrue(len(finite))
        self.assertEqual(float(finite.values[-1]), 1.0)

    def test_independent_metrics_do_not_overweight_same_cycle_reanchors(self) -> None:
        cycle = pd.Timestamp("2026-07-10T00:00:00")
        archive = None
        for minute, value in ((0, 40.0), (15, 58.0)):
            issue = cycle + pd.Timedelta(minutes=minute)
            forecast = xr.Dataset(
                {"BatterySOCForecast": (("time",), [60.0, value])},
                coords={"time": [issue, cycle + pd.Timedelta(hours=3)]},
                attrs={"initial_soc_time": issue.isoformat(), "ecmwf_cycle_time": cycle.isoformat()},
            )
            archive = append_forecast_archive(forecast, self.tmp_archive_path)
        frame = pd.DataFrame(
            {"BatterySOC": [60.0, 60.0, 58.0]},
            index=pd.DatetimeIndex([cycle, cycle + pd.Timedelta(minutes=15), cycle + pd.Timedelta(hours=3)]),
        )

        raw = evaluate_forecast_archive(archive, frame)
        independent = evaluate_independent_forecast_archive(archive, frame)

        self.assertGreater(float(raw["soc_mae_0_6h"]), 0.0)
        self.assertAlmostEqual(float(independent["soc_mae_0_6h"]), 0.0)
        self.assertEqual(int(independent["soc_independent_cycles"]), 1)

    def test_cached_reanchors_are_archived_but_not_adaptive_evidence(self) -> None:
        cycle = pd.Timestamp("2026-07-10T00:00:00")
        valid_time = cycle + pd.Timedelta(hours=3)
        independent_issue = xr.Dataset(
            {"BatterySOCForecast": (("time",), [60.0, 58.0])},
            coords={"time": [cycle, valid_time]},
            attrs={
                "initial_soc_time": cycle.isoformat(),
                "ecmwf_cycle_time": cycle.isoformat(),
                "forecast_refresh_kind": "ecmwf_cycle",
                "forecast_verification_eligible": "true",
                "independent_cycle": "true",
            },
        )
        cached_time = cycle + pd.Timedelta(minutes=15)
        cached_issue = xr.Dataset(
            {"BatterySOCForecast": (("time",), [60.0, 20.0])},
            coords={"time": [cached_time, valid_time]},
            attrs={
                "initial_soc_time": cached_time.isoformat(),
                "ecmwf_cycle_time": cycle.isoformat(),
                "forecast_refresh_kind": "cached_reanchor",
                "forecast_verification_eligible": "false",
                "independent_cycle": "false",
            },
        )
        append_forecast_archive(independent_issue, self.tmp_archive_path)
        archive = append_forecast_archive(cached_issue, self.tmp_archive_path)
        frame = pd.DataFrame(
            {"BatterySOC": [60.0, 60.0, 58.0]},
            index=pd.DatetimeIndex([cycle, cached_time, valid_time]),
        )

        metrics = evaluate_independent_forecast_archive(archive, frame)

        self.assertEqual(
            archive["ForecastRefreshKind"].values.tolist(),
            ["ecmwf_cycle", "cached_reanchor"],
        )
        self.assertEqual(
            archive["ForecastVerificationEligible"].values.tolist(),
            [True, False],
        )
        self.assertAlmostEqual(float(metrics["soc_mae_0_6h"]), 0.0)
        self.assertEqual(int(metrics["soc_independent_cycles"]), 1)

    def test_build_ensemble_starts_every_member_at_actual_soc(self) -> None:
        power_times = pd.date_range("2026-07-10T00:00:00", periods=49, freq="1h")
        power = xr.Dataset(
            {
                "BatterySOC": (("time",), np.linspace(70.0, 66.0, len(power_times))),
                "ACOutputWatts": (("time",), np.full(len(power_times), 100.0)),
                "DCInverterWatts": (("time",), np.full(len(power_times), 20.0)),
            },
            coords={"time": power_times},
        )
        forecast_times = pd.date_range(power_times[-1], periods=6, freq="3h")
        accumulated = np.stack(
            [np.arange(len(forecast_times), dtype=float) * 3 * 3600 * value for value in (100.0, 200.0, 300.0)]
        )
        solar = xr.Dataset(
            {"ssrd": (("number", "time"), accumulated)},
            coords={"number": [1, 2, 3], "time": forecast_times},
        )
        deterministic = xr.Dataset(
            attrs={
                "solar_calibration_factor_w_per_wm2": "1.0",
                "battery_capacity_kwh": "26",
                "load_bias_correction_w": "0",
                "forecast_load_w": "455.15",
                "load_model": "mode_conditioned_energy_balance_v5",
                "load_model_version": "5",
                "load_mode": "DC-Only + CL61",
                "load_mode_source": "pdu_signature",
                "load_mode_active_kits": "CL61",
                "load_mode_signature": "PDUOutlet5Watts>=5W",
            }
        )

        ensemble = build_ensemble_dataset(power, deterministic, solar, horizon_hours=15)

        np.testing.assert_allclose(ensemble["BatterySOCForecastEnsemble"].values[:, 0], 66.0)
        ensemble_loads = ensemble["ForecastLoadWattsEnsemble"].values
        np.testing.assert_allclose(
            ensemble_loads,
            np.repeat(ensemble_loads[[0], :], ensemble_loads.shape[0], axis=0),
        )
        self.assertTrue(np.all(ensemble["BatterySOCForecastP10"] <= ensemble["BatterySOCForecastP50"]))
        self.assertTrue(np.all(ensemble["BatterySOCForecastP50"] <= ensemble["BatterySOCForecastP90"]))
        self.assertTrue(
            np.all(
                (ensemble[SOC_BELOW_THRESHOLD_PROBABILITY_FIELD] >= 0)
                & (ensemble[SOC_BELOW_THRESHOLD_PROBABILITY_FIELD] <= 1)
            )
        )
        self.assertEqual(
            float(ensemble.attrs["minimum_operational_soc_pct"]),
            MINIMUM_OPERATIONAL_SOC_PCT,
        )
        self.assertEqual(ensemble.attrs["load_model"], "mode_conditioned_energy_balance_v5")
        self.assertEqual(ensemble.attrs["load_model_version"], "5")
        self.assertEqual(ensemble.attrs["load_mode"], "DC-Only + CL61")
        self.assertEqual(ensemble.attrs["load_mode_signature"], "PDUOutlet5Watts>=5W")
        self.assertEqual(float(ensemble.attrs["forecast_load_w"]), 455.15)
        self.assertEqual(ensemble.attrs["scenario_scope"], "current_system_only")
        self.assertEqual(
            ensemble.attrs["load_uncertainty"],
            "stationary exact-state load distribution independently paired with ECMWF solar members",
        )

    def test_ensemble_generation_uses_exact_deterministic_power_cutoff(self) -> None:
        root = Path(self._tmp.name)
        power_path = root / "live_power.zarr"
        deterministic_path = root / "deterministic.zarr"
        solar_path = root / "ensemble.nc"
        output_path = root / "ensemble.zarr"
        archive_path = root / "ensemble_archive.zarr"
        skill_path = root / "ensemble_skill.zarr"
        cutoff = pd.Timestamp("2026-07-10T04:00:00")
        power_times = pd.date_range("2026-07-10T00:00:00", periods=6, freq="1h")
        xr.Dataset(
            {
                "BatterySOC": (("time",), [70.0, 69.0, 68.0, 67.0, 66.0, 12.0]),
                "ACOutputWatts": (("time",), np.full(6, 100.0)),
                "DCInverterWatts": (("time",), np.full(6, 20.0)),
            },
            coords={"time": power_times},
        ).to_zarr(power_path, mode="w", consolidated=True)
        xr.Dataset(
            attrs={
                "initial_soc_time": cutoff.isoformat(),
                "initial_soc_pct": "66",
                "solar_calibration_factor_w_per_wm2": "1",
                "battery_capacity_kwh": "26",
                "load_bias_correction_w": "0",
                "forecast_load_p10_w": "110",
                "forecast_load_p50_w": "120",
                "forecast_load_p90_w": "130",
                "load_state_contract": "finite_operating_state_v1",
                "source_cycle_set_id": "deterministic-cycle-set",
                "source_manifest_digest": f"sha256:{'a' * 64}",
                "forecast_identity_id": "deterministic-forecast-identity",
            }
        ).to_zarr(deterministic_path, mode="w", consolidated=True)
        solar_times = pd.date_range(cutoff, periods=3, freq="3h")
        xr.Dataset(
            {
                "ssrd": (
                    ("number", "time"),
                    np.stack(
                        [
                            np.arange(3, dtype=float) * 3 * 3600 * irradiance
                            for irradiance in (100.0, 200.0, 300.0)
                        ]
                    ),
                )
            },
            coords={"number": [1, 2, 3], "time": solar_times},
        ).to_netcdf(solar_path)

        generate_ensemble_forecast(
            power_zarr=power_path,
            deterministic_zarr=deterministic_path,
            output_zarr=output_path,
            archive_zarr=archive_path,
            skill_zarr=skill_path,
            input_forecast=solar_path,
            horizon_hours=6,
            power_cutoff_time=cutoff.isoformat(),
        )

        with xr.open_zarr(output_path, chunks={}) as forecast:
            ensemble_source_cycle_set_id = str(forecast.attrs["source_cycle_set_id"])
            self.assertEqual(pd.Timestamp(forecast.attrs["initial_soc_time"]), cutoff)
            self.assertEqual(float(forecast.attrs["initial_soc_pct"]), 66.0)
            self.assertEqual(
                pd.Timestamp(forecast.attrs["power_input_cutoff_time_utc"]), cutoff
            )
            self.assertTrue(
                forecast.attrs["source_cycle_set_id"].startswith(
                    "power-ensemble-source-set-v1-"
                )
            )
            self.assertNotEqual(
                forecast.attrs["source_cycle_set_id"], "deterministic-cycle-set"
            )
            self.assertEqual(
                forecast.attrs["deterministic_source_cycle_set_id"],
                "deterministic-cycle-set",
            )
            self.assertEqual(
                forecast.attrs["deterministic_source_manifest_digest"],
                f"sha256:{'a' * 64}",
            )
            self.assertRegex(
                forecast.attrs["ensemble_site_forcing_sha256"], r"^sha256:[0-9a-f]{64}$"
            )
            self.assertRegex(
                forecast.attrs["source_manifest_digest"], r"^sha256:[0-9a-f]{64}$"
            )
            self.assertNotEqual(
                forecast.attrs["forecast_identity_id"],
                "deterministic-forecast-identity",
            )
            np.testing.assert_allclose(
                forecast["BatterySOCForecastEnsemble"].isel(time=0).values,
                66.0,
            )
        with xr.open_zarr(archive_path, chunks={}) as archive:
            self.assertEqual(
                str(archive["SourceCycleSetID"].values[-1]),
                ensemble_source_cycle_set_id,
            )
        with xr.open_zarr(skill_path, chunks={}) as skill:
            self.assertLessEqual(pd.Timestamp(skill["time"].values[-1]), cutoff)

    def test_ensemble_power_cutoff_fails_without_an_exact_soc_anchor(self) -> None:
        power = xr.Dataset(
            {"BatterySOC": (("time",), [70.0, 69.0])},
            coords={"time": pd.date_range("2026-07-10T01:00:00", periods=2, freq="1h")},
        )

        with self.assertRaisesRegex(ValueError, "No APS power data exist"):
            _power_at_or_before_cutoff(power, pd.Timestamp("2026-07-10T00:00:00"))
        with self.assertRaisesRegex(ValueError, "exact ensemble cutoff"):
            _power_at_or_before_cutoff(power, pd.Timestamp("2026-07-10T01:30:00"))

    def test_ensemble_power_cutoff_keeps_a_lazy_bounded_window(self) -> None:
        times = pd.date_range("2026-06-20T12:00:00", periods=22 * 24 + 1, freq="1h")
        cutoff = pd.Timestamp("2026-07-11T10:00:00")
        power = xr.Dataset(
            {
                "BatterySOC": (("time",), np.linspace(90.0, 60.0, len(times))),
                "ACOutputWatts": (("time",), np.full(len(times), 120.0)),
                "UnusedDiagnostic": (("time",), np.arange(len(times), dtype=float)),
            },
            coords={"time": times},
        ).chunk({"time": 24})

        selected = _power_at_or_before_cutoff(power, cutoff, history_days=14)

        expected_start = (cutoff - pd.Timedelta(days=14)).normalize()
        self.assertEqual(pd.Timestamp(selected["time"].values[0]), expected_start)
        self.assertEqual(pd.Timestamp(selected["time"].values[-1]), cutoff)
        self.assertNotIn("UnusedDiagnostic", selected)
        self.assertIsNotNone(getattr(selected["ACOutputWatts"].data, "chunks", None))
        self.assertEqual(selected.attrs["ensemble_power_history_days"], "14")
        self.assertEqual(
            pd.Timestamp(selected.attrs["ensemble_power_window_start_utc"]),
            expected_start,
        )
        self.assertEqual(
            pd.Timestamp(selected.attrs["ensemble_power_window_end_utc"]),
            cutoff,
        )

    def test_ensemble_power_history_covers_long_lead_skill_references(self) -> None:
        times = pd.date_range("2026-06-20", periods=16 * 24 + 1, freq="1h")
        power = xr.Dataset(
            {"BatterySOC": (("time",), np.linspace(90.0, 70.0, len(times)))},
            coords={"time": times},
        )

        with self.assertRaisesRegex(ValueError, "must retain at least 12 days"):
            _power_at_or_before_cutoff(
                power,
                pd.Timestamp(times[-1]),
                history_days=ENSEMBLE_POWER_MIN_HISTORY_DAYS - 0.01,
            )

        selected = _power_at_or_before_cutoff(
            power,
            pd.Timestamp(times[-1]),
            history_days=ENSEMBLE_POWER_MIN_HISTORY_DAYS,
        )
        self.assertEqual(
            pd.Timestamp(selected["time"].values[0]),
            (pd.Timestamp(times[-1]) - pd.Timedelta(days=12)).normalize(),
        )

    def test_ensemble_power_cutoff_does_not_materialise_post_cutoff_state(self) -> None:
        times = pd.date_range("2026-07-10T00:00:00", periods=6, freq="1h")
        power = xr.Dataset(
            {
                "BatterySOC": (("time",), [70.0, 69.0, 68.0, 67.0, 12.0, 11.0]),
                "ACOutputWatts": (("time",), [100.0, 100.0, 100.0, 100.0, 900.0, 900.0]),
            },
            coords={"time": times},
        )
        cutoff = pd.Timestamp("2026-07-10T03:00:00")

        selected = _power_at_or_before_cutoff(power, cutoff)

        self.assertEqual(pd.Timestamp(selected["time"].values[-1]), cutoff)
        self.assertEqual(float(selected["BatterySOC"].values[-1]), 67.0)
        self.assertEqual(float(selected["ACOutputWatts"].values[-1]), 100.0)

    def test_ensemble_legacy_load_fallback_matches_full_retained_history(self) -> None:
        power_times = pd.date_range("2026-06-20", periods=20 * 24 + 1, freq="1h")
        cutoff = pd.Timestamp(power_times[-1])
        load = 180.0 + 20.0 * np.sin(np.arange(len(power_times)) / 12.0)
        power = xr.Dataset(
            {
                "BatterySOC": (("time",), np.linspace(90.0, 70.0, len(power_times))),
                "ACOutputWatts": (("time",), load),
                "DCInverterWatts": (("time",), np.full(len(power_times), 20.0)),
            },
            coords={"time": power_times},
        )
        forecast_times = pd.date_range(cutoff, periods=3, freq="3h")
        solar = xr.Dataset(
            {
                "ssrd": (
                    ("number", "time"),
                    np.stack(
                        [
                            np.arange(len(forecast_times), dtype=float) * 3 * 3600 * value
                            for value in (100.0, 200.0, 300.0)
                        ]
                    ),
                )
            },
            coords={"number": [1, 2, 3], "time": forecast_times},
        )
        deterministic = xr.Dataset(
            attrs={
                "solar_calibration_factor_w_per_wm2": "1",
                "battery_capacity_kwh": "26",
                "load_bias_correction_w": "0",
                "forecast_load_p10_w": "180",
                "forecast_load_p50_w": "200",
                "forecast_load_p90_w": "220",
            }
        )
        retained = _power_at_or_before_cutoff(power, cutoff, history_days=14)

        with patch(
            "generate_power_soc_ensemble._power_frame",
            wraps=__import__("generate_power_soc_ensemble")._power_frame,
        ) as power_frame:
            bounded = build_ensemble_dataset(retained, deterministic, solar, horizon_hours=6)
        full = build_ensemble_dataset(power, deterministic, solar, horizon_hours=6)

        self.assertEqual(list(power_frame.call_args_list[0].args[0].data_vars), ["BatterySOC"])
        self.assertIn("ACOutputWatts", power_frame.call_args_list[1].args[0].data_vars)
        for name in (
            "BatterySOCForecastEnsemble",
            "ForecastSolarWattsEnsemble",
            "ForecastLoadWattsEnsemble",
        ):
            np.testing.assert_allclose(bounded[name], full[name])

    def test_ensemble_skill_matches_full_history_for_long_lead_reference(self) -> None:
        end = pd.Timestamp("2026-07-20T00:00:00")
        issue = end - pd.Timedelta(days=11)
        valid = issue + pd.Timedelta(hours=95)
        forecast = xr.Dataset(
            {
                "BatterySOCForecastEnsemble": (
                    ("member", "time"),
                    np.asarray([[80.0, 76.0], [80.0, 75.0], [80.0, 74.0]]),
                )
            },
            coords={"member": [1, 2, 3], "time": [issue, valid]},
            attrs={
                "initial_soc_time": issue.isoformat(),
                "ecmwf_cycle_time": issue.isoformat(),
            },
        )
        archive = append_ensemble_archive(forecast, self.tmp_ensemble_archive_path)
        power_times = pd.date_range(end - pd.Timedelta(days=20), end, freq="1h")
        power = xr.Dataset(
            {
                "BatterySOC": (
                    ("time",),
                    np.linspace(90.0, 70.0, len(power_times)),
                ),
                "ACOutputWatts": (("time",), np.full(len(power_times), 200.0)),
            },
            coords={"time": power_times},
        )
        retained = _power_at_or_before_cutoff(power, end, history_days=14)

        with patch(
            "generate_power_soc_ensemble._power_frame",
            wraps=__import__("generate_power_soc_ensemble")._power_frame,
        ) as power_frame:
            bounded = build_ensemble_skill_dataset(archive, retained)
        full = build_ensemble_skill_dataset(archive, power)

        self.assertEqual(list(power_frame.call_args.args[0].data_vars), ["BatterySOC"])
        np.testing.assert_array_equal(bounded["time"], full["time"])
        for name in bounded.data_vars:
            np.testing.assert_allclose(bounded[name], full[name], equal_nan=True)
        self.assertTrue(np.isfinite(bounded["ForecastSOCCRPS_48_96h"]).any())

    def test_ensemble_load_members_span_one_stationary_state_distribution(self) -> None:
        power_times = pd.date_range("2026-07-10T00:00:00", periods=49, freq="1h")
        power = xr.Dataset(
            {
                "BatterySOC": (("time",), np.linspace(70.0, 66.0, len(power_times))),
                "ACOutputWatts": (("time",), np.full(len(power_times), 100.0)),
                "DCInverterWatts": (("time",), np.full(len(power_times), 20.0)),
            },
            coords={"time": power_times},
        )
        forecast_times = pd.date_range(power_times[-1], periods=6, freq="3h")
        accumulated = np.stack(
            [np.arange(len(forecast_times), dtype=float) * 3 * 3600 * value for value in (100.0, 200.0, 300.0, 400.0, 500.0)]
        )
        solar = xr.Dataset(
            {"ssrd": (("number", "time"), accumulated)},
            coords={"number": [1, 2, 3, 4, 5], "time": forecast_times},
        )
        deterministic = xr.Dataset(
            attrs={
                "solar_calibration_factor_w_per_wm2": "1.0",
                "battery_capacity_kwh": "26",
                "load_bias_correction_w": "0",
                "forecast_load_p10_w": "400",
                "forecast_load_p50_w": "500",
                "forecast_load_p90_w": "600",
                "load_anchor_method": "learned_exact_state_distribution",
                "load_mode_learning_observations": "12",
                "load_model": "finite_controlled_state_v8",
                "load_model_version": "8",
                "load_mode": "DC-Only + Radar",
                "load_state_contract": "finite_operating_state_v1",
            }
        )

        ensemble = build_ensemble_dataset(power, deterministic, solar, horizon_hours=15)

        loads = ensemble["ForecastLoadWattsEnsemble"].values
        np.testing.assert_allclose(np.diff(loads, axis=1), 0.0)
        self.assertGreater(float(np.ptp(loads[:, 0])), 0.0)
        self.assertGreaterEqual(float(loads[:, 0].min()), 400.0)
        self.assertLessEqual(float(loads[:, 0].max()), 600.0)
        self.assertEqual(ensemble.attrs["load_state_contract"], "finite_operating_state_v1")

    def test_ensemble_tracks_startup_and_fan_uncertainty_inside_one_state(self) -> None:
        power_times = pd.date_range("2026-07-20T00:00:00", periods=49, freq="1h")
        power = xr.Dataset(
            {
                "BatterySOC": (("time",), np.linspace(95.0, 92.0, len(power_times))),
                "ACOutputWatts": (("time",), np.full(len(power_times), 250.0)),
                "DCInverterWatts": (("time",), np.full(len(power_times), 25.0)),
            },
            coords={"time": power_times},
        )
        forecast_times = pd.date_range(power_times[-1], periods=6, freq="3h")
        accumulated = np.stack(
            [np.arange(len(forecast_times), dtype=float) * 3 * 3600 * value for value in (100.0, 200.0, 300.0, 400.0, 500.0)]
        )
        solar = xr.Dataset(
            {"ssrd": (("number", "time"), accumulated)},
            coords={"number": [1, 2, 3, 4, 5], "time": forecast_times},
        )
        dynamics = {
            "schema_version": 3,
            "state": "dc_cl61",
            "current_phase": "startup",
            "state_started_at": power_times[-1].isoformat(),
            "phase_started_at": power_times[-1].isoformat(),
            "startup_duration_p10_minutes": 60.0,
            "startup_duration_p50_minutes": 120.0,
            "startup_duration_p90_minutes": 180.0,
            "phase_profiles": {
                "startup": {"p10_w": 450.0, "p50_w": 470.0, "p90_w": 500.0, "sample_count": 16},
                "fan_low": {"p10_w": 260.0, "p50_w": 275.0, "p90_w": 290.0, "sample_count": 48},
                "fan_high": {"p10_w": 425.0, "p50_w": 450.0, "p90_w": 480.0, "sample_count": 24},
            },
            "phase_weights": {"fan_low": 0.67, "fan_high": 0.33},
            "phase_dwell_minutes": {"fan_low": 360.0, "fan_high": 180.0},
            "sample_count": 88,
            "episode_count": 3,
            "change_count": 7,
        }
        deterministic = xr.Dataset(
            attrs={
                "solar_calibration_factor_w_per_wm2": "1.0",
                "battery_capacity_kwh": "26",
                "forecast_load_p10_w": "260",
                "forecast_load_p50_w": "275",
                "forecast_load_p90_w": "500",
                "load_model": "finite_controlled_state_phases_v10",
                "load_model_version": "10",
                "load_mode": "DC-Only + CL61",
                "load_state_contract": "finite_operating_state_phases_v2",
                "load_state_hold_policy": "hold_confirmed_state_allow_detected_phase_or_explicit_schedule_transition",
                "load_state_dynamics": json.dumps(dynamics),
            }
        )

        ensemble = build_ensemble_dataset(power, deterministic, solar, horizon_hours=15)

        loads = np.asarray(ensemble["ForecastLoadWattsEnsemble"].values)
        phases = np.asarray(ensemble["ForecastLoadPhaseCodeEnsemble"].values)
        self.assertEqual(loads.shape, phases.shape)
        self.assertTrue(np.all(phases[:, 0] == PHASE_CODES[PHASE_STARTUP]))
        self.assertGreater(np.count_nonzero(np.diff(phases, axis=1)), 0)
        self.assertGreater(np.count_nonzero(np.diff(loads, axis=1)), 0)
        self.assertEqual(
            ensemble.attrs["load_uncertainty"],
            "exact-state startup duration and fan-phase uncertainty independently paired with ECMWF solar members",
        )

    def test_ensemble_reanchors_when_soc_or_load_mode_changes_within_same_cycle(self) -> None:
        deterministic_attrs = {
            "initial_soc_time": "2026-07-16T12:29:34",
            "initial_soc_pct": "57",
            "solar_calibration_factor_w_per_wm2": "1.82",
            "battery_capacity_kwh": "26",
            "load_bias_correction_w": "0",
            "forecast_load_w": "455.15",
            "load_model": "mode_conditioned_energy_balance_v5",
            "load_model_version": "5",
            "load_mode": "DC-Only + CL61",
            "load_mode_source": "pdu_signature",
            "load_mode_active_kits": "CL61",
            "load_mode_signature": "PDUOutlet5Watts>=5W",
            "load_state_uncertainty_source": "elapsed_time_exact_state_phase_distribution",
            "load_state_dynamics": '{"current_phase":"fan_low"}',
            "load_state_dynamics_signature": "phase-profile-a",
            "forecast_identity_id": "deterministic-identity",
            "source_cycle_set_id": "deterministic-source-set",
            "source_manifest_digest": f"sha256:{'b' * 64}",
        }
        matching_attrs = dict(deterministic_attrs)

        self.assertEqual(_ensemble_refresh_reasons(matching_attrs, deterministic_attrs), [])

        composite_attrs = {
            **matching_attrs,
            "forecast_identity_id": "ensemble-composite-identity",
            "source_cycle_set_id": "ensemble-composite-source-set",
            "source_manifest_digest": f"sha256:{'c' * 64}",
            "deterministic_forecast_identity_id": "deterministic-identity",
            "deterministic_source_cycle_set_id": "deterministic-source-set",
            "deterministic_source_manifest_digest": f"sha256:{'b' * 64}",
        }
        self.assertEqual(_ensemble_refresh_reasons(composite_attrs, deterministic_attrs), [])

        stale_dynamics = dict(matching_attrs)
        stale_dynamics["load_state_dynamics_signature"] = "phase-profile-b"
        self.assertEqual(
            _ensemble_refresh_reasons(stale_dynamics, deterministic_attrs),
            ["load_state_dynamics_signature"],
        )

        stale_attrs = dict(matching_attrs)
        stale_attrs.update(
            {
                "initial_soc_time": "2026-07-16T08:36:12",
                "initial_soc_pct": "61",
                "forecast_load_w": "223",
                "load_model_version": "3",
                "load_mode": "DC-Only",
            }
        )
        reasons = _ensemble_refresh_reasons(stale_attrs, deterministic_attrs)

        self.assertIn("initial_soc_time", reasons)
        self.assertIn("initial_soc_pct", reasons)
        self.assertIn("forecast_load_w", reasons)
        self.assertIn("load_model_version", reasons)
        self.assertIn("load_mode", reasons)

    def test_operational_threshold_refresh_replaces_legacy_probability(self) -> None:
        ensemble = xr.Dataset(
            {
                "BatterySOCForecastEnsemble": (
                    ("member", "time"),
                    [[50.0, 35.0], [45.0, 30.0]],
                ),
                "BatterySOCBelow20Probability": (("time",), [0.0, 0.0]),
            },
            coords={"member": [1, 2], "time": pd.date_range("2026-07-10", periods=2, freq="3h")},
        )

        refreshed = apply_operational_soc_threshold(ensemble)

        self.assertNotIn("BatterySOCBelow20Probability", refreshed)
        np.testing.assert_allclose(refreshed[SOC_BELOW_THRESHOLD_PROBABILITY_FIELD], [0.0, 1.0])
        self.assertEqual(
            float(refreshed.attrs["minimum_operational_soc_pct"]),
            MINIMUM_OPERATIONAL_SOC_PCT,
        )

    def test_ensemble_archive_produces_probabilistic_skill(self) -> None:
        issue = pd.Timestamp("2026-07-10T00:00:00")
        times = pd.date_range(issue, periods=5, freq="3h")
        members = np.array(
            [
                [60.0, 58.0, 56.0, 54.0, 52.0],
                [60.0, 59.0, 57.0, 55.0, 53.0],
                [60.0, 57.0, 55.0, 53.0, 51.0],
            ],
            dtype=np.float32,
        )
        forecast = xr.Dataset(
            {"BatterySOCForecastEnsemble": (("member", "time"), members)},
            coords={"member": [1, 2, 3], "time": times},
            attrs={"initial_soc_time": issue.isoformat()},
        )
        archive = append_ensemble_archive(forecast, self.tmp_ensemble_archive_path)
        power_times = pd.date_range(issue, times[-1], freq="15min")
        power = xr.Dataset(
            {"BatterySOC": (("time",), np.linspace(60.0, 52.0, len(power_times)))},
            coords={"time": power_times},
        )

        skill = build_ensemble_skill_dataset(archive, power, retention_days=1)

        self.assertIn("ForecastSOCCRPS_0_6h", skill)
        self.assertIn("ForecastSOCIntervalCoverage80", skill)
        self.assertIn(SOC_BELOW_THRESHOLD_BRIER_FIELD, skill)
        self.assertIn("ForecastSOCCRPSSamples_0_6h", skill)
        self.assertIn("ForecastSOCCRPSCycles_0_6h", skill)
        self.assertIn("ForecastSOCCRPSSkill_0_6h", skill)
        self.assertNotIn("ForecastSOCBelow20Brier", skill)
        self.assertTrue(np.isfinite(skill["ForecastSOCCRPS_0_6h"].values).any())
        finite_coverage = skill["ForecastSOCIntervalCoverage80"].values
        finite_coverage = finite_coverage[np.isfinite(finite_coverage)]
        self.assertTrue(np.all((finite_coverage >= 0.0) & (finite_coverage <= 1.0)))

    def test_ensemble_skill_filters_semantic_identity_and_cached_reanchors(self) -> None:
        common_new_identity = {
            "forecast_model_contract_id": "new-contract",
            "forecast_system_version": "power-v12",
            "feature_set_version": "features-v4",
            "feature_set_digest": "feature-digest",
            "forecast_code_revision": "revision-new",
            "candidate_lane": "D_physical_solar_load_residual",
            "local_feature_contract_id": "local-v1",
            "baseline_control_contract_id": "baseline-v10",
            "baseline_control_system_version": "power-v10",
        }

        def ensemble_issue(
            issue: pd.Timestamp,
            *,
            target_soc: float,
            contract: str,
            source_cycle: str,
            independent: bool,
        ) -> xr.Dataset:
            attrs = {
                **common_new_identity,
                "initial_soc_time": issue.isoformat(),
                "ecmwf_cycle_time": issue.floor("3h").isoformat(),
                "source_cycle_set_id": source_cycle,
                "forecast_model_contract_id": contract,
                "forecast_refresh_kind": "ecmwf_cycle" if independent else "cached_reanchor",
                "forecast_verification_eligible": str(independent).lower(),
                "independent_cycle": str(independent).lower(),
            }
            members = np.asarray(
                [[50.0, target_soc], [50.0, target_soc]], dtype=np.float32
            )
            return xr.Dataset(
                {"BatterySOCForecastEnsemble": (("member", "time"), members)},
                coords={
                    "member": [1, 2],
                    "time": [issue.to_datetime64(), (issue + pd.Timedelta(hours=3)).to_datetime64()],
                },
                attrs=attrs,
            )

        archive = append_ensemble_archive(
            ensemble_issue(
                pd.Timestamp("2026-07-10T00:00:00"),
                target_soc=10.0,
                contract="old-contract",
                source_cycle="old-cycle",
                independent=True,
            ),
            self.tmp_ensemble_archive_path,
        )
        archive = append_ensemble_archive(
            ensemble_issue(
                pd.Timestamp("2026-07-10T01:00:00"),
                target_soc=50.0,
                contract="new-contract",
                source_cycle="new-cycle",
                independent=True,
            ),
            self.tmp_ensemble_archive_path,
        )
        archive = append_ensemble_archive(
            ensemble_issue(
                pd.Timestamp("2026-07-10T02:00:00"),
                target_soc=5.0,
                contract="new-contract",
                source_cycle="new-cycle",
                independent=False,
            ),
            self.tmp_ensemble_archive_path,
        )
        power_times = pd.date_range("2026-07-10T00:00:00", periods=21, freq="15min")
        power = xr.Dataset(
            {"BatterySOC": (("time",), np.full(len(power_times), 50.0))},
            coords={"time": power_times},
        )

        skill = build_ensemble_skill_dataset(archive, power, retention_days=1)

        self.assertEqual(
            archive["ForecastVerificationEligible"].values.tolist(),
            [True, True, False],
        )
        self.assertEqual(archive["IndependentCycle"].values.tolist(), [True, True, False])
        self.assertEqual(archive["ForecastModelContractID"].values.tolist(), [
            "old-contract",
            "new-contract",
            "new-contract",
        ])
        finite_crps = skill["ForecastSOCCRPS_0_6h"].values
        finite_crps = finite_crps[np.isfinite(finite_crps)]
        self.assertTrue(len(finite_crps))
        np.testing.assert_allclose(finite_crps, 0.0)
        finite_cycles = skill["ForecastSOCCRPSCycles_0_6h"].values
        finite_cycles = finite_cycles[np.isfinite(finite_cycles)]
        np.testing.assert_allclose(finite_cycles, 1.0)
        self.assertEqual(skill.attrs["forecast_model_contract_id"], "new-contract")

    def test_ensemble_guidance_marks_immature_long_range_scores_not_verified(self) -> None:
        times = pd.date_range("2026-07-10T00:00:00", periods=2, freq="1h")
        summary = xr.Dataset(
            {
                "ForecastSOCCRPS_0_6h": (("time",), [1.0, 1.2]),
                "ForecastSOCCRPSSamples_0_6h": (("time",), [24.0, 24.0]),
                "ForecastSOCCRPSCycles_0_6h": (("time",), [12.0, 12.0]),
                "ForecastSOCCRPSSkill_0_6h": (("time",), [0.1, 0.1]),
                "ForecastSOCIntervalCoverage80": (("time",), [0.8, 0.8]),
                "ForecastSOCIntervalCoverage80Samples": (("time",), [24.0, 24.0]),
                "ForecastSOCIntervalCoverage80Cycles": (("time",), [12.0, 12.0]),
                SOC_BELOW_THRESHOLD_BRIER_FIELD: (("time",), [0.05, 0.05]),
                f"{SOC_BELOW_THRESHOLD_BRIER_FIELD}Samples": (("time",), [24.0, 24.0]),
                f"{SOC_BELOW_THRESHOLD_BRIER_FIELD}Cycles": (("time",), [12.0, 12.0]),
            },
            coords={"time": times},
        )

        guidance = build_power_verification_guidance("soc_ensemble_skill", summary)

        self.assertIsNotNone(guidance)
        metrics = {metric["id"]: metric for metric in guidance["metrics"]}
        self.assertEqual(metrics["soc-crps-0_6h"]["status"], "Insufficient evidence")
        self.assertEqual(metrics["soc-crps-48_96h"]["valueText"], "Not yet verified")
        self.assertEqual(metrics["soc-coverage"]["status"], "Insufficient evidence")

    def test_display_summary_merges_forecast_fields(self) -> None:
        power_times = pd.date_range("2026-07-10T00:00:00", periods=3, freq="1h")
        power = xr.Dataset(
            {
                "BatterySOC": (("time",), [70.0, 69.5, 69.0]),
                "ACOutputWatts": (("time",), [10.0, 10.0, 10.0]),
            },
            coords={"time": power_times},
        )
        forecast_times = pd.date_range("2026-07-10T03:00:00", periods=3, freq="3h")
        forecast = xr.Dataset(
            {
                "BatterySOCForecast": (("time",), [68.0, 67.0, 66.0]),
                "BatterySOCForecast_Load100W": (("time",), [68.0, 67.5, 67.0]),
                "BatterySOCForecast_Load600W": (("time",), [68.0, 63.0, 58.0]),
                "ECMWFSolarIrradiance": (("time",), [100.0, 200.0, 100.0]),
                "ForecastLoadWatts": (("time",), [220.0, 220.0, 220.0]),
            },
            coords={"time": forecast_times},
            attrs={
                "load_mode": "DC-Only",
                "load_model": "mode_conditioned_energy_balance_v5",
                "load_model_version": "5",
                "load_mode_source": "ac_output",
                "load_mode_active_kits": "",
                "load_mode_signature": "ACOutputWatts<=25W",
                "load_mode_learning_ready": "true",
                "load_mode_learning_reason": "stable",
                "load_mode_learning_observations": "2",
                "load_mode_pdu_active_watts": "nan",
                "load_measurement": "battery_discharge_when_solar_zero",
                "load_balance_measurement": "solar_generation_minus_battery_power",
            },
        )
        skill = xr.Dataset(
            {
                "ForecastSOCMAE_0_6h_Verified": (("time",), [1.0, 1.5, 2.0]),
                "ForecastLoadMAE24h": (("time",), [10.0, 11.0, 12.0]),
                "ForecastSolarMAE24h": (("time",), [20.0, 21.0, 22.0]),
            },
            coords={"time": power_times},
        )
        hindcast = xr.Dataset(
            {
                "BatterySOCObservedHindcast": (("time",), [70.0, 69.5, 69.0]),
                "BatterySOCHindcast_6h": (("time",), [71.0, 70.0, 68.0]),
            },
            coords={"time": power_times},
        )
        ensemble = xr.Dataset(
            {
                "BatterySOCForecastP10": (("time",), [65.0, 60.0, 55.0]),
                "BatterySOCForecastP90": (("time",), [72.0, 75.0, 78.0]),
                SOC_BELOW_THRESHOLD_PROBABILITY_FIELD: (("time",), [0.0, 0.0, 0.1]),
            },
            coords={"time": forecast_times},
        )
        operating = xr.Dataset(
            {
                "ScenarioSOCP10": (("scenario", "time"), [[67.0, 65.0, 63.0], [68.0, 67.0, 66.0], [66.0, 61.0, 56.0], [67.0, 66.0, 64.0], [66.0, 63.0, 60.0]]),
                "ScenarioSOCP50": (("scenario", "time"), [[68.0, 66.0, 64.0], [69.0, 68.0, 67.0], [67.0, 62.0, 57.0], [68.0, 67.0, 65.0], [67.0, 64.0, 61.0]]),
                "ScenarioSOCP90": (("scenario", "time"), [[69.0, 67.0, 65.0], [70.0, 69.0, 68.0], [68.0, 63.0, 58.0], [69.0, 68.0, 66.0], [68.0, 65.0, 62.0]]),
                "ScenarioLoadP50Watts": (("scenario", "time"), np.full((5, 3), 220.0)),
                "ScenarioBelow40Probability": (("scenario", "time"), np.zeros((5, 3))),
                "ScenarioModeCode": (("scenario", "time"), [[1, 1, 1], [0, 0, 0], [1, 1, 1], [0, 1, 1], [2, 2, 2]]),
                "SolarP50Watts": (("time",), [100.0, 200.0, 100.0]),
                "scenario_label": (("scenario",), ["Current Mode", "DC-Only", "DC + CL61", "Optimized CL61", "DC + Radar"]),
            },
            coords={
                "scenario": ["current_mode", "dc_only", "cl61_continuous", "optimized_cl61", "learned_dc_radar"],
                "time": forecast_times,
            },
            attrs={
                "current_mode": "dc_cl61",
                "current_mode_label": "DC + CL61",
                "current_mode_confidence": "0.98",
                "model": "hybrid_state_space_v5",
                "model_version": "5",
                "initial_soc_time": forecast_times[0].isoformat(),
                "generated_at_utc": "2026-07-20T20:05:00+00:00",
                "operating_decision_horizon_hours": "96",
            },
        )
        ensemble.attrs["initial_soc_time"] = forecast_times[0].isoformat()

        summary = build_power_display_summary_dataset(
            power,
            forecast_ds=forecast,
            forecast_skill_ds=skill,
            hindcast_ds=hindcast,
            ensemble_forecast_ds=ensemble,
            operating_scenarios_ds=operating,
            freq="1h",
        )

        self.assertIn("BatterySOCForecast", summary)
        self.assertIn("BatterySOCForecast_Load100W", summary)
        self.assertIn("BatterySOCForecast_Load600W", summary)
        self.assertIn("ECMWFSolarIrradiance", summary)
        self.assertIn("ForecastSOCMAE_0_6h_Verified", summary)
        self.assertIn("ForecastLoadMAE24h", summary)
        self.assertIn("ForecastSolarMAE24h", summary)
        self.assertIn("BatterySOCHindcast_6h", summary)
        self.assertIn("BatterySOCForecastP10", summary)
        self.assertIn(SOC_BELOW_THRESHOLD_PROBABILITY_FIELD, summary)
        self.assertIn("OperatingDCOnlySOCP50", summary)
        self.assertIn("OperatingCL61OptimizedSOCP10", summary)
        self.assertIn("OperatingCL61OptimizedModeCode", summary)
        self.assertIn("OperatingLearned1SOCP50", summary)
        self.assertIn("SystemAsIsDecisionSOCP50", summary)
        np.testing.assert_allclose(
            summary["SystemAsIsDecisionSOCP50"],
            summary["OperatingCurrentSOCP50"],
            equal_nan=True,
        )
        np.testing.assert_allclose(
            summary["SystemAsIsDecisionSOCP10"],
            summary["OperatingCurrentSOCP10"],
            equal_nan=True,
        )
        self.assertEqual(summary.attrs["system_as_is_decision_source"], "operating_scenario")
        self.assertEqual(summary.attrs["operating_initial_soc_time"], forecast_times[0].isoformat())
        self.assertEqual(summary.attrs["operating_decision_horizon_hours"], "96")
        self.assertEqual(summary.attrs["operating_learned_1_label"], "DC + Radar")
        self.assertEqual(summary.attrs["operating_current_mode_label"], "DC + CL61")
        self.assertEqual(summary.attrs["forecast_load_mode"], "DC-Only")
        self.assertEqual(summary.attrs["forecast_load_model"], "mode_conditioned_energy_balance_v5")
        self.assertEqual(summary.attrs["forecast_load_mode_signature"], "ACOutputWatts<=25W")
        self.assertEqual(summary.attrs["forecast_load_mode_learning_ready"], "true")
        self.assertEqual(summary.attrs["forecast_load_measurement"], "battery_discharge_when_solar_zero")
        self.assertEqual(summary.attrs["forecast_load_balance_measurement"], "solar_generation_minus_battery_power")
        self.assertEqual(float(summary.attrs["minimum_operational_soc_pct"]), 40.0)
        self.assertGreater(summary.sizes["time"], power.sizes["time"])

        display_with_anchor = power.copy()
        display_with_anchor.attrs["forecast_initial_soc_time"] = forecast_times[0].isoformat()
        merged = merge_operating_scenarios_into_display_summary(display_with_anchor, operating)
        self.assertIn("OperatingLearned1SOCP50", merged)
        np.testing.assert_allclose(
            merged["SystemAsIsDecisionSOCP50"],
            merged["OperatingCurrentSOCP50"],
            equal_nan=True,
        )
        self.assertEqual(merged.attrs["operating_learned_1_label"], "DC + Radar")
        self.assertEqual(pd.Timestamp(merged["time"].values[-1]), forecast_times[-1])

    def test_operating_plan_anchor_within_refresh_window_is_displayed(self) -> None:
        times = pd.date_range("2026-07-20T20:00:00", periods=3, freq="1h")
        display = xr.Dataset(
            {"BatterySOCForecastP50": (("time",), [90.0, 88.0, 86.0])},
            coords={"time": times},
            attrs={"forecast_initial_soc_time": "2026-07-20T20:15:00"},
        )
        operating = xr.Dataset(
            {"ScenarioSOCP50": (("scenario", "time"), [[90.0, 80.0, 70.0]])},
            coords={"scenario": ["optimized_cl61"], "time": times},
            attrs={"initial_soc_time": "2026-07-20T20:00:00"},
        )

        merged = merge_operating_scenarios_into_display_summary(display, operating)

        self.assertIn("OperatingCL61OptimizedSOCP50", merged)
        self.assertEqual(merged.attrs["operating_planning_status"], "ready")

    def test_system_as_is_decision_uses_one_ensemble_fallback(self) -> None:
        times = pd.date_range("2026-07-20T20:00:00", periods=3, freq="1h")
        display = xr.Dataset(
            {
                "BatterySOCForecastP10": (("time",), [88.0, 82.0, 76.0]),
                "BatterySOCForecastP50": (("time",), [90.0, 86.0, 82.0]),
                "BatterySOCForecastP90": (("time",), [92.0, 90.0, 88.0]),
                SOC_BELOW_THRESHOLD_PROBABILITY_FIELD: (("time",), [0.0, 0.0, 0.1]),
                "BatterySOCForecast": (("time",), [99.0, 99.0, 99.0]),
            },
            coords={"time": times},
        )

        merged = merge_operating_scenarios_into_display_summary(display, None)

        np.testing.assert_allclose(merged["SystemAsIsDecisionSOCP50"], [90.0, 86.0, 82.0])
        self.assertEqual(merged.attrs["system_as_is_decision_source"], "ensemble_fallback")

    def test_mismatched_operating_plan_anchor_is_withheld_from_display(self) -> None:
        times = pd.date_range("2026-07-20T20:00:00", periods=3, freq="1h")
        display = xr.Dataset(
            {
                "BatterySOCForecastP50": (("time",), [90.0, 88.0, 86.0]),
                "OperatingCL61OptimizedSOCP50": (("time",), [90.0, 80.0, 70.0]),
            },
            coords={"time": times},
            attrs={"forecast_initial_soc_time": "2026-07-20T20:29:00"},
        )
        operating = xr.Dataset(
            {
                "ScenarioSOCP50": (("scenario", "time"), [[90.0, 80.0, 70.0]]),
            },
            coords={"scenario": ["optimized_cl61"], "time": times},
            attrs={"initial_soc_time": "2026-07-20T20:00:00"},
        )

        merged = merge_operating_scenarios_into_display_summary(display, operating)

        self.assertNotIn("OperatingCL61OptimizedSOCP50", merged)
        self.assertEqual(merged.attrs["operating_planning_status"], "unavailable")
        self.assertIn("does not match", merged.attrs["operating_planning_status_reason"])

    def test_newer_reanchored_operating_plan_is_displayed(self) -> None:
        times = pd.date_range("2026-07-20T20:00:00", periods=3, freq="1h")
        display = xr.Dataset(
            {"BatterySOCForecastP50": (("time",), [90.0, 88.0, 86.0])},
            coords={"time": times},
            attrs={"forecast_initial_soc_time": "2026-07-20T20:00:00"},
        )
        operating = xr.Dataset(
            {"ScenarioSOCP50": (("scenario", "time"), [[90.0, 80.0, 70.0]])},
            coords={"scenario": ["optimized_cl61"], "time": times},
            attrs={
                "initial_soc_time": "2026-07-20T21:00:00",
                "planning_forecast_initial_soc_time": "2026-07-20T08:00:00",
            },
        )

        merged = merge_operating_scenarios_into_display_summary(display, operating)

        self.assertIn("OperatingCL61OptimizedSOCP50", merged)
        self.assertEqual(merged.attrs["operating_planning_status"], "ready")

    def test_all_soc_decision_panels_draw_40_percent_operational_minimum(self) -> None:
        times = pd.date_range("2026-07-10T00:00:00", periods=4, freq="3h")
        ds = xr.Dataset(
            {
                "BatterySOCForecast": (("time",), [60.0, 55.0, 50.0, 45.0]),
                "BatterySOCForecastP10": (("time",), [58.0, 50.0, 43.0, 35.0]),
                "BatterySOCForecastP90": (("time",), [62.0, 60.0, 57.0, 52.0]),
                SOC_BELOW_THRESHOLD_PROBABILITY_FIELD: (("time",), [0.0, 0.0, 0.0, 0.5]),
                "BatterySOCObservedHindcast": (("time",), [65.0, 60.0, 55.0, 50.0]),
                "BatterySOCHindcast_6h": (("time",), [64.0, 59.0, 54.0, 49.0]),
                "OperatingDCOnlySOCP50": (("time",), [60.0, 58.0, 56.0, 54.0]),
                "OperatingCL61ContinuousSOCP50": (("time",), [60.0, 50.0, 40.0, 30.0]),
                "OperatingCL61OptimizedSOCP50": (("time",), [60.0, 57.0, 53.0, 48.0]),
                "OperatingCL61OptimizedSOCP10": (("time",), [58.0, 54.0, 49.0, 42.0]),
                "OperatingSuggested1SOCP50": (("time",), [60.0, 50.0, 40.0, 30.0]),
                "OperatingSuggested2SOCP50": (("time",), [60.0, 48.0, 36.0, 24.0]),
                "OperatingSuggested3SOCP50": (("time",), [60.0, 47.0, 34.0, 21.0]),
                "OperatingSuggested4SOCP50": (("time",), [60.0, 44.0, 28.0, 12.0]),
                "OperatingSuggested5SOCP50": (("time",), [60.0, 51.0, 42.0, 33.0]),
                "OperatingSuggested6SOCP50": (("time",), [60.0, 54.0, 48.0, 42.0]),
                "OperatingSuggested7SOCP50": (("time",), [60.0, 53.0, 46.0, 39.0]),
                "OperatingSuggested8SOCP50": (("time",), [60.0, 43.0, 25.0, 7.0]),
            },
            coords={"time": times},
        )

        figure = build_summary_plotly(ds, "power")

        references = [trace for trace in figure.data if trace.name == MINIMUM_OPERATIONAL_SOC_REFERENCE_LABEL]
        self.assertEqual(len(references), 4)
        for trace in references:
            np.testing.assert_allclose(trace.y, MINIMUM_OPERATIONAL_SOC_PCT)

        scenario_panel = next(trace for trace in figure.data if trace.name == "All instruments + UAS tier 3")
        np.testing.assert_allclose(scenario_panel.y, [60.0, 43.0, 25.0, 7.0])

    def test_unavailable_operating_product_removes_baked_stale_recommendations(self) -> None:
        times = pd.date_range("2026-07-10T00:00:00", periods=3, freq="1h")
        display = xr.Dataset(
            {"OperatingCL61OptimizedSOCP50": (("time",), [80.0, 70.0, 60.0]), "BatterySOC": (("time",), [90.0, 89.0, 88.0])},
            coords={"time": times},
        )
        unavailable = xr.Dataset(
            coords={"scenario": np.asarray([], dtype=str), "time": np.asarray([], dtype="datetime64[ns]")},
            attrs={"planning_status": "unavailable", "planning_status_reason": "SOC anchor mismatch"},
        )

        merged = merge_operating_scenarios_into_display_summary(display, unavailable)

        self.assertNotIn("OperatingCL61OptimizedSOCP50", merged)
        self.assertEqual(merged.attrs["operating_planning_status"], "unavailable")
        self.assertEqual(merged.attrs["operating_planning_status_reason"], "SOC anchor mismatch")

    def test_only_assigned_pdu_outlet_loads_are_displayed(self) -> None:
        times = pd.date_range("2026-07-10T00:00:00", periods=4, freq="15min")
        ds = xr.Dataset(
            {field: (("time",), np.zeros(len(times))) for field in PDU_WATT_FIELDS},
            coords={"time": times},
        )

        panels = _active_panels(ds, "power")
        pdu_rows = next(rows for panel, rows in panels if panel.key == "pdu_outlet_power")
        figure = build_summary_plotly(ds, "power")

        self.assertEqual(len(pdu_rows), 4)
        self.assertTrue(all(np.allclose(values, 0.0) for _trace, values in pdu_rows))
        outlet_names = {trace.name for trace in figure.data}
        self.assertEqual({"UAS", "CL61", "Radar", "HATPRO"} & outlet_names, {"UAS", "CL61", "Radar", "HATPRO"})
        self.assertFalse({"Outlet 1", "Outlet 2", "Outlet 3", "Outlet 7"} & outlet_names)


if __name__ == "__main__":
    unittest.main()
