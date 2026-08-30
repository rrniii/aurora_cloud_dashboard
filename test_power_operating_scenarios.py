from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import xarray as xr

from power_operating_scenarios import (
    COMPONENT_INDEX,
    MODE_DC_ONLY,
    OperatingEvent,
    SCENARIO_CL61,
    SCENARIO_CURRENT,
    SCENARIO_DC_ONLY,
    SCENARIO_OPTIMIZED,
    SCENARIO_P50_CONTINUATION,
    _align_ensemble_solar_contract,
    _validate_scenario_invariants,
    build_observation_frame,
    build_operating_scenarios,
    describe_cl61_schedule,
    apply_p50_continuation_rule,
    evaluate_p50_continuation_rule,
    evaluate_custom_schedule,
    fit_operating_model,
    integrate_soc_members,
    mode_from_code,
    mode_id,
    mode_kits,
    load_operating_events,
    optimize_cl61_schedule,
    optimize_cl61_primary_schedule,
    optimize_priority_schedule,
    _tier_profile_members,
)
from power_scenario_catalog import SUGGESTED_OPERATING_SCENARIOS
from power_state_catalog import (
    LEARNED_POWER_STATE_IDS,
    POWER_STATE_SCENARIO_IDS,
    UAS_CHARGE_DURATION_HOURS,
    UAS_CHARGE_EMPIRICAL_INCREMENT_P10_W,
    UAS_CHARGE_EMPIRICAL_INCREMENT_P50_W,
    UAS_CHARGE_EMPIRICAL_INCREMENT_P90_W,
    UAS_CHARGE_ESTIMATE_W,
    canonical_uas_tier,
    tier_is_learning_source,
)
from power_load_dynamics import LoadDistribution, StateLoadDynamics
from generate_power_operating_scenarios import (
    _verification_for_record,
    _planning_forecast_provenance,
    _validate_operating_inputs,
    generate as generate_operating_products,
    scenario_publication_signature,
)


def _training_data() -> tuple[xr.Dataset, xr.Dataset]:
    times = pd.date_range("2026-07-15T00:00:00", periods=49, freq="15min")
    cl61_active = times >= pd.Timestamp("2026-07-15T06:00:00")
    solar = np.zeros(len(times), dtype=float)
    load = np.where(cl61_active, 420.0, 200.0)
    power = xr.Dataset(
        {
            "BatterySOC": (("time",), np.linspace(90.0, 86.0, len(times))),
            "BatteryWatts": (("time",), solar - load),
            "SolarWatts_East": (("time",), solar / 3.0),
            "SolarWatts_South": (("time",), solar / 3.0),
            "SolarWatts_West": (("time",), solar / 3.0),
            "ACOutputWatts": (("time",), np.where(cl61_active, 220.0, 0.0)),
            "DCInverterWatts": (("time",), np.full(len(times), 8.0)),
        },
        coords={"time": times},
    )
    pdu = xr.Dataset(
        {
            "PDUOutlet4Watts": (("time",), np.zeros(len(times))),
            "PDUOutlet4State": (("time",), np.zeros(len(times))),
            "PDUOutlet5Watts": (("time",), np.where(cl61_active, 220.0, 0.0)),
            "PDUOutlet5State": (("time",), cl61_active.astype(float)),
            "PDUOutlet6Watts": (("time",), np.zeros(len(times))),
            "PDUOutlet6State": (("time",), np.zeros(len(times))),
            "PDUOutlet8Watts": (("time",), np.zeros(len(times))),
            "PDUOutlet8State": (("time",), np.zeros(len(times))),
        },
        coords={"time": times},
    )
    return power, pdu


def _forecast_inputs(issue: pd.Timestamp, horizon_hours: int = 96) -> tuple[xr.Dataset, xr.Dataset]:
    times = pd.date_range(issue, issue + pd.Timedelta(hours=horizon_hours), freq="3h")
    solar = np.full(len(times), 520.0, dtype=float)
    load = np.full(len(times), 640.0, dtype=float)
    deterministic = xr.Dataset(
        {
            "ForecastSolarWatts": (("time",), solar),
            "ForecastLoadWatts": (("time",), load),
        },
        coords={"time": times},
        attrs={"battery_capacity_kwh": "26", "initial_soc_time": issue.isoformat()},
    )
    members = np.vstack([solar * factor for factor in np.linspace(0.75, 1.25, 20)])
    ensemble = xr.Dataset(
        {
            "ForecastSolarWattsEnsemble": (("member", "time"), members),
            "ForecastLoadWattsEnsemble": (("member", "time"), np.tile(load, (20, 1))),
        },
        coords={"member": np.arange(1, 21), "time": times},
    )
    return deterministic, ensemble


class OperatingScenarioTests(unittest.TestCase):
    def test_canonical_power_state_catalog_matches_the_operator_contract(self) -> None:
        self.assertEqual(
            LEARNED_POWER_STATE_IDS,
            (
                "uas_tier_1",
                "uas_tier_1_charging",
                "uas_tier_2",
                "uas_tier_2_charging",
                "uas_tier_3",
                "uas_tier_3_charging",
                "uas_tier_4",
                "uas_tier_5",
                "cl61",
                "cl61_heater_on",
            ),
        )
        self.assertEqual(len(POWER_STATE_SCENARIO_IDS), 10)
        self.assertEqual(canonical_uas_tier(11), 1)
        self.assertEqual(canonical_uas_tier(12), 2)
        self.assertTrue(tier_is_learning_source(11, 1))
        self.assertTrue(tier_is_learning_source(12, 2))
        self.assertFalse(tier_is_learning_source(1, 1))
        self.assertFalse(tier_is_learning_source(2, 2))

    def test_unknown_uas_tiers_are_ignored_by_learning(self) -> None:
        power, pdu = _training_data()
        times = pd.DatetimeIndex(power["time"].values)
        tiers = pd.Series(
            [11.0, np.nan, 999.0, 12.0],
            index=times[:4],
            dtype=float,
        )

        observations = build_observation_frame(
            power,
            pdu,
            uas_dock1_tier=tiers,
            uas_dock2_tier=tiers,
            lookback_days=2,
        )

        self.assertTrue(observations.loc[times[0], "uas_tier_learning_eligible"])
        self.assertFalse(observations.loc[times[1], "uas_tier_learning_eligible"])
        self.assertFalse(observations.loc[times[2], "uas_tier_learning_eligible"])
        self.assertTrue(observations.loc[times[3], "uas_tier_learning_eligible"])

    def test_mixed_uas_dock_pairs_are_never_single_tier_training_data(self) -> None:
        power, pdu = _training_data()
        times = pd.DatetimeIndex(power["time"].values)
        dock1 = pd.Series(11.0, index=times)
        dock2 = pd.Series(12.0, index=times)

        observations = build_observation_frame(
            power,
            pdu,
            uas_dock1_tier=dock1,
            uas_dock2_tier=dock2,
            lookback_days=2,
        )

        self.assertFalse(observations["uas_tier_learning_eligible"].any())
        self.assertFalse(observations["uas_pair_consistent"].any())
        self.assertTrue(observations["uas_effective_tier"].isna().all())
        self.assertEqual(str(observations["uas_pair_state"].iloc[-1]), "dock1_11__dock2_12")

    def test_p50_continuation_holds_only_currently_on_controlled_instruments(self) -> None:
        times = pd.date_range("2026-08-09T00:00:00", periods=6, freq="1h")
        decision = evaluate_p50_continuation_rule(
            times=times,
            soc_p50=[100.0, 82.0, 61.0, 95.0, 92.0, 88.0],
            current_mode=mode_id(("CL61", "UAS")),
            horizon_hours=5,
        )

        modes = apply_p50_continuation_rule(
            [MODE_DC_ONLY] * len(times),
            decision,
        )

        self.assertTrue(decision.eligible)
        self.assertEqual(decision.held_instruments, ("CL61",))
        self.assertEqual(decision.recovery_time, times[3])
        self.assertEqual(decision.minimum_p50_soc_before_recovery, 61.0)
        self.assertTrue(all("CL61" in mode_kits(value) for value in modes[:4]))
        self.assertTrue(all("CL61" not in mode_kits(value) for value in modes[4:]))
        self.assertTrue(all("UAS" not in mode_kits(value) for value in modes))

    def test_p50_continuation_rejects_a_floor_breach_before_recovery(self) -> None:
        times = pd.date_range("2026-08-09T00:00:00", periods=4, freq="1h")

        decision = evaluate_p50_continuation_rule(
            times=times,
            soc_p50=[90.0, 39.0, 70.0, 95.0],
            current_mode=mode_id(("CL61",)),
        )

        self.assertFalse(decision.eligible)
        self.assertEqual(decision.reason_code, "p50_breaches_minimum_before_recovery")
        self.assertIsNone(decision.hold_through_index)

    def test_p50_continuation_rejects_a_forecast_without_recovery(self) -> None:
        times = pd.date_range("2026-08-09T00:00:00", periods=4, freq="1h")

        decision = evaluate_p50_continuation_rule(
            times=times,
            soc_p50=[90.0, 82.0, 76.0, 70.0],
            current_mode=mode_id(("CL61",)),
        )

        self.assertFalse(decision.eligible)
        self.assertEqual(decision.reason_code, "p50_does_not_recover")

    def test_p50_continuation_keeps_current_instrument_on_when_soc_stays_recovered(self) -> None:
        times = pd.date_range("2026-08-09T00:00:00", periods=5, freq="1h")

        decision = evaluate_p50_continuation_rule(
            times=times,
            soc_p50=[100.0, 99.0, 98.0, 97.0, 96.0],
            current_mode=mode_id(("Radar",)),
            horizon_hours=4,
        )
        modes = apply_p50_continuation_rule([MODE_DC_ONLY] * len(times), decision)

        self.assertTrue(decision.eligible)
        self.assertEqual(decision.reason_code, "p50_stays_at_recovery_target")
        self.assertEqual(decision.recovery_time, times[-1])
        self.assertTrue(all("Radar" in mode_kits(value) for value in modes))

    def test_validation_reanchors_a_planning_forecast_with_an_old_soc_anchor(self) -> None:
        power, _ = _training_data()
        forecast, _ = _forecast_inputs(pd.Timestamp("2026-07-15T00:00:00"), horizon_hours=240)
        forecast.attrs["initial_soc_time"] = "2026-07-13T00:00:00"

        anchor_time, anchor_soc, _ = _validate_operating_inputs(
            power,
            forecast,
            planning_hours=96,
            max_power_age_minutes=None,
            now=pd.Timestamp("2026-07-15T12:00:00"),
        )

        self.assertEqual(anchor_time, pd.Timestamp(power.time.values[-1]))
        self.assertEqual(anchor_soc, float(power.BatterySOC.values[-1]))
    def test_archived_decision_verifies_against_actual_soc_and_mode(self) -> None:
        times = pd.date_range("2026-07-18T00:00:00", periods=3, freq="1h")
        record = {
            "forecast_trace": {
                "time_utc": [value.isoformat() for value in times],
                "soc_p50_pct": [80.0, 79.0, 78.0],
                "mode_code": [0, 0, 1],
            }
        }
        power = xr.Dataset(
            {"BatterySOC": (("time",), [80.0, 78.0, 77.0])},
            coords={"time": times},
        )
        state = xr.Dataset(
            {"OperatingModeCode": (("time",), [0, 0, 1])},
            coords={"time": times},
        )

        verification = _verification_for_record(record, power=power, operating_state=state)

        self.assertIsNotNone(verification)
        self.assertEqual(verification["status"], "complete")
        self.assertAlmostEqual(verification["soc_mae_pct"], 2.0 / 3.0)
        self.assertEqual(verification["mode_adherence_fraction"], 1.0)

    def test_planning_provenance_preserves_cached_cycle_identity(self) -> None:
        times = pd.date_range("2026-07-18T00:00:00", periods=3, freq="1h")
        forecast = xr.Dataset(
            {"ForecastSolarWatts": (("time",), [10.0, 20.0, 30.0])},
            coords={"time": times},
            attrs={
                "generated_at_utc": "2026-07-18T00:05:00+00:00",
                "initial_soc_time": "2026-07-18T00:00:00",
                "forecast_refresh_kind": "cached_reanchor",
                "forecast_verification_eligible": "false",
            },
        )

        provenance = _planning_forecast_provenance(forecast)

        self.assertEqual(provenance["planning_forecast_refresh_kind"], "cached_reanchor")
        self.assertEqual(provenance["planning_forecast_initial_soc_time"], "2026-07-18T00:00:00")
        self.assertEqual(provenance["planning_forecast_time_coverage_start"], "2026-07-18T00:00:00")
        self.assertEqual(provenance["planning_forecast_time_coverage_end"], "2026-07-18T02:00:00")

    def test_mode_code_round_trip_supports_combinations(self) -> None:
        value = mode_id(("CL61", "Radar"))
        code = 0
        for kit in mode_kits(value):
            code |= {"CL61": 1, "Radar": 2}[kit]
        self.assertEqual(mode_from_code(code), value)

    def test_model_learns_dc_baseline_and_cl61_increment(self) -> None:
        power, pdu = _training_data()

        result = fit_operating_model(power, pdu, lookback_days=2)

        self.assertEqual(result.current_mode, mode_id(("CL61",)))
        self.assertGreater(result.current_confidence, 0.90)
        self.assertIn(MODE_DC_ONLY, result.learned_modes)
        self.assertIn(mode_id(("CL61",)), result.learned_modes)
        self.assertIn(MODE_DC_ONLY, result.mode_load_profiles)
        self.assertIn(mode_id(("CL61",)), result.mode_load_profiles)
        self.assertIn("mode_load_profiles", result.state_dataset.attrs)
        self.assertAlmostEqual(result.component_mean[COMPONENT_INDEX["DC"]], 200.0, delta=25.0)
        self.assertAlmostEqual(result.component_mean[COMPONENT_INDEX["CL61"]], 220.0, delta=20.0)
        probabilities = result.state_dataset["OperatingModeProbability"].values
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)

    def test_exact_state_confirmation_requires_all_assigned_pdu_outlets(self) -> None:
        power, pdu = _training_data()
        partial_pdu = pdu[["PDUOutlet5Watts", "PDUOutlet5State"]]

        observations = build_observation_frame(power, partial_pdu, lookback_days=2)

        self.assertFalse(observations["direct_state_confirmed"].any())

    def test_unconfirmed_high_load_does_not_contaminate_dc_only_profile(self) -> None:
        times = pd.date_range("2026-07-15T00:00:00", periods=32, freq="15min")
        load = np.r_[np.full(12, 200.0), np.full(20, 750.0)]
        power = xr.Dataset(
            {
                "BatterySOC": (("time",), np.linspace(90.0, 86.0, len(times))),
                "BatteryWatts": (("time",), -load),
                "SolarWatts_East": (("time",), np.zeros(len(times))),
                "SolarWatts_South": (("time",), np.zeros(len(times))),
                "SolarWatts_West": (("time",), np.zeros(len(times))),
                "ACOutputWatts": (("time",), np.zeros(len(times))),
                "DCInverterWatts": (("time",), np.full(len(times), 8.0)),
            },
            coords={"time": times},
        )
        pdu_times = times[:12]
        pdu = xr.Dataset(
            {
                f"PDUOutlet{outlet}{metric}": (("time",), np.zeros(len(pdu_times)))
                for outlet in (4, 5, 6, 8)
                for metric in ("Watts", "State")
            },
            coords={"time": pdu_times},
        )

        result = fit_operating_model(power, pdu, lookback_days=2)

        profile = result.mode_load_profiles[MODE_DC_ONLY]
        self.assertLess(max(value.p90_w for value in profile.phase_profiles.values()), 300.0)
        self.assertLess(result.component_mean[COMPONENT_INDEX["DC"]], 300.0)
        self.assertGreater(result.state["confirmed_state_observation_count"], 8)

    def test_incompatible_saved_phase_profile_is_relearned(self) -> None:
        power, pdu = _training_data()
        first = fit_operating_model(power, pdu, lookback_days=2)
        poisoned = json.loads(json.dumps(first.state))
        saved = poisoned["mode_load_profiles"][MODE_DC_ONLY]
        saved["schema_version"] = 1
        for phase in saved["phase_profiles"].values():
            phase.update({"p10_w": 850.0, "p50_w": 900.0, "p90_w": 950.0})

        second = fit_operating_model(power, pdu, raw_state=poisoned, lookback_days=2)

        rebuilt = second.state["mode_load_profiles"][MODE_DC_ONLY]
        self.assertEqual(second.state["new_observation_count"], 0)
        self.assertEqual(rebuilt["schema_version"], 3)
        self.assertLess(rebuilt["phase_profiles"]["steady"]["p90_w"], 300.0)

    def test_saved_state_does_not_retrain_the_same_observations(self) -> None:
        power, pdu = _training_data()
        first = fit_operating_model(power, pdu, lookback_days=2)

        second = fit_operating_model(power, pdu, raw_state=first.state, lookback_days=2)

        np.testing.assert_allclose(second.component_mean, first.component_mean, atol=1e-9)
        np.testing.assert_allclose(second.component_covariance, first.component_covariance, atol=1e-9)
        self.assertEqual(second.state["new_observation_count"], 0)

    def test_repeated_cl61_load_levels_become_two_regimes(self) -> None:
        power, pdu = _training_data()
        active = np.asarray(pdu["PDUOutlet5Watts"].values) >= 5.0
        cl61 = np.where(active, np.where(np.arange(len(active)) % 2 == 0, 39.0, 222.0), 0.0)
        pdu["PDUOutlet5Watts"] = (("time",), cl61)
        result = fit_operating_model(power, pdu, lookback_days=2)

        regimes = result.component_regimes["CL61"]
        self.assertEqual(len(regimes), 2)
        self.assertLess(regimes[0]["mean_w"], 50.0)
        self.assertGreater(regimes[1]["mean_w"], 200.0)
        self.assertGreater(result.component_covariance[COMPONENT_INDEX["CL61"], COMPONENT_INDEX["CL61"]], 1000.0)

    def test_short_pdu_combination_is_visible_with_observed_maturity(self) -> None:
        power, pdu = _training_data()
        radar = np.zeros(power.sizes["time"], dtype=float)
        radar[-3:] = 285.0
        pdu["PDUOutlet6Watts"] = (("time",), radar)
        pdu["PDUOutlet6State"] = (("time",), (radar > 0).astype(float))
        result = fit_operating_model(power, pdu, lookback_days=2)
        mode = mode_id(("CL61", "Radar"))

        self.assertIn(mode, result.observed_modes)
        self.assertEqual(result.mode_maturity[mode], "observed")
        self.assertNotIn(mode, result.learned_modes)

    def test_operator_event_file_is_loaded_without_overriding_pdu_mode(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.csv"
            path.write_text("time_utc,action,kit,note\n2026-07-15T06:00:00Z,on,CL61,test\n", encoding="utf-8")
            events = load_operating_events(path)
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0].active)
        self.assertEqual(events[0].kit, "CL61")

    def test_uas_charge_operator_events_are_accepted(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.csv"
            path.write_text(
                "time_utc,action,kit,note\n"
                "2026-07-15T02:00:00Z,on,UASCharge,charging\n"
                "2026-07-15T05:00:00Z,off,UASCharge,complete\n",
                encoding="utf-8",
            )
            events = load_operating_events(path)

        self.assertEqual([value.kit for value in events], ["UASCharge", "UASCharge"])
        self.assertEqual([value.active for value in events], [True, False])

    def test_stale_pdu_evidence_does_not_keep_cl61_active(self) -> None:
        power, pdu = _training_data()
        later_times = pd.date_range("2026-07-15T12:15:00", periods=5, freq="15min")
        later = xr.Dataset(
            {
                "BatterySOC": (("time",), np.linspace(85.9, 85.5, len(later_times))),
                "BatteryWatts": (("time",), np.full(len(later_times), -430.0)),
                "SolarWatts_East": (("time",), np.zeros(len(later_times))),
                "SolarWatts_South": (("time",), np.zeros(len(later_times))),
                "SolarWatts_West": (("time",), np.zeros(len(later_times))),
                "ACOutputWatts": (("time",), np.full(len(later_times), 230.0)),
                "DCInverterWatts": (("time",), np.full(len(later_times), 8.0)),
            },
            coords={"time": later_times},
        )
        extended = xr.concat([power, later], dim="time")

        result = fit_operating_model(extended, pdu, lookback_days=2)

        self.assertEqual(result.current_mode, "unknown_ac")

    def test_operating_scenarios_reject_stale_power_or_expired_solar(self) -> None:
        power, _ = _training_data()
        issue = pd.Timestamp(power["time"].values[-1])
        deterministic, _ = _forecast_inputs(issue, horizon_hours=96)

        with self.assertRaisesRegex(ValueError, "stale SOC/load input"):
            _validate_operating_inputs(
                power,
                deterministic,
                planning_hours=96,
                max_power_age_minutes=20,
                now=issue + pd.Timedelta(minutes=21),
            )

        short_forecast, _ = _forecast_inputs(issue, horizon_hours=90)
        with self.assertRaisesRegex(ValueError, "minimum decision horizon"):
            _validate_operating_inputs(
                power,
                short_forecast,
                planning_hours=240,
                minimum_horizon_hours=96,
                max_power_age_minutes=None,
                now=issue,
            )

    def test_named_scenarios_replace_fixed_watt_curves(self) -> None:
        power, pdu = _training_data()
        model = fit_operating_model(power, pdu, lookback_days=2)
        issue = pd.Timestamp(power["time"].values[-1])
        deterministic, ensemble = _forecast_inputs(issue)

        scenarios = build_operating_scenarios(
            power,
            deterministic,
            model,
            ensemble=ensemble,
            horizon_hours=96,
            optimization_hours=96,
        )

        scenario_ids = set(str(value) for value in scenarios["scenario"].values)
        self.assertTrue(
            {
                SCENARIO_DC_ONLY,
                SCENARIO_CL61,
                SCENARIO_OPTIMIZED,
                SCENARIO_P50_CONTINUATION,
            }.issubset(scenario_ids)
        )
        self.assertNotIn("BatterySOCForecast_Load100W", scenarios)
        self.assertEqual(float(scenarios["ScenarioSOCP50"].isel(time=0).min()), 86.0)
        optimized = scenarios.sel(scenario=SCENARIO_OPTIMIZED)
        self.assertGreaterEqual(float(optimized["ScenarioMinimumP10SOC"]), 40.0)
        expected_active = sum(
            ((np.asarray(optimized["ScenarioModeCode"].values, dtype=np.int16) & bit) > 0).astype(np.int8)
            for bit in (1, 2, 4)
        )
        np.testing.assert_array_equal(
            optimized["ScenarioActiveInstrumentCount"].values,
            expected_active,
        )
        daily = json.loads(scenarios.attrs["optimized_daily_operations"])
        self.assertAlmostEqual(
            sum(float(value["total_instrument_hours"]) for value in daily.values()),
            float(scenarios.attrs["optimized_total_instrument_hours"]),
        )
        for value in daily.values():
            for kit in ("CL61", "Radar", "HATPRO"):
                self.assertAlmostEqual(
                    float(value[f"{kit}_on_hours"]) + float(value[f"{kit}_off_hours"]),
                    float(value["available_clock_hours"]),
                )
        self.assertEqual(scenarios.attrs["optimized_phase_aware_search"], "true")
        self.assertEqual(scenarios.attrs["control_authority"], "advisory_only")
        self.assertEqual(scenarios.attrs["p50_continuation_control_authority"], "advisory_only")
        self.assertIn(scenarios.attrs["p50_continuation_status"], {"eligible", "not_eligible", "not_applicable"})

    def test_unsafe_zero_schedule_is_not_described_as_a_recommendation(self) -> None:
        diagnostic = describe_cl61_schedule(
            current_mode=mode_id(("CL61", "Radar", "HATPRO")),
            safe=False,
            collection_hours=0.0,
            minimum_p10_soc=0.0,
        )

        self.assertEqual(diagnostic.status, "no_safe_schedule")
        self.assertEqual(diagnostic.base_mode, mode_id(("Radar", "HATPRO")))
        self.assertEqual(diagnostic.blocking_instruments, ("Radar", "HATPRO"))
        self.assertTrue(diagnostic.operator_action_required)
        self.assertIn("not a recommendation to switch CL61 off", diagnostic.reason)

    def test_scenario_contract_explains_unsafe_dc_baseline(self) -> None:
        power, pdu = _training_data()
        model = fit_operating_model(power, pdu, lookback_days=2)
        model.current_mode = mode_id(("CL61", "Radar", "HATPRO"))
        model.current_confidence = 1.0
        model.component_mean[:] = [200.0, 220.0, 320.0, 250.0, 0.0, 0.0]
        model.component_covariance[:] = np.diag([4.0, 4.0, 4.0, 4.0, 4.0, 4.0])
        model.mode_load_profiles = {}
        issue = pd.Timestamp(power["time"].values[-1])
        deterministic, ensemble = _forecast_inputs(issue)
        deterministic["ForecastSolarWatts"][:] = 0.0
        ensemble["ForecastSolarWattsEnsemble"][:] = 0.0

        scenarios = build_operating_scenarios(
            power,
            deterministic,
            model,
            ensemble=ensemble,
            horizon_hours=96,
            optimization_hours=96,
        )

        optimized = scenarios.sel(scenario=SCENARIO_OPTIMIZED)
        # A currently-on CL61 is preserved in the diagnostic trace instead of
        # being silently shed at the first scheduler boundary.
        self.assertEqual(float(optimized["ScenarioCollectionHours"]), 96.0)
        self.assertEqual(float(optimized["ScenarioSafe"]), 0.0)
        self.assertEqual(scenarios.attrs["optimized_status"], "no_safe_schedule")
        self.assertEqual(
            json.loads(scenarios.attrs["optimized_blocking_instruments"]),
            [],
        )
        self.assertEqual(
            json.loads(scenarios.attrs["optimized_priority_order"]),
            ["CL61", "Radar", "HATPRO"],
        )
        self.assertIn("not an instruction to switch CL61 off", scenarios.attrs["optimized_reason"])
        self.assertEqual(scenarios.attrs["optimized_operator_action_required"], "true")

    def test_phase_aware_joint_search_retains_safe_single_instrument_subsets(self) -> None:
        power, pdu = _training_data()
        model = fit_operating_model(power, pdu, lookback_days=2)
        cl61_mode = mode_id(("CL61",))
        model.current_mode = cl61_mode
        model.current_confidence = 1.0
        unsafe_combined_modes = (
            mode_id(("CL61", "Radar")),
            mode_id(("CL61", "HATPRO")),
            mode_id(("Radar", "HATPRO")),
            mode_id(("CL61", "Radar", "HATPRO")),
        )
        model.mode_load_profiles = {
            unsafe_mode: StateLoadDynamics(
                state=unsafe_mode,
                current_phase="steady",
                state_started_at="2026-07-15T06:00:00",
                phase_started_at="2026-07-15T06:00:00",
                startup_duration_p10_minutes=0.0,
                startup_duration_p50_minutes=0.0,
                startup_duration_p90_minutes=0.0,
                phase_profiles={"steady": LoadDistribution(5000.0, 5000.0, 5000.0, 24)},
                phase_weights={"steady": 1.0},
                phase_dwell_minutes={"steady": 24.0 * 60.0},
                sample_count=24,
                episode_count=2,
                change_count=0,
            )
            for unsafe_mode in unsafe_combined_modes
        }
        issue = pd.Timestamp(power["time"].values[-1])
        deterministic, ensemble = _forecast_inputs(issue)

        scenarios = build_operating_scenarios(
            power,
            deterministic,
            model,
            ensemble=ensemble,
            horizon_hours=96,
            optimization_hours=96,
        )

        optimized = scenarios.sel(scenario=SCENARIO_OPTIMIZED)
        active_count = np.asarray(optimized["ScenarioActiveInstrumentCount"].values, dtype=np.int8)
        self.assertGreater(int(np.max(active_count[1:])), 0)
        self.assertLessEqual(int(np.max(active_count[1:])), 1)
        self.assertGreater(float(scenarios.attrs["optimized_total_instrument_hours"]), 0.0)
        self.assertEqual(scenarios.attrs["optimized_phase_validation_fallback"], "false")
        self.assertGreaterEqual(float(optimized["ScenarioMinimumP10SOC"]), 40.0)

    def test_native_ensemble_is_preserved_for_the_decision_horizon(self) -> None:
        power, pdu = _training_data()
        model = fit_operating_model(power, pdu, lookback_days=2)
        issue = pd.Timestamp(power["time"].values[-1])
        deterministic, ensemble = _forecast_inputs(issue)
        deterministic["ECMWFSolarIrradiance"] = deterministic["ForecastSolarWatts"] / 2.0
        ensemble["ECMWFSolarIrradianceEnsemble"] = ensemble["ForecastSolarWattsEnsemble"] / 4.0
        deterministic.attrs["solar_calibration_factor_w_per_wm2"] = "2"
        deterministic.attrs["solar_calibration_contract_id"] = "solar-a"
        ensemble.attrs["solar_calibration_contract_id"] = "solar-b"

        aligned, metadata = _align_ensemble_solar_contract(deterministic, ensemble)
        scenarios = build_operating_scenarios(power, deterministic, model, ensemble=ensemble)

        self.assertIsNotNone(aligned)
        np.testing.assert_allclose(
            aligned["ForecastSolarWattsEnsemble"].values,
            ensemble["ForecastSolarWattsEnsemble"].values,
        )
        self.assertEqual(metadata["solar_ensemble_recalibrated"], "false")
        self.assertEqual(metadata["solar_contracts_differ"], "true")
        self.assertEqual(scenarios.attrs["solar_calibration_contract_id"], "solar-b")
        self.assertEqual(scenarios.attrs["solar_ensemble_source_calibration_contract_id"], "solar-b")
        self.assertEqual(scenarios.attrs["solar_decision_calibration_contract_id"], "solar-b")
        self.assertEqual(scenarios.attrs["solar_reserve_tail_calibration_contract_id"], "solar-a")
        self.assertEqual(scenarios.attrs["solar_ensemble_recalibrated"], "false")

    def test_mismatched_solar_contract_without_raw_members_keeps_native_members(self) -> None:
        power, pdu = _training_data()
        model = fit_operating_model(power, pdu, lookback_days=2)
        issue = pd.Timestamp(power["time"].values[-1])
        deterministic, ensemble = _forecast_inputs(issue)
        deterministic["ECMWFSolarIrradiance"] = deterministic["ForecastSolarWatts"] / 2.0
        deterministic.attrs["solar_calibration_contract_id"] = "solar-a"
        ensemble.attrs["solar_calibration_contract_id"] = "solar-b"

        scenarios = build_operating_scenarios(power, deterministic, model, ensemble=ensemble)

        self.assertEqual(scenarios.attrs["solar_decision_calibration_contract_id"], "solar-b")
        self.assertEqual(scenarios.attrs["solar_reserve_tail_calibration_contract_id"], "solar-a")

    def test_scenario_publication_signature_ignores_generation_time(self) -> None:
        power, pdu = _training_data()
        model = fit_operating_model(power, pdu, lookback_days=2)
        issue = pd.Timestamp(power["time"].values[-1])
        deterministic, ensemble = _forecast_inputs(issue)
        scenarios = build_operating_scenarios(
            power,
            deterministic,
            model,
            ensemble=ensemble,
            horizon_hours=96,
        )
        first = scenario_publication_signature(scenarios)
        scenarios.attrs["generated_at_utc"] = "2099-01-01T00:00:00+00:00"

        self.assertEqual(scenario_publication_signature(scenarios), first)
        scenarios.attrs["optimized_status"] = "changed"
        self.assertNotEqual(scenario_publication_signature(scenarios), first)
        scenarios.attrs["optimized_status"] = "safe_schedule"
        first = scenario_publication_signature(scenarios)
        scenarios["ScenarioLoadP50Watts"].values[0, 0] += 50.0
        self.assertNotEqual(scenario_publication_signature(scenarios), first)

    def test_current_scenario_uses_the_finite_operating_state_and_soc_anchor(self) -> None:
        power, pdu = _training_data()
        model = fit_operating_model(power, pdu, lookback_days=2)
        issue = pd.Timestamp(power["time"].values[-1])
        deterministic, ensemble = _forecast_inputs(issue)

        scenarios = build_operating_scenarios(
            power,
            deterministic,
            model,
            ensemble=ensemble,
            horizon_hours=96,
        )
        current = scenarios.sel(scenario=SCENARIO_CURRENT)

        self.assertEqual(pd.Timestamp(scenarios.time.values[0]), issue)
        self.assertEqual(float(current["ScenarioSOCP50"].isel(time=0)), 86.0)
        self.assertAlmostEqual(
            float(current["ScenarioLoadP50Watts"].isel(time=0)),
            420.0,
            places=3,
        )
        self.assertEqual(
            scenarios.attrs["current_system_load_source"],
            "fresh_exact_state_phase_model",
        )
        self.assertEqual(
            scenarios.attrs["load_baseline_source"],
            "finite_state_component_model_for_all_operational_scenarios",
        )
        self.assertEqual(scenarios.attrs["load_state_contract"], "finite_operating_state_phases_v2")
        self.assertIn("ScenarioLoadPhaseCode", scenarios)
        self.assertIn("ScenarioLoadPhaseEpoch", scenarios)
        np.testing.assert_allclose(np.diff(current["ScenarioLoadP50Watts"].values), 0.0)

    def test_current_scenario_prefers_the_fresh_operating_phase_over_stale_planning_load(self) -> None:
        power, pdu = _training_data()
        model = fit_operating_model(power, pdu, lookback_days=2)
        issue = pd.Timestamp(power["time"].values[-1])
        deterministic, ensemble = _forecast_inputs(issue)
        deterministic.attrs.update(
            {
                "load_state_contract": "finite_operating_state_phases_v2",
                "load_state_hold_policy": "hold_confirmed_state_allow_detected_phase_or_explicit_schedule_transition",
                "load_exact_state_id": "dc_cl61",
                "load_current_phase": "fan_low",
            }
        )
        ensemble["ForecastLoadWattsEnsemble"][:] = 515.0

        scenarios = build_operating_scenarios(
            power,
            deterministic,
            model,
            ensemble=ensemble,
            horizon_hours=96,
        )
        current = scenarios.sel(scenario=SCENARIO_CURRENT)
        cl61 = scenarios.sel(scenario=SCENARIO_CL61)

        self.assertFalse(np.allclose(current["ScenarioLoadP50Watts"].values, 515.0))
        np.testing.assert_allclose(
            current["ScenarioLoadP50Watts"].values,
            cl61["ScenarioLoadP50Watts"].values,
        )
        self.assertEqual(
            scenarios.attrs["current_system_load_source"],
            "fresh_exact_state_phase_model",
        )

    def test_cl61_heater_phase_is_the_current_cl61_scenario_load(self) -> None:
        times = pd.date_range("2026-08-01T06:00:00", periods=97, freq="15min")
        low_high = np.asarray(
            [270.0] * 16
            + [460.0] * 16
            + [270.0] * 16
            + [460.0] * 16
            + [270.0] * 16
            + [460.0] * 17
        )
        power = xr.Dataset(
            {
                "BatterySOC": (("time",), np.linspace(96.0, 90.0, len(times))),
                "BatteryWatts": (("time",), -low_high),
                "SolarWatts_East": (("time",), np.zeros(len(times))),
                "SolarWatts_South": (("time",), np.zeros(len(times))),
                "SolarWatts_West": (("time",), np.zeros(len(times))),
                "ACOutputWatts": (("time",), low_high - 240.0),
                "DCInverterWatts": (("time",), np.full(len(times), 8.0)),
            },
            coords={"time": times},
        )
        pdu = xr.Dataset(
            {
                "PDUOutlet4Watts": (("time",), np.zeros(len(times))),
                "PDUOutlet4State": (("time",), np.zeros(len(times))),
                "PDUOutlet5Watts": (("time",), np.where(low_high > 400.0, 223.0, 38.0)),
                "PDUOutlet5State": (("time",), np.ones(len(times))),
                "PDUOutlet6Watts": (("time",), np.zeros(len(times))),
                "PDUOutlet6State": (("time",), np.zeros(len(times))),
                "PDUOutlet8Watts": (("time",), np.zeros(len(times))),
                "PDUOutlet8State": (("time",), np.zeros(len(times))),
            },
            coords={"time": times},
        )
        model = fit_operating_model(power, pdu, lookback_days=2)
        deterministic, ensemble = _forecast_inputs(pd.Timestamp(times[-1]))
        ensemble["ForecastLoadWattsEnsemble"][:] = 270.0

        scenarios = build_operating_scenarios(
            power,
            deterministic,
            model,
            ensemble=ensemble,
            horizon_hours=96,
        )
        current = scenarios.sel(scenario=SCENARIO_CURRENT)
        cl61 = scenarios.sel(scenario=SCENARIO_CL61)
        cl61_normal = scenarios.sel(scenario="state_cl61")
        cl61_heater = scenarios.sel(scenario="state_cl61_heater_on")

        self.assertEqual(model.current_mode, mode_id(("CL61",)))
        self.assertEqual(model.mode_load_profiles[model.current_mode].current_phase, "fan_high")
        self.assertIn("cl61", model.cl61_state_profiles)
        self.assertIn("cl61_heater_on", model.cl61_state_profiles)
        np.testing.assert_allclose(
            current["ScenarioLoadP50Watts"].values,
            cl61["ScenarioLoadP50Watts"].values,
        )
        self.assertGreater(float(current["ScenarioLoadP50Watts"].isel(time=0)), 400.0)
        self.assertGreater(
            float(cl61_heater["ScenarioLoadP50Watts"].isel(time=0)),
            float(cl61_normal["ScenarioLoadP50Watts"].isel(time=0)),
        )
        self.assertTrue(
            np.all(
                cl61_heater["ScenarioPowerState"].values
                == "dc_cl61__cl61_heater_on"
            )
        )

    def test_scenario_validation_rejects_load_drift_inside_one_mode(self) -> None:
        power, pdu = _training_data()
        model = fit_operating_model(power, pdu, lookback_days=2)
        issue = pd.Timestamp(power["time"].values[-1])
        deterministic, ensemble = _forecast_inputs(issue)
        scenarios = build_operating_scenarios(power, deterministic, model, ensemble=ensemble)
        current_index = int(np.flatnonzero(scenarios["scenario"].values == SCENARIO_CURRENT)[0])
        for name in ("ScenarioLoadP10Watts", "ScenarioLoadP50Watts", "ScenarioLoadP90Watts"):
            scenarios[name].values[current_index, 1] += 25.0

        with self.assertRaisesRegex(ValueError, "without an operating-state or phase transition"):
            _validate_scenario_invariants(scenarios)

    def test_all_instruments_uas_tier3_uses_tier_profile_and_stays_above_dc(self) -> None:
        power, pdu = _training_data()
        pdu["PDUOutlet4Watts"] = (("time",), np.full(power.sizes["time"], 108.0))
        pdu["PDUOutlet4State"] = (("time",), np.ones(power.sizes["time"]))
        pdu["PDUOutlet6Watts"] = (("time",), np.full(power.sizes["time"], 285.0))
        pdu["PDUOutlet6State"] = (("time",), np.ones(power.sizes["time"]))
        pdu["PDUOutlet8Watts"] = (("time",), np.full(power.sizes["time"], 230.0))
        pdu["PDUOutlet8State"] = (("time",), np.ones(power.sizes["time"]))
        tier = pd.Series(3.0, index=pd.DatetimeIndex(power["time"].values))
        model = fit_operating_model(
            power,
            pdu,
            uas_dock1_tier=tier,
            uas_dock2_tier=tier,
            lookback_days=2,
        )
        issue = pd.Timestamp(power["time"].values[-1])
        deterministic, ensemble = _forecast_inputs(issue)

        scenarios = build_operating_scenarios(
            power,
            deterministic,
            model,
            ensemble=ensemble,
            horizon_hours=96,
        )

        tier3 = scenarios.sel(scenario="suggested_all_uas_tier3")
        dc_only = scenarios.sel(scenario=SCENARIO_DC_ONLY)
        np.testing.assert_array_equal(tier3["ScenarioUASEffectiveTier"].values, 3)
        self.assertTrue(
            np.all(
                tier3["ScenarioLoadP50Watts"].values
                >= dc_only["ScenarioLoadP50Watts"].values
            )
        )
        self.assertEqual(
            str(tier3["scenario_mode_maturity"].item()),
            "provisional",
        )

    def test_tier_11_and_12_train_canonical_tier_1_and_2_profiles(self) -> None:
        power, pdu = _training_data()
        sample_count = power.sizes["time"]
        raw_tiers = np.r_[
            np.full(13, 11.0),
            np.full(12, 12.0),
            np.full(12, 11.0),
            np.full(sample_count - 37, 12.0),
        ]
        uas_watts = np.where(raw_tiers == 11.0, 115.0, 205.0)
        pdu["PDUOutlet4Watts"] = (("time",), uas_watts)
        pdu["PDUOutlet4State"] = (("time",), np.ones(sample_count))
        tiers = pd.Series(raw_tiers, index=pd.DatetimeIndex(power["time"].values))

        result = fit_operating_model(
            power,
            pdu,
            uas_dock1_tier=tiers,
            uas_dock2_tier=tiers,
            lookback_days=2,
        )

        self.assertEqual(set(result.uas_tier_profiles), {"1", "2"})
        self.assertEqual(result.uas_tier_profiles["1"]["source_effective_tiers"], [11])
        self.assertEqual(result.uas_tier_profiles["2"]["source_effective_tiers"], [12])
        self.assertEqual(result.uas_tier_profiles["1"]["maturity"], "reliable_proxy")
        self.assertEqual(result.uas_tier_profiles["2"]["maturity"], "reliable_proxy")
        self.assertAlmostEqual(result.uas_tier_profiles["1"]["p50_w"], 115.0)
        self.assertAlmostEqual(result.uas_tier_profiles["2"]["p50_w"], 205.0)
        self.assertEqual(result.current_uas_effective_tier, 12)
        self.assertEqual(result.current_uas_tier, 2)
        np.testing.assert_array_equal(
            result.state_dataset["UASCanonicalTier"].values,
            np.where(raw_tiers == 11.0, 1.0, 2.0),
        )

    def test_uas_charge_uses_the_empirical_energy_prior_then_returns_to_base_tier(self) -> None:
        power, pdu = _training_data()
        sample_count = power.sizes["time"]
        pdu["PDUOutlet4Watts"] = (("time",), np.full(sample_count, 108.0))
        pdu["PDUOutlet4State"] = (("time",), np.ones(sample_count))
        tiers = pd.Series(3.0, index=pd.DatetimeIndex(power["time"].values))
        model = fit_operating_model(
            power,
            pdu,
            uas_dock1_tier=tiers,
            uas_dock2_tier=tiers,
            lookback_days=2,
        )
        issue = pd.Timestamp(power["time"].values[-1])
        deterministic, ensemble = _forecast_inputs(issue)

        scenarios = build_operating_scenarios(
            power,
            deterministic,
            model,
            ensemble=ensemble,
            horizon_hours=96,
        )
        base = scenarios.sel(scenario="state_uas_tier_3")
        charging = scenarios.sel(scenario="state_uas_tier_3_charging")
        difference = (
            charging["ScenarioLoadP50Watts"].values
            - base["ScenarioLoadP50Watts"].values
        )
        difference_p10 = (
            charging["ScenarioLoadP10Watts"].values
            - base["ScenarioLoadP10Watts"].values
        )
        difference_p90 = (
            charging["ScenarioLoadP90Watts"].values
            - base["ScenarioLoadP90Watts"].values
        )

        active = np.asarray(charging["ScenarioUASCharging"].values, dtype=bool)
        self.assertTrue(active[0])
        self.assertFalse(active[1:].any())
        # The scenario stores finite-member quantiles, so discrete ranks are
        # close to rather than identically equal to the empirical prior.
        self.assertAlmostEqual(float(difference_p10[0]), UAS_CHARGE_EMPIRICAL_INCREMENT_P10_W, delta=8.0)
        self.assertAlmostEqual(float(difference[0]), UAS_CHARGE_EMPIRICAL_INCREMENT_P50_W, delta=8.0)
        self.assertAlmostEqual(float(difference_p90[0]), UAS_CHARGE_EMPIRICAL_INCREMENT_P90_W, delta=8.0)
        np.testing.assert_allclose(difference[~active], 0.0)
        self.assertLess(UAS_CHARGE_DURATION_HOURS, 1.0)
        self.assertEqual(str(charging["scenario_mode_maturity"].item()), "estimated")
        self.assertEqual(
            str(charging["ScenarioPowerState"].isel(time=0).item()),
            "dc_uas__uas_tier_3_charging",
        )
        self.assertEqual(
            str(charging["ScenarioPowerState"].isel(time=4).item()),
            "dc_uas__uas_tier_3",
        )

    def test_single_explicit_charge_episode_remains_provisional(self) -> None:
        power, pdu = _training_data()
        sample_count = power.sizes["time"]
        uas_watts = np.full(sample_count, 105.0)
        uas_watts[10:22] = 405.0
        pdu["PDUOutlet4Watts"] = (("time",), uas_watts)
        pdu["PDUOutlet4State"] = (("time",), np.ones(sample_count))
        tiers = pd.Series(3.0, index=pd.DatetimeIndex(power["time"].values))
        times = pd.DatetimeIndex(power["time"].values)
        events = (
            OperatingEvent(times[10], "UASCharge", True, "charge started"),
            OperatingEvent(times[22], "UASCharge", False, "charge complete"),
        )

        result = fit_operating_model(
            power,
            pdu,
            uas_dock1_tier=tiers,
            uas_dock2_tier=tiers,
            events=events,
            lookback_days=2,
        )
        charge = result.uas_charge_profiles["3"]

        self.assertEqual(charge["maturity"], "provisional")
        self.assertEqual(charge["episode_count"], 1.0)
        self.assertEqual(charge["observed_hours"], 3.0)
        self.assertEqual(charge["observed_days"], 1.0)
        self.assertAlmostEqual(charge["increment_p50_w"], 300.0)
        self.assertAlmostEqual(charge["duration_p50_hours"], 3.0)
        self.assertAlmostEqual(charge["duration_hours"], UAS_CHARGE_DURATION_HOURS)

    def test_provisional_uas_tier_profile_uses_conservative_fallback(self) -> None:
        members = _tier_profile_members(
            {
                "p10_w": 170.0,
                "p50_w": 175.0,
                "p90_w": 180.0,
                "maturity": "provisional",
            },
            1_000,
            seed=4,
        )

        self.assertAlmostEqual(float(np.median(members)), 108.0, delta=1.0)
        self.assertLess(float(np.quantile(members, 0.10)), 60.0)
        self.assertGreater(float(np.quantile(members, 0.90)), 295.0)

    def test_uas_tier_requires_three_separate_episodes_for_reliability(self) -> None:
        power, pdu = _training_data()
        sample_count = power.sizes["time"]
        pdu["PDUOutlet4Watts"] = (("time",), np.full(sample_count, 175.0))
        pdu["PDUOutlet4State"] = (("time",), np.ones(sample_count))
        tier_values = np.full(sample_count, 3.0)
        tier_values[8:10] = 2.0
        tier_values[18:20] = 2.0
        tiers = pd.Series(tier_values, index=pd.DatetimeIndex(power["time"].values))

        result = fit_operating_model(
            power,
            pdu,
            uas_dock1_tier=tiers,
            uas_dock2_tier=tiers,
            lookback_days=2,
        )
        tier3 = result.uas_tier_profiles["3"]

        self.assertEqual(tier3["episode_count"], 3.0)
        self.assertGreaterEqual(tier3["observed_hours"], 6.0)
        self.assertEqual(tier3["maturity"], "reliable")

    def test_soc_integration_honours_member_specific_capacity(self) -> None:
        times = pd.date_range("2026-07-15T00:00:00", periods=3, freq="1h")
        solar = np.zeros((2, len(times)))
        load = np.full((2, len(times)), 1_000.0)

        soc = integrate_soc_members(
            initial_soc=80.0,
            times=times,
            solar_members_w=solar,
            load_members_w=load,
            capacity_kwh=26.0,
            member_capacity_kwh=np.array([10.0, 20.0]),
            member_charge_efficiency=np.ones(2),
            member_discharge_efficiency=np.ones(2),
        )

        self.assertAlmostEqual(float(soc[0, -1]), 60.0)
        self.assertAlmostEqual(float(soc[1, -1]), 70.0)

    def test_optimizer_enforces_minimum_run_and_daily_start_limit(self) -> None:
        times = pd.date_range("2026-07-15T00:00:00", periods=97, freq="1h")
        solar = np.full((20, len(times)), 500.0)
        components = np.tile(np.array([200.0, 220.0, 300.0, 250.0, 200.0, 250.0]), (20, 1))

        result = optimize_cl61_schedule(
            times=times,
            solar_members_w=solar,
            component_members=components,
            initial_soc=70.0,
            capacity_kwh=26.0,
            base_mode=MODE_DC_ONLY,
            horizon_hours=96,
        )

        on = np.asarray(["CL61" in mode_kits(value) for value in result.modes], dtype=bool)
        starts = np.flatnonzero(on & ~np.r_[False, on[:-1]])
        for start in starts:
            stop_candidates = np.flatnonzero(~on[start:])
            stop = start + int(stop_candidates[0]) if stop_candidates.size else len(on)
            self.assertGreaterEqual(stop - start, 12)
        start_days = [times[index].date() for index in starts]
        self.assertEqual(len(start_days), len(set(start_days)))
        self.assertGreaterEqual(result.minimum_p10_soc, 40.0)

    def test_priority_optimizer_maximizes_additive_controlled_energy(self) -> None:
        times = pd.date_range("2026-07-15T00:00:00", periods=241, freq="1h")
        solar = np.zeros((20, len(times)))
        components = np.tile(
            np.array([40.0, 40.0, 250.0, 200.0, 0.0, 0.0]),
            (20, 1),
        )

        result = optimize_priority_schedule(
            times=times,
            solar_members_w=solar,
            component_members=components,
            initial_soc=100.0,
            capacity_kwh=26.0,
            base_mode=mode_id(("CL61",)),
            horizon_hours=96,
        )

        self.assertTrue(result.safe)
        self.assertGreaterEqual(result.minimum_p10_soc, 40.0)
        self.assertEqual(result.instrument_hours, {"CL61": 15.0, "Radar": 12.0, "HATPRO": 12.0})
        self.assertEqual(result.total_instrument_hours, 39.0)
        self.assertAlmostEqual(result.controlled_energy_kwh, 6.0)
        for kit in ("CL61", "Radar", "HATPRO"):
            on = np.asarray([kit in mode_kits(value) for value in result.modes[:97]], dtype=bool)
            starts = np.flatnonzero(on[1:] & ~on[:-1]) + 1
            self.assertEqual(len(starts), len({times[index].date() for index in starts}))
            for start in starts:
                stop_candidates = np.flatnonzero(~on[start:])
                stop = start + int(stop_candidates[0]) if stop_candidates.size else len(on)
                self.assertGreaterEqual(stop - start, 12)
        self.assertTrue(
            all(
                not ({"CL61", "Radar", "HATPRO"} & set(mode_kits(value)))
                for value in result.modes[97:]
            )
        )

    def test_additive_optimizer_uses_priority_only_to_break_equal_plans(self) -> None:
        times = pd.date_range("2026-07-15T00:00:00", periods=241, freq="1h")
        solar = np.zeros((20, len(times)))
        components = np.tile(
            np.array([40.0, 490.0, 490.0, 490.0, 0.0, 0.0]),
            (20, 1),
        )

        result = optimize_priority_schedule(
            times=times,
            solar_members_w=solar,
            component_members=components,
            initial_soc=100.0,
            capacity_kwh=26.0,
            base_mode=MODE_DC_ONLY,
            horizon_hours=96,
        )

        self.assertTrue(result.safe)
        self.assertEqual(result.instrument_hours, {"CL61": 12.0, "Radar": 0.0, "HATPRO": 0.0})
        self.assertAlmostEqual(result.controlled_energy_kwh, 5.88)

    def test_cl61_primary_policy_reserves_cl61_before_other_instruments(self) -> None:
        times = pd.date_range("2026-07-15T00:00:00", periods=241, freq="1h")
        solar = np.zeros((20, len(times)))
        components = np.tile(
            np.array([40.0, 40.0, 250.0, 200.0, 0.0, 0.0]),
            (20, 1),
        )

        combined, primary = optimize_cl61_primary_schedule(
            times=times,
            solar_members_w=solar,
            component_members=components,
            initial_soc=100.0,
            capacity_kwh=26.0,
            base_mode=MODE_DC_ONLY,
            horizon_hours=96,
        )

        self.assertTrue(primary.safe)
        self.assertTrue(combined.safe)
        self.assertEqual(primary.instrument_hours, {"CL61": 96.0})
        self.assertEqual(combined.instrument_hours["CL61"], 96.0)
        self.assertEqual(combined.instrument_hours["Radar"], 0.0)
        self.assertEqual(combined.instrument_hours["HATPRO"], 0.0)
        self.assertTrue(
            all("CL61" in mode_kits(value) for value in combined.modes[1:97])
        )

    def test_cl61_primary_policy_does_not_shed_an_existing_cl61(self) -> None:
        times = pd.date_range("2026-07-15T00:00:00", periods=241, freq="1h")
        solar = np.zeros((20, len(times)))
        components = np.tile(
            np.array([40.0, 20.0, 400.0, 300.0, 0.0, 0.0]),
            (20, 1),
        )

        combined, primary = optimize_cl61_primary_schedule(
            times=times,
            solar_members_w=solar,
            component_members=components,
            initial_soc=100.0,
            capacity_kwh=26.0,
            base_mode=mode_id(("CL61",)),
            horizon_hours=96,
        )

        self.assertTrue(primary.safe)
        self.assertTrue(combined.safe)
        self.assertEqual(combined.starts, 0)
        self.assertTrue(all("CL61" in mode_kits(value) for value in combined.modes))

    def test_cl61_primary_policy_holds_other_current_pdu_loads(self) -> None:
        times = pd.date_range("2026-07-15T00:00:00", periods=241, freq="1h")
        solar = np.zeros((20, len(times)))
        components = np.tile(
            np.array([40.0, 20.0, 400.0, 300.0, 0.0, 0.0]),
            (20, 1),
        )

        combined, primary = optimize_cl61_primary_schedule(
            times=times,
            solar_members_w=solar,
            component_members=components,
            initial_soc=100.0,
            capacity_kwh=26.0,
            base_mode=mode_id(("CL61", "Radar")),
            horizon_hours=96,
        )

        self.assertTrue(all("Radar" in mode_kits(value) for value in primary.modes))
        self.assertTrue(all("Radar" in mode_kits(value) for value in combined.modes))
        self.assertTrue(all("CL61" in mode_kits(value) for value in combined.modes))

    def test_optimizer_protects_reserve_through_full_planning_horizon(self) -> None:
        times = pd.date_range("2026-07-15T00:00:00", periods=241, freq="1h")
        solar = np.zeros((20, len(times)))
        components = np.tile(np.array([40.0, 100.0, 0.0, 0.0, 0.0, 0.0]), (20, 1))

        result = optimize_cl61_schedule(
            times=times,
            solar_members_w=solar,
            component_members=components,
            initial_soc=80.0,
            capacity_kwh=26.0,
            base_mode=MODE_DC_ONLY,
            horizon_hours=96,
        )

        on = np.asarray(["CL61" in mode_kits(value) for value in result.modes], dtype=bool)
        self.assertEqual(len(result.modes), len(times))
        self.assertEqual(int(np.count_nonzero(on)), 0)
        self.assertTrue(result.safe)
        self.assertGreaterEqual(result.minimum_p10_soc, 40.0)
        self.assertLess(result.minimum_p10_soc, 45.0)

    def test_custom_schedule_reacts_to_start_and_duration(self) -> None:
        power, pdu = _training_data()
        model = fit_operating_model(power, pdu, lookback_days=2)
        issue = pd.Timestamp(power["time"].values[-1])
        deterministic, ensemble = _forecast_inputs(issue)
        scenarios = build_operating_scenarios(power, deterministic, model, ensemble=ensemble, horizon_hours=96)

        short = evaluate_custom_schedule(scenarios, start_time=issue + pd.Timedelta(hours=6), duration_hours=12)
        long = evaluate_custom_schedule(scenarios, start_time=issue + pd.Timedelta(hours=6), duration_hours=24)

        self.assertEqual(short["collection_hours"], 12.0)
        self.assertEqual(long["collection_hours"], 24.0)
        self.assertLessEqual(long["final_p10_soc"], short["final_p10_soc"] + 1e-6)

    def test_custom_schedule_supports_each_learned_instrument_load(self) -> None:
        power, pdu = _training_data()
        model = fit_operating_model(power, pdu, lookback_days=2)
        issue = pd.Timestamp(power["time"].values[-1])
        deterministic, ensemble = _forecast_inputs(issue)
        scenarios = build_operating_scenarios(power, deterministic, model, ensemble=ensemble, horizon_hours=96)

        radar = evaluate_custom_schedule(
            scenarios,
            start_time=issue + pd.Timedelta(hours=6),
            duration_hours=12,
            kit="Radar",
        )

        self.assertEqual(radar["kit"], "Radar")
        self.assertEqual(radar["collection_hours"], 12.0)
        self.assertTrue(any("Radar" in mode_kits(mode) for mode in radar["modes"]))
        self.assertGreater(float(np.nanmax(radar["load_p50_w"])), 0.0)

    def test_planning_horizon_extends_short_ensemble_with_deterministic_shape(self) -> None:
        power, pdu = _training_data()
        model = fit_operating_model(power, pdu, lookback_days=2)
        issue = pd.Timestamp(power["time"].values[-1])
        deterministic_times = pd.date_range(issue, issue + pd.Timedelta(hours=240), freq="3h")
        deterministic_solar = np.maximum(600.0 * np.sin(np.arange(len(deterministic_times)) * np.pi / 8.0), 0.0)
        deterministic = xr.Dataset(
            {"ForecastSolarWatts": (("time",), deterministic_solar)},
            coords={"time": deterministic_times},
            attrs={"battery_capacity_kwh": "26"},
        )
        ensemble_times = deterministic_times[:33]
        ensemble = xr.Dataset(
            {
                "ForecastSolarWattsEnsemble": (
                    ("member", "time"),
                    np.vstack([deterministic_solar[:33] * factor for factor in np.linspace(0.8, 1.2, 20)]),
                )
            },
            coords={"member": np.arange(1, 21), "time": ensemble_times},
        )

        scenarios = build_operating_scenarios(power, deterministic, model, ensemble=ensemble, horizon_hours=240)

        self.assertEqual(scenarios.sizes["time"], 241)
        self.assertEqual(scenarios.attrs["uncertainty_extrapolated"], "true")
        tail_solar = scenarios["SolarP50Watts"].isel(time=slice(-24, None)).values
        self.assertGreater(float(np.nanmax(tail_solar)), 0.0)
        self.assertGreater(float(np.nanmax(tail_solar) - np.nanmin(tail_solar)), 0.0)
        optimized_codes = scenarios.sel(scenario=SCENARIO_OPTIMIZED)["ScenarioModeCode"].values
        # The current observed CL61 state remains held through the full
        # planning horizon under the CL61-first continuation policy.
        self.assertTrue(np.all((optimized_codes[97:] & 1) == 1))

    def test_generator_persists_versioned_state_scenarios_and_recommendation(self) -> None:
        power, pdu = _training_data()
        issue = pd.Timestamp(power["time"].values[-1])
        deterministic, ensemble = _forecast_inputs(issue)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {
                "power": root / "power.zarr",
                "pdu": root / "pdu.zarr",
                "forecast": root / "forecast.zarr",
                "ensemble": root / "ensemble.zarr",
                "state": root / "operating_state.zarr",
                "scenarios": root / "operating_scenarios.zarr",
                "model": root / "model.json",
                "recommendations": root / "recommendations.json",
                "automation_intent": root / "cl61_automation_intent.json",
                "automation_status": root / "cl61_automation_status.json",
                "automation_history": root / "cl61_automation_history.jsonl",
            }
            power.to_zarr(paths["power"], mode="w", consolidated=True)
            pdu.to_zarr(paths["pdu"], mode="w", consolidated=True)
            deterministic.to_zarr(paths["forecast"], mode="w", consolidated=True)
            ensemble.to_zarr(paths["ensemble"], mode="w", consolidated=True)

            generate_operating_products(
                power_zarr=paths["power"],
                pdu_zarr=paths["pdu"],
                forecast_zarr=paths["forecast"],
                ensemble_zarr=paths["ensemble"],
                state_output=paths["state"],
                scenario_output=paths["scenarios"],
                model_state=paths["model"],
                bootstrap_state=None,
                recommendation_archive=paths["recommendations"],
                planning_hours=96,
                optimization_hours=96,
                lookback_days=2,
                automation_intent_output=paths["automation_intent"],
                automation_status_output=paths["automation_status"],
                automation_history_output=paths["automation_history"],
                automation_shadow_enabled=True,
            )

            state = xr.open_zarr(paths["state"], chunks={})
            scenarios = xr.open_zarr(paths["scenarios"], chunks={})
            try:
                self.assertEqual(state.attrs["model_version"], "11")
                self.assertEqual(scenarios.attrs["control_authority"], "advisory_only")
                self.assertIn("optimized_cl61", set(str(value) for value in scenarios["scenario"].values))
                scenario_ids = [str(value) for value in scenarios["scenario"].values]
                scenario_labels = [str(value) for value in scenarios["scenario_label"].values]
                for definition in SUGGESTED_OPERATING_SCENARIOS:
                    self.assertIn(definition.scenario_id, scenario_ids)
                    index = scenario_ids.index(definition.scenario_id)
                    self.assertEqual(scenario_labels[index], definition.label)
                    codes = np.asarray(scenarios["ScenarioModeCode"].isel(scenario=index).values)
                    expected_mode = mode_id(definition.instruments)
                    self.assertTrue(all(mode_from_code(value) == expected_mode for value in codes))
            finally:
                state.close()
                scenarios.close()
            self.assertTrue(paths["model"].exists())
            self.assertTrue(paths["recommendations"].exists())
            automation_status = json.loads(paths["automation_status"].read_text(encoding="utf-8"))
            self.assertEqual(automation_status["mode"], "observe_only")
            self.assertFalse(automation_status["capability"])
            self.assertEqual(automation_status["target"]["pdu_outlet"], 5)
            self.assertTrue(paths["automation_intent"].exists())
            self.assertTrue(paths["automation_history"].exists())
            archive = json.loads(paths["recommendations"].read_text(encoding="utf-8"))
            self.assertEqual(archive["schema_version"], 4)
            record = archive["recommendations"][-1]
            self.assertEqual(record["decision_horizon_hours"], 96)
            self.assertEqual(record["safety_constraint"], "P10 SOC must remain at or above 40%")
            self.assertEqual(record["instrument_priority"], ["CL61", "Radar", "HATPRO"])
            self.assertEqual(set(record["instrument_hours"]), {"CL61", "Radar", "HATPRO"})
            self.assertEqual(set(record["instrument_starts"]), {"CL61", "Radar", "HATPRO"})
            self.assertEqual(record["minimum_run_hours"], 12.0)
            self.assertEqual(record["maximum_starts_per_utc_day"], 1)
            self.assertIn(record["recommendation_status"], {"safe_schedule", "reserve_only"})
            self.assertIn("reason", record)
            self.assertIn("blocking_instruments", record)
            self.assertEqual(
                record["p50_continuation"]["control_authority"],
                "advisory_only",
            )
            self.assertEqual(
                record["p50_continuation"]["scenario_id"],
                SCENARIO_P50_CONTINUATION,
            )
            self.assertEqual(
                len(record["forecast_trace"]["p50_continuation_mode_code"]),
                96,
            )
            self.assertEqual(
                len(record["forecast_trace"]["p50_continuation_soc_p50_pct"]),
                96,
            )
            self.assertEqual(len(record["forecast_trace"]["time_utc"]), 96)
            self.assertEqual(len(record["forecast_trace"]["soc_p50_pct"]), 96)
            self.assertTrue(record["recommended_mode_windows"])
            verification = record["verification"]
            self.assertIsNotNone(verification)
            self.assertGreaterEqual(verification["coverage_hours"], 0.0)
            self.assertIn("soc_mae_pct", verification)


if __name__ == "__main__":
    unittest.main()
