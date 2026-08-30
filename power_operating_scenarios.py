#!/usr/bin/env python3
"""Learn APS operating states and evaluate state-aware SOC scenarios."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from power_soc_thresholds import MINIMUM_OPERATIONAL_SOC_PCT
from power_load_contract import (
    CONTROLLED_LOAD_CONTRACT,
    STATE_HOLD_POLICY,
    ControlledLoadEstimate,
    validate_state_held_load,
)
from power_load_dynamics import (
    LOAD_PHASE_SCHEMA_VERSION,
    PHASE_CODES,
    PHASE_FAN_HIGH,
    PHASE_FAN_LOW,
    PHASE_STEADY,
    StateLoadDynamics,
    controlled_load_member_profiles,
    force_startup,
    learn_state_load_dynamics,
)
from power_scenario_catalog import (
    SUGGESTED_OPERATING_SCENARIOS,
    SUGGESTED_OPERATING_SCENARIO_IDS,
)
from power_state_catalog import (
    LEARNED_POWER_STATE_IDS,
    POWER_STATE_SCENARIOS,
    POWER_STATE_SCENARIO_IDS,
    UAS_CHARGE_DURATION_HOURS,
    UAS_CHARGE_EMPIRICAL_DURATION_P10_HOURS,
    UAS_CHARGE_EMPIRICAL_DURATION_P50_HOURS,
    UAS_CHARGE_EMPIRICAL_DURATION_P90_HOURS,
    UAS_CHARGE_EMPIRICAL_ENERGY_P10_WH,
    UAS_CHARGE_EMPIRICAL_ENERGY_P50_WH,
    UAS_CHARGE_EMPIRICAL_ENERGY_P90_WH,
    UAS_CHARGE_EMPIRICAL_INCREMENT_P10_W,
    UAS_CHARGE_EMPIRICAL_INCREMENT_P50_W,
    UAS_CHARGE_EMPIRICAL_INCREMENT_P90_W,
    UAS_CHARGE_ESTIMATE_W,
    UAS_CHARGE_EVENT_KIT,
    UAS_CHARGE_PLANNING_ENERGY_WH,
    UAS_CHARGE_PRIOR_SOURCE,
    UAS_CHARGE_TIERS,
    UAS_TIER_LEARNING_SOURCES,
    canonical_uas_tier,
    operating_load_state_id,
    state_catalog_records,
    tier_is_learning_source,
    uas_state_id,
    uas_state_label,
)
from power_battery_model import BatteryModel

MODEL_NAME = "hybrid_canonical_uas_cl61_states_v11"
MODEL_VERSION = 11
STATE_SCHEMA_VERSION = 7
SCENARIO_SCHEMA_VERSION = 12

KIT_ORDER = ("CL61", "Radar", "HATPRO", "UAS")
KIT_BITS = {name: 1 << index for index, name in enumerate(KIT_ORDER)}
KIT_OUTLETS = {"UAS": 4, "CL61": 5, "Radar": 6, "HATPRO": 8}
OPERATING_PRIORITY = ("CL61", "Radar", "HATPRO")
SCHEDULE_POLICY_CL61_PRIMARY = "cl61_primary_v1"
SCHEDULE_POLICY_ENERGY_MAXIMISING = "energy_maximising_v1"
DEFAULT_SCHEDULE_POLICY = SCHEDULE_POLICY_CL61_PRIMARY
COMPONENTS = ("DC",) + KIT_ORDER + ("UnknownAC",)
COMPONENT_INDEX = {name: index for index, name in enumerate(COMPONENTS)}

MODE_DC_ONLY = "dc_only"
MODE_UNKNOWN_AC = "unknown_ac"
PDU_ACTIVE_W = 5.0
AC_ACTIVE_W = 25.0
PDU_FRESHNESS_MINUTES = 60.0
OBSERVATION_FREQUENCY = "15min"
MIN_CALIBRATED_SAMPLES = 4
MIN_RELIABLE_SAMPLES = 12
MIN_REGIME_SAMPLES = 4
MIN_RUN_HOURS = 12
MAX_STARTS_PER_UTC_DAY = 1
P50_CONTINUATION_RECOVERY_SOC_PCT = 95.0
P50_CONTINUATION_MINIMUM_SOC_PCT = MINIMUM_OPERATIONAL_SOC_PCT
UAS_TIER_RELIABLE_EPISODES = 3
UAS_TIER_RELIABLE_HOURS = 6.0
UAS_PROXY_TIER_RELIABLE_EPISODES = 2
UAS_CHARGE_RELIABLE_EPISODES = 20
UAS_CHARGE_RELIABLE_HOURS = 5.0
UAS_CHARGE_RELIABLE_DAYS = 5
UAS_TIER3_FALLBACK_P10_W = 55.0
UAS_TIER3_FALLBACK_P50_W = 108.0
UAS_TIER3_FALLBACK_P90_W = 302.0

SCENARIO_CURRENT = "current_mode"
SCENARIO_DC_ONLY = "dc_only"
SCENARIO_CL61 = "cl61_continuous"
SCENARIO_OPTIMIZED = "optimized_cl61"
SCENARIO_P50_CONTINUATION = "p50_continuation"
CORE_SCENARIOS = (
    SCENARIO_CURRENT,
    SCENARIO_DC_ONLY,
    SCENARIO_CL61,
    SCENARIO_OPTIMIZED,
    SCENARIO_P50_CONTINUATION,
)
DEFAULT_EVENTS_PATH = Path(__file__).with_name("config") / "power_operating_events.csv"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class OperatingEvent:
    time: pd.Timestamp
    kit: str
    active: bool
    note: str = ""


def load_operating_events(path: Path | None = DEFAULT_EVENTS_PATH) -> tuple[OperatingEvent, ...]:
    """Load operator intent annotations; PDU telemetry remains electrical truth."""
    if path is None or not Path(path).exists():
        return ()
    events: list[OperatingEvent] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            kit = str(row.get("kit", "")).strip()
            action = str(row.get("action", "")).strip().lower()
            timestamp = pd.to_datetime(row.get("time_utc"), errors="coerce", utc=True)
            if kit not in (*KIT_ORDER, UAS_CHARGE_EVENT_KIT) or action not in {"on", "off"} or pd.isna(timestamp):
                continue
            events.append(
                OperatingEvent(
                    time=pd.Timestamp(timestamp).tz_convert("UTC").tz_localize(None),
                    kit=kit,
                    active=action == "on",
                    note=str(row.get("note", "")).strip(),
                )
            )
    return tuple(sorted(events, key=lambda value: value.time))


def _mode_catalog() -> tuple[str, ...]:
    combinations = [mode_from_code(value) for value in range(1 << len(KIT_ORDER))]
    return tuple(combinations + [MODE_UNKNOWN_AC])


def mode_id(active_kits: Iterable[str], *, unknown_ac: bool = False) -> str:
    active_set = set(active_kits)
    kits = tuple(name for name in KIT_ORDER if name in active_set)
    if kits:
        return "dc_" + "_".join(name.lower() for name in kits)
    return MODE_UNKNOWN_AC if unknown_ac else MODE_DC_ONLY


def mode_kits(value: str) -> tuple[str, ...]:
    text = str(value or "").lower()
    if text in {MODE_DC_ONLY, MODE_UNKNOWN_AC}:
        return ()
    return tuple(name for name in KIT_ORDER if name.lower() in text.split("_"))


def mode_label(value: str) -> str:
    if value == MODE_DC_ONLY:
        return "DC-Only"
    if value == MODE_UNKNOWN_AC:
        return "Unknown AC Load"
    kits = mode_kits(value)
    if kits:
        return "DC + " + " + ".join(kits)
    return str(value).replace("_", " ").title()


@dataclass(frozen=True)
class CL61ScheduleDiagnostic:
    """Explain whether the CL61-only optimiser found an actionable plan."""

    status: str
    reason_code: str
    reason: str
    base_mode: str
    base_mode_label: str
    blocking_instruments: tuple[str, ...]
    operator_action_required: bool


def _human_list(values: Sequence[str]) -> str:
    items = [str(value) for value in values if str(value)]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def describe_cl61_schedule(
    *,
    current_mode: str,
    safe: bool,
    collection_hours: float,
    minimum_p10_soc: float,
    minimum_soc: float = MINIMUM_OPERATIONAL_SOC_PCT,
) -> CL61ScheduleDiagnostic:
    """Classify a CL61 plan without implying control over other instruments."""
    fixed_kits = tuple(name for name in KIT_ORDER if name != "CL61" and name in mode_kits(current_mode))
    base_mode = mode_id(fixed_kits)
    base_label = mode_label(base_mode)
    minimum_text = (
        f"{float(minimum_p10_soc):.1f}%"
        if np.isfinite(float(minimum_p10_soc))
        else "an unknown value"
    )
    fixed_detail = (
        f"{_human_list(fixed_kits)} remain fixed on"
        if fixed_kits
        else "the station DC baseline remains fixed"
    )
    if not safe:
        reason = (
            f"No CL61 schedule can keep P10 SOC at or above {float(minimum_soc):g}%. "
            f"With CL61 off, the fixed {base_label} baseline still reaches {minimum_text}; "
            f"{fixed_detail} because this optimiser controls only CL61. The zero trace is an "
            "unsafe fallback, not a recommendation to switch CL61 off. Operator review is required."
        )
        return CL61ScheduleDiagnostic(
            status="no_safe_schedule",
            reason_code="fixed_baseline_below_reserve",
            reason=reason,
            base_mode=base_mode,
            base_mode_label=base_label,
            blocking_instruments=fixed_kits,
            operator_action_required=True,
        )
    if float(collection_hours) <= 0.0:
        reason = (
            f"No CL61 collection interval satisfies the {float(minimum_soc):g}% P10 reserve while "
            f"the fixed {base_label} baseline continues. Keeping CL61 off is the reserve-preserving "
            "advisory plan; other instrument states are not changed."
        )
        return CL61ScheduleDiagnostic(
            status="reserve_only",
            reason_code="no_safe_collection_window",
            reason=reason,
            base_mode=base_mode,
            base_mode_label=base_label,
            blocking_instruments=fixed_kits,
            operator_action_required=False,
        )
    reason = (
        f"A safe advisory CL61 schedule provides {float(collection_hours):.0f} collection hours while "
        f"holding the fixed {base_label} baseline unchanged and keeping P10 SOC at or above "
        f"{float(minimum_soc):g}%."
    )
    return CL61ScheduleDiagnostic(
        status="safe_schedule",
        reason_code="safe_collection_window",
        reason=reason,
        base_mode=base_mode,
        base_mode_label=base_label,
        blocking_instruments=fixed_kits,
        operator_action_required=False,
    )


def describe_priority_schedule(
    *,
    current_mode: str,
    safe: bool,
    instrument_hours: Mapping[str, float],
    minimum_p10_soc: float,
    priorities: Sequence[str] = OPERATING_PRIORITY,
    minimum_soc: float = MINIMUM_OPERATIONAL_SOC_PCT,
    controlled_energy_kwh: float | None = None,
    policy: str = SCHEDULE_POLICY_ENERGY_MAXIMISING,
    continuation_required: bool = False,
    held_instruments: Sequence[str] = (),
) -> CL61ScheduleDiagnostic:
    """Explain the additive advisory plan without implying automatic PDU control."""
    controlled = tuple(str(value) for value in priorities)
    fixed_kits = tuple(
        name for name in KIT_ORDER if name not in controlled and name in mode_kits(current_mode)
    )
    base_mode = mode_id(fixed_kits)
    base_label = mode_label(base_mode)
    minimum_text = (
        f"{float(minimum_p10_soc):.1f}%"
        if np.isfinite(float(minimum_p10_soc))
        else "an unknown value"
    )
    priority_text = " > ".join(controlled)
    if not safe:
        if policy == SCHEDULE_POLICY_CL61_PRIMARY and continuation_required:
            reason = (
                f"The existing CL61 is deliberately retained in this diagnostic, but its "
                f"CL61-first continuation cannot keep P10 SOC at or above {float(minimum_soc):g}% "
                "through the planning horizon. This is not an instruction to switch CL61 off; "
                "automatic control remains unavailable and operator review is required."
            )
            return CL61ScheduleDiagnostic(
                status="no_safe_schedule",
                reason_code="existing_cl61_continuation_below_reserve",
                reason=reason,
                base_mode=base_mode,
                base_mode_label=base_label,
                blocking_instruments=fixed_kits,
                operator_action_required=True,
            )
        fixed_detail = (
            f"The non-controlled {_human_list(fixed_kits)} load remains on. "
            if fixed_kits
            else ""
        )
        reason = (
            f"No {priority_text} timetable can keep P10 SOC at or above "
            f"{float(minimum_soc):g}%. With all three controlled instruments off, "
            f"the fixed {base_label} baseline reaches {minimum_text}. {fixed_detail}"
            "The zero traces are an unsafe fallback, not an instruction to change PDU outlets. "
            "Operator review is required."
        )
        return CL61ScheduleDiagnostic(
            status="no_safe_schedule",
            reason_code="fixed_baseline_below_reserve",
            reason=reason,
            base_mode=base_mode,
            base_mode_label=base_label,
            blocking_instruments=fixed_kits,
            operator_action_required=True,
        )

    held = {str(name) for name in held_instruments}
    selected = tuple(
        (
            f"{name} {float(instrument_hours.get(name, 0.0)):.0f} h "
            "(held observed state)"
        )
        if name in held
        else f"{name} {float(instrument_hours.get(name, 0.0)):.0f} h"
        for name in controlled
        if float(instrument_hours.get(name, 0.0)) > 0.0
    )
    if not selected:
        reason = (
            f"No controlled-instrument interval satisfies the {float(minimum_soc):g}% P10 reserve. "
            f"The reserve-preserving advisory timetable keeps {priority_text} off; it does not "
            "operate the PDU automatically."
        )
        return CL61ScheduleDiagnostic(
            status="reserve_only",
            reason_code="no_safe_collection_window",
            reason=reason,
            base_mode=base_mode,
            base_mode_label=base_label,
            blocking_instruments=fixed_kits,
            operator_action_required=False,
        )

    total_hours = float(sum(max(float(instrument_hours.get(name, 0.0)), 0.0) for name in controlled))
    energy_text = (
        f" and {float(controlled_energy_kwh):.2f} kWh of additive instrument energy"
        if controlled_energy_kwh is not None and np.isfinite(float(controlled_energy_kwh))
        else ""
    )
    if policy == SCHEDULE_POLICY_CL61_PRIMARY:
        objective = (
            "The CL61-first advisory scheduler reserves the feasible CL61 timetable first, "
            "then adds Radar and HATPRO only from residual reserve. Existing non-CL61 "
            "instruments are held at their observed state rather than implicitly switched."
        )
        reason_code = "safe_cl61_primary_schedule"
    else:
        objective = (
            "The additive advisory scheduler maximises controlled energy first, then total "
            f"instrument-hours; {priority_text} breaks otherwise-equal plans."
        )
        reason_code = "safe_priority_schedule"
    reason = (
        f"{objective} This plan provides {_human_list(selected)}, totalling "
        f"{total_hours:.0f} instrument-hours{energy_text}, while keeping full-horizon P10 SOC "
        f"at or above {float(minimum_soc):g}%. It never operates PDU outlets automatically."
    )
    return CL61ScheduleDiagnostic(
        status="safe_schedule",
        reason_code=reason_code,
        reason=reason,
        base_mode=base_mode,
        base_mode_label=base_label,
        blocking_instruments=fixed_kits,
        operator_action_required=False,
    )


def mode_code(value: str) -> int:
    if value == MODE_UNKNOWN_AC:
        return 1 << len(KIT_ORDER)
    result = 0
    for kit in mode_kits(value):
        result |= KIT_BITS[kit]
    return result


def mode_from_code(value: int) -> str:
    if int(value) & (1 << len(KIT_ORDER)):
        return MODE_UNKNOWN_AC
    return mode_id(name for name, bit in KIT_BITS.items() if int(value) & bit)


def _power_frame(
    power: xr.Dataset,
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    fields = (
        "BatterySOC",
        "BatteryWatts",
        "SolarWatts_East",
        "SolarWatts_South",
        "SolarWatts_West",
        "ACOutputWatts",
        "DCInverterWatts",
    )
    if "time" not in power:
        return pd.DataFrame()
    source = power
    if start is not None or end is not None:
        source = power.sel(time=slice(start, end))
    values = {
        name: np.asarray(source[name].values, dtype=np.float64)
        for name in fields
        if name in source and source[name].dims == ("time",)
    }
    if not values:
        return pd.DataFrame()
    frame = pd.DataFrame(values, index=pd.DatetimeIndex(source["time"].values)).sort_index()
    return frame.loc[~frame.index.duplicated(keep="last")]


def observed_total_load(frame: pd.DataFrame) -> pd.Series:
    solar_names = ("SolarWatts_East", "SolarWatts_South", "SolarWatts_West")
    if "BatteryWatts" in frame and all(name in frame for name in solar_names):
        solar = frame[list(solar_names)].sum(axis=1, min_count=len(solar_names))
        balanced = (solar - frame["BatteryWatts"]).clip(lower=0.0)
        if np.isfinite(balanced.to_numpy(dtype=np.float64)).any():
            balanced.name = "ObservedLoadWatts"
            return balanced
    names = [name for name in ("ACOutputWatts", "DCInverterWatts") if name in frame]
    if not names:
        return pd.Series(dtype=np.float64, name="ObservedLoadWatts")
    result = frame[names].sum(axis=1, min_count=1).clip(lower=0.0)
    result.name = "ObservedLoadWatts"
    return result


def _pdu_frame(
    pdu: xr.Dataset | None,
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    if pdu is None or "time" not in pdu:
        return pd.DataFrame()
    source = pdu
    if start is not None or end is not None:
        source = pdu.sel(time=slice(start, end))
    values: dict[str, np.ndarray] = {}
    for kit, outlet in KIT_OUTLETS.items():
        for metric in ("Watts", "State"):
            name = f"PDUOutlet{outlet}{metric}"
            if name in source and source[name].dims == ("time",):
                values[f"{kit}_{metric.lower()}"] = np.asarray(source[name].values, dtype=np.float64)
    if not values:
        return pd.DataFrame()
    frame = pd.DataFrame(values, index=pd.DatetimeIndex(source["time"].values)).sort_index()
    return frame.loc[~frame.index.duplicated(keep="last")]


def _aligned_uas_tier(series: pd.Series | None, index: pd.DatetimeIndex) -> pd.Series:
    """Return a recent raw dock-tier series aligned to operating observations."""
    if series is None or series.empty:
        return pd.Series(np.nan, index=index, dtype=np.float64)
    tier = pd.Series(
        pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64),
        index=pd.DatetimeIndex(series.index),
        dtype=np.float64,
    ).sort_index()
    if tier.index.tz is not None:
        tier.index = tier.index.tz_convert("UTC").tz_localize(None)
    tier = tier.loc[~tier.index.duplicated(keep="last")]
    return tier.reindex(
        index,
        method="ffill",
        tolerance=pd.Timedelta(minutes=30),
    )


def build_observation_frame(
    power: xr.Dataset,
    pdu: xr.Dataset | None,
    *,
    end: pd.Timestamp | None = None,
    lookback_days: float = 7.0,
    frequency: str = OBSERVATION_FREQUENCY,
    events: Sequence[OperatingEvent] = (),
    uas_tier: pd.Series | None = None,
    uas_dock1_tier: pd.Series | None = None,
    uas_dock2_tier: pd.Series | None = None,
) -> pd.DataFrame:
    if "time" not in power or power.sizes.get("time", 0) == 0:
        return pd.DataFrame()
    power_times = pd.DatetimeIndex(power["time"].values)
    end = pd.Timestamp(end if end is not None else power_times.max())
    start = end - pd.Timedelta(days=float(lookback_days))
    power_frame = _power_frame(power, start=start, end=end)
    if power_frame.empty:
        return pd.DataFrame()
    observed = pd.DataFrame(index=power_frame.resample(frequency).median().index)
    observed["load_w"] = observed_total_load(power_frame).resample(frequency).median()
    for name in ("BatterySOC", "ACOutputWatts"):
        if name in power_frame:
            observed[name] = power_frame[name].resample(frequency).median()

    pdu_frame = _pdu_frame(pdu, start=start, end=end)
    if not pdu_frame.empty:
        pdu_frame = pdu_frame.loc[(pdu_frame.index >= start) & (pdu_frame.index <= end)]
        pdu_samples = pdu_frame.resample(frequency).median()
        for name in pdu_samples:
            observed[name] = pdu_samples[name].reindex(observed.index, method="nearest", tolerance=pd.Timedelta(frequency))
    # The Menapia producer emits (dock1_tier, dock2_tier).  A combined PDU
    # branch cannot identify the component draw of a mixed pair, so only an
    # exact matching pair is allowed to train a canonical single-tier profile.
    if uas_dock1_tier is not None or uas_dock2_tier is not None:
        dock1 = _aligned_uas_tier(uas_dock1_tier, observed.index)
        dock2 = _aligned_uas_tier(uas_dock2_tier, observed.index)
        observed["uas_dock1_tier"] = dock1
        observed["uas_dock2_tier"] = dock2
        pair_complete = np.isfinite(dock1) & np.isfinite(dock2)
        pair_consistent = pair_complete & np.isclose(dock1, dock2)
        observed["uas_pair_consistent"] = pair_consistent
        observed["uas_pair_state"] = [
            f"dock1_{int(left)}__dock2_{int(right)}" if complete else ""
            for left, right, complete in zip(dock1, dock2, pair_complete, strict=True)
        ]
        observed["uas_effective_tier"] = dock1.where(pair_consistent)
        observed["uas_canonical_tier"] = observed["uas_effective_tier"].map(canonical_uas_tier)
        observed["uas_tier_source"] = np.where(
            pair_consistent,
            "matching_dock_pair",
            np.where(pair_complete, "mixed_dock_pair", "incomplete_dock_pair"),
        )
    elif uas_tier is not None and not uas_tier.empty:
        # Retain the legacy single-field adapter only for historical replay.
        # It is never presented as two-dock evidence in new products.
        effective = _aligned_uas_tier(uas_tier, observed.index)
        observed["uas_effective_tier"] = effective
        observed["uas_canonical_tier"] = effective.map(canonical_uas_tier)
        observed["uas_pair_consistent"] = False
        observed["uas_pair_state"] = "legacy_single_tier"
        observed["uas_tier_source"] = "legacy_single_tier"
    else:
        observed["uas_effective_tier"] = np.nan
        observed["uas_canonical_tier"] = np.nan
        observed["uas_pair_consistent"] = False
        observed["uas_pair_state"] = ""
        observed["uas_tier_source"] = "unavailable"
    observed["uas_tier_learning_eligible"] = [
        bool(
            source == "matching_dock_pair"
            and pd.notna(canonical)
            and tier_is_learning_source(raw, int(canonical))
        )
        for raw, canonical, source in zip(
            observed["uas_effective_tier"],
            observed["uas_canonical_tier"],
            observed["uas_tier_source"],
            strict=True,
        )
    ]

    # Charging is an explicitly annotated state, not a wattage guess.  The
    # current field estimate is used until UASCharge on/off events provide a
    # complete observed episode for learning.
    observed["uas_charging"] = np.nan
    for event in (value for value in events if value.kit == UAS_CHARGE_EVENT_KIT):
        observed.loc[observed.index >= event.time, "uas_charging"] = float(event.active)

    mode_values: list[str] = []
    evidence_values: list[str] = []
    confidence_values: list[float] = []
    confirmed_values: list[bool] = []
    for _, row in observed.iterrows():
        active: list[str] = []
        pdu_evidence_count = 0
        for kit in KIT_ORDER:
            watts = row.get(f"{kit}_watts", np.nan)
            state = row.get(f"{kit}_state", np.nan)
            if np.isfinite(watts):
                pdu_evidence_count += 1
                if float(watts) >= PDU_ACTIVE_W:
                    active.append(kit)
            elif np.isfinite(state):
                pdu_evidence_count += 1
                if float(state) >= 0.5:
                    active.append(kit)
        has_pdu_evidence = pdu_evidence_count > 0
        direct_state_confirmed = pdu_evidence_count == len(KIT_ORDER)
        ac_active = bool(np.isfinite(row.get("ACOutputWatts", np.nan)) and row["ACOutputWatts"] > AC_ACTIVE_W)
        if active:
            selected = mode_id(active)
            evidence = "pdu_signature"
            confidence = 0.995 if ac_active else 0.90
        elif has_pdu_evidence and not ac_active:
            selected = MODE_DC_ONLY
            evidence = "pdu_and_ac"
            confidence = 0.995
        elif ac_active:
            selected = MODE_UNKNOWN_AC
            evidence = "ac_output"
            confidence = 0.80
        else:
            selected = MODE_DC_ONLY
            evidence = "ac_output"
            confidence = 0.90
        mode_values.append(selected)
        evidence_values.append(evidence)
        confidence_values.append(confidence)
        confirmed_values.append(direct_state_confirmed)
    observed["direct_mode"] = mode_values
    observed["mode_evidence"] = evidence_values
    observed["direct_confidence"] = confidence_values
    observed["direct_state_confirmed"] = confirmed_values
    observed["operator_event"] = ""
    observed["operator_event_agreement"] = np.nan
    tolerance = pd.Timedelta(minutes=5)
    for event in events:
        if event.time < observed.index.min() - tolerance or event.time > observed.index.max() + tolerance:
            continue
        nearest = observed.index.get_indexer([event.time], method="nearest", tolerance=tolerance)
        if nearest.size == 0 or nearest[0] < 0:
            continue
        index = int(nearest[0])
        if event.kit == UAS_CHARGE_EVENT_KIT:
            observed.iloc[index, observed.columns.get_loc("operator_event")] = (
                f"UAS charging {'on' if event.active else 'off'}"
            )
            continue
        raw_window = pdu_frame.loc[event.time - tolerance : event.time + tolerance] if not pdu_frame.empty else pd.DataFrame()
        watts = raw_window.get(f"{event.kit}_watts", pd.Series(dtype=float)).to_numpy(dtype=np.float64)
        states = raw_window.get(f"{event.kit}_state", pd.Series(dtype=float)).to_numpy(dtype=np.float64)
        active = np.where(np.isfinite(watts), watts >= PDU_ACTIVE_W, states >= 0.5)
        actual = bool(np.any(active == event.active)) if active.size else False
        observed.iloc[index, observed.columns.get_loc("operator_event")] = f"{event.kit} {'on' if event.active else 'off'}"
        observed.iloc[index, observed.columns.get_loc("operator_event_agreement")] = float(actual == event.active)
    subset = [name for name in ("load_w", "BatterySOC", "ACOutputWatts") if name in observed]
    return observed.dropna(how="all", subset=subset) if subset else observed


def _default_component_means(observations: pd.DataFrame) -> np.ndarray:
    means = np.array([200.0, 220.0, 300.0, 250.0, 200.0, 250.0], dtype=np.float64)
    if observations.empty:
        return means
    confirmed = observations.get(
        "direct_state_confirmed",
        pd.Series(False, index=observations.index, dtype=bool),
    ).fillna(False).astype(bool)
    dc = observations.loc[
        confirmed & (observations["direct_mode"] == MODE_DC_ONLY),
        "load_w",
    ].dropna()
    if not dc.empty:
        means[COMPONENT_INDEX["DC"]] = max(float(dc.median()), 0.0)
    for kit in KIT_ORDER:
        field = f"{kit}_watts"
        if field in observations:
            values = observations.loc[observations[field] >= PDU_ACTIVE_W, field].dropna()
            if not values.empty:
                means[COMPONENT_INDEX[kit]] = max(float(values.median()), 0.0)
    return means


def _state_is_compatible(raw_state: Mapping[str, Any] | None) -> bool:
    if not isinstance(raw_state, Mapping):
        return False
    try:
        return (
            int(raw_state.get("model_version", 0) or 0) == MODEL_VERSION
            and int(raw_state.get("schema_version", 0) or 0) == STATE_SCHEMA_VERSION
        )
    except (TypeError, ValueError):
        return False


def _bootstrap_components(raw_state: Mapping[str, Any] | None, observations: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = _default_component_means(observations)
    covariance = np.diag(np.array([60.0, 100.0, 150.0, 120.0, 100.0, 150.0]) ** 2)
    counts = np.zeros(len(COMPONENTS), dtype=np.int64)
    if not _state_is_compatible(raw_state):
        return means, covariance, counts
    component_state = raw_state.get("components")
    if isinstance(component_state, Mapping):
        for name, index in COMPONENT_INDEX.items():
            entry = component_state.get(name)
            if not isinstance(entry, Mapping):
                continue
            try:
                value = float(entry.get("mean_w", np.nan))
                variance = float(entry.get("variance_w2", np.nan))
                count = int(entry.get("observation_count", 0))
            except (TypeError, ValueError):
                continue
            if np.isfinite(value) and value >= 0.0:
                means[index] = value
            if np.isfinite(variance) and variance > 0.0:
                covariance[index, index] = variance
            counts[index] = max(count, 0)
        saved_covariance = raw_state.get("covariance_w2")
        try:
            candidate = np.asarray(saved_covariance, dtype=np.float64)
        except (TypeError, ValueError):
            candidate = np.empty((0, 0), dtype=np.float64)
        if candidate.shape == covariance.shape and np.isfinite(candidate).all():
            candidate = (candidate + candidate.T) / 2.0
            eigenvalues, eigenvectors = np.linalg.eigh(candidate)
            covariance = eigenvectors @ np.diag(np.clip(eigenvalues, 1e-6, None)) @ eigenvectors.T
        return means, covariance, counts

    registry = raw_state.get("load_mode_registry")
    if isinstance(registry, Mapping):
        dc_entry = registry.get("DC-Only")
        if isinstance(dc_entry, Mapping):
            try:
                means[0] = max(float(dc_entry.get("learned_level_w", means[0])), 0.0)
            except (TypeError, ValueError):
                pass
    return means, covariance, counts


def _mode_design(value: str) -> np.ndarray:
    design = np.zeros(len(COMPONENTS), dtype=np.float64)
    design[COMPONENT_INDEX["DC"]] = 1.0
    if value == MODE_UNKNOWN_AC:
        design[COMPONENT_INDEX["UnknownAC"]] = 1.0
    for kit in mode_kits(value):
        design[COMPONENT_INDEX[kit]] = 1.0
    return design


def _kalman_update(
    mean: np.ndarray,
    covariance: np.ndarray,
    design: np.ndarray,
    observation: float,
    measurement_variance: float,
    *,
    innovation_limit_sigma: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    predicted = float(design @ mean)
    innovation = float(observation - predicted)
    innovation_variance = float(design @ covariance @ design + measurement_variance)
    if not np.isfinite(innovation_variance) or innovation_variance <= 0.0:
        return mean, covariance, innovation, False
    limit = float(innovation_limit_sigma * np.sqrt(innovation_variance))
    clipped = bool(abs(innovation) > limit)
    innovation_used = float(np.clip(innovation, -limit, limit))
    gain = covariance @ design / innovation_variance
    updated_mean = np.clip(mean + gain * innovation_used, 0.0, None)
    identity = np.eye(len(mean), dtype=np.float64)
    update_matrix = identity - np.outer(gain, design)
    updated_covariance = update_matrix @ covariance @ update_matrix.T + np.outer(gain, gain) * measurement_variance
    updated_covariance = (updated_covariance + updated_covariance.T) / 2.0
    return updated_mean, updated_covariance, innovation, clipped


def _robust_location_scale(values: np.ndarray) -> tuple[float, float, np.ndarray]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return 0.0, 1.0, finite
    median = float(np.nanmedian(finite))
    mad = float(np.nanmedian(np.abs(finite - median)))
    scale = max(1.4826 * mad, 3.0)
    retained = finite[np.abs(finite - median) <= max(5.0 * scale, 20.0)]
    if not retained.size:
        retained = finite
    return float(np.nanmedian(retained)), max(float(1.4826 * np.nanmedian(np.abs(retained - np.nanmedian(retained)))), 3.0), retained


def _component_regimes(observations: pd.DataFrame) -> dict[str, list[dict[str, float]]]:
    """Derive robust one/two-regime component distributions from direct evidence.

    A large isolated gap is retained only when both sides contain repeated samples.
    That preserves the CL61 high/low regimes while rejecting the single UAS spike.
    """
    regimes: dict[str, list[dict[str, float]]] = {}
    confirmed = observations.get(
        "direct_state_confirmed",
        pd.Series(False, index=observations.index, dtype=bool),
    ).fillna(False).astype(bool)
    dc_values = observations.loc[
        confirmed & (observations["direct_mode"] == MODE_DC_ONLY),
        "load_w",
    ].to_numpy(dtype=np.float64)
    for component in COMPONENTS:
        if component == "DC":
            values = dc_values
        elif component == "UnknownAC":
            values = observations.loc[observations["direct_mode"] == MODE_UNKNOWN_AC, "load_w"].to_numpy(dtype=np.float64)
        else:
            field = f"{component}_watts"
            values = observations.loc[observations.get(field, pd.Series(index=observations.index, dtype=float)) >= PDU_ACTIVE_W, field].to_numpy(dtype=np.float64) if field in observations else np.empty(0)
        values = values[np.isfinite(values)]
        if not values.size:
            regimes[component] = []
            continue
        ordered = np.sort(values)
        split = int(np.argmax(np.diff(ordered))) + 1 if ordered.size >= 2 else 0
        left, right = ordered[:split], ordered[split:]
        gap = float(ordered[split] - ordered[split - 1]) if 0 < split < ordered.size else 0.0
        _, left_scale, _ = _robust_location_scale(left) if left.size else (0.0, 3.0, left)
        _, right_scale, _ = _robust_location_scale(right) if right.size else (0.0, 3.0, right)
        separated = gap >= max(20.0, 3.0 * np.sqrt(left_scale * left_scale + right_scale * right_scale))
        if left.size >= MIN_REGIME_SAMPLES and right.size >= MIN_REGIME_SAMPLES and separated:
            groups = (left, right)
        else:
            _, _, retained = _robust_location_scale(values)
            groups = (retained,)
        component_regimes: list[dict[str, float]] = []
        total = float(sum(len(group) for group in groups))
        for group in groups:
            mean, scale, retained = _robust_location_scale(group)
            component_regimes.append({"mean_w": mean, "std_w": scale, "weight": len(retained) / total, "sample_count": float(len(retained))})
        regimes[component] = component_regimes
    return regimes


def _regime_component_moments(regimes: Mapping[str, Sequence[Mapping[str, float]]], component: str) -> tuple[float, float] | None:
    values = list(regimes.get(component, ()))
    if not values:
        return None
    weights = np.asarray([float(value["weight"]) for value in values], dtype=np.float64)
    weights /= weights.sum()
    means = np.asarray([float(value["mean_w"]) for value in values], dtype=np.float64)
    variances = np.asarray([float(value["std_w"]) ** 2 for value in values], dtype=np.float64)
    mean = float(weights @ means)
    return mean, float(weights @ (variances + (means - mean) ** 2))


def _episode_count(mask: pd.Series, index: pd.DatetimeIndex) -> int:
    continuous = index.to_series().diff().le(pd.Timedelta(minutes=30)).fillna(False)
    starts = mask & ~(mask.shift(fill_value=False) & continuous)
    return int(starts.sum())


def _episode_durations_hours(mask: pd.Series, index: pd.DatetimeIndex) -> np.ndarray:
    continuous = index.to_series().diff().le(pd.Timedelta(minutes=30)).fillna(False)
    starts = mask & ~(mask.shift(fill_value=False) & continuous)
    groups = starts.cumsum()
    counts = groups.loc[mask].value_counts(sort=False).to_numpy(dtype=np.float64)
    sample_hours = float(pd.Timedelta(OBSERVATION_FREQUENCY) / pd.Timedelta(hours=1))
    return counts * sample_hours


def _uas_tier_profiles(observations: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if "uas_effective_tier" not in observations or "UAS_watts" not in observations:
        return {}
    profiles: dict[str, dict[str, Any]] = {}
    raw_tiers = pd.to_numeric(observations["uas_effective_tier"], errors="coerce")
    watts = pd.to_numeric(observations["UAS_watts"], errors="coerce")
    charging = pd.to_numeric(
        observations.get("uas_charging", pd.Series(np.nan, index=observations.index)),
        errors="coerce",
    )
    learning_eligible = observations.get(
        "uas_tier_learning_eligible",
        pd.Series(False, index=observations.index),
    ).fillna(False).astype(bool)
    for tier_value, source_tiers in UAS_TIER_LEARNING_SOURCES.items():
        selected_mask = (
            raw_tiers.isin(source_tiers)
            & learning_eligible
            & np.isfinite(watts)
            & ~(charging >= 0.5)
        )
        selected = observations.loc[selected_mask, ["uas_effective_tier", "UAS_watts"]]
        if selected.empty:
            continue
        episodes = _episode_count(selected_mask, pd.DatetimeIndex(observations.index))
        hours = float(len(selected) * pd.Timedelta(OBSERVATION_FREQUENCY) / pd.Timedelta(hours=1))
        values = selected["UAS_watts"].to_numpy(dtype=np.float64)
        p10, p50, p90 = np.nanquantile(values, (0.10, 0.50, 0.90))
        is_proxy = tier_value in {1, 2}
        minimum_episodes = (
            UAS_PROXY_TIER_RELIABLE_EPISODES if is_proxy else UAS_TIER_RELIABLE_EPISODES
        )
        reliable = episodes >= minimum_episodes and hours >= UAS_TIER_RELIABLE_HOURS
        profiles[str(tier_value)] = {
            "state_id": uas_state_id(tier_value),
            "label": uas_state_label(tier_value),
            "canonical_tier": int(tier_value),
            "source_effective_tiers": list(source_tiers),
            "p10_w": float(max(p10, 0.0)),
            "p50_w": float(max(p50, 0.0)),
            "p90_w": float(max(p90, 0.0)),
            "sample_count": float(len(values)),
            "episode_count": float(episodes),
            "observed_hours": hours,
            "maturity": (
                "reliable_proxy" if reliable and is_proxy else
                "reliable" if reliable else
                "provisional_proxy" if is_proxy else
                "provisional"
            ),
        }
    return profiles


def _uas_charge_profiles(
    observations: pd.DataFrame,
    tier_profiles: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    raw_tiers = pd.to_numeric(
        observations.get("uas_effective_tier", pd.Series(np.nan, index=observations.index)),
        errors="coerce",
    )
    watts = pd.to_numeric(
        observations.get("UAS_watts", pd.Series(np.nan, index=observations.index)),
        errors="coerce",
    )
    charging = pd.to_numeric(
        observations.get("uas_charging", pd.Series(np.nan, index=observations.index)),
        errors="coerce",
    )
    learning_eligible = observations.get(
        "uas_tier_learning_eligible",
        pd.Series(False, index=observations.index),
    ).fillna(False).astype(bool)
    for tier_value in UAS_CHARGE_TIERS:
        state_id = uas_state_id(tier_value, charging=True)
        source_tiers = UAS_TIER_LEARNING_SOURCES[tier_value]
        selected_mask = (
            raw_tiers.isin(source_tiers)
            & learning_eligible
            & (charging >= 0.5)
            & np.isfinite(watts)
        )
        selected = watts.loc[selected_mask]
        base_profile = tier_profiles.get(str(tier_value), {})
        base_p50 = float(base_profile.get("p50_w", 0.0))
        if selected.empty:
            p10 = UAS_CHARGE_EMPIRICAL_INCREMENT_P10_W
            p50 = UAS_CHARGE_EMPIRICAL_INCREMENT_P50_W
            p90 = UAS_CHARGE_EMPIRICAL_INCREMENT_P90_W
            episodes = 0
            hours = 0.0
            active_days = 0
            duration_p10 = UAS_CHARGE_EMPIRICAL_DURATION_P10_HOURS
            duration_p50 = UAS_CHARGE_EMPIRICAL_DURATION_P50_HOURS
            duration_p90 = UAS_CHARGE_EMPIRICAL_DURATION_P90_HOURS
            energy_p10 = UAS_CHARGE_EMPIRICAL_ENERGY_P10_WH
            energy_p50 = UAS_CHARGE_EMPIRICAL_ENERGY_P50_WH
            energy_p90 = UAS_CHARGE_EMPIRICAL_ENERGY_P90_WH
            maturity = "estimated"
        else:
            increments = np.clip(selected.to_numpy(dtype=np.float64) - base_p50, 0.0, None)
            p10, p50, p90 = np.nanquantile(increments, (0.10, 0.50, 0.90))
            episodes = _episode_count(selected_mask, pd.DatetimeIndex(observations.index))
            hours = float(len(selected) * pd.Timedelta(OBSERVATION_FREQUENCY) / pd.Timedelta(hours=1))
            active_days = int(
                len(pd.DatetimeIndex(observations.index[selected_mask]).normalize().unique())
            )
            durations = _episode_durations_hours(
                selected_mask, pd.DatetimeIndex(observations.index)
            )
            duration_p10, duration_p50, duration_p90 = np.nanquantile(
                durations, (0.10, 0.50, 0.90)
            )
            sample_hours = float(
                pd.Timedelta(OBSERVATION_FREQUENCY) / pd.Timedelta(hours=1)
            )
            energy_p10, energy_p50, energy_p90 = np.nanquantile(
                increments * sample_hours,
                (0.10, 0.50, 0.90),
            )
            maturity = (
                "reliable"
                if episodes >= UAS_CHARGE_RELIABLE_EPISODES
                and hours >= UAS_CHARGE_RELIABLE_HOURS
                and active_days >= UAS_CHARGE_RELIABLE_DAYS
                and bool(base_profile)
                else "provisional"
            )
        profiles[str(tier_value)] = {
            "state_id": state_id,
            "label": uas_state_label(tier_value, charging=True),
            "base_state_id": uas_state_id(tier_value),
            "canonical_tier": int(tier_value),
            "source_effective_tiers": list(source_tiers),
            "increment_p10_w": float(max(p10, 0.0)),
            "increment_p50_w": float(max(p50, 0.0)),
            "increment_p90_w": float(max(p90, p50, 0.0)),
            "sample_count": float(len(selected)),
            "episode_count": float(episodes),
            "observed_hours": hours,
            "observed_days": float(active_days),
            "duration_p10_hours": float(duration_p10),
            "duration_p50_hours": float(duration_p50),
            "duration_p90_hours": float(duration_p90),
            "duration_hours": float(
                duration_p50
                if maturity == "reliable"
                else UAS_CHARGE_DURATION_HOURS
            ),
            "energy_p10_wh": float(max(energy_p10, 0.0)),
            "energy_p50_wh": float(max(energy_p50, 0.0)),
            "energy_p90_wh": float(max(energy_p90, energy_p50, 0.0)),
            "planning_energy_wh": float(UAS_CHARGE_PLANNING_ENERGY_WH),
            "prior_source": UAS_CHARGE_PRIOR_SOURCE,
            "maturity": maturity,
            "fallback_increment_w": float(UAS_CHARGE_ESTIMATE_W),
        }
    return profiles


def _cl61_state_profiles(observations: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if "CL61_watts" not in observations:
        return {}
    values = pd.to_numeric(observations["CL61_watts"], errors="coerce")
    selected = values.loc[np.isfinite(values) & (values >= PDU_ACTIVE_W)].to_numpy(dtype=np.float64)
    if not selected.size:
        return {}
    ordered = np.sort(selected)
    split = int(np.argmax(np.diff(ordered))) + 1 if ordered.size >= 2 else 0
    gap = float(ordered[split] - ordered[split - 1]) if 0 < split < ordered.size else 0.0
    normal = ordered[:split]
    heater = ordered[split:]
    separated = (
        normal.size >= MIN_REGIME_SAMPLES
        and heater.size >= MIN_REGIME_SAMPLES
        and gap >= 20.0
    )
    groups = {
        "cl61": normal if separated else ordered,
        "cl61_heater_on": heater if separated else np.empty(0, dtype=np.float64),
    }
    threshold = (
        float((normal[-1] + heater[0]) / 2.0)
        if separated
        else float("nan")
    )
    profiles: dict[str, dict[str, Any]] = {}
    for state_id, group in groups.items():
        if not group.size:
            continue
        p10, p50, p90 = np.nanquantile(group, (0.10, 0.50, 0.90))
        profiles[state_id] = {
            "state_id": state_id,
            "label": "CL61 (heater on)" if state_id == "cl61_heater_on" else "CL61",
            "p10_w": float(max(p10, 0.0)),
            "p50_w": float(max(p50, 0.0)),
            "p90_w": float(max(p90, p50, 0.0)),
            "sample_count": float(len(group)),
            "maturity": "reliable" if len(group) >= MIN_RELIABLE_SAMPLES else "observed",
            "classification_threshold_w": threshold,
        }
    return profiles


def _cl61_state_series(
    observations: pd.DataFrame,
    profiles: Mapping[str, Mapping[str, Any]],
) -> pd.Series:
    values = pd.Series("off", index=observations.index, dtype=object)
    modes = observations.get("direct_mode", pd.Series(MODE_DC_ONLY, index=observations.index))
    active = modes.astype(str).map(lambda value: "CL61" in mode_kits(value))
    values.loc[active] = "cl61"
    heater = profiles.get("cl61_heater_on", {})
    threshold = float(heater.get("classification_threshold_w", np.nan))
    if np.isfinite(threshold) and "CL61_watts" in observations:
        watts = pd.to_numeric(observations["CL61_watts"], errors="coerce")
        values.loc[active & (watts >= threshold)] = "cl61_heater_on"
    return values


def _tier_profile_members(
    profile: Mapping[str, float | str] | None,
    count: int,
    *,
    seed: int,
) -> np.ndarray:
    if profile is None or str(profile.get("maturity", "provisional")) not in {
        "reliable",
        "reliable_proxy",
    }:
        p10, p50, p90 = UAS_TIER3_FALLBACK_P10_W, UAS_TIER3_FALLBACK_P50_W, UAS_TIER3_FALLBACK_P90_W
    else:
        p10 = float(profile.get("p10_w", UAS_TIER3_FALLBACK_P10_W))
        p50 = float(profile.get("p50_w", UAS_TIER3_FALLBACK_P50_W))
        p90 = float(profile.get("p90_w", UAS_TIER3_FALLBACK_P90_W))
    ordered = np.maximum.accumulate(np.asarray([max(p10, 0.0), max(p50, 0.0), max(p90, 0.0)]))
    rng = np.random.default_rng(seed)
    quantiles = (np.arange(max(int(count), 1), dtype=np.float64) + 0.5) / max(int(count), 1)
    rng.shuffle(quantiles)
    return np.interp(quantiles, (0.0, 0.10, 0.50, 0.90, 1.0), (ordered[0], ordered[0], ordered[1], ordered[2], ordered[2]))


def _charge_increment_members(
    profile: Mapping[str, Any] | None,
    count: int,
    *,
    seed: int,
) -> np.ndarray:
    if profile is None or str(profile.get("maturity", "estimated")) != "reliable":
        values = (
            UAS_CHARGE_EMPIRICAL_INCREMENT_P10_W,
            UAS_CHARGE_EMPIRICAL_INCREMENT_P50_W,
            UAS_CHARGE_EMPIRICAL_INCREMENT_P90_W,
        )
    else:
        values = (
            float(profile.get("increment_p10_w", UAS_CHARGE_ESTIMATE_W)),
            float(profile.get("increment_p50_w", UAS_CHARGE_ESTIMATE_W)),
            float(profile.get("increment_p90_w", UAS_CHARGE_ESTIMATE_W)),
        )
    ordered = np.maximum.accumulate(np.clip(np.asarray(values, dtype=np.float64), 0.0, None))
    rng = np.random.default_rng(seed)
    quantiles = (np.arange(max(int(count), 1), dtype=np.float64) + 0.5) / max(int(count), 1)
    rng.shuffle(quantiles)
    return np.interp(
        quantiles,
        (0.0, 0.10, 0.50, 0.90, 1.0),
        (ordered[0], ordered[0], ordered[1], ordered[2], ordered[2]),
    )


@dataclass
class OperatingModelResult:
    state_dataset: xr.Dataset
    state: dict[str, Any]
    component_mean: np.ndarray
    component_covariance: np.ndarray
    learned_modes: tuple[str, ...]
    observed_modes: tuple[str, ...]
    mode_maturity: dict[str, str]
    component_regimes: dict[str, list[dict[str, float]]]
    mode_load_profiles: dict[str, StateLoadDynamics]
    uas_tier_profiles: dict[str, dict[str, Any]]
    uas_charge_profiles: dict[str, dict[str, Any]]
    cl61_state_profiles: dict[str, dict[str, Any]]
    current_mode: str
    current_confidence: float
    current_load_state: str
    current_uas_effective_tier: int | None
    current_uas_tier: int | None
    current_uas_charging: bool
    current_uas_state: str | None
    current_cl61_state: str


def fit_operating_model(
    power: xr.Dataset,
    pdu: xr.Dataset | None,
    *,
    raw_state: Mapping[str, Any] | None = None,
    end: pd.Timestamp | None = None,
    lookback_days: float = 7.0,
    events: Sequence[OperatingEvent] = (),
    uas_tier: pd.Series | None = None,
    uas_dock1_tier: pd.Series | None = None,
    uas_dock2_tier: pd.Series | None = None,
) -> OperatingModelResult:
    observations = build_observation_frame(
        power,
        pdu,
        end=end,
        lookback_days=lookback_days,
        events=events,
        uas_tier=uas_tier,
        uas_dock1_tier=uas_dock1_tier,
        uas_dock2_tier=uas_dock2_tier,
    )
    if observations.empty:
        raise ValueError("No APS/PDU observations are available for operating-state learning")
    compatible_state = _state_is_compatible(raw_state)
    mean, covariance, counts = _bootstrap_components(raw_state, observations)
    process_variance = np.array([1.0, 4.0, 6.0, 5.0, 4.0, 8.0], dtype=np.float64)
    last_trained = pd.NaT
    if compatible_state:
        candidate = pd.to_datetime(raw_state.get("last_observation_time_utc"), errors="coerce")
        if not pd.isna(candidate):
            last_trained = pd.Timestamp(candidate)
            if last_trained.tzinfo is not None:
                last_trained = last_trained.tz_convert("UTC").tz_localize(None)
    train_mask = np.ones(len(observations), dtype=bool) if pd.isna(last_trained) else observations.index > last_trained

    mode_names = _mode_catalog()
    posterior = np.full(len(mode_names), 1.0 / len(mode_names), dtype=np.float64)
    previous_mode = str(raw_state.get("current_mode", MODE_DC_ONLY)) if compatible_state else MODE_DC_ONLY
    if previous_mode in mode_names:
        posterior[:] = 0.02 / max(len(mode_names) - 1, 1)
        posterior[mode_names.index(previous_mode)] = 0.98

    selected_modes: list[str] = []
    confidences: list[float] = []
    estimated_loads: list[float] = []
    innovations: list[float] = []
    outliers: list[float] = []
    probabilities = np.zeros((len(observations), len(mode_names)), dtype=np.float64)

    for row_index, (observation_time, row) in enumerate(observations.iterrows()):
        should_train = bool(train_mask[row_index])
        if should_train:
            covariance = covariance + np.diag(process_variance)
        direct_mode = str(row["direct_mode"])
        direct_confidence = float(row["direct_confidence"])
        prior = 0.985 * posterior + 0.015 / len(mode_names)
        emissions = np.ones(len(mode_names), dtype=np.float64)
        finite_load = np.isfinite(row.get("load_w", np.nan))
        for index, candidate in enumerate(mode_names):
            if direct_mode == candidate:
                emissions[index] *= max(direct_confidence, 0.5)
            else:
                emissions[index] *= max((1.0 - direct_confidence) / max(len(mode_names) - 1, 1), 1e-5)
            if finite_load:
                design = _mode_design(candidate)
                expected = float(design @ mean)
                variance = max(float(design @ covariance @ design + 75.0**2), 1.0)
                residual = float(row["load_w"] - expected)
                emissions[index] *= float(np.exp(-0.5 * residual * residual / variance) / np.sqrt(variance))
        posterior = prior * emissions
        if not np.isfinite(posterior).all() or posterior.sum() <= 0.0:
            posterior = np.full(len(mode_names), 1.0 / len(mode_names), dtype=np.float64)
        else:
            posterior /= posterior.sum()
        selected_index = int(np.argmax(posterior))
        selected_mode = mode_names[selected_index]
        probabilities[row_index] = posterior

        design = _mode_design(selected_mode)
        estimated_load = float(design @ mean)
        innovation = np.nan
        clipped = False
        if finite_load and should_train and bool(row.get("direct_state_confirmed", False)):
            training_design = _mode_design(direct_mode)
            mean, covariance, innovation, clipped = _kalman_update(
                mean,
                covariance,
                training_design,
                float(row["load_w"]),
                75.0**2,
            )
            counts[np.flatnonzero(training_design)] += 1
        for kit in KIT_ORDER:
            field = f"{kit}_watts"
            if not should_train or field not in row or not np.isfinite(row[field]) or float(row[field]) < PDU_ACTIVE_W:
                continue
            direct_design = np.zeros(len(COMPONENTS), dtype=np.float64)
            direct_design[COMPONENT_INDEX[kit]] = 1.0
            mean, covariance, _, _ = _kalman_update(
                mean,
                covariance,
                direct_design,
                float(row[field]),
                25.0**2,
            )
            counts[COMPONENT_INDEX[kit]] += 1

        selected_modes.append(selected_mode)
        confidences.append(float(posterior[selected_index]))
        estimated_loads.append(estimated_load)
        innovations.append(float(innovation))
        outliers.append(float(clipped))

    mode_counts = pd.Series(selected_modes).value_counts()
    observed_modes = tuple(
        value
        for value in mode_names
        if value != MODE_UNKNOWN_AC and int(mode_counts.get(value, 0)) > 0
    )
    mode_maturity = {
        value: (
            "reliable" if int(mode_counts.get(value, 0)) >= MIN_RELIABLE_SAMPLES
            else "calibrated" if int(mode_counts.get(value, 0)) >= MIN_CALIBRATED_SAMPLES
            else "observed"
        )
        for value in observed_modes
    }
    learned_modes = tuple(value for value in observed_modes if mode_maturity[value] in {"calibrated", "reliable"})
    new_observation_count = int(np.count_nonzero(train_mask))
    saved_regimes = raw_state.get("component_regimes") if compatible_state else None
    component_regimes = (
        {str(name): list(values) for name, values in saved_regimes.items()}
        if new_observation_count == 0 and isinstance(saved_regimes, Mapping)
        else _component_regimes(observations)
    )
    saved_mode_profiles = raw_state.get("mode_load_profiles") if compatible_state else None
    mode_load_profiles: dict[str, StateLoadDynamics] = {}
    if new_observation_count == 0 and isinstance(saved_mode_profiles, Mapping):
        for name, value in saved_mode_profiles.items():
            if not isinstance(value, Mapping):
                continue
            try:
                profile = StateLoadDynamics.from_dict(value)
                if profile.state == str(name):
                    mode_load_profiles[str(name)] = profile
            except (KeyError, TypeError, ValueError):
                continue
    saved_cl61_profiles = raw_state.get("cl61_state_profiles") if compatible_state else None
    cl61_state_profiles = (
        {str(name): dict(values) for name, values in saved_cl61_profiles.items()}
        if new_observation_count == 0 and isinstance(saved_cl61_profiles, Mapping)
        else _cl61_state_profiles(observations)
    )
    observations["cl61_state"] = _cl61_state_series(observations, cl61_state_profiles)
    observations["operating_load_state"] = [
        operating_load_state_id(
            str(mode),
            uas_tier=(
                int(tier)
                if "UAS" in mode_kits(str(mode)) and pd.notna(tier)
                else None
            ),
            uas_charging=bool(pd.notna(charging) and float(charging) >= 0.5),
            cl61_heater_on=str(cl61_state) == "cl61_heater_on",
        )
        for mode, tier, charging, cl61_state in zip(
            observations["direct_mode"],
            observations.get(
                "uas_canonical_tier",
                pd.Series(np.nan, index=observations.index),
            ),
            observations.get(
                "uas_charging",
                pd.Series(np.nan, index=observations.index),
            ),
            observations["cl61_state"],
            strict=True,
        )
    ]

    phase_observations = observations.loc[
        observations["direct_state_confirmed"].fillna(False).astype(bool)
    ]
    confirmed_modes = sorted(
        set(str(value) for value in phase_observations["direct_mode"])
        - {MODE_UNKNOWN_AC}
    )
    for name in confirmed_modes:
        if new_observation_count == 0 and name in mode_load_profiles:
            continue
        profile = learn_state_load_dynamics(phase_observations, name)
        if profile is not None and profile.sample_count >= 8:
            mode_load_profiles[name] = profile
    canonical_phase_observations = phase_observations.loc[
        [
            (
                "UAS" not in mode_kits(str(mode))
                or bool(eligible)
            )
            for mode, eligible in zip(
                phase_observations["direct_mode"],
                phase_observations.get(
                    "uas_tier_learning_eligible",
                    pd.Series(False, index=phase_observations.index),
                ),
                strict=True,
            )
        ]
    ]
    confirmed_load_states = sorted(
        set(str(value) for value in canonical_phase_observations["operating_load_state"])
        - {MODE_UNKNOWN_AC}
    )
    for name in confirmed_load_states:
        if name in confirmed_modes:
            continue
        if new_observation_count == 0 and name in mode_load_profiles:
            continue
        profile = learn_state_load_dynamics(
            canonical_phase_observations,
            name,
            mode_column="operating_load_state",
        )
        if profile is not None and profile.sample_count >= 8:
            mode_load_profiles[name] = profile
    saved_tier_profiles = raw_state.get("uas_tier_profiles") if compatible_state else None
    uas_tier_profiles = (
        {str(name): dict(values) for name, values in saved_tier_profiles.items()}
        if new_observation_count == 0 and isinstance(saved_tier_profiles, Mapping)
        else _uas_tier_profiles(observations)
    )
    saved_charge_profiles = raw_state.get("uas_charge_profiles") if compatible_state else None
    uas_charge_profiles = (
        {str(name): dict(values) for name, values in saved_charge_profiles.items()}
        if new_observation_count == 0 and isinstance(saved_charge_profiles, Mapping)
        else _uas_charge_profiles(observations, uas_tier_profiles)
    )
    if new_observation_count > 0 or not compatible_state:
        for name, index in COMPONENT_INDEX.items():
            moment = _regime_component_moments(component_regimes, name)
            if moment is None:
                continue
            mean[index], variance = moment
            covariance[index, index] = max(variance, 9.0)
        eigenvalues, eigenvectors = np.linalg.eigh((covariance + covariance.T) / 2.0)
        covariance = eigenvectors @ np.diag(np.clip(eigenvalues, 1e-6, None)) @ eigenvectors.T
    current_mode = selected_modes[-1]
    current_confidence = confidences[-1]
    latest_row = observations.iloc[-1]
    latest_raw_tier = latest_row.get("uas_effective_tier", np.nan)
    current_uas_effective_tier = (
        int(latest_raw_tier) if pd.notna(latest_raw_tier) else None
    )
    latest_canonical_tier = latest_row.get("uas_canonical_tier", np.nan)
    current_uas_tier = (
        int(latest_canonical_tier)
        if pd.notna(latest_canonical_tier) and "UAS" in mode_kits(current_mode)
        else None
    )
    latest_charging = latest_row.get("uas_charging", np.nan)
    current_uas_charging = bool(
        current_uas_tier in UAS_CHARGE_TIERS
        and pd.notna(latest_charging)
        and float(latest_charging) >= 0.5
    )
    current_uas_state = (
        uas_state_id(current_uas_tier, charging=current_uas_charging)
        if current_uas_tier is not None
        else None
    )
    current_uas_pair_state = str(latest_row.get("uas_pair_state", "") or "")
    current_uas_pair_consistent = bool(latest_row.get("uas_pair_consistent", False))
    current_cl61_state = (
        str(latest_row.get("cl61_state", "cl61"))
        if "CL61" in mode_kits(current_mode)
        else "off"
    )
    current_load_state = operating_load_state_id(
        current_mode,
        uas_tier=current_uas_tier,
        uas_charging=current_uas_charging,
        cl61_heater_on=current_cl61_state == "cl61_heater_on",
    )
    operating_load_states = np.asarray(
        [
            operating_load_state_id(
                mode,
                uas_tier=(
                    int(tier)
                    if "UAS" in mode_kits(mode) and pd.notna(tier)
                    else None
                ),
                uas_charging=bool(pd.notna(charging) and float(charging) >= 0.5),
                cl61_heater_on=str(cl61_state) == "cl61_heater_on",
            )
            for mode, tier, charging, cl61_state in zip(
                selected_modes,
                observations.get(
                    "uas_canonical_tier",
                    pd.Series(np.nan, index=observations.index),
                ),
                observations.get(
                    "uas_charging",
                    pd.Series(np.nan, index=observations.index),
                ),
                observations["cl61_state"],
                strict=True,
            )
        ],
        dtype=str,
    )

    catalog = state_catalog_records()
    learned_state_p10: list[float] = []
    learned_state_p50: list[float] = []
    learned_state_p90: list[float] = []
    learned_state_samples: list[float] = []
    learned_state_maturity: list[str] = []
    for definition in catalog:
        state_id = str(definition["id"])
        if state_id.startswith("uas_tier_"):
            tier_value = int(state_id.split("_")[2])
            base = uas_tier_profiles.get(str(tier_value), {})
            charging_state = state_id.endswith("_charging")
            charge = uas_charge_profiles.get(str(tier_value), {}) if charging_state else {}
            learned_state_p10.append(
                float(base.get("p10_w", np.nan))
                + (float(charge.get("increment_p10_w", UAS_CHARGE_ESTIMATE_W)) if charging_state else 0.0)
            )
            learned_state_p50.append(
                float(base.get("p50_w", np.nan))
                + (float(charge.get("increment_p50_w", UAS_CHARGE_ESTIMATE_W)) if charging_state else 0.0)
            )
            learned_state_p90.append(
                float(base.get("p90_w", np.nan))
                + (float(charge.get("increment_p90_w", UAS_CHARGE_ESTIMATE_W)) if charging_state else 0.0)
            )
            learned_state_samples.append(float(charge.get("sample_count", 0.0) if charging_state else base.get("sample_count", 0.0)))
            learned_state_maturity.append(str(charge.get("maturity", "estimated") if charging_state else base.get("maturity", "unobserved")))
        else:
            profile = cl61_state_profiles.get(state_id, {})
            learned_state_p10.append(float(profile.get("p10_w", np.nan)))
            learned_state_p50.append(float(profile.get("p50_w", np.nan)))
            learned_state_p90.append(float(profile.get("p90_w", np.nan)))
            learned_state_samples.append(float(profile.get("sample_count", 0.0)))
            learned_state_maturity.append(str(profile.get("maturity", "unobserved")))
    latest_observation_time = pd.Timestamp(observations.index[-1])
    if not pd.isna(last_trained) and new_observation_count == 0:
        latest_observation_time = last_trained
    state_ds = xr.Dataset(
        {
            "OperatingModeCode": (("time",), np.asarray([mode_code(value) for value in selected_modes], dtype=np.int16)),
            "OperatingModeConfidence": (("time",), np.asarray(confidences, dtype=np.float32)),
            "ObservedLoadWatts": (("time",), observations["load_w"].to_numpy(dtype=np.float32)),
            "EstimatedModeLoadWatts": (("time",), np.asarray(estimated_loads, dtype=np.float32)),
            "LoadInnovationWatts": (("time",), np.asarray(innovations, dtype=np.float32)),
            "LoadObservationOutlier": (("time",), np.asarray(outliers, dtype=np.float32)),
            "DirectStateConfirmed": (
                ("time",),
                observations["direct_state_confirmed"].to_numpy(dtype=np.uint8),
            ),
            "OperatingEventAgreement": (("time",), observations["operator_event_agreement"].to_numpy(dtype=np.float32)),
            "OperatingModeProbability": (("time", "mode"), probabilities.astype(np.float32)),
            "OperatingLoadState": (("time",), operating_load_states),
            "ComponentRegimeMeanWatts": (("component", "regime"), np.asarray([
                [component_regimes.get(component, [{}] * 2)[index].get("mean_w", np.nan) if index < len(component_regimes.get(component, ())) else np.nan for index in range(2)]
                for component in COMPONENTS
            ], dtype=np.float32)),
            "ComponentRegimeStdWatts": (("component", "regime"), np.asarray([
                [component_regimes.get(component, [{}] * 2)[index].get("std_w", np.nan) if index < len(component_regimes.get(component, ())) else np.nan for index in range(2)]
                for component in COMPONENTS
            ], dtype=np.float32)),
            "ComponentRegimeWeight": (("component", "regime"), np.asarray([
                [component_regimes.get(component, [{}] * 2)[index].get("weight", np.nan) if index < len(component_regimes.get(component, ())) else np.nan for index in range(2)]
                for component in COMPONENTS
            ], dtype=np.float32)),
            "ComponentRegimeSampleCount": (("component", "regime"), np.asarray([
                [component_regimes.get(component, [{}] * 2)[index].get("sample_count", np.nan) if index < len(component_regimes.get(component, ())) else np.nan for index in range(2)]
                for component in COMPONENTS
            ], dtype=np.float32)),
            "UASEffectiveTier": (
                ("time",),
                observations.get(
                    "uas_effective_tier",
                    pd.Series(np.nan, index=observations.index),
                ).to_numpy(dtype=np.float32),
            ),
            "UASDock1Tier": (
                ("time",),
                observations.get(
                    "uas_dock1_tier",
                    pd.Series(np.nan, index=observations.index),
                ).to_numpy(dtype=np.float32),
            ),
            "UASDock2Tier": (
                ("time",),
                observations.get(
                    "uas_dock2_tier",
                    pd.Series(np.nan, index=observations.index),
                ).to_numpy(dtype=np.float32),
            ),
            "UASPairConsistent": (
                ("time",),
                observations.get(
                    "uas_pair_consistent",
                    pd.Series(False, index=observations.index),
                ).to_numpy(dtype=np.uint8),
            ),
            "UASPairState": (
                ("time",),
                observations.get(
                    "uas_pair_state",
                    pd.Series("", index=observations.index),
                ).astype(str).to_numpy(dtype=str),
            ),
            "UASCanonicalTier": (
                ("time",),
                observations.get(
                    "uas_canonical_tier",
                    pd.Series(np.nan, index=observations.index),
                ).to_numpy(dtype=np.float32),
            ),
            "UASChargingState": (
                ("time",),
                observations.get(
                    "uas_charging",
                    pd.Series(np.nan, index=observations.index),
                ).to_numpy(dtype=np.float32),
            ),
            "CL61StateCode": (
                ("time",),
                observations["cl61_state"].map(
                    {"off": 0, "cl61": 1, "cl61_heater_on": 2}
                ).to_numpy(dtype=np.int8),
            ),
            "LearnedStateLoadP10Watts": (
                ("learned_state",),
                np.asarray(learned_state_p10, dtype=np.float32),
            ),
            "LearnedStateLoadP50Watts": (
                ("learned_state",),
                np.asarray(learned_state_p50, dtype=np.float32),
            ),
            "LearnedStateLoadP90Watts": (
                ("learned_state",),
                np.asarray(learned_state_p90, dtype=np.float32),
            ),
            "LearnedStateSampleCount": (
                ("learned_state",),
                np.asarray(learned_state_samples, dtype=np.float32),
            ),
        },
        coords={
            "time": observations.index.to_numpy(dtype="datetime64[ns]"),
            "mode": np.asarray(mode_names, dtype=str),
            "component": np.asarray(COMPONENTS, dtype=str),
            "regime": np.asarray(("low", "high"), dtype=str),
            "learned_state": np.asarray(LEARNED_POWER_STATE_IDS, dtype=str),
            "learned_state_label": (
                ("learned_state",),
                np.asarray([str(value["label"]) for value in catalog], dtype=str),
            ),
            "learned_state_maturity": (
                ("learned_state",),
                np.asarray(learned_state_maturity, dtype=str),
            ),
        },
        attrs={
            "power_operating_state_product": "true",
            "schema_version": str(STATE_SCHEMA_VERSION),
            "model": MODEL_NAME,
            "model_version": str(MODEL_VERSION),
            "load_phase_schema_version": str(LOAD_PHASE_SCHEMA_VERSION),
            "generated_at_utc": _utc_now(),
            "current_mode": current_mode,
            "current_mode_label": mode_label(current_mode),
            "current_mode_confidence": f"{current_confidence:.6g}",
            "current_load_state": current_load_state,
            "current_uas_effective_tier": (
                str(current_uas_effective_tier)
                if current_uas_effective_tier is not None
                else ""
            ),
            "current_uas_canonical_tier": (
                str(current_uas_tier) if current_uas_tier is not None else ""
            ),
            "current_uas_charging": str(current_uas_charging).lower(),
            "current_uas_state": current_uas_state or "",
            "current_uas_pair_state": current_uas_pair_state,
            "current_uas_pair_consistent": str(current_uas_pair_consistent).lower(),
            "current_cl61_state": current_cl61_state,
            "learned_power_state_catalog": json.dumps(catalog, sort_keys=True),
            "learned_modes": json.dumps(list(learned_modes)),
            "observed_modes": json.dumps(list(observed_modes)),
            "mode_maturity": json.dumps(mode_maturity, sort_keys=True),
            "operator_event_count": str(len(events)),
            "operator_event_agreements": str(int(np.nansum(observations["operator_event_agreement"].to_numpy(dtype=np.float64)))),
            "component_names": json.dumps(list(COMPONENTS)),
            "mode_load_profiles": json.dumps(
                {name: profile.to_dict() for name, profile in mode_load_profiles.items()},
                sort_keys=True,
            ),
            "component_mean_w": json.dumps({name: float(mean[index]) for index, name in enumerate(COMPONENTS)}),
            "component_std_w": json.dumps(
                {name: float(np.sqrt(max(covariance[index, index], 0.0))) for index, name in enumerate(COMPONENTS)}
            ),
            "uas_tier_profiles": json.dumps(uas_tier_profiles, sort_keys=True),
            "uas_charge_profiles": json.dumps(uas_charge_profiles, sort_keys=True),
            "cl61_state_profiles": json.dumps(cl61_state_profiles, sort_keys=True),
            "observation_frequency": OBSERVATION_FREQUENCY,
            "last_observation_time_utc": latest_observation_time.isoformat(),
            "new_observation_count": str(new_observation_count),
            "confirmed_state_observation_count": str(int(observations["direct_state_confirmed"].sum())),
        },
    )
    state_ds["OperatingModeCode"].attrs["mode_mapping"] = json.dumps(
        {str(mode_code(value)): mode_label(value) for value in mode_names}, sort_keys=True
    )
    state_ds["OperatingModeConfidence"].attrs["units"] = "1"
    state_ds["OperatingLoadState"].attrs["description"] = (
        "composed PDU mode, canonical UAS tier/charge state, and CL61 heater state"
    )
    state_ds["DirectStateConfirmed"].attrs.update(
        {
            "long_name": "complete four-outlet PDU state vector available",
            "flag_values": "0, 1",
        }
    )
    state_ds["UASEffectiveTier"].attrs["description"] = (
        "raw shared tier retained only when Dock 1 and Dock 2 exactly agree; "
        "Tier 11 and 12 remain visible here"
    )
    for name, dock in (("UASDock1Tier", 1), ("UASDock2Tier", 2)):
        state_ds[name].attrs.update(
            {
                "description": f"raw Menapia Dock {dock} tier",
                "units": "tier",
            }
        )
    state_ds["UASPairConsistent"].attrs.update(
        {
            "description": "Dock 1 and Dock 2 raw tier match exactly; only then single-tier learning is eligible",
            "flag_values": "0, 1",
        }
    )
    state_ds["UASPairState"].attrs["description"] = "raw dock-pair state used to exclude mixed pairs from single-tier learning"
    state_ds["UASCanonicalTier"].attrs.update(
        {
            "description": "canonical UAS operating tier after mapping 11 to 1 and 12 to 2",
            "units": "tier",
        }
    )
    state_ds["UASChargingState"].attrs.update(
        {
            "description": (
                "explicit UASCharge operator annotation; charging is never inferred from watts"
            ),
            "flag_values": "0, 1",
        }
    )
    state_ds["CL61StateCode"].attrs.update(
        {
            "description": "CL61 state classified from the learned low/high load regimes",
            "flag_values": "0, 1, 2",
            "flag_meanings": "off cl61 cl61_heater_on",
        }
    )
    for name in (
        "LearnedStateLoadP10Watts",
        "LearnedStateLoadP50Watts",
        "LearnedStateLoadP90Watts",
    ):
        state_ds[name].attrs["units"] = "W"
    state_ds["LearnedStateSampleCount"].attrs["units"] = "count"
    for name in ("ObservedLoadWatts", "EstimatedModeLoadWatts", "LoadInnovationWatts"):
        state_ds[name].attrs["units"] = "W"

    state = {
        "schema_version": STATE_SCHEMA_VERSION,
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "load_phase_schema_version": LOAD_PHASE_SCHEMA_VERSION,
        "updated_at_utc": _utc_now(),
        "current_mode": current_mode,
        "current_mode_confidence": current_confidence,
        "current_load_state": current_load_state,
        "current_uas_effective_tier": current_uas_effective_tier,
        "current_uas_canonical_tier": current_uas_tier,
        "current_uas_charging": current_uas_charging,
        "current_uas_state": current_uas_state,
        "current_uas_pair_state": current_uas_pair_state,
        "current_uas_pair_consistent": current_uas_pair_consistent,
        "current_cl61_state": current_cl61_state,
        "learned_power_state_catalog": catalog,
        "last_observation_time_utc": latest_observation_time.isoformat(),
        "new_observation_count": new_observation_count,
        "confirmed_state_observation_count": int(observations["direct_state_confirmed"].sum()),
        "learned_modes": list(learned_modes),
        "observed_modes": list(observed_modes),
        "mode_maturity": mode_maturity,
        "component_regimes": component_regimes,
        "mode_load_profiles": {
            name: profile.to_dict() for name, profile in mode_load_profiles.items()
        },
        "uas_tier_profiles": uas_tier_profiles,
        "uas_charge_profiles": uas_charge_profiles,
        "cl61_state_profiles": cl61_state_profiles,
        "components": {
            name: {
                "mean_w": float(mean[index]),
                "variance_w2": float(max(covariance[index, index], 0.0)),
                "observation_count": int(counts[index]),
            }
            for index, name in enumerate(COMPONENTS)
        },
        "covariance_w2": covariance.tolist(),
        "mode_sample_counts": {str(name): int(value) for name, value in mode_counts.items()},
    }
    return OperatingModelResult(
        state_dataset=state_ds,
        state=state,
        component_mean=mean,
        component_covariance=covariance,
        learned_modes=learned_modes,
        observed_modes=observed_modes,
        mode_maturity=mode_maturity,
        component_regimes=component_regimes,
        mode_load_profiles=mode_load_profiles,
        uas_tier_profiles=uas_tier_profiles,
        uas_charge_profiles=uas_charge_profiles,
        cl61_state_profiles=cl61_state_profiles,
        current_mode=current_mode,
        current_confidence=current_confidence,
        current_load_state=current_load_state,
        current_uas_effective_tier=current_uas_effective_tier,
        current_uas_tier=current_uas_tier,
        current_uas_charging=current_uas_charging,
        current_uas_state=current_uas_state,
        current_cl61_state=current_cl61_state,
    )


def _latest_soc(power: xr.Dataset) -> tuple[pd.Timestamp, float]:
    if "time" not in power or "BatterySOC" not in power or power.sizes.get("time", 0) == 0:
        raise ValueError("Power data do not contain BatterySOC")
    count = int(power.sizes["time"])
    starts = tuple(dict.fromkeys((max(count - 100_000, 0), 0)))
    for start_index in starts:
        view = power[["BatterySOC"]].isel(time=slice(start_index, None))
        values = np.asarray(view["BatterySOC"].values, dtype=np.float64)
        finite_indices = np.flatnonzero(np.isfinite(values))
        if finite_indices.size:
            index = int(finite_indices[-1])
            return pd.Timestamp(view["time"].values[index]), float(values[index])
    raise ValueError("Power data do not contain a finite BatterySOC sample")


def _align_ensemble_solar_contract(
    deterministic: xr.Dataset,
    ensemble: xr.Dataset | None,
) -> tuple[xr.Dataset | None, dict[str, str]]:
    """Preserve the native decision ensemble and describe reserve-tail provenance.

    The ensemble is the authoritative weather distribution for the actionable
    96-hour decision period. Recalibrating it with an older long-range planning
    cycle made the system and scenario cards disagree. The deterministic
    planning product is still used to extend the reserve tail after native
    ensemble coverage ends, and its separate contract is recorded explicitly.
    """
    target_contract = str(deterministic.attrs.get("solar_calibration_contract_id", ""))
    source_contract = str(ensemble.attrs.get("solar_calibration_contract_id", "")) if ensemble is not None else ""
    metadata = {
        "solar_ensemble_source_calibration_contract_id": source_contract,
        "solar_decision_calibration_contract_id": source_contract or target_contract,
        "solar_reserve_tail_calibration_contract_id": target_contract,
        "solar_ensemble_recalibrated": "false",
        "solar_contracts_differ": str(bool(source_contract and target_contract and source_contract != target_contract)).lower(),
    }
    return ensemble, metadata


def _hourly_solar_members(
    deterministic: xr.Dataset,
    ensemble: xr.Dataset | None,
    *,
    issue_time: pd.Timestamp,
    horizon_hours: int,
) -> tuple[pd.DatetimeIndex, np.ndarray, dict[str, str]]:
    # Every scenario is anchored to the physical SOC observation.  Flooring
    # this timestamp created forecast points before the observation and made
    # the scenario panel incomparable with the system-as-is ensemble.
    times = pd.DatetimeIndex(
        issue_time + pd.to_timedelta(np.arange(int(horizon_hours) + 1), unit="h")
    )
    if "ForecastSolarWatts" not in deterministic or "time" not in deterministic:
        raise ValueError("Deterministic forecast does not contain ForecastSolarWatts")
    deterministic_times = pd.DatetimeIndex(deterministic["time"].values)
    deterministic_series = pd.Series(
        np.asarray(deterministic["ForecastSolarWatts"].values, dtype=np.float64),
        index=deterministic_times,
    )
    deterministic_hourly = (
        deterministic_series.reindex(times.union(deterministic_times))
        .interpolate("time", limit_area="inside")
        .reindex(times)
        .ffill()
        .bfill()
        .clip(lower=0.0)
    )
    if ensemble is not None and "ForecastSolarWattsEnsemble" in ensemble and "time" in ensemble:
        source_times = pd.DatetimeIndex(ensemble["time"].values)
        rows = []
        native = np.asarray(
            ensemble["ForecastSolarWattsEnsemble"].transpose("member", "time").values,
            dtype=np.float64,
        )
        valid_native_times = source_times[np.any(np.isfinite(native), axis=0)] if native.size else pd.DatetimeIndex([])
        native_end = valid_native_times.max() if len(valid_native_times) else pd.NaT
        rank_factors = np.linspace(0.75, 1.25, max(native.shape[0], 1))
        deterministic_values = deterministic_hourly.to_numpy(dtype=np.float64)
        for member_index, values in enumerate(native):
            series = (
                pd.Series(values, index=source_times)
                .reindex(times.union(source_times))
                .interpolate("time", limit_area="inside")
                .reindex(times)
            )
            native_values = series.to_numpy(dtype=np.float64)
            ratio_mask = np.isfinite(native_values) & (deterministic_values >= 25.0)
            ratio = float(np.nanmedian(native_values[ratio_mask] / deterministic_values[ratio_mask])) if ratio_mask.any() else float(rank_factors[member_index])
            ratio = float(np.clip(ratio, 0.20, 1.80))
            fallback = deterministic_values * ratio
            if not pd.isna(native_end):
                lead = np.maximum((times - native_end) / pd.Timedelta(hours=1), 0.0).to_numpy(dtype=np.float64)
                widened_ratio = 1.0 + (ratio - 1.0) * (1.0 + np.minimum(lead / 96.0, 0.5))
                fallback = deterministic_values * np.clip(widened_ratio, 0.10, 2.00)
            combined = np.where(np.isfinite(native_values), native_values, fallback)
            rows.append(np.clip(combined, 0.0, None))
        if rows:
            return times, np.asarray(rows, dtype=np.float64), {
                "solar_member_source": "native_ensemble_with_deterministic_extension",
                "native_ensemble_end_time": "" if pd.isna(native_end) else pd.Timestamp(native_end).isoformat(),
                "uncertainty_extrapolated": str(bool(not pd.isna(native_end) and times[-1] > native_end)).lower(),
            }
    return times, deterministic_hourly.to_numpy(dtype=np.float64)[None, :], {
        "solar_member_source": "deterministic_only",
        "native_ensemble_end_time": "",
        "uncertainty_extrapolated": "false",
    }


def _current_system_load_members(
    deterministic: xr.Dataset,
    ensemble: xr.Dataset | None,
    times: pd.DatetimeIndex,
    member_count: int,
    *,
    fallback: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return system-as-is load and learned phase codes on ``times``."""
    source = ensemble if ensemble is not None and "ForecastLoadWattsEnsemble" in ensemble else None
    if source is not None and "time" in source:
        source_times = pd.DatetimeIndex(source["time"].values)
        values = np.asarray(
            source["ForecastLoadWattsEnsemble"].transpose("member", "time").values,
            dtype=np.float64,
        )
        rows = []
        for row in values:
            aligned = (
                pd.Series(row, index=source_times)
                .reindex(times.union(source_times))
                .interpolate("time", limit_area="inside")
                .reindex(times)
                .ffill()
                .bfill()
            )
            rows.append(aligned.to_numpy(dtype=np.float64))
        if rows:
            result = np.asarray(rows, dtype=np.float64)
            if result.shape[0] == member_count and np.isfinite(result).all():
                phases = np.full(result.shape, PHASE_CODES[PHASE_STEADY], dtype=np.int8)
                if "ForecastLoadPhaseCodeEnsemble" in source:
                    native_phases = np.asarray(
                        source["ForecastLoadPhaseCodeEnsemble"].transpose("member", "time").values,
                        dtype=np.float64,
                    )
                    if native_phases.shape[0] == member_count:
                        aligned_phases = []
                        for row in native_phases:
                            aligned = (
                                pd.Series(row, index=source_times)
                                .reindex(times.union(source_times))
                                .ffill()
                                .bfill()
                                .reindex(times)
                            )
                            aligned_phases.append(aligned.to_numpy(dtype=np.int8))
                        phases = np.asarray(aligned_phases, dtype=np.int8)
                return np.clip(result, 0.0, None), phases

    if "ForecastLoadWatts" in deterministic and "time" in deterministic:
        source_times = pd.DatetimeIndex(deterministic["time"].values)
        aligned = (
            pd.Series(np.asarray(deterministic["ForecastLoadWatts"].values, dtype=np.float64), index=source_times)
            .reindex(times.union(source_times))
            .interpolate("time", limit_area="inside")
            .reindex(times)
            .ffill()
            .bfill()
            .clip(lower=0.0)
        )
        phases = np.full((member_count, len(times)), PHASE_CODES[PHASE_STEADY], dtype=np.int8)
        if "ForecastLoadPhaseCode" in deterministic:
            aligned_phase = (
                pd.Series(
                    np.asarray(deterministic["ForecastLoadPhaseCode"].values, dtype=np.float64),
                    index=source_times,
                )
                .reindex(times.union(source_times))
                .ffill()
                .bfill()
                .reindex(times)
            )
            phases[:] = aligned_phase.to_numpy(dtype=np.int8)
        return np.tile(aligned.to_numpy(dtype=np.float64), (member_count, 1)), phases
    if fallback is not None:
        values = np.asarray(fallback, dtype=np.float64)
        if values.shape == (member_count, len(times)) and np.isfinite(values).all():
            return (
                np.clip(values, 0.0, None),
                np.full(values.shape, PHASE_CODES[PHASE_STEADY], dtype=np.int8),
            )
    raise ValueError("System forecast does not contain ForecastLoadWatts")


def _battery_member_parameters(
    ensemble: xr.Dataset | None,
    model: BatteryModel,
    member_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fields = (
        ("BatteryUsableCapacityKWhEnsemble", model.usable_capacity_kwh),
        ("BatteryChargeEfficiencyEnsemble", model.charge_efficiency),
        ("BatteryDischargeEfficiencyEnsemble", model.discharge_efficiency),
    )
    values: list[np.ndarray] = []
    for name, fallback in fields:
        if ensemble is not None and name in ensemble and ensemble[name].dims == ("member",):
            candidate = np.asarray(ensemble[name].values, dtype=np.float64)
            if candidate.shape == (member_count,) and np.isfinite(candidate).all():
                values.append(candidate)
                continue
        values.append(np.full(member_count, float(fallback), dtype=np.float64))
    return values[0], values[1], values[2]


def _component_members(
    mean: np.ndarray,
    covariance: np.ndarray,
    count: int,
    *,
    seed: int,
    regimes: Mapping[str, Sequence[Mapping[str, float]]] | None = None,
) -> np.ndarray:
    count = max(int(count), 1)
    if count == 1:
        return mean[None, :]
    safe_covariance = np.asarray(covariance, dtype=np.float64).copy()
    safe_covariance = (safe_covariance + safe_covariance.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(safe_covariance)
    safe_covariance = eigenvectors @ np.diag(np.clip(eigenvalues, 1e-6, None)) @ eigenvectors.T
    rng = np.random.default_rng(seed)
    values = rng.multivariate_normal(mean, safe_covariance, size=count)
    for name, index in COMPONENT_INDEX.items():
        entries = list((regimes or {}).get(name, ()))
        if not entries:
            continue
        weights = np.asarray([float(entry["weight"]) for entry in entries], dtype=np.float64)
        weights /= weights.sum()
        choices = rng.choice(len(entries), size=count, p=weights)
        values[:, index] = np.asarray(
            [rng.normal(float(entries[choice]["mean_w"]), max(float(entries[choice]["std_w"]), 3.0)) for choice in choices],
            dtype=np.float64,
        )
    return np.clip(values, 0.0, None)


def _load_members_for_modes(component_members: np.ndarray, modes: Sequence[str]) -> np.ndarray:
    components = np.asarray(component_members, dtype=np.float64)
    design = np.asarray([_mode_design(value) for value in modes], dtype=np.float64)
    if components.ndim != 2 or components.shape[1] != len(COMPONENTS):
        raise ValueError("Component members must be a member x component array")
    if not np.isfinite(components).all() or not np.isfinite(design).all():
        raise ValueError("Operating-mode load inputs must be finite")
    # The explicit contraction avoids spurious Accelerate/BLAS warnings seen
    # with a transposed, non-contiguous design matrix on macOS.
    loads = np.einsum("mc,tc->mt", components, design, optimize=False)
    return np.clip(loads, 0.0, None)


def _member_soc_delta_percent(
    net_power_w: np.ndarray,
    hours: float,
    model: BatteryModel,
    capacities_kwh: np.ndarray,
    charge_efficiencies: np.ndarray,
    discharge_efficiencies: np.ndarray,
) -> np.ndarray:
    net_w = np.asarray(net_power_w, dtype=np.float64) - model.parasitic_load_w
    charging_w = np.minimum(np.clip(net_w, 0.0, None), model.max_charge_w)
    discharging_w = np.minimum(np.clip(-net_w, 0.0, None), model.max_discharge_w)
    stored_w = charging_w * charge_efficiencies - discharging_w / discharge_efficiencies
    return 100.0 * stored_w * max(float(hours), 0.0) / (1000.0 * capacities_kwh)


def integrate_soc_members(
    *,
    initial_soc: float,
    times: pd.DatetimeIndex,
    solar_members_w: np.ndarray,
    load_members_w: np.ndarray,
    capacity_kwh: float,
    battery_model: BatteryModel | None = None,
    member_capacity_kwh: np.ndarray | None = None,
    member_charge_efficiency: np.ndarray | None = None,
    member_discharge_efficiency: np.ndarray | None = None,
) -> np.ndarray:
    solar = np.asarray(solar_members_w, dtype=np.float64)
    load = np.asarray(load_members_w, dtype=np.float64)
    if solar.ndim != 2 or load.ndim != 2 or solar.shape != load.shape:
        raise ValueError("Solar and load members must be matching member x time arrays")
    model = battery_model or BatteryModel(
        usable_capacity_kwh=capacity_kwh,
        charge_efficiency=1.0,
        discharge_efficiency=1.0,
        max_charge_w=20_000.0,
        max_discharge_w=20_000.0,
    )
    member_count = solar.shape[0]
    capacities = np.asarray(member_capacity_kwh, dtype=np.float64) if member_capacity_kwh is not None else np.full(member_count, model.usable_capacity_kwh)
    charge_efficiencies = np.asarray(member_charge_efficiency, dtype=np.float64) if member_charge_efficiency is not None else np.full(member_count, model.charge_efficiency)
    discharge_efficiencies = np.asarray(member_discharge_efficiency, dtype=np.float64) if member_discharge_efficiency is not None else np.full(member_count, model.discharge_efficiency)
    if capacities.shape != (member_count,) or charge_efficiencies.shape != (member_count,) or discharge_efficiencies.shape != (member_count,):
        raise ValueError("Battery member parameters must match the forecast member dimension")
    capacities = np.clip(capacities, 10.0, 40.0)
    charge_efficiencies = np.clip(charge_efficiencies, 0.65, 1.0)
    discharge_efficiencies = np.clip(discharge_efficiencies, 0.65, 1.0)
    soc = np.full(solar.shape, np.nan, dtype=np.float64)
    soc[:, 0] = float(np.clip(initial_soc, 0.0, 100.0))
    for index in range(1, len(times)):
        hours = max(float((times[index] - times[index - 1]) / pd.Timedelta(hours=1)), 0.0)
        delta_soc = _member_soc_delta_percent(
            solar[:, index] - load[:, index],
            hours,
            model,
            capacities,
            charge_efficiencies,
            discharge_efficiencies,
        )
        soc[:, index] = np.clip(soc[:, index - 1] + delta_soc, 0.0, 100.0)
    return soc


@dataclass
class ScheduleResult:
    modes: tuple[str, ...]
    collection_hours: float
    start_time: pd.Timestamp | None
    stop_time: pd.Timestamp | None
    minimum_p10_soc: float
    final_p10_soc: float
    safe: bool
    starts: int
    instrument_hours: dict[str, float] = field(default_factory=dict)
    total_instrument_hours: float = 0.0
    controlled_energy_kwh: float = 0.0


@dataclass(frozen=True)
class P50ContinuationDecision:
    """Describe a forecast-only rule for equipment that is already on."""

    eligible: bool
    status: str
    reason_code: str
    reason: str
    held_instruments: tuple[str, ...]
    recovery_time: pd.Timestamp | None
    hold_through_index: int | None
    minimum_p50_soc_before_recovery: float


def evaluate_p50_continuation_rule(
    *,
    times: pd.DatetimeIndex,
    soc_p50: Sequence[float],
    current_mode: str,
    controlled_instruments: Sequence[str] = OPERATING_PRIORITY,
    horizon_hours: int = 96,
    recovery_soc: float = P50_CONTINUATION_RECOVERY_SOC_PCT,
    minimum_soc: float = P50_CONTINUATION_MINIMUM_SOC_PCT,
) -> P50ContinuationDecision:
    """Allow currently-on instruments to bridge a safe median-SOC recovery.

    This is a recommendation rule only. It never starts an instrument and has
    no control path to a PDU. The conservative P10 schedule remains a separate
    scenario and is used whenever this median rule is not eligible.
    """
    forecast_times = pd.DatetimeIndex(times)
    values = np.asarray(soc_p50, dtype=np.float64).reshape(-1)
    if len(forecast_times) != len(values):
        raise ValueError("P50 continuation times and SOC values must have equal length")

    held = tuple(
        str(instrument)
        for instrument in controlled_instruments
        if str(instrument) in mode_kits(current_mode)
    )
    count = min(len(forecast_times), max(int(horizon_hours), 0) + 1)
    unavailable = P50ContinuationDecision(
        eligible=False,
        status="not_applicable" if not held else "unavailable",
        reason_code="no_currently_on_controlled_instruments" if not held else "insufficient_forecast",
        reason=(
            "No controlled instrument is currently on, so there is nothing for the P50 continuation rule to hold."
            if not held
            else "The P50 continuation rule needs a complete future median-SOC forecast."
        ),
        held_instruments=held,
        recovery_time=None,
        hold_through_index=None,
        minimum_p50_soc_before_recovery=float("nan"),
    )
    if not held or count < 2:
        return unavailable
    decision_values = values[:count]
    if not np.isfinite(decision_values).all():
        return unavailable

    below_target = np.flatnonzero(decision_values[1:] < float(recovery_soc) - 1e-9) + 1
    if decision_values[0] >= float(recovery_soc) and below_target.size == 0:
        recovery_index = count - 1
        reason_code = "p50_stays_at_recovery_target"
    else:
        search_start = int(below_target[0] + 1) if decision_values[0] >= float(recovery_soc) else 1
        recovery_candidates = np.flatnonzero(
            decision_values[search_start:] >= float(recovery_soc) - 1e-9
        )
        if recovery_candidates.size == 0:
            return P50ContinuationDecision(
                eligible=False,
                status="not_eligible",
                reason_code="p50_does_not_recover",
                reason=(
                    f"P50 SOC does not recover to {float(recovery_soc):g}% within the "
                    f"{int(horizon_hours)}-hour decision horizon, so the conservative P10 plan remains in use."
                ),
                held_instruments=held,
                recovery_time=None,
                hold_through_index=None,
                minimum_p50_soc_before_recovery=float(np.nanmin(decision_values)),
            )
        recovery_index = search_start + int(recovery_candidates[0])
        reason_code = "safe_p50_recovery"

    minimum_before_recovery = float(np.nanmin(decision_values[: recovery_index + 1]))
    recovery_time = pd.Timestamp(forecast_times[recovery_index])
    if minimum_before_recovery < float(minimum_soc) - 1e-9:
        return P50ContinuationDecision(
            eligible=False,
            status="not_eligible",
            reason_code="p50_breaches_minimum_before_recovery",
            reason=(
                f"P50 SOC falls to {minimum_before_recovery:.1f}% before recovering to "
                f"{float(recovery_soc):g}%, below the {float(minimum_soc):g}% floor. "
                "The conservative P10 plan remains in use."
            ),
            held_instruments=held,
            recovery_time=recovery_time,
            hold_through_index=None,
            minimum_p50_soc_before_recovery=minimum_before_recovery,
        )

    held_text = _human_list(held)
    return P50ContinuationDecision(
        eligible=True,
        status="eligible",
        reason_code=reason_code,
        reason=(
            f"The advisory P50 continuation scenario keeps {held_text} on through "
            f"{recovery_time.isoformat()} because median SOC stays at or above "
            f"{float(minimum_soc):g}% and reaches {float(recovery_soc):g}%. "
            "It does not operate any PDU outlet."
        ),
        held_instruments=held,
        recovery_time=recovery_time,
        hold_through_index=recovery_index,
        minimum_p50_soc_before_recovery=minimum_before_recovery,
    )


def apply_p50_continuation_rule(
    modes: Sequence[str],
    decision: P50ContinuationDecision,
) -> tuple[str, ...]:
    """Overlay an eligible P50 hold on a copy of the P10 advisory modes."""
    result = [str(value) for value in modes]
    if not decision.eligible or decision.hold_through_index is None:
        return tuple(result)
    last = min(int(decision.hold_through_index), len(result) - 1)
    for index in range(last + 1):
        result[index] = mode_id(set(mode_kits(result[index])) | set(decision.held_instruments))
    return tuple(result)


@dataclass
class _Candidate:
    modes: tuple[str, ...]
    soc: np.ndarray
    on: bool
    run_hours: float
    last_start_day: object | None
    starts: int
    on_hours: float
    minimum_p10: float
    first_start_index: int | None


def _candidate_score(candidate: _Candidate) -> tuple[float, float, float, float]:
    first = candidate.first_start_index if candidate.first_start_index is not None else 10**6
    return (float(candidate.on_hours), float(-candidate.starts), float(candidate.minimum_p10), float(-first))


@dataclass
class _AdditiveCandidate:
    """One phase-aware joint schedule candidate for all controlled instruments."""

    modes: tuple[str, ...]
    soc: np.ndarray
    active: tuple[bool, ...]
    run_hours: tuple[float, ...]
    last_start_days: tuple[object | None, ...]
    starts: tuple[int, ...]
    instrument_hours: tuple[float, ...]
    controlled_energy_wh: float
    minimum_p10: float
    first_active_index: int | None
    segment_start_index: int


def _additive_candidate_score(candidate: _AdditiveCandidate) -> tuple[float, ...]:
    """Rank by additive power use, total use, then declared instrument priority."""
    first = candidate.first_active_index if candidate.first_active_index is not None else 10**6
    return (
        float(candidate.controlled_energy_wh),
        float(sum(candidate.instrument_hours)),
        *(float(value) for value in candidate.instrument_hours),
        float(-sum(candidate.starts)),
        float(candidate.minimum_p10),
        float(np.nanmedian(candidate.soc)),
        float(-first),
    )


def optimize_cl61_schedule(
    *,
    times: pd.DatetimeIndex,
    solar_members_w: np.ndarray,
    component_members: np.ndarray,
    initial_soc: float,
    capacity_kwh: float,
    battery_model: BatteryModel | None = None,
    member_capacity_kwh: np.ndarray | None = None,
    member_charge_efficiency: np.ndarray | None = None,
    member_discharge_efficiency: np.ndarray | None = None,
    base_mode: str,
    horizon_hours: int = 96,
    minimum_soc: float = MINIMUM_OPERATIONAL_SOC_PCT,
    minimum_run_hours: int = MIN_RUN_HOURS,
    max_starts_per_day: int = MAX_STARTS_PER_UTC_DAY,
    beam_width: int = 300,
) -> ScheduleResult:
    full_times = pd.DatetimeIndex(times)
    full_solar = np.asarray(solar_members_w, dtype=np.float64)
    if len(full_times) == 0:
        raise ValueError("CL61 optimization requires at least one forecast time")
    if full_solar.ndim != 2 or full_solar.shape[1] < len(full_times):
        raise ValueError("Solar members must be a member x time array covering the forecast")
    full_solar = full_solar[:, : len(full_times)]
    energy_model = battery_model or BatteryModel(
        usable_capacity_kwh=capacity_kwh,
        charge_efficiency=1.0,
        discharge_efficiency=1.0,
        max_charge_w=20_000.0,
        max_discharge_w=20_000.0,
    )
    member_count = full_solar.shape[0]
    capacities = (
        np.asarray(member_capacity_kwh, dtype=np.float64)
        if member_capacity_kwh is not None
        else np.full(member_count, energy_model.usable_capacity_kwh)
    )
    charge_efficiencies = (
        np.asarray(member_charge_efficiency, dtype=np.float64)
        if member_charge_efficiency is not None
        else np.full(member_count, energy_model.charge_efficiency)
    )
    discharge_efficiencies = (
        np.asarray(member_discharge_efficiency, dtype=np.float64)
        if member_discharge_efficiency is not None
        else np.full(member_count, energy_model.discharge_efficiency)
    )
    if any(values.shape != (member_count,) for values in (capacities, charge_efficiencies, discharge_efficiencies)):
        raise ValueError("Battery member parameters must match the solar member dimension")
    capacities = np.clip(capacities, 10.0, 40.0)
    charge_efficiencies = np.clip(charge_efficiencies, 0.65, 1.0)
    discharge_efficiencies = np.clip(discharge_efficiencies, 0.65, 1.0)
    decision_count = min(len(full_times), int(horizon_hours) + 1)
    decision_times = full_times[:decision_count]
    base_kits = set(mode_kits(base_mode))
    base_kits.discard("CL61")
    off_mode = mode_id(base_kits)
    on_mode = mode_id(base_kits | {"CL61"})
    off_load = _load_members_for_modes(component_members, [off_mode])[:, 0]
    on_load = _load_members_for_modes(component_members, [on_mode])[:, 0]
    initial_members = np.full(full_solar.shape[0], float(initial_soc), dtype=np.float64)
    candidates = [
        _Candidate(
            modes=(off_mode,),
            soc=initial_members,
            on=False,
            run_hours=0,
            last_start_day=None,
            starts=0,
            on_hours=0,
            minimum_p10=float(initial_soc),
            first_start_index=None,
        )
    ]
    search_complete = decision_count == 1
    for index in range(1, decision_count):
        next_candidates: list[_Candidate] = []
        day = decision_times[index].date()
        for candidate in candidates:
            choices = (True,) if candidate.on and candidate.run_hours < minimum_run_hours else (False, True)
            for turn_on in choices:
                is_start = bool(turn_on and not candidate.on)
                if is_start and candidate.last_start_day == day:
                    continue
                if is_start and max_starts_per_day <= 0:
                    continue
                load = on_load if turn_on else off_load
                hours = max(
                    float((decision_times[index] - decision_times[index - 1]) / pd.Timedelta(hours=1)),
                    0.0,
                )
                soc = np.clip(
                    candidate.soc
                    + _member_soc_delta_percent(
                        full_solar[:, index] - load,
                        hours,
                        energy_model,
                        capacities,
                        charge_efficiencies,
                        discharge_efficiencies,
                    ),
                    0.0,
                    100.0,
                )
                p10 = float(np.nanquantile(soc, 0.10))
                if p10 < float(minimum_soc) - 1e-9:
                    continue
                next_candidates.append(
                    _Candidate(
                        modes=candidate.modes + (on_mode if turn_on else off_mode,),
                        soc=soc,
                        on=turn_on,
                        run_hours=candidate.run_hours + 1 if turn_on else 0,
                        last_start_day=day if is_start else candidate.last_start_day,
                        starts=candidate.starts + int(is_start),
                        on_hours=candidate.on_hours + int(turn_on),
                        minimum_p10=min(candidate.minimum_p10, p10),
                        first_start_index=index if is_start and candidate.first_start_index is None else candidate.first_start_index,
                    )
                )
        if not next_candidates:
            candidates = []
            break
        grouped: dict[tuple[bool, int, object | None, int], list[_Candidate]] = {}
        for candidate in next_candidates:
            key = (
                candidate.on,
                min(candidate.run_hours, minimum_run_hours),
                candidate.last_start_day,
                int(np.nanmedian(candidate.soc) // 2),
            )
            grouped.setdefault(key, []).append(candidate)
        reduced = [max(values, key=_candidate_score) for values in grouped.values()]
        ranked = sorted(reduced, key=_candidate_score, reverse=True)
        baseline = next(
            (candidate for candidate in next_candidates if candidate.starts == 0 and candidate.on_hours == 0),
            None,
        )
        width = max(int(beam_width), 1)
        candidates = ranked[:width]
        if baseline is not None and not any(value.starts == 0 and value.on_hours == 0 for value in candidates):
            candidates = ranked[: width - 1] + [baseline]
        search_complete = index == decision_count - 1

    valid = [] if not search_complete else [
        candidate for candidate in candidates if not candidate.on or candidate.run_hours >= minimum_run_hours
    ]

    def extend_candidate(candidate: _Candidate) -> tuple[_Candidate, tuple[str, ...], float, float]:
        modes = list(candidate.modes)
        soc = candidate.soc.copy()
        minimum_p10 = float(candidate.minimum_p10)
        for index in range(decision_count, len(full_times)):
            hours = max(float((full_times[index] - full_times[index - 1]) / pd.Timedelta(hours=1)), 0.0)
            soc = np.clip(
                soc
                + _member_soc_delta_percent(
                    full_solar[:, index] - off_load,
                    hours,
                    energy_model,
                    capacities,
                    charge_efficiencies,
                    discharge_efficiencies,
                ),
                0.0,
                100.0,
            )
            minimum_p10 = min(minimum_p10, float(np.nanquantile(soc, 0.10)))
            modes.append(off_mode)
        return candidate, tuple(modes), minimum_p10, float(np.nanquantile(soc, 0.10))

    evaluated = [extend_candidate(candidate) for candidate in valid]
    safe_evaluated = [value for value in evaluated if value[2] >= float(minimum_soc) - 1e-9]
    if safe_evaluated:
        best, modes, minimum_p10, final_p10 = max(
            safe_evaluated,
            key=lambda value: (
                float(value[0].on_hours),
                float(-value[0].starts),
                float(value[2]),
                float(-(value[0].first_start_index if value[0].first_start_index is not None else 10**6)),
            ),
        )
    else:
        modes = tuple(off_mode for _ in full_times)
        off_loads = np.repeat(off_load[:, np.newaxis], len(full_times), axis=1)
        off_soc = integrate_soc_members(
            initial_soc=initial_soc,
            times=full_times,
            solar_members_w=full_solar,
            load_members_w=off_loads,
            capacity_kwh=capacity_kwh,
            battery_model=battery_model,
            member_capacity_kwh=capacities,
            member_charge_efficiency=charge_efficiencies,
            member_discharge_efficiency=discharge_efficiencies,
        )
        off_p10 = np.nanquantile(off_soc, 0.10, axis=0)
        minimum_p10 = float(np.nanmin(off_p10))
        final_p10 = float(off_p10[-1])
        best = _Candidate(
            modes=modes[:decision_count],
            soc=off_soc[:, decision_count - 1],
            on=False,
            run_hours=0,
            last_start_day=None,
            starts=0,
            on_hours=0,
            minimum_p10=minimum_p10,
            first_start_index=None,
        )

    on_indices = [index for index, value in enumerate(modes) if "CL61" in mode_kits(value)]
    start_time = full_times[on_indices[0]] if on_indices else None
    stop_time = None
    if on_indices and on_indices[-1] + 1 < len(full_times):
        stop_time = full_times[on_indices[-1] + 1]
    return ScheduleResult(
        modes=modes,
        collection_hours=float(best.on_hours),
        start_time=start_time,
        stop_time=stop_time,
        minimum_p10_soc=float(minimum_p10),
        final_p10_soc=float(final_p10),
        safe=bool(minimum_p10 >= minimum_soc),
        starts=int(best.starts),
    )


def _mode_with_kit(mode: str, kit: str, active: bool) -> str:
    kits = set(mode_kits(mode))
    kits.discard(kit)
    if active:
        kits.add(kit)
    return mode_id(kits)


def _instrument_hours(
    times: pd.DatetimeIndex,
    modes: Sequence[str],
    kit: str,
    *,
    limit: int | None = None,
) -> float:
    count = min(len(times), len(modes), limit if limit is not None else len(times))
    return float(
        sum(
            max(float((times[index] - times[index - 1]) / pd.Timedelta(hours=1)), 0.0)
            for index in range(1, count)
            if kit in mode_kits(modes[index])
        )
    )


def _instrument_starts(modes: Sequence[str], kit: str, *, limit: int | None = None) -> int:
    count = min(len(modes), limit if limit is not None else len(modes))
    return int(
        sum(
            kit in mode_kits(modes[index]) and kit not in mode_kits(modes[index - 1])
            for index in range(1, count)
        )
    )


def _daily_operating_summary(
    times: pd.DatetimeIndex,
    modes: Sequence[str],
    *,
    kits: Sequence[str],
    component_power_w: Mapping[str, float],
    limit: int,
) -> dict[str, dict[str, float]]:
    """Integrate binary schedules into auditable UTC-day operating totals."""
    count = min(len(times), len(modes), max(int(limit), 0))
    summary: dict[str, dict[str, float]] = {}
    for index in range(1, count):
        interval_start = pd.Timestamp(times[index - 1])
        interval_end = pd.Timestamp(times[index])
        if interval_end <= interval_start:
            continue
        active = tuple(kit for kit in kits if kit in mode_kits(modes[index]))
        cursor = interval_start
        while cursor < interval_end:
            next_day = cursor.normalize() + pd.Timedelta(days=1)
            stop = min(interval_end, next_day)
            hours = float((stop - cursor) / pd.Timedelta(hours=1))
            day = cursor.date().isoformat()
            row = summary.setdefault(
                day,
                {
                    **{kit: 0.0 for kit in kits},
                    "available_clock_hours": 0.0,
                    "active_clock_hours": 0.0,
                    "total_instrument_hours": 0.0,
                    "controlled_energy_kwh": 0.0,
                },
            )
            row["available_clock_hours"] += hours
            if active:
                row["active_clock_hours"] += hours
            for kit in active:
                row[kit] += hours
            row["total_instrument_hours"] += hours * len(active)
            row["controlled_energy_kwh"] += (
                hours * sum(max(float(component_power_w.get(kit, 0.0)), 0.0) for kit in active) / 1000.0
            )
            cursor = stop
    for row in summary.values():
        row["idle_clock_hours"] = max(
            row["available_clock_hours"] - row["active_clock_hours"],
            0.0,
        )
        for kit in kits:
            row[f"{kit}_on_hours"] = row[kit]
            row[f"{kit}_off_hours"] = max(
                row["available_clock_hours"] - row[kit],
                0.0,
            )
        for key, value in tuple(row.items()):
            row[key] = round(float(value), 6)
    return summary


def _optimize_one_instrument_schedule(
    *,
    times: pd.DatetimeIndex,
    solar_members_w: np.ndarray,
    component_members: np.ndarray,
    initial_soc: float,
    capacity_kwh: float,
    base_modes: Sequence[str],
    kit: str,
    initially_on: bool,
    battery_model: BatteryModel | None = None,
    member_capacity_kwh: np.ndarray | None = None,
    member_charge_efficiency: np.ndarray | None = None,
    member_discharge_efficiency: np.ndarray | None = None,
    horizon_hours: int = 96,
    minimum_soc: float = MINIMUM_OPERATIONAL_SOC_PCT,
    minimum_run_hours: int = MIN_RUN_HOURS,
    max_starts_per_day: int = MAX_STARTS_PER_UTC_DAY,
    beam_width: int = 300,
) -> ScheduleResult:
    """Add one instrument to an already-safe higher-priority timetable."""
    if kit not in KIT_ORDER:
        raise ValueError(f"Unknown instrument: {kit}")
    if max_starts_per_day not in {0, 1}:
        raise ValueError("The advisory scheduler supports at most one start per UTC day")

    full_times = pd.DatetimeIndex(times)
    full_solar = np.asarray(solar_members_w, dtype=np.float64)
    baseline = tuple(str(value) for value in base_modes)
    if len(full_times) == 0:
        raise ValueError("Instrument optimization requires at least one forecast time")
    if len(baseline) != len(full_times):
        raise ValueError("The higher-priority timetable must cover the full forecast")
    if full_solar.ndim != 2 or full_solar.shape[1] < len(full_times):
        raise ValueError("Solar members must be a member x time array covering the forecast")
    full_solar = full_solar[:, : len(full_times)]

    energy_model = battery_model or BatteryModel(
        usable_capacity_kwh=capacity_kwh,
        charge_efficiency=1.0,
        discharge_efficiency=1.0,
        max_charge_w=20_000.0,
        max_discharge_w=20_000.0,
    )
    member_count = full_solar.shape[0]
    capacities = (
        np.asarray(member_capacity_kwh, dtype=np.float64)
        if member_capacity_kwh is not None
        else np.full(member_count, energy_model.usable_capacity_kwh)
    )
    charge_efficiencies = (
        np.asarray(member_charge_efficiency, dtype=np.float64)
        if member_charge_efficiency is not None
        else np.full(member_count, energy_model.charge_efficiency)
    )
    discharge_efficiencies = (
        np.asarray(member_discharge_efficiency, dtype=np.float64)
        if member_discharge_efficiency is not None
        else np.full(member_count, energy_model.discharge_efficiency)
    )
    if any(
        values.shape != (member_count,)
        for values in (capacities, charge_efficiencies, discharge_efficiencies)
    ):
        raise ValueError("Battery member parameters must match the solar member dimension")
    capacities = np.clip(capacities, 10.0, 40.0)
    charge_efficiencies = np.clip(charge_efficiencies, 0.65, 1.0)
    discharge_efficiencies = np.clip(discharge_efficiencies, 0.65, 1.0)

    off_modes = tuple(_mode_with_kit(mode, kit, False) for mode in baseline)
    on_modes = tuple(_mode_with_kit(mode, kit, True) for mode in baseline)
    off_loads = _load_members_for_modes(component_members, off_modes)
    on_loads = _load_members_for_modes(component_members, on_modes)
    decision_count = min(len(full_times), int(horizon_hours) + 1)
    decision_times = full_times[:decision_count]
    initial_members = np.full(member_count, float(initial_soc), dtype=np.float64)
    candidates = [
        _Candidate(
            modes=((on_modes[0] if initially_on else off_modes[0]),),
            soc=initial_members,
            on=bool(initially_on),
            # A state already active at issue time was not started by this
            # advisory plan, so it may be shed at the first decision boundary.
            run_hours=minimum_run_hours if initially_on else 0,
            last_start_day=None,
            starts=0,
            on_hours=0,
            minimum_p10=float(initial_soc),
            first_start_index=None,
        )
    ]
    search_complete = decision_count == 1
    for index in range(1, decision_count):
        next_candidates: list[_Candidate] = []
        day = decision_times[index].date()
        step_hours = max(
            float((decision_times[index] - decision_times[index - 1]) / pd.Timedelta(hours=1)),
            0.0,
        )
        for candidate in candidates:
            choices = (True,) if candidate.on and candidate.run_hours < minimum_run_hours else (False, True)
            for turn_on in choices:
                is_start = bool(turn_on and not candidate.on)
                if is_start and (
                    max_starts_per_day <= 0
                    or candidate.last_start_day == day
                ):
                    continue
                load = on_loads[:, index] if turn_on else off_loads[:, index]
                soc = np.clip(
                    candidate.soc
                    + _member_soc_delta_percent(
                        full_solar[:, index] - load,
                        step_hours,
                        energy_model,
                        capacities,
                        charge_efficiencies,
                        discharge_efficiencies,
                    ),
                    0.0,
                    100.0,
                )
                p10 = float(np.nanquantile(soc, 0.10))
                if p10 < float(minimum_soc) - 1e-9:
                    continue
                next_candidates.append(
                    _Candidate(
                        modes=candidate.modes + ((on_modes[index] if turn_on else off_modes[index]),),
                        soc=soc,
                        on=turn_on,
                        run_hours=(candidate.run_hours + step_hours) if turn_on else 0,
                        last_start_day=day if is_start else candidate.last_start_day,
                        starts=candidate.starts + int(is_start),
                        on_hours=candidate.on_hours + (step_hours if turn_on else 0),
                        minimum_p10=min(candidate.minimum_p10, p10),
                        first_start_index=(
                            index
                            if is_start and candidate.first_start_index is None
                            else candidate.first_start_index
                        ),
                    )
                )
        if not next_candidates:
            candidates = []
            break
        grouped: dict[tuple[bool, int, object | None, int], list[_Candidate]] = {}
        for candidate in next_candidates:
            key = (
                candidate.on,
                min(int(candidate.run_hours), minimum_run_hours),
                candidate.last_start_day,
                int(np.nanmedian(candidate.soc) // 2),
            )
            grouped.setdefault(key, []).append(candidate)
        ranked = sorted(
            (max(values, key=_candidate_score) for values in grouped.values()),
            key=_candidate_score,
            reverse=True,
        )
        width = max(int(beam_width), 1)
        candidates = ranked[:width]
        baseline_candidate = next(
            (
                value
                for value in next_candidates
                if not value.on and value.starts == 0 and value.on_hours == 0
            ),
            None,
        )
        if baseline_candidate is not None and all(
            value is not baseline_candidate for value in candidates
        ):
            candidates = candidates[: max(width - 1, 0)] + [baseline_candidate]
        search_complete = index == decision_count - 1

    valid = [] if not search_complete else [
        candidate
        for candidate in candidates
        if not candidate.on or candidate.run_hours >= minimum_run_hours
    ]

    def extend_candidate(candidate: _Candidate) -> tuple[_Candidate, tuple[str, ...], float, float]:
        modes = list(candidate.modes)
        soc = candidate.soc.copy()
        minimum_p10 = float(candidate.minimum_p10)
        for index in range(decision_count, len(full_times)):
            step_hours = max(
                float((full_times[index] - full_times[index - 1]) / pd.Timedelta(hours=1)),
                0.0,
            )
            soc = np.clip(
                soc
                + _member_soc_delta_percent(
                    full_solar[:, index] - off_loads[:, index],
                    step_hours,
                    energy_model,
                    capacities,
                    charge_efficiencies,
                    discharge_efficiencies,
                ),
                0.0,
                100.0,
            )
            minimum_p10 = min(minimum_p10, float(np.nanquantile(soc, 0.10)))
            modes.append(off_modes[index])
        return candidate, tuple(modes), minimum_p10, float(np.nanquantile(soc, 0.10))

    evaluated = [extend_candidate(candidate) for candidate in valid]
    safe_evaluated = [value for value in evaluated if value[2] >= float(minimum_soc) - 1e-9]
    if safe_evaluated:
        best, modes, minimum_p10, final_p10 = max(
            safe_evaluated,
            key=lambda value: (
                float(value[0].on_hours),
                float(-value[0].starts),
                float(value[2]),
                float(-(value[0].first_start_index if value[0].first_start_index is not None else 10**6)),
            ),
        )
    else:
        modes = off_modes
        off_soc = integrate_soc_members(
            initial_soc=initial_soc,
            times=full_times,
            solar_members_w=full_solar,
            load_members_w=off_loads,
            capacity_kwh=capacity_kwh,
            battery_model=battery_model,
            member_capacity_kwh=capacities,
            member_charge_efficiency=charge_efficiencies,
            member_discharge_efficiency=discharge_efficiencies,
        )
        off_p10 = np.nanquantile(off_soc, 0.10, axis=0)
        minimum_p10 = float(np.nanmin(off_p10))
        final_p10 = float(off_p10[-1])
        best = _Candidate(
            modes=modes[:decision_count],
            soc=off_soc[:, decision_count - 1],
            on=False,
            run_hours=0,
            last_start_day=None,
            starts=0,
            on_hours=0,
            minimum_p10=minimum_p10,
            first_start_index=None,
        )

    on_indices = [index for index, value in enumerate(modes) if kit in mode_kits(value)]
    start_time = full_times[on_indices[0]] if on_indices else None
    stop_time = None
    if on_indices and on_indices[-1] + 1 < len(full_times):
        stop_time = full_times[on_indices[-1] + 1]
    return ScheduleResult(
        modes=modes,
        collection_hours=float(best.on_hours),
        start_time=start_time,
        stop_time=stop_time,
        minimum_p10_soc=float(minimum_p10),
        final_p10_soc=float(final_p10),
        safe=bool(minimum_p10 >= minimum_soc),
        starts=int(best.starts),
    )


def optimize_priority_schedule(
    *,
    times: pd.DatetimeIndex,
    solar_members_w: np.ndarray,
    component_members: np.ndarray,
    initial_soc: float,
    capacity_kwh: float,
    base_mode: str,
    priorities: Sequence[str] = OPERATING_PRIORITY,
    battery_model: BatteryModel | None = None,
    member_capacity_kwh: np.ndarray | None = None,
    member_charge_efficiency: np.ndarray | None = None,
    member_discharge_efficiency: np.ndarray | None = None,
    horizon_hours: int = 96,
    minimum_soc: float = MINIMUM_OPERATIONAL_SOC_PCT,
    minimum_run_hours: int = MIN_RUN_HOURS,
    max_starts_per_day: int = MAX_STARTS_PER_UTC_DAY,
    beam_width: int = 600,
    mode_load_profiles: Mapping[str, StateLoadDynamics] | None = None,
    current_mode: str | None = None,
    required_active: Mapping[str, Sequence[bool]] | None = None,
    seed: int = 0,
) -> ScheduleResult:
    """Jointly maximise safe additive controlled energy across all instruments.

    The primary objective is additive component energy, followed by total
    instrument-hours. ``priorities`` is a deterministic tie-break, not a set of
    sequential budgets. Learned exact-state startup/fan phases are included in
    the search itself, so a lower-priority addition can be rejected without
    erasing a safe higher-value subset. ``required_active`` pins named
    instruments to a supplied timeline. It is used by the CL61-first policy to
    reserve the CL61 timetable before Radar and HATPRO are considered.
    """
    controlled = tuple(str(value) for value in priorities)
    if not controlled or len(set(controlled)) != len(controlled):
        raise ValueError("Operating priorities must contain unique instruments")
    unknown = tuple(value for value in controlled if value not in KIT_ORDER)
    if unknown:
        raise ValueError(f"Unknown operating priorities: {', '.join(unknown)}")

    if max_starts_per_day not in {0, 1}:
        raise ValueError("The advisory scheduler supports at most one start per UTC day")

    full_times = pd.DatetimeIndex(times)
    full_solar = np.asarray(solar_members_w, dtype=np.float64)
    components = np.asarray(component_members, dtype=np.float64)
    if len(full_times) == 0:
        raise ValueError("Instrument optimization requires at least one forecast time")
    if full_solar.ndim != 2 or full_solar.shape[1] < len(full_times):
        raise ValueError("Solar members must be a member x time array covering the forecast")
    full_solar = full_solar[:, : len(full_times)]
    member_count = full_solar.shape[0]
    if components.shape != (member_count, len(COMPONENTS)):
        raise ValueError("Component members must match the solar members and component schema")

    required_by_kit: dict[str, np.ndarray] = {}
    for kit, values in (required_active or {}).items():
        if kit not in controlled:
            raise ValueError(f"Required instrument {kit!r} is not in the controlled schedule")
        required = np.asarray(values, dtype=bool).reshape(-1)
        if required.shape != (len(full_times),):
            raise ValueError(
                f"Required {kit} activity must match the complete forecast time axis"
            )
        required_by_kit[kit] = required

    energy_model = battery_model or BatteryModel(
        usable_capacity_kwh=capacity_kwh,
        charge_efficiency=1.0,
        discharge_efficiency=1.0,
        max_charge_w=20_000.0,
        max_discharge_w=20_000.0,
    )
    capacities = (
        np.asarray(member_capacity_kwh, dtype=np.float64)
        if member_capacity_kwh is not None
        else np.full(member_count, energy_model.usable_capacity_kwh)
    )
    charge_efficiencies = (
        np.asarray(member_charge_efficiency, dtype=np.float64)
        if member_charge_efficiency is not None
        else np.full(member_count, energy_model.charge_efficiency)
    )
    discharge_efficiencies = (
        np.asarray(member_discharge_efficiency, dtype=np.float64)
        if member_discharge_efficiency is not None
        else np.full(member_count, energy_model.discharge_efficiency)
    )
    if any(
        values.shape != (member_count,)
        for values in (capacities, charge_efficiencies, discharge_efficiencies)
    ):
        raise ValueError("Battery member parameters must match the solar member dimension")
    capacities = np.clip(capacities, 10.0, 40.0)
    charge_efficiencies = np.clip(charge_efficiencies, 0.65, 1.0)
    discharge_efficiencies = np.clip(discharge_efficiencies, 0.65, 1.0)

    fixed_kits = set(mode_kits(base_mode)) - set(controlled)
    initial_active = tuple(
        bool(required_by_kit[kit][0])
        if kit in required_by_kit
        else kit in mode_kits(base_mode)
        for kit in controlled
    )
    initial_mode = mode_id(
        fixed_kits | {kit for position, kit in enumerate(controlled) if initial_active[position]}
    )
    component_power_w = {
        kit: max(float(np.nanmedian(components[:, COMPONENT_INDEX[kit]])), 0.0)
        for kit in controlled
    }
    profiles = mode_load_profiles or {}
    held_current_mode = str(current_mode or base_mode)
    segment_load_cache: dict[tuple[str, int], np.ndarray] = {}

    def segment_load(mode: str, segment_start: int, index: int) -> np.ndarray:
        key = (str(mode), int(segment_start))
        if key not in segment_load_cache:
            baseline = _load_members_for_modes(components, (mode,))[:, 0]
            profile = profiles.get(str(mode))
            if profile is None:
                loads = np.repeat(baseline[:, np.newaxis], len(full_times) - segment_start, axis=1)
            else:
                if segment_start > 0 or str(mode) != held_current_mode:
                    profile = force_startup(profile, full_times[segment_start])
                fallback = ControlledLoadEstimate(
                    p10_w=float(np.nanquantile(baseline, 0.10)),
                    p50_w=float(np.nanquantile(baseline, 0.50)),
                    p90_w=float(np.nanquantile(baseline, 0.90)),
                    source="finite_state_component_fallback",
                    sample_count=int(len(baseline)),
                ).validated()
                loads, _ = controlled_load_member_profiles(
                    profile,
                    full_times[segment_start:],
                    fallback,
                    member_count,
                    seed=int(
                        seed
                        + segment_start * 1009
                        + sum(ord(character) for character in str(mode))
                    ),
                )
            segment_load_cache[key] = np.asarray(loads, dtype=np.float64)
        return segment_load_cache[key][:, index - segment_start]

    initial_members = np.full(member_count, float(initial_soc), dtype=np.float64)
    candidates = [
        _AdditiveCandidate(
            modes=(initial_mode,),
            soc=initial_members,
            active=initial_active,
            # Existing loads were not started by this advisory plan and may be
            # shed at the first decision boundary.
            run_hours=tuple(float(minimum_run_hours) if value else 0.0 for value in initial_active),
            last_start_days=tuple(None for _ in controlled),
            starts=tuple(0 for _ in controlled),
            instrument_hours=tuple(0.0 for _ in controlled),
            controlled_energy_wh=0.0,
            minimum_p10=float(initial_soc),
            # t0 is an observed state, not a scheduled interval. Track the
            # first future operating interval so equally valuable plans prefer
            # earlier science collection.
            first_active_index=None,
            segment_start_index=0,
        )
    ]
    decision_count = min(len(full_times), int(horizon_hours) + 1)
    search_complete = decision_count == 1
    for index in range(1, decision_count):
        day = full_times[index].date()
        step_hours = max(
            float((full_times[index] - full_times[index - 1]) / pd.Timedelta(hours=1)),
            0.0,
        )
        next_candidates: list[_AdditiveCandidate] = []
        for candidate in candidates:
            for mask in range(1 << len(controlled)):
                active = tuple(bool(mask & (1 << position)) for position in range(len(controlled)))
                if any(
                    bool(required_by_kit[kit][index]) != active[position]
                    for position, kit in enumerate(controlled)
                    if kit in required_by_kit
                ):
                    continue
                if any(
                    candidate.active[position]
                    and candidate.run_hours[position] < float(minimum_run_hours) - 1e-9
                    and not active[position]
                    for position in range(len(controlled))
                ):
                    continue
                is_start = tuple(
                    active[position] and not candidate.active[position]
                    for position in range(len(controlled))
                )
                if any(
                    is_start[position]
                    and (
                        max_starts_per_day <= 0
                        or candidate.last_start_days[position] == day
                    )
                    for position in range(len(controlled))
                ):
                    continue

                mode = mode_id(fixed_kits | {
                    kit for position, kit in enumerate(controlled) if active[position]
                })
                segment_start = (
                    candidate.segment_start_index
                    if mode == candidate.modes[-1]
                    else index
                )
                load = segment_load(mode, segment_start, index)
                soc = np.clip(
                    candidate.soc
                    + _member_soc_delta_percent(
                        full_solar[:, index] - load,
                        step_hours,
                        energy_model,
                        capacities,
                        charge_efficiencies,
                        discharge_efficiencies,
                    ),
                    0.0,
                    100.0,
                )
                p10 = float(np.nanquantile(soc, 0.10))
                if p10 < float(minimum_soc) - 1e-9:
                    continue
                run_hours = tuple(
                    (
                        candidate.run_hours[position] + step_hours
                        if candidate.active[position]
                        else step_hours
                    )
                    if active[position]
                    else 0.0
                    for position in range(len(controlled))
                )
                instrument_hours = tuple(
                    candidate.instrument_hours[position] + (step_hours if active[position] else 0.0)
                    for position in range(len(controlled))
                )
                last_start_days = tuple(
                    day if is_start[position] else candidate.last_start_days[position]
                    for position in range(len(controlled))
                )
                starts = tuple(
                    candidate.starts[position] + int(is_start[position])
                    for position in range(len(controlled))
                )
                controlled_power = sum(
                    component_power_w[kit]
                    for position, kit in enumerate(controlled)
                    if active[position]
                )
                next_candidates.append(
                    _AdditiveCandidate(
                        modes=candidate.modes + (mode,),
                        soc=soc,
                        active=active,
                        run_hours=run_hours,
                        last_start_days=last_start_days,
                        starts=starts,
                        instrument_hours=instrument_hours,
                        controlled_energy_wh=(
                            candidate.controlled_energy_wh + controlled_power * step_hours
                        ),
                        minimum_p10=min(candidate.minimum_p10, p10),
                        first_active_index=(
                            index
                            if any(active) and candidate.first_active_index is None
                            else candidate.first_active_index
                        ),
                        segment_start_index=segment_start,
                    )
                )
        if not next_candidates:
            candidates = []
            break

        grouped: dict[tuple[object, ...], _AdditiveCandidate] = {}
        for candidate in next_candidates:
            key = (
                candidate.active,
                tuple(min(int(value), int(minimum_run_hours)) for value in candidate.run_hours),
                candidate.last_start_days,
                candidate.segment_start_index,
                int(np.nanmedian(candidate.soc)),
                int(candidate.minimum_p10),
            )
            incumbent = grouped.get(key)
            if incumbent is None or _additive_candidate_score(candidate) > _additive_candidate_score(incumbent):
                grouped[key] = candidate
        ranked = sorted(grouped.values(), key=_additive_candidate_score, reverse=True)
        width = max(int(beam_width), 1)
        # A global power-first cut can retain only plans that spend too much
        # reserve to survive the post-decision tail. Keep a stratified frontier
        # across current P10 SOC so lower-energy candidates remain available for
        # the final full-horizon safety check.
        by_soc_bucket: dict[int, list[_AdditiveCandidate]] = {}
        for candidate in ranked:
            bucket = int(float(np.nanquantile(candidate.soc, 0.10)) // 2.0)
            by_soc_bucket.setdefault(bucket, []).append(candidate)
        bucket_count = max(len(by_soc_bucket), 1)
        per_bucket = max(width // bucket_count, 1)
        candidates = []
        selected_ids: set[int] = set()
        for bucket in sorted(by_soc_bucket, reverse=True):
            for candidate in by_soc_bucket[bucket][:per_bucket]:
                if len(candidates) >= width:
                    break
                candidates.append(candidate)
                selected_ids.add(id(candidate))
        if len(candidates) < width:
            for candidate in ranked:
                if id(candidate) in selected_ids:
                    continue
                candidates.append(candidate)
                if len(candidates) >= width:
                    break
        reserve_candidate = next(
            (
                value
                for value in next_candidates
                if not any(value.active)
                and not any(value.starts)
                and sum(value.instrument_hours) == 0.0
            ),
            None,
        )
        if reserve_candidate is not None and all(value is not reserve_candidate for value in candidates):
            candidates = candidates[: max(width - 1, 0)] + [reserve_candidate]
        search_complete = index == decision_count - 1

    valid = [] if not search_complete else [
        candidate
        for candidate in candidates
        if all(
            not candidate.active[position]
            or candidate.run_hours[position] >= float(minimum_run_hours) - 1e-9
            for position in range(len(controlled))
        )
    ]

    def extend_candidate(
        candidate: _AdditiveCandidate,
    ) -> tuple[_AdditiveCandidate, tuple[str, ...], float, float]:
        modes = list(candidate.modes)
        soc = candidate.soc.copy()
        minimum_p10 = float(candidate.minimum_p10)
        previous_mode = modes[-1]
        segment_start = candidate.segment_start_index
        for index in range(decision_count, len(full_times)):
            required_mode = mode_id(
                fixed_kits
                | {
                    kit
                    for position, kit in enumerate(controlled)
                    if kit in required_by_kit and bool(required_by_kit[kit][index])
                }
            )
            if previous_mode != required_mode:
                previous_mode = required_mode
                segment_start = index
            step_hours = max(
                float((full_times[index] - full_times[index - 1]) / pd.Timedelta(hours=1)),
                0.0,
            )
            load = segment_load(required_mode, segment_start, index)
            soc = np.clip(
                soc
                + _member_soc_delta_percent(
                    full_solar[:, index] - load,
                    step_hours,
                    energy_model,
                    capacities,
                    charge_efficiencies,
                    discharge_efficiencies,
                ),
                0.0,
                100.0,
            )
            minimum_p10 = min(minimum_p10, float(np.nanquantile(soc, 0.10)))
            modes.append(required_mode)
        return candidate, tuple(modes), minimum_p10, float(np.nanquantile(soc, 0.10))

    evaluated = [extend_candidate(candidate) for candidate in valid]
    safe_evaluated = [value for value in evaluated if value[2] >= float(minimum_soc) - 1e-9]
    if safe_evaluated:
        best, modes, minimum_p10, final_p10 = max(
            safe_evaluated,
            key=lambda value: (
                *_additive_candidate_score(value[0]),
                float(value[2]),
                float(value[3]),
            ),
        )
    else:
        # Preserve the issue-time state at t0 and any explicitly required
        # future activity. A failed schedule must never turn a required CL61
        # continuation into a deceptively safe all-off trace.
        fallback_modes = tuple(
            initial_mode
            if index == 0
            else mode_id(
                fixed_kits
                | {
                    kit
                    for position, kit in enumerate(controlled)
                    if kit in required_by_kit and bool(required_by_kit[kit][index])
                }
            )
            for index in range(len(full_times))
        )
        fallback_loads = np.empty((member_count, len(full_times)), dtype=np.float64)
        segment_start = 0
        previous_mode = fallback_modes[0]
        fallback_loads[:, 0] = segment_load(previous_mode, segment_start, 0)
        for index in range(1, len(full_times)):
            if fallback_modes[index] != previous_mode:
                previous_mode = fallback_modes[index]
                segment_start = index
            fallback_loads[:, index] = segment_load(previous_mode, segment_start, index)
        fallback_soc = integrate_soc_members(
            initial_soc=initial_soc,
            times=full_times,
            solar_members_w=full_solar,
            load_members_w=fallback_loads,
            capacity_kwh=capacity_kwh,
            battery_model=energy_model,
            member_capacity_kwh=capacities,
            member_charge_efficiency=charge_efficiencies,
            member_discharge_efficiency=discharge_efficiencies,
        )
        fallback_p10 = np.nanquantile(fallback_soc, 0.10, axis=0)
        minimum_p10 = float(np.nanmin(fallback_p10))
        final_p10 = float(fallback_p10[-1])
        modes = fallback_modes
        fallback_instrument_hours = tuple(
            _instrument_hours(full_times, fallback_modes, kit, limit=decision_count)
            for kit in controlled
        )
        fallback_starts = tuple(
            _instrument_starts(fallback_modes, kit, limit=decision_count)
            for kit in controlled
        )
        fallback_energy_wh = sum(
            fallback_instrument_hours[position] * component_power_w[kit]
            for position, kit in enumerate(controlled)
        )
        best = _AdditiveCandidate(
            modes=fallback_modes[:decision_count],
            soc=fallback_soc[:, decision_count - 1],
            active=tuple(kit in mode_kits(fallback_modes[decision_count - 1]) for kit in controlled),
            run_hours=tuple(0.0 for _ in controlled),
            last_start_days=tuple(None for _ in controlled),
            starts=fallback_starts,
            instrument_hours=fallback_instrument_hours,
            controlled_energy_wh=fallback_energy_wh,
            minimum_p10=minimum_p10,
            first_active_index=None,
            segment_start_index=1 if len(full_times) > 1 else 0,
        )

    instrument_hours = {
        kit: float(best.instrument_hours[position])
        for position, kit in enumerate(controlled)
    }
    if np.isclose(minimum_p10, float(minimum_soc), rtol=0.0, atol=1e-9):
        minimum_p10 = float(minimum_soc)
    cl61_on = [index for index, value in enumerate(modes) if "CL61" in mode_kits(value)]
    start_time = full_times[cl61_on[0]] if cl61_on else None
    stop_time = None
    if cl61_on and cl61_on[-1] + 1 < len(full_times):
        stop_time = full_times[cl61_on[-1] + 1]
    return ScheduleResult(
        modes=modes,
        collection_hours=float(instrument_hours.get("CL61", 0.0)),
        start_time=start_time,
        stop_time=stop_time,
        minimum_p10_soc=float(minimum_p10),
        final_p10_soc=float(final_p10),
        safe=bool(minimum_p10 >= float(minimum_soc) - 1e-9),
        starts=int(best.starts[controlled.index("CL61")]) if "CL61" in controlled else 0,
        instrument_hours=instrument_hours,
        total_instrument_hours=float(sum(instrument_hours.values())),
        controlled_energy_kwh=float(best.controlled_energy_wh / 1000.0),
    )


def optimize_cl61_primary_schedule(
    *,
    times: pd.DatetimeIndex,
    solar_members_w: np.ndarray,
    component_members: np.ndarray,
    initial_soc: float,
    capacity_kwh: float,
    base_mode: str,
    battery_model: BatteryModel | None = None,
    member_capacity_kwh: np.ndarray | None = None,
    member_charge_efficiency: np.ndarray | None = None,
    member_discharge_efficiency: np.ndarray | None = None,
    horizon_hours: int = 96,
    minimum_soc: float = MINIMUM_OPERATIONAL_SOC_PCT,
    minimum_run_hours: int = MIN_RUN_HOURS,
    max_starts_per_day: int = MAX_STARTS_PER_UTC_DAY,
    beam_width: int = 600,
    mode_load_profiles: Mapping[str, StateLoadDynamics] | None = None,
    seed: int = 0,
) -> tuple[ScheduleResult, ScheduleResult]:
    """Return a CL61-first plan and its reserved CL61 timetable.

    The first pass searches only CL61 against the non-controlled station
    baseline.  When CL61 is already on, its activity is pinned through the
    planning horizon instead of silently shedding it at the first decision
    boundary.  Existing Radar/HATPRO states are also held: the initial
    controller is CL61-only, so the model must not assume that another live
    PDU outlet has changed.  The second pass fixes those timelines, then
    maximises additive Radar/HATPRO use only from residual reserve.  It
    therefore cannot trade CL61 science for a higher-power combination or
    claim energy from switching a non-automated live instrument off.
    """
    full_times = pd.DatetimeIndex(times)
    if len(full_times) == 0:
        raise ValueError("CL61-first optimization requires at least one forecast time")
    current_kits = set(mode_kits(base_mode))
    fixed_kits = current_kits - set(OPERATING_PRIORITY)
    existing_non_cl61 = (current_kits & set(OPERATING_PRIORITY)) - {"CL61"}
    primary_base_mode = mode_id(fixed_kits | existing_non_cl61)
    initially_on = "CL61" in current_kits
    primary_current_mode = mode_id(fixed_kits | existing_non_cl61 | ({"CL61"} if initially_on else set()))
    required_primary = (
        {"CL61": np.ones(len(full_times), dtype=bool)} if initially_on else None
    )
    primary = optimize_priority_schedule(
        times=full_times,
        solar_members_w=solar_members_w,
        component_members=component_members,
        initial_soc=initial_soc,
        capacity_kwh=capacity_kwh,
        base_mode=primary_base_mode,
        priorities=("CL61",),
        battery_model=battery_model,
        member_capacity_kwh=member_capacity_kwh,
        member_charge_efficiency=member_charge_efficiency,
        member_discharge_efficiency=member_discharge_efficiency,
        horizon_hours=horizon_hours,
        minimum_soc=minimum_soc,
        minimum_run_hours=minimum_run_hours,
        max_starts_per_day=max_starts_per_day,
        beam_width=max(1, int(beam_width) // 2),
        mode_load_profiles=mode_load_profiles,
        current_mode=primary_current_mode,
        required_active=required_primary,
        seed=seed,
    )
    cl61_timeline = np.asarray(
        ["CL61" in mode_kits(value) for value in primary.modes], dtype=bool
    )
    primary_start_mode = mode_id(fixed_kits | ({"CL61"} if bool(cl61_timeline[0]) else set()))
    required_combined = {"CL61": cl61_timeline}
    for kit in existing_non_cl61:
        required_combined[kit] = np.ones(len(full_times), dtype=bool)
    combined = optimize_priority_schedule(
        times=full_times,
        solar_members_w=solar_members_w,
        component_members=component_members,
        initial_soc=initial_soc,
        capacity_kwh=capacity_kwh,
        base_mode=primary_start_mode,
        priorities=OPERATING_PRIORITY,
        battery_model=battery_model,
        member_capacity_kwh=member_capacity_kwh,
        member_charge_efficiency=member_charge_efficiency,
        member_discharge_efficiency=member_discharge_efficiency,
        horizon_hours=horizon_hours,
        minimum_soc=minimum_soc,
        minimum_run_hours=minimum_run_hours,
        max_starts_per_day=max_starts_per_day,
        beam_width=beam_width,
        mode_load_profiles=mode_load_profiles,
        # The combined first step is exactly the observed station state: the
        # reserved CL61 timeline and any other live PDU instruments reproduce
        # ``base_mode`` at t0.  Preserve its learned phase rather than
        # spuriously charging an already-running Radar/HATPRO/CL61 startup
        # transient to the future plan.
        current_mode=base_mode,
        required_active=required_combined,
        seed=seed,
    )
    return combined, primary


def _schedule_modes(
    times: pd.DatetimeIndex,
    base_mode: str,
    start: pd.Timestamp,
    duration_hours: int,
    kit: str,
) -> tuple[str, ...]:
    if kit not in KIT_ORDER:
        raise ValueError(f"Unknown instrument: {kit}")
    base_kits = set(mode_kits(base_mode))
    base_kits.discard(kit)
    off_mode = mode_id(base_kits)
    on_mode = mode_id(base_kits | {kit})
    stop = pd.Timestamp(start) + pd.Timedelta(hours=int(duration_hours))
    return tuple(on_mode if pd.Timestamp(start) <= value < stop else off_mode for value in times)


def _scenario_members(
    modes: Sequence[str],
    *,
    times: pd.DatetimeIndex,
    solar_members: np.ndarray,
    component_members: np.ndarray,
    initial_soc: float,
    capacity_kwh: float,
    battery_model: BatteryModel | None = None,
    member_capacity_kwh: np.ndarray | None = None,
    member_charge_efficiency: np.ndarray | None = None,
    member_discharge_efficiency: np.ndarray | None = None,
    mode_load_profiles: Mapping[str, StateLoadDynamics] | None = None,
    current_mode: str | None = None,
    profile_states: Sequence[str] | None = None,
    forced_phases: Sequence[str | None] | None = None,
    load_adjustments_w: np.ndarray | None = None,
    adjustment_phase_codes: Sequence[int] | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loads = _load_members_for_modes(component_members, modes)
    phase_codes = np.full(loads.shape, PHASE_CODES[PHASE_STEADY], dtype=np.int8)
    profiles = mode_load_profiles or {}
    states = tuple(profile_states) if profile_states is not None else tuple(str(value) for value in modes)
    forced_phase_values = (
        tuple(forced_phases)
        if forced_phases is not None
        else tuple(None for _ in modes)
    )
    if len(states) != len(modes) or len(forced_phase_values) != len(modes):
        raise ValueError("Scenario profile state and phase arrays must match the mode timeline")
    start = 0
    while start < len(modes):
        stop = start + 1
        while (
            stop < len(modes)
            and modes[stop] == modes[start]
            and states[stop] == states[start]
            and forced_phase_values[stop] == forced_phase_values[start]
        ):
            stop += 1
        profile = profiles.get(str(states[start])) or profiles.get(str(modes[start]))
        if profile is not None:
            forced_phase = forced_phase_values[start]
            if forced_phase is not None and forced_phase in profile.phase_profiles:
                profile = replace(
                    profile,
                    current_phase=str(forced_phase),
                    state_started_at=pd.Timestamp(times[start]).isoformat(),
                    phase_started_at=pd.Timestamp(times[start]).isoformat(),
                )
            elif start > 0 or str(modes[start]) != str(current_mode):
                profile = force_startup(profile, times[start])
            baseline = loads[:, start]
            fallback = ControlledLoadEstimate(
                p10_w=float(np.nanquantile(baseline, 0.10)),
                p50_w=float(np.nanquantile(baseline, 0.50)),
                p90_w=float(np.nanquantile(baseline, 0.90)),
                source="finite_state_component_fallback",
                sample_count=int(len(baseline)),
            ).validated()
            profiled, segment_phases = controlled_load_member_profiles(
                profile,
                times[start:stop],
                fallback,
                loads.shape[0],
                seed=int(seed + start * 1009 + sum(ord(character) for character in str(modes[start]))),
            )
            loads[:, start:stop] = profiled
            phase_codes[:, start:stop] = segment_phases
        start = stop
    if load_adjustments_w is not None:
        adjustments = np.asarray(load_adjustments_w, dtype=np.float64)
        if adjustments.shape == (len(times),):
            adjustments = np.repeat(adjustments[None, :], loads.shape[0], axis=0)
        if adjustments.shape != loads.shape or not np.isfinite(adjustments).all():
            raise ValueError("Scenario load adjustments must be finite member x time values")
        loads = np.clip(loads + adjustments, 0.0, None)
    if adjustment_phase_codes is not None:
        codes = np.asarray(adjustment_phase_codes, dtype=np.int8)
        if codes.shape != (len(times),):
            raise ValueError("Scenario adjustment phase codes must match the time axis")
        phase_codes[:] = codes[None, :]
    validate_state_held_load(
        np.asarray([mode_code(value) for value in modes], dtype=np.int16),
        loads,
        phase_codes=phase_codes,
    )
    soc = integrate_soc_members(
        initial_soc=initial_soc,
        times=times,
        solar_members_w=solar_members,
        load_members_w=loads,
        capacity_kwh=capacity_kwh,
        battery_model=battery_model,
        member_capacity_kwh=member_capacity_kwh,
        member_charge_efficiency=member_charge_efficiency,
        member_discharge_efficiency=member_discharge_efficiency,
    )
    return loads, soc, phase_codes


def _validate_scenario_invariants(output: xr.Dataset) -> None:
    for prefix in ("ScenarioLoad", "ScenarioSOC"):
        p10 = np.asarray(output[f"{prefix}P10{'Watts' if prefix == 'ScenarioLoad' else ''}"].values)
        p50 = np.asarray(output[f"{prefix}P50{'Watts' if prefix == 'ScenarioLoad' else ''}"].values)
        p90 = np.asarray(output[f"{prefix}P90{'Watts' if prefix == 'ScenarioLoad' else ''}"].values)
        if np.any(p10 > p50 + 1e-5) or np.any(p50 > p90 + 1e-5):
            raise ValueError(f"{prefix} quantiles are not ordered")
    dc = np.asarray(output.sel(scenario=SCENARIO_DC_ONLY)["ScenarioLoadP50Watts"].values, dtype=np.float64)
    if not np.isfinite(dc).all() or np.nanmedian(dc) <= 0.0:
        raise ValueError("DC-only scenario must retain a positive learned station baseline")
    if "suggested_all_uas_tier3" in set(str(value) for value in output["scenario"].values):
        all_on = np.asarray(
            output.sel(scenario="suggested_all_uas_tier3")["ScenarioLoadP50Watts"].values,
            dtype=np.float64,
        )
        if np.any(all_on + 1e-5 < dc):
            raise ValueError("All-instruments UAS tier 3 load cannot be below DC-only load")
    for scenario_index in range(int(output.sizes.get("scenario", 0))):
        mode_codes = np.asarray(
            output["ScenarioModeCode"].isel(scenario=scenario_index).values,
            dtype=np.int16,
        )
        load_quantiles = np.stack(
            [
                np.asarray(output[name].isel(scenario=scenario_index).values, dtype=np.float64)
                for name in (
                    "ScenarioLoadP10Watts",
                    "ScenarioLoadP50Watts",
                    "ScenarioLoadP90Watts",
                )
            ]
        )
        phase_epochs = np.asarray(
            output["ScenarioLoadPhaseEpoch"].isel(scenario=scenario_index).values,
            dtype=np.int16,
        )
        validate_state_held_load(mode_codes, load_quantiles, phase_codes=phase_epochs)


def build_operating_scenarios(
    power: xr.Dataset,
    deterministic: xr.Dataset,
    model: OperatingModelResult,
    *,
    ensemble: xr.Dataset | None = None,
    horizon_hours: int = 240,
    optimization_hours: int = 96,
    capacity_kwh: float | None = None,
) -> xr.Dataset:
    deterministic_solar_contract = str(deterministic.attrs.get("solar_calibration_contract_id", ""))
    ensemble_solar_contract = str(ensemble.attrs.get("solar_calibration_contract_id", "")) if ensemble is not None else ""
    ensemble, solar_alignment_metadata = _align_ensemble_solar_contract(
        deterministic,
        ensemble,
    )
    issue_time, initial_soc = _latest_soc(power)
    available_end = pd.Timestamp(deterministic["time"].values[-1]) if "time" in deterministic else issue_time
    available_hours = max(int((available_end - issue_time) / pd.Timedelta(hours=1)), 1)
    actual_horizon = min(int(horizon_hours), available_hours)
    capacity = float(capacity_kwh or deterministic.attrs.get("battery_capacity_kwh", 26.0))
    battery_model = (
        BatteryModel.from_attrs(deterministic.attrs, default_capacity_kwh=capacity)
        if "battery_energy_model" in deterministic.attrs
        else BatteryModel(
            usable_capacity_kwh=capacity,
            charge_efficiency=1.0,
            discharge_efficiency=1.0,
            max_charge_w=20_000.0,
            max_discharge_w=20_000.0,
        )
    )
    capacity = battery_model.usable_capacity_kwh
    times, solar_members, solar_metadata = _hourly_solar_members(
        deterministic,
        ensemble,
        issue_time=issue_time,
        horizon_hours=actual_horizon,
    )
    solar_metadata.update(
        {
            "solar_calibration_contract_id": ensemble_solar_contract or deterministic_solar_contract,
            "solar_calibration_factor_w_per_wm2": str(
                deterministic.attrs.get("solar_calibration_factor_w_per_wm2", "")
            ),
            "solar_mos_factor_by_lead_bucket": str(
                deterministic.attrs.get("solar_mos_factor_by_lead_bucket", "{}")
            ),
            **solar_alignment_metadata,
        }
    )
    member_count = solar_members.shape[0]
    member_capacity, member_charge_efficiency, member_discharge_efficiency = _battery_member_parameters(
        ensemble,
        battery_model,
        member_count,
    )
    seed = int(issue_time.value % (2**32 - 1))
    component_members = _component_members(
        model.component_mean,
        model.component_covariance,
        member_count,
        seed=seed,
        regimes=model.component_regimes,
    )
    base_mode = model.current_mode if model.current_mode != MODE_UNKNOWN_AC else MODE_DC_ONLY
    planning_component_members = component_members.copy()
    if model.current_uas_tier is not None and "UAS" in mode_kits(base_mode):
        planning_component_members[:, COMPONENT_INDEX["UAS"]] = _tier_profile_members(
            model.uas_tier_profiles.get(str(model.current_uas_tier)),
            member_count,
            seed=seed + model.current_uas_tier * 1009,
        )
    optimizer_profiles = {
        name: profile
        for name, profile in model.mode_load_profiles.items()
        if "__" not in name
        and not (
            model.current_uas_tier is not None
            and "UAS" in mode_kits(name)
        )
    }
    modeled_current_load = _load_members_for_modes(
        planning_component_members,
        tuple(base_mode for _ in times),
    )
    upstream_current_load, _ = _current_system_load_members(
        deterministic,
        ensemble,
        times,
        member_count,
        fallback=modeled_current_load,
    )
    current_system_load = modeled_current_load
    current_system_load_source = "scenario_component_model"
    current_modes_for_rule = tuple(base_mode for _ in times)
    _, current_soc_for_rule, _ = _scenario_members(
        current_modes_for_rule,
        times=times,
        solar_members=solar_members,
        component_members=planning_component_members,
        initial_soc=initial_soc,
        capacity_kwh=capacity,
        battery_model=battery_model,
        member_capacity_kwh=member_capacity,
        member_charge_efficiency=member_charge_efficiency,
        member_discharge_efficiency=member_discharge_efficiency,
        mode_load_profiles=model.mode_load_profiles,
        current_mode=base_mode,
        profile_states=tuple(model.current_load_state for _ in times),
        seed=seed,
    )
    p50_continuation = evaluate_p50_continuation_rule(
        times=times,
        soc_p50=np.nanquantile(current_soc_for_rule, 0.50, axis=0),
        current_mode=base_mode,
        horizon_hours=min(optimization_hours, actual_horizon),
    )
    optimized, cl61_primary = optimize_cl61_primary_schedule(
        times=times,
        solar_members_w=solar_members,
        component_members=planning_component_members,
        initial_soc=initial_soc,
        capacity_kwh=capacity,
        battery_model=battery_model,
        member_capacity_kwh=member_capacity,
        member_charge_efficiency=member_charge_efficiency,
        member_discharge_efficiency=member_discharge_efficiency,
        base_mode=base_mode,
        horizon_hours=min(optimization_hours, actual_horizon),
        mode_load_profiles=optimizer_profiles,
        seed=seed,
    )
    def completed_modes(result: ScheduleResult) -> list[str]:
        modes = list(result.modes)
        if len(modes) < len(times):
            # Outside the explicit decision horizon, retain the actually
            # observed station state rather than modelling an implicit PDU
            # switch-off of CL61, Radar, or HATPRO.
            modes.extend([mode_id(set(mode_kits(base_mode)))] * (len(times) - len(modes)))
        return modes

    def published_phase_minimum(modes: Sequence[str]) -> float:
        _, phase_soc, _ = _scenario_members(
            modes,
            times=times,
            solar_members=solar_members,
            component_members=planning_component_members,
            initial_soc=initial_soc,
            capacity_kwh=capacity,
            battery_model=battery_model,
            member_capacity_kwh=member_capacity,
            member_charge_efficiency=member_charge_efficiency,
            member_discharge_efficiency=member_discharge_efficiency,
            mode_load_profiles=optimizer_profiles,
            current_mode=base_mode,
            seed=seed,
        )
        return float(np.nanmin(np.nanquantile(phase_soc, 0.10, axis=0)))

    optimized_modes = completed_modes(optimized)
    optimized_phase_minimum_p10 = published_phase_minimum(optimized_modes)
    # The secondary search may contain combinations whose exact composed
    # phase profile is more conservative than its component search estimate.
    # In that case retain the already-validated CL61-first timetable instead
    # of publishing an unsafe higher-power combination.
    phase_validation_fallback = False
    if optimized.safe and optimized_phase_minimum_p10 < MINIMUM_OPERATIONAL_SOC_PCT - 1e-6:
        optimized = cl61_primary
        optimized_modes = completed_modes(optimized)
        optimized_phase_minimum_p10 = published_phase_minimum(optimized_modes)
        phase_validation_fallback = True
    p50_continuation_modes = apply_p50_continuation_rule(
        optimized_modes,
        p50_continuation,
    )

    scenario_modes: dict[str, tuple[str, ...]] = {
        SCENARIO_CURRENT: tuple(base_mode for _ in times),
        SCENARIO_DC_ONLY: tuple(MODE_DC_ONLY for _ in times),
        SCENARIO_OPTIMIZED: tuple(optimized_modes),
        SCENARIO_P50_CONTINUATION: p50_continuation_modes,
    }
    scenario_uas_tiers: dict[str, int] = {}
    scenario_uas_charge_hours: dict[str, float] = {}
    scenario_cl61_phases: dict[str, str] = {}
    for definition in SUGGESTED_OPERATING_SCENARIOS:
        scenario_modes[definition.scenario_id] = tuple(
            mode_id(definition.instruments) for _ in times
        )
        if definition.uas_effective_tier is not None:
            scenario_uas_tiers[definition.scenario_id] = int(definition.uas_effective_tier)
    for definition in POWER_STATE_SCENARIOS:
        scenario_modes[definition.scenario_id] = tuple(
            mode_id(definition.instruments) for _ in times
        )
        if definition.uas_tier is not None:
            scenario_uas_tiers[definition.scenario_id] = int(definition.uas_tier)
        if definition.uas_charging:
            scenario_uas_charge_hours[definition.scenario_id] = float(
                model.uas_charge_profiles.get(str(definition.uas_tier), {}).get(
                    "duration_hours", UAS_CHARGE_DURATION_HOURS
                )
            )
        if definition.cl61_phase is not None:
            scenario_cl61_phases[definition.scenario_id] = str(
                definition.cl61_phase
            )
    for observed_mode in model.observed_modes:
        if observed_mode in {MODE_DC_ONLY, mode_id(("CL61",))}:
            continue
        scenario_modes.setdefault(f"learned_{observed_mode}", tuple(observed_mode for _ in times))

    scenario_ids = tuple(scenario_modes)
    labels = {
        SCENARIO_CURRENT: f"Current: {mode_label(base_mode)}",
        SCENARIO_DC_ONLY: "DC-Only",
        SCENARIO_CL61: "DC + CL61 Continuously On",
        SCENARIO_OPTIMIZED: "CL61-first Instrument Schedule",
        SCENARIO_P50_CONTINUATION: (
            f"P50 continuation: keep {_human_list(p50_continuation.held_instruments)} on"
            if p50_continuation.eligible
            else "P50 continuation unavailable: P10 priority plan"
        ),
    }
    labels.update(
        {definition.scenario_id: definition.label for definition in SUGGESTED_OPERATING_SCENARIOS}
    )
    labels.update(
        {definition.scenario_id: definition.label for definition in POWER_STATE_SCENARIOS}
    )
    power_state_definitions = {
        definition.scenario_id: definition for definition in POWER_STATE_SCENARIOS
    }
    load_p10: list[np.ndarray] = []
    load_p50: list[np.ndarray] = []
    load_p90: list[np.ndarray] = []
    soc_p10: list[np.ndarray] = []
    soc_p50: list[np.ndarray] = []
    soc_p90: list[np.ndarray] = []
    below_probability: list[np.ndarray] = []
    mode_codes: list[np.ndarray] = []
    load_phase_codes: list[np.ndarray] = []
    load_phase_epochs: list[np.ndarray] = []
    collection_hours: list[float] = []
    minimum_p10: list[float] = []
    final_p10: list[float] = []
    minimum_p50: list[float] = []
    final_p50: list[float] = []
    starts: list[int] = []
    start_times: list[np.datetime64] = []
    stop_times: list[np.datetime64] = []
    safe_values: list[float] = []
    uas_tier_values: list[np.ndarray] = []
    uas_charging_values: list[np.ndarray] = []
    power_state_values: list[np.ndarray] = []
    active_instrument_counts: list[np.ndarray] = []
    for scenario_id, modes in scenario_modes.items():
        has_uas = any("UAS" in mode_kits(value) for value in modes)
        tier = scenario_uas_tiers.get(scenario_id)
        if tier is None and has_uas:
            tier = model.current_uas_tier
        scenario_components = component_members.copy()
        if tier is not None:
            scenario_components[:, COMPONENT_INDEX["UAS"]] = _tier_profile_members(
                model.uas_tier_profiles.get(str(tier)),
                member_count,
                seed=seed + tier * 1009,
            )
        charge_hours = scenario_uas_charge_hours.get(scenario_id, 0.0)
        if (
            charge_hours <= 0.0
            and model.current_uas_charging
            and scenario_id
            in {SCENARIO_CURRENT, SCENARIO_OPTIMIZED, SCENARIO_P50_CONTINUATION}
        ):
            charge_hours = float(
                model.uas_charge_profiles.get(str(tier), {}).get(
                    "duration_hours", UAS_CHARGE_DURATION_HOURS
                )
            )
        charge_mask = np.zeros(len(times), dtype=bool)
        load_adjustments = None
        adjustment_codes = None
        if tier in UAS_CHARGE_TIERS and charge_hours > 0.0 and has_uas:
            charge_mask = (
                (times - times[0]) / pd.Timedelta(hours=1)
                <= float(charge_hours) + 1e-9
            )
            increments = _charge_increment_members(
                model.uas_charge_profiles.get(str(tier)),
                member_count,
                seed=seed + int(tier) * 2029,
            )
            load_adjustments = np.zeros((member_count, len(times)), dtype=np.float64)
            load_adjustments[:, charge_mask] = increments[:, None]
            adjustment_codes = np.full(
                len(times), PHASE_CODES[PHASE_STEADY], dtype=np.int8
            )
            adjustment_codes[charge_mask] = PHASE_CODES.get(
                "startup", PHASE_CODES[PHASE_STEADY]
            )
        forced_cl61_phase = scenario_cl61_phases.get(scenario_id)
        forced_phases = tuple(forced_cl61_phase for _ in times)
        heater_state = (
            forced_cl61_phase == PHASE_FAN_HIGH
            or (
                scenario_id in {SCENARIO_CURRENT, SCENARIO_CL61}
                and model.current_cl61_state == "cl61_heater_on"
            )
        )
        profile_states = tuple(
            operating_load_state_id(
                value,
                uas_tier=(tier if "UAS" in mode_kits(value) else None),
                # Charge increment learning is separate from the base exact
                # state profile, so it is added exactly once below.
                uas_charging=False,
                cl61_heater_on=heater_state and "CL61" in mode_kits(value),
            )
            for value in modes
        )
        scenario_profiles = model.mode_load_profiles
        if tier is not None:
            # A generic UAS mode may contain a mixture of field tiers.  When a
            # canonical tier is requested, use its exact composed profile when
            # available and otherwise fall back to the tier component profile.
            scenario_profiles = {
                name: profile
                for name, profile in model.mode_load_profiles.items()
                if "__" in name or "UAS" not in mode_kits(name)
            }
        loads, soc, member_load_phases = _scenario_members(
            modes,
            times=times,
            solar_members=solar_members,
            component_members=scenario_components,
            initial_soc=initial_soc,
            capacity_kwh=capacity,
            battery_model=battery_model,
            member_capacity_kwh=member_capacity,
            member_charge_efficiency=member_charge_efficiency,
            member_discharge_efficiency=member_discharge_efficiency,
            mode_load_profiles=scenario_profiles,
            current_mode=base_mode,
            profile_states=profile_states,
            forced_phases=forced_phases,
            load_adjustments_w=load_adjustments,
            adjustment_phase_codes=adjustment_codes,
            seed=seed,
        )
        if scenario_id == SCENARIO_CURRENT:
            # The operating-state job runs much more often than the ECMWF
            # planning retrieval. Keep the solar cycle, but anchor current load
            # to the freshly detected exact state and its autonomous phase.
            # Reusing an older planning load here can otherwise leave the
            # current trace in CL61's low phase after its heater/blower starts.
            current_system_load = loads.copy()
            current_system_load_source = (
                "fresh_exact_state_phase_model"
                if base_mode in model.mode_load_profiles
                else "scenario_component_model"
            )
        load_p10.append(np.nanquantile(loads, 0.10, axis=0))
        load_p50.append(np.nanquantile(loads, 0.50, axis=0))
        load_p90.append(np.nanquantile(loads, 0.90, axis=0))
        p10 = np.nanquantile(soc, 0.10, axis=0)
        soc_p10.append(p10)
        p50 = np.nanquantile(soc, 0.50, axis=0)
        soc_p50.append(p50)
        soc_p90.append(np.nanquantile(soc, 0.90, axis=0))
        below_probability.append(np.mean(soc < MINIMUM_OPERATIONAL_SOC_PCT, axis=0))
        scenario_mode_codes = np.asarray([mode_code(value) for value in modes], dtype=np.int16)
        mode_codes.append(scenario_mode_codes)
        active_instrument_counts.append(
            np.asarray(
                [sum(kit in mode_kits(value) for kit in OPERATING_PRIORITY) for value in modes],
                dtype=np.int8,
            )
        )
        central_phases = np.asarray(
            [
                int(np.bincount(member_load_phases[:, index].astype(np.int64)).argmax())
                for index in range(member_load_phases.shape[1])
            ],
            dtype=np.int8,
        )
        phase_changed = np.r_[
            False,
            np.any(member_load_phases[:, 1:] != member_load_phases[:, :-1], axis=0)
            | (scenario_mode_codes[1:] != scenario_mode_codes[:-1]),
        ]
        load_phase_codes.append(central_phases)
        load_phase_epochs.append(np.cumsum(phase_changed).astype(np.int16))
        on = np.asarray(["CL61" in mode_kits(value) for value in modes], dtype=bool)
        collection_hours.append(float(np.count_nonzero(on[1:])))
        scenario_minimum_p10 = float(np.nanmin(p10))
        if np.isclose(
            scenario_minimum_p10,
            MINIMUM_OPERATIONAL_SOC_PCT,
            rtol=0.0,
            atol=1e-6,
        ):
            scenario_minimum_p10 = float(MINIMUM_OPERATIONAL_SOC_PCT)
        minimum_p10.append(scenario_minimum_p10)
        final_p10.append(float(p10[-1]))
        safe_values.append(
            float(scenario_minimum_p10 >= MINIMUM_OPERATIONAL_SOC_PCT - 1e-6)
        )
        minimum_p50.append(float(np.nanmin(p50)))
        final_p50.append(float(p50[-1]))
        uas_tier_values.append(np.full(len(times), tier if tier is not None else -1, dtype=np.int16))
        uas_charging_values.append(charge_mask.astype(np.int8))
        power_state_values.append(
            np.asarray(
                [
                    operating_load_state_id(
                        value,
                        uas_tier=(tier if "UAS" in mode_kits(value) else None),
                        uas_charging=bool(charge_mask[index]),
                        cl61_heater_on=(
                            heater_state and "CL61" in mode_kits(value)
                        ),
                    )
                    for index, value in enumerate(modes)
                ],
                dtype=str,
            )
        )
        transitions = np.flatnonzero(on & ~np.r_[False, on[:-1]])
        stops_found = np.flatnonzero(~on & np.r_[False, on[:-1]])
        starts.append(int(len(transitions)))
        start_times.append(times[transitions[0]].to_datetime64() if transitions.size else np.datetime64("NaT", "ns"))
        stop_times.append(times[stops_found[-1]].to_datetime64() if stops_found.size else np.datetime64("NaT", "ns"))
        labels.setdefault(scenario_id, mode_label(modes[0]))

    def scenario_maturity(scenario_id: str) -> str:
        if scenario_id in CORE_SCENARIOS:
            return "core"
        definition = power_state_definitions.get(scenario_id)
        if definition is not None:
            if definition.uas_tier is not None:
                profiles = (
                    model.uas_charge_profiles
                    if definition.uas_charging
                    else model.uas_tier_profiles
                )
                fallback = "estimated" if definition.uas_charging else "unobserved"
                return str(
                    profiles.get(str(definition.uas_tier), {}).get(
                        "maturity", fallback
                    )
                )
            return str(
                model.cl61_state_profiles.get(definition.state_id, {}).get(
                    "maturity", "unobserved"
                )
            )
        if scenario_id in SUGGESTED_OPERATING_SCENARIO_IDS:
            if scenario_id in scenario_uas_tiers:
                return str(
                    model.uas_tier_profiles.get(
                        str(scenario_uas_tiers[scenario_id]), {}
                    ).get("maturity", "provisional")
                )
            return "suggested"
        return model.mode_maturity.get(
            scenario_id.removeprefix("learned_"), "observed"
        )

    optimized_index = scenario_ids.index(SCENARIO_OPTIMIZED)
    decision_count = min(len(times), int(optimization_hours) + 1)
    optimized_instrument_hours = {
        kit: _instrument_hours(
            times,
            optimized_modes,
            kit,
            limit=decision_count,
        )
        for kit in OPERATING_PRIORITY
    }
    optimized_instrument_starts = {
        kit: _instrument_starts(
            optimized_modes,
            kit,
            limit=decision_count,
        )
        for kit in OPERATING_PRIORITY
    }
    optimized_total_instrument_hours = float(sum(optimized_instrument_hours.values()))
    optimized_component_power_w = {
        kit: max(float(np.nanmedian(component_members[:, COMPONENT_INDEX[kit]])), 0.0)
        for kit in OPERATING_PRIORITY
    }
    optimized_daily_operations = _daily_operating_summary(
        times,
        optimized_modes,
        kits=OPERATING_PRIORITY,
        component_power_w=optimized_component_power_w,
        limit=decision_count,
    )
    optimized_controlled_energy_kwh = float(
        sum(
            float(row.get("controlled_energy_kwh", 0.0))
            for row in optimized_daily_operations.values()
        )
    )
    optimized_diagnostic = describe_priority_schedule(
        current_mode=base_mode,
        safe=bool(safe_values[optimized_index] >= 0.5),
        instrument_hours=optimized_instrument_hours,
        minimum_p10_soc=minimum_p10[optimized_index],
        priorities=OPERATING_PRIORITY,
        controlled_energy_kwh=optimized_controlled_energy_kwh,
        policy=DEFAULT_SCHEDULE_POLICY,
        continuation_required=("CL61" in mode_kits(base_mode)),
        held_instruments=(set(mode_kits(base_mode)) & set(OPERATING_PRIORITY)) - {"CL61"},
    )

    output = xr.Dataset(
        {
            "ScenarioLoadP10Watts": (("scenario", "time"), np.asarray(load_p10, dtype=np.float32)),
            "ScenarioLoadP50Watts": (("scenario", "time"), np.asarray(load_p50, dtype=np.float32)),
            "ScenarioLoadP90Watts": (("scenario", "time"), np.asarray(load_p90, dtype=np.float32)),
            "ScenarioSOCP10": (("scenario", "time"), np.asarray(soc_p10, dtype=np.float32)),
            "ScenarioSOCP50": (("scenario", "time"), np.asarray(soc_p50, dtype=np.float32)),
            "ScenarioSOCP90": (("scenario", "time"), np.asarray(soc_p90, dtype=np.float32)),
            "ScenarioBelow40Probability": (("scenario", "time"), np.asarray(below_probability, dtype=np.float32)),
            "ScenarioModeCode": (("scenario", "time"), np.asarray(mode_codes, dtype=np.int16)),
            "ScenarioActiveInstrumentCount": (
                ("scenario", "time"),
                np.asarray(active_instrument_counts, dtype=np.int8),
            ),
            "ScenarioLoadPhaseCode": (("scenario", "time"), np.asarray(load_phase_codes, dtype=np.int8)),
            "ScenarioLoadPhaseEpoch": (("scenario", "time"), np.asarray(load_phase_epochs, dtype=np.int16)),
            "ScenarioCollectionHours": (("scenario",), np.asarray(collection_hours, dtype=np.float32)),
            "ScenarioMinimumP10SOC": (("scenario",), np.asarray(minimum_p10, dtype=np.float32)),
            "ScenarioFinalP10SOC": (("scenario",), np.asarray(final_p10, dtype=np.float32)),
            "ScenarioMinimumP50SOC": (("scenario",), np.asarray(minimum_p50, dtype=np.float32)),
            "ScenarioFinalP50SOC": (("scenario",), np.asarray(final_p50, dtype=np.float32)),
            "ScenarioSafe": (("scenario",), np.asarray(safe_values, dtype=np.float32)),
            "ScenarioStarts": (("scenario",), np.asarray(starts, dtype=np.int16)),
            "ScenarioStartTime": (("scenario",), np.asarray(start_times, dtype="datetime64[ns]")),
            "ScenarioStopTime": (("scenario",), np.asarray(stop_times, dtype="datetime64[ns]")),
            "ScenarioUASEffectiveTier": (("scenario", "time"), np.asarray(uas_tier_values, dtype=np.int16)),
            "ScenarioUASCanonicalTier": (("scenario", "time"), np.asarray(uas_tier_values, dtype=np.int16)),
            "ScenarioUASCharging": (("scenario", "time"), np.asarray(uas_charging_values, dtype=np.int8)),
            "ScenarioPowerState": (("scenario", "time"), np.asarray(power_state_values, dtype=str)),
            "SolarEnsembleWatts": (("member", "time"), solar_members.astype(np.float32)),
            "SolarP10Watts": (("time",), np.nanquantile(solar_members, 0.10, axis=0).astype(np.float32)),
            "SolarP50Watts": (("time",), np.nanquantile(solar_members, 0.50, axis=0).astype(np.float32)),
            "SolarP90Watts": (("time",), np.nanquantile(solar_members, 0.90, axis=0).astype(np.float32)),
            "ComponentLoadWatts": (("member", "component"), planning_component_members.astype(np.float32)),
            "BatteryUsableCapacityKWhEnsemble": (("member",), member_capacity.astype(np.float32)),
            "BatteryChargeEfficiencyEnsemble": (("member",), member_charge_efficiency.astype(np.float32)),
            "BatteryDischargeEfficiencyEnsemble": (("member",), member_discharge_efficiency.astype(np.float32)),
        },
        coords={
            "scenario": np.asarray(scenario_ids, dtype=str),
            "scenario_label": (("scenario",), np.asarray([labels[value] for value in scenario_ids], dtype=str)),
            "scenario_mode_maturity": (
                ("scenario",),
                np.asarray(
                    [scenario_maturity(value) for value in scenario_ids],
                    dtype=str,
                ),
            ),
            "time": times.to_numpy(dtype="datetime64[ns]"),
            "member": np.arange(1, member_count + 1, dtype=np.int16),
            "component": np.asarray(COMPONENTS, dtype=str),
        },
        attrs={
            "power_operating_scenarios_product": "true",
            "schema_version": str(SCENARIO_SCHEMA_VERSION),
            "model": MODEL_NAME,
            "model_version": str(MODEL_VERSION),
            "load_phase_schema_version": str(LOAD_PHASE_SCHEMA_VERSION),
            "generated_at_utc": _utc_now(),
            "initial_soc_time": issue_time.isoformat(),
            "initial_soc_pct": f"{initial_soc:.6g}",
            "battery_capacity_kwh": f"{capacity:.6g}",
            "current_mode": model.current_mode,
            "current_mode_label": mode_label(model.current_mode),
            "current_mode_confidence": f"{model.current_confidence:.6g}",
            "current_mode_maturity": model.mode_maturity.get(model.current_mode, "observed"),
            "current_load_state": model.current_load_state,
            "current_uas_effective_tier": (
                str(model.current_uas_effective_tier)
                if model.current_uas_effective_tier is not None
                else ""
            ),
            "current_uas_canonical_tier": (
                str(model.current_uas_tier) if model.current_uas_tier is not None else ""
            ),
            "current_uas_charging": str(model.current_uas_charging).lower(),
            "current_uas_state": model.current_uas_state or "",
            "current_cl61_state": model.current_cl61_state,
            "observed_modes": json.dumps(list(model.observed_modes)),
            "mode_maturity": json.dumps(model.mode_maturity, sort_keys=True),
            "learned_power_state_catalog": json.dumps(
                state_catalog_records(), sort_keys=True
            ),
            "power_state_scenario_ids": json.dumps(list(POWER_STATE_SCENARIO_IDS)),
            "uas_tier_learning_sources": json.dumps(
                UAS_TIER_LEARNING_SOURCES, sort_keys=True
            ),
            "uas_charge_event_kit": UAS_CHARGE_EVENT_KIT,
            "uas_charge_estimated_increment_w": f"{UAS_CHARGE_ESTIMATE_W:g}",
            "uas_charge_estimated_duration_hours": f"{UAS_CHARGE_DURATION_HOURS:g}",
            "uas_charge_planning_energy_wh": f"{UAS_CHARGE_PLANNING_ENERGY_WH:g}",
            "uas_charge_prior_source": UAS_CHARGE_PRIOR_SOURCE,
            "uas_charge_promotion_minimum_complete_episodes": str(UAS_CHARGE_RELIABLE_EPISODES),
            "uas_charge_promotion_minimum_days": str(UAS_CHARGE_RELIABLE_DAYS),
            "scenario_base_mode": base_mode,
            "load_baseline_source": "finite_state_component_model_for_all_operational_scenarios",
            "load_state_contract": CONTROLLED_LOAD_CONTRACT,
            "load_state_hold_policy": STATE_HOLD_POLICY,
            "current_system_load_source": current_system_load_source,
            "current_system_load_p50_w": f"{float(np.nanmedian(current_system_load)):.6g}",
            "modeled_current_load_p50_w": f"{float(np.nanmedian(modeled_current_load)):.6g}",
            "upstream_current_load_p50_w": f"{float(np.nanmedian(upstream_current_load)):.6g}",
            "current_load_model_disagreement_w": f"{float(np.nanmedian(upstream_current_load - modeled_current_load)):.6g}",
            "mode_load_profiles": json.dumps(
                {name: profile.to_dict() for name, profile in model.mode_load_profiles.items()},
                sort_keys=True,
            ),
            "uas_tier_profiles": json.dumps(model.uas_tier_profiles, sort_keys=True),
            "uas_charge_profiles": json.dumps(model.uas_charge_profiles, sort_keys=True),
            "cl61_state_profiles": json.dumps(model.cl61_state_profiles, sort_keys=True),
            "forecast_horizon_hours": str(actual_horizon),
            "optimization_horizon_hours": str(min(optimization_hours, actual_horizon)),
            "minimum_operational_soc_pct": f"{MINIMUM_OPERATIONAL_SOC_PCT:g}",
            "operating_decision_horizon_hours": str(min(optimization_hours, actual_horizon)),
            "operating_safety_constraint": f"P10 SOC >= {MINIMUM_OPERATIONAL_SOC_PCT:g}% across the full planning horizon",
            "operating_optimization_objective": (
                "reserve the feasible CL61 timetable first, then maximize Radar and HATPRO "
                "within the residual reserve; preserve an already-on CL61 through the planning horizon"
            ),
            "operating_schedule_policy": DEFAULT_SCHEDULE_POLICY,
            "operating_available_schedule_policies": json.dumps(
                [SCHEDULE_POLICY_CL61_PRIMARY, SCHEDULE_POLICY_ENERGY_MAXIMISING]
            ),
            "optimized_schedule_policy": DEFAULT_SCHEDULE_POLICY,
            "optimized_priority_order": json.dumps(list(OPERATING_PRIORITY)),
            "optimized_controlled_instruments": json.dumps(list(OPERATING_PRIORITY)),
            "optimized_held_existing_instruments": json.dumps(
                sorted((set(mode_kits(base_mode)) & set(OPERATING_PRIORITY)) - {"CL61"})
            ),
            "cl61_primary_reserved_hours": f"{cl61_primary.collection_hours:.6g}",
            "cl61_primary_minimum_p10_soc": f"{cl61_primary.minimum_p10_soc:.6g}",
            "cl61_primary_safe": str(cl61_primary.safe).lower(),
            "cl61_primary_starts": str(cl61_primary.starts),
            "cl61_primary_continuation_required": str("CL61" in mode_kits(base_mode)).lower(),
            "energy_maximising_comparison_policy": SCHEDULE_POLICY_ENERGY_MAXIMISING,
            "energy_maximising_comparison_status": "available_on_request",
            "optimized_instrument_hours": json.dumps(optimized_instrument_hours, sort_keys=True),
            "optimized_instrument_starts": json.dumps(optimized_instrument_starts, sort_keys=True),
            "optimized_total_instrument_hours": f"{optimized_total_instrument_hours:.6g}",
            "optimized_controlled_energy_kwh": f"{optimized_controlled_energy_kwh:.6g}",
            "optimized_daily_operations": json.dumps(optimized_daily_operations, sort_keys=True),
            "optimized_active_instrument_count_max": str(
                int(np.nanmax(active_instrument_counts[optimized_index]))
            ),
            "minimum_controlled_run_hours": str(MIN_RUN_HOURS),
            "max_controlled_starts_per_utc_day": str(MAX_STARTS_PER_UTC_DAY),
            "optimized_phase_validation_fallback": str(phase_validation_fallback).lower(),
            "optimized_phase_aware_search": "true",
            "optimized_phase_validation_minimum_p10_soc": f"{optimized_phase_minimum_p10:.6g}",
            # Legacy keys remain for consumers that have not yet adopted the
            # three-instrument schedule metadata.
            "minimum_cl61_run_hours": str(MIN_RUN_HOURS),
            "max_cl61_starts_per_utc_day": str(MAX_STARTS_PER_UTC_DAY),
            "control_authority": "advisory_only",
            "p50_continuation_control_authority": "advisory_only",
            "p50_continuation_eligible": str(p50_continuation.eligible).lower(),
            "p50_continuation_status": p50_continuation.status,
            "p50_continuation_reason_code": p50_continuation.reason_code,
            "p50_continuation_reason": p50_continuation.reason,
            "p50_continuation_held_instruments": json.dumps(list(p50_continuation.held_instruments)),
            "p50_continuation_recovery_soc_pct": f"{P50_CONTINUATION_RECOVERY_SOC_PCT:g}",
            "p50_continuation_minimum_soc_pct": f"{P50_CONTINUATION_MINIMUM_SOC_PCT:g}",
            "p50_continuation_recovery_time_utc": (
                p50_continuation.recovery_time.isoformat()
                if p50_continuation.recovery_time is not None
                else ""
            ),
            "p50_continuation_minimum_soc_before_recovery_pct": (
                f"{p50_continuation.minimum_p50_soc_before_recovery:.6g}"
                if np.isfinite(p50_continuation.minimum_p50_soc_before_recovery)
                else ""
            ),
            "p50_continuation_fallback": "existing_p10_priority_schedule",
            "optimized_safe": str(bool(safe_values[optimized_index] >= 0.5)).lower(),
            "optimized_collection_hours": f"{collection_hours[optimized_index]:.6g}",
            "optimized_minimum_p10_soc": f"{minimum_p10[optimized_index]:.6g}",
            "optimized_status": optimized_diagnostic.status,
            "optimized_reason_code": optimized_diagnostic.reason_code,
            "optimized_reason": optimized_diagnostic.reason,
            "optimized_base_mode": optimized_diagnostic.base_mode,
            "optimized_base_mode_label": optimized_diagnostic.base_mode_label,
            "optimized_blocking_instruments": json.dumps(list(optimized_diagnostic.blocking_instruments)),
            "optimized_operator_action_required": str(optimized_diagnostic.operator_action_required).lower(),
            **solar_metadata,
            **battery_model.attrs(),
        },
    )
    for name in (
        "ScenarioSOCP10",
        "ScenarioSOCP50",
        "ScenarioSOCP90",
        "ScenarioMinimumP10SOC",
        "ScenarioFinalP10SOC",
        "ScenarioMinimumP50SOC",
        "ScenarioFinalP50SOC",
    ):
        output[name].attrs["units"] = "%"
    for name in (
        "ScenarioLoadP10Watts",
        "ScenarioLoadP50Watts",
        "ScenarioLoadP90Watts",
        "SolarEnsembleWatts",
        "SolarP10Watts",
        "SolarP50Watts",
        "SolarP90Watts",
        "ComponentLoadWatts",
    ):
        output[name].attrs["units"] = "W"
    output["ScenarioBelow40Probability"].attrs["units"] = "1"
    output["ScenarioActiveInstrumentCount"].attrs["units"] = "count"
    output["ScenarioActiveInstrumentCount"].attrs["description"] = (
        "additive count of scheduled CL61, Radar, and HATPRO states (0 to 3)"
    )
    output["ScenarioUASEffectiveTier"].attrs["units"] = "tier"
    output["ScenarioUASEffectiveTier"].attrs["description"] = (
        "deprecated compatibility alias for ScenarioUASCanonicalTier"
    )
    output["ScenarioUASCanonicalTier"].attrs.update(
        {
            "units": "tier",
            "description": "canonical tier; raw 11 is Tier 1 and raw 12 is Tier 2",
        }
    )
    output["ScenarioUASCharging"].attrs.update(
        {
            "description": "UAS charging load is active for this forecast time",
            "flag_values": "0, 1",
        }
    )
    output["ScenarioPowerState"].attrs["description"] = (
        "composed PDU mode, canonical UAS tier/charge state, and CL61 heater state"
    )
    output["BatteryUsableCapacityKWhEnsemble"].attrs["units"] = "kWh"
    output["BatteryChargeEfficiencyEnsemble"].attrs["units"] = "1"
    output["BatteryDischargeEfficiencyEnsemble"].attrs["units"] = "1"
    output["ScenarioModeCode"].attrs["mode_mapping"] = json.dumps(
        {str(mode_code(value)): mode_label(value) for value in {mode for modes in scenario_modes.values() for mode in modes}},
        sort_keys=True,
    )
    output["ScenarioLoadPhaseCode"].attrs["phase_mapping"] = json.dumps(
        {str(code): name for name, code in PHASE_CODES.items()}, sort_keys=True
    )
    output["ScenarioLoadPhaseEpoch"].attrs["description"] = (
        "increments whenever the operating state or any learned startup/fan phase changes"
    )
    _validate_scenario_invariants(output)
    return output


def evaluate_custom_schedule(
    scenarios: xr.Dataset,
    *,
    start_time: pd.Timestamp,
    duration_hours: int,
    kit: str = "CL61",
) -> dict[str, Any]:
    times = pd.DatetimeIndex(scenarios["time"].values)
    components = tuple(str(value) for value in scenarios["component"].values)
    if components != COMPONENTS:
        raise ValueError("Scenario component schema is not compatible with this model version")
    solar = np.asarray(scenarios["SolarEnsembleWatts"].values, dtype=np.float64)
    component_members = np.asarray(scenarios["ComponentLoadWatts"].values, dtype=np.float64)
    base_mode = str(scenarios.attrs.get("scenario_base_mode", scenarios.attrs.get("current_mode", MODE_DC_ONLY)))
    modes = _schedule_modes(times, base_mode, pd.Timestamp(start_time), int(duration_hours), kit)
    loads = _load_members_for_modes(component_members, modes)
    soc = integrate_soc_members(
        initial_soc=float(scenarios.attrs["initial_soc_pct"]),
        times=times,
        solar_members_w=solar,
        load_members_w=loads,
        capacity_kwh=float(scenarios.attrs["battery_capacity_kwh"]),
        battery_model=(
            BatteryModel.from_attrs(scenarios.attrs)
            if "battery_energy_model" in scenarios.attrs
            else None
        ),
    )
    p10 = np.nanquantile(soc, 0.10, axis=0)
    return {
        "time": times,
        "modes": modes,
        "kit": kit,
        "mode_codes": np.asarray([mode_code(value) for value in modes], dtype=np.int16),
        "load_p50_w": np.nanquantile(loads, 0.50, axis=0),
        "soc_p10": p10,
        "soc_p50": np.nanquantile(soc, 0.50, axis=0),
        "soc_p90": np.nanquantile(soc, 0.90, axis=0),
        "below_40_probability": np.mean(soc < MINIMUM_OPERATIONAL_SOC_PCT, axis=0),
        "collection_hours": float(np.count_nonzero([kit in mode_kits(value) for value in modes[1:]])),
        "minimum_p10_soc": float(np.nanmin(p10)),
        "final_p10_soc": float(p10[-1]),
        "safe": bool(np.nanmin(p10) >= MINIMUM_OPERATIONAL_SOC_PCT),
    }
