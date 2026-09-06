#!/usr/bin/env python3
"""Summary and housekeeping plotting helpers for 1D Aurora instruments.

The helpers in this module define the curated panel layouts, human-readable
labels, static quicklook PNG generation, and interactive Plotly summaries used
by the Meteorology, Radiation, Aurora Power Supply, and Operations views.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import xarray as xr

from quicklook_time_axis import apply_quicklook_time_axis
from power_scenario_catalog import SUGGESTED_OPERATING_SCENARIOS
from power_soc_thresholds import (
    MINIMUM_OPERATIONAL_SOC_LABEL,
    MINIMUM_OPERATIONAL_SOC_PCT,
    MINIMUM_OPERATIONAL_SOC_REFERENCE_LABEL,
    SOC_BELOW_THRESHOLD_BRIER_FIELD,
    SOC_BELOW_THRESHOLD_PROBABILITY_FIELD,
    SOC_REFERENCE_PANEL_KEYS,
)
from time_gap_breaks import insert_time_gap_breaks

MAX_TIME_SAMPLES = int(os.environ.get("AURORA_QUICKLOOK_MAX_TIME_SAMPLES", "2200"))
INTERACTIVE_MAX_TIME_SAMPLES = int(os.environ.get("AURORA_INTERACTIVE_MAX_TIME_SAMPLES", "1600"))
MAX_TIME_TICKS = int(os.environ.get("AURORA_QUICKLOOK_MAX_TIME_TICKS", "16"))
OVERVIEW_LABEL = "Overview"
# Reserve a fixed right-side gutter for per-panel legends so they sit beyond the
# secondary-axis labels in both the interactive Plotly view and saved PNGs.
MATPLOTLIB_PANEL_RIGHT = 0.72
MATPLOTLIB_LEGEND_X = 1.12
PLOTLY_SUMMARY_PANEL_DOMAIN_END = 0.78
PLOTLY_SUMMARY_LEGEND_X = 0.91
PLOTLY_SUMMARY_RIGHT_MARGIN = 110
PLOTLY_SUMMARY_PANEL_HEIGHT = 225
PLOTLY_SUMMARY_POWER_PANEL_HEIGHT = 330
PLOTLY_SUMMARY_MAX_HEIGHT = 1650
PLOTLY_SUMMARY_POWER_MAX_HEIGHT = 7000
PLOTLY_SUMMARY_POWER_PANEL_GAP = 88
MATPLOTLIB_Y_HEADROOM_FRACTION = 0.28
MATPLOTLIB_Y_FOOTROOM_FRACTION = 0.04
SUMMARY_DISPLAY_START_ATTR = "summary_display_start"
SUMMARY_DISPLAY_END_ATTR = "summary_display_end"
POWER_CUMULATIVE_CONTEXT_DAYS = int(os.environ.get("AURORA_POWER_CUMULATIVE_CONTEXT_DAYS", "7"))
POWER_DISPLAY_ENERGY_FREQ = os.environ.get("AURORA_POWER_DISPLAY_ENERGY_FREQ", "1min")
POWER_DISPLAY_SUMMARY_FREQ = os.environ.get("AURORA_POWER_DISPLAY_SUMMARY_FREQ", POWER_DISPLAY_ENERGY_FREQ)
POWER_DISPLAY_ENERGY_ATTR = "power_display_energy_product"
POWER_DISPLAY_SUMMARY_ATTR = "power_display_summary_product"
POWER_FORECAST_DISPLAY_HOURS = float(
    os.environ.get("AURORA_POWER_FORECAST_DISPLAY_HOURS", "96")
)
POWER_SOC_PROJECTION_HOURS = float(os.environ.get("AURORA_POWER_SOC_PROJECTION_HOURS", "24"))
POWER_SOC_PROJECTION_STEP_MINUTES = float(os.environ.get("AURORA_POWER_SOC_PROJECTION_STEP_MINUTES", "5"))
POWER_SOC_PROJECTION_POLY_DEGREE = int(os.environ.get("AURORA_POWER_SOC_PROJECTION_POLY_DEGREE", "1"))
OPERATING_SCENARIO_ANCHOR_TOLERANCE_MINUTES = float(
    os.environ.get("AURORA_OPERATING_SCENARIO_ANCHOR_TOLERANCE_MINUTES", "20")
)
POWER_PANEL_TIME_GROUPS = OrderedDict(
    (
        (
            "observed",
            (
                "renewables",
                "battery_charging",
                "output_power",
                "ass_dc_power",
                "pdu_outlet_power",
                "cumulative_power",
                "output_voltage",
                "thermal_state",
            ),
        ),
        ("forecast_24h", ("soc_projection", "soc_24h_forecast")),
        (
            "forecast_96h",
            (
                "ecmwf_solar_forecast",
                "soc_ecmwf_forecast",
                "operating_plan_scenarios",
                "operating_plan_schedule",
            ),
        ),
        (
            "verification",
            (
                "soc_hindcast",
                "soc_forecast_skill",
                "soc_ensemble_skill",
                "forecast_power_skill",
            ),
        ),
    )
)
POWER_PANEL_TIME_GROUP_BY_KEY = {
    panel_key: group_name
    for group_name, panel_keys in POWER_PANEL_TIME_GROUPS.items()
    for panel_key in panel_keys
}
POWER_PANEL_TIME_AXIS_LABELS = {
    "observed": "Time (UTC)",
    "forecast_24h": "Forecast Time (UTC)",
    "forecast_96h": "Forecast Time (UTC)",
    "verification": "Verification Time (UTC)",
}
PDU_OUTLET_COUNT = 8
PDU_DISPLAY_SUMMARY_FIELDS = tuple(
    f"PDUOutlet{outlet}{metric}"
    for outlet in range(1, PDU_OUTLET_COUNT + 1)
    for metric in ("Watts", "Amps", "State")
)
PDU_WATT_FIELDS = tuple(f"PDUOutlet{outlet}Watts" for outlet in range(1, PDU_OUTLET_COUNT + 1))
PDU_OUTLET_LABELS = {
    4: "UAS",
    5: "CL61",
    6: "Radar",
    8: "HATPRO",
}
# The PDU records all eight physical outlets, but only these outlets are
# assigned to station equipment and belong in the operator-facing plot.
PDU_DISPLAY_OUTLETS = tuple(PDU_OUTLET_LABELS)
POWER_DISPLAY_ENERGY_MAP = {
    "SolarYield_East": "PowerDisplaySolarYield_East",
    "SolarYield_South": "PowerDisplaySolarYield_South",
    "SolarYield_West": "PowerDisplaySolarYield_West",
    "CumulativePowerGeneratedTotal": "PowerDisplayCumulativePowerGeneratedTotal",
    "CumulativePowerUtilised": "PowerDisplayCumulativePowerUtilised",
}
POWER_DISPLAY_SUMMARY_FIELDS = (
    "SolarWatts_East",
    "SolarWatts_South",
    "SolarWatts_West",
    "SolarVolts_East",
    "SolarVolts_South",
    "SolarVolts_West",
    "BatteryAmps",
    "BatteryWatts",
    "ACOutputWatts",
    "DCInverterWatts",
    "BatterySOC",
    "ACOutputVolts",
    "DCInverterVolts",
    "InternalTemperature",
    "HeatsinkTemperature",
    "TempSensor1",
    "TempSensor2",
    "TempSensor3",
    "TempSensor4",
)
POWER_DISPLAY_SUMMARY_CONTEXT_FIELDS = ("watts_on_48vdc_Avg",)
POWER_SOC_FORECAST_FIELDS = (
    "BatterySOCForecast",
    "ECMWFSolarIrradiance",
    "ForecastSolarWatts",
    "ForecastLoadWatts",
    "BatterySOCForecast_Load100W",
    "BatterySOCForecast_Load200W",
    "BatterySOCForecast_Load300W",
    "BatterySOCForecast_Load400W",
    "BatterySOCForecast_Load500W",
    "BatterySOCForecast_Load600W",
)
POWER_SOC_FORECAST_SKILL_FIELDS = (
    "ForecastVerificationSamples",
    "ForecastIndependentCycles",
    "ForecastSOCMAE_0_6h_Verified",
    "ForecastSOCMAE_6_24h_Verified",
    "ForecastSOCMAE_24_48h_Verified",
    "ForecastSOCMAE_48_96h_Verified",
    "ForecastSOCBias_0_6h_Verified",
    "ForecastSOCBias_6_24h_Verified",
    "ForecastSOCBias_24_48h_Verified",
    "ForecastSOCBias_48_96h_Verified",
    "ForecastSOCSkill_0_6h",
    "ForecastLoadMAE24h",
    "ForecastLoadBias24h",
    "ForecastLoadSkill24h",
    "ForecastSolarMAE24h",
    "ForecastSolarBias24h",
    "ForecastSolarSkill24h",
    "ForecastSOCMAESamples_0_6h",
    "ForecastSOCMAESamples_6_24h",
    "ForecastSOCMAESamples_24_48h",
    "ForecastSOCMAESamples_48_96h",
    "ForecastSOCMAECycles_0_6h",
    "ForecastSOCMAECycles_6_24h",
    "ForecastSOCMAECycles_24_48h",
    "ForecastSOCMAECycles_48_96h",
    "ForecastSOCSkill_0_6h",
    "ForecastSOCSkill_6_24h",
    "ForecastSOCSkill_24_48h",
    "ForecastSOCSkill_48_96h",
    "ForecastSOCReadiness_0_6h",
    "ForecastSOCReadiness_6_24h",
    "ForecastSOCReadiness_24_48h",
    "ForecastSOCReadiness_48_96h",
    "ForecastLoadVerificationSamples",
    "ForecastLoadIndependentCycles",
    "ForecastSolarVerificationSamples",
    "ForecastSolarIndependentCycles",
)
POWER_SOC_HINDCAST_FIELDS = (
    "BatterySOCObservedHindcast",
    "BatterySOCHindcast_6h",
    "BatterySOCHindcast_24h",
    "BatterySOCHindcast_48h",
    "BatterySOCHindcast_72h",
)
SOC_HINDCAST_EXPLANATION = (
    "Solid green is measured battery state of charge. Broken lines are archived forecasts "
    "for the same plotted time; each label states how long before that time the forecast was issued."
)
POWER_SOC_ENSEMBLE_FORECAST_FIELDS = (
    "BatterySOCForecastP10",
    "BatterySOCForecastP50",
    "BatterySOCForecastP90",
    "BatterySOCForecastMinimum",
    "BatterySOCForecastMaximum",
    SOC_BELOW_THRESHOLD_PROBABILITY_FIELD,
)
SYSTEM_AS_IS_DECISION_FIELD_SOURCES = OrderedDict(
    (
        ("SystemAsIsDecisionSOCP10", ("OperatingCurrentSOCP10", ("BatterySOCForecastP10",))),
        (
            "SystemAsIsDecisionSOCP50",
            ("OperatingCurrentSOCP50", ("BatterySOCForecastP50", "BatterySOCForecast")),
        ),
        ("SystemAsIsDecisionSOCP90", ("OperatingCurrentSOCP90", ("BatterySOCForecastP90",))),
        (
            "SystemAsIsDecisionBelow40Probability",
            ("OperatingCurrentBelow40Probability", (SOC_BELOW_THRESHOLD_PROBABILITY_FIELD,)),
        ),
    )
)
SYSTEM_AS_IS_DECISION_FIELDS = tuple(SYSTEM_AS_IS_DECISION_FIELD_SOURCES)
POWER_SOC_ENSEMBLE_SKILL_FIELDS = (
    "ForecastSOCCRPS_0_6h",
    "ForecastSOCCRPS_6_24h",
    "ForecastSOCCRPS_24_48h",
    "ForecastSOCCRPS_48_96h",
    "ForecastSOCIntervalCoverage80",
    SOC_BELOW_THRESHOLD_BRIER_FIELD,
    "ForecastEnsembleCycles",
    "ForecastSOCCRPSSkill_0_6h",
    "ForecastSOCCRPSSkill_6_24h",
    "ForecastSOCCRPSSkill_24_48h",
    "ForecastSOCCRPSSkill_48_96h",
    "ForecastSOCCRPSSamples_0_6h",
    "ForecastSOCCRPSSamples_6_24h",
    "ForecastSOCCRPSSamples_24_48h",
    "ForecastSOCCRPSSamples_48_96h",
    "ForecastSOCCRPSCycles_0_6h",
    "ForecastSOCCRPSCycles_6_24h",
    "ForecastSOCCRPSCycles_24_48h",
    "ForecastSOCCRPSCycles_48_96h",
    "ForecastSOCIntervalCoverage80Samples",
    "ForecastSOCIntervalCoverage80Cycles",
    f"{SOC_BELOW_THRESHOLD_BRIER_FIELD}Reference",
    f"{SOC_BELOW_THRESHOLD_BRIER_FIELD}Skill",
    f"{SOC_BELOW_THRESHOLD_BRIER_FIELD}Samples",
    f"{SOC_BELOW_THRESHOLD_BRIER_FIELD}Cycles",
)
OPERATING_SCENARIO_PREFIXES = OrderedDict(
    (
        ("current_mode", "OperatingCurrent"),
        ("dc_only", "OperatingDCOnly"),
        ("cl61_continuous", "OperatingCL61Continuous"),
        ("optimized_cl61", "OperatingCL61Optimized"),
        ("p50_continuation", "OperatingP50Continuation"),
    )
)
OPERATING_SUGGESTED_PREFIXES = OrderedDict(
    (definition.scenario_id, f"OperatingSuggested{index}")
    for index, definition in enumerate(SUGGESTED_OPERATING_SCENARIOS, start=1)
)
MAX_OPERATING_LEARNED_SCENARIOS = 6
OPERATING_LEARNED_PREFIXES = tuple(
    f"OperatingLearned{index}" for index in range(1, MAX_OPERATING_LEARNED_SCENARIOS + 1)
)
OPERATING_SCENARIO_SOURCE_FIELDS = (
    ("ScenarioSOCP10", "SOCP10"),
    ("ScenarioSOCP50", "SOCP50"),
    ("ScenarioSOCP90", "SOCP90"),
    ("ScenarioLoadP10Watts", "LoadP10Watts"),
    ("ScenarioLoadP50Watts", "LoadP50Watts"),
    ("ScenarioLoadP90Watts", "LoadP90Watts"),
    ("ScenarioBelow40Probability", "Below40Probability"),
    ("ScenarioModeCode", "ModeCode"),
    ("ScenarioActiveInstrumentCount", "ActiveCount"),
)
OPERATING_SCENARIO_DISPLAY_FIELDS = tuple(
    f"{prefix}{suffix}"
    for prefix in (
        tuple(OPERATING_SCENARIO_PREFIXES.values())
        + tuple(OPERATING_SUGGESTED_PREFIXES.values())
        + OPERATING_LEARNED_PREFIXES
    )
    for _source, suffix in OPERATING_SCENARIO_SOURCE_FIELDS
) + (
    "OperatingCL61OptimizedCL61On",
    "OperatingCL61OptimizedRadarOn",
    "OperatingCL61OptimizedHATPROOn",
    "OperatingCL61OptimizedActiveCount",
    "OperatingSolarP10Watts",
    "OperatingSolarP50Watts",
    "OperatingSolarP90Watts",
)
POWER_FUTURE_DISPLAY_FIELDS = tuple(
    dict.fromkeys(
        POWER_SOC_FORECAST_FIELDS
        + POWER_SOC_ENSEMBLE_FORECAST_FIELDS
        + SYSTEM_AS_IS_DECISION_FIELDS
        + OPERATING_SCENARIO_DISPLAY_FIELDS
    )
)
OPERATING_MODE_BITS = (
    ("CL61", 1, "rgba(192, 86, 71, 0.16)"),
    ("Radar", 2, "rgba(63, 109, 181, 0.16)"),
    ("HATPRO", 4, "rgba(134, 91, 170, 0.16)"),
    ("UAS", 8, "rgba(211, 138, 55, 0.16)"),
)
OPERATING_SCHEDULE_SHADE_PANELS = {"ecmwf_solar_forecast"}
FAST_SONIC_TO_LOGGER_AVG = {
    "metek_x_out": "metek_x_out_Avg",
    "metek_y_out": "metek_y_out_Avg",
    "metek_z_out": "metek_z_out_Avg",
    "metek_T_out": "metek_T_out_Avg",
    "metek_InclX_out": "metek_InclX_out_Avg",
    "metek_InclY_out": "metek_InclY_out_Avg",
}
FAST_GAS_TO_LOGGER_AVG = {
    "licor_co2_out": "licor_co2_out_Avg",
    "licor_h2o_out": "licor_h2o_out_Avg",
    "licor_pr_out": "licor_pr_out_Avg",
    "licor_t_out": "licor_t_out_Avg",
    "licor_diag_out": "licor_diag_out_Avg",
    "licor_co2_str_out": "licor_co2_str_out_Avg",
}


@dataclass(frozen=True)
class TraceSpec:
    var: str
    label: str
    color: str
    axis: str = "left"
    scale: float = 1.0
    dash: str | None = None
    step: bool = False
    valid_min: float | None = None
    valid_max: float | None = None
    skip_if_all_zero: bool = False
    smooth_minutes: float | None = None
    break_on_day_change: bool = False
    projection_lookback_minutes: float | None = None
    projection_horizon_hours: float = POWER_SOC_PROJECTION_HOURS
    projection_degree: int = POWER_SOC_PROJECTION_POLY_DEGREE
    display_horizon_hours: float | None = None
    line_width: float = 2.0
    opacity: float = 1.0


@dataclass(frozen=True)
class PanelSpec:
    key: str
    label: str
    left_axis_label: str
    right_axis_label: str | None
    traces: tuple[TraceSpec, ...]
    description: str | None = None
    display_horizon_hours: float | None = None


@dataclass(frozen=True)
class CL61SchedulePresentation:
    status: str
    title: str
    trace_label: str
    annotation: str
    summary: str
    explanation: str


def _schedule_attr(ds: xr.Dataset | None, name: str) -> str:
    if ds is None:
        return ""
    for key in (f"operating_{name}", name):
        value = str(ds.attrs.get(key, "")).strip()
        if value:
            return value
    return ""


def cl61_schedule_presentation(ds: xr.Dataset | None) -> CL61SchedulePresentation:
    """Describe legacy CL61 or current priority plans without overstating safety."""
    status = _schedule_attr(ds, "optimized_status")
    priority_order = _schedule_attr(ds, "optimized_priority_order")
    priority_plan = bool(priority_order)
    cl61_primary = _schedule_attr(ds, "optimized_schedule_policy") == "cl61_primary_v1"
    continuation_required = _schedule_attr(ds, "cl61_primary_continuation_required").lower() == "true"
    safe_text = _schedule_attr(ds, "optimized_safe").lower()
    collection_text = _schedule_attr(ds, "optimized_collection_hours")
    try:
        collection_hours = float(collection_text)
    except ValueError:
        collection_hours = float("nan")
    if not status:
        if safe_text == "false":
            status = "no_safe_schedule"
        elif safe_text == "true" and np.isfinite(collection_hours):
            status = "safe_schedule" if collection_hours > 0.0 else "reserve_only"
        else:
            status = "unknown"

    reason = _schedule_attr(ds, "optimized_reason")
    base_label = _schedule_attr(ds, "optimized_base_mode_label") or "fixed station load"
    if status == "no_safe_schedule":
        explanation = reason or (
            (
                "The existing CL61 state is held in this advisory forecast, but the operating SOC reserve is not met. "
                "This is not an instruction to switch CL61 off; operator review is required."
            )
            if continuation_required
            else (
                f"Even with CL61 off, the {base_label} baseline falls below the operational SOC reserve. "
                "The zero trace is an unsafe fallback, not an instruction to switch CL61 off. Operator review is required."
            )
        )
        return CL61SchedulePresentation(
            status=status,
            title=(
                "No Feasible CL61-first Schedule"
                if cl61_primary
                else "No Feasible Instrument Schedule"
                if priority_plan
                else "No Feasible CL61 Schedule"
            ),
            trace_label=(
                "Unsafe held CL61 continuation"
                if continuation_required
                else "Unsafe fallback (CL61 off)"
            ),
            annotation=(
                "Existing CL61 held; operator review required"
                if continuation_required
                else "No safe instrument plan; operator review required"
                if priority_plan
                else "No safe CL61-only plan; operator review required"
            ),
            summary=(
                "The CL61-first scheduler preserves the existing CL61 state but cannot preserve the operating SOC reserve."
                if continuation_required
                else "The CL61-first scheduler cannot preserve the operating SOC reserve."
                if cl61_primary
                else "The priority scheduler could not satisfy the SOC reserve even with its controlled instruments off."
                if priority_plan
                else "The CL61-only optimiser could not satisfy the SOC reserve under the fixed current loads."
            ),
            explanation=explanation,
        )
    if status == "reserve_only":
        explanation = reason or (
            "No CL61 collection interval satisfies the operational SOC reserve while the other current loads remain fixed."
        )
        return CL61SchedulePresentation(
            status=status,
            title=(
                "CL61-first Reserve-Only Plan"
                if cl61_primary
                else "Additive Instrument Reserve-Only Plan"
                if priority_plan
                else "CL61 Reserve-Only Plan"
            ),
            trace_label="CL61 remains off",
            annotation=(
                "No safe controlled-instrument window"
                if priority_plan
                else "No safe CL61 collection window"
            ),
            summary=(
                "No safe CL61-first collection window is available, so the advisory plan holds the reserve."
                if cl61_primary
                else "Keeping CL61, Radar, and HATPRO off is the reserve-preserving advisory plan."
                if priority_plan
                else "Keeping CL61 off is the only reserve-preserving CL61 plan under the fixed current loads."
            ),
            explanation=explanation,
        )
    if status == "safe_schedule":
        explanation = reason or (
            (
                "The priority scheduler found an advisory CL61, Radar, and HATPRO timetable "
                "that satisfies the operational SOC reserve."
            )
            if priority_plan
            else "The optimiser found a CL61 collection schedule that satisfies the operational SOC reserve while other instrument states remain fixed."
        )
        return CL61SchedulePresentation(
            status=status,
            title=(
                "Recommended CL61-first Instrument Schedule"
                if cl61_primary
                else "Recommended Additive Instrument Schedule"
                if priority_plan
                else "Recommended CL61 Collection Schedule"
            ),
            trace_label="Recommended CL61 schedule",
            annotation=(
                "No controlled-instrument interval selected"
                if priority_plan
                else "No CL61 collection interval selected"
            ),
            summary=(
                "CL61 is reserved first; Radar and HATPRO use only the remaining safe reserve."
                if cl61_primary
                else "The power-maximising additive on/off timetable; CL61, Radar, then HATPRO break ties."
                if priority_plan
                else "The on/off timetable for the feasible advisory CL61 plan."
            ),
            explanation=explanation,
        )
    return CL61SchedulePresentation(
        status="unknown",
        title="CL61 Collection Schedule",
        trace_label="CL61 schedule",
        annotation="No CL61 collection interval in this plan",
        summary="The CL61 schedule status is not available in this forecast product.",
        explanation="Regenerate the operating-plan product before interpreting a zero schedule as operational advice.",
    )


def power_trace_label(ds: xr.Dataset, trace: TraceSpec) -> str:
    """Return a trace label that reflects CL61 plan feasibility."""
    if trace.var == "ForecastLoadWatts":
        mode = str(ds.attrs.get("forecast_load_mode", ds.attrs.get("load_mode", ""))).strip()
        if mode:
            return f"Forecast Load ({mode})"
    if trace.var == "OperatingCurrentSOCP50":
        mode = str(ds.attrs.get("operating_current_mode_label", "")).strip()
        if mode:
            maturity = str(ds.attrs.get("operating_current_mode_maturity", "")).strip()
            suffix = f" ({maturity})" if maturity else ""
            return f"Current load / system as-is: {mode}{suffix}"
    if trace.var == "OperatingCL61OptimizedActiveCount":
        return "Total active instruments (additive sum)"
    if trace.var == "OperatingP50ContinuationSOCP50":
        eligible = _schedule_attr(ds, "p50_continuation_eligible").lower() == "true"
        instruments_text = _schedule_attr(ds, "p50_continuation_held_instruments")
        try:
            instruments = [str(value) for value in json.loads(instruments_text)]
        except (TypeError, ValueError, json.JSONDecodeError):
            instruments = []
        if eligible and instruments:
            return f"P50 continuation: keep {' + '.join(instruments)} on to median recovery"
        return "P50 continuation unavailable: conservative P10 plan"
    if trace.var in {
        "OperatingCL61OptimizedCL61On",
        "OperatingCL61OptimizedRadarOn",
        "OperatingCL61OptimizedHATPROOn",
    }:
        instrument = trace.var.removeprefix("OperatingCL61Optimized").removesuffix("On")
        presentation = cl61_schedule_presentation(ds)
        try:
            held_instruments = {
                str(value) for value in json.loads(_schedule_attr(ds, "optimized_held_existing_instruments"))
            }
        except (TypeError, ValueError, json.JSONDecodeError):
            held_instruments = set()
        if instrument == "CL61":
            return presentation.trace_label
        if instrument in held_instruments:
            return f"Observed {instrument} held (not scheduled)"
        if presentation.status == "no_safe_schedule":
            return f"Unsafe fallback ({instrument} off)"
        if presentation.status == "reserve_only":
            return f"{instrument} remains off"
        return f"Recommended {instrument} schedule"
    if trace.var == "OperatingCL61OptimizedLoadP50Watts":
        priority_plan = bool(_schedule_attr(ds, "optimized_priority_order"))
        presentation = cl61_schedule_presentation(ds)
        status = presentation.status
        held_cl61 = _schedule_attr(ds, "cl61_primary_continuation_required").lower() == "true"
        if status == "no_safe_schedule":
            if held_cl61:
                return "Unsafe held-CL61 continuation load"
            return (
                "Unsafe all-controlled-off fallback load"
                if priority_plan
                else "Unsafe fallback load (CL61 off)"
            )
        if status == "reserve_only":
            return (
                "Reserve-only controlled-instrument load"
                if priority_plan
                else "Reserve-only load (CL61 off)"
            )
    if trace.var == "OperatingCL61OptimizedSOCP50":
        priority_plan = bool(_schedule_attr(ds, "optimized_priority_order"))
        presentation = cl61_schedule_presentation(ds)
        status = presentation.status
        held_cl61 = _schedule_attr(ds, "cl61_primary_continuation_required").lower() == "true"
        if status == "no_safe_schedule":
            if held_cl61:
                return "Unsafe held-CL61 continuation"
            return (
                "Unsafe all-controlled-off fallback"
                if priority_plan
                else "Unsafe CL61-off fallback"
            )
        if status == "reserve_only":
            return (
                "Reserve-only controlled-instrument plan"
                if priority_plan
                else "Reserve-only CL61-off plan"
            )
    for index, prefix in enumerate(OPERATING_LEARNED_PREFIXES, start=1):
        if trace.var == f"{prefix}SOCP50":
            mode = str(ds.attrs.get(f"operating_learned_{index}_label", "")).strip()
            if mode:
                return f"Comparison: {mode}"
    return trace.label


VERIFICATION_MIN_SAMPLES = 20
VERIFICATION_MIN_CYCLES = 30


def _latest_finite_value(ds: xr.Dataset, name: str) -> float:
    if name not in ds:
        return float("nan")
    values = np.asarray(ds[name].values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    return float(values[-1]) if values.size else float("nan")


def _verification_evidence(samples: float, cycles: float) -> tuple[str, str]:
    if not np.isfinite(samples) or samples <= 0:
        return ("Not yet verified", "learning")
    if not np.isfinite(cycles) or samples < VERIFICATION_MIN_SAMPLES or cycles < VERIFICATION_MIN_CYCLES:
        cycle_text = "unknown" if not np.isfinite(cycles) else str(int(round(cycles)))
        return (f"Learning: {int(round(samples))} samples, {cycle_text} ECMWF cycles", "learning")
    return (f"{int(round(samples))} samples, {int(round(cycles))} independent ECMWF cycles", "neutral")


def _skill_status(skill: float, evidence_status: str) -> tuple[str, str]:
    if evidence_status == "learning":
        return ("Insufficient evidence", "learning")
    if not np.isfinite(skill):
        return ("No usable persistence baseline", "neutral")
    if skill > 0.0:
        return ("Better than persistence", "good")
    return ("Not better than persistence", "caution")


def _wilson_interval(successes: float, samples: float, *, z: float = 1.96) -> tuple[float, float]:
    if samples <= 0:
        return (float("nan"), float("nan"))
    proportion = successes / samples
    denominator = 1.0 + z**2 / samples
    centre = (proportion + z**2 / (2.0 * samples)) / denominator
    margin = z * math.sqrt((proportion * (1.0 - proportion) + z**2 / (4.0 * samples)) / samples) / denominator
    return (centre - margin, centre + margin)


def build_power_verification_guidance(panel_key: str, ds: xr.Dataset) -> dict[str, object] | None:
    """Build compact, common score guidance for browser and native Power cards."""
    if panel_key == "soc_forecast_skill":
        metrics = []
        for bucket, label in (("0_6h", "0-6 h"), ("6_24h", "6-24 h"), ("24_48h", "24-48 h"), ("48_96h", "48-96 h")):
            value = _latest_finite_value(ds, f"ForecastSOCMAE_{bucket}_Verified")
            bias = _latest_finite_value(ds, f"ForecastSOCBias_{bucket}_Verified")
            samples = _latest_finite_value(ds, f"ForecastSOCMAESamples_{bucket}")
            cycles = _latest_finite_value(ds, f"ForecastSOCMAECycles_{bucket}")
            evidence, evidence_level = _verification_evidence(samples, cycles)
            status, level = _skill_status(_latest_finite_value(ds, f"ForecastSOCSkill_{bucket}"), evidence_level)
            ready = _latest_finite_value(ds, f"ForecastSOCReadiness_{bucket}")
            if evidence_level != "learning" and np.isfinite(ready):
                status, level = ("Meets 10-point target", "good") if ready >= 0.5 else ("Outside 10-point target", "caution")
            metrics.append(
                {
                    "id": f"soc-mae-{bucket}",
                    "label": f"SOC MAE {label}",
                    "valueText": "Not yet verified" if not np.isfinite(value) else f"{value:.2f} percentage points",
                    "direction": "Lower is better",
                    "reference": (
                        f"Bias {'not yet verified' if not np.isfinite(bias) else f'{bias:+.2f} points'}; "
                        "target: MAE and absolute bias below 10 points"
                    ),
                    "status": status,
                    "evidence": evidence,
                    "level": level,
                }
            )
        return {
            "title": "How to read SOC verification",
            "summary": "Average absolute miss and signed error of immutable issue-time SOC forecasts against later APS measurements. Certification requires 30 independent ECMWF cycles.",
            "metrics": metrics,
        }

    if panel_key == "soc_ensemble_skill":
        metrics = []
        for bucket, label in (("0_6h", "0-6 h"), ("6_24h", "6-24 h"), ("24_48h", "24-48 h"), ("48_96h", "48-96 h")):
            value = _latest_finite_value(ds, f"ForecastSOCCRPS_{bucket}")
            samples = _latest_finite_value(ds, f"ForecastSOCCRPSSamples_{bucket}")
            cycles = _latest_finite_value(ds, f"ForecastSOCCRPSCycles_{bucket}")
            evidence, evidence_level = _verification_evidence(samples, cycles)
            status, level = _skill_status(_latest_finite_value(ds, f"ForecastSOCCRPSSkill_{bucket}"), evidence_level)
            metrics.append(
                {
                    "id": f"soc-crps-{bucket}",
                    "label": f"SOC CRPS {label}",
                    "valueText": "Not yet verified" if not np.isfinite(value) else f"{value:.2f} percentage points",
                    "direction": "Lower is better",
                    "reference": "Scores ensemble centre and uncertainty against persistence",
                    "status": status,
                    "evidence": evidence,
                    "level": level,
                }
            )
        coverage = _latest_finite_value(ds, "ForecastSOCIntervalCoverage80")
        coverage_samples = _latest_finite_value(ds, "ForecastSOCIntervalCoverage80Samples")
        coverage_cycles = _latest_finite_value(ds, "ForecastSOCIntervalCoverage80Cycles")
        evidence, evidence_level = _verification_evidence(coverage_samples, coverage_cycles)
        if evidence_level == "learning":
            coverage_status, coverage_level = ("Insufficient evidence", "learning")
        elif np.isfinite(coverage):
            lower, upper = _wilson_interval(coverage * coverage_samples, coverage_samples)
            if upper < 0.8:
                coverage_status, coverage_level = ("Too narrow / overconfident", "caution")
            elif lower > 0.8:
                coverage_status, coverage_level = ("Too wide / underconfident", "caution")
            else:
                coverage_status, coverage_level = ("Consistent with 80% target", "good")
        else:
            coverage_status, coverage_level = ("Not yet verified", "learning")
        metrics.append(
            {
                "id": "soc-coverage",
                "label": "P10-P90 coverage",
                "valueText": "Not yet verified" if not np.isfinite(coverage) else f"{coverage:.2f}",
                "direction": "Target is 0.80",
                "reference": "Observed SOC should fall inside the central 80% ensemble interval about 80% of the time",
                "status": coverage_status,
                "evidence": evidence,
                "level": coverage_level,
            }
        )
        brier = _latest_finite_value(ds, SOC_BELOW_THRESHOLD_BRIER_FIELD)
        brier_samples = _latest_finite_value(ds, f"{SOC_BELOW_THRESHOLD_BRIER_FIELD}Samples")
        brier_cycles = _latest_finite_value(ds, f"{SOC_BELOW_THRESHOLD_BRIER_FIELD}Cycles")
        evidence, evidence_level = _verification_evidence(brier_samples, brier_cycles)
        brier_status, brier_level = _skill_status(
            _latest_finite_value(ds, f"{SOC_BELOW_THRESHOLD_BRIER_FIELD}Skill"), evidence_level
        )
        metrics.append(
            {
                "id": "soc-brier",
                "label": f"Below {MINIMUM_OPERATIONAL_SOC_LABEL} Brier score",
                "valueText": "Not yet verified" if not np.isfinite(brier) else f"{brier:.3f}",
                "direction": "Lower is better; 0 is perfect",
                "reference": "Compared with predicting the recent below-threshold event frequency",
                "status": brier_status,
                "evidence": evidence,
                "level": brier_level,
            }
        )
        return {
            "title": "How to read ensemble verification",
            "summary": "CRPS checks both the likely SOC and the honesty of ensemble uncertainty. Coverage and Brier check calibration and the 40% safety-risk probability.",
            "metrics": metrics,
        }

    if panel_key == "forecast_power_skill":
        metrics = []
        for kind, label, sample_name, cycle_name in (
            ("Solar", "Solar forecast", "ForecastSolarVerificationSamples", "ForecastSolarIndependentCycles"),
            ("Load", "Station load forecast", "ForecastLoadVerificationSamples", "ForecastLoadIndependentCycles"),
        ):
            value = _latest_finite_value(ds, f"Forecast{kind}MAE24h")
            bias = _latest_finite_value(ds, f"Forecast{kind}Bias24h")
            samples = _latest_finite_value(ds, sample_name)
            cycles = _latest_finite_value(ds, cycle_name)
            evidence, evidence_level = _verification_evidence(samples, cycles)
            status, level = _skill_status(_latest_finite_value(ds, f"Forecast{kind}Skill24h"), evidence_level)
            bias_text = "not yet verified" if not np.isfinite(bias) else f"{bias:+.1f} W"
            metrics.append(
                {
                    "id": f"{kind.lower()}-mae",
                    "label": f"{label} MAE",
                    "valueText": "Not yet verified" if not np.isfinite(value) else f"{value:.1f} W",
                    "direction": "Lower is better",
                    "reference": f"Bias {bias_text}; positive means forecast exceeds observed",
                    "status": status,
                    "evidence": evidence,
                    "level": level,
                }
            )
        return {
            "title": "How to read solar and load verification",
            "summary": "Solar is forecast panel charging. Station load is solar generation minus APS battery flow when available, including the main DC load; AC plus inverter power is the fallback.",
            "metrics": metrics,
        }
    return None


def build_power_forecast_info(panel_key: str, ds: xr.Dataset | None = None) -> dict[str, object] | None:
    """Return the implementation note shown by every Power forecast plot.

    This deliberately describes the operational meaning of each panel rather
    than repeating its trace labels.  It is kept here so the Panel browser and
    native mobile API cannot drift apart.
    """
    minimum = MINIMUM_OPERATIONAL_SOC_LABEL
    notes: dict[str, dict[str, object]] = {
        "soc_projection": {
            "title": "SOC projection",
            "summary": "A short display-only extrapolation of the latest measured battery state of charge.",
            "implementation": "Two low-degree polynomial fits use the latest 30 minutes and latest 2 hours of APS SOC, then extend for 24 hours. It is a trend indicator, not the weather-informed operating forecast.",
            "metrics": [
                {"label": "30 min / 2 h fits", "detail": "Short recent trends; they can diverge when the battery behaviour changes."},
                {"label": "Use", "detail": "Immediate trend context only; use the ECMWF forecast for planning."},
            ],
        },
        "soc_24h_forecast": {
            "title": "SOC next 24 h forecast",
            "summary": "The near-term weather-informed SOC outlook used for immediate station context.",
            "implementation": "It starts from the latest valid APS SOC and propagates forecast solar charging against the latest confirmed finite instrument state and its currently detected load phase. That state is held fixed unless an explicit operating schedule changes it. This is the central forecast only; uncertainty is shown in the 96-hour ensemble panel.",
            "metrics": [
                {"label": "Starting point", "detail": "Latest valid APS battery SOC."},
                {"label": "Load assumption", "detail": "Latest confirmed instrument state and sustained within-state load phase, held fixed."},
            ],
        },
        "soc_ecmwf_forecast": {
            "title": "SOC 96 h forecast",
            "summary": "The same 96-hour system-as-is forecast used as the reference in the instrument-scenario comparison.",
            "implementation": "Every member starts from one latest valid APS SOC observation and one latest confirmed finite instrument state. The controllable state is fixed; the load distribution represents its current sustained phase plus only recurrent startup or fan phases learned within that same state. The ensemble combines that within-state load uncertainty with ECMWF solar weather and bounded battery-capacity and efficiency uncertainty. The displayed P10, median, P90, and threshold risk are all from this single re-anchored decision snapshot.",
            "metrics": [
                {"label": "P10", "detail": "Lower-end SOC outcome across ECMWF solar members; use it for conservative planning."},
                {"label": "Central / P90", "detail": "Central estimate and upper-end solar outcome for the same system-as-is load."},
                {"label": "Load uncertainty", "detail": "Only recurrent startup and fan behaviour within the confirmed instrument state; it never blends different controllable states."},
                {"label": f"Probability below {minimum}", "detail": f"The fraction of ensemble members below the {minimum} operational reference at each time."},
            ],
        },
        "soc_hindcast": {
            "title": "Measured SOC versus earlier forecasts",
            "summary": "A retrospective check, not a new prediction.",
            "implementation": "Each dashed trace is an archived forecast issued 6, 24, 48, or 72 hours before its valid time. It is compared with the later APS measurement to expose timing or bias errors.",
            "metrics": [
                {"label": "Measured SOC", "detail": "Later APS observation used as the verification reference."},
                {"label": "Lead time", "detail": "How far in advance that archived forecast was issued."},
            ],
        },
        "operating_plan_scenarios": {
            "title": "Suggested instrument-mode SOC forecasts",
            "summary": "The current exact instrument state is the reference, with a P50 continuation recommendation and eight explicit instrument combinations, including all instruments with UAS at tier 3.",
            "implementation": "The system-as-is line uses the latest confirmed finite instrument state and its currently detected sustained load phase. The P50 continuation line holds only controlled instruments that are already on until median SOC next reaches 95%, but only when median SOC stays at or above 40% before that recovery; otherwise it follows the conservative P10 priority timetable. It never starts an instrument. Each fixed comparison line uses the learned load distribution for exactly the named state; states are never blended. Recurrent startup and fan phases add uncertainty only within their own state. The UAS tier-3 load is learned separately and remains provisional until at least three independent episodes and six observed hours are available. All lines share the forecast issue, initial SOC, solar ensemble, and calibrated battery assumptions. They are advisory and never operate PDU outlets.",
            "metrics": [
                {"label": "P50", "detail": "The median SOC path for each named instrument combination."},
                {"label": "Current state", "detail": "Reference SOC path if the latest confirmed instrument state and detected load phase continue."},
                {"label": "P50 continuation", "detail": "Keeps only currently-on controlled instruments on through the next 95% median-SOC recovery when the path first stays above the 40% floor. It is a scenario, not control."},
                {"label": "Common basis", "detail": "Every line starts from the same measured SOC and uses the same weather forecast."},
                {"label": "State loads", "detail": "The learned distribution for each exact named state, including recurrent startup or fan phases where supported."},
                {"label": "UAS tier 3", "detail": "Tier-specific load estimate; provisional until the minimum independent evidence gate is met."},
            ],
        },
        "operating_plan_schedule": {
            "title": "Additive instrument operating schedule",
            "summary": "The advisory additive schedule and whether its SOC constraint is feasible.",
            "implementation": "Each instrument trace is 1 when selected and 0 when off; the total trace is their additive sum from 0 to 3. Stored UTC-day summaries make each instrument's on-hours plus off-hours equal the available hours in that day. The joint scheduler maximises controlled energy first, then total instrument-hours. CL61, then Radar, then HATPRO breaks otherwise-equivalent plans. It includes learned startup and fan phases while searching, requires full-horizon P10 SOC at or above the operational reserve, 12-hour minimum runs, and no more than one scheduled start per UTC day. UAS and the DC baseline are not controlled. The plan is advisory and never operates PDU outlets.",
            "metrics": [
                {"label": "1 / 0", "detail": "Each instrument selected on / off; always read this with the feasibility status."},
                {"label": "Additive sum", "detail": "Total active instruments is CL61 + Radar + HATPRO at each hour (0 to 3). Integrating it gives daily instrument-hours, up to 72 per complete UTC day."},
                {"label": "Objective", "detail": "Maximise safe controlled energy, then total instrument-hours; CL61 > Radar > HATPRO is the tie-break."},
                {"label": "Constraint", "detail": f"P10 SOC is held at or above the {minimum} operational reference over the planning horizon."},
            ],
        },
        "ecmwf_solar_forecast": {
            "title": "ECMWF solar and load forecast",
            "summary": "The weather input and electrical assumptions used by the SOC forecast and operating plans.",
            "implementation": "ECMWF irradiance is converted to expected solar charging using the currently learned station conversion. The current-load trace holds the detected system configuration fixed; the recommended-instrument load trace applies the priority scheduler result and must be read with its feasibility status.",
            "metrics": [
                {"label": "ECMWF solar", "detail": "Forecast irradiance from the meteorological ensemble input."},
                {"label": "Solar charging", "detail": "Estimated battery charging after the learned solar conversion."},
                {"label": "Load traces", "detail": "Current system-as-is load versus the prioritized CL61, Radar, and HATPRO plan. An unsafe fallback is diagnostic, not operational advice."},
            ],
        },
        "soc_forecast_skill": {
            "title": "SOC forecast verification",
            "summary": "How closely archived SOC forecasts matched later APS measurements.",
            "implementation": "The rolling 24-hour verification groups archived forecasts by lead time and calculates mean absolute error (MAE) against later observed SOC. It compares each forecast with persistence from the issue-time SOC.",
            "metrics": [
                {"label": "MAE", "detail": "Mean absolute error in SOC percentage points; lower is better."},
                {"label": "Lead buckets", "detail": "0-6, 6-24, 24-48, and 48-96 hours separate short- and longer-range performance."},
                {"label": "Independent cycles", "detail": "Number of distinct ECMWF forecast cycles contributing evidence."},
            ],
        },
        "soc_ensemble_skill": {
            "title": "SOC ensemble verification",
            "summary": "Whether the ensemble is accurate and whether its stated uncertainty is appropriately wide.",
            "implementation": "CRPS evaluates the ensemble centre and spread against later APS SOC. P10-P90 coverage checks whether the central 80% interval contains observations about 80% of the time; Brier score checks the forecast probability of dropping below the operational minimum.",
            "metrics": [
                {"label": "CRPS", "detail": "Combined accuracy and uncertainty score in SOC percentage points; lower is better."},
                {"label": "P10-P90 coverage", "detail": "Target is 0.80; lower is overconfident and higher is underconfident."},
                {"label": "Brier score", "detail": f"Probability error for SOC below {minimum}; 0 is perfect and lower is better."},
            ],
        },
        "forecast_power_skill": {
            "title": "Solar and load forecast verification",
            "summary": "How well the electrical inputs used by the SOC forecast matched later station data.",
            "implementation": "Solar MAE compares forecast charging with observed charging. Load MAE compares the station load model with the APS power balance, including the main DC load where available; AC plus inverter power is the fallback. Scores are separated by load-model version.",
            "metrics": [
                {"label": "MAE", "detail": "Average absolute forecast miss in watts; lower is better."},
                {"label": "Bias", "detail": "Forecast minus observed; positive solar means too much charging forecast, positive load means too much consumption forecast."},
                {"label": "Versioned evidence", "detail": "Load metrics only compare rows produced by the same load-model version."},
            ],
        },
    }
    info = notes.get(panel_key)
    if info is None:
        return None
    if panel_key == "operating_plan_schedule" and ds is not None:
        presentation = cl61_schedule_presentation(ds)
        metrics = list(info["metrics"])
        cl61_primary = _schedule_attr(ds, "optimized_schedule_policy") == "cl61_primary_v1"
        implementation = str(info["implementation"])
        if cl61_primary:
            implementation = (
                "Each instrument trace is 1 when selected and 0 when off; the total trace is "
                "their additive sum from 0 to 3. The scheduler first reserves the feasible CL61 "
                "timetable, then adds Radar and HATPRO only from residual reserve. It includes "
                "learned startup and fan phases, requires full-horizon P10 SOC at or above the "
                "operational reserve, uses 12-hour minimum runs, and permits no more than one "
                "scheduled start per instrument per UTC day. The plan is advisory and never operates "
                "PDU outlets."
            )
            for metric in metrics:
                if metric.get("label") == "Objective":
                    metric["detail"] = "Reserve CL61 first; Radar and HATPRO may use only remaining safe reserve."
        total_hours = _schedule_attr(ds, "optimized_total_instrument_hours")
        energy_kwh = _schedule_attr(ds, "optimized_controlled_energy_kwh")
        try:
            total_hours_value = float(total_hours)
        except ValueError:
            total_hours_value = float("nan")
        try:
            energy_kwh_value = float(energy_kwh)
        except ValueError:
            energy_kwh_value = float("nan")
        if np.isfinite(total_hours_value) or np.isfinite(energy_kwh_value):
            total_text = (
                f"{total_hours_value:.0f} instrument-hours"
                if np.isfinite(total_hours_value)
                else "instrument-hours unavailable"
            )
            energy_text = (
                f"{energy_kwh_value:.2f} kWh controlled energy"
                if np.isfinite(energy_kwh_value)
                else "controlled energy unavailable"
            )
            metrics.append({"label": "Selected total", "detail": f"{total_text}; {energy_text}."})
        info = {
            **info,
            "title": presentation.title,
            "summary": presentation.summary,
            "implementation": f"{presentation.explanation} {implementation}",
            "metrics": metrics,
        }
    return {"id": panel_key, **info}


COLOR = {
    "teal": "#0b7285",
    "light_blue": "#7fb6d6",
    "blue": "#4d6fb3",
    "purple": "#7768b8",
    "brown": "#4f7d8d",
    "olive": "#7a9964",
    "red": "#c05647",
    "magenta": "#9f6b9f",
    "green": "#4f8c63",
    "slate": "#718195",
    "black": "#22313f",
}
PDU_OUTLET_COLORS = (
    COLOR["teal"],
    COLOR["light_blue"],
    COLOR["blue"],
    COLOR["purple"],
    COLOR["brown"],
    COLOR["olive"],
    COLOR["red"],
    COLOR["magenta"],
)

PLOT_TEXT = "#22313f"
PLOT_LINE = "#c5d0da"
PLOT_GRID = "#e5eaef"
PLOT_BORDER = "#d8e1e8"


SUMMARY_INSTRUMENTS = ("vaisalamet", "asfs-logger", "asfs-fast-sonic", "power", "ops-monitor")

DISPLAY_NAMES = {
    "vaisalamet": "Meteorology",
    "asfs-logger": "Radiation",
    "asfs-fast-sonic": "ASFS Fast Sonic",
    "power": "Aurora Power Supply",
    "ops-monitor": "Operations",
}

HOUSEKEEPING_LABELS = {
    "vaisalamet": "HK_Met",
    "asfs-logger": "HK_ASFS",
    "power": "HK_APS",
    "ops-monitor": "HK_Operations",
}

QUICKLOOK_PREFIX = {
    "vaisalamet": "vaisalamet",
    "asfs-logger": "asfs_logger",
    "asfs-fast-sonic": "asfs_fast_sonic",
    "power": "power",
    "ops-monitor": "ops_monitor",
}

LEGACY_ALIAS_PREFIX = {
    "vaisalamet": "vaisalamet",
    "asfs-logger": "asfs_logger",
    "power": "power",
}

STATUS_TOKENS = (
    "alarm",
    "bits",
    "critical_error",
    "dev_",
    "discrepancy",
    "err_",
    "error",
    "failure",
    "locked",
    "not_available",
    "not_ready",
    "not_reliable",
    "online",
    "over_range",
    "qc",
    "quality",
    "senspathstate",
    "sensor_fail",
    "sensor_failure",
    "state",
    "status",
    "under_range",
    "warning",
)

HUMAN_LABELS = {
    "baro_hPa": "Pressure",
    "h1_t": "HMP1 Air Temperature",
    "t2_t": "T2 Air Temperature",
    "h1_td": "Dew Point",
    "h1_rh": "Relative Humidity",
    "h1_e": "Vapor Pressure",
    "h1_ah": "Absolute Humidity",
    "h1_mr": "Mixing Ratio",
    "h1_online": "HMP1 Online",
    "t2_online": "T2 Online",
    "h1_error_status": "HMP1 Error Status",
    "t2_error_status": "T2 Error Status",
    "baro_err_pressure_meas_err": "Pressure Measurement Error",
    "baro_err_pressure_oor": "Pressure Out of Range",
    "baro_st_sensor_failure": "Pressure Sensor Failure",
    "baro_st_value_locked": "Pressure Value Locked",
    "batt_volt_Avg": "Battery Voltage",
    "amp_meter_48vdc_Avg": "48 V Current",
    "watts_on_48vdc_Avg": "48 V Power",
    "PTemp_Avg": "Panel Temperature",
    "metek_x_out_Avg": "Metek U Wind",
    "metek_y_out_Avg": "Metek V Wind",
    "metek_z_out_Avg": "Metek W Wind",
    "metek_T_out_Avg": "Sonic Temperature",
    "metek_InclX_out_Avg": "Metek Tilt X",
    "metek_InclY_out_Avg": "Metek Tilt Y",
    "MetekWindSpeed": "Metek Wind Speed",
    "MetekWindDirection": "Metek Wind Direction",
    "spn1_tot_Avg": "Total Radiation",
    "spn1_dif_Avg": "Diffuse Radiation",
    "sr30_swd_Irr_Avg": "Downwelling Shortwave",
    "sr30_swu_Irr_Avg": "Upwelling Shortwave",
    "sr30_swd_DegC_Avg": "Downwelling SR30 Body Temperature",
    "sr30_swu_DegC_Avg": "Upwelling SR30 Body Temperature",
    "sr30_swd_tilt_Avg": "Downwelling SR30 Tilt",
    "sr30_swu_tilt_Avg": "Upwelling SR30 Tilt",
    "sr30_swd_rot_Avg": "Downwelling SR30 Rotation",
    "sr30_swu_rot_Avg": "Upwelling SR30 Rotation",
    "sr30_swd_fantach_Avg": "Downwelling SR30 Fan Tach",
    "sr30_swu_fantach_Avg": "Upwelling SR30 Fan Tach",
    "sr30_swd_fanstate_Avg": "Downwelling SR30 Fan State",
    "sr30_swu_fanstate_Avg": "Upwelling SR30 Fan State",
    "sr30_swd_heatstate_Avg": "Downwelling SR30 Heater State",
    "sr30_swu_heatstate_Avg": "Upwelling SR30 Heater State",
    "ir20_lwd_Wm2_Avg": "Downwelling Longwave",
    "ir20_lwu_Wm2_Avg": "Upwelling Longwave",
    "ir20_lwd_DegC_Avg": "Downwelling IR20 Body Temperature",
    "ir20_lwu_DegC_Avg": "Upwelling IR20 Body Temperature",
    "ir20_lwd_fan_Avg": "Downwelling IR20 Fan",
    "ir20_lwu_fan_Avg": "Upwelling IR20 Fan",
    "fp_A_Wm2_Avg": "Flux Plate A",
    "sr50_dist_Avg": "SR50 Distance",
    "sr50_qc_Avg": "SR50 Quality",
    "kt15_amb_Avg": "KT15 Ambient Temperature",
    "kt15_tem_Avg": "KT15 Surface Temperature",
    "licor_co2_out_Avg": "LI-COR CO2",
    "licor_h2o_out_Avg": "LI-COR H2O",
    "licor_pr_out_Avg": "LI-COR Pressure",
    "licor_t_out_Avg": "LI-COR Temperature",
    "licor_diag_out_Avg": "LI-COR Diagnostic",
    "licor_co2_str_out_Avg": "LI-COR CO2 Strength",
    "vaisala_T_Avg": "ASFS Vaisala Temperature",
    "vaisala_RH_Avg": "ASFS Vaisala Relative Humidity",
    "vaisala_P_Avg": "ASFS Vaisala Pressure",
    "metek_x_out": "Metek U Wind",
    "metek_y_out": "Metek V Wind",
    "metek_z_out": "Metek W Wind",
    "metek_T_out": "Sonic Temperature",
    "metek_InclX_out": "Tilt X",
    "metek_InclY_out": "Tilt Y",
    "metek_quality_out": "Metek Quality",
    "metek_senspathstate_out": "Sensor Path State",
    "ACOutputWatts": "AC Output Power",
    "DCInverterWatts": "DC Inverter Power",
    "ACOutputVolts": "AC Output Voltage",
    "DCInverterVolts": "DC Inverter Voltage",
    "BatteryWatts": "Battery Power",
    "BatteryAmps": "Battery Current",
    "BatterySOC": "State of Charge",
    "BatteryState": "Battery State",
    "BattsOnline": "Batteries Online",
    "InternalTemperature": "Internal Temperature",
    "HeatsinkTemperature": "Heatsink Temperature",
    "TempSensor1": "Temperature Sensor 1",
    "TempSensor2": "Temperature Sensor 2",
    "TempSensor3": "Temperature Sensor 3",
    "TempSensor4": "Temperature Sensor 4",
    "SolarWatts_East": "Solar East Power",
    "SolarWatts_South": "Solar South Power",
    "SolarWatts_West": "Solar West Power",
    "SolarVolts_East": "Solar East Voltage",
    "SolarVolts_South": "Solar South Voltage",
    "SolarVolts_West": "Solar West Voltage",
    "SolarAmps_East": "Solar East Current",
    "SolarAmps_South": "Solar South Current",
    "SolarAmps_West": "Solar West Current",
    "SolarYield_East": "East Solar Generated",
    "SolarYield_South": "South Solar Generated",
    "SolarYield_West": "West Solar Generated",
    "PowerDisplaySolarYield_East": "East Solar Generated",
    "PowerDisplaySolarYield_South": "South Solar Generated",
    "PowerDisplaySolarYield_West": "West Solar Generated",
    "CumulativePowerGeneratedTotal": "Total Generated",
    "CumulativePowerUtilised": "Power Utilised",
    "PowerDisplayCumulativePowerGeneratedTotal": "Total Generated",
    "PowerDisplayCumulativePowerUtilised": "Power Utilised",
    "BatterySOCForecast": "ECMWF SOC Forecast",
    "ECMWFSolarIrradiance": "ECMWF Solar Power",
    "ForecastSolarWatts": "Forecast Solar Charging",
    "ForecastLoadWatts": "Forecast Load",
    "BatterySOCForecast_Load100W": "SOC Forecast 100 W Load",
    "BatterySOCForecast_Load200W": "SOC Forecast 200 W Load",
    "BatterySOCForecast_Load300W": "SOC Forecast 300 W Load",
    "BatterySOCForecast_Load400W": "SOC Forecast 400 W Load",
    "BatterySOCForecast_Load500W": "SOC Forecast 500 W Load",
    "BatterySOCForecast_Load600W": "SOC Forecast 600 W Load",
    "ForecastVerificationSamples": "Verified Forecast Samples",
    "ForecastIndependentCycles": "Independent ECMWF Cycles",
    "ForecastSOCMAE_0_6h_Verified": "SOC MAE 0-6 h",
    "ForecastSOCMAE_6_24h_Verified": "SOC MAE 6-24 h",
    "ForecastSOCMAE_24_48h_Verified": "SOC MAE 24-48 h",
    "ForecastSOCMAE_48_96h_Verified": "SOC MAE 48-96 h",
    "ForecastSOCBias_0_6h_Verified": "SOC Bias 0-6 h",
    "ForecastSOCBias_6_24h_Verified": "SOC Bias 6-24 h",
    "ForecastSOCBias_24_48h_Verified": "SOC Bias 24-48 h",
    "ForecastSOCBias_48_96h_Verified": "SOC Bias 48-96 h",
    "ForecastSOCReadiness_0_6h": "SOC Target Ready 0-6 h",
    "ForecastSOCReadiness_6_24h": "SOC Target Ready 6-24 h",
    "ForecastSOCReadiness_24_48h": "SOC Target Ready 24-48 h",
    "ForecastSOCReadiness_48_96h": "SOC Target Ready 48-96 h",
    "ForecastSOCSkill_0_6h": "SOC Skill 0-6 h",
    "ForecastSolarMAE24h": "Solar MAE 24 h",
    "ForecastSolarBias24h": "Solar Bias 24 h",
    "ForecastSolarSkill24h": "Solar Skill 24 h",
    "ForecastLoadMAE24h": "Load MAE 24 h",
    "ForecastLoadBias24h": "Load Bias 24 h",
    "ForecastLoadSkill24h": "Load Skill 24 h",
    "BatterySOCObservedHindcast": "Observed SOC",
    "BatterySOCHindcast_6h": "Forecast Issued 6 h Earlier",
    "BatterySOCHindcast_24h": "Forecast Issued 24 h Earlier",
    "BatterySOCHindcast_48h": "Forecast Issued 48 h Earlier",
    "BatterySOCHindcast_72h": "Forecast Issued 72 h Earlier",
    "BatterySOCForecastP10": "ECMWF Ensemble P10",
    "BatterySOCForecastP50": "ECMWF Ensemble Median",
    "BatterySOCForecastP90": "ECMWF Ensemble P90",
    "BatterySOCForecastMinimum": "ECMWF Ensemble Minimum",
    "BatterySOCForecastMaximum": "ECMWF Ensemble Maximum",
    SOC_BELOW_THRESHOLD_PROBABILITY_FIELD: f"Probability SOC Below {MINIMUM_OPERATIONAL_SOC_LABEL}",
    "SystemAsIsDecisionSOCP10": "System As-Is Decision P10",
    "SystemAsIsDecisionSOCP50": "System As-Is Decision Median",
    "SystemAsIsDecisionSOCP90": "System As-Is Decision P90",
    "SystemAsIsDecisionBelow40Probability": f"System As-Is Probability SOC Below {MINIMUM_OPERATIONAL_SOC_LABEL}",
    "ForecastSOCCRPS_0_6h": "SOC CRPS 0-6 h",
    "ForecastSOCCRPS_6_24h": "SOC CRPS 6-24 h",
    "ForecastSOCCRPS_24_48h": "SOC CRPS 24-48 h",
    "ForecastSOCCRPS_48_96h": "SOC CRPS 48-96 h",
    "ForecastSOCIntervalCoverage80": "P10-P90 Coverage",
    SOC_BELOW_THRESHOLD_BRIER_FIELD: f"Below {MINIMUM_OPERATIONAL_SOC_LABEL} Brier Score",
    "ForecastEnsembleCycles": "Verified Ensemble Cycles",
    "SolarState_East": "Solar East State",
    "SolarState_South": "Solar South State",
    "SolarState_West": "Solar West State",
    "AlarmBits": "Alarm Bits",
    "FaultBits": "Fault Bits",
    "HeatsinkTempAlarm": "Heatsink Alarm",
    "InternalTempAlarm": "Internal Alarm",
    "time_discrepancy": "Clock Discrepancy",
}
for _outlet in range(1, PDU_OUTLET_COUNT + 1):
    _outlet_label = PDU_OUTLET_LABELS.get(_outlet, f"PDU Outlet {_outlet}")
    HUMAN_LABELS.setdefault(f"PDUOutlet{_outlet}Watts", f"{_outlet_label} Power")
    HUMAN_LABELS.setdefault(f"PDUOutlet{_outlet}Amps", f"{_outlet_label} Current")
    HUMAN_LABELS.setdefault(f"PDUOutlet{_outlet}State", f"{_outlet_label} State")

HUMAN_UNITS = {
    "baro_hPa": "hPa",
    "h1_t": "C",
    "t2_t": "C",
    "h1_td": "C",
    "h1_rh": "%",
    "h1_e": "hPa",
    "h1_ah": "g m^-3",
    "h1_mr": "g kg^-1",
    "h1_online": "state",
    "t2_online": "state",
    "h1_error_status": "state",
    "t2_error_status": "state",
    "baro_err_pressure_meas_err": "state",
    "baro_err_pressure_oor": "state",
    "baro_st_sensor_failure": "state",
    "baro_st_value_locked": "state",
    "batt_volt_Avg": "V",
    "amp_meter_48vdc_Avg": "A",
    "watts_on_48vdc_Avg": "W",
    "PTemp_Avg": "C",
    "metek_x_out_Avg": "m s^-1",
    "metek_y_out_Avg": "m s^-1",
    "metek_z_out_Avg": "m s^-1",
    "metek_T_out_Avg": "C",
    "metek_InclX_out_Avg": "deg",
    "metek_InclY_out_Avg": "deg",
    "MetekWindSpeed": "m s^-1",
    "MetekWindDirection": "deg",
    "spn1_tot_Avg": "W m^-2",
    "spn1_dif_Avg": "W m^-2",
    "sr30_swd_Irr_Avg": "W m^-2",
    "sr30_swu_Irr_Avg": "W m^-2",
    "sr30_swd_DegC_Avg": "C",
    "sr30_swu_DegC_Avg": "C",
    "sr30_swd_tilt_Avg": "deg",
    "sr30_swu_tilt_Avg": "deg",
    "sr30_swd_rot_Avg": "deg",
    "sr30_swu_rot_Avg": "deg",
    "sr30_swd_fantach_Avg": "Hz",
    "sr30_swu_fantach_Avg": "Hz",
    "sr30_swd_fanstate_Avg": "state",
    "sr30_swu_fanstate_Avg": "state",
    "sr30_swd_heatstate_Avg": "state",
    "sr30_swu_heatstate_Avg": "state",
    "ir20_lwd_Wm2_Avg": "W m^-2",
    "ir20_lwu_Wm2_Avg": "W m^-2",
    "ir20_lwd_DegC_Avg": "C",
    "ir20_lwu_DegC_Avg": "C",
    "ir20_lwd_fan_Avg": "Hz",
    "ir20_lwu_fan_Avg": "Hz",
    "fp_A_Wm2_Avg": "W m^-2",
    "fp_B_Wm2_Avg": "W m^-2",
    "sr50_dist_Avg": "m",
    "sr50_qc_Avg": "state",
    "kt15_amb_Avg": "C",
    "kt15_tem_Avg": "C",
    "licor_co2_out_Avg": "mmol m^-3",
    "licor_h2o_out_Avg": "mmol m^-3",
    "licor_pr_out_Avg": "kPa",
    "licor_t_out_Avg": "C",
    "licor_diag_out_Avg": "code",
    "licor_co2_str_out_Avg": "%",
    "vaisala_T_Avg": "C",
    "vaisala_RH_Avg": "%",
    "vaisala_P_Avg": "hPa",
    "metek_x_out": "m s^-1",
    "metek_y_out": "m s^-1",
    "metek_z_out": "m s^-1",
    "metek_T_out": "C",
    "metek_InclX_out": "deg",
    "metek_InclY_out": "deg",
    "metek_msec_out": "ms",
    "metek_quality_out": "state",
    "metek_senspathstate_out": "state",
    "ACOutputAmps": "A",
    "ACOutputHZ": "Hz",
    "ACOutputVolts": "V",
    "ACOutputWatts": "W",
    "ACkWh": "kWh",
    "ACnHours": "h",
    "BatteryAmps": "A",
    "BatterySOC": "%",
    "BatteryState": "state",
    "BatteryWatts": "W",
    "BattsOnline": "state",
    "DCInverterAmps": "A",
    "DCInverterVolts": "V",
    "DCInverterWatts": "W",
    "FaultBits": "bits",
    "AlarmBits": "bits",
    "HeatsinkTempAlarm": "state",
    "HeatsinkTemperature": "C",
    "InternalTempAlarm": "state",
    "InternalTemperature": "C",
    "MaxSolarWatts_East": "W",
    "MaxSolarWatts_South": "W",
    "MaxSolarWatts_West": "W",
    "SolarAmps_East": "A",
    "SolarAmps_South": "A",
    "SolarAmps_West": "A",
    "SolarState_East": "state",
    "SolarState_South": "state",
    "SolarState_West": "state",
    "SolarVolts_East": "V",
    "SolarVolts_South": "V",
    "SolarVolts_West": "V",
    "SolarWatts_East": "W",
    "SolarWatts_South": "W",
    "SolarWatts_West": "W",
    "SolarYield_East": "kWh",
    "SolarYield_South": "kWh",
    "SolarYield_West": "kWh",
    "PowerDisplaySolarYield_East": "kWh",
    "PowerDisplaySolarYield_South": "kWh",
    "PowerDisplaySolarYield_West": "kWh",
    "CumulativePowerGeneratedTotal": "kWh",
    "CumulativePowerUtilised": "kWh",
    "PowerDisplayCumulativePowerGeneratedTotal": "kWh",
    "PowerDisplayCumulativePowerUtilised": "kWh",
    "BatterySOCForecast": "%",
    "ECMWFSolarIrradiance": "W/m2",
    "ForecastSolarWatts": "W",
    "ForecastLoadWatts": "W",
    "BatterySOCForecast_Load100W": "%",
    "BatterySOCForecast_Load200W": "%",
    "BatterySOCForecast_Load300W": "%",
    "BatterySOCForecast_Load400W": "%",
    "BatterySOCForecast_Load500W": "%",
    "BatterySOCForecast_Load600W": "%",
    "ForecastVerificationSamples": "samples",
    "ForecastIndependentCycles": "cycles",
    "ForecastSOCMAE_0_6h_Verified": "percentage points",
    "ForecastSOCMAE_6_24h_Verified": "percentage points",
    "ForecastSOCMAE_24_48h_Verified": "percentage points",
    "ForecastSOCMAE_48_96h_Verified": "percentage points",
    "ForecastSOCBias_0_6h_Verified": "percentage points",
    "ForecastSOCBias_6_24h_Verified": "percentage points",
    "ForecastSOCBias_24_48h_Verified": "percentage points",
    "ForecastSOCBias_48_96h_Verified": "percentage points",
    "ForecastSOCReadiness_0_6h": "1",
    "ForecastSOCReadiness_6_24h": "1",
    "ForecastSOCReadiness_24_48h": "1",
    "ForecastSOCReadiness_48_96h": "1",
    "ForecastSOCSkill_0_6h": "1",
    "ForecastSolarMAE24h": "W",
    "ForecastSolarBias24h": "W",
    "ForecastSolarSkill24h": "1",
    "ForecastLoadMAE24h": "W",
    "ForecastLoadBias24h": "W",
    "ForecastLoadSkill24h": "1",
    "BatterySOCObservedHindcast": "%",
    "BatterySOCHindcast_6h": "%",
    "BatterySOCHindcast_24h": "%",
    "BatterySOCHindcast_48h": "%",
    "BatterySOCHindcast_72h": "%",
    "BatterySOCForecastP10": "%",
    "BatterySOCForecastP50": "%",
    "BatterySOCForecastP90": "%",
    "BatterySOCForecastMinimum": "%",
    "BatterySOCForecastMaximum": "%",
    SOC_BELOW_THRESHOLD_PROBABILITY_FIELD: "1",
    "SystemAsIsDecisionSOCP10": "%",
    "SystemAsIsDecisionSOCP50": "%",
    "SystemAsIsDecisionSOCP90": "%",
    "SystemAsIsDecisionBelow40Probability": "1",
    "ForecastSOCCRPS_0_6h": "percentage points",
    "ForecastSOCCRPS_6_24h": "percentage points",
    "ForecastSOCCRPS_24_48h": "percentage points",
    "ForecastSOCCRPS_48_96h": "percentage points",
    "ForecastSOCIntervalCoverage80": "1",
    SOC_BELOW_THRESHOLD_BRIER_FIELD: "1",
    "ForecastEnsembleCycles": "cycles",
    "TempSensor1": "C",
    "TempSensor2": "C",
    "TempSensor3": "C",
    "TempSensor4": "C",
    "TotCapacity": "capacity units",
    "time_discrepancy": "s",
    "scantime": "s",
}
for _outlet in range(1, PDU_OUTLET_COUNT + 1):
    HUMAN_UNITS.setdefault(f"PDUOutlet{_outlet}Watts", "W")
    HUMAN_UNITS.setdefault(f"PDUOutlet{_outlet}Amps", "A")
    HUMAN_UNITS.setdefault(f"PDUOutlet{_outlet}State", "state")

DISPLAY_SCALE = {}

SUMMARY_SOURCE_INSTRUMENTS = {
    "vaisalamet": ("vaisalamet", "asfs-logger", "asfs-fast-sonic"),
    "asfs-logger": ("asfs-logger",),
    "asfs-fast-sonic": ("asfs-fast-sonic",),
    "power": ("power", "asfs-logger"),
    "ops-monitor": ("ops-monitor",),
}

SUMMARY_LAYOUTS: dict[str, tuple[PanelSpec, ...]] = {
    "vaisalamet": (
        PanelSpec(
            "air_temperature",
            "Air Temperature",
            "Air Temperature [C]",
            None,
            (
                TraceSpec("h1_t", "HMP1 Air Temperature", COLOR["teal"]),
                TraceSpec("t2_t", "T2 Air Temperature", COLOR["light_blue"]),
                TraceSpec("vaisala_T_Avg", "ASFS Vaisala Temperature", COLOR["green"]),
                TraceSpec("metek_T_out_Avg", "Sonic Temperature", COLOR["brown"]),
            ),
        ),
        PanelSpec(
            "humidity",
            "Humidity / Dew Point",
            "Dew Point [C]",
            "Relative Humidity [%]",
            (
                TraceSpec("h1_td", "Dew Point", COLOR["purple"]),
                TraceSpec("h1_rh", "Relative Humidity", COLOR["brown"], axis="right"),
                TraceSpec("vaisala_RH_Avg", "ASFS Vaisala Relative Humidity", COLOR["green"], axis="right"),
            ),
        ),
        PanelSpec(
            "pressure",
            "Pressure",
            "Pressure [hPa]",
            None,
            (
                TraceSpec("baro_hPa", "Pressure", COLOR["green"]),
                TraceSpec("vaisala_P_Avg", "ASFS Vaisala Pressure", COLOR["teal"]),
            ),
        ),
        PanelSpec(
            "met",
            "Met",
            "Metek U / V Wind [m/s]",
            "Metek W Wind [m/s]",
            (
                TraceSpec("metek_x_out_Avg", "Metek U Wind", COLOR["teal"]),
                TraceSpec("metek_y_out_Avg", "Metek V Wind", COLOR["light_blue"]),
                TraceSpec("metek_z_out_Avg", "Metek W Wind", COLOR["purple"], axis="right"),
            ),
        ),
        PanelSpec(
            "metek_wind_speed_direction",
            "Metek Wind Speed / Direction",
            "Wind Speed [m/s]",
            "Wind Direction [deg]",
            (
                TraceSpec("MetekWindSpeed", "Wind Speed", COLOR["teal"], valid_min=0.0, valid_max=100.0),
                TraceSpec("MetekWindDirection", "Wind Direction", COLOR["purple"], axis="right", valid_min=0.0, valid_max=360.0),
            ),
        ),
    ),
    "asfs-logger": (
        PanelSpec(
            "shortwave_radiation",
            "Shortwave Radiation",
            "Radiation [W m^-2]",
            None,
            (
                TraceSpec("spn1_tot_Avg", "Total Radiation", COLOR["brown"]),
                TraceSpec("spn1_dif_Avg", "Diffuse Radiation", COLOR["purple"]),
                TraceSpec("sr30_swd_Irr_Avg", "Downwelling Shortwave", COLOR["brown"]),
                TraceSpec("sr30_swu_Irr_Avg", "Upwelling Shortwave", COLOR["olive"]),
            ),
        ),
        PanelSpec(
            "longwave_radiation",
            "Longwave Radiation",
            "Radiation [W m^-2]",
            None,
            (
                TraceSpec("ir20_lwd_Wm2_Avg", "Downwelling Longwave", COLOR["teal"]),
                TraceSpec("ir20_lwu_Wm2_Avg", "Upwelling Longwave", COLOR["light_blue"]),
            ),
        ),
        PanelSpec(
            "flux_plates",
            "Flux Plate",
            "Flux [W m^-2]",
            None,
            (
                TraceSpec("fp_A_Wm2_Avg", "Flux Plate A", COLOR["teal"]),
            ),
        ),
        PanelSpec(
            "surface_temperature",
            "Surface / Snow",
            "KT15 Surface Temperature [C]",
            "SR50 Distance [m]",
            (
                TraceSpec("kt15_tem_Avg", "KT15 Surface Temperature", COLOR["magenta"]),
                TraceSpec("sr50_dist_Avg", "SR50 Distance", COLOR["olive"], axis="right"),
            ),
        ),
    ),
    "asfs-logger-hk": (
        PanelSpec(
            "logger_power",
            "Logger Power",
            "Voltage [V]",
            "Current [A] / Power [W]",
            (
                TraceSpec("batt_volt_Avg", "Battery Voltage", COLOR["teal"]),
                TraceSpec("amp_meter_48vdc_Avg", "48 V Current", COLOR["purple"], axis="right"),
                TraceSpec("watts_on_48vdc_Avg", "48 V Power", COLOR["brown"], axis="right"),
            ),
        ),
        PanelSpec(
            "logger_thermal_scan",
            "Logger Thermal / Scan",
            "Panel Temperature [C]",
            "Scan Time [s]",
            (
                TraceSpec("PTemp_Avg", "Panel Temperature", COLOR["brown"]),
                TraceSpec("scantime", "Scan Time", COLOR["slate"], axis="right"),
            ),
        ),
        PanelSpec(
            "asfs_met_licor",
            "ASFS Met / LI-COR",
            "CO2 / H2O [mmol m^-3]",
            "Signal Strength [%]",
            (
                TraceSpec("licor_co2_out_Avg", "CO2 Output", COLOR["teal"]),
                TraceSpec("licor_h2o_out_Avg", "H2O Output", COLOR["light_blue"]),
                TraceSpec("licor_co2_str_out_Avg", "CO2 Signal Strength", COLOR["green"], axis="right", valid_min=0.0, valid_max=100.0),
            ),
        ),
        PanelSpec(
            "licor_diagnostics",
            "LI-COR Diagnostics",
            "Diagnostic Code",
            None,
            (
                TraceSpec("licor_diag_out_Avg", "Diagnostic Code", COLOR["red"], step=True),
            ),
        ),
        PanelSpec(
            "shortwave_mechanics",
            "SR30 Orientation",
            "Tilt [deg]",
            "Rotation [deg]",
            (
                TraceSpec("sr30_swd_tilt_Avg", "Downwelling Tilt", COLOR["teal"], valid_min=-5.0, valid_max=20.0),
                TraceSpec("sr30_swu_tilt_Avg", "Upwelling Tilt", COLOR["light_blue"], valid_min=150.0, valid_max=200.0),
                TraceSpec("sr30_swd_rot_Avg", "Downwelling Rotation", COLOR["purple"], axis="right", valid_min=-20.0, valid_max=360.0),
                TraceSpec("sr30_swu_rot_Avg", "Upwelling Rotation", COLOR["olive"], axis="right", valid_min=-20.0, valid_max=360.0),
            ),
        ),
        PanelSpec(
            "shortwave_support",
            "SR30 Fans / Heaters",
            "Fan Tach [Hz]",
            "State",
            (
                TraceSpec("sr30_swd_fantach_Avg", "Downwelling Fan Tach", COLOR["teal"]),
                TraceSpec("sr30_swu_fantach_Avg", "Upwelling Fan Tach", COLOR["light_blue"]),
                TraceSpec("sr30_swd_fanstate_Avg", "Downwelling Fan State", COLOR["green"], axis="right", step=True),
                TraceSpec("sr30_swu_fanstate_Avg", "Upwelling Fan State", COLOR["olive"], axis="right", step=True),
                TraceSpec("sr30_swd_heatstate_Avg", "Downwelling Heater State", COLOR["purple"], axis="right", step=True),
                TraceSpec("sr30_swu_heatstate_Avg", "Upwelling Heater State", COLOR["brown"], axis="right", step=True),
            ),
        ),
        PanelSpec(
            "longwave_support",
            "IR20 Support",
            "Fan Tach [Hz]",
            "Body Temperature [C]",
            (
                TraceSpec("ir20_lwd_fan_Avg", "Downwelling IR20 Fan", COLOR["teal"]),
                TraceSpec("ir20_lwu_fan_Avg", "Upwelling IR20 Fan", COLOR["light_blue"]),
                TraceSpec("ir20_lwd_DegC_Avg", "Downwelling IR20 Body Temp", COLOR["purple"], axis="right"),
                TraceSpec("ir20_lwu_DegC_Avg", "Upwelling IR20 Body Temp", COLOR["brown"], axis="right"),
            ),
        ),
        PanelSpec(
            "sensor_variability",
            "Sensor Variability",
            "Standard Deviation",
            "SR50 QC",
            (
                TraceSpec("kt15_tem_Std", "KT15 Surface Std", COLOR["magenta"]),
                TraceSpec("spn1_tot_Std", "SPN1 Total Std", COLOR["brown"]),
                TraceSpec("sr50_dist_Std", "SR50 Distance Std", COLOR["teal"]),
                TraceSpec("sr50_qc_Avg", "SR50 Quality", COLOR["olive"], axis="right", step=True),
            ),
        ),
    ),
    "asfs-fast-sonic": (
        PanelSpec(
            "met",
            "Met",
            "Metek U / V Wind [m/s]",
            "Metek W Wind [m/s]",
            (
                TraceSpec("metek_x_out", "Metek U Wind", COLOR["teal"], valid_min=-100.0, valid_max=100.0),
                TraceSpec("metek_y_out", "Metek V Wind", COLOR["light_blue"], valid_min=-100.0, valid_max=100.0),
                TraceSpec("metek_z_out", "Metek W Wind", COLOR["purple"], axis="right", valid_min=-30.0, valid_max=30.0),
            ),
        ),
        PanelSpec(
            "tilt_temperature",
            "Tilt / Temperature",
            "Tilt [deg]",
            "Sonic Temperature [C]",
            (
                TraceSpec("metek_InclX_out", "Tilt X", COLOR["brown"], valid_min=-10.0, valid_max=360.0),
                TraceSpec("metek_InclY_out", "Tilt Y", COLOR["olive"], valid_min=-10.0, valid_max=360.0),
                TraceSpec("metek_T_out", "Sonic Temperature", COLOR["magenta"], axis="right", valid_min=-50.0, valid_max=50.0),
            ),
        ),
        PanelSpec(
            "quality",
            "Quality",
            "Quality",
            "State",
            (
                TraceSpec("metek_quality_out", "Metek Quality", COLOR["red"], step=True),
                TraceSpec("metek_senspathstate_out", "Sensor Path State", COLOR["slate"], axis="right", step=True),
            ),
        ),
    ),
    "power": (
        PanelSpec(
            "renewables",
            "Renewables",
            "Solar Power [W]",
            "Solar Voltage [V]",
            (
                TraceSpec("SolarWatts_East", "Solar East Power", COLOR["brown"]),
                TraceSpec("SolarWatts_South", "Solar South Power", COLOR["purple"]),
                TraceSpec("SolarWatts_West", "Solar West Power", COLOR["magenta"]),
                TraceSpec("SolarVolts_East", "Solar East Voltage", COLOR["brown"], axis="right", dash="dash", valid_min=0.0, valid_max=200.0),
                TraceSpec("SolarVolts_South", "Solar South Voltage", COLOR["purple"], axis="right", dash="dash", valid_min=0.0, valid_max=200.0),
                TraceSpec("SolarVolts_West", "Solar West Voltage", COLOR["magenta"], axis="right", dash="dash", valid_min=0.0, valid_max=200.0),
            ),
        ),
        PanelSpec(
            "battery_charging",
            "Battery Charging",
            "Current In/Out [A]",
            "Power In/Out [W]",
            (
                TraceSpec("BatteryAmps", "Current In/Out", COLOR["teal"], valid_min=-250.0, valid_max=250.0, smooth_minutes=30.0),
                TraceSpec("BatteryWatts", "Power In/Out", COLOR["light_blue"], axis="right", valid_min=-10000.0, valid_max=10000.0, smooth_minutes=30.0),
            ),
        ),
        PanelSpec(
            "output_power",
            "Output Power",
            "AC Output Power [W]",
            "DC Inverter Power [W]",
            (
                TraceSpec("ACOutputWatts", "AC Output Power", COLOR["red"], valid_min=0.0, valid_max=10000.0),
                TraceSpec("DCInverterWatts", "DC Inverter Power", COLOR["teal"], axis="right", valid_min=0.0, valid_max=10000.0),
            ),
        ),
        PanelSpec(
            "ass_dc_power",
            "ASS 48 V DC Power",
            "ASS 48 V DC Power [W]",
            None,
            (
                TraceSpec("watts_on_48vdc_Avg", "ASS 48 V DC Power", COLOR["purple"]),
            ),
        ),
        PanelSpec(
            "pdu_outlet_power",
            "ASS PDU Outlet Power",
            "PDU Outlet Power [W]",
            None,
            tuple(
                TraceSpec(
                    field_name,
                    PDU_OUTLET_LABELS.get(outlet, f"Outlet {outlet}"),
                    PDU_OUTLET_COLORS[(outlet - 1) % len(PDU_OUTLET_COLORS)],
                    valid_min=0.0,
                    valid_max=5000.0,
                )
                for outlet in PDU_DISPLAY_OUTLETS
                for field_name in (f"PDUOutlet{outlet}Watts",)
            ),
        ),
        PanelSpec(
            "cumulative_power",
            "Cumulative Energy & State of Charge",
            "SOC [%]",
            "Cumulative Energy [kWh]",
            (
                TraceSpec("BatterySOC", "State of Charge", COLOR["green"], valid_min=0.0, valid_max=100.0),
                TraceSpec("SolarYield_East", "East Solar Generated", COLOR["brown"], axis="right", break_on_day_change=True),
                TraceSpec("SolarYield_South", "South Solar Generated", COLOR["purple"], axis="right", break_on_day_change=True),
                TraceSpec("SolarYield_West", "West Solar Generated", COLOR["magenta"], axis="right", break_on_day_change=True),
                TraceSpec("CumulativePowerGeneratedTotal", "Total Generated", COLOR["green"], axis="right", break_on_day_change=True),
                TraceSpec("CumulativePowerUtilised", "Utilised", COLOR["teal"], axis="right", break_on_day_change=True),
            ),
        ),
        PanelSpec(
            "output_voltage",
            "Output Voltage",
            "AC Output Voltage [V]",
            "DC Inverter Voltage [V]",
            (
                TraceSpec("ACOutputVolts", "AC Output Voltage", COLOR["brown"], valid_min=180.0, valid_max=260.0),
                TraceSpec("DCInverterVolts", "DC Inverter Voltage", COLOR["magenta"], axis="right", valid_min=40.0, valid_max=70.0),
            ),
        ),
        PanelSpec(
            "thermal_state",
            "Thermal State",
            "Temperature [C]",
            "Temperature [C]",
            (
                TraceSpec("InternalTemperature", "Internal Temperature", COLOR["red"], valid_min=-40.0, valid_max=100.0),
                TraceSpec("HeatsinkTemperature", "Heatsink Temperature", COLOR["brown"], valid_min=-40.0, valid_max=120.0),
                TraceSpec("TempSensor1", "Temperature Sensor 1", COLOR["teal"], axis="right", valid_min=-40.0, valid_max=100.0),
                TraceSpec("TempSensor2", "Temperature Sensor 2", COLOR["light_blue"], axis="right", valid_min=-40.0, valid_max=100.0),
                TraceSpec("TempSensor3", "Temperature Sensor 3", COLOR["purple"], axis="right", valid_min=-40.0, valid_max=100.0),
                TraceSpec("TempSensor4", "Temperature Sensor 4", COLOR["olive"], axis="right", valid_min=-40.0, valid_max=100.0),
            ),
        ),
        PanelSpec(
            "soc_24h_forecast",
            "SOC Next 24 h Forecast",
            "SOC [%]",
            None,
            (
                TraceSpec(
                    "SystemAsIsDecisionSOCP50",
                    "System as-is - central",
                    COLOR["green"],
                    valid_min=0.0,
                    valid_max=100.0,
                    display_horizon_hours=24.0,
                ),
            ),
        ),
        PanelSpec(
            "soc_ecmwf_forecast",
            "SOC 96 h Forecast",
            "SOC [%]",
            f"Probability SOC Below {MINIMUM_OPERATIONAL_SOC_LABEL} [%]",
            (
                TraceSpec("SystemAsIsDecisionSOCP10", "System as-is - P10", COLOR["blue"], dash="dash", valid_min=0.0, valid_max=100.0, line_width=2.2),
                TraceSpec("SystemAsIsDecisionSOCP90", "System as-is - P90", COLOR["light_blue"], dash="dot", valid_min=0.0, valid_max=100.0, line_width=2.2),
                TraceSpec("SystemAsIsDecisionSOCP50", "System as-is - central", COLOR["green"], valid_min=0.0, valid_max=100.0, line_width=3.0),
                TraceSpec("SystemAsIsDecisionBelow40Probability", f"Probability Below {MINIMUM_OPERATIONAL_SOC_LABEL}", COLOR["red"], axis="right", scale=100.0, valid_min=0.0, valid_max=1.0),
            ),
            display_horizon_hours=POWER_FORECAST_DISPLAY_HOURS,
        ),
        PanelSpec(
            "soc_hindcast",
            "Battery SOC: Measured vs Earlier Forecasts",
            "SOC [%]",
            None,
            (
                TraceSpec("BatterySOCObservedHindcast", "Measured battery SOC", COLOR["green"], valid_min=0.0, valid_max=100.0, step=True),
                TraceSpec("BatterySOCHindcast_6h", "Forecast issued 6 h before valid time", COLOR["blue"], dash="dash", valid_min=0.0, valid_max=100.0),
                TraceSpec("BatterySOCHindcast_24h", "Forecast issued 24 h before valid time", COLOR["teal"], dash="dot", valid_min=0.0, valid_max=100.0),
                TraceSpec("BatterySOCHindcast_48h", "Forecast issued 48 h before valid time", COLOR["purple"], dash="dashdot", valid_min=0.0, valid_max=100.0),
                TraceSpec("BatterySOCHindcast_72h", "Forecast issued 72 h before valid time", COLOR["slate"], dash="longdash", valid_min=0.0, valid_max=100.0),
            ),
            description=SOC_HINDCAST_EXPLANATION,
        ),
        PanelSpec(
            "operating_plan_scenarios",
            "Suggested Instrument-Mode SOC Forecasts",
            "SOC [%]",
            None,
            (
                TraceSpec(
                    "OperatingCurrentSOCP50",
                    "Current load / system as-is",
                    COLOR["black"],
                    valid_min=0.0,
                    valid_max=100.0,
                    line_width=3.2,
                ),
                TraceSpec(
                    "OperatingP50ContinuationSOCP50",
                    "P50 continuation rule",
                    COLOR["teal"],
                    dash="longdash",
                    valid_min=0.0,
                    valid_max=100.0,
                    line_width=3.0,
                ),
                TraceSpec("OperatingSuggested1SOCP50", "CL61", COLOR["red"], valid_min=0.0, valid_max=100.0, line_width=2.1),
                TraceSpec("OperatingSuggested2SOCP50", "CL61 + Radar", COLOR["blue"], valid_min=0.0, valid_max=100.0, line_width=2.1),
                TraceSpec("OperatingSuggested3SOCP50", "CL61 + HATPRO", COLOR["purple"], valid_min=0.0, valid_max=100.0, line_width=2.1),
                TraceSpec("OperatingSuggested4SOCP50", "CL61 + HATPRO + Radar", COLOR["magenta"], valid_min=0.0, valid_max=100.0, line_width=2.4),
                TraceSpec("OperatingSuggested5SOCP50", "HATPRO + Radar", COLOR["green"], dash="dash", valid_min=0.0, valid_max=100.0, line_width=2.1),
                TraceSpec("OperatingSuggested6SOCP50", "Radar", COLOR["slate"], dash="dash", valid_min=0.0, valid_max=100.0, line_width=2.1),
                TraceSpec("OperatingSuggested7SOCP50", "HATPRO", COLOR["brown"], dash="dash", valid_min=0.0, valid_max=100.0, line_width=2.1),
                TraceSpec(
                    "OperatingSuggested8SOCP50",
                    "All instruments + UAS tier 3",
                    COLOR["teal"],
                    dash="dashdot",
                    valid_min=0.0,
                    valid_max=100.0,
                    line_width=2.6,
                ),
            ),
            display_horizon_hours=POWER_FORECAST_DISPLAY_HOURS,
        ),
        PanelSpec(
            "operating_plan_schedule",
            "Recommended Additive Instrument Schedule",
            "Additive activity (sum 0-3; each instrument 0/1)",
            None,
            (
                TraceSpec("OperatingCL61OptimizedActiveCount", "Total active instruments", COLOR["green"], step=True, valid_min=0.0, valid_max=3.0, line_width=3.5),
                TraceSpec("OperatingCL61OptimizedCL61On", "Recommended CL61 schedule", COLOR["red"], step=True, valid_min=0.0, valid_max=1.0, line_width=3.0),
                TraceSpec("OperatingCL61OptimizedRadarOn", "Recommended Radar schedule", COLOR["blue"], dash="dash", step=True, valid_min=0.0, valid_max=1.0, line_width=2.5),
                TraceSpec("OperatingCL61OptimizedHATPROOn", "Recommended HATPRO schedule", COLOR["purple"], dash="dot", step=True, valid_min=0.0, valid_max=1.0, line_width=2.5),
            ),
            display_horizon_hours=POWER_FORECAST_DISPLAY_HOURS,
        ),
        PanelSpec(
            "ecmwf_solar_forecast",
            "ECMWF Solar & Load Forecast",
            "ECMWF Solar [W/m2]",
            "Forecast Charging / Load [W]",
            (
                TraceSpec("ECMWFSolarIrradiance", "ECMWF Solar Power", COLOR["brown"], valid_min=0.0),
                TraceSpec("ForecastSolarWatts", "Forecast Solar Charging", COLOR["green"], axis="right", dash="dot", valid_min=0.0),
                TraceSpec("OperatingCurrentLoadP50Watts", "Current Instrument Load", COLOR["slate"], axis="right", dash="dot", valid_min=0.0),
                TraceSpec("OperatingCL61OptimizedLoadP50Watts", "Recommended Instrument Load", COLOR["red"], axis="right", dash="dashdot", valid_min=0.0, line_width=3.0),
            ),
            display_horizon_hours=POWER_FORECAST_DISPLAY_HOURS,
        ),
        PanelSpec(
            "soc_forecast_skill",
            "SOC Forecast Verification",
            "SOC MAE [percentage points]",
            "Independent ECMWF Cycles [count]",
            (
                TraceSpec("ForecastSOCMAE_0_6h_Verified", "SOC MAE 0-6 h", COLOR["blue"], valid_min=0.0),
                TraceSpec("ForecastSOCMAE_6_24h_Verified", "SOC MAE 6-24 h", COLOR["teal"], valid_min=0.0),
                TraceSpec("ForecastSOCMAE_24_48h_Verified", "SOC MAE 24-48 h", COLOR["purple"], valid_min=0.0),
                TraceSpec("ForecastSOCMAE_48_96h_Verified", "SOC MAE 48-96 h", COLOR["slate"], valid_min=0.0),
                TraceSpec("ForecastIndependentCycles", "Independent ECMWF Cycles", COLOR["brown"], axis="right", dash="dot", valid_min=0.0),
            ),
            "Rolling 24-hour verification of archived SOC forecasts against later APS measurements. Lower MAE is better.",
        ),
        PanelSpec(
            "soc_ensemble_skill",
            "SOC Ensemble Verification",
            "SOC CRPS [percentage points]",
            "Coverage / Brier [0-1]",
            (
                TraceSpec("ForecastSOCCRPS_0_6h", "SOC CRPS 0-6 h", COLOR["blue"], valid_min=0.0),
                TraceSpec("ForecastSOCCRPS_6_24h", "SOC CRPS 6-24 h", COLOR["teal"], valid_min=0.0),
                TraceSpec("ForecastSOCCRPS_24_48h", "SOC CRPS 24-48 h", COLOR["purple"], valid_min=0.0),
                TraceSpec("ForecastSOCCRPS_48_96h", "SOC CRPS 48-96 h", COLOR["slate"], valid_min=0.0),
                TraceSpec("ForecastSOCIntervalCoverage80", "P10-P90 Coverage", COLOR["olive"], axis="right", valid_min=0.0, valid_max=1.0),
                TraceSpec(SOC_BELOW_THRESHOLD_BRIER_FIELD, f"Below {MINIMUM_OPERATIONAL_SOC_LABEL} Brier Score", COLOR["red"], axis="right", dash="dash", valid_min=0.0, valid_max=1.0),
            ),
            "Rolling 24-hour ensemble verification. CRPS measures forecast accuracy and uncertainty; coverage targets 0.80; Brier assesses the probability of SOC below 40%.",
        ),
        PanelSpec(
            "forecast_power_skill",
            "Solar and Load Forecast Verification",
            "Forecast Error [W]",
            None,
            (
                TraceSpec("ForecastSolarMAE24h", "Solar MAE 24 h", COLOR["brown"], valid_min=0.0),
                TraceSpec("ForecastSolarBias24h", "Solar Bias 24 h", COLOR["olive"]),
                TraceSpec("ForecastLoadMAE24h", "Load MAE 24 h", COLOR["red"], dash="dash", valid_min=0.0),
                TraceSpec("ForecastLoadBias24h", "Load Bias 24 h", COLOR["purple"], dash="dot"),
            ),
            "Rolling 24-hour forecast error in watts. Bias is forecast minus observed: positive solar means too much charging; positive load means too much consumption.",
        ),
    ),
    "ops-monitor": (
        PanelSpec(
            "source_disk_use",
            "Host Disk Use",
            "Used [%]",
            None,
            (
                TraceSpec("host_celine_source_used_pct", "CL61 Root", COLOR["teal"]),
                TraceSpec("host_celine_data_used_pct", "CL61 Data", COLOR["blue"]),
                TraceSpec("host_ass_data_used_pct", "ASS Data", COLOR["slate"]),
                TraceSpec("host_ass_root_used_pct", "ASS Root", COLOR["purple"]),
                TraceSpec("host_aps_data_used_pct", "APS Data", COLOR["brown"]),
                TraceSpec("host_aps_root_used_pct", "APS Root", COLOR["olive"]),
            ),
        ),
        PanelSpec(
            "aurora_storage",
            "Aurora / Archive Storage",
            "Used [%]",
            None,
            (
                TraceSpec("aurora_project_used_pct", "Aurora Raw (/project)", COLOR["teal"]),
                TraceSpec("aurora_data_used_pct", "AURORA Cloud Products (/data/aurora)", COLOR["purple"]),
                TraceSpec("aurora_root_used_pct", "AURORA Cloud Root (/)", COLOR["olive"]),
                TraceSpec("gws_storage_used_pct", "JASMIN GWS", COLOR["green"]),
            ),
        ),
        PanelSpec(
            "aps_battery_voltage",
            "APS Electrical / Thermal",
            "Voltage [V] / SOC [%]",
            "Temperature [C]",
            (
                TraceSpec("aps_battery_voltage_v", "DC Inverter Voltage", COLOR["brown"]),
                TraceSpec("aps_battery_soc_pct", "State of Charge", COLOR["green"]),
                TraceSpec("aps_internal_temp_c", "Internal Temperature", COLOR["red"], axis="right"),
            ),
        ),
        PanelSpec(
            "local_coverage",
            "Local Mirror Coverage",
            "Coverage [%]",
            None,
            (
                TraceSpec("cl61_local_coverage_pct", "CL61", COLOR["teal"]),
                TraceSpec("radar_local_coverage_pct", "Radar", COLOR["blue"]),
                TraceSpec("hatpro_local_coverage_pct", "HATPRO", COLOR["black"]),
                TraceSpec("vaisalamet_local_coverage_pct", "Meteorology", COLOR["green"]),
                TraceSpec("asfs_logger_local_coverage_pct", "Radiation", COLOR["purple"]),
                TraceSpec("asfs_fast_sonic_local_coverage_pct", "ASFS Fast Sonic", COLOR["magenta"]),
                TraceSpec("power_local_coverage_pct", "Aurora Power Supply", COLOR["brown"]),
                TraceSpec("pdu_local_coverage_pct", "ASS PDU", COLOR["red"]),
                TraceSpec("wxcam_local_coverage_pct", "WXcam", COLOR["olive"]),
            ),
        ),
        PanelSpec(
            "local_lag",
            "Local Mirror Lag",
            "Lag [min]",
            None,
            (
                TraceSpec("cl61_local_lag_min", "CL61", COLOR["teal"]),
                TraceSpec("radar_local_lag_min", "Radar", COLOR["blue"]),
                TraceSpec("hatpro_local_lag_min", "HATPRO", COLOR["black"]),
                TraceSpec("vaisalamet_local_lag_min", "Meteorology", COLOR["green"]),
                TraceSpec("asfs_logger_local_lag_min", "Radiation", COLOR["purple"]),
                TraceSpec("asfs_fast_sonic_local_lag_min", "ASFS Fast Sonic", COLOR["magenta"]),
                TraceSpec("power_local_lag_min", "Aurora Power Supply", COLOR["brown"]),
                TraceSpec("pdu_local_lag_min", "ASS PDU", COLOR["red"]),
                TraceSpec("wxcam_local_lag_min", "WXcam", COLOR["olive"]),
            ),
        ),
        PanelSpec(
            "gws_coverage",
            "JASMIN Mirror Coverage",
            "Coverage [%]",
            None,
            (
                TraceSpec("cl61_gws_coverage_pct", "CL61", COLOR["teal"]),
                TraceSpec("radar_gws_coverage_pct", "Radar", COLOR["blue"]),
                TraceSpec("hatpro_gws_coverage_pct", "HATPRO", COLOR["black"]),
                TraceSpec("vaisalamet_gws_coverage_pct", "Meteorology", COLOR["green"]),
                TraceSpec("asfs_logger_gws_coverage_pct", "Radiation", COLOR["purple"]),
                TraceSpec("asfs_fast_sonic_gws_coverage_pct", "ASFS Fast Sonic", COLOR["magenta"]),
                TraceSpec("power_gws_coverage_pct", "Aurora Power Supply", COLOR["brown"]),
                TraceSpec("pdu_gws_coverage_pct", "ASS PDU", COLOR["red"]),
                TraceSpec("wxcam_gws_coverage_pct", "WXcam", COLOR["olive"]),
            ),
        ),
        PanelSpec(
            "prune_gates",
            "Prune / Product Gates",
            "Stream Count",
            "State",
            (
                TraceSpec("streams_product_gate_ok_count", "Product Gates OK", COLOR["teal"]),
                TraceSpec("streams_prune_ready_count", "Prune Ready", COLOR["blue"]),
                TraceSpec("streams_local_issue_count", "Local Mirror Issues", COLOR["brown"]),
                TraceSpec("streams_gws_issue_count", "JASMIN Mirror Issues", COLOR["red"]),
                TraceSpec("gws_available_state", "JASMIN Available", COLOR["green"], axis="right", step=True),
            ),
        ),
        PanelSpec(
            "service_health",
            "Service Health",
            "Failure Count",
            "State",
            (
                TraceSpec("source_host_probe_fail_count", "Source Host Probe Failures", COLOR["red"]),
                TraceSpec("failed_source_sync_unit_count", "Source Sync Failures", COLOR["brown"]),
                TraceSpec("failed_processing_unit_count", "Processing Failures", COLOR["purple"]),
                TraceSpec("failed_transfer_unit_count", "Transfer Failures", COLOR["magenta"]),
                TraceSpec("mirror_verify_timer_active_state", "Mirror Verify Timer", COLOR["teal"], axis="right", step=True),
            ),
        ),
    ),
    # Archived Operations housekeeping should focus on exceptions and drift,
    # not on every raw metric collected in the live snapshot stream.
    "ops-monitor-hk": (
        PanelSpec(
            "storage_use",
            "Storage Use",
            "Used [%]",
            None,
            (
                TraceSpec("host_celine_source_used_pct", "CL61 Root Disk", COLOR["teal"]),
                TraceSpec("host_celine_data_used_pct", "CL61 Data Disk", COLOR["blue"]),
                TraceSpec("host_ass_data_used_pct", "ASS Data Disk", COLOR["slate"]),
                TraceSpec("host_ass_root_used_pct", "ASS Root Disk", COLOR["purple"]),
                TraceSpec("host_aps_data_used_pct", "APS Data Disk", COLOR["brown"]),
                TraceSpec("host_aps_root_used_pct", "APS Root Disk", COLOR["olive"]),
                TraceSpec("aurora_project_used_pct", "Aurora Raw (/project)", COLOR["olive"]),
                TraceSpec("aurora_data_used_pct", "AURORA Cloud Products (/data/aurora)", COLOR["magenta"]),
                TraceSpec("gws_storage_used_pct", "JASMIN GWS", COLOR["green"]),
            ),
        ),
        PanelSpec(
            "mirror_issue_counts",
            "Mirror Issue Counts",
            "Local Issues [count]",
            "GWS Issues [count]",
            (
                TraceSpec("cl61_local_issue_count", "CL61 Local", COLOR["teal"], step=True, skip_if_all_zero=True),
                TraceSpec("radar_local_issue_count", "Radar Local", COLOR["blue"], step=True, skip_if_all_zero=True),
                TraceSpec("hatpro_local_issue_count", "HATPRO Local", COLOR["black"], step=True, skip_if_all_zero=True),
                TraceSpec("vaisalamet_local_issue_count", "Meteorology Local", COLOR["green"], step=True, skip_if_all_zero=True),
                TraceSpec("asfs_logger_local_issue_count", "Radiation Local", COLOR["purple"], step=True, skip_if_all_zero=True),
                TraceSpec("asfs_fast_sonic_local_issue_count", "ASFS Fast Sonic Local", COLOR["magenta"], step=True, skip_if_all_zero=True),
                TraceSpec("power_local_issue_count", "APS Local", COLOR["brown"], step=True, skip_if_all_zero=True),
                TraceSpec("pdu_local_issue_count", "PDU Local", COLOR["red"], step=True, skip_if_all_zero=True),
                TraceSpec("cl61_gws_issue_count", "CL61 GWS", COLOR["teal"], axis="right", dash="dot", step=True, skip_if_all_zero=True),
                TraceSpec("radar_gws_issue_count", "Radar GWS", COLOR["blue"], axis="right", dash="dot", step=True, skip_if_all_zero=True),
                TraceSpec("hatpro_gws_issue_count", "HATPRO GWS", COLOR["black"], axis="right", dash="dot", step=True, skip_if_all_zero=True),
                TraceSpec("vaisalamet_gws_issue_count", "Meteorology GWS", COLOR["green"], axis="right", dash="dot", step=True, skip_if_all_zero=True),
                TraceSpec("asfs_logger_gws_issue_count", "Radiation GWS", COLOR["purple"], axis="right", dash="dot", step=True, skip_if_all_zero=True),
                TraceSpec("asfs_fast_sonic_gws_issue_count", "ASFS Fast Sonic GWS", COLOR["magenta"], axis="right", dash="dot", step=True, skip_if_all_zero=True),
                TraceSpec("power_gws_issue_count", "APS GWS", COLOR["brown"], axis="right", dash="dot", step=True, skip_if_all_zero=True),
                TraceSpec("pdu_gws_issue_count", "PDU GWS", COLOR["red"], axis="right", dash="dot", step=True, skip_if_all_zero=True),
            ),
        ),
        PanelSpec(
            "mirror_lag",
            "Mirror Lag Outliers",
            "Local Lag [min]",
            "GWS Lag [min]",
            (
                TraceSpec("cl61_local_lag_min", "CL61 Local", COLOR["teal"], valid_min=2.0),
                TraceSpec("radar_local_lag_min", "Radar Local", COLOR["blue"], valid_min=2.0),
                TraceSpec("hatpro_local_lag_min", "HATPRO Local", COLOR["black"], valid_min=2.0),
                TraceSpec("vaisalamet_local_lag_min", "Meteorology Local", COLOR["green"], valid_min=2.0),
                TraceSpec("asfs_logger_local_lag_min", "Radiation Local", COLOR["purple"], valid_min=2.0),
                TraceSpec("asfs_fast_sonic_local_lag_min", "ASFS Fast Sonic Local", COLOR["magenta"], valid_min=2.0),
                TraceSpec("power_local_lag_min", "APS Local", COLOR["brown"], valid_min=2.0),
                TraceSpec("pdu_local_lag_min", "PDU Local", COLOR["red"], valid_min=2.0),
                TraceSpec("wxcam_local_lag_min", "WXcam Local", COLOR["olive"], valid_min=2.0),
                TraceSpec("cl61_gws_lag_min", "CL61 GWS", COLOR["teal"], axis="right", dash="dot", valid_min=10.0),
                TraceSpec("radar_gws_lag_min", "Radar GWS", COLOR["blue"], axis="right", dash="dot", valid_min=10.0),
                TraceSpec("hatpro_gws_lag_min", "HATPRO GWS", COLOR["black"], axis="right", dash="dot", valid_min=10.0),
                TraceSpec("vaisalamet_gws_lag_min", "Meteorology GWS", COLOR["green"], axis="right", dash="dot", valid_min=10.0),
                TraceSpec("asfs_logger_gws_lag_min", "Radiation GWS", COLOR["purple"], axis="right", dash="dot", valid_min=10.0),
                TraceSpec("asfs_fast_sonic_gws_lag_min", "ASFS Fast Sonic GWS", COLOR["magenta"], axis="right", dash="dot", valid_min=10.0),
                TraceSpec("power_gws_lag_min", "APS GWS", COLOR["brown"], axis="right", dash="dot", valid_min=10.0),
                TraceSpec("pdu_gws_lag_min", "PDU GWS", COLOR["red"], axis="right", dash="dot", valid_min=10.0),
                TraceSpec("wxcam_gws_lag_min", "WXcam GWS", COLOR["olive"], axis="right", dash="dot", valid_min=10.0),
            ),
        ),
        PanelSpec(
            "wxcam_backfill",
            "WXcam Backfill",
            "Coverage [%]",
            "Issue Count",
            (
                TraceSpec("wxcam_local_coverage_pct", "WXcam Local Coverage", COLOR["olive"]),
                TraceSpec("wxcam_gws_coverage_pct", "WXcam GWS Coverage", COLOR["green"]),
                TraceSpec("wxcam_local_issue_count", "WXcam Local Issues", COLOR["brown"], axis="right", step=True, skip_if_all_zero=True),
                TraceSpec("wxcam_gws_issue_count", "WXcam GWS Issues", COLOR["red"], axis="right", step=True, skip_if_all_zero=True),
            ),
        ),
        PanelSpec(
            "gate_blocks",
            "Blocked Streams",
            "Blocked Streams [count]",
            None,
            (
                TraceSpec("streams_product_gate_block_count", "Product Gate Blocked", COLOR["teal"], step=True, skip_if_all_zero=True),
                TraceSpec("streams_prune_block_count", "Prune Blocked", COLOR["blue"], step=True, skip_if_all_zero=True),
                TraceSpec("streams_local_issue_count", "Streams with Local Issues", COLOR["brown"], step=True, skip_if_all_zero=True),
                TraceSpec("streams_gws_issue_count", "Streams with GWS Issues", COLOR["red"], step=True, skip_if_all_zero=True),
            ),
        ),
        PanelSpec(
            "source_freshness",
            "Source Freshness",
            "Stale Streams [count]",
            "Recent State",
            (
                TraceSpec("streams_source_stale_count", "Stale Streams", COLOR["red"], step=True, skip_if_all_zero=True),
                TraceSpec("cl61_source_recent_state", "CL61 Recent", COLOR["teal"], axis="right", step=True, skip_if_all_zero=True),
                TraceSpec("radar_source_recent_state", "Radar Recent", COLOR["blue"], axis="right", step=True, skip_if_all_zero=True),
                TraceSpec("hatpro_source_recent_state", "HATPRO Recent", COLOR["black"], axis="right", step=True, skip_if_all_zero=True),
                TraceSpec("vaisalamet_source_recent_state", "Meteorology Recent", COLOR["green"], axis="right", step=True, skip_if_all_zero=True),
                TraceSpec("asfs_logger_source_recent_state", "Radiation Recent", COLOR["purple"], axis="right", step=True, skip_if_all_zero=True),
                TraceSpec("asfs_fast_sonic_source_recent_state", "ASFS Fast Sonic Recent", COLOR["magenta"], axis="right", step=True, skip_if_all_zero=True),
                TraceSpec("power_source_recent_state", "APS Recent", COLOR["brown"], axis="right", step=True, skip_if_all_zero=True),
                TraceSpec("pdu_source_recent_state", "PDU Recent", COLOR["red"], axis="right", step=True, skip_if_all_zero=True),
                TraceSpec("wxcam_source_recent_state", "WXcam Recent", COLOR["olive"], axis="right", step=True, skip_if_all_zero=True),
            ),
        ),
        PanelSpec(
            "service_problems",
            "Service / Transfer Problems",
            "Failure Count",
            "Problem State",
            (
                TraceSpec("source_host_probe_fail_count", "Source Host Probe Failures", COLOR["red"], step=True, skip_if_all_zero=True),
                TraceSpec("failed_source_sync_unit_count", "Source Sync Failures", COLOR["brown"], step=True, skip_if_all_zero=True),
                TraceSpec("failed_processing_unit_count", "Processing Failures", COLOR["purple"], step=True, skip_if_all_zero=True),
                TraceSpec("failed_transfer_unit_count", "Transfer Failures", COLOR["magenta"], step=True, skip_if_all_zero=True),
                TraceSpec("gws_unavailable_state", "GWS Unavailable", COLOR["green"], axis="right", step=True, skip_if_all_zero=True),
                TraceSpec("mirror_verify_problem_state", "Mirror Verify Problem", COLOR["teal"], axis="right", step=True, skip_if_all_zero=True),
                TraceSpec("ops_monitor_append_problem_state", "Ops Append Problem", COLOR["blue"], axis="right", step=True, skip_if_all_zero=True),
                TraceSpec("ops_monitor_alerts_problem_state", "Ops Alert Problem", COLOR["black"], axis="right", step=True, skip_if_all_zero=True),
                TraceSpec("ops_monitor_quicklooks_problem_state", "Ops Quicklook Problem", COLOR["slate"], axis="right", step=True, skip_if_all_zero=True),
                TraceSpec("dashboard_perf_log_stale_state", "Dashboard Perf Log Stale", COLOR["olive"], axis="right", step=True, skip_if_all_zero=True),
            ),
        ),
        PanelSpec(
            "dashboard_perf_log",
            "Dashboard Performance Log",
            "Age [min]",
            None,
            (
                TraceSpec("dashboard_perf_log_age_min", "Perf Log Age", COLOR["olive"], valid_min=30.0),
            ),
        ),
    ),
}

CURATED_HOUSEKEEPING_LAYOUTS = {
    "asfs-logger": "asfs-logger-hk",
}


def is_summary_instrument(instrument: str) -> bool:
    return instrument in SUMMARY_INSTRUMENTS


def display_name(instrument: str) -> str:
    return DISPLAY_NAMES.get(instrument, instrument)


def housekeeping_label(instrument: str) -> str | None:
    return HOUSEKEEPING_LABELS.get(instrument)


def summary_source_instruments(instrument: str) -> tuple[str, ...]:
    return SUMMARY_SOURCE_INSTRUMENTS.get(instrument, (instrument,))


def default_interactive_label(instrument: str) -> str:
    return OVERVIEW_LABEL


def default_calendar_label(instrument: str) -> str:
    """Return the default dated quicklook group label.

    The function name is kept for compatibility with older app code that called
    the dated PNG browser a calendar; the visible UI is now Science Quicklooks.
    """
    return OVERVIEW_LABEL


def widget_group_options(instrument: str) -> OrderedDict[str, dict[str, object]]:
    return OrderedDict(
        [
            (
                OVERVIEW_LABEL,
                {
                    "label": OVERVIEW_LABEL,
                    "clim": (0.0, 1.0),
                    "log": False,
                    "colorscale": "Viridis",
                },
            )
        ]
    )


def quicklook_prefix(instrument: str) -> str:
    return QUICKLOOK_PREFIX[instrument]


def summary_latest_png(quicklook_dir: Path, instrument: str) -> Path:
    return quicklook_dir / f"{quicklook_prefix(instrument)}__summary__latest.png"


def summary_daily_png(quicklook_dir: Path, instrument: str, day: pd.Timestamp | str) -> Path:
    stamp = pd.Timestamp(day).strftime("%Y%m%d")
    return quicklook_dir / f"{quicklook_prefix(instrument)}__summary__{stamp}.png"


def housekeeping_latest_png(quicklook_dir: Path, instrument: str) -> Path | None:
    label = housekeeping_label(instrument)
    if label is None:
        return None
    key = label.lower()
    return quicklook_dir / f"{quicklook_prefix(instrument)}__{key}__latest.png"


def housekeeping_daily_png(quicklook_dir: Path, instrument: str, day: pd.Timestamp | str) -> Path | None:
    label = housekeeping_label(instrument)
    if label is None:
        return None
    key = label.lower()
    stamp = pd.Timestamp(day).strftime("%Y%m%d")
    return quicklook_dir / f"{quicklook_prefix(instrument)}__{key}__{stamp}.png"


def legacy_latest_png(quicklook_dir: Path, instrument: str) -> Path | None:
    prefix = LEGACY_ALIAS_PREFIX.get(instrument)
    if prefix is None:
        return None
    return quicklook_dir / "latest.png"


def legacy_daily_png(quicklook_dir: Path, instrument: str, day: pd.Timestamp | str) -> Path | None:
    prefix = LEGACY_ALIAS_PREFIX.get(instrument)
    if prefix is None:
        return None
    stamp = pd.Timestamp(day).strftime("%Y%m%d")
    return quicklook_dir / f"{prefix}_{stamp}.png"


def clear_generated_quicklooks(quicklook_dir: Path, instrument: str) -> None:
    prefix = quicklook_prefix(instrument)
    for png in quicklook_dir.glob(f"{prefix}*.png"):
        png.unlink()
    legacy_prefix = LEGACY_ALIAS_PREFIX.get(instrument)
    if legacy_prefix:
        for png in quicklook_dir.glob(f"{legacy_prefix}_*.png"):
            png.unlink()
        legacy_latest = legacy_latest_png(quicklook_dir, instrument)
        if legacy_latest and legacy_latest.exists():
            legacy_latest.unlink()


def refresh_legacy_aliases(
    quicklook_dir: Path,
    instrument: str,
    day_png: Path | None = None,
    latest_png: Path | None = None,
) -> None:
    if day_png is not None:
        token = day_png.stem.rsplit("__", 1)[-1]
        legacy_day = legacy_daily_png(quicklook_dir, instrument, token)
        if legacy_day:
            shutil.copyfile(day_png, legacy_day)
    if latest_png is not None:
        legacy_latest = legacy_latest_png(quicklook_dir, instrument)
        if legacy_latest:
            shutil.copyfile(latest_png, legacy_latest)


def calendar_date_tokens(quicklook_dir: Path, instrument: str) -> list[str]:
    """List daily Science Quicklook tokens.

    The public UI no longer uses the word calendar, but these helper names are
    retained because existing quicklook and performance code still imports them.
    """
    prefix = quicklook_prefix(instrument)
    tokens: list[str] = []
    for png in sorted(quicklook_dir.glob(f"{prefix}__summary__*.png")):
        suffix = png.stem.split("__")[-1]
        if suffix == "latest":
            continue
        tokens.append(suffix)
    return tokens


def calendar_product_paths(quicklook_dir: Path, instrument: str, token: str) -> list[tuple[str, Path]]:
    """Return science and housekeeping PNGs associated with a dated token."""
    paths: list[tuple[str, Path]] = []
    if token == "latest":
        summary = summary_latest_png(quicklook_dir, instrument)
        if summary.exists():
            paths.append((display_name(instrument), summary))
        hk = housekeeping_latest_png(quicklook_dir, instrument)
        if hk and hk.exists():
            paths.append((housekeeping_label(instrument) or "Housekeeping", hk))
        return paths

    summary = summary_daily_png(quicklook_dir, instrument, token)
    if summary.exists():
        paths.append((display_name(instrument), summary))
    hk = housekeeping_daily_png(quicklook_dir, instrument, token)
    if hk and hk.exists():
        paths.append((housekeeping_label(instrument) or "Housekeeping", hk))
    return paths


def human_label(name: str) -> str:
    if name in HUMAN_LABELS:
        return HUMAN_LABELS[name]
    tokens = [token for token in name.split("_") if token]
    if tokens and tokens[-1] in {"pct", "gb", "count", "min", "state", "bytes"}:
        tokens = tokens[:-1]
    token_map = {
        "ac": "AC",
        "aps": "APS",
        "asfs": "ASFS",
        "cl61": "CL61",
        "gws": "GWS",
        "hk": "HK",
        "radar": "Radar",
        "utc": "UTC",
        "wxcam": "WXcam",
    }
    parts = [token_map.get(token.lower(), token.replace("-", " ").title()) for token in tokens]
    return " ".join(parts) if parts else name.replace("_", " ")


def human_unit(name: str) -> str | None:
    if name in HUMAN_UNITS:
        return HUMAN_UNITS[name]
    lower = name.lower()
    if lower.endswith("_pct"):
        return "%"
    if lower.endswith("_gb"):
        return "GB"
    if lower.endswith("_min"):
        return "min"
    if lower.endswith("_count"):
        return "count"
    if lower.endswith("_state"):
        return "state"
    if lower.endswith("_bytes"):
        return "B"
    if "volt" in lower:
        return "V"
    if "watt" in lower:
        return "W"
    if "amp" in lower:
        return "A"
    if lower.endswith("hz") or "_hz" in lower:
        return "Hz"
    if "yield" in lower or lower.endswith("kwh"):
        return "kWh"
    if "temp" in lower or lower.endswith(("_t", "_td")) or "_amb_" in lower or "_tem_" in lower:
        return "C"
    if lower.endswith("_rh") or "_rh_" in lower or lower == "batterystate":
        return "%"
    if lower.startswith("baro") or lower.endswith("_hpa") or lower == "h1_e":
        return "hPa"
    if "dist" in lower:
        return "m"
    if "incl" in lower:
        return "deg"
    if "msec" in lower:
        return "ms"
    if lower == "scantime" or lower == "time_discrepancy":
        return "s"
    if "co2" in lower and "str" not in lower:
        return "ppm"
    if "h2o" in lower:
        return "mmol mol^-1"
    if "co2_str" in lower:
        return "%"
    if any(token in lower for token in STATUS_TOKENS):
        return "state"
    return None


def human_axis_label(name: str) -> str:
    label = human_label(name)
    unit = human_unit(name)
    return f"{label} [{unit}]" if unit else label


def display_scale(name: str) -> float:
    return DISPLAY_SCALE.get(name, 1.0)


def summary_trace_vars(instrument: str) -> set[str]:
    return {trace.var for panel in SUMMARY_LAYOUTS.get(instrument, ()) for trace in panel.traces}


def combine_summary_datasets(instrument: str, *datasets: xr.Dataset | None) -> xr.Dataset:
    """Merge 1D time-series sources, using later sources to fill gaps.

    This is used for summary instruments whose displayed variables may come
    from more than one Zarr store. Existing values keep priority, while NaNs
    or missing times can be filled by an independent source such as ASFS fast
    sonic.
    """
    merged_inputs: list[xr.Dataset] = []
    for ds in datasets:
        if ds is None or "time" not in ds or ds.sizes.get("time", 0) == 0:
            continue
        keep_names = [name for name, da in ds.data_vars.items() if da.dims == ("time",)]
        if not keep_names:
            continue
        subset = ds[keep_names].sortby("time")
        merged_inputs.append(subset)
    if not merged_inputs:
        return xr.Dataset()
    merged = merged_inputs[0]
    for subset in merged_inputs[1:]:
        merged, aligned = xr.align(merged, subset, join="outer")
        assignments = {}
        for name, da in aligned.data_vars.items():
            if name in merged.data_vars:
                filled = merged[name].combine_first(da)
                filled.attrs = dict(merged[name].attrs)
                assignments[name] = filled
            else:
                assignments[name] = da
        if assignments:
            merged = merged.assign(**assignments)
    merged = merged.sortby("time")
    merged.attrs["summary_instrument"] = instrument
    return merged


def fast_sonic_metek_summary_dataset(ds: xr.Dataset, freq: str = "1min") -> xr.Dataset:
    """Resample high-rate ASFS fast-sonic Metek fields onto summary names."""
    if ds is None or "time" not in ds or ds.sizes.get("time", 0) == 0:
        return xr.Dataset()
    if any(name in ds.data_vars for name in FAST_SONIC_TO_LOGGER_AVG.values()):
        keep = [
            name
            for name in FAST_SONIC_TO_LOGGER_AVG.values()
            if name in ds.data_vars and ds[name].dims == ("time",)
        ]
        return ds[keep].sortby("time") if keep else xr.Dataset()
    keep = [name for name in FAST_SONIC_TO_LOGGER_AVG if name in ds and ds[name].dims == ("time",)]
    if not keep:
        return xr.Dataset()
    frame = pd.DataFrame(
        {FAST_SONIC_TO_LOGGER_AVG[name]: np.asarray(ds[name].values, dtype=np.float64) for name in keep},
        index=pd.DatetimeIndex(ds["time"].values),
    )
    frame = frame[~frame.index.isna()].sort_index()
    frame = frame.resample(freq).mean().dropna(how="all")
    if frame.empty:
        return xr.Dataset()
    return xr.Dataset(
        {name: (("time",), frame[name].to_numpy(dtype=np.float32)) for name in frame.columns},
        coords={"time": frame.index.to_numpy(dtype="datetime64[ns]")},
        attrs={"source": "derived from ASFS fast-sonic high-rate Metek fields", "frequency": freq},
    )


def fast_gas_licor_summary_dataset(ds: xr.Dataset, freq: str = "1min") -> xr.Dataset:
    """Resample high-rate ASFS fast-gas LI-COR fields onto summary names."""
    if ds is None or "time" not in ds or ds.sizes.get("time", 0) == 0:
        return xr.Dataset()
    if any(name in ds.data_vars for name in FAST_GAS_TO_LOGGER_AVG.values()):
        keep = [
            name
            for name in FAST_GAS_TO_LOGGER_AVG.values()
            if name in ds.data_vars and ds[name].dims == ("time",)
        ]
        return ds[keep].sortby("time") if keep else xr.Dataset()
    keep = [name for name in FAST_GAS_TO_LOGGER_AVG if name in ds and ds[name].dims == ("time",)]
    if not keep:
        return xr.Dataset()
    frame = pd.DataFrame(
        {FAST_GAS_TO_LOGGER_AVG[name]: np.asarray(ds[name].values, dtype=np.float64) for name in keep},
        index=pd.DatetimeIndex(ds["time"].values),
    )
    frame = frame[~frame.index.isna()].sort_index()
    frame = frame.resample(freq).mean().dropna(how="all")
    if frame.empty:
        return xr.Dataset()
    return xr.Dataset(
        {name: (("time",), frame[name].to_numpy(dtype=np.float32)) for name in frame.columns},
        coords={"time": frame.index.to_numpy(dtype="datetime64[ns]")},
        attrs={"source": "derived from ASFS fast-gas LI-COR fields", "frequency": freq},
    )


def augment_meteorology_from_fast_sonic(ds: xr.Dataset) -> xr.Dataset:
    """Fill Meteorology Metek summary fields from the high-rate sonic stream.

    The ASFS science/logger stream carries one-minute Metek averages alongside
    radiation data. When that slow table has a source gap, the independent
    fast-sonic files can still provide the same Metek components. This helper
    maps those raw fast-sonic variables onto the one-minute summary names and
    only fills places where the slow-table values are missing.
    """
    if ds is None or "time" not in ds or ds.sizes.get("time", 0) == 0:
        return ds
    assignments: dict[str, xr.DataArray] = {}
    for source_name, target_name in FAST_SONIC_TO_LOGGER_AVG.items():
        if source_name not in ds or ds[source_name].dims != ("time",):
            continue
        source = ds[source_name].copy(deep=False)
        source.attrs = dict(source.attrs)
        source.attrs["derived_from"] = source_name
        if target_name in ds and ds[target_name].dims == ("time",):
            filled = ds[target_name].combine_first(source)
            filled.attrs = dict(ds[target_name].attrs)
            filled.attrs["gap_fill_source"] = source_name
            assignments[target_name] = filled
        else:
            source.name = target_name
            assignments[target_name] = source
    return ds.assign(**assignments) if assignments else ds


def augment_asfs_from_fast_gas(ds: xr.Dataset) -> xr.Dataset:
    """Fill ASFS LI-COR summary fields from the independent fast-gas stream."""
    if ds is None or "time" not in ds or ds.sizes.get("time", 0) == 0:
        return ds
    assignments: dict[str, xr.DataArray] = {}
    for source_name, target_name in FAST_GAS_TO_LOGGER_AVG.items():
        if source_name not in ds or ds[source_name].dims != ("time",):
            continue
        source = ds[source_name].copy(deep=False)
        source.attrs = dict(source.attrs)
        source.attrs["derived_from"] = source_name
        if target_name in ds and ds[target_name].dims == ("time",):
            filled = ds[target_name].combine_first(source)
            filled.attrs = dict(ds[target_name].attrs)
            filled.attrs["gap_fill_source"] = source_name
            assignments[target_name] = filled
        else:
            source.name = target_name
            assignments[target_name] = source
    return ds.assign(**assignments) if assignments else ds


def _daily_cumulative_energy_kwh(times: pd.DatetimeIndex, power_w: np.ndarray) -> np.ndarray:
    """Integrate power to kWh, resetting the displayed total at each UTC day."""
    cumulative_kwh = np.zeros(len(times), dtype=np.float64)
    if len(times) <= 1:
        return cumulative_kwh

    day_starts = times.normalize()
    time_ns = times.asi8.astype(np.float64)
    for idx in range(1, len(times)):
        if day_starts[idx] != day_starts[idx - 1]:
            # Start each day visibly from zero rather than carrying yesterday's
            # utilised energy into the new UTC day.
            cumulative_kwh[idx] = 0.0
            continue
        dt_hours = max((time_ns[idx] - time_ns[idx - 1]) / 3.6e12, 0.0)
        incremental_kwh = 0.5 * (power_w[idx] + power_w[idx - 1]) * dt_hours / 1000.0
        cumulative_kwh[idx] = cumulative_kwh[idx - 1] + incremental_kwh
    return cumulative_kwh


def _daily_cumulative_counter_delta(times: pd.DatetimeIndex, counter_kwh: np.ndarray) -> np.ndarray:
    """Convert daily-ish energy counters into UTC-day cumulative increments.

    The APS solar-yield counters can reset tens of minutes after midnight. For
    display, accumulate only positive counter changes within each UTC day and
    ignore reset drops, so the plotted generation starts cleanly at midnight.
    """
    cumulative_kwh = np.full(len(times), np.nan, dtype=np.float64)
    if len(times) == 0:
        return cumulative_kwh

    day_starts = times.normalize()
    current_day = None
    running_total = 0.0
    last_value = np.nan
    for idx, (day_start, raw_value) in enumerate(zip(day_starts, counter_kwh, strict=False)):
        if current_day is None or day_start != current_day:
            current_day = day_start
            running_total = 0.0
            last_value = np.nan
        if not np.isfinite(raw_value):
            continue
        if np.isfinite(last_value):
            delta = float(raw_value - last_value)
            if delta > 0.0:
                running_total += delta
        cumulative_kwh[idx] = running_total
        last_value = float(raw_value)
    return cumulative_kwh


def _display_energy_assignments(ds: xr.Dataset) -> dict[str, xr.DataArray]:
    """Map compact Power display-energy variables onto the standard plot names."""
    assignments: dict[str, xr.DataArray] = {}
    for target_name, source_name in POWER_DISPLAY_ENERGY_MAP.items():
        if source_name not in ds:
            continue
        da = ds[source_name].copy(deep=False)
        da.attrs = dict(da.attrs)
        da.attrs["units"] = "kWh"
        assignments[target_name] = da
    return assignments


def build_power_display_energy_dataset(
    ds: xr.Dataset,
    freq: str = POWER_DISPLAY_ENERGY_FREQ,
) -> xr.Dataset:
    """Build a compact Power display product for cumulative energy traces.

    The raw APS Zarr stays authoritative. This derived product stores only the
    one-minute cumulative kWh traces needed by the dashboard so interactive
    plotting does not need to read many days of one-second samples to compute
    solar generation and utilised energy.
    """
    if "time" not in ds or ds.sizes.get("time", 0) == 0:
        return xr.Dataset()

    ds = ds.sortby("time")
    times = pd.DatetimeIndex(ds["time"].values)
    frame: dict[str, np.ndarray] = {}
    generated_arrays: list[np.ndarray] = []
    for field_name in ("SolarYield_East", "SolarYield_South", "SolarYield_West"):
        if field_name not in ds:
            continue
        generated = _daily_cumulative_counter_delta(times, np.asarray(ds[field_name].values, dtype=np.float64))
        frame[POWER_DISPLAY_ENERGY_MAP[field_name]] = generated
        generated_arrays.append(generated)

    if generated_arrays:
        valid_generated = np.zeros(len(times), dtype=bool)
        for values in generated_arrays:
            valid_generated |= np.isfinite(values)
        total_generated = np.full(len(times), np.nan, dtype=np.float64)
        if np.any(valid_generated):
            summed = np.zeros(int(np.count_nonzero(valid_generated)), dtype=np.float64)
            for values in generated_arrays:
                summed += np.nan_to_num(values[valid_generated], nan=0.0)
            total_generated[valid_generated] = summed
        frame[POWER_DISPLAY_ENERGY_MAP["CumulativePowerGeneratedTotal"]] = total_generated
    else:
        total_generated = np.full(len(times), np.nan, dtype=np.float64)

    if "ACOutputWatts" in ds or "DCInverterWatts" in ds:
        ac_power = np.asarray(
            ds["ACOutputWatts"].values if "ACOutputWatts" in ds else np.full(len(times), np.nan),
            dtype=np.float64,
        )
        dc_power = np.asarray(
            ds["DCInverterWatts"].values if "DCInverterWatts" in ds else np.full(len(times), np.nan),
            dtype=np.float64,
        )
        valid_power = np.isfinite(ac_power) | np.isfinite(dc_power)
        utilised = np.full(len(times), np.nan, dtype=np.float64)
        if np.any(valid_power):
            utilised_power_w = np.nan_to_num(ac_power[valid_power], nan=0.0) + np.nan_to_num(dc_power[valid_power], nan=0.0)
            utilised_power_w = np.clip(utilised_power_w, a_min=0.0, a_max=None)
            utilised[valid_power] = _daily_cumulative_energy_kwh(times[valid_power], utilised_power_w)
        frame[POWER_DISPLAY_ENERGY_MAP["CumulativePowerUtilised"]] = utilised
    else:
        utilised = np.full(len(times), np.nan, dtype=np.float64)

    if not frame:
        return xr.Dataset()

    display_frame = pd.DataFrame(frame, index=times).resample(freq).last().dropna(how="all")
    if display_frame.empty:
        return xr.Dataset()
    out = xr.Dataset(
        {name: (("time",), display_frame[name].to_numpy(dtype=np.float32)) for name in display_frame.columns},
        coords={"time": display_frame.index.to_numpy(dtype="datetime64[ns]")},
        attrs={
            POWER_DISPLAY_ENERGY_ATTR: "true",
            "source": "derived from power.zarr",
            "frequency": freq,
            "description": "Display-only one-minute cumulative APS energy traces for dashboard plotting.",
        },
    )
    for name in out.data_vars:
        out[name].attrs["units"] = "kWh"
    return out


def _time_frame_from_dataset(ds: xr.Dataset, fields: tuple[str, ...]) -> pd.DataFrame:
    """Load selected 1D time-series fields into a sorted pandas frame."""
    if ds is None or "time" not in ds or ds.sizes.get("time", 0) == 0:
        return pd.DataFrame()
    names = [name for name in fields if name in ds and ds[name].dims == ("time",)]
    if not names:
        return pd.DataFrame()
    times = pd.DatetimeIndex(ds["time"].values)
    frame = pd.DataFrame(
        {name: np.asarray(ds[name].values, dtype=np.float64) for name in names},
        index=times,
    )
    frame = frame[~frame.index.isna()].sort_index()
    return frame[~frame.index.duplicated(keep="last")]


def _operating_scenario_frame(ds: xr.Dataset | None) -> pd.DataFrame:
    """Flatten named scenario/time fields into display-summary time series."""
    if (
        ds is None
        or str(ds.attrs.get("planning_status", "ready")).strip() != "ready"
        or "time" not in ds
        or "scenario" not in ds
        or ds.sizes.get("time", 0) == 0
    ):
        return pd.DataFrame()
    scenario_ids = [str(value) for value in ds["scenario"].values]
    times = pd.DatetimeIndex(ds["time"].values)
    values: dict[str, np.ndarray] = {}
    for scenario_id, prefix in OPERATING_SCENARIO_PREFIXES.items():
        if scenario_id not in scenario_ids:
            continue
        index = scenario_ids.index(scenario_id)
        for source_name, suffix in OPERATING_SCENARIO_SOURCE_FIELDS:
            if source_name not in ds or ds[source_name].dims != ("scenario", "time"):
                continue
            values[f"{prefix}{suffix}"] = np.asarray(ds[source_name].isel(scenario=index).values, dtype=np.float64)
        if scenario_id == "optimized_cl61" and "ScenarioModeCode" in ds:
            codes = np.asarray(ds["ScenarioModeCode"].isel(scenario=index).values, dtype=np.int64)
            values["OperatingCL61OptimizedCL61On"] = ((codes & 1) > 0).astype(np.float64)
            values["OperatingCL61OptimizedRadarOn"] = ((codes & 2) > 0).astype(np.float64)
            values["OperatingCL61OptimizedHATPROOn"] = ((codes & 4) > 0).astype(np.float64)
            values["OperatingCL61OptimizedActiveCount"] = (
                values["OperatingCL61OptimizedCL61On"]
                + values["OperatingCL61OptimizedRadarOn"]
                + values["OperatingCL61OptimizedHATPROOn"]
            )
    for scenario_id, prefix in OPERATING_SUGGESTED_PREFIXES.items():
        if scenario_id not in scenario_ids:
            continue
        index = scenario_ids.index(scenario_id)
        for source_name, suffix in OPERATING_SCENARIO_SOURCE_FIELDS:
            if source_name not in ds or ds[source_name].dims != ("scenario", "time"):
                continue
            values[f"{prefix}{suffix}"] = np.asarray(
                ds[source_name].isel(scenario=index).values,
                dtype=np.float64,
            )
    current_mode = str(ds.attrs.get("current_mode", ""))
    learned_ids = [
        value
        for value in scenario_ids
        if value.startswith("learned_") and value != f"learned_{current_mode}"
    ][:MAX_OPERATING_LEARNED_SCENARIOS]
    for slot, scenario_id in enumerate(learned_ids, start=1):
        index = scenario_ids.index(scenario_id)
        prefix = f"OperatingLearned{slot}"
        for source_name, suffix in OPERATING_SCENARIO_SOURCE_FIELDS:
            if source_name not in ds or ds[source_name].dims != ("scenario", "time"):
                continue
            values[f"{prefix}{suffix}"] = np.asarray(ds[source_name].isel(scenario=index).values, dtype=np.float64)
    for quantile in ("P10", "P50", "P90"):
        source_name = f"Solar{quantile}Watts"
        if source_name in ds and ds[source_name].dims == ("time",):
            values[f"OperatingSolar{quantile}Watts"] = np.asarray(ds[source_name].values, dtype=np.float64)
    if not values:
        return pd.DataFrame()
    frame = pd.DataFrame(values, index=times)
    frame = frame[~frame.index.isna()].sort_index()
    return frame[~frame.index.duplicated(keep="last")]


def _with_system_as_is_decision_fields(
    ds: xr.Dataset,
    *,
    prefer_operating_scenario: bool,
) -> tuple[xr.Dataset, str]:
    """Attach one internally consistent system-as-is decision contract.

    The operational scenario product is re-anchored from the freshest APS SOC
    and finite instrument state. When its complete current-state distribution
    is present, both the 96-hour card and the scenario reference must use it.
    Older deployments can still fall back to one ensemble product; the helper
    never mixes a deterministic median with ensemble bounds.
    """
    operating_sources = {
        target: operating_source
        for target, (operating_source, _fallback_sources) in SYSTEM_AS_IS_DECISION_FIELD_SOURCES.items()
    }
    fallback_sources: dict[str, str] = {}
    for target, (_operating_source, candidates) in SYSTEM_AS_IS_DECISION_FIELD_SOURCES.items():
        source = next((name for name in candidates if name in ds and ds[name].dims == ("time",)), None)
        if source is not None:
            fallback_sources[target] = source

    use_operating = prefer_operating_scenario and all(
        source in ds and ds[source].dims == ("time",)
        for source in operating_sources.values()
    )
    if use_operating:
        selected_sources = operating_sources
        source_kind = "operating_scenario"
    elif len(fallback_sources) == len(SYSTEM_AS_IS_DECISION_FIELDS):
        selected_sources = fallback_sources
        source_kind = (
            "ensemble_fallback"
            if fallback_sources.get("SystemAsIsDecisionSOCP50") == "BatterySOCForecastP50"
            else "legacy_forecast_fallback"
        )
    else:
        return ds, "unavailable"

    result = ds.copy()
    result.attrs = dict(ds.attrs)
    for target, source in selected_sources.items():
        result[target] = result[source].copy(deep=False)
        result[target].attrs = dict(result[source].attrs)
        result[target].attrs["units"] = (
            "1" if target.endswith("Below40Probability") else "%"
        )
    result.attrs["system_as_is_decision_source"] = source_kind
    return result, source_kind


def operating_mode_intervals(
    times: pd.DatetimeIndex,
    codes: np.ndarray,
    *,
    interval_end_aligned: bool = False,
) -> list[tuple[pd.Timestamp, pd.Timestamp, str, str]]:
    """Return contiguous planned-instrument intervals for forecast plot bands."""
    values = np.asarray(codes, dtype=np.float64)
    if len(times) != len(values) or len(times) == 0:
        return []
    if interval_end_aligned and len(times) > 1:
        # Forecast schedules store the state used to integrate the interval
        # ending at each timestamp. Shift only the presentation coordinates;
        # the physical forecast product remains unchanged.
        times = times[:-1]
        values = values[1:]
    valid = np.isfinite(values)
    if not valid.any():
        return []
    positive_steps = np.diff(times.view("i8"))
    positive_steps = positive_steps[positive_steps > 0]
    step = pd.Timedelta(int(np.median(positive_steps)), unit="ns") if positive_steps.size else pd.Timedelta(hours=1)
    intervals: list[tuple[pd.Timestamp, pd.Timestamp, str, str]] = []
    start = 0
    integer_codes = np.where(valid, values, -1).astype(np.int64)
    for index in range(1, len(times) + 1):
        if index < len(times) and integer_codes[index] == integer_codes[start]:
            continue
        code = int(integer_codes[start])
        if code >= 0:
            active = [(name, color) for name, bit, color in OPERATING_MODE_BITS if code & bit]
            if active:
                end = pd.Timestamp(times[index]) if index < len(times) else pd.Timestamp(times[-1]) + step
                label = " + ".join(name for name, _color in active)
                intervals.append((pd.Timestamp(times[start]), end, label, active[0][1]))
        start = index
    return intervals


def _add_operating_schedule_bands(fig: go.Figure, ds: xr.Dataset, *, row: int) -> None:
    """Shade the optimised planned-instrument periods behind related forecast plots."""
    field = "OperatingCL61OptimizedModeCode"
    if field not in ds or "time" not in ds:
        return
    times = pd.DatetimeIndex(ds["time"].values)
    for start, end, label, color in operating_mode_intervals(
        times,
        np.asarray(ds[field].values),
        interval_end_aligned=True,
    ):
        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor=color,
            opacity=1.0,
            line_width=0,
            layer="below",
            annotation_text=f"{label} on",
            annotation_position="top left",
            annotation_font_size=9,
            annotation_font_color=PLOT_TEXT,
            row=row,
            col=1,
        )


def power_panel_label(ds: xr.Dataset, panel: PanelSpec) -> str:
    """Mark forecast plots that are using a re-anchored cached ECMWF cycle."""
    label = (
        cl61_schedule_presentation(ds).title
        if panel.key == "operating_plan_schedule"
        else panel.label
    )
    if panel.key not in {"ecmwf_solar_forecast", "soc_ecmwf_forecast", "operating_plan_scenarios", "operating_plan_schedule"}:
        return label
    if str(ds.attrs.get("operating_planning_forecast_refresh_kind", "")).strip() == "cached_reanchor":
        return f"{label} [Cached forecast - reduced confidence]"
    return label


def _operating_scenario_attrs(
    ds: xr.Dataset | None,
    *,
    status_override: str | None = None,
    reason_override: str | None = None,
) -> dict[str, str]:
    if ds is None:
        return {}
    attrs: dict[str, str] = {}
    status = status_override or str(ds.attrs.get("planning_status", "ready")).strip() or "ready"
    attrs["operating_planning_status"] = status
    reason = reason_override if reason_override is not None else str(ds.attrs.get("planning_status_reason", "")).strip()
    if reason:
        attrs["operating_planning_status_reason"] = reason
    if "scenario" not in ds or status != "ready":
        return attrs
    for source_name, target_name in (
        ("current_mode", "operating_current_mode"),
        ("current_mode_label", "operating_current_mode_label"),
        ("current_mode_confidence", "operating_current_mode_confidence"),
        ("current_mode_maturity", "operating_current_mode_maturity"),
        ("model", "operating_model"),
        ("model_version", "operating_model_version"),
        ("observed_modes", "operating_observed_modes"),
        ("mode_maturity", "operating_mode_maturity"),
        ("operating_event_count", "operating_event_count"),
        ("forecast_horizon_hours", "operating_forecast_horizon_hours"),
        ("optimization_horizon_hours", "operating_optimization_horizon_hours"),
        ("control_authority", "operating_control_authority"),
        ("solar_member_source", "operating_solar_member_source"),
        ("native_ensemble_end_time", "operating_native_ensemble_end_time"),
        ("initial_soc_time", "operating_initial_soc_time"),
        ("generated_at_utc", "operating_generated_at_utc"),
        ("operating_decision_horizon_hours", "operating_decision_horizon_hours"),
        ("planning_forecast_generated_at_utc", "operating_planning_forecast_generated_at_utc"),
        ("planning_forecast_initial_soc_time", "operating_planning_forecast_initial_soc_time"),
        ("planning_forecast_refresh_kind", "operating_planning_forecast_refresh_kind"),
        ("planning_forecast_verification_eligible", "operating_planning_forecast_verification_eligible"),
        ("planning_forecast_time_coverage_start", "operating_planning_forecast_time_coverage_start"),
        ("planning_forecast_time_coverage_end", "operating_planning_forecast_time_coverage_end"),
        ("optimized_safe", "operating_optimized_safe"),
        ("optimized_collection_hours", "operating_optimized_collection_hours"),
        ("optimized_minimum_p10_soc", "operating_optimized_minimum_p10_soc"),
        ("optimized_status", "operating_optimized_status"),
        ("optimized_reason_code", "operating_optimized_reason_code"),
        ("optimized_reason", "operating_optimized_reason"),
        ("optimized_base_mode", "operating_optimized_base_mode"),
        ("optimized_base_mode_label", "operating_optimized_base_mode_label"),
        ("optimized_blocking_instruments", "operating_optimized_blocking_instruments"),
        ("optimized_operator_action_required", "operating_optimized_operator_action_required"),
        ("optimized_schedule_policy", "operating_optimized_schedule_policy"),
        ("optimized_priority_order", "operating_optimized_priority_order"),
        ("optimized_controlled_instruments", "operating_optimized_controlled_instruments"),
        ("optimized_held_existing_instruments", "operating_optimized_held_existing_instruments"),
        ("optimized_instrument_hours", "operating_optimized_instrument_hours"),
        ("optimized_instrument_starts", "operating_optimized_instrument_starts"),
        ("optimized_total_instrument_hours", "operating_optimized_total_instrument_hours"),
        ("optimized_controlled_energy_kwh", "operating_optimized_controlled_energy_kwh"),
        ("optimized_daily_operations", "operating_optimized_daily_operations"),
        ("optimized_active_instrument_count_max", "operating_optimized_active_instrument_count_max"),
        ("optimized_phase_aware_search", "operating_optimized_phase_aware_search"),
        ("optimized_phase_validation_minimum_p10_soc", "operating_optimized_phase_validation_minimum_p10_soc"),
        ("p50_continuation_control_authority", "operating_p50_continuation_control_authority"),
        ("p50_continuation_eligible", "operating_p50_continuation_eligible"),
        ("p50_continuation_status", "operating_p50_continuation_status"),
        ("p50_continuation_reason_code", "operating_p50_continuation_reason_code"),
        ("p50_continuation_reason", "operating_p50_continuation_reason"),
        ("p50_continuation_held_instruments", "operating_p50_continuation_held_instruments"),
        ("p50_continuation_recovery_soc_pct", "operating_p50_continuation_recovery_soc_pct"),
        ("p50_continuation_minimum_soc_pct", "operating_p50_continuation_minimum_soc_pct"),
        ("p50_continuation_recovery_time_utc", "operating_p50_continuation_recovery_time_utc"),
        (
            "p50_continuation_minimum_soc_before_recovery_pct",
            "operating_p50_continuation_minimum_soc_before_recovery_pct",
        ),
        ("p50_continuation_fallback", "operating_p50_continuation_fallback"),
        ("minimum_controlled_run_hours", "operating_minimum_controlled_run_hours"),
        ("max_controlled_starts_per_utc_day", "operating_max_controlled_starts_per_utc_day"),
    ):
        if source_name in ds.attrs:
            attrs[target_name] = str(ds.attrs[source_name])
    scenario_ids = [str(value) for value in ds["scenario"].values]
    current_mode = str(ds.attrs.get("current_mode", ""))
    learned_ids = [
        value
        for value in scenario_ids
        if value.startswith("learned_") and value != f"learned_{current_mode}"
    ][:MAX_OPERATING_LEARNED_SCENARIOS]
    labels = [str(value) for value in ds["scenario_label"].values] if "scenario_label" in ds else scenario_ids
    for slot, definition in enumerate(SUGGESTED_OPERATING_SCENARIOS, start=1):
        if definition.scenario_id in scenario_ids:
            attrs[f"operating_suggested_{slot}_label"] = labels[
                scenario_ids.index(definition.scenario_id)
            ]
    for slot, scenario_id in enumerate(learned_ids, start=1):
        attrs[f"operating_learned_{slot}_label"] = labels[scenario_ids.index(scenario_id)]
    return attrs


def _operating_scenario_alignment(
    display_ds: xr.Dataset,
    operating_scenarios_ds: xr.Dataset | None,
) -> tuple[bool, str]:
    """Require operating plans and the system forecast to share one SOC anchor.

    A scenario plan is only comparable with the 96-hour system forecast when
    both calculations start at the same physical APS SOC observation.  The
    timestamp is part of the forecast contract, not display metadata.
    """
    if operating_scenarios_ds is None:
        return False, "Operating-plan product is unavailable"
    status = str(operating_scenarios_ds.attrs.get("planning_status", "ready")).strip() or "ready"
    if status != "ready":
        return False, str(operating_scenarios_ds.attrs.get("planning_status_reason", "Operating plan is unavailable"))
    forecast_anchor = _summary_display_timestamp(display_ds.attrs.get("forecast_initial_soc_time"))
    # Scenarios are re-integrated from the latest measured SOC. Compare that
    # scenario anchor with the system forecast; the planning forecast anchor is
    # retained separately as solar-cycle provenance.
    plan_anchor = _summary_display_timestamp(operating_scenarios_ds.attrs.get("initial_soc_time"))
    if forecast_anchor is None or plan_anchor is None:
        return False, "Missing SOC anchor required to compare the system forecast and operating plan"
    # A newer scenario has been re-integrated from a newer physical SOC sample
    # and remains authoritative for its own panel. Only withhold a plan that is
    # materially older than the system forecast it is being compared with.
    plan_age_seconds = float((forecast_anchor - plan_anchor) / pd.Timedelta(seconds=1))
    if plan_age_seconds > OPERATING_SCENARIO_ANCHOR_TOLERANCE_MINUTES * 60.0:
        return (
            False,
            "Operating-plan SOC anchor does not match the current system forecast "
            f"({plan_anchor.isoformat()} versus {forecast_anchor.isoformat()})",
        )
    return True, ""


def merge_operating_scenarios_into_display_summary(
    display_ds: xr.Dataset,
    operating_scenarios_ds: xr.Dataset | None,
) -> xr.Dataset:
    """Merge the compact scenario contract into an existing display summary."""
    # Display summaries may contain an older pre-merged operating plan. The
    # standalone scenario product is authoritative, including its unavailable
    # state, so never retain those stale traces.
    stale_fields = [
        name
        for name in display_ds.data_vars
        if name.startswith("Operating") or name in SYSTEM_AS_IS_DECISION_FIELDS
    ]
    base_ds = display_ds.drop_vars(stale_fields) if stale_fields else display_ds
    aligned, alignment_reason = _operating_scenario_alignment(base_ds, operating_scenarios_ds)
    frame = _operating_scenario_frame(operating_scenarios_ds) if aligned else pd.DataFrame()
    if frame.empty:
        base_ds = base_ds.copy()
        base_ds.attrs = dict(display_ds.attrs)
        base_ds, _decision_source = _with_system_as_is_decision_fields(
            base_ds,
            prefer_operating_scenario=False,
        )
        base_ds.attrs.update(
            _operating_scenario_attrs(
                operating_scenarios_ds,
                status_override="unavailable" if not aligned else None,
                reason_override=alignment_reason if not aligned else None,
            )
        )
        return base_ds
    scenario_ds = xr.Dataset(
        {name: (("time",), frame[name].to_numpy(dtype=np.float32)) for name in frame.columns},
        coords={"time": frame.index.to_numpy(dtype="datetime64[ns]")},
    )
    for name in scenario_ds.data_vars:
        if "SOC" in name:
            scenario_ds[name].attrs["units"] = "%"
        elif "Probability" in name:
            scenario_ds[name].attrs["units"] = "1"
        elif "Watts" in name:
            scenario_ds[name].attrs["units"] = "W"
    merged = xr.merge((base_ds, scenario_ds), join="outer", compat="override", combine_attrs="override").sortby("time")
    merged.attrs = dict(display_ds.attrs)
    merged.attrs.update(_operating_scenario_attrs(operating_scenarios_ds))
    merged, _decision_source = _with_system_as_is_decision_fields(
        merged,
        prefer_operating_scenario=True,
    )
    merged.attrs["time_coverage_end"] = pd.Timestamp(merged["time"].values[-1]).isoformat()
    return merged


def _resample_display_frame(frame: pd.DataFrame, freq: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame.resample(freq).mean().dropna(how="all")


def build_power_display_summary_dataset(
    power_ds: xr.Dataset,
    ass_power_ds: xr.Dataset | None = None,
    pdu_ds: xr.Dataset | None = None,
    forecast_ds: xr.Dataset | None = None,
    forecast_skill_ds: xr.Dataset | None = None,
    hindcast_ds: xr.Dataset | None = None,
    ensemble_forecast_ds: xr.Dataset | None = None,
    ensemble_skill_ds: xr.Dataset | None = None,
    operating_scenarios_ds: xr.Dataset | None = None,
    freq: str = POWER_DISPLAY_SUMMARY_FREQ,
) -> xr.Dataset:
    """Build one-minute APS traces for fast dashboard plotting.

    The raw APS, ASFS logger, and ASS PDU Zarrs remain authoritative. This derived store
    keeps only the fields used by the curated Power summary panels, resampled
    to the dashboard display cadence, plus the cumulative-energy variables
    already produced for the APS cumulative panel.
    """
    if "time" not in power_ds or power_ds.sizes.get("time", 0) == 0:
        return xr.Dataset()

    frames: list[pd.DataFrame] = []
    sorted_power = power_ds.sortby("time")
    power_times = pd.DatetimeIndex(sorted_power["time"].values)
    power_start = power_times.min()
    power_end = power_times.max()

    power_frame = _time_frame_from_dataset(sorted_power, POWER_DISPLAY_SUMMARY_FIELDS)
    power_frame = _resample_display_frame(power_frame, freq)
    if not power_frame.empty:
        frames.append(power_frame)

    energy = build_power_display_energy_dataset(power_ds, freq=freq)
    if energy.sizes.get("time", 0):
        frames.append(energy.to_dataframe())

    if ass_power_ds is not None:
        ass_frame = _time_frame_from_dataset(ass_power_ds.sortby("time"), POWER_DISPLAY_SUMMARY_CONTEXT_FIELDS)
        if not ass_frame.empty:
            ass_frame = ass_frame[(ass_frame.index >= power_start) & (ass_frame.index <= power_end)]
        ass_frame = _resample_display_frame(ass_frame, freq)
        if not ass_frame.empty:
            frames.append(ass_frame)

    if pdu_ds is not None:
        pdu_frame = _time_frame_from_dataset(pdu_ds.sortby("time"), PDU_DISPLAY_SUMMARY_FIELDS)
        if not pdu_frame.empty:
            pdu_frame = pdu_frame[(pdu_frame.index >= power_start) & (pdu_frame.index <= power_end)]
        pdu_frame = _resample_display_frame(pdu_frame, freq)
        if not pdu_frame.empty:
            frames.append(pdu_frame)

    if forecast_ds is not None:
        forecast_frame = _time_frame_from_dataset(forecast_ds.sortby("time"), POWER_SOC_FORECAST_FIELDS)
        if not forecast_frame.empty:
            frames.append(forecast_frame)

    if forecast_skill_ds is not None:
        skill_frame = _time_frame_from_dataset(forecast_skill_ds.sortby("time"), POWER_SOC_FORECAST_SKILL_FIELDS)
        skill_frame = _resample_display_frame(skill_frame, freq)
        if not skill_frame.empty:
            frames.append(skill_frame)

    if hindcast_ds is not None:
        hindcast_frame = _time_frame_from_dataset(hindcast_ds.sortby("time"), POWER_SOC_HINDCAST_FIELDS)
        if not hindcast_frame.empty:
            frames.append(hindcast_frame)

    if ensemble_forecast_ds is not None:
        ensemble_frame = _time_frame_from_dataset(
            ensemble_forecast_ds.sortby("time"), POWER_SOC_ENSEMBLE_FORECAST_FIELDS
        )
        if not ensemble_frame.empty:
            frames.append(ensemble_frame)

    if ensemble_skill_ds is not None:
        ensemble_skill_frame = _time_frame_from_dataset(
            ensemble_skill_ds.sortby("time"), POWER_SOC_ENSEMBLE_SKILL_FIELDS
        )
        ensemble_skill_frame = _resample_display_frame(ensemble_skill_frame, freq)
        if not ensemble_skill_frame.empty:
            frames.append(ensemble_skill_frame)

    forecast_anchor = _summary_display_timestamp(
        ensemble_forecast_ds.attrs.get("initial_soc_time") if ensemble_forecast_ds is not None else None
    )
    planning_anchor = _summary_display_timestamp(
        operating_scenarios_ds.attrs.get("initial_soc_time") if operating_scenarios_ds is not None else None
    )
    operating_alignment_reason = ""
    operating_aligned = operating_scenarios_ds is not None
    if operating_scenarios_ds is not None:
        if forecast_anchor is None or planning_anchor is None:
            operating_aligned = False
            operating_alignment_reason = "Missing SOC anchor required to compare the system forecast and operating plan"
        elif (
            float((forecast_anchor - planning_anchor) / pd.Timedelta(seconds=1))
            > OPERATING_SCENARIO_ANCHOR_TOLERANCE_MINUTES * 60.0
        ):
            operating_aligned = False
            operating_alignment_reason = (
                "Operating-plan SOC anchor does not match the current system forecast "
                f"({planning_anchor.isoformat()} versus {forecast_anchor.isoformat()})"
            )
    operating_frame = _operating_scenario_frame(operating_scenarios_ds) if operating_aligned else pd.DataFrame()
    if not operating_frame.empty:
        frames.append(operating_frame)

    if not frames:
        return xr.Dataset()

    display_frame = pd.concat(frames, axis=1).sort_index()
    display_frame = display_frame.loc[:, ~display_frame.columns.duplicated(keep="last")]
    display_frame = display_frame.dropna(how="all")
    if display_frame.empty:
        return xr.Dataset()

    start = pd.Timestamp(display_frame.index.min()).isoformat()
    end = pd.Timestamp(display_frame.index.max()).isoformat()
    summary_attrs = {
        POWER_DISPLAY_SUMMARY_ATTR: "true",
        "source": "derived from power.zarr plus optional ASS/PDU, SOC forecast verification, ensemble, and learned operating-scenario products",
        "frequency": freq,
        "time_coverage_start": start,
        "time_coverage_end": end,
        "minimum_operational_soc_pct": f"{MINIMUM_OPERATIONAL_SOC_PCT:g}",
        "description": "Display-only one-minute APS summary traces for fast dashboard plotting.",
    }
    if forecast_ds is not None:
        for source_name, target_name in (
            ("initial_soc_time", "forecast_initial_soc_time"),
            ("generated_at_utc", "forecast_generated_at_utc"),
            ("forecast_system_version", "forecast_system_version"),
            ("forecast_model_contract_id", "forecast_model_contract_id"),
            ("forecast_identity_id", "forecast_identity_id"),
            ("feature_set_version", "forecast_feature_set_version"),
            ("feature_set_digest", "forecast_feature_set_digest"),
            ("training_cutoff_utc", "forecast_training_cutoff_utc"),
            ("observation_cutoff_utc", "forecast_observation_cutoff_utc"),
            ("forecast_code_revision", "forecast_code_revision"),
            ("source_cycle_set_id", "forecast_source_cycle_set_id"),
            ("source_manifest_digest", "forecast_source_manifest_digest"),
            ("adaptive_calibration_state_id", "forecast_adaptive_calibration_state_id"),
            ("degraded_mode_code", "forecast_degraded_mode_code"),
            ("forecast_refresh_kind", "forecast_refresh_kind"),
            ("forecast_verification_eligible", "forecast_verification_eligible"),
            ("independent_cycle", "forecast_independent_cycle"),
            ("load_mode", "forecast_load_mode"),
            ("load_model", "forecast_load_model"),
            ("load_model_version", "forecast_load_model_version"),
            ("load_mode_source", "forecast_load_mode_source"),
            ("load_mode_active_kits", "forecast_load_mode_active_kits"),
            ("load_mode_signature", "forecast_load_mode_signature"),
            ("load_mode_learning_ready", "forecast_load_mode_learning_ready"),
            ("load_mode_learning_reason", "forecast_load_mode_learning_reason"),
            ("load_mode_learning_observations", "forecast_load_mode_learning_observations"),
            ("load_mode_pdu_active_watts", "forecast_load_mode_pdu_active_watts"),
            ("load_measurement", "forecast_load_measurement"),
            ("load_balance_measurement", "forecast_load_balance_measurement"),
            ("solar_model_name", "forecast_solar_model_name"),
            ("solar_power_semantics", "forecast_solar_power_semantics"),
            ("solar_forcing_mode", "forecast_solar_forcing_mode"),
            ("solar_calibration_contract_id", "forecast_solar_calibration_contract_id"),
            ("load_current_phase", "forecast_load_current_phase"),
            ("load_residual_model_status", "forecast_load_residual_model_status"),
            ("soc_bias_correction_method", "forecast_soc_bias_correction_method"),
            ("soc_physical_consistency_status", "forecast_soc_physical_consistency_status"),
            ("battery_usable_capacity_kwh", "forecast_battery_usable_capacity_kwh"),
            ("battery_charge_efficiency", "forecast_battery_charge_efficiency"),
            ("battery_discharge_efficiency", "forecast_battery_discharge_efficiency"),
            ("battery_parasitic_load_w", "forecast_battery_parasitic_load_w"),
            ("battery_max_charge_w", "forecast_battery_max_charge_w"),
            ("battery_max_discharge_w", "forecast_battery_max_discharge_w"),
            ("load_bias_correction_w", "forecast_load_bias_correction_w"),
            ("ecmwf_provider_requested", "forecast_ecmwf_provider_requested"),
            ("ecmwf_provider_effective", "forecast_ecmwf_provider_effective"),
            ("ecmwf_provider_fallback_reason", "forecast_ecmwf_provider_fallback_reason"),
            ("ecmwf_cycle_time", "forecast_ecmwf_cycle_time"),
            ("selected_grid_latitude", "forecast_ecmwf_grid_latitude"),
            ("selected_grid_longitude", "forecast_ecmwf_grid_longitude"),
            ("selected_grid_distance_km", "forecast_ecmwf_grid_distance_km"),
        ):
            if source_name in forecast_ds.attrs:
                summary_attrs[target_name] = str(forecast_ds.attrs[source_name])
    if ensemble_forecast_ds is not None:
        for source_name, target_name in (
            ("initial_soc_time", "forecast_initial_soc_time"),
            ("generated_at_utc", "forecast_generated_at_utc"),
        ):
            if source_name in ensemble_forecast_ds.attrs:
                summary_attrs[target_name] = str(ensemble_forecast_ds.attrs[source_name])
    summary_attrs.update(
        _operating_scenario_attrs(
            operating_scenarios_ds,
            status_override="unavailable" if not operating_aligned else None,
            reason_override=operating_alignment_reason if not operating_aligned else None,
        )
    )
    out = xr.Dataset(
        {name: (("time",), display_frame[name].to_numpy(dtype=np.float32)) for name in display_frame.columns},
        coords={"time": display_frame.index.to_numpy(dtype="datetime64[ns]")},
        attrs=summary_attrs,
    )
    out, _decision_source = _with_system_as_is_decision_fields(
        out,
        prefer_operating_scenario=operating_aligned and not operating_frame.empty,
    )
    for name in out.data_vars:
        unit = human_unit(name)
        if unit:
            out[name].attrs["units"] = unit
    for name in POWER_DISPLAY_ENERGY_MAP.values():
        if name in out:
            out[name].attrs["units"] = "kWh"
    if "BatterySOCForecast" in out:
        out["BatterySOCForecast"].attrs["units"] = "%"
    for name in OPERATING_SCENARIO_DISPLAY_FIELDS:
        if name not in out:
            continue
        if "SOC" in name:
            out[name].attrs["units"] = "%"
        elif "Probability" in name:
            out[name].attrs["units"] = "1"
        elif "Watts" in name:
            out[name].attrs["units"] = "W"
    for name in (
        "BatterySOCForecast_Load100W",
        "BatterySOCForecast_Load200W",
        "BatterySOCForecast_Load300W",
        "BatterySOCForecast_Load400W",
        "BatterySOCForecast_Load500W",
        "BatterySOCForecast_Load600W",
    ):
        if name in out:
            out[name].attrs["units"] = "%"
    if "ECMWFSolarIrradiance" in out:
        out["ECMWFSolarIrradiance"].attrs["units"] = "W m-2"
    for name in ("ForecastSolarWatts", "ForecastLoadWatts"):
        if name in out:
            out[name].attrs["units"] = "W"
    for name in (
        "ForecastSOCMAE_0_6h_Verified",
        "ForecastSOCMAE_6_24h_Verified",
        "ForecastSOCMAE_24_48h_Verified",
        "ForecastSOCMAE_48_96h_Verified",
        "ForecastSOCBias_0_6h_Verified",
    ):
        if name in out:
            out[name].attrs["units"] = "percentage points"
    for name in ("ForecastSolarMAE24h", "ForecastSolarBias24h", "ForecastLoadMAE24h", "ForecastLoadBias24h"):
        if name in out:
            out[name].attrs["units"] = "W"
    for name in ("ForecastSOCSkill_0_6h", "ForecastSolarSkill24h", "ForecastLoadSkill24h"):
        if name in out:
            out[name].attrs["units"] = "1"
    if "ForecastVerificationSamples" in out:
        out["ForecastVerificationSamples"].attrs["units"] = "samples"
    if "ForecastIndependentCycles" in out:
        out["ForecastIndependentCycles"].attrs["units"] = "cycles"
    for name in POWER_SOC_HINDCAST_FIELDS:
        if name in out:
            out[name].attrs["units"] = "%"
    for name in POWER_SOC_ENSEMBLE_FORECAST_FIELDS + POWER_SOC_ENSEMBLE_SKILL_FIELDS:
        if name in out:
            out[name].attrs["units"] = human_unit(name)
    for name in out.data_vars:
        if name.endswith("Samples"):
            out[name].attrs["units"] = "samples"
        elif name.endswith("Cycles"):
            out[name].attrs["units"] = "cycles"
        elif "Skill" in name:
            out[name].attrs["units"] = "1"
    return out


def _summary_display_timestamp(value: object) -> pd.Timestamp | None:
    if value in (None, ""):
        return None
    try:
        timestamp = pd.Timestamp(value)
    except Exception:
        return None
    if pd.isna(timestamp):
        return None
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp


def _crop_to_summary_display_window(ds: xr.Dataset, times: pd.DatetimeIndex) -> xr.Dataset:
    start = _summary_display_timestamp(ds.attrs.get(SUMMARY_DISPLAY_START_ATTR))
    end = _summary_display_timestamp(ds.attrs.get(SUMMARY_DISPLAY_END_ATTR))
    if start is None and end is None:
        return ds
    mask = np.ones(len(times), dtype=bool)
    if start is not None:
        mask &= times >= start
    if end is not None:
        mask &= times <= end
    forecast_names = [name for name in POWER_FUTURE_DISPLAY_FIELDS if name in ds]
    if end is not None and forecast_names:
        forecast_valid = np.zeros(len(times), dtype=bool)
        for name in forecast_names:
            forecast_valid |= np.isfinite(np.asarray(ds[name].values, dtype=np.float64))
        forecast_end = end + pd.Timedelta(hours=POWER_FORECAST_DISPLAY_HOURS)
        forecast_mask = forecast_valid & (times <= forecast_end)
        if start is not None:
            forecast_mask &= times >= start
        mask |= forecast_mask
    return ds.isel(time=mask)


def _metek_wind_assignments(ds: xr.Dataset) -> dict[str, xr.DataArray]:
    """Derive horizontal wind speed and meteorological direction from Metek U/V."""
    if "metek_x_out_Avg" not in ds or "metek_y_out_Avg" not in ds or "time" not in ds:
        return {}
    u = np.asarray(ds["metek_x_out_Avg"].values, dtype=np.float64)
    v = np.asarray(ds["metek_y_out_Avg"].values, dtype=np.float64)
    valid = np.isfinite(u) & np.isfinite(v)
    speed = np.full(len(u), np.nan, dtype=np.float64)
    direction = np.full(len(u), np.nan, dtype=np.float64)
    speed[valid] = np.hypot(u[valid], v[valid])
    # Meteorological convention: direction wind is coming from, clockwise from north.
    direction[valid] = (270.0 - np.degrees(np.arctan2(v[valid], u[valid]))) % 360.0
    return {
        "MetekWindSpeed": xr.DataArray(
            speed,
            coords={"time": ds["time"]},
            dims=("time",),
            attrs={"units": "m s^-1", "description": "Horizontal wind speed derived from metek_x_out_Avg and metek_y_out_Avg."},
        ),
        "MetekWindDirection": xr.DataArray(
            direction,
            coords={"time": ds["time"]},
            dims=("time",),
            attrs={
                "units": "degree",
                "description": "Meteorological wind direction derived from metek_x_out_Avg and metek_y_out_Avg.",
            },
        ),
    }


def prepare_summary_dataset(ds: xr.Dataset, instrument: str) -> xr.Dataset:
    if instrument not in {"power", "vaisalamet"} or "time" not in ds or ds.sizes.get("time", 0) == 0:
        return ds

    if instrument == "power":
        ds, _decision_source = _with_system_as_is_decision_fields(
            ds,
            prefer_operating_scenario=True,
        )

    times = pd.DatetimeIndex(ds["time"].values)
    if len(times) == 0:
        return ds

    assignments: dict[str, xr.DataArray] = {}
    if instrument == "vaisalamet":
        ds = augment_meteorology_from_fast_sonic(ds)
        assignments.update(_metek_wind_assignments(ds))
        prepared = ds.assign(**assignments) if assignments else ds
        prepared_times = pd.DatetimeIndex(prepared["time"].values)
        return _crop_to_summary_display_window(prepared, prepared_times)

    display_assignments = _display_energy_assignments(ds)
    if display_assignments:
        prepared = ds.assign(**display_assignments)
        prepared_times = pd.DatetimeIndex(prepared["time"].values)
        return _crop_to_summary_display_window(prepared, prepared_times)

    generated_fields = [name for name in ("SolarYield_East", "SolarYield_South", "SolarYield_West") if name in ds]
    for field_name in generated_fields:
        generated = _daily_cumulative_counter_delta(times, np.asarray(ds[field_name].values, dtype=np.float64))
        assignments[field_name] = xr.DataArray(
            generated,
            coords={"time": ds["time"]},
            dims=("time",),
            attrs={"units": "kWh"},
        )

    if "ACOutputWatts" in ds or "DCInverterWatts" in ds:
        ac_power = np.asarray(
            ds["ACOutputWatts"].values if "ACOutputWatts" in ds else np.full(len(times), np.nan),
            dtype=np.float64,
        )
        dc_power = np.asarray(
            ds["DCInverterWatts"].values if "DCInverterWatts" in ds else np.full(len(times), np.nan),
            dtype=np.float64,
        )
        # The power summary can include ASFS logger overlay traces on a merged
        # time grid. Only integrate rows where APS output power was actually
        # sampled, otherwise ASFS-only timestamps would look like zero APS load.
        valid_power = np.isfinite(ac_power) | np.isfinite(dc_power)
        utilised_full = np.full(len(times), np.nan, dtype=np.float64)
        if np.any(valid_power):
            utilised_power_w = np.nan_to_num(ac_power[valid_power], nan=0.0) + np.nan_to_num(dc_power[valid_power], nan=0.0)
            utilised_power_w = np.clip(utilised_power_w, a_min=0.0, a_max=None)
            utilised_full[valid_power] = _daily_cumulative_energy_kwh(times[valid_power], utilised_power_w)

        assignments["CumulativePowerUtilised"] = xr.DataArray(
            utilised_full,
            coords={"time": ds["time"]},
            dims=("time",),
            attrs={"units": "kWh"},
        )

    if generated_fields:
        total_generated = np.full(len(times), np.nan, dtype=np.float64)
        valid_generated = np.zeros(len(times), dtype=bool)
        field_values: list[np.ndarray] = []
        for field_name in generated_fields:
            values = np.asarray(assignments[field_name].values, dtype=np.float64)
            field_values.append(values)
            valid_generated |= np.isfinite(values)
        if np.any(valid_generated):
            summed = np.zeros(int(np.count_nonzero(valid_generated)), dtype=np.float64)
            for values in field_values:
                summed += np.nan_to_num(values[valid_generated], nan=0.0)
            total_generated[valid_generated] = summed
        assignments["CumulativePowerGeneratedTotal"] = xr.DataArray(
            total_generated,
            coords={"time": ds["time"]},
            dims=("time",),
            attrs={"units": "kWh"},
        )

    prepared = ds.assign(**assignments) if assignments else ds
    prepared_times = pd.DatetimeIndex(prepared["time"].values)
    return _crop_to_summary_display_window(prepared, prepared_times)


def numeric_time_vars(ds: xr.Dataset) -> list[str]:
    names: list[str] = []
    for name, da in ds.data_vars.items():
        if da.dims != ("time",):
            continue
        if name == "RECORD":
            continue
        if np.issubdtype(da.dtype, np.number):
            names.append(name)
    return names


def downsample_time(ds: xr.Dataset, max_time_samples: int = MAX_TIME_SAMPLES) -> xr.Dataset:
    if "time" not in ds:
        return ds
    count = ds.sizes.get("time", 0)
    if count > max_time_samples:
        step = int(np.ceil(count / max_time_samples))
        ds = ds.isel(time=slice(None, None, step))
    return ds


def is_status_like_var(name: str) -> bool:
    lower = name.lower()
    return any(token in lower for token in STATUS_TOKENS)


def _clean_values(values: np.ndarray, trace: TraceSpec) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).copy()
    if trace.valid_min is not None:
        arr[arr < trace.valid_min] = np.nan
    if trace.valid_max is not None:
        arr[arr > trace.valid_max] = np.nan
    if trace.scale != 1.0:
        arr *= trace.scale
    return arr


def _has_signal(values: np.ndarray, trace: TraceSpec) -> bool:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return False
    return not (trace.skip_if_all_zero and np.allclose(finite, 0.0))


def _panel_series(ds: xr.Dataset, panel: PanelSpec) -> list[tuple[TraceSpec, np.ndarray]]:
    rows: list[tuple[TraceSpec, np.ndarray]] = []
    for trace in panel.traces:
        if trace.var not in ds:
            continue
        values = _clean_values(ds[trace.var].values, trace)
        if not _has_signal(values, trace):
            continue
        rows.append((trace, values))
    return rows


def _active_panels(ds: xr.Dataset, instrument: str) -> list[tuple[PanelSpec, list[tuple[TraceSpec, np.ndarray]]]]:
    panels: list[tuple[PanelSpec, list[tuple[TraceSpec, np.ndarray]]]] = []
    for panel in SUMMARY_LAYOUTS[instrument]:
        rows = _panel_series(ds, panel)
        if rows:
            panels.append((panel, rows))
    return panels


def _power_panel_time_group(panel_key: str) -> str:
    return POWER_PANEL_TIME_GROUP_BY_KEY.get(panel_key, "observed")


def _order_power_panels_by_time_group(
    panels: list[tuple[PanelSpec, list[tuple[TraceSpec, np.ndarray]]]],
) -> list[tuple[PanelSpec, list[tuple[TraceSpec, np.ndarray]]]]:
    """Keep Power panels with compatible time axes together on desktop."""
    group_rank = {name: index for index, name in enumerate(POWER_PANEL_TIME_GROUPS)}
    panel_rank = {
        panel_key: index
        for panel_keys in POWER_PANEL_TIME_GROUPS.values()
        for index, panel_key in enumerate(panel_keys)
    }
    original_rank = {panel.key: index for index, (panel, _rows) in enumerate(panels)}
    return sorted(
        panels,
        key=lambda item: (
            group_rank.get(_power_panel_time_group(item[0].key), 0),
            panel_rank.get(item[0].key, original_rank[item[0].key]),
        ),
    )


def _time_index(ds: xr.Dataset) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(ds["time"].values) if "time" in ds else pd.DatetimeIndex([])


def _slice_dataset_time_limits(ds: xr.Dataset, x_limits) -> xr.Dataset:
    if x_limits is None or "time" not in ds:
        return ds
    start, end = (pd.Timestamp(value) for value in x_limits)
    if start.tz is not None:
        start = start.tz_convert("UTC").tz_localize(None)
    if end.tz is not None:
        end = end.tz_convert("UTC").tz_localize(None)
    if pd.isna(start) or pd.isna(end) or end <= start:
        return ds
    times = _time_index(ds)
    mask = (times >= start) & (times <= end)
    return ds.isel(time=mask).sortby("time")


def _trace_time_values(times: pd.DatetimeIndex, values: np.ndarray) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Drop merged-grid NaNs so each trace renders on its own real sampling cadence."""
    if len(times) == 0:
        return times, values
    finite = np.isfinite(values)
    if not np.any(finite):
        return pd.DatetimeIndex([]), np.asarray([], dtype=np.float64)
    return times[finite], values[finite]


def _downsample_trace(
    times: pd.DatetimeIndex,
    values: np.ndarray,
    max_time_samples: int,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Downsample one trace after dropping merged-grid NaNs.

    Summary datasets can merge one-second APS data with one-minute ASFS context.
    Downsampling each rendered trace separately preserves the lower-cadence
    context lines instead of skipping them on the dense merged time grid.
    """
    count = min(len(times), values.size)
    if count == 0:
        return pd.DatetimeIndex([]), np.asarray([], dtype=np.float64)
    times = times[:count]
    values = np.asarray(values[:count], dtype=np.float64)
    if count <= max_time_samples:
        return times, values
    keep_count = max(2, int(max_time_samples))
    # A uniform stride can hide short APS load spikes and voltage dips. Keep the
    # local extrema from each time bucket instead, while retaining chronological
    # order and a strict trace-point budget for browser payloads.
    bucket_count = max(1, (keep_count - 2) // 2)
    edges = np.linspace(0, count, bucket_count + 1, dtype=int)
    selected = [0, count - 1]
    for left, right in zip(edges[:-1], edges[1:]):
        if right <= left:
            continue
        bucket = values[left:right]
        finite = np.flatnonzero(np.isfinite(bucket))
        if len(finite) == 0:
            continue
        finite_values = bucket[finite]
        selected.extend(
            (
                left + int(finite[np.argmin(finite_values)]),
                left + int(finite[np.argmax(finite_values)]),
            )
        )
    keep = np.unique(np.asarray(selected, dtype=int))
    if len(keep) > keep_count:
        keep = keep[np.linspace(0, len(keep) - 1, keep_count, dtype=int)]
    return times[keep], values[keep]


def _smooth_trace_values(
    times: pd.DatetimeIndex,
    values: np.ndarray,
    trace: TraceSpec,
) -> np.ndarray:
    """Apply optional display-only smoothing before browser/PNG downsampling."""
    if trace.smooth_minutes is None or trace.smooth_minutes <= 0 or len(times) < 3:
        return values
    series = pd.Series(np.asarray(values, dtype=np.float64), index=times)
    window = f"{float(trace.smooth_minutes):g}min"
    return series.rolling(window, center=True, min_periods=1).mean().to_numpy(dtype=np.float64)


def _insert_day_breaks(
    times: pd.DatetimeIndex,
    values: np.ndarray,
    trace: TraceSpec,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Break display lines where UTC-day cumulative counters reset."""
    if not trace.break_on_day_change or len(times) < 2:
        return times, values
    day_starts = times.normalize()
    if not np.any(day_starts[1:] != day_starts[:-1]):
        return times, values

    out_times = []
    out_values: list[float] = []
    for idx, (timestamp, value) in enumerate(zip(times, values, strict=False)):
        if idx > 0 and day_starts[idx] != day_starts[idx - 1]:
            out_times.append(timestamp)
            out_values.append(np.nan)
        out_times.append(timestamp)
        out_values.append(float(value))
    return pd.DatetimeIndex(out_times), np.asarray(out_values, dtype=np.float64)


def _insert_line_gap_breaks(times: pd.DatetimeIndex, values: np.ndarray) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Insert NaNs into line traces so outages render as visible white breaks."""
    if len(times) < 2 or values.size < 2:
        return times, values
    expanded_times, expanded_values = insert_time_gap_breaks(times, np.asarray(values, dtype=np.float64)[None, :], time_axis=1)
    return pd.DatetimeIndex(expanded_times), expanded_values[0]


def _downsample_trace_with_gap_breaks(
    times: pd.DatetimeIndex,
    values: np.ndarray,
    max_time_samples: int,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Retain real source gaps without treating downsampling gaps as outages."""
    expanded_times, expanded_values = _insert_line_gap_breaks(times, values)
    gap_mask = ~np.isfinite(expanded_values)
    gap_times = expanded_times[gap_mask]
    finite_times = expanded_times[~gap_mask]
    finite_values = expanded_values[~gap_mask]
    if len(gap_times) == 0:
        return _downsample_trace(finite_times, finite_values, max_time_samples)

    # Each real outage has one or two NaN markers. Keep those markers, but do
    # not let an unusually fragmented source make a browser trace unbounded.
    keep_count = max(2, int(max_time_samples))
    gap_budget = max(0, keep_count - 2)
    if len(gap_times) > gap_budget:
        gap_times = gap_times[np.linspace(0, len(gap_times) - 1, gap_budget, dtype=int)]
    finite_budget = max(2, keep_count - len(gap_times))
    sampled_times, sampled_values = _downsample_trace(finite_times, finite_values, finite_budget)
    combined_times = sampled_times.append(gap_times).sort_values()
    combined_values = np.empty(len(combined_times), dtype=np.float64)
    sampled_lookup = {timestamp: value for timestamp, value in zip(sampled_times, sampled_values, strict=False)}
    for index, timestamp in enumerate(combined_times):
        combined_values[index] = sampled_lookup.get(timestamp, np.nan)
    return combined_times, combined_values


def _projection_trace_values(
    times: pd.DatetimeIndex,
    values: np.ndarray,
    trace: TraceSpec,
    max_time_samples: int,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Fit recent SOC with a low-degree polynomial and extrapolate display-only values."""
    if trace.projection_lookback_minutes is None or trace.projection_lookback_minutes <= 0:
        return pd.DatetimeIndex([]), np.asarray([], dtype=np.float64)
    trace_times, trace_values = _trace_time_values(times, values)
    if len(trace_times) < 2:
        return pd.DatetimeIndex([]), np.asarray([], dtype=np.float64)

    end_time = trace_times.max()
    lookback_start = end_time - pd.Timedelta(minutes=float(trace.projection_lookback_minutes))
    fit_mask = trace_times >= lookback_start
    fit_times = trace_times[fit_mask]
    fit_values = trace_values[fit_mask]
    finite = np.isfinite(fit_values)
    fit_times = fit_times[finite]
    fit_values = fit_values[finite]
    if len(fit_times) < 2:
        return pd.DatetimeIndex([]), np.asarray([], dtype=np.float64)

    degree = max(1, min(int(trace.projection_degree), len(fit_times) - 1))
    fit_x = ((fit_times - end_time) / pd.Timedelta(minutes=1)).to_numpy(dtype=np.float64)
    try:
        coeffs = np.polyfit(fit_x, fit_values.astype(np.float64), degree)
    except Exception:
        return pd.DatetimeIndex([]), np.asarray([], dtype=np.float64)

    horizon = pd.Timedelta(hours=float(trace.projection_horizon_hours))
    step_minutes = max(float(POWER_SOC_PROJECTION_STEP_MINUTES), 1.0)
    projection_end = end_time + horizon
    projection_times = pd.date_range(start=fit_times.min(), end=projection_end, freq=f"{step_minutes:g}min")
    if len(projection_times) == 0 or projection_times[-1] < projection_end:
        projection_times = projection_times.append(pd.DatetimeIndex([projection_end]))
    projection_x = ((projection_times - end_time) / pd.Timedelta(minutes=1)).to_numpy(dtype=np.float64)
    projection_values = np.polyval(coeffs, projection_x)
    projection_values = np.clip(projection_values, 0.0, 100.0)
    return _downsample_trace(pd.DatetimeIndex(projection_times), projection_values, max_time_samples)


def _trace_plot_values(
    times: pd.DatetimeIndex,
    values: np.ndarray,
    max_time_samples: int,
    trace: TraceSpec,
    display_horizon_hours: float | None = None,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    if trace.projection_lookback_minutes is not None:
        return _projection_trace_values(times, values, trace, max_time_samples)
    trace_times, trace_values = _trace_time_values(times, values)
    horizon_candidates = [
        float(value)
        for value in (trace.display_horizon_hours, display_horizon_hours)
        if value is not None
    ]
    if horizon_candidates and len(trace_times):
        display_end = trace_times.min() + pd.Timedelta(hours=min(horizon_candidates))
        display_mask = trace_times <= display_end
        trace_times = trace_times[display_mask]
        trace_values = trace_values[display_mask]
    trace_values = _smooth_trace_values(trace_times, trace_values, trace)
    trace_times, trace_values = _downsample_trace_with_gap_breaks(trace_times, trace_values, max_time_samples)
    return _insert_day_breaks(trace_times, trace_values, trace)


def _matplotlib_linestyle(dash: str | None) -> str:
    if dash in {"dash", "dashdot", "dot"}:
        return {"dash": "--", "dashdot": "-.", "dot": ":"}[dash]
    return "-"


def _plotly_time_tick_options(start: pd.Timestamp, end: pd.Timestamp) -> dict[str, object]:
    duration = end - start
    if duration <= pd.Timedelta(hours=12):
        tick_hours = 1
    elif duration <= pd.Timedelta(hours=30):
        tick_hours = 3
    elif duration <= pd.Timedelta(hours=72):
        tick_hours = 6
    elif duration <= pd.Timedelta(hours=120):
        tick_hours = 12
    else:
        tick_hours = 24
    tick0 = start.normalize() if tick_hours >= 24 else start.floor(f"{tick_hours}h")
    return {
        "tickmode": "linear",
        "tick0": tick0.to_pydatetime(),
        "dtick": tick_hours * 60 * 60 * 1000,
        "tickformat": "%a %d %b<br>%H:%M UTC",
    }


def _include_zero_in_limits(limits: tuple[float, float] | None) -> tuple[float, float] | None:
    """Expand an axis range just enough to keep the zero reference visible."""
    if limits is None:
        return None
    lower, upper = limits
    if lower <= 0.0 <= upper:
        return limits
    span = max(upper - lower, max(abs(lower), abs(upper), 1.0) * 0.1)
    if lower > 0.0:
        return -0.04 * span, upper
    return lower, 0.08 * span


def _axis_tick_values(
    limits: tuple[float, float] | None,
    step: float = 2.0,
) -> tuple[list[float], list[str]]:
    """Build fixed-step numeric tick labels for secondary energy axes."""
    if limits is None:
        return [], []
    lower, upper = limits
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        return [], []
    if not np.isfinite(step) or step <= 0:
        return [], []
    start = np.floor(lower / step) * step
    stop = np.ceil(upper / step) * step
    values = np.arange(start, stop + step * 0.5, step, dtype=float)
    values = values[(values >= lower - step * 0.05) & (values <= upper + step * 0.05)]
    if values.size == 0:
        return [], []
    decimals = 0 if step >= 1 and np.isclose(step, round(step)) else int(max(0, np.ceil(-np.log10(step)))) + 1
    labels = []
    for value in values:
        if abs(value) < step * 1.0e-6:
            labels.append("0")
        elif decimals == 0:
            labels.append(f"{value:.0f}")
        else:
            labels.append(f"{value:.{decimals}f}".rstrip("0").rstrip("."))
    return values.tolist(), labels


def _axis_tick_step(limits: tuple[float, float] | None, target_ticks: int = 6) -> float:
    """Choose a quiet 1/2/5-style tick step for a numeric axis range."""
    if limits is None:
        return 1.0
    lower, upper = limits
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        return 1.0
    raw_step = (upper - lower) / max(target_ticks - 1, 1)
    if raw_step <= 0 or not np.isfinite(raw_step):
        return 1.0
    magnitude = 10 ** np.floor(np.log10(raw_step))
    for multiplier in (1.0, 2.0, 5.0, 10.0):
        step = multiplier * magnitude
        if step >= raw_step:
            return float(step)
    return float(10.0 * magnitude)


def _padded_axis_limits(
    series: list[np.ndarray],
    headroom: float = MATPLOTLIB_Y_HEADROOM_FRACTION,
    footroom: float = MATPLOTLIB_Y_FOOTROOM_FRACTION,
) -> tuple[float, float] | None:
    finite_parts = [np.asarray(values, dtype=np.float64)[np.isfinite(values)] for values in series]
    finite_parts = [values for values in finite_parts if values.size]
    if not finite_parts:
        return None
    values = np.concatenate(finite_parts)
    lower = float(np.nanmin(values))
    upper = float(np.nanmax(values))
    if not np.isfinite(lower) or not np.isfinite(upper):
        return None
    span = upper - lower
    if span <= 0:
        scale = max(abs(upper), 1.0)
        span = scale * 0.1
        lower -= span * 0.5
        upper += span * 0.5
    return (
        lower - span * footroom,
        upper + span * headroom,
    )


def _apply_matplotlib_axis_padding(ax, series: list[np.ndarray]) -> tuple[float, float] | None:
    """Add y-range headroom so boxed panel labels do not sit on top of traces."""
    limits = _padded_axis_limits(series)
    if limits is not None:
        ax.set_ylim(*limits)
    return limits


def _window_title(suffix: str, instrument: str) -> str:
    return f"{display_name(instrument)} - {suffix}"


def plot_housekeeping_timeseries(
    ds: xr.Dataset,
    instrument: str,
    title: str,
    output: Path,
    max_time_samples: int = MAX_TIME_SAMPLES,
    exclude_vars: set[str] | None = None,
    x_limits=None,
    max_time_ticks: int = MAX_TIME_TICKS,
) -> list[str]:
    curated_layout = CURATED_HOUSEKEEPING_LAYOUTS.get(instrument)
    if curated_layout:
        save_summary_png(
            ds,
            curated_layout,
            title,
            output,
            max_time_samples=max_time_samples,
            x_limits=x_limits,
            max_time_ticks=max_time_ticks,
        )
        return sorted(summary_trace_vars(curated_layout))

    ds = _slice_dataset_time_limits(ds, x_limits)
    ds = downsample_time(ds, max_time_samples=max_time_samples)
    times = _time_index(ds)
    names = [name for name in numeric_time_vars(ds) if not exclude_vars or name not in exclude_vars]
    if len(times) == 0 or not names:
        raise ValueError(f"No numeric {instrument} time-series variables available")

    max_height = 42.0 if instrument == "power" else 34.0
    per_var = 1.0 if instrument == "power" else 1.1
    height = max(8.0, min(max_height, per_var * len(names)))
    fig, axes = plt.subplots(len(names), 1, figsize=(13, height), sharex=True, squeeze=False)
    axes = axes[:, 0]
    colors = [COLOR["teal"], COLOR["red"], COLOR["green"], COLOR["purple"], COLOR["brown"], COLOR["magenta"], COLOR["olive"], COLOR["blue"]]
    for idx, (ax, name) in enumerate(zip(axes, names, strict=False)):
        values = np.asarray(ds[name].values, dtype=np.float64) * display_scale(name)
        trace_times, trace_values = _trace_time_values(times, values)
        trace_times, trace_values = _insert_line_gap_breaks(trace_times, trace_values)
        drawstyle = "steps-post" if is_status_like_var(name) else "default"
        if len(trace_times):
            ax.plot(trace_times, trace_values, color=colors[idx % len(colors)], linewidth=0.8, drawstyle=drawstyle)
        ax.set_ylabel(human_axis_label(name), fontsize=7, rotation=0, ha="right", va="center")
        ax.grid(True, color=PLOT_GRID, linewidth=0.4)
        ax.tick_params(axis="y", labelsize=7)

    apply_quicklook_time_axis(
        axes[-1],
        times,
        label_rotation=90,
        label_size=8,
        x_limits=x_limits,
        max_ticks=max_time_ticks,
    )
    axes[-1].set_xlabel("Time (UTC)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.subplots_adjust(left=0.27, bottom=0.08, top=0.96, hspace=0.18)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Wrote {output}")
    return names


def plot_housekeeping_last_24h(
    zarr_path: Path,
    output: Path,
    instrument: str,
    max_time_samples: int = MAX_TIME_SAMPLES,
) -> list[str]:
    ds = xr.open_zarr(zarr_path, chunks={})
    if "time" not in ds:
        raise KeyError("Dataset is missing a time coordinate")
    time_index = _time_index(ds)
    if len(time_index) == 0:
        raise ValueError("Dataset contains no time samples")
    end_time = time_index.max()
    start_time = end_time - timedelta(hours=24)
    mask = (time_index >= start_time) & (time_index <= end_time)
    if not mask.any():
        raise ValueError("No data in latest 24h")
    window = ds.isel(time=mask).sortby("time")
    title = _window_title(f"{housekeeping_label(instrument) or 'Housekeeping'} - Latest 24 hours", instrument)
    return plot_housekeeping_timeseries(
        window,
        instrument=instrument,
        title=title,
        output=output,
        max_time_samples=max_time_samples,
        x_limits=(start_time, end_time),
    )


def _apply_time_axis_matplotlib(ax, times: pd.DatetimeIndex, *, x_limits=None, max_time_ticks: int = MAX_TIME_TICKS) -> None:
    apply_quicklook_time_axis(ax, times, label_rotation=0, label_size=9, x_limits=x_limits, max_ticks=max_time_ticks)


def save_summary_png(
    ds: xr.Dataset,
    instrument: str,
    title: str,
    output: Path,
    max_time_samples: int = MAX_TIME_SAMPLES,
    x_limits=None,
    max_time_ticks: int = MAX_TIME_TICKS,
) -> int:
    ds = prepare_summary_dataset(ds, instrument)
    ds = _slice_dataset_time_limits(ds, x_limits)
    times = _time_index(ds)
    panels = _active_panels(ds, instrument)
    if len(times) == 0 or not panels:
        raise ValueError(f"No summary time-series panels available for {instrument}")

    fig, axes = plt.subplots(len(panels), 1, figsize=(14, max(7.5, 2.6 * len(panels))), sharex=True, squeeze=False)
    axes = axes[:, 0]
    for ax, (panel, rows) in zip(axes, panels, strict=False):
        right_ax = ax.twinx() if panel.right_axis_label else None
        left_color = None
        right_color = None
        left_axis_values: list[np.ndarray] = []
        right_axis_values: list[np.ndarray] = []
        for trace, values in rows:
            target = right_ax if trace.axis == "right" and right_ax is not None else ax
            drawstyle = (
                "steps-pre"
                if trace.step and panel.key == "operating_plan_schedule"
                else "steps-post" if trace.step else "default"
            )
            trace_times, trace_values = _trace_plot_values(
                times,
                values,
                max_time_samples,
                trace,
                panel.display_horizon_hours,
            )
            if len(trace_times) == 0:
                continue
            trace_label = power_trace_label(ds, trace)
            target.plot(
                trace_times,
                trace_values,
                color=trace.color,
                linewidth=1.25,
                linestyle=_matplotlib_linestyle(trace.dash),
                drawstyle=drawstyle,
                label=trace_label,
            )
            if target is right_ax:
                right_axis_values.append(trace_values)
            else:
                left_axis_values.append(trace_values)
            if trace.axis == "right" and right_color is None:
                right_color = trace.color
            if trace.axis == "left" and left_color is None:
                left_color = trace.color

        left_limits = _apply_matplotlib_axis_padding(ax, left_axis_values)
        if panel.key in SOC_REFERENCE_PANEL_KEYS:
            left_axis_values.append(np.array([MINIMUM_OPERATIONAL_SOC_PCT], dtype=np.float64))
            left_limits = _apply_matplotlib_axis_padding(ax, left_axis_values)
        if right_ax is not None:
            right_limits = _apply_matplotlib_axis_padding(right_ax, right_axis_values)
            if panel.right_axis_label == panel.left_axis_label:
                common_limits = _padded_axis_limits(left_axis_values + right_axis_values)
                if common_limits is not None:
                    ax.set_ylim(*common_limits)
                    right_ax.set_ylim(*common_limits)
            elif left_limits is not None and right_limits is None:
                right_ax.set_ylim(*left_limits)
            if panel.key == "cumulative_power":
                right_limits = _include_zero_in_limits(right_limits)
                if right_limits is not None:
                    right_ax.set_ylim(*right_limits)
                    tick_values, tick_labels = _axis_tick_values(right_limits, step=_axis_tick_step(right_limits))
                    if tick_values:
                        right_ax.set_yticks(tick_values)
                        right_ax.set_yticklabels(tick_labels)

        ax.set_facecolor("white")
        ax.grid(True, color=PLOT_GRID, linewidth=0.5)
        if panel.key in SOC_REFERENCE_PANEL_KEYS:
            ax.axhline(
                MINIMUM_OPERATIONAL_SOC_PCT,
                color=COLOR["black"],
                linewidth=1.4,
                linestyle="--",
                label=MINIMUM_OPERATIONAL_SOC_REFERENCE_LABEL,
            )
        ax.tick_params(axis="y", colors=left_color or COLOR["black"], labelsize=9)
        ax.set_ylabel(panel.left_axis_label, color=left_color or COLOR["black"], fontsize=11)
        if right_ax is not None:
            right_ax.tick_params(axis="y", colors=right_color or COLOR["black"], labelsize=9)
            right_ax.set_ylabel(panel.right_axis_label or "", color=right_color or COLOR["black"], fontsize=11)

        ax.text(
            0.01,
            0.94,
            panel.label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            bbox=dict(facecolor="white", edgecolor=PLOT_TEXT, linewidth=1.0, boxstyle="square,pad=0.3"),
        )

        handles_left, labels_left = ax.get_legend_handles_labels()
        handles_right, labels_right = right_ax.get_legend_handles_labels() if right_ax is not None else ([], [])
        handles = handles_left + handles_right
        labels = labels_left + labels_right
        if handles:
            ax.legend(
                handles,
                labels,
                loc="upper left",
                bbox_to_anchor=(MATPLOTLIB_LEGEND_X, 1.0),
                borderaxespad=0.0,
                fontsize=8,
                frameon=False,
                ncol=1,
            )

    _apply_time_axis_matplotlib(axes[-1], times, x_limits=x_limits, max_time_ticks=max_time_ticks)
    axes[-1].set_xlabel("Time (UTC)", fontsize=12)
    fig.suptitle(title, fontsize=15)
    fig.tight_layout()
    fig.subplots_adjust(left=0.08, right=MATPLOTLIB_PANEL_RIGHT, bottom=0.08, top=0.95, hspace=0.05)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Wrote {output}")
    return len(panels)


def plot_summary_last_24h(
    zarr_path: Path,
    output: Path,
    instrument: str,
    max_time_samples: int = MAX_TIME_SAMPLES,
) -> int:
    ds = xr.open_zarr(zarr_path, chunks={})
    if "time" not in ds:
        raise KeyError("Dataset is missing a time coordinate")
    time_index = _time_index(ds)
    if len(time_index) == 0:
        raise ValueError("Dataset contains no time samples")
    end_time = time_index.max()
    start_time = end_time - timedelta(hours=24)
    if instrument == "power":
        context_start = (pd.Timestamp(start_time) - pd.Timedelta(days=max(0, POWER_CUMULATIVE_CONTEXT_DAYS))).normalize()
    else:
        context_start = start_time
    mask = (time_index >= context_start) & (time_index <= end_time)
    if not mask.any():
        raise ValueError("No data in latest 24h")
    window = ds.isel(time=mask).sortby("time")
    if instrument == "power":
        window = window.copy(deep=False)
        window.attrs[SUMMARY_DISPLAY_START_ATTR] = pd.Timestamp(start_time).isoformat()
        window.attrs[SUMMARY_DISPLAY_END_ATTR] = pd.Timestamp(end_time).isoformat()
    title = _window_title("Latest 24 hours", instrument)
    return save_summary_png(
        window,
        instrument=instrument,
        title=title,
        output=output,
        max_time_samples=max_time_samples,
        x_limits=(start_time, end_time),
    )


def build_summary_plotly(
    ds: xr.Dataset,
    instrument: str,
    title: str | None = None,
    max_time_samples: int = INTERACTIVE_MAX_TIME_SAMPLES,
    x_limits=None,
    panel_groups: set[str] | None = None,
) -> go.Figure:
    ds = prepare_summary_dataset(ds, instrument)
    times = _time_index(ds)
    panels = _active_panels(ds, instrument)
    if instrument == "power" and panel_groups is not None:
        panels = [
            panel_rows
            for panel_rows in panels
            if _power_panel_time_group(panel_rows[0].key) in panel_groups
        ]
    if len(times) == 0 or not panels:
        raise ValueError(f"No summary time-series panels available for {instrument}")

    if instrument == "power":
        panels = _order_power_panels_by_time_group(panels)
    panel_domain_end = PLOTLY_SUMMARY_PANEL_DOMAIN_END
    legend_x = PLOTLY_SUMMARY_LEGEND_X
    right_margin = PLOTLY_SUMMARY_RIGHT_MARGIN
    if instrument == "power":
        per_panel_height = PLOTLY_SUMMARY_POWER_PANEL_HEIGHT
        max_height = PLOTLY_SUMMARY_POWER_MAX_HEIGHT
        requested_height = (
            per_panel_height * len(panels)
            + PLOTLY_SUMMARY_POWER_PANEL_GAP * max(0, len(panels) - 1)
            + 90
        )
        figure_height = max(520, min(max_height, requested_height))
        vertical_spacing = PLOTLY_SUMMARY_POWER_PANEL_GAP / figure_height if len(panels) > 1 else 0.0
    else:
        per_panel_height = PLOTLY_SUMMARY_PANEL_HEIGHT
        max_height = PLOTLY_SUMMARY_MAX_HEIGHT
        figure_height = max(520, min(max_height, per_panel_height * len(panels) + 90))
        vertical_spacing = 0.028 if len(panels) >= 6 else 0.04
    separate_time_axes = instrument == "power" and any(
        _power_panel_time_group(panel.key) != "observed" for panel, _rows in panels
    )
    base_time_start = times.min()
    base_time_end = times.max()
    if instrument == "power":
        display_start = _summary_display_timestamp(ds.attrs.get(SUMMARY_DISPLAY_START_ATTR))
        display_end = _summary_display_timestamp(ds.attrs.get(SUMMARY_DISPLAY_END_ATTR))
        if display_start is not None and display_end is not None and display_end > display_start:
            base_time_start = display_start
            base_time_end = display_end
    if x_limits is not None:
        try:
            requested_start, requested_end = (pd.Timestamp(value) for value in x_limits)
            if requested_start.tz is not None:
                requested_start = requested_start.tz_convert("UTC").tz_localize(None)
            if requested_end.tz is not None:
                requested_end = requested_end.tz_convert("UTC").tz_localize(None)
            if pd.notna(requested_start) and pd.notna(requested_end) and requested_end > requested_start:
                base_time_start = requested_start
                base_time_end = requested_end
        except Exception:
            pass

    fig = make_subplots(
        rows=len(panels),
        cols=1,
        shared_xaxes=not separate_time_axes,
        vertical_spacing=vertical_spacing,
        specs=[[{"secondary_y": panel.right_axis_label is not None}] for panel, _rows in panels],
        subplot_titles=[power_panel_label(ds, panel) if instrument == "power" else panel.label for panel, _rows in panels],
    )

    panel_height = (1.0 - vertical_spacing * (len(panels) - 1)) / len(panels)
    legend_layouts: dict[str, dict[str, object]] = {}
    plot_time_start = base_time_start
    plot_time_end = base_time_end
    panel_x_ranges: dict[int, tuple[pd.Timestamp, pd.Timestamp]] = {}
    axis_label_annotations: list[dict[str, object]] = []
    for row_index, (panel, rows) in enumerate(panels, start=1):
        legend_name = "legend" if row_index == 1 else f"legend{row_index}"
        panel_top = 1.0 - (row_index - 1) * (panel_height + vertical_spacing)
        legend_layouts[legend_name] = dict(
            x=legend_x,
            xanchor="left",
            y=max(0.02, panel_top - 0.02),
            yanchor="top",
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor=PLOT_BORDER,
            borderwidth=1,
            font=dict(size=9, color=PLOT_TEXT),
            itemsizing="constant",
            tracegroupgap=2,
        )
        left_color = None
        right_color = None
        left_axis_values: list[np.ndarray] = []
        right_axis_values: list[np.ndarray] = []
        left_axis_has_finite_data = False
        right_axis_has_finite_data = False
        panel_time_start: pd.Timestamp | None = None
        panel_time_end: pd.Timestamp | None = None
        panel_time_group = _power_panel_time_group(panel.key) if instrument == "power" else "observed"
        for trace, values in rows:
            secondary = trace.axis == "right" and panel.right_axis_label is not None
            if secondary and right_color is None:
                right_color = trace.color
            if not secondary and left_color is None:
                left_color = trace.color
            trace_times, trace_values = _trace_plot_values(
                times,
                values,
                max_time_samples,
                trace,
                panel.display_horizon_hours,
            )
            if len(trace_times) == 0:
                continue
            trace_start = trace_times.min()
            trace_end = trace_times.max()
            if separate_time_axes and panel_time_group != "observed":
                if panel_time_start is None or trace_start < panel_time_start:
                    panel_time_start = trace_start
                if panel_time_end is None or trace_end > panel_time_end:
                    panel_time_end = trace_end
            elif not separate_time_axes:
                if trace_start < plot_time_start:
                    plot_time_start = trace_start
                if trace_end > plot_time_end:
                    plot_time_end = trace_end
            if secondary:
                right_axis_values.append(trace_values)
                right_axis_has_finite_data = right_axis_has_finite_data or bool(np.isfinite(trace_values).any())
            else:
                left_axis_values.append(trace_values)
                left_axis_has_finite_data = left_axis_has_finite_data or bool(np.isfinite(trace_values).any())
            trace_label = power_trace_label(ds, trace)
            # Power's standard current view contains many independent line
            # traces. WebGL avoids creating thousands of SVG nodes on phones
            # and browsers, while stepped state schedules stay SVG so their
            # horizontal/vertical transitions remain exact.
            trace_type = go.Scattergl if instrument == "power" and not trace.step else go.Scatter
            fig.add_trace(
                trace_type(
                    x=trace_times,
                    y=trace_values,
                    mode="lines",
                    name=trace_label,
                    legend=legend_name,
                    line=dict(
                        color=trace.color,
                        width=trace.line_width,
                        dash=trace.dash or "solid",
                        shape=(
                            "vh"
                            if trace.step and panel.key == "operating_plan_schedule"
                            else "hv" if trace.step else "linear"
                        ),
                    ),
                    opacity=trace.opacity,
                    hovertemplate=f"Time=%{{x}}<br>{trace_label}=%{{y:.6g}}<extra></extra>",
                    connectgaps=False,
                    showlegend=True,
                ),
                row=row_index,
                col=1,
                secondary_y=secondary,
            )
        if instrument == "power" and panel.key in OPERATING_SCHEDULE_SHADE_PANELS:
            _add_operating_schedule_bands(fig, ds, row=row_index)
        if instrument == "power" and panel.key == "operating_plan_schedule":
            schedule_fields = (
                "OperatingCL61OptimizedCL61On",
                "OperatingCL61OptimizedRadarOn",
                "OperatingCL61OptimizedHATPROOn",
            )
            schedule_has_on = any(
                np.any(
                    np.isfinite(np.asarray(ds[name].values, dtype=np.float64))
                    & (np.asarray(ds[name].values, dtype=np.float64) > 0)
                )
                for name in schedule_fields
                if name in ds
            )
            if not schedule_has_on:
                presentation = cl61_schedule_presentation(ds)
                fig.add_annotation(
                    text=presentation.annotation,
                    xref=f"x{row_index}" if row_index > 1 else "x",
                    yref=f"y{row_index}" if row_index > 1 else "y",
                    x=panel_time_start + (panel_time_end - panel_time_start) / 2 if panel_time_start is not None and panel_time_end is not None else base_time_start + (base_time_end - base_time_start) / 2,
                    y=0.5,
                    xanchor="center",
                    yanchor="middle",
                    showarrow=False,
                    font=dict(color=COLOR["slate"], size=12),
                )
        if panel.key in SOC_REFERENCE_PANEL_KEYS:
            reference_start = panel_time_start or base_time_start
            reference_end = panel_time_end or base_time_end
            if pd.notna(reference_start) and pd.notna(reference_end) and reference_end > reference_start:
                fig.add_trace(
                    go.Scatter(
                        x=[reference_start, reference_end],
                        y=[MINIMUM_OPERATIONAL_SOC_PCT, MINIMUM_OPERATIONAL_SOC_PCT],
                        mode="lines",
                        name=MINIMUM_OPERATIONAL_SOC_REFERENCE_LABEL,
                        legend=legend_name,
                        line=dict(color=COLOR["black"], width=2.0, dash="dash"),
                        hovertemplate=f"{MINIMUM_OPERATIONAL_SOC_REFERENCE_LABEL}<extra></extra>",
                        showlegend=True,
                    ),
                    row=row_index,
                    col=1,
                    secondary_y=False,
                )
                left_axis_has_finite_data = True
        if panel.right_axis_label is not None and right_axis_has_finite_data and not left_axis_has_finite_data:
            anchor_start = panel_time_start or base_time_start
            anchor_end = panel_time_end or base_time_end
            fig.add_trace(
                go.Scatter(
                    x=[anchor_start, anchor_end],
                    y=[0.0, 0.0],
                    mode="lines",
                    name="Primary axis anchor",
                    line=dict(width=0),
                    opacity=0.0,
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=row_index,
                col=1,
                secondary_y=False,
            )
            left_axis_values.append(np.array([0.0], dtype=np.float64))
        left_range = _padded_axis_limits(left_axis_values, headroom=0.08, footroom=0.04)
        right_range = _padded_axis_limits(right_axis_values, headroom=0.08, footroom=0.04)
        if panel.key in SOC_REFERENCE_PANEL_KEYS:
            left_axis_values.append(np.array([MINIMUM_OPERATIONAL_SOC_PCT], dtype=np.float64))
            left_range = _padded_axis_limits(left_axis_values, headroom=0.08, footroom=0.04)
        if panel.key == "cumulative_power":
            right_range = _include_zero_in_limits(right_range)
        if panel.right_axis_label == panel.left_axis_label:
            common_range = _padded_axis_limits(left_axis_values + right_axis_values, headroom=0.08, footroom=0.04)
            if common_range is not None:
                left_range = common_range
                right_range = common_range
        fig.update_yaxes(
            title_text=panel.left_axis_label,
            automargin=True,
            showgrid=True,
            gridcolor=PLOT_GRID,
            linecolor=PLOT_LINE,
            tickfont=dict(color=left_color or COLOR["black"], size=10),
            title_font=dict(color=left_color or COLOR["black"], size=11),
            range=list(left_range) if left_range is not None else None,
            row=row_index,
            col=1,
            secondary_y=False,
        )
        if panel.right_axis_label is not None:
            right_tick_values: list[float] = []
            right_tick_labels: list[str] = []
            if panel.key == "cumulative_power":
                right_tick_values, right_tick_labels = _axis_tick_values(right_range, step=_axis_tick_step(right_range))
            fig.update_yaxes(
                title_text=panel.right_axis_label,
                automargin=True,
                showgrid=False,
                gridcolor=PLOT_GRID,
                zeroline=False,
                zerolinecolor=PLOT_GRID,
                zerolinewidth=1,
                linecolor=PLOT_LINE,
                tickfont=dict(color=right_color or COLOR["black"], size=10),
                title_font=dict(color=right_color or COLOR["black"], size=11),
                range=list(right_range) if right_range is not None else None,
                tickmode="array" if right_tick_values else "auto",
                tickvals=right_tick_values or None,
                ticktext=right_tick_labels or None,
                ticks="outside" if panel.key == "cumulative_power" else "",
                ticklen=5 if panel.key == "cumulative_power" else None,
                row=row_index,
                col=1,
                secondary_y=True,
            )
        axis_label_y = panel_top - min(0.028, panel_height * 0.32)
        axis_label_annotations.append(
            dict(
                x=0.0,
                y=axis_label_y,
                xref="paper",
                yref="paper",
                text=panel.left_axis_label,
                showarrow=False,
                xanchor="left",
                yanchor="top",
                align="left",
                font=dict(color=left_color or COLOR["black"], size=10),
                bgcolor="rgba(255,255,255,0.78)",
                bordercolor="rgba(0,0,0,0)",
            )
        )
        if panel.right_axis_label is not None:
            axis_label_annotations.append(
                dict(
                    x=panel_domain_end,
                    y=axis_label_y,
                    xref="paper",
                    yref="paper",
                    text=panel.right_axis_label,
                    showarrow=False,
                    xanchor="right",
                    yanchor="top",
                    align="right",
                    font=dict(color=right_color or COLOR["black"], size=10),
                    bgcolor="rgba(255,255,255,0.78)",
                    bordercolor="rgba(0,0,0,0)",
                )
            )
        if separate_time_axes and panel_time_group != "observed" and panel_time_start is not None and panel_time_end is not None:
            panel_x_ranges[row_index] = (panel_time_start, panel_time_end)
        else:
            panel_x_ranges[row_index] = (base_time_start, base_time_end)

    if separate_time_axes:
        group_ranges: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
        for row_index, (panel, _rows) in enumerate(panels, start=1):
            group_name = _power_panel_time_group(panel.key)
            start, end = panel_x_ranges[row_index]
            if group_name in group_ranges:
                group_start, group_end = group_ranges[group_name]
                group_ranges[group_name] = (min(group_start, start), max(group_end, end))
            else:
                group_ranges[group_name] = (start, end)

        group_axis_roots: dict[str, str] = {}
        for row_index, (panel, _rows) in enumerate(panels, start=1):
            group_name = _power_panel_time_group(panel.key)
            start, end = group_ranges[group_name]
            tick_options = _plotly_time_tick_options(start, end)
            fig.update_xaxes(
                domain=[0.0, panel_domain_end],
                showgrid=True,
                gridcolor=PLOT_GRID,
                linecolor=PLOT_LINE,
                tickfont=dict(color=PLOT_TEXT, size=11),
                range=[start, end],
                showticklabels=True,
                title_text=POWER_PANEL_TIME_AXIS_LABELS[group_name],
                title_standoff=10,
                **tick_options,
                row=row_index,
                col=1,
            )
            axis_reference = "x" if row_index == 1 else f"x{row_index}"
            if group_name in group_axis_roots:
                fig.update_xaxes(matches=group_axis_roots[group_name], row=row_index, col=1)
            else:
                group_axis_roots[group_name] = axis_reference
    else:
        tick_options = _plotly_time_tick_options(plot_time_start, plot_time_end)
        fig.update_xaxes(
            domain=[0.0, panel_domain_end],
            showgrid=True,
            gridcolor=PLOT_GRID,
            linecolor=PLOT_LINE,
            tickfont=dict(color=PLOT_TEXT, size=11),
            range=[plot_time_start, plot_time_end],
            **tick_options,
        )
        fig.update_xaxes(title_text="Time (UTC)", row=len(panels), col=1)
    fig.update_layout(
        showlegend=True,
        height=figure_height,
        margin=dict(l=80, r=right_margin, t=60, b=70),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color=PLOT_TEXT, size=12),
        title=dict(text=title or display_name(instrument), x=0.01, xanchor="left", font=dict(size=17, color=PLOT_TEXT)),
        **legend_layouts,
    )
    for ann in fig.layout.annotations:
        ann.update(
            x=0.01,
            xref="paper",
            xanchor="left",
            bgcolor="white",
            bordercolor=PLOT_TEXT,
            borderwidth=1,
            font=dict(size=12, color=PLOT_TEXT),
        )
    for annotation in axis_label_annotations:
        fig.add_annotation(**annotation)
    return fig
