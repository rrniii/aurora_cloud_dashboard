from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr

from grouped_timeseries import (
    POWER_PANEL_TIME_GROUP_BY_KEY,
    PLOTLY_SUMMARY_POWER_MAX_HEIGHT,
    PLOTLY_SUMMARY_POWER_PANEL_GAP,
    PLOTLY_SUMMARY_POWER_PANEL_HEIGHT,
    SUMMARY_DISPLAY_END_ATTR,
    SUMMARY_DISPLAY_START_ATTR,
    SUMMARY_LAYOUTS,
    PanelSpec,
    TraceSpec,
    _downsample_trace,
    _downsample_trace_with_gap_breaks,
    _plotly_time_tick_options,
    build_power_forecast_info,
    build_summary_plotly,
    cl61_schedule_presentation,
    operating_mode_intervals,
    power_panel_label,
    power_trace_label,
)


def test_power_time_ticks_include_date_time_and_utc() -> None:
    options = _plotly_time_tick_options(pd.Timestamp("2026-07-19T00:00:00"), pd.Timestamp("2026-07-20T00:00:00"))

    assert options["tickformat"] == "%a %d %b<br>%H:%M UTC"


def test_radiation_layout_excludes_noisy_flux_plate_b() -> None:
    panel = next(panel for panel in SUMMARY_LAYOUTS["asfs-logger"] if panel.key == "flux_plates")

    assert panel.label == "Flux Plate"
    assert [(trace.var, trace.label) for trace in panel.traces] == [
        ("fp_A_Wm2_Avg", "Flux Plate A"),
    ]


def test_trace_downsampling_retains_short_extrema_within_budget() -> None:
    times = pd.date_range("2026-07-15T00:00:00", periods=100, freq="min")
    values = np.zeros(100)
    values[17] = 99.0
    values[83] = -42.0

    sampled_times, sampled_values = _downsample_trace(times, values, max_time_samples=20)

    assert len(sampled_times) <= 20
    assert 99.0 in sampled_values
    assert -42.0 in sampled_values
    assert sampled_times.is_monotonic_increasing


def test_trace_downsampling_preserves_real_gaps_without_artificial_breaks() -> None:
    times = pd.date_range("2026-07-15T00:00:00", periods=40, freq="min").append(
        pd.date_range("2026-07-15T03:00:00", periods=40, freq="min")
    )
    values = np.linspace(0.0, 1.0, len(times))

    sampled_times, sampled_values = _downsample_trace_with_gap_breaks(times, values, max_time_samples=20)

    assert len(sampled_times) <= 20
    assert sampled_times.is_monotonic_increasing
    assert np.isnan(sampled_values).any()


def _power_layout_dataset() -> xr.Dataset:
    times = pd.date_range("2026-07-15T00:00:00", periods=33, freq="3h")
    values = np.linspace(1.0, 2.0, len(times))
    return xr.Dataset(
        {
            "observed_1": (("time",), values * 100.0),
            "observed_2": (("time",), values),
            "forecast_24": (("time",), 70.0 - values),
            "forecast_input": (("time",), values * 200.0),
            "forecast_96": (("time",), 65.0 - values),
            "forecast_load": (("time",), 60.0 - values),
            "hindcast": (("time",), 68.0 - values),
            "forecast_skill": (("time",), values),
            "ensemble_skill": (("time",), values / 2.0),
            "power_skill": (("time",), values * 10.0),
        },
        coords={"time": times},
    )


def test_soc_hindcast_labels_explain_forecast_issue_time() -> None:
    panel = next(panel for panel in SUMMARY_LAYOUTS["power"] if panel.key == "soc_hindcast")

    assert panel.label == "Battery SOC: Measured vs Earlier Forecasts"
    assert panel.description is not None
    assert "before that time" in panel.description
    assert [trace.label for trace in panel.traces] == [
        "Measured battery SOC",
        "Forecast issued 6 h before valid time",
        "Forecast issued 24 h before valid time",
        "Forecast issued 48 h before valid time",
        "Forecast issued 72 h before valid time",
    ]


def test_every_power_forecast_panel_has_shared_implementation_info() -> None:
    forecast_groups = {"forecast_24h", "forecast_96h", "verification"}
    panels = [
        panel
        for panel in SUMMARY_LAYOUTS["power"]
        if POWER_PANEL_TIME_GROUP_BY_KEY.get(panel.key) in forecast_groups
    ]

    for panel in panels:
        info = build_power_forecast_info(panel.key)
        assert info is not None, panel.key
        assert info["title"]
        assert info["implementation"]
        assert info["metrics"]


def test_every_96_hour_power_panel_clips_its_rendered_traces_to_96_hours() -> None:
    anchor = pd.Timestamp("2026-09-06T00:00:00")
    times = pd.date_range(anchor, anchor + pd.Timedelta(hours=240), freq="1h")
    fields: dict[str, tuple[tuple[str], np.ndarray]] = {}
    for panel in SUMMARY_LAYOUTS["power"]:
        if POWER_PANEL_TIME_GROUP_BY_KEY.get(panel.key) != "forecast_96h":
            continue
        for trace in panel.traces:
            if trace.var.endswith("Probability"):
                values = np.linspace(0.0, 1.0, len(times))
            elif trace.var.endswith("ActiveCount"):
                values = np.ones(len(times))
            elif trace.var.endswith("On") or trace.var.endswith("ModeCode"):
                values = np.zeros(len(times))
            else:
                values = np.linspace(80.0, 60.0, len(times))
            fields[trace.var] = (("time",), values)
    dataset = xr.Dataset(
        fields,
        coords={"time": times},
        attrs={
            "forecast_initial_soc_time": anchor.isoformat(),
            "operating_initial_soc_time": anchor.isoformat(),
            "operating_decision_horizon_hours": "96",
        },
    )

    figure = build_summary_plotly(
        dataset,
        "power",
        panel_groups={"forecast_96h"},
    )

    expected_end = anchor + pd.Timedelta(hours=96)
    assert figure.data
    for trace in figure.data:
        trace_times = pd.DatetimeIndex(pd.to_datetime(trace.x))
        assert trace_times.min() >= anchor, trace.name
        assert trace_times.max() <= expected_end, trace.name
    for index in range(1, 5):
        axis = getattr(figure.layout, "xaxis" if index == 1 else f"xaxis{index}")
        assert pd.Timestamp(axis.range[0]) == anchor
        assert pd.Timestamp(axis.range[1]) == expected_end


def test_system_soc_quantiles_have_distinct_accessible_trace_styles() -> None:
    panel = next(
        panel
        for panel in SUMMARY_LAYOUTS["power"]
        if panel.key == "soc_ecmwf_forecast"
    )
    traces = {trace.var: trace for trace in panel.traces}
    p10 = traces["SystemAsIsDecisionSOCP10"]
    p50 = traces["SystemAsIsDecisionSOCP50"]
    p90 = traces["SystemAsIsDecisionSOCP90"]

    assert p10.color != p90.color
    assert (p10.dash or "solid") != (p90.dash or "solid")
    assert p50.line_width > max(p10.line_width, p90.line_width)
    assert len(
        {
            (p10.color, p10.dash or "solid"),
            (p50.color, p50.dash or "solid"),
            (p90.color, p90.dash or "solid"),
        }
    ) == 3


def test_operating_schedule_uses_interval_end_step_alignment() -> None:
    anchor = pd.Timestamp("2026-09-06T00:00:00")
    times = pd.date_range(anchor, periods=4, freq="1h")
    dataset = xr.Dataset(
        {
            "OperatingCL61OptimizedActiveCount": (("time",), [0.0, 1.0, 1.0, 0.0]),
            "OperatingCL61OptimizedCL61On": (("time",), [0.0, 1.0, 1.0, 0.0]),
        },
        coords={"time": times},
    )

    figure = build_summary_plotly(
        dataset,
        "power",
        panel_groups={"forecast_96h"},
    )

    assert figure.data
    assert all(trace.line.shape == "vh" for trace in figure.data)


def test_operating_scenarios_start_with_current_state_reference() -> None:
    panel = next(panel for panel in SUMMARY_LAYOUTS["power"] if panel.key == "operating_plan_scenarios")

    assert panel.traces[0].var == "OperatingCurrentSOCP50"
    assert panel.traces[0].label == "Current load / system as-is"
    assert panel.traces[1].var == "OperatingP50ContinuationSOCP50"
    assert panel.traces[0].line_width > max(trace.line_width for trace in panel.traces[1:])
    info = build_power_forecast_info(panel.key)
    assert info is not None
    assert "current exact instrument state is the reference" in info["summary"]
    assert "P50 continuation" in info["summary"]
    assert "never operate PDU outlets" in info["implementation"]
    assert "states are never blended" in info["implementation"]


def test_unsafe_cl61_fallback_is_labelled_as_infeasible() -> None:
    panel = next(panel for panel in SUMMARY_LAYOUTS["power"] if panel.key == "operating_plan_schedule")
    ds = xr.Dataset(
        attrs={
            "operating_optimized_safe": "false",
            "operating_optimized_status": "no_safe_schedule",
            "operating_optimized_reason": "Radar and HATPRO remain fixed on. This is not a recommendation.",
            "operating_optimized_base_mode_label": "DC + Radar + HATPRO",
        }
    )

    presentation = cl61_schedule_presentation(ds)
    info = build_power_forecast_info(panel.key, ds)

    assert presentation.title == "No Feasible CL61 Schedule"
    assert presentation.trace_label == "Unsafe fallback (CL61 off)"
    assert power_panel_label(ds, panel) == presentation.title
    assert info is not None
    assert info["title"] == presentation.title
    assert "not a recommendation" in info["implementation"]

    load_trace = next(
        trace
        for candidate in SUMMARY_LAYOUTS["power"]
        if candidate.key == "ecmwf_solar_forecast"
        for trace in candidate.traces
        if trace.var == "OperatingCL61OptimizedLoadP50Watts"
    )
    assert power_trace_label(ds, load_trace) == "Unsafe fallback load (CL61 off)"


def test_unsafe_existing_cl61_is_labelled_as_held_not_switched_off() -> None:
    panel = next(panel for panel in SUMMARY_LAYOUTS["power"] if panel.key == "operating_plan_schedule")
    ds = xr.Dataset(
        attrs={
            "operating_optimized_safe": "false",
            "operating_optimized_status": "no_safe_schedule",
            "operating_optimized_schedule_policy": "cl61_primary_v1",
            "operating_cl61_primary_continuation_required": "true",
            "operating_optimized_held_existing_instruments": '["Radar"]',
        }
    )

    presentation = cl61_schedule_presentation(ds)
    cl61_trace = next(trace for trace in panel.traces if trace.var == "OperatingCL61OptimizedCL61On")
    radar_trace = next(trace for trace in panel.traces if trace.var == "OperatingCL61OptimizedRadarOn")
    load_trace = next(
        trace
        for candidate in SUMMARY_LAYOUTS["power"]
        if candidate.key == "ecmwf_solar_forecast"
        for trace in candidate.traces
        if trace.var == "OperatingCL61OptimizedLoadP50Watts"
    )

    assert presentation.trace_label == "Unsafe held CL61 continuation"
    assert power_trace_label(ds, cl61_trace) == "Unsafe held CL61 continuation"
    assert power_trace_label(ds, radar_trace) == "Observed Radar held (not scheduled)"
    assert power_trace_label(ds, load_trace) == "Unsafe held-CL61 continuation load"


def test_priority_schedule_uses_additive_sum_and_three_instrument_labels() -> None:
    panel = next(panel for panel in SUMMARY_LAYOUTS["power"] if panel.key == "operating_plan_schedule")
    ds = xr.Dataset(
        attrs={
            "operating_optimized_safe": "true",
            "operating_optimized_status": "safe_schedule",
            "operating_optimized_priority_order": '["CL61", "Radar", "HATPRO"]',
            "operating_optimized_reason": (
                "A safe advisory priority schedule keeps P10 SOC at or above 40%."
            ),
        }
    )

    presentation = cl61_schedule_presentation(ds)
    labels = [power_trace_label(ds, trace) for trace in panel.traces]

    assert presentation.title == "Recommended Additive Instrument Schedule"
    assert labels == [
        "Total active instruments (additive sum)",
        "Recommended CL61 schedule",
        "Recommended Radar schedule",
        "Recommended HATPRO schedule",
    ]
    implementation = build_power_forecast_info(panel.key, ds)["implementation"]
    assert "maximises controlled energy first" in implementation
    assert "CL61, then Radar, then HATPRO breaks" in implementation
    assert "additive sum from 0 to 3" in implementation


def _power_layout_panels() -> tuple[PanelSpec, ...]:
    panel_specs = (
        ("renewables", "Renewables", "observed_1"),
        ("battery_charging", "Battery Charging", "observed_2"),
        ("soc_24h_forecast", "SOC Next 24 h Forecast", "forecast_24"),
        ("ecmwf_solar_forecast", "ECMWF Solar & Load Forecast", "forecast_input"),
        ("soc_ecmwf_forecast", "SOC 96 h Forecast", "forecast_96"),
        ("operating_plan_scenarios", "Learned Operating-Mode SOC Plans", "forecast_load"),
        ("soc_hindcast", "SOC Hindcast: Forecasts vs Observed", "hindcast"),
        ("soc_forecast_skill", "SOC Forecast Verification", "forecast_skill"),
        ("soc_ensemble_skill", "SOC Ensemble Verification", "ensemble_skill"),
        ("forecast_power_skill", "Solar and Load Forecast Verification", "power_skill"),
    )
    return tuple(
        PanelSpec(key, label, "Value", None, (TraceSpec(variable, label, "#0b7285"),))
        for key, label, variable in panel_specs
    )


def test_power_desktop_panels_are_tall_and_grouped_by_time_axis() -> None:
    ds = _power_layout_dataset()
    times = pd.DatetimeIndex(ds["time"].values)

    with patch.dict(SUMMARY_LAYOUTS, {"power": _power_layout_panels()}):
        figure = build_summary_plotly(ds, "power", x_limits=(times[0], times[8]))

    expected_titles = [
        "Renewables",
        "Battery Charging",
        "SOC Next 24 h Forecast",
        "ECMWF Solar & Load Forecast",
        "SOC 96 h Forecast",
        "Learned Operating-Mode SOC Plans",
        "SOC Hindcast: Forecasts vs Observed",
        "SOC Forecast Verification",
        "SOC Ensemble Verification",
        "Solar and Load Forecast Verification",
    ]
    assert [annotation.text for annotation in figure.layout.annotations[: len(expected_titles)]] == expected_titles
    expected_height = min(
        PLOTLY_SUMMARY_POWER_MAX_HEIGHT,
        PLOTLY_SUMMARY_POWER_PANEL_HEIGHT * len(expected_titles)
        + PLOTLY_SUMMARY_POWER_PANEL_GAP * (len(expected_titles) - 1)
        + 90,
    )
    assert figure.layout.height == expected_height
    first_axis = figure.layout.yaxis.domain
    second_axis = figure.layout.yaxis2.domain
    assert (first_axis[0] - second_axis[1]) * figure.layout.height >= PLOTLY_SUMMARY_POWER_PANEL_GAP - 1

    xaxes = [
        getattr(figure.layout, "xaxis" if index == 1 else f"xaxis{index}")
        for index in range(1, len(expected_titles) + 1)
    ]
    assert all(axis.showticklabels for axis in xaxes)
    assert [axis.title.text for axis in xaxes] == [
        "Time (UTC)",
        "Time (UTC)",
        "Forecast Time (UTC)",
        "Forecast Time (UTC)",
        "Forecast Time (UTC)",
        "Forecast Time (UTC)",
        "Verification Time (UTC)",
        "Verification Time (UTC)",
        "Verification Time (UTC)",
        "Verification Time (UTC)",
    ]
    assert [axis.matches for axis in xaxes] == [None, "x", None, None, "x4", "x4", None, "x7", "x7", "x7"]


def test_power_panel_groups_split_current_from_forecast() -> None:
    ds = _power_layout_dataset()

    with patch.dict(SUMMARY_LAYOUTS, {"power": _power_layout_panels()}):
        current = build_summary_plotly(ds, "power", panel_groups={"observed"})
        forecast = build_summary_plotly(
            ds,
            "power",
            panel_groups={"forecast_24h", "forecast_96h", "verification"},
        )

    assert [annotation.text for annotation in current.layout.annotations[:2]] == [
        "Renewables",
        "Battery Charging",
    ]
    forecast_titles = [annotation.text for annotation in forecast.layout.annotations[:8]]
    assert "Renewables" not in forecast_titles
    assert forecast_titles == [
        "SOC Next 24 h Forecast",
        "ECMWF Solar & Load Forecast",
        "SOC 96 h Forecast",
        "Learned Operating-Mode SOC Plans",
        "SOC Hindcast: Forecasts vs Observed",
        "SOC Forecast Verification",
        "SOC Ensemble Verification",
        "Solar and Load Forecast Verification",
    ]


def test_non_power_summary_height_is_unchanged() -> None:
    times = pd.date_range("2026-07-15T00:00:00", periods=5, freq="1h")
    ds = xr.Dataset({"h1_t": (("time",), np.linspace(0.0, 1.0, len(times)))}, coords={"time": times})

    figure = build_summary_plotly(ds, "vaisalamet")

    assert figure.layout.height < PLOTLY_SUMMARY_POWER_PANEL_HEIGHT * 4


def test_operating_mode_intervals_identify_each_planned_instrument() -> None:
    times = pd.date_range("2026-07-15T00:00:00", periods=5, freq="1h")

    intervals = operating_mode_intervals(times, np.array([0, 1, 1, 2, 0]))

    assert [(label, start, end) for start, end, label, _color in intervals] == [
        ("CL61", times[1], times[3]),
        ("Radar", times[3], times[4]),
    ]


def test_right_axis_only_panel_retains_its_primary_subplot_anchor() -> None:
    times = pd.date_range("2026-07-15T00:00:00", periods=5, freq="1h")
    ds = xr.Dataset({"cycles": (("time",), np.arange(len(times), dtype=float))}, coords={"time": times})
    panels = (
        PanelSpec(
            "soc_forecast_skill",
            "SOC Forecast Verification",
            "SOC MAE [percentage points]",
            "Independent ECMWF Cycles [count]",
            (TraceSpec("cycles", "Independent ECMWF Cycles", "#4f7d8d", axis="right"),),
        ),
    )

    with patch.dict(SUMMARY_LAYOUTS, {"power": panels}):
        figure = build_summary_plotly(ds, "power")

    assert len(figure.data) == 2
    right_trace, anchor_trace = figure.data
    assert right_trace.yaxis == "y2"
    assert anchor_trace.yaxis == "y"
    assert anchor_trace.showlegend is False
    assert anchor_trace.opacity == 0.0
    assert figure.layout.yaxis.domain == (0.0, 1.0)
    assert figure.layout.yaxis2.overlaying == "y"


def test_power_prewarm_observed_axes_use_measured_display_window() -> None:
    times = pd.date_range("2026-07-15T10:00:00", periods=41, freq="3h")
    observed_end_index = 8
    observed = np.full(len(times), np.nan)
    observed[: observed_end_index + 1] = np.linspace(50.0, 60.0, observed_end_index + 1)
    forecast = np.full(len(times), np.nan)
    forecast[observed_end_index:] = np.linspace(60.0, 75.0, len(times) - observed_end_index)
    ds = xr.Dataset(
        {
            "BatterySOC": (("time",), observed),
            "BatterySOCForecast": (("time",), forecast),
        },
        coords={"time": times},
        attrs={
            SUMMARY_DISPLAY_START_ATTR: times[0].isoformat(),
            SUMMARY_DISPLAY_END_ATTR: times[observed_end_index].isoformat(),
        },
    )
    panels = (
        PanelSpec("cumulative_power", "Observed SOC", "SOC [%]", None, (TraceSpec("BatterySOC", "Observed", "#468b61"),)),
        PanelSpec(
            "soc_ecmwf_forecast",
            "SOC 96 h Forecast",
            "SOC [%]",
            None,
            (TraceSpec("BatterySOCForecast", "Forecast", "#468b61"),),
        ),
    )

    with patch.dict(SUMMARY_LAYOUTS, {"power": panels}):
        figure = build_summary_plotly(ds, "power")

    assert list(figure.layout.xaxis.range) == [times[0], times[observed_end_index]]
    assert figure.layout.xaxis2.range[1] == times[-1]
