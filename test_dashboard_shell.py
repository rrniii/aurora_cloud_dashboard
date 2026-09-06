from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr

import app


class DashboardShellTests(TestCase):
    def test_naive_utc_datetime_rejects_non_finite_timestamp(self) -> None:
        self.assertIsNone(app._as_naive_utc_datetime(pd.NaT))
        self.assertIsNone(app._as_naive_utc_datetime(np.datetime64("NaT")))

    def test_science_quicklook_falls_back_when_latest_bound_is_nat(self) -> None:
        start = datetime(2026, 9, 6, 0, 0)
        end = datetime(2026, 9, 6, 12, 0)
        with TemporaryDirectory() as tmpdir:
            latest_path = Path(tmpdir) / "latest.png"
            latest_path.touch()
            with (
                patch.object(app, "_cfg", return_value={"quicklook_dir": tmpdir}),
                patch.object(app, "_is_stacked_timeseries_instrument", return_value=True),
                patch.object(app, "_dataset_time_bounds", return_value=(pd.NaT, pd.NaT)),
                patch.object(app, "summary_latest_png", return_value=latest_path),
            ):
                resolved = app._science_quicklook_path_for_interactive(
                    "power",
                    start=start,
                    end=end,
                )

        self.assertEqual(resolved, latest_path)

    def test_auroracam_detail_panes_refresh_without_rebuilding_thumbnail_grid(self) -> None:
        with patch.object(app, "_auroracam_viewer_markup", return_value="<div>selected camera</div>") as render:
            app._refresh_auroracam_detail()

        render.assert_called_once_with(
            app.auroracam_camera.value,
            app.auroracam_date.value,
            app.auroracam_time.value,
        )
        self.assertEqual(app.auroracam_detail.object, "<div>selected camera</div>")
        self.assertEqual(app.mobile_auroracam_detail.object, "<div>selected camera</div>")

    def test_browser_performance_probe_is_development_only(self) -> None:
        with (
            patch.object(app, "SITE_ENV", "development"),
            patch.dict("os.environ", {"AURORA_BROWSER_RUM_ENABLED": "1"}, clear=False),
        ):
            probe = app._browser_performance_probe()

        self.assertIsInstance(probe, app.BrowserPerformanceProbe)
        self.assertIn("browser_first_power_plot", probe._esm)
        self.assertIn("browser_power_section_switch", probe._esm)

        with (
            patch.object(app, "SITE_ENV", "production"),
            patch.dict("os.environ", {"AURORA_BROWSER_RUM_ENABLED": "1"}, clear=False),
        ):
            self.assertIsNone(app._browser_performance_probe())

    def test_browser_performance_probe_rejects_unknown_events(self) -> None:
        probe = app.BrowserPerformanceProbe()
        events = []
        with patch.object(app, "_perf_log", side_effect=lambda event, **fields: events.append((event, fields))):
            probe._handle_msg({"event": "untrusted", "duration_ms": 1})
            probe._handle_msg({"event": "browser_document_ready", "duration_ms": 12.5, "path": "/app"})

        self.assertEqual(events, [("browser_document_ready", {"duration_ms": 12.5, "instrument": "power", "path": "/app"})])

    def test_power_section_prewarm_paths_are_distinct(self) -> None:
        original = app.power_view_select.value
        try:
            app.power_view_select.value = "current"
            current = app._prewarmed_interactive_path("power")
            app.power_view_select.value = "forecast"
            forecast = app._prewarmed_interactive_path("power")
        finally:
            app.power_view_select.value = original

        self.assertEqual(current.name, "power_current_latest_interactive.json")
        self.assertEqual(forecast.name, "power_forecast_latest_interactive.json")

    def test_power_live_window_uses_a_fresh_section_prewarm(self) -> None:
        end = datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)
        start = end - app.DEFAULT_WINDOW
        cache_key = (
            "power", "current", app._interactive_final_quality("power"), "power_latest_5min", "power_latest_5min",
            start.isoformat(), end.isoformat(), 0, 1, "", "", 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        )
        with TemporaryDirectory() as tmpdir, patch.dict(
            "os.environ", {"AURORA_INTERACTIVE_PREWARM_DIR": tmpdir}, clear=False
        ):
            path = Path(tmpdir) / "power_current_latest_interactive.json"
            path.write_text('{"data":[],"layout":{}}', encoding="utf-8")
            self.assertTrue(app._cache_key_targets_latest_prewarm(cache_key, "power"))

    def test_power_live_window_does_not_open_raw_data_when_prewarmed(self) -> None:
        end = datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)
        start = end - app.DEFAULT_WINDOW
        with TemporaryDirectory() as tmpdir, patch.dict(
            "os.environ", {"AURORA_INTERACTIVE_PREWARM_DIR": tmpdir}, clear=False
        ), patch.object(app, "_dataset_time_bounds", side_effect=AssertionError("raw data should not be read")):
            (Path(tmpdir) / "power_current_latest_interactive.json").write_text("{}", encoding="utf-8")
            self.assertTrue(app._is_power_latest_window(start, end, "power"))

    def test_power_section_window_reads_only_the_selected_compact_store(self) -> None:
        times = pd.date_range("2026-07-20T00:00:00", periods=5, freq="1h")
        dataset = xr.Dataset(
            {"BatterySOC": (("time",), np.arange(len(times), dtype=float))},
            coords={"time": times},
        )
        with TemporaryDirectory() as tmpdir:
            current_path = Path(tmpdir) / "power_current_display.zarr"
            dataset.to_zarr(current_path, mode="w", consolidated=True)
            previous = dict(app._POWER_DISPLAY_SECTION_DS)
            previous_times = dict(app._POWER_DISPLAY_SECTION_REFRESHED_AT)
            app._POWER_DISPLAY_SECTION_DS.clear()
            app._POWER_DISPLAY_SECTION_REFRESHED_AT.clear()
            try:
                with patch.dict("os.environ", {"POWER_CURRENT_DISPLAY_ZARR_PATH": str(current_path)}, clear=False):
                    result = app._open_power_display_summary_window(times[1], times[3], section="current")
            finally:
                app._refresh_power_display_energy_dataset()
                app._POWER_DISPLAY_SECTION_DS.update(previous)
                app._POWER_DISPLAY_SECTION_REFRESHED_AT.update(previous_times)

        self.assertIsNotNone(result)
        self.assertEqual(list(pd.DatetimeIndex(result["time"].values)), list(times[1:4]))

    def test_power_background_prepare_uses_selected_section_without_panel_mutation(self) -> None:
        times = pd.date_range("2026-07-20T00:00:00", periods=3, freq="1h")
        dataset = xr.Dataset(
            {"BatterySOC": (("time",), np.array([70.0, 71.0, 72.0]))},
            coords={"time": times},
        )
        figure = app.go.Figure()
        with (
            patch.object(app, "_power_display_summary_path", return_value=Path("/tmp")),
            patch.object(app, "_open_power_display_summary_window", return_value=dataset) as open_display,
            patch.object(app, "build_summary_plotly", return_value=figure) as build_plot,
        ):
            prepared, metrics = app._prepare_stacked_timeseries_figure(
                "power",
                times[0],
                times[-1],
                "downsampled",
                power_section="forecast",
            )

        self.assertIs(prepared, figure)
        self.assertEqual(metrics["status"], "ok")
        self.assertEqual(open_display.call_args.kwargs["section"], "forecast")
        self.assertEqual(
            build_plot.call_args.kwargs["panel_groups"],
            {"forecast_24h", "forecast_96h", "verification"},
        )

    def test_background_preparation_executor_stays_bounded(self) -> None:
        self.assertGreaterEqual(app._BACKGROUND_PREPARATION_EXECUTOR._max_workers, 1)
        self.assertLessEqual(app._BACKGROUND_PREPARATION_EXECUTOR._max_workers, 4)

    def test_power_query_selects_power_before_interactive_callbacks(self) -> None:
        original_instrument = app.instrument_select.value
        original_view = app.power_view_select.value
        try:
            with patch.object(app, "_request_query_args", return_value={"tab": "power", "power_view": "forecast"}):
                app._apply_query_state()
            self.assertEqual(app.instrument_select.value, "power")
            self.assertEqual(app.power_view_select.value, "forecast")
        finally:
            app.instrument_select.value = original_instrument
            app.power_view_select.value = original_view

    def test_slow_interactive_render_emits_a_budget_event(self) -> None:
        events = []
        with (
            patch.object(app, "INTERACTIVE_RENDER_BUDGET_MS", 0),
            patch.object(app, "_perf_log", side_effect=lambda event, **fields: events.append((event, fields))),
        ):
            with app._timed_perf("interactive_view_update", instrument="power") as details:
                details["status"] = "ok"

        self.assertEqual(events[0][0], "interactive_view_update")
        self.assertEqual(events[1][0], "interactive_render_budget_exceeded")
        self.assertEqual(events[1][1]["source_event"], "interactive_view_update")

    def test_forecast_info_control_uses_deployed_panel_widget_api(self) -> None:
        panel = next(
            panel
            for panel in app.SUMMARY_LAYOUTS["power"]
            if panel.key == "soc_ecmwf_forecast"
        )

        control = app._forecast_plot_info_control(panel, xr.Dataset())

        self.assertIsNotNone(control)
        self.assertEqual(control.objects[0].objects[-1].name, "Info")

    def test_desktop_forecast_plot_card_keeps_info_with_its_plot(self) -> None:
        panel = next(
            panel
            for panel in app.SUMMARY_LAYOUTS["power"]
            if panel.key == "soc_ecmwf_forecast"
        )
        times = pd.date_range("2026-07-20T00:00:00", periods=4, freq="1h")
        dataset = xr.Dataset(
            {
                trace.var: (("time",), np.linspace(90.0, 80.0, len(times)))
                for trace in panel.traces
            },
            coords={"time": times},
        )

        card = app._power_plot_card(dataset, panel, mobile=False)

        self.assertIsNotNone(card)
        self.assertIn("desktop-power-plot-card", card.css_classes)
        self.assertIn("forecast-plot-info-control", card.objects[0].css_classes)
        self.assertIsInstance(card.objects[-1], app.pn.pane.Plotly)
        self.assertIn("desktop-power-figure", card.objects[-1].css_classes)

    def test_custom_operating_plan_clips_to_96_hours_and_distinguishes_quantiles(self) -> None:
        anchor = pd.Timestamp("2026-09-06T00:00:00")
        times = pd.date_range(anchor, anchor + pd.Timedelta(hours=240), freq="1h")
        result = {
            "time": times,
            "modes": ["dc_only"] * len(times),
            "kit": "CL61",
            "mode_codes": np.zeros(len(times), dtype=np.int16),
            "load_p50_w": np.full(len(times), 250.0),
            "soc_p10": np.linspace(88.0, 20.0, len(times)),
            "soc_p50": np.linspace(90.0, 24.0, len(times)),
            "soc_p90": np.linspace(92.0, 28.0, len(times)),
            "below_40_probability": np.zeros(len(times)),
            "collection_hours": 12.0,
            "minimum_p10_soc": 20.0,
            "final_p10_soc": 20.0,
            "safe": False,
        }
        scenarios = xr.Dataset(
            attrs={
                "current_mode_label": "DC-Only",
                "current_mode_maturity": "observed",
                "current_mode_confidence": "1",
            }
        )

        with (
            patch.object(app, "_get_power_operating_scenarios_dataset", return_value=scenarios),
            patch.object(app, "evaluate_custom_schedule", return_value=result),
        ):
            view = app._build_custom_cl61_plan_view(anchor, 12, "CL61")

        plot = next(item for item in view.objects if isinstance(item, app.pn.pane.Plotly))
        figure = plot.object
        expected_end = anchor + pd.Timedelta(hours=96)
        for trace in figure.data:
            trace_times = pd.DatetimeIndex(pd.to_datetime(trace.x))
            self.assertGreaterEqual(trace_times.min(), anchor, trace.name)
            self.assertLessEqual(trace_times.max(), expected_end, trace.name)
        traces = {trace.name: trace for trace in figure.data}
        p10 = traces["SOC P10"]
        p50 = traces["SOC Median"]
        p90 = traces["SOC P90"]
        self.assertNotEqual(p10.line.color, p90.line.color)
        self.assertNotEqual(p10.line.dash or "solid", p90.line.dash or "solid")
        self.assertGreater(p50.line.width, max(p10.line.width, p90.line.width))

    def test_missing_power_forecast_warning_has_explicit_visible_contrast(self) -> None:
        with patch.object(app, "_open_power_display_summary_window", return_value=None):
            warning = app._browser_power_briefing(
                "power",
                datetime(2026, 9, 5, tzinfo=timezone.utc),
                datetime(2026, 9, 6, tzinfo=timezone.utc),
            )

        self.assertTrue(warning.visible)
        self.assertEqual(
            warning.object,
            "Power forecast data is not available.",
        )
        self.assertIn("power-forecast-unavailable", warning.css_classes)
        stylesheet = Path(app.__file__).with_name("assets") / "dashboard.css"
        css = stylesheet.read_text(encoding="utf-8")
        rule = css.split(".power-forecast-unavailable {", 1)[1].split("}", 1)[0]
        self.assertIn("color:", rule)
        self.assertIn("background:", rule)
        self.assertIn("border:", rule)
        self.assertNotIn("display: none", rule)
        self.assertNotIn("visibility: hidden", rule)
        self.assertNotIn("opacity: 0", rule)

    def test_archived_decision_uses_96_hour_cl61_count_in_summary(self) -> None:
        record = {
            "decision_horizon_hours": 96,
            # Legacy archives can contain a full safety-tail count here.
            "collection_hours": 55.0,
            "instrument_hours": {"CL61": 46.0, "Radar": 49.0, "HATPRO": 55.0},
            "minimum_p10_soc": 40.0,
            "issued_at_utc": "2026-09-05T12:00:00Z",
            "recommended_mode_windows": [],
            "verification": {"status": "awaiting_measurements"},
        }
        with TemporaryDirectory() as tmpdir:
            archive = Path(tmpdir) / "recommendations.json"
            archive.write_text(
                json.dumps({"recommendations": [record]}),
                encoding="utf-8",
            )
            with patch.object(
                app,
                "_power_operating_recommendations_path",
                return_value=archive,
            ):
                view = app._operating_decision_audit_view()

        markup = view.objects[0].object
        label = (
            "<div class='operating-plan-metric__label'>"
            "Planned CL61 collection (decision window)</div>"
        )
        value = "<div class='operating-plan-metric__value'>46 h</div>"
        self.assertIn(label + value, markup)
        self.assertNotIn(label + "<div class='operating-plan-metric__value'>55 h</div>", markup)

    def test_desktop_shell_has_full_named_tabs(self) -> None:
        labels = [label for label, _slug, _panel in app.DESKTOP_TAB_SPECS]

        self.assertEqual(
            labels,
            [
                "Overview",
                "Interactive Data Browser",
                "Power",
                "Science Quicklooks",
                "House Keeping Quicklooks",
                "AURORACam",
                "UAS",
                "Operations Dashboard",
            ],
        )
        self.assertEqual(len(app.desktop_tabs), len(labels))
        # Content stays lazy, but stable tab hosts remain attached so Bokeh
        # does not drop model patches while a tab is activated.
        self.assertFalse(app.desktop_tabs.dynamic)
        self.assertIn("Overview", labels)

    def test_uas_science_quicklook_is_not_a_generic_browser_choice(self) -> None:
        self.assertEqual(app.science_instrument.options["UAS"], "uas")
        self.assertNotIn("UAS", app.instrument_select.options)
        self.assertNotIn("UAS", app.hk_instrument.options)
        with TemporaryDirectory() as tmpdir, patch.object(app, "UAS_QUICKLOOK_DIR", Path(tmpdir)):
            (Path(tmpdir) / "uas__summary__20260827.png").write_bytes(b"dated")
            (Path(tmpdir) / "uas__summary__latest.png").write_bytes(b"latest")
            options = app._quicklook_options("uas", mode="science")

        self.assertEqual(list(options), ["20260827", "Latest available"])

    def test_uas_profile_figure_has_four_linked_panels_and_sensor_traces(self) -> None:
        detail = {
            "flight": {
                "id": "flight-1",
                "title": "Menapia Flight 1",
                "startTimeUTC": "2026-08-28T10:00:00Z",
                "durationSeconds": 2,
            },
            "series": {
                "timeUTC": [
                    "2026-08-28T10:00:00Z",
                    "2026-08-28T10:00:01Z",
                    "2026-08-28T10:00:02Z",
                ],
                "temperatureC": {"SN0122": [8.0, 8.1, 8.2], "SN0123": [8.2, 8.3, 8.4]},
                "pressureHpa": {"SN0122": [1005.0, 1005.1, 1005.2], "SN0123": [1006.0, 1006.1, 1006.2]},
                "relativeHumidityPct": {"SN0122": [75.0, 76.0, 77.0], "SN0123": [74.0, 75.0, 76.0]},
                "altitudeM": [-60.0, 48.0, -60.0],
            },
        }

        figure = app._uas_flight_figure(detail)

        self.assertEqual(len(figure.data), 7)
        self.assertEqual([trace.name for trace in figure.data[:2]], ["SN0122", "SN0123"])
        self.assertEqual(figure.data[1].line.dash, "dash")
        self.assertEqual(figure.data[2].mode, "lines+markers")
        self.assertFalse(figure.data[2].connectgaps)
        self.assertEqual(figure.data[2].marker.size, 4)
        self.assertEqual(figure.layout.yaxis.title.text, "Temperature (°C)")
        self.assertEqual(figure.layout.yaxis2.title.text, "Pressure (hPa)")
        self.assertEqual(figure.layout.yaxis3.title.text, "Relative humidity (%)")
        self.assertEqual(figure.layout.yaxis4.title.text, "Fused altitude (m)")
        self.assertEqual(figure.layout.xaxis4.title.text, "Time (UTC)")
        self.assertEqual(figure.layout.hovermode, "x unified")

    def test_uas_latest_defaults_newest_but_preserves_manual_and_historical_selection(self) -> None:
        latest_listing = {
            "generatedAt": "2026-08-28T12:31:00Z",
            "lastRunAt": "2026-08-28T14:00:00Z",
            "latestFlightID": "flight-new",
            "availableDays": ["2026-08-28", "2026-08-27"],
            "selectedDay": "2026-08-28",
            "status": {"state": "fresh", "level": "green", "title": "Current", "detail": "Current"},
            "flights": [
                {
                    "id": "flight-new",
                    "dayUTC": "2026-08-28",
                    "title": "Flight new",
                    "startTimeUTC": "2026-08-28T12:00:00Z",
                    "durationSeconds": 60,
                    "quality": {"level": "green", "warnings": []},
                },
                {
                    "id": "flight-before",
                    "dayUTC": "2026-08-28",
                    "title": "Flight before",
                    "startTimeUTC": "2026-08-28T11:00:00Z",
                    "durationSeconds": 60,
                    "quality": {"level": "green", "warnings": []},
                },
            ],
        }
        historical_listing = {
            **latest_listing,
            "selectedDay": "2026-08-27",
            "flights": [
                {
                    "id": "flight-old",
                    "dayUTC": "2026-08-27",
                    "title": "Flight old",
                    "startTimeUTC": "2026-08-27T10:00:00Z",
                    "durationSeconds": 60,
                    "quality": {"level": "green", "warnings": []},
                }
            ],
        }
        status_markup = app._uas_flight_status_markup(latest_listing)
        self.assertIn("Last product build", status_markup)
        self.assertIn("2026-08-28 14:00 UTC", status_markup)
        self.assertIn("Product content", status_markup)
        self.assertIn("2026-08-28 12:31 UTC", status_markup)
        original_day_options = app.uas_flight_day.options
        original_day = app.uas_flight_day.value
        original_flight_options = app.uas_flight_select.options
        original_flight = app.uas_flight_select.value
        original_manual_selection = app._uas_flight_manual_selection
        try:
            app._uas_flight_widget_guard = True
            app.uas_flight_day.options = {"Latest available": "latest", "27 August": "2026-08-27"}
            app.uas_flight_day.value = "latest"
            app.uas_flight_select.options = {"Before": "flight-before"}
            app.uas_flight_select.value = "flight-before"
            app._uas_flight_manual_selection = False
            app._uas_flight_widget_guard = False
            with (
                patch.object(
                    app.mobile_catalog,
                    "uas_flights",
                    side_effect=lambda day="latest": latest_listing if day == "latest" else historical_listing,
                ),
                patch.object(app, "_refresh_uas_selected_flight"),
            ):
                app._refresh_uas_flight_products(preserve_selection=True)
                self.assertEqual(app.uas_flight_select.value, "flight-new")

                app._uas_flight_widget_guard = True
                app.uas_flight_select.value = "flight-before"
                app._uas_flight_widget_guard = False
                app._on_uas_flight_selection_change(SimpleNamespace(old="flight-new", new="flight-before"))
                self.assertTrue(app._uas_flight_manual_selection)
                app._refresh_uas_flight_products(preserve_selection=True)
                self.assertEqual(app.uas_flight_select.value, "flight-before")

                app._uas_flight_widget_guard = True
                app.uas_flight_day.value = "2026-08-27"
                app.uas_flight_select.options = {"Old": "flight-old"}
                app.uas_flight_select.value = "flight-old"
                app._uas_flight_widget_guard = False
                app._refresh_uas_flight_products(preserve_selection=True)
                self.assertEqual(app.uas_flight_select.value, "flight-old")
        finally:
            app._uas_flight_widget_guard = True
            app.uas_flight_day.options = original_day_options
            app.uas_flight_day.value = original_day
            app.uas_flight_select.options = original_flight_options
            app.uas_flight_select.value = original_flight
            app._uas_flight_manual_selection = original_manual_selection
            app._uas_flight_widget_guard = False

    def test_science_wxcam_to_uas_hides_and_resets_image_type_control(self) -> None:
        original_guard = app._instrument_guard
        original_visible = app.science_image_type.visible
        original_options = app.science_image_type.options
        original_value = app.science_image_type.value
        try:
            app._instrument_guard = True
            app.science_image_type.options = ["FISH HDR"]
            app.science_image_type.value = "FISH HDR"
            app.science_image_type.visible = True
            app._instrument_guard = False
            with (
                patch.object(app, "_refresh_ql_options"),
                patch.object(app.ql_date.param, "trigger"),
            ):
                app._on_science_instrument_change(SimpleNamespace(old="wxcam", new="uas"))

            self.assertFalse(app.science_image_type.visible)
            self.assertEqual(list(app.science_image_type.options), [])
            self.assertIsNone(app.science_image_type.value)
        finally:
            app._instrument_guard = True
            app.science_image_type.options = original_options
            app.science_image_type.value = original_value
            app.science_image_type.visible = original_visible
            app._instrument_guard = original_guard

    def test_uas_share_state_includes_and_restores_day_and_flight(self) -> None:
        original_day = app._UAS_PENDING_DAY_QUERY
        original_flight = app._UAS_PENDING_FLIGHT_QUERY
        try:
            app._UAS_PENDING_DAY_QUERY = None
            app._UAS_PENDING_FLIGHT_QUERY = None
            with patch.object(
                app,
                "_request_query_args",
                return_value={
                    "tab": "uas",
                    "uas_day": "2026-08-27",
                    "uas_flight": "flight-old",
                },
            ):
                app._apply_query_state()
            params = app._view_query_params("uas")
        finally:
            app._UAS_PENDING_DAY_QUERY = original_day
            app._UAS_PENDING_FLIGHT_QUERY = original_flight

        self.assertEqual(params["uas_day"], "2026-08-27")
        self.assertEqual(params["uas_flight"], "flight-old")

    def test_desktop_interactive_and_power_tabs_use_distinct_hosts(self) -> None:
        self.assertIsNot(app.TAB_PANEL_BY_SLUG["interactive"], app.TAB_PANEL_BY_SLUG["power"])

        app._sync_browser_tab_instrument("power")
        self.assertEqual(app.interactive_tab_host.objects, [])
        self.assertEqual(app.power_tab_host.objects, [app.interactive_tab])

        app._sync_browser_tab_instrument("interactive")
        self.assertEqual(app.interactive_tab_host.objects, [app.interactive_tab])
        self.assertEqual(app.power_tab_host.objects, [])

    def test_desktop_tab_labels_scroll_without_abbreviating(self) -> None:
        stylesheet = Path(app.__file__).with_name("assets") / "dashboard.css"
        css = stylesheet.read_text(encoding="utf-8")
        self.assertIn(":host(.desktop-tabs) .bk-header", css)
        self.assertIn("overflow-x: auto", css)
        self.assertIn(":host(.desktop-tabs) .bk-tab", css)
        self.assertIn("white-space: nowrap", css)

    def test_dashboard_stylesheet_is_a_single_cacheable_static_asset(self) -> None:
        stylesheet = Path(app.__file__).with_name("assets") / "dashboard.css"

        self.assertTrue(stylesheet.is_file())
        self.assertRegex(
            app.DASHBOARD_STYLESHEET,
            r"^/dashboard-assets/dashboard\.css\?v=[0-9a-f]{12}$",
        )

    def test_quicklook_images_override_panel_hidden_wrapper(self) -> None:
        stylesheet = Path(app.__file__).with_name("assets") / "dashboard.css"
        css = stylesheet.read_text(encoding="utf-8")
        quicklook_rule = css.split(".quicklook-image__img {", 1)[1].split("}", 1)[0]

        self.assertIn("visibility: visible !important;", quicklook_rule)

    def test_desktop_controls_keep_compact_navigation_rows(self) -> None:
        controls_body = app.controls.objects[0]
        first_row_names = [widget.name for widget in controls_body.objects[0].objects]
        second_row_names = [widget.name for widget in controls_body.objects[1].objects]

        self.assertEqual(first_row_names, ["Instrument", "Start (UTC)", "End (UTC)", "Live Off"])
        self.assertEqual(second_row_names, ["Previous Day", "Reset View Defaults", "Next Day/Current Day"])

    def test_phone_shell_keeps_operational_groups(self) -> None:
        self.assertEqual(list(app.MOBILE_TAB_OPTIONS), ["Overview", "Power", "Plots", "Camera", "Ops"])

    def test_browser_overview_uses_one_icon_led_instrument_status_list(self) -> None:
        overview = {
            "instrumentPower": [
                {"id": "vaisalamet", "title": "Meteorology", "systemImage": "cloud.sun", "state": "Collecting", "level": "green", "detail": "Latest sample 1 min old"},
                {"id": "asfs-logger", "title": "Radiation", "systemImage": "sun.max", "state": "Collecting", "level": "green", "detail": "Latest sample 1 min old"},
                {"id": "uas", "title": "UAS", "systemImage": "airplane", "state": "On", "level": "green", "detail": "PDU sample 2 min old"},
                {"id": "ceilometer", "title": "CL61", "systemImage": "laser.burst", "state": "Off", "level": "unknown", "detail": "PDU sample 2 min old"},
                {"id": "cloud-radar", "title": "Cloud Radar", "systemImage": "dot.radiowaves.left.and.right", "state": "On", "level": "green", "detail": "PDU sample 2 min old"},
                {"id": "hatpro", "title": "HATPRO", "systemImage": "antenna.radiowaves.left.and.right", "state": "Off", "level": "unknown", "detail": "PDU sample 2 min old"},
            ]
        }
        with patch.object(app.mobile_catalog, "overview", return_value=overview):
            markup = app._browser_overview_instrument_markup()

        self.assertIn("Instrument status", markup)
        self.assertNotIn("PDU-controlled instruments", markup)
        self.assertNotIn("Collection-only instruments", markup)
        self.assertEqual(markup.count("data-instrument-id="), 6)
        for instrument_id, system_image in (
            ("vaisalamet", "cloud.sun"),
            ("asfs-logger", "sun.max"),
            ("uas", "airplane"),
            ("ceilometer", "laser.burst"),
            ("cloud-radar", "dot.radiowaves.left.and.right"),
            ("hatpro", "antenna.radiowaves.left.and.right"),
        ):
            self.assertIn(f"data-instrument-id='{instrument_id}'", markup)
            self.assertIn(f"data-instrument-icon='{system_image}'", markup)

    def test_overview_refreshes_when_selected(self) -> None:
        with patch.object(app, "_refresh_browser_overview") as refresh:
            app._ensure_active_tab_loaded("overview")

        refresh.assert_called_once_with()

    def test_overview_refresh_is_idempotent_after_initial_render(self) -> None:
        original_loaded = app._BROWSER_OVERVIEW_LOADED
        try:
            app._BROWSER_OVERVIEW_LOADED = True
            with (
                patch.object(app, "_mobile_overview_markup", side_effect=AssertionError("overview rebuilt")),
                patch.object(app, "_browser_overview_instrument_markup", side_effect=AssertionError("overview rebuilt")),
            ):
                app._refresh_browser_overview()
        finally:
            app._BROWSER_OVERVIEW_LOADED = original_loaded

    def test_overview_refresh_updates_stable_attached_panes(self) -> None:
        original_loaded = app._BROWSER_OVERVIEW_LOADED
        original_summary = app.browser_overview_summary_pane.object
        original_instruments = app.browser_overview_instrument_pane.object
        original_objects = tuple(app.browser_overview_container.objects)
        try:
            app._BROWSER_OVERVIEW_LOADED = False
            with (
                patch.object(app, "_mobile_overview_markup", return_value="<div>AURORA Overview fixture</div>"),
                patch.object(app, "_browser_overview_instrument_markup", return_value="<div>Instrument fixture</div>"),
            ):
                app._refresh_browser_overview()

            self.assertEqual(tuple(app.browser_overview_container.objects), original_objects)
            self.assertEqual(app.browser_overview_summary_pane.object, "<div>AURORA Overview fixture</div>")
            self.assertEqual(app.browser_overview_instrument_pane.object, "<div>Instrument fixture</div>")
        finally:
            app.browser_overview_summary_pane.object = original_summary
            app.browser_overview_instrument_pane.object = original_instruments
            app._BROWSER_OVERVIEW_LOADED = original_loaded

    def test_housekeeping_refresh_updates_stable_attached_panes(self) -> None:
        original_quicklook = app.housekeeping_quicklook_pane.object
        original_status = app.hk_status_pane.object
        original_availability = app.hk_availability_pane.object
        original_objects = tuple(app.housekeeping_quicklook_container.objects)
        try:
            with (
                patch.object(app, "_housekeeping_quicklook_image", return_value=app.pn.pane.HTML("<img class='quicklook-image__img'>")),
                patch.object(app, "_current_hk_status_markup", return_value="<div>HK status fixture</div>"),
                patch.object(app, "_current_hk_availability_markup", return_value="<div>HK availability fixture</div>"),
            ):
                app._refresh_housekeeping_tab_content()

            self.assertEqual(tuple(app.housekeeping_quicklook_container.objects), original_objects)
            self.assertIn("quicklook-image__img", app.housekeeping_quicklook_pane.object)
            self.assertEqual(app.hk_status_pane.object, "<div>HK status fixture</div>")
            self.assertEqual(app.hk_availability_pane.object, "<div>HK availability fixture</div>")
        finally:
            app.housekeeping_quicklook_pane.object = original_quicklook
            app.hk_status_pane.object = original_status
            app.hk_availability_pane.object = original_availability

    def test_tab_is_selected_before_its_load_is_scheduled(self) -> None:
        original_slug = app.ACTIVE_TAB_SLUG
        original_active = app.desktop_tabs.active
        try:
            def assert_selected(slug: str) -> None:
                self.assertEqual(slug, "overview")
                self.assertEqual(app.desktop_tabs.active, app.TAB_INDEX_BY_SLUG[slug])

            with (
                patch.object(app, "_schedule_active_tab_load", side_effect=assert_selected) as schedule,
                patch.object(app, "_refresh_share_and_download_state"),
            ):
                app._set_active_tab("overview")

            schedule.assert_called_once_with("overview")
        finally:
            app.ACTIVE_TAB_SLUG = original_slug
            app._desktop_tab_syncing = True
            try:
                app.desktop_tabs.active = original_active
            finally:
                app._desktop_tab_syncing = False

    def test_only_stable_direct_link_tabs_are_prepared_during_bootstrap(self) -> None:
        original_loaded_tabs = set(app._LOADED_TABS)
        try:
            app._LOADED_TABS.discard("housekeeping")
            with (
                patch.object(app, "_refresh_browser_overview") as overview,
                patch.object(app, "_refresh_housekeeping_tab_content") as housekeeping,
            ):
                app._prepare_initial_stable_tab_content("interactive")
                overview.assert_not_called()
                housekeeping.assert_not_called()

                app._prepare_initial_stable_tab_content("overview")
                overview.assert_called_once_with()

                app._prepare_initial_stable_tab_content("housekeeping")
                housekeeping.assert_called_once_with()
                self.assertIn("housekeeping", app._LOADED_TABS)
        finally:
            app._LOADED_TABS.clear()
            app._LOADED_TABS.update(original_loaded_tabs)

    def test_overview_uses_cached_power_timestamp_without_opening_display_zarr(self) -> None:
        snapshot = {
            "time_utc": "2026-07-29T12:00:00Z",
            "aps_battery_power_time_utc": "2026-07-29T11:58:00Z",
        }
        with (
            patch.object(app, "_ops_read_snapshot", return_value=snapshot),
            patch.object(app, "_mobile_power_latest_measured_time", side_effect=AssertionError("display Zarr opened")),
            patch.object(app, "_mobile_auroracam_freshness", return_value=("Radar", "2m ago", "green")),
            patch.object(app.mobile_catalog, "environmental_signal_cards", return_value=[]),
        ):
            markup = app._mobile_overview_markup()

        self.assertIn("11:58 UTC", markup)

    def test_overview_selection_does_not_enable_interactive_rendering(self) -> None:
        original_bootstrapping = app._APP_BOOTSTRAPPING
        try:
            app._APP_BOOTSTRAPPING = False
            with (
                patch.object(app, "_refresh_browser_overview"),
                patch.object(app, "_enable_browser_interactive_render") as enable,
            ):
                app._ensure_active_tab_loaded("overview")
        finally:
            app._APP_BOOTSTRAPPING = original_bootstrapping

        enable.assert_not_called()

    def test_interactive_selection_enables_renderer_on_demand(self) -> None:
        original_bootstrapping = app._APP_BOOTSTRAPPING
        original_enabled = app._INTERACTIVE_RENDER_ENABLED
        try:
            app._APP_BOOTSTRAPPING = False
            app._INTERACTIVE_RENDER_ENABLED = False
            with (
                patch.object(app, "_sync_browser_tab_instrument"),
                patch.object(app, "_enable_browser_interactive_render") as enable,
            ):
                app._ensure_active_tab_loaded("interactive")
        finally:
            app._APP_BOOTSTRAPPING = original_bootstrapping
            app._INTERACTIVE_RENDER_ENABLED = original_enabled

        enable.assert_called_once_with()

    def test_empty_pdu_instrument_view_explains_intentional_power_off(self) -> None:
        with patch.object(
            app.mobile_catalog,
            "pdu_instrument_status",
            return_value={"state": "Off", "detail": "PDU sample 2 min old"},
        ):
            figure = app._empty_interactive_figure("Ceilometer", "No samples", start=datetime(2026, 7, 19), end=datetime(2026, 7, 20))

        annotation = figure.layout.annotations[0]
        self.assertIn("INTENTIONAL POWER-OFF", annotation.text)
        self.assertIn("Data collection is paused", annotation.text)
        self.assertEqual(annotation.bgcolor, "#edf8f6")

    def test_operations_marks_stale_pdu_off_streams_as_paused(self) -> None:
        snapshot = {
            "time_utc": "2026-07-19T12:00:00Z",
            "cl61_source_recent_state": 0,
            "cl61_source_age_min": 447,
            "radar_source_recent_state": 0,
            "radar_source_age_min": 451,
            "hatpro_source_recent_state": 0,
            "hatpro_source_age_min": 451,
            "vaisalamet_source_recent_state": 1,
            "asfs_logger_source_recent_state": 1,
            "asfs_fast_sonic_source_recent_state": 1,
            "power_source_recent_state": 1,
            "wxcam_source_recent_state": 1,
            "source_host_probe_fail_count": 0,
        }
        with patch.object(app.mobile_catalog, "pdu_outlet_states", return_value={5: False, 6: True, 8: False}):
            paused = app._ops_expected_paused_prefixes()
            recent, stale, paused_count = app._ops_source_health(snapshot, paused)

        self.assertEqual(paused, {"cl61", "hatpro"})
        self.assertEqual((recent, stale, paused_count), (5, 1, 2))
        self.assertIn("Paused - PDU outlet off", app._ops_source_freshness_text(snapshot, "cl61", intentionally_paused=True))

    def test_combined_operations_series_allows_all_missing_columns(self) -> None:
        dataset = xr.Dataset({"source_age": (("time",), np.array([np.nan, np.nan]))})

        combined = app._ops_combined_series(dataset, ("source_age",))

        self.assertTrue(np.isnan(combined).all())

    def test_live_query_uses_current_window_instead_of_stale_url_dates(self) -> None:
        current_start = datetime(2026, 7, 15, 10, 30)
        current_end = datetime(2026, 7, 16, 10, 30)
        with patch.object(app, "_last_24h_utc_window", return_value=(current_start, current_end)):
            state = app._query_interactive_time_state(
                {"start": "2026-07-15T07:04:01", "end": "2026-07-16T07:04:01", "live": "1"},
                "power",
            )

        self.assertEqual(state, (current_start, current_end, True))

    def test_non_live_query_preserves_historical_window(self) -> None:
        state = app._query_interactive_time_state(
            {"start": "2026-07-15T07:04:01", "end": "2026-07-16T07:04:01", "live": "0"},
            "power",
        )

        self.assertEqual(
            state,
            (datetime(2026, 7, 15, 7, 4, 1), datetime(2026, 7, 16, 7, 4, 1), False),
        )

    def test_power_time_bounds_ignore_forecast_only_rows(self) -> None:
        times = pd.date_range("2026-07-16T08:00:00", periods=7, freq="1h")
        measured = np.array([50.0, 51.0, 52.0, np.nan, np.nan, np.nan, np.nan])
        forecast = np.array([np.nan, np.nan, 52.0, 53.0, 54.0, 55.0, 56.0])
        dataset = xr.Dataset(
            {"BatterySOC": (("time",), measured), "BatterySOCForecast": (("time",), forecast)},
            coords={"time": times},
        )

        lower, upper, raw_count, valid_count = app._time_bounds_from_power_display_dataset(dataset)

        self.assertEqual(lower, datetime(2026, 7, 16, 8, 0))
        self.assertEqual(upper, datetime(2026, 7, 16, 10, 0))
        self.assertEqual(raw_count, 7)
        self.assertEqual(valid_count, 3)

    def test_operating_scenario_cache_reopens_an_incomplete_mirror(self) -> None:
        incomplete = xr.Dataset(coords={"time": pd.date_range("2026-07-20", periods=2, freq="1h")})
        complete = xr.Dataset(
            {
                "SolarEnsembleWatts": (("member", "time"), np.ones((1, 2))),
                "ComponentLoadWatts": (("member", "component"), np.ones((1, 1))),
            },
            coords={
                "time": pd.date_range("2026-07-20", periods=2, freq="1h"),
                "member": [0],
                "component": ["DC"],
            },
        )
        with (
            patch.object(app, "_POWER_OPERATING_SCENARIOS_DS", incomplete),
            patch.object(app, "_power_operating_scenario_paths") as paths,
            patch.object(app.xr, "open_zarr", side_effect=(incomplete, complete)) as open_zarr,
        ):
            paths.return_value = (app.Path(__file__), app.Path(app.__file__))
            result = app._get_power_operating_scenarios_dataset()

        self.assertIs(result, complete)
        self.assertEqual(open_zarr.call_count, 2)
