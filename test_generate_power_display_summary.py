from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import xarray as xr
from dask import array as da

from generate_power_display_summary import (
    _open_optional_zarr,
    _recent_observation_views,
    _release_generation_lock,
    _section_subset,
    _try_generation_lock,
    _validated_history_days,
    _write_metadata,
    _write_zarr_atomic,
)
from grouped_timeseries import (
    POWER_DISPLAY_ENERGY_MAP,
    POWER_PANEL_TIME_GROUP_BY_KEY,
    SUMMARY_LAYOUTS,
    build_power_display_summary_dataset,
)


class PowerDisplaySummaryMetadataTests(unittest.TestCase):
    def test_metadata_records_time_bounds_and_dimensions(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "power_display_summary.zarr"
            display = xr.Dataset(
                {"power": ("time", np.asarray([1.0, 2.0]))},
                coords={"time": np.asarray(["2026-07-17T00:00", "2026-07-17T00:01"], dtype="datetime64[m]")},
            )
            path = _write_metadata(output, display)
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["time_count"], 2)
        self.assertEqual(payload["variable_count"], 1)
        self.assertEqual(str(np.datetime64(payload["time_start_utc"])), "2026-07-17T00:00:00")
        self.assertEqual(str(np.datetime64(payload["time_end_utc"])), "2026-07-17T00:01:00")

    def test_section_products_contain_only_their_panel_variables(self) -> None:
        times = np.asarray(["2026-07-17T00:00", "2026-07-17T00:01"], dtype="datetime64[m]")
        fields = {
            trace.var
            for panel in SUMMARY_LAYOUTS["power"]
            for trace in panel.traces
        }
        display = xr.Dataset(
            {name: ("time", np.asarray([1.0, 2.0])) for name in fields},
            coords={"time": times},
        )

        current = _section_subset(display, "current")
        forecast = _section_subset(display, "forecast")
        current_fields = {
            trace.var
            for panel in SUMMARY_LAYOUTS["power"]
            if POWER_PANEL_TIME_GROUP_BY_KEY.get(panel.key, "observed") == "observed"
            for trace in panel.traces
        }
        forecast_fields = fields - current_fields

        self.assertEqual(set(current.data_vars), current_fields)
        self.assertEqual(set(forecast.data_vars), forecast_fields)

    def test_generation_lock_skips_overlapping_build(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "power_display_summary.zarr"
            first = _try_generation_lock(output)
            self.assertIsNotNone(first)
            try:
                self.assertIsNone(_try_generation_lock(output))
            finally:
                _release_generation_lock(first)

            second = _try_generation_lock(output)
            self.assertIsNotNone(second)
            _release_generation_lock(second)

    def test_atomic_store_has_consolidated_metadata(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "display.zarr"
            display = xr.Dataset(
                {"power": ("time", np.asarray([1.0, 2.0]))},
                coords={"time": np.asarray(["2026-07-17T00:00", "2026-07-17T00:01"], dtype="datetime64[m]")},
            )
            _write_zarr_atomic(display, output, chunk_time=1)
            opened = xr.open_zarr(output, consolidated=True)
            try:
                self.assertEqual(opened.sizes["time"], 2)
            finally:
                opened.close()

    def test_eager_optional_store_is_stable_after_source_replacement(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "forecast.zarr"
            first = xr.Dataset(
                {"soc": (("scenario", "time"), np.asarray([[90.0, 91.0]]))},
                coords={"scenario": ["current"], "time": [0, 1]},
            )
            second = xr.Dataset(
                {"soc": (("scenario", "time"), np.asarray([[80.0, 81.0, 82.0]]))},
                coords={"scenario": ["current"], "time": [0, 1, 2]},
            )
            first.to_zarr(output, mode="w", consolidated=True)
            snapshot = _open_optional_zarr(output, "forecast", eager=True)
            self.assertIsNotNone(snapshot)

            _write_zarr_atomic(second, output, chunk_time=3)

            np.testing.assert_array_equal(snapshot["soc"].values, [[90.0, 91.0]])
            self.assertEqual(snapshot.sizes["time"], 2)

    def test_recent_observation_view_keeps_arrays_lazy_and_prunes_fields(self) -> None:
        times = pd.date_range("2026-07-01T00:00:00", periods=12 * 24 + 1, freq="1h")
        source_values = da.arange(len(times), chunks=24)
        power = xr.Dataset(
            {
                "BatterySOC": (("time",), source_values),
                "unused": (("time",), source_values + 1),
            },
            coords={"time": times},
        )

        recent, _ass, _pdu, start, end = _recent_observation_views(
            power,
            None,
            None,
            history_days=8,
        )

        self.assertEqual(start, pd.Timestamp("2026-07-05T00:00:00"))
        self.assertEqual(end, pd.Timestamp("2026-07-13T00:00:00"))
        self.assertNotIn("unused", recent)
        self.assertIsInstance(recent["BatterySOC"].data, da.Array)
        self.assertLess(recent.sizes["time"], power.sizes["time"])

    def test_recent_observation_summary_matches_full_history_inside_retained_window(self) -> None:
        times = pd.date_range("2026-07-01T00:00:00", periods=12 * 24 * 6, freq="10min")
        within_day = (times.hour * 6 + times.minute // 10).to_numpy(dtype=np.float64)
        power = xr.Dataset(
            {
                "BatterySOC": (("time",), 85.0 - np.arange(len(times), dtype=np.float64) / 10000.0),
                "ACOutputWatts": (("time",), 100.0 + np.sin(np.arange(len(times)) / 17.0)),
                "DCInverterWatts": (("time",), np.full(len(times), 25.0)),
                "SolarYield_East": (("time",), within_day * 0.01),
                "SolarYield_South": (("time",), within_day * 0.02),
                "SolarYield_West": (("time",), within_day * 0.03),
            },
            coords={"time": times},
        )
        ass = xr.Dataset(
            {"watts_on_48vdc_Avg": (("time",), np.full(len(times), 42.0))},
            coords={"time": times},
        )
        pdu = xr.Dataset(
            {"PDUOutlet5Watts": (("time",), np.full(len(times), 31.0))},
            coords={"time": times},
        )
        full = build_power_display_summary_dataset(power, ass, pdu, freq="1h")

        recent_power, recent_ass, recent_pdu, start, end = _recent_observation_views(
            power.chunk({"time": 144}),
            ass.chunk({"time": 144}),
            pdu.chunk({"time": 144}),
            history_days=8,
        )
        recent = build_power_display_summary_dataset(recent_power, recent_ass, recent_pdu, freq="1h")
        expected = full.sel(time=slice(start, end))

        xr.testing.assert_allclose(recent, expected)
        for name in POWER_DISPLAY_ENERGY_MAP.values():
            if name in expected:
                np.testing.assert_allclose(recent[name], expected[name], equal_nan=True)

    def test_history_shorter_than_display_contract_is_rejected(self) -> None:
        for value in (7, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "must retain at least"):
                _validated_history_days(value)


if __name__ == "__main__":
    unittest.main()
