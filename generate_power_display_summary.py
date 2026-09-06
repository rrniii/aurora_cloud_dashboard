#!/usr/bin/env python3
"""Generate compact Aurora Power Supply display-summary Zarr products."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import xarray as xr
import zarr

from grouped_timeseries import (
    PDU_DISPLAY_SUMMARY_FIELDS,
    POWER_CUMULATIVE_CONTEXT_DAYS,
    POWER_DISPLAY_ENERGY_ATTR,
    POWER_DISPLAY_ENERGY_MAP,
    POWER_DISPLAY_SUMMARY_ATTR,
    POWER_DISPLAY_SUMMARY_CONTEXT_FIELDS,
    POWER_DISPLAY_SUMMARY_FIELDS,
    POWER_DISPLAY_SUMMARY_FREQ,
    POWER_PANEL_TIME_GROUP_BY_KEY,
    SUMMARY_LAYOUTS,
    build_power_display_summary_dataset,
)

POWER_DISPLAY_MIN_HISTORY_DAYS = max(4.0, float(POWER_CUMULATIVE_CONTEXT_DAYS) + 1.0)
POWER_DISPLAY_HISTORY_DAYS = float(
    os.environ.get("AURORA_POWER_DISPLAY_HISTORY_DAYS", str(max(14.0, POWER_DISPLAY_MIN_HISTORY_DAYS)))
)
POWER_DISPLAY_SOURCE_FIELDS = tuple(
    dict.fromkeys(POWER_DISPLAY_SUMMARY_FIELDS + tuple(POWER_DISPLAY_ENERGY_MAP))
)

POWER_ZARR_PATH = Path(os.environ.get("POWER_ZARR_PATH", "/data/aurora/products/power/power.zarr"))
ASFS_LOGGER_ZARR_PATH = Path(os.environ.get("ASFS_LOGGER_ZARR_PATH", "/data/aurora/products/asfs_logger/asfs_logger.zarr"))
PDU_ZARR_PATH = Path(os.environ.get("PDU_ZARR_PATH", "/data/aurora/products/power/pdu.zarr"))
POWER_SOC_FORECAST_ZARR_PATH = Path(
    os.environ.get("POWER_SOC_FORECAST_ZARR_PATH", "/data/aurora/products/power/power_soc_forecast.zarr")
)
POWER_SOC_FORECAST_SKILL_ZARR_PATH = Path(
    os.environ.get("POWER_SOC_FORECAST_SKILL_ZARR_PATH", "/data/aurora/products/power/power_soc_forecast_skill.zarr")
)
POWER_SOC_HINDCAST_ZARR_PATH = Path(
    os.environ.get("POWER_SOC_HINDCAST_ZARR_PATH", "/data/aurora/products/power/power_soc_hindcast.zarr")
)
POWER_SOC_ENSEMBLE_ZARR_PATH = Path(
    os.environ.get("POWER_SOC_ENSEMBLE_ZARR_PATH", "/data/aurora/products/power/power_soc_ensemble_forecast.zarr")
)
POWER_SOC_ENSEMBLE_SKILL_ZARR_PATH = Path(
    os.environ.get("POWER_SOC_ENSEMBLE_SKILL_ZARR_PATH", "/data/aurora/products/power/power_soc_ensemble_skill.zarr")
)
POWER_OPERATING_SCENARIOS_ZARR_PATH = Path(
    os.environ.get(
        "POWER_OPERATING_SCENARIOS_ZARR_PATH",
        "/data/aurora/products/power/power_operating_scenarios.zarr",
    )
)
POWER_DISPLAY_SUMMARY_ZARR_PATH = Path(
    os.environ.get("POWER_DISPLAY_SUMMARY_ZARR_PATH", "/data/aurora/products/power/power_display_summary.zarr")
)
POWER_DISPLAY_ENERGY_ZARR_PATH = Path(
    os.environ.get("POWER_DISPLAY_ENERGY_ZARR_PATH", "/data/aurora/products/power/power_display_energy.zarr")
)
POWER_CURRENT_DISPLAY_ZARR_PATH = Path(
    os.environ.get("POWER_CURRENT_DISPLAY_ZARR_PATH", "/data/aurora/products/power/power_current_display.zarr")
)
POWER_FORECAST_DISPLAY_ZARR_PATH = Path(
    os.environ.get("POWER_FORECAST_DISPLAY_ZARR_PATH", "/data/aurora/products/power/power_forecast_display.zarr")
)
POWER_DISPLAY_MANIFEST_PATH = Path(
    os.environ.get("POWER_DISPLAY_MANIFEST_PATH", "/data/aurora/products/power/power_display_manifest.json")
)


def _metadata_path(output_zarr: Path) -> Path:
    return output_zarr.with_name("power_display_summary_metadata.json")


def _write_metadata(output_zarr: Path, display: xr.Dataset) -> Path:
    """Write the tiny initial-dashboard metadata sidecar atomically."""
    times = display["time"].values if "time" in display.coords else []
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source_zarr": str(output_zarr),
        "time_count": int(display.sizes.get("time", 0)),
        "variable_count": len(display.data_vars),
        "time_start_utc": str(times[0]) if len(times) else "",
        "time_end_utc": str(times[-1]) if len(times) else "",
    }
    for name in (
        "observation_history_days",
        "observation_window_start_utc",
        "observation_window_end_utc",
    ):
        if name in display.attrs:
            payload[name] = str(display.attrs[name])
    path = _metadata_path(output_zarr)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def _write_zarr_atomic(ds: xr.Dataset, output_zarr: Path, chunk_time: int = 1440) -> None:
    output_zarr.parent.mkdir(parents=True, exist_ok=True)
    # The generator lock serialises complete builds; this unique path also
    # keeps an interrupted build from being mistaken for the next build.
    tmp = output_zarr.with_name(f"{output_zarr.name}.tmp.{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    ds.chunk({"time": chunk_time}).to_zarr(tmp, mode="w", consolidated=True)
    if output_zarr.exists():
        shutil.rmtree(output_zarr)
    tmp.rename(output_zarr)
    zarr.consolidate_metadata(output_zarr)


def _open_optional_zarr(
    path: Path,
    label: str,
    *,
    eager: bool = False,
    attempts: int = 3,
) -> xr.Dataset | None:
    """Open an optional store, eagerly snapshotting small replace-in-place products."""
    if not path.exists():
        return None
    last_error: Exception | None = None
    for attempt in range(max(int(attempts), 1)):
        opened: xr.Dataset | None = None
        try:
            opened = xr.open_zarr(path, chunks={})
            if not eager:
                return opened
            loaded = opened.load()
            opened.close()
            return loaded
        except Exception as exc:
            last_error = exc
            if opened is not None:
                opened.close()
            if attempt + 1 < attempts:
                time.sleep(0.2)
    print(f"Could not open {label} Zarr for Power display summary: {last_error}")
    return None


def _validated_history_days(history_days: float) -> float:
    """Validate the retained observed window needed by every Power surface."""
    value = float(history_days)
    if not math.isfinite(value) or value < POWER_DISPLAY_MIN_HISTORY_DAYS:
        raise ValueError(
            "Power display history must retain at least "
            f"{POWER_DISPLAY_MIN_HISTORY_DAYS:g} days (96-hour mobile coverage and "
            f"{POWER_CUMULATIVE_CONTEXT_DAYS}-day cumulative context)"
        )
    return value


def _time_window_view(
    dataset: xr.Dataset | None,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp | None,
    fields: tuple[str, ...] | None = None,
) -> xr.Dataset | None:
    """Return a lazy, field-pruned time view without reading source data arrays."""
    if dataset is None or "time" not in dataset or dataset.sizes.get("time", 0) == 0:
        return dataset
    if fields is not None:
        names = [name for name in fields if name in dataset]
        dataset = dataset[names]
    times = pd.DatetimeIndex(dataset["time"].values)
    if times.is_monotonic_increasing:
        return dataset.sel(time=slice(start, end))
    mask = (~times.isna()) & (times >= start)
    if end is not None:
        mask &= times <= end
    return dataset.isel(time=mask)


def _recent_observation_views(
    power: xr.Dataset,
    ass_logger: xr.Dataset | None,
    pdu: xr.Dataset | None,
    *,
    history_days: float,
) -> tuple[xr.Dataset, xr.Dataset | None, xr.Dataset | None, pd.Timestamp, pd.Timestamp]:
    """Crop high-rate observations before pandas can materialise full history.

    APS currently contains millions of one-second rows. Only the recent live
    window is needed by the precomputed display products; older interactive
    requests continue to use the dashboard's existing raw-Zarr fallback.
    Starting at UTC midnight preserves the daily cumulative-energy traces for
    every retained day.
    """
    retained_days = _validated_history_days(history_days)
    if "time" not in power or power.sizes.get("time", 0) == 0:
        raise ValueError("Power Zarr contains no time samples")
    times = pd.DatetimeIndex(power["time"].values)
    valid_times = times[~times.isna()]
    if not len(valid_times):
        raise ValueError("Power Zarr contains no valid timestamps")
    end = pd.Timestamp(valid_times.max())
    start = (end - pd.Timedelta(days=retained_days)).normalize()
    recent_power = _time_window_view(
        power,
        start=start,
        end=end,
        fields=POWER_DISPLAY_SOURCE_FIELDS,
    )
    recent_ass_logger = _time_window_view(
        ass_logger,
        start=start,
        end=end,
        fields=POWER_DISPLAY_SUMMARY_CONTEXT_FIELDS,
    )
    recent_pdu = _time_window_view(
        pdu,
        start=start,
        end=end,
        fields=PDU_DISPLAY_SUMMARY_FIELDS,
    )
    assert recent_power is not None
    return recent_power, recent_ass_logger, recent_pdu, start, end


def _energy_subset(summary: xr.Dataset, freq: str) -> xr.Dataset:
    names = [name for name in POWER_DISPLAY_ENERGY_MAP.values() if name in summary]
    if not names:
        return xr.Dataset()
    out = summary[names].copy(deep=False)
    out.attrs = {
        POWER_DISPLAY_ENERGY_ATTR: "true",
        "source": "derived from power_display_summary.zarr",
        "frequency": freq,
        "description": "Display-only one-minute cumulative APS energy traces for dashboard plotting.",
    }
    for name in out.data_vars:
        out[name].attrs["units"] = "kWh"
    return out


def _section_subset(summary: xr.Dataset, section: str) -> xr.Dataset:
    """Return only the variables plotted by one Power browser section."""
    groups = {"observed"} if section == "current" else {"forecast_24h", "forecast_96h", "verification"}
    names = tuple(
        dict.fromkeys(
            trace.var
            for panel in SUMMARY_LAYOUTS["power"]
            if POWER_PANEL_TIME_GROUP_BY_KEY.get(panel.key, "observed") in groups
            for trace in panel.traces
            if trace.var in summary
        )
    )
    if not names:
        return xr.Dataset(coords={"time": summary["time"]})
    out = summary[list(names)].copy(deep=False)
    out.attrs = dict(summary.attrs)
    out.attrs.update(
        {
            POWER_DISPLAY_SUMMARY_ATTR: "true",
            "display_section": section,
            "description": f"Display-only Power {section} variables for the dashboard.",
        }
    )
    # Remove the empty timeline introduced when observed and forecast sources
    # are aligned. This keeps the forecast product compact and avoids full-array
    # validity scans in the browser request path.
    return out.dropna(dim="time", how="all")


def _display_descriptor(path: Path, display: xr.Dataset) -> dict[str, object]:
    times = display["time"].values if "time" in display.coords else []
    return {
        "path": str(path),
        "time_count": int(display.sizes.get("time", 0)),
        "variable_count": len(display.data_vars),
        "time_start_utc": str(times[0]) if len(times) else "",
        "time_end_utc": str(times[-1]) if len(times) else "",
    }


def _write_manifest(path: Path, current: xr.Dataset, forecast: xr.Dataset, current_path: Path, forecast_path: Path) -> None:
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "products": {
            "current": _display_descriptor(current_path, current),
            "forecast": _display_descriptor(forecast_path, forecast),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _try_generation_lock(output_zarr: Path):
    """Return a non-blocking lock handle, or ``None`` when a build is active."""
    lock_path = output_zarr.with_name(f".{output_zarr.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


def _release_generation_lock(handle) -> None:
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    handle.close()


def _generate_unlocked(
    power_zarr: Path = POWER_ZARR_PATH,
    output_zarr: Path = POWER_DISPLAY_SUMMARY_ZARR_PATH,
    ass_logger_zarr: Path = ASFS_LOGGER_ZARR_PATH,
    pdu_zarr: Path = PDU_ZARR_PATH,
    forecast_zarr: Path = POWER_SOC_FORECAST_ZARR_PATH,
    forecast_skill_zarr: Path = POWER_SOC_FORECAST_SKILL_ZARR_PATH,
    hindcast_zarr: Path = POWER_SOC_HINDCAST_ZARR_PATH,
    ensemble_forecast_zarr: Path = POWER_SOC_ENSEMBLE_ZARR_PATH,
    ensemble_skill_zarr: Path = POWER_SOC_ENSEMBLE_SKILL_ZARR_PATH,
    operating_scenarios_zarr: Path = POWER_OPERATING_SCENARIOS_ZARR_PATH,
    energy_output_zarr: Path | None = POWER_DISPLAY_ENERGY_ZARR_PATH,
    current_output_zarr: Path | None = POWER_CURRENT_DISPLAY_ZARR_PATH,
    forecast_output_zarr: Path | None = POWER_FORECAST_DISPLAY_ZARR_PATH,
    manifest_path: Path | None = POWER_DISPLAY_MANIFEST_PATH,
    freq: str = POWER_DISPLAY_SUMMARY_FREQ,
    history_days: float = POWER_DISPLAY_HISTORY_DAYS,
) -> Path:
    """Build the derived one-minute display-summary store from Power inputs."""
    power = xr.open_zarr(power_zarr, chunks={})
    ass_logger = _open_optional_zarr(ass_logger_zarr, "ASFS logger")
    pdu = _open_optional_zarr(pdu_zarr, "ASS PDU")
    power, ass_logger, pdu, observation_start, observation_end = _recent_observation_views(
        power,
        ass_logger,
        pdu,
        history_days=history_days,
    )
    forecast = _open_optional_zarr(forecast_zarr, "Power SOC forecast", eager=True)
    forecast_skill = _open_optional_zarr(forecast_skill_zarr, "Power SOC forecast skill", eager=True)
    hindcast = _open_optional_zarr(hindcast_zarr, "Power SOC hindcast", eager=True)
    ensemble_forecast = _open_optional_zarr(
        ensemble_forecast_zarr,
        "Power SOC ensemble forecast",
        eager=True,
    )
    ensemble_skill = _open_optional_zarr(
        ensemble_skill_zarr,
        "Power SOC ensemble skill",
        eager=True,
    )
    operating_scenarios = _open_optional_zarr(
        operating_scenarios_zarr,
        "Power operating scenarios",
        eager=True,
    )
    display = build_power_display_summary_dataset(
        power,
        ass_logger,
        pdu,
        forecast,
        forecast_skill,
        hindcast,
        ensemble_forecast,
        ensemble_skill,
        operating_scenarios,
        freq=freq,
    )
    if display.sizes.get("time", 0) == 0:
        raise ValueError("No display-summary samples could be generated from the Power Zarr")

    display.attrs[POWER_DISPLAY_SUMMARY_ATTR] = "true"
    display.attrs["observation_history_days"] = f"{float(history_days):g}"
    display.attrs["observation_window_start_utc"] = observation_start.isoformat()
    display.attrs["observation_window_end_utc"] = observation_end.isoformat()
    _write_zarr_atomic(display, output_zarr)
    metadata_path = _write_metadata(output_zarr, display)
    print(f"Wrote {output_zarr} with {display.sizes.get('time', 0)} samples and {len(display.data_vars)} variables")
    print(f"Wrote {metadata_path}")

    if energy_output_zarr is not None:
        energy = _energy_subset(display, freq)
        if energy.sizes.get("time", 0) and len(energy.data_vars):
            _write_zarr_atomic(energy, energy_output_zarr)
            print(f"Wrote {energy_output_zarr} with {energy.sizes.get('time', 0)} samples")
    if current_output_zarr is not None and forecast_output_zarr is not None:
        current = _section_subset(display, "current")
        forecast_display = _section_subset(display, "forecast")
        _write_zarr_atomic(current, current_output_zarr)
        _write_zarr_atomic(forecast_display, forecast_output_zarr)
        print(f"Wrote {current_output_zarr} with {current.sizes.get('time', 0)} samples")
        print(f"Wrote {forecast_output_zarr} with {forecast_display.sizes.get('time', 0)} samples")
        if manifest_path is not None:
            _write_manifest(manifest_path, current, forecast_display, current_output_zarr, forecast_output_zarr)
            print(f"Wrote {manifest_path}")
    return output_zarr


def generate(*args, **kwargs) -> Path:
    """Build one Power display generation at a time for all output products."""
    output_zarr = kwargs.get("output_zarr")
    if output_zarr is None:
        output_zarr = args[1] if len(args) > 1 else POWER_DISPLAY_SUMMARY_ZARR_PATH
    output_zarr = Path(output_zarr)
    lock = _try_generation_lock(output_zarr)
    if lock is None:
        print(f"Power display-summary build already running for {output_zarr}; skipping duplicate request")
        return output_zarr
    try:
        return _generate_unlocked(*args, **kwargs)
    finally:
        _release_generation_lock(lock)


def write_metadata_only(output_zarr: Path = POWER_DISPLAY_SUMMARY_ZARR_PATH) -> Path:
    """Backfill initial-dashboard metadata without rebuilding the Zarr product."""
    display = xr.open_zarr(output_zarr, chunks={}, consolidated=True)
    try:
        return _write_metadata(output_zarr, display)
    finally:
        display.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the compact Power display-summary Zarr")
    parser.add_argument("--power-zarr", type=Path, default=POWER_ZARR_PATH)
    parser.add_argument("--asfs-logger-zarr", type=Path, default=ASFS_LOGGER_ZARR_PATH)
    parser.add_argument("--pdu-zarr", type=Path, default=PDU_ZARR_PATH)
    parser.add_argument("--forecast-zarr", type=Path, default=POWER_SOC_FORECAST_ZARR_PATH)
    parser.add_argument("--forecast-skill-zarr", type=Path, default=POWER_SOC_FORECAST_SKILL_ZARR_PATH)
    parser.add_argument("--hindcast-zarr", type=Path, default=POWER_SOC_HINDCAST_ZARR_PATH)
    parser.add_argument("--ensemble-forecast-zarr", type=Path, default=POWER_SOC_ENSEMBLE_ZARR_PATH)
    parser.add_argument("--ensemble-skill-zarr", type=Path, default=POWER_SOC_ENSEMBLE_SKILL_ZARR_PATH)
    parser.add_argument("--operating-scenarios-zarr", type=Path, default=POWER_OPERATING_SCENARIOS_ZARR_PATH)
    parser.add_argument("--output-zarr", type=Path, default=POWER_DISPLAY_SUMMARY_ZARR_PATH)
    parser.add_argument("--energy-output-zarr", type=Path, default=POWER_DISPLAY_ENERGY_ZARR_PATH)
    parser.add_argument("--current-output-zarr", type=Path, default=POWER_CURRENT_DISPLAY_ZARR_PATH)
    parser.add_argument("--forecast-output-zarr", type=Path, default=POWER_FORECAST_DISPLAY_ZARR_PATH)
    parser.add_argument("--manifest-path", type=Path, default=POWER_DISPLAY_MANIFEST_PATH)
    parser.add_argument("--no-energy-output", action="store_true", help="Do not refresh the legacy cumulative-energy display Zarr")
    parser.add_argument("--freq", default=POWER_DISPLAY_SUMMARY_FREQ)
    parser.add_argument(
        "--history-days",
        type=float,
        default=POWER_DISPLAY_HISTORY_DAYS,
        help=(
            "Observed history retained in compact display products "
            f"(default: {POWER_DISPLAY_HISTORY_DAYS:g}; minimum: {POWER_DISPLAY_MIN_HISTORY_DAYS:g})"
        ),
    )
    parser.add_argument("--write-metadata-only", action="store_true")
    args = parser.parse_args()
    if args.write_metadata_only:
        print(f"Wrote {write_metadata_only(args.output_zarr)}")
        return
    generate(
        power_zarr=args.power_zarr,
        output_zarr=args.output_zarr,
        ass_logger_zarr=args.asfs_logger_zarr,
        pdu_zarr=args.pdu_zarr,
        forecast_zarr=args.forecast_zarr,
        forecast_skill_zarr=args.forecast_skill_zarr,
        hindcast_zarr=args.hindcast_zarr,
        ensemble_forecast_zarr=args.ensemble_forecast_zarr,
        ensemble_skill_zarr=args.ensemble_skill_zarr,
        operating_scenarios_zarr=args.operating_scenarios_zarr,
        energy_output_zarr=None if args.no_energy_output else args.energy_output_zarr,
        current_output_zarr=args.current_output_zarr,
        forecast_output_zarr=args.forecast_output_zarr,
        manifest_path=args.manifest_path,
        freq=args.freq,
        history_days=args.history_days,
    )


if __name__ == "__main__":
    main()
