#!/usr/bin/env python3
"""
Generate one PNG per day from the ceilometer Zarr store.

For each day up to (but not including) today:
- Slice 00:00–23:59 for that date
- Plot beta_att (log10) and linear_depol_ratio stacked
- Label hours along the bottom; at noon also show the date on its own line
- Save to quicklooks/ceilometer/ceilometer_YYYYMMDD.png
Skip any PNG that already exists.
"""

from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.dates import HourLocator, DateFormatter
import numpy as np
import pandas as pd
import xarray as xr

# --- Paths and constants ---
ZARR_PATH = Path("/mnt/data/cl61/gamb2le_depolarisation_lidar_ceilometer_aurora_20251201.zarr")
QUICKLOOK_DIR = Path("/home/aurora/aurora_cloud_dashboard/quicklooks/ceilometer")
BETA_CMAP = "cividis"
BETA_VMIN = 1e-7
BETA_VMAX = 1e-3
LDR_VMIN = 0.0
LDR_VMAX = 0.5
RANGE_MAX = 8000


def _ensure_dir(path: Path) -> None:
    """Create parent directory if missing."""
    path.mkdir(parents=True, exist_ok=True)


def _plot_day(ds_day: xr.Dataset, range_coord: str, date_label: str, output: Path) -> None:
    """Render and save a two-panel quicklook for one day (no plot title)."""
    # Mirror the formatting used in plot_ceilometer_zarr_last24h.py so ticks are readable.
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)
    vars_and_titles = [
        ("beta_att", "Beta Attenuated"),
        ("linear_depol_ratio", "Linear Depolarization Ratio"),
    ]

    for ax, (var, title) in zip(axes, vars_and_titles):
        da = ds_day[var].transpose("time", range_coord)
        data = da.values
        if var == "beta_att":
            data = np.where(data > 0, data, np.nan)
            mesh = ax.pcolormesh(
                da["time"].values,
                da[range_coord].values,
                data.T,
                shading="auto",
                norm=LogNorm(vmin=BETA_VMIN, vmax=BETA_VMAX),
                cmap=BETA_CMAP,
            )
            cbar = fig.colorbar(mesh, ax=ax, pad=0.01)
            cbar.set_label("beta_att (m⁻¹ sr⁻¹)")
        else:
            mesh = ax.pcolormesh(
                da["time"].values,
                da[range_coord].values,
                data.T,
                shading="auto",
                vmin=LDR_VMIN,
                vmax=LDR_VMAX,
            )
            cbar = fig.colorbar(mesh, ax=ax, pad=0.01)
            cbar.set_label("linear_depol_ratio")
        ax.set_ylabel(range_coord)
        ax.set_title(title)
        ax.set_ylim(0, RANGE_MAX)

    # Hourly ticks and labels exactly as in plot_ceilometer_zarr_last24h.py
    major_locator = HourLocator()
    formatter = DateFormatter("%H:%M")
    for ax in axes:
        ax.xaxis.set_major_locator(major_locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.tick_params(axis="x", labelrotation=90)
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(90)
            lbl.set_ha("right")
    axes[0].tick_params(axis="x", labelbottom=False)  # only bottom plot shows labels
    axes[-1].set_xlabel("Time (UTC)")

    fig.suptitle(date_label)
    fig.tight_layout()
    # Give rotated labels room to breathe.
    fig.subplots_adjust(bottom=0.22)
    _ensure_dir(output.parent)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Wrote {output}")


def main():
    ds = xr.open_zarr(ZARR_PATH, chunks={})
    if "time" not in ds:
        raise KeyError("Dataset is missing a 'time' coordinate.")
    if "beta_att" not in ds or "linear_depol_ratio" not in ds:
        raise KeyError("Dataset must contain beta_att and linear_depol_ratio.")

    # Determine range coordinate and time index
    range_coord = "range" if "range" in ds.coords else _pick_range_coord(ds)
    time_index = pd.DatetimeIndex(ds["time"].values)
    tz = time_index.tz
    today = (pd.Timestamp.now(tz=tz) if tz is not None else pd.Timestamp.utcnow()).date()

    # Unique dates available, excluding today
    dates = pd.Series(time_index.date).unique()
    dates = sorted(d for d in dates if d < today)

    _ensure_dir(QUICKLOOK_DIR)

    for d in dates:
        out_path = QUICKLOOK_DIR / f"ceilometer_{pd.Timestamp(d).strftime('%Y%m%d')}.png"
        if out_path.exists():
            continue
        start = pd.Timestamp(d)
        if tz is not None:
            start = start.tz_localize(tz)
        end = start + timedelta(days=1) - timedelta(milliseconds=1)

        mask = (time_index >= start) & (time_index <= end)
        if not mask.any():
            continue
        ds_day = ds.isel(time=mask).sortby("time")
        ds_day = ds_day.sel({range_coord: slice(0, RANGE_MAX)})
        # Skip if less than 1 hour of data or no range samples
        if ds_day.sizes.get("time", 0) < 1 or ds_day.sizes.get(range_coord, 0) == 0:
            continue
        # Require at least ~1 hour of coverage (>= 2 time samples)
        times = pd.to_datetime(ds_day["time"].values)
        if len(times) < 2 or (times.max() - times.min()) < np.timedelta64(1, "h"):
            continue
        _plot_day(ds_day, range_coord, pd.Timestamp(d).strftime("%Y-%m-%d"), out_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate daily ceilometer quicklook PNGs.")
    parser.add_argument(
        "--force", action="store_true", help="Delete existing quicklooks and regenerate all."
    )
    args = parser.parse_args()

    if args.force and QUICKLOOK_DIR.exists():
        for png in QUICKLOOK_DIR.glob("ceilometer_*.png"):
            png.unlink()
        print("Deleted existing quicklook PNGs.")

    main()
