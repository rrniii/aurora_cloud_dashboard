#!/usr/bin/env python3
"""
Generate daily quicklook PNGs for the cloud radar Zarr store.
One PNG per day (00:00–23:59 UTC) with reflectivity and SLDR panels.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, HourLocator
import numpy as np
import pandas as pd
import xarray as xr

ZARR_PATH = Path("/mnt/data/ass/rpgfmcw94/cloud_radar.zarr")
QUICKLOOK_DIR = Path("/home/aurora/aurora_cloud_dashboard/quicklooks/cloud_radar")
REFL_VMIN = -40
REFL_VMAX = 20
SLDR_VMIN = -30
SLDR_VMAX = 10
RANGE_MAX = 12000


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _plot_day(ds_day: xr.Dataset, date_label: str, output: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)
    vars_titles = [
        ("ze_dbz", "Reflectivity (dBZ)", REFL_VMIN, REFL_VMAX, "Reflectivity (dBZ)"),
        ("sldr_db", "SLDR (dB)", SLDR_VMIN, SLDR_VMAX, "SLDR (dB)"),
    ]

    for ax, (var, title, vmin, vmax, cbar_label) in zip(axes, vars_titles):
        da = ds_day[var].transpose("time", "range")
        data = da.values
        mesh = ax.pcolormesh(
            da["time"].values,
            da["range"].values,
            data.T,
            shading="auto",
            vmin=vmin,
            vmax=vmax,
            cmap="cividis" if var == "ze_dbz" else "viridis",
        )
        cbar = fig.colorbar(mesh, ax=ax, pad=0.01)
        cbar.set_label(cbar_label)
        ax.set_ylabel("range (m)")
        ax.set_title(title)
        ax.set_ylim(0, RANGE_MAX)

    locator = HourLocator()
    formatter = DateFormatter("%H:%M")
    for ax in axes:
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.tick_params(axis="x", labelrotation=90)
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(90)
            lbl.set_ha("right")
    axes[0].tick_params(axis="x", labelbottom=False)
    axes[-1].set_xlabel("Time (UTC)")

    fig.suptitle(date_label)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    _ensure_dir(output.parent)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Wrote {output}")


def main(force: bool = False):
    ds = xr.open_zarr(ZARR_PATH, chunks={})
    if "time" not in ds:
        raise KeyError("Dataset missing time coordinate")
    if "ze_dbz" not in ds or "sldr_db" not in ds:
        raise KeyError("Dataset missing ze_dbz or sldr_db")

    time_index = pd.DatetimeIndex(ds["time"].values)
    today = pd.Timestamp.utcnow().replace(tzinfo=None).date()
    dates = sorted(d for d in pd.Series(time_index.date).unique() if d < today)

    _ensure_dir(QUICKLOOK_DIR)

    if force:
        for png in QUICKLOOK_DIR.glob("cloud_radar_*.png"):
            png.unlink()
        print("Deleted existing quicklook PNGs.")

    for d in dates:
        out = QUICKLOOK_DIR / f"cloud_radar_{pd.Timestamp(d).strftime('%Y%m%d')}.png"
        if out.exists():
            continue
        start = pd.Timestamp(d)
        end = start + timedelta(days=1) - timedelta(milliseconds=1)
        mask = (time_index >= start) & (time_index <= end)
        if not mask.any():
            continue
        ds_day = ds.isel(time=mask).sortby("time")
        ds_day = ds_day.sel({"range": slice(0, RANGE_MAX)})
        if ds_day.sizes.get("time", 0) < 2:
            continue
        _plot_day(ds_day, pd.Timestamp(d).strftime("%Y-%m-%d"), out)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate cloud radar quicklook PNGs")
    parser.add_argument("--force", action="store_true", help="Regenerate all quicklooks")
    args = parser.parse_args()
    main(force=args.force)
