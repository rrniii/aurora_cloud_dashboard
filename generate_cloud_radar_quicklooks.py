#!/usr/bin/env python3
"""
Generate daily quicklook PNGs for the cloud radar Zarr store.
One PNG per day (00:00–23:59 UTC) with C1ZE (log) and C1MeanVel panels.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.dates import DateFormatter, HourLocator
import numpy as np
import pandas as pd
import xarray as xr

ZARR_PATH = Path("/mnt/data/ass/rpgfmcw94/cloud_radar.zarr")
QUICKLOOK_DIR = Path("/home/aurora/aurora_cloud_dashboard/quicklooks/cloud_radar")
ZE_VMIN = -30.0
ZE_VMAX = 10.0
VEL_VMIN = -5.0
VEL_VMAX = 5.0
SPEC_VMIN = 0.0
SPEC_VMAX = 3.0
SLDR_VMIN = -100.0
SLDR_VMAX = -10.0
RHV_VMIN = 0.8
RHV_VMAX = 1.0
SRCX_VMIN = 0.8
SRCX_VMAX = 1.0
SKEW_VMIN = -2.0
SKEW_VMAX = 2.0
KURT_VMIN = 0.0
KURT_VMAX = 8.0
RANGE_MAX = 9000


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _plot_day(ds_day: xr.Dataset, date_label: str, output: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)
    vars_titles = [
        ("ZE_dBZ", "ZE (dBZ)", ZE_VMIN, ZE_VMAX, "ZE (dBZ)", "cividis", "linear"),
        ("MeanVel", "Mean Velocity", VEL_VMIN, VEL_VMAX, "Velocity (m/s)", "RdBu_r", "linear"),
        ("SpecWidth", "Spectrum Width (m/s)", SPEC_VMIN, SPEC_VMAX, "Spec Width (m/s)", "plasma", "linear"),
        ("SLDR", "SLDR (dB)", SLDR_VMIN, SLDR_VMAX, "SLDR (dB)", "RdBu_r", "linear"),
        ("RHV", "RHV", RHV_VMIN, RHV_VMAX, "RHV", "viridis", "linear"),
        ("SRCX", "SRCX", SRCX_VMIN, SRCX_VMAX, "SRCX", "viridis", "linear"),
        ("Skew", "Skew", SKEW_VMIN, SKEW_VMAX, "Skew", "RdBu_r", "linear"),
        ("Kurt", "Kurtosis", KURT_VMIN, KURT_VMAX, "Kurtosis", "magma", "linear"),
    ]

    for ax, (var, title, vmin, vmax, cbar_label, cmap, scale) in zip(axes, vars_titles):
        da = ds_day[var].transpose("time", "range")
        data = da.values
        mesh = ax.pcolormesh(
            da["time"].values,
            da["range"].values,
            data.T,
            shading="auto",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
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
    if "ZE_dBZ" not in ds or "MeanVel" not in ds:
        raise KeyError("Dataset missing ZE_dBZ or MeanVel")

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
