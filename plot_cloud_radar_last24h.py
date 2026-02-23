#!/usr/bin/env python3
"""
Render the last 24 hours of cloud radar data (Chirp 1) to a PNG for the dashboard.
Two panels: C1ZE (log-scaled) and C1MeanVel.
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.dates import DateFormatter, HourLocator
import numpy as np
import pandas as pd
import xarray as xr

ZARR_DEFAULT = Path("/mnt/data/ass/rpgfmcw94/cloud_radar.zarr")
OUTPUT_DEFAULT = Path("last24h_cloudradar.png")

ZE_VMIN = -30.0
ZE_VMAX = 10.0
VEL_VMIN = -5.0
VEL_VMAX = 5.0
RANGE_MAX = 9000


def plot_last_24h(zarr_path: Path, output: Path):
    ds = xr.open_zarr(zarr_path, chunks={})
    if "time" not in ds:
        raise KeyError("Dataset missing time coordinate")
    if "ZE_dBZ" not in ds or "MeanVel" not in ds:
        raise KeyError("Dataset missing ZE_dBZ or MeanVel")

    time_index = pd.DatetimeIndex(ds["time"].values)
    end_time = pd.Timestamp.utcnow().replace(tzinfo=None)
    start_time = end_time - timedelta(hours=24)

    mask = (time_index >= start_time) & (time_index <= end_time)
    if not mask.any():
        raise ValueError("No data in last 24h")
    window = ds.isel(time=mask).sortby("time")
    window = window.sel({"range": slice(0, RANGE_MAX)})

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)
    vars_titles = [
        ("ZE_dBZ", "ZE (dBZ)", ZE_VMIN, ZE_VMAX, "ZE (dBZ)", "cividis", "linear"),
        ("MeanVel", "Mean Velocity", VEL_VMIN, VEL_VMAX, "Velocity (m/s)", "RdBu_r", "linear"),
    ]

    for ax, (var, title, vmin, vmax, cbar_label, cmap, scale) in zip(axes, vars_titles):
        da = window[var]
        data = da.transpose("time", "range").values
        mesh = ax.pcolormesh(
            da["time"].values,
            da["range"].values,
            data.T,
            shading="auto",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
        ax.set_ylabel("range (m)")
        ax.set_title(title)
        ax.set_ylim(0, RANGE_MAX)
        cbar = fig.colorbar(mesh, ax=ax, pad=0.01)
        cbar.set_label(cbar_label)

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

    fig.suptitle("Cloud Radar – Last 24 hours")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(output, dpi=150)
    print(f"Wrote {output}")


def main():
    zarr_path = ZARR_DEFAULT
    output = OUTPUT_DEFAULT
    if len(sys.argv) >= 2:
        zarr_path = Path(sys.argv[1])
    if len(sys.argv) >= 3:
        output = Path(sys.argv[2])
    plot_last_24h(zarr_path, output)


if __name__ == "__main__":
    main()
