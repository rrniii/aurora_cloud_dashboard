#!/usr/bin/env python3
"""
Plot time series of all 1D (time-only) variables in the ceilometer Zarr store.

This captures status/housekeeping fields (including nested groups such as
status/Data_generation_status or monitoring/internal_heater) by flattening
group names to var strings and selecting any variable whose only dimension
is time. Output is a single PNG with stacked subplots.
"""

from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, AutoDateLocator
import xarray as xr

# Paths
ZARR_PATH = Path("/mnt/data/cl61/gamb2le_depolarisation_lidar_ceilometer_aurora_20251201.zarr")
OUTPUT_PNG = Path("status_timeseries.png")


def _find_time_only_vars(ds: xr.Dataset):
    """Return a list of variable names whose only dimension is time."""
    return [v for v in ds.data_vars if ds[v].dims == ("time",)]


def main():
    ds = xr.open_zarr(ZARR_PATH, consolidated=True)

    status_vars = _find_time_only_vars(ds)
    if not status_vars:
        raise RuntimeError("No time-only status variables found.")

    n = len(status_vars)
    ncols = 2
    nrows = ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3 * nrows), sharex=True)
    axes = axes.flat if n > 1 else [axes]

    # Format x-axis once
    locator = AutoDateLocator()
    formatter = DateFormatter("%Y-%m-%d\n%H:%M")

    for ax, var in zip(axes, status_vars):
        da = ds[var].load()
        times = da["time"].values
        ax.plot(times, da.values, lw=0.8)
        ax.set_title(var)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

    # Hide any unused axes
    for ax in axes[n:]:
        ax.set_visible(False)

    axes[-1].set_xlabel("Time (UTC)")
    fig.suptitle("Ceilometer status / housekeeping time series", y=0.995)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12)
    fig.savefig(OUTPUT_PNG, dpi=150)
    print(f"Wrote {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
