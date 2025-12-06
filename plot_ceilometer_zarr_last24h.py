#!/usr/bin/env python3
"""
Plot the last 24 hours of beta_att and linear_depol_ratio from a Zarr store.

Adjust ZARR_PATH at the bottom to point to your Zarr directory.
Outputs a PNG with two panels (beta_att on top, linear_depol_ratio on bottom).
"""

import sys
from datetime import timedelta

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.dates import DateFormatter, HourLocator
import numpy as np
import pandas as pd
import xarray as xr

BETA_CMAP = "cividis"


def _pick_range_coord(ds):
    """Return the best-guess range/height coordinate name."""
    for cand in ("range", "height", "altitude", "distance"):
        if cand in ds.coords:
            return cand
    raise KeyError("No range/height-like coordinate found (tried range/height/altitude/distance).")


def plot_last_24h(zarr_path, output_png="last24h.png"):
    # Open lazily; plotting will materialize the needed data.
    ds = xr.open_zarr(zarr_path, chunks={})

    if "time" not in ds:
        raise KeyError("Dataset is missing a 'time' coordinate.")

    range_coord = _pick_range_coord(ds)

    time_index = pd.DatetimeIndex(ds["time"].values)
    tz = time_index.tz  # None for naive, tzinfo for aware

    # Align our query window's tz-awareness with the dataset.
    if tz is None:
        end_time = pd.Timestamp.utcnow().replace(tzinfo=None)
    else:
        end_time = pd.Timestamp.now(tz=tz)
    start_time = end_time - timedelta(hours=24)

    # Use boolean masking to avoid pandas slice errors on unsorted/duplicate times.
    mask = (time_index >= start_time) & (time_index <= end_time)
    if not mask.any():
        raise ValueError("No data found in the last 24 hours.")
    ds_window = ds.isel(time=mask)
    ds_window = ds_window.sortby("time")
    # Limit range/height to 0–8000
    ds_window = ds_window.sel({range_coord: slice(0, 8000)})
    if ds_window.dims.get(range_coord, 0) == 0:
        raise ValueError("No data in range 0–8000.")
    if ds_window.time.size == 0:
        raise ValueError("No data found in the last 24 hours.")

    needed_vars = ["beta_att", "linear_depol_ratio"]
    missing = [v for v in needed_vars if v not in ds_window]
    if missing:
        raise KeyError(f"Missing variables in dataset: {', '.join(missing)}")

    # Match the quicklook styling: same figsize, shared axes.
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)
    vars_and_titles = [
        ("beta_att", "Beta Attenuated"),
        ("linear_depol_ratio", "Linear Depolarization Ratio"),
    ]

    for ax, (var, title) in zip(axes, vars_and_titles):
        da = ds_window[var]
        data = da.transpose("time", range_coord).values

        if var == "beta_att":
            data = np.where(data > 0, data, np.nan)
            if np.all(np.isnan(data)):
                raise ValueError("beta_att has no positive values to plot.")
            vmin = 1e-7
            vmax = 1e-3
            norm = LogNorm(vmin=vmin, vmax=vmax)
            mesh = ax.pcolormesh(
                da["time"].values,
                da[range_coord].values,
                data.T,
                shading="auto",
                norm=norm,
                cmap=BETA_CMAP,
            )
        else:
            mesh = ax.pcolormesh(
                da["time"].values,
                da[range_coord].values,
                data.T,
                shading="auto",
                vmin=0,
                vmax=0.5,
            )
        ax.set_ylabel(range_coord)
        ax.set_title(title)
        ax.set_ylim(0, 8000)
        cbar = fig.colorbar(mesh, ax=ax, pad=0.01)
        if var == "beta_att":
            cbar.set_label("beta_att (m⁻¹ sr⁻¹)")
        else:
            cbar.set_label("linear_depol_ratio")

    # Time ticks: labels on the hour, angled labels (match quicklook style).
    major_locator = HourLocator()
    formatter = DateFormatter("%H:%M")
    axes[-1].xaxis.set_major_locator(major_locator)
    axes[-1].xaxis.set_major_formatter(formatter)
    axes[-1].tick_params(axis="x", labelrotation=90)
    for lbl in axes[-1].get_xticklabels():
        lbl.set_rotation(90)
        lbl.set_ha("right")
    axes[0].xaxis.set_major_locator(major_locator)
    axes[0].xaxis.set_major_formatter(formatter)
    axes[0].tick_params(axis="x", labelrotation=90, labelbottom=False)  # hide top labels
    for lbl in axes[0].get_xticklabels():
        lbl.set_rotation(90)
        lbl.set_ha("right")
    axes[-1].set_xlabel("Time (UTC)")

    fig.suptitle("Last 24 hours")
    # Match quicklook spacing so rotated hour labels stay visible.
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(output_png, dpi=150)
    print(f"Wrote {output_png}")


if __name__ == "__main__":
    # EDIT THIS to point to your Zarr store
    ZARR_PATH = "/mnt/data/cl61/gamb2le_depolarisation_lidar_ceilometer_aurora_20251201.zarr"
    OUTPUT_PNG = "last24h.png"

    # Allow optional CLI args: zarr_path [output_png]
    if len(sys.argv) >= 2:
        ZARR_PATH = sys.argv[1]
    if len(sys.argv) >= 3:
        OUTPUT_PNG = sys.argv[2]

    plot_last_24h(ZARR_PATH, OUTPUT_PNG)
