#!/usr/bin/env python3
"""
Append any new NetCDF files into an existing Zarr store.

This script:
1) Opens the target Zarr store to find the most recent time coordinate.
2) Finds NetCDF files matching a pattern whose encoded timestamp is newer.
3) Appends those files along the time dimension.

Filenames are expected to contain a timestamp like YYYYMMDD_HHMMSS, e.g.:
    gamb2le_depolarisation_lidar_ceilometer_aurora_20251203_000056.nc
"""

import glob
import os

import pandas as pd
import xarray as xr

from netcdf_to_zarr import choose_engine, extract_datetime_from_name


def append_new_files(
    input_dir,
    pattern,
    zarr_path,
    chunks="auto",
    engine="h5netcdf",
    append_dim="time",
):
    if not os.path.isdir(zarr_path):
        raise FileNotFoundError(f"Zarr store not found: {zarr_path}")

    engine = choose_engine(engine)

    ds_existing = xr.open_zarr(zarr_path, chunks={})
    if append_dim not in ds_existing:
        raise ValueError(f"Append dimension '{append_dim}' not found in Zarr store.")

    last_time = pd.to_datetime(ds_existing[append_dim].max().values).to_pydatetime()
    print(f"Latest {append_dim} in Zarr: {last_time}")

    file_pattern = os.path.join(input_dir, pattern)
    all_files = sorted(glob.glob(file_pattern))
    if not all_files:
        print(f"No files found matching pattern: {file_pattern}")
        return

    new_files = []
    for f in all_files:
        ts = extract_datetime_from_name(f)
        if ts is None:
            print(f"Skipping file without parsable timestamp: {f}")
            continue
        if ts > last_time:
            new_files.append((ts, f))

    if not new_files:
        print("No new files to append.")
        return

    new_files.sort(key=lambda x: x[0])
    ordered_files = [f for _, f in new_files]

    print(f"Appending {len(ordered_files)} new files:")
    for f in ordered_files:
        print("  ", os.path.basename(f))

    ds_new = xr.open_mfdataset(
        ordered_files,
        combine="by_coords",
        data_vars="minimal",
        coords="minimal",
        compat="override",
        chunks=chunks,
        engine=engine,
        parallel=False,
    )

    ds_new = ds_new.sortby(append_dim)

    ds_new.to_zarr(zarr_path, mode="a", append_dim=append_dim)
    print("Append complete.")


if __name__ == "__main__":
    # EDIT THESE VALUES FOR YOUR SETUP
    INPUT_DIR = "/mnt/data/cl61"
    PATTERN = "gamb2le_depolarisation_lidar_ceilometer_aurora_*.nc"
    ZARR_PATH = "/mnt/data/cl61/gamb2le_depolarisation_lidar_ceilometer_aurora_20251201.zarr"

    # Chunking: "auto" or None generally avoids misaligned chunk warnings.
    CHUNKS = "auto"

    ENGINE = "h5netcdf"
    APPEND_DIM = "time"

    append_new_files(
        input_dir=INPUT_DIR,
        pattern=PATTERN,
        zarr_path=ZARR_PATH,
        chunks=CHUNKS,
        engine=ENGINE,
        append_dim=APPEND_DIM,
    )
