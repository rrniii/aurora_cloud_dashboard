#!/usr/bin/env python3
"""
Combine NetCDF files into a single Zarr store, filtering by calendar day
encoded in the filename as YYYYMMDD.

Example filename pattern:
    gamb2le_depolarisation_lidar_ceilometer_aurora_20251203_000056.nc
"""

import os
import glob
import re
from datetime import datetime, date
import xarray as xr


def extract_date_from_name(path):
    """
    Extract a date from filenames like:
        gamb2le_depolarisation_lidar_ceilometer_aurora_20251203_000056.nc

    Returns
    -------
    datetime.date or None
        Parsed date if found, otherwise None.
    """
    name = os.path.basename(path)
    # Look for 8 digits (YYYYMMDD) followed by underscore and 6 digits (HHMMSS)
    match = re.search(r"(\d{8})_\d{6}", name)
    if not match:
        return None
    date_part = match.group(1)
    return datetime.strptime(date_part, "%Y%m%d").date()


def extract_datetime_from_name(path):
    """
    Extract a full datetime from filenames like:
        gamb2le_depolarisation_lidar_ceilometer_aurora_20251203_000056.nc
    Returns
    -------
    datetime.datetime or None
    """
    name = os.path.basename(path)
    match = re.search(r"(\d{8})_(\d{6})", name)
    if not match:
        return None
    ts = match.group(1) + match.group(2)
    return datetime.strptime(ts, "%Y%m%d%H%M%S")


def choose_engine(preferred="h5netcdf"):
    """
    Choose a backend engine, falling back to netcdf4 if h5netcdf is missing.
    """
    if preferred == "h5netcdf":
        try:
            import h5netcdf  # noqa: F401
        except ModuleNotFoundError:
            print("h5netcdf not installed; falling back to netcdf4 engine.")
            return "netcdf4"
    return preferred


def make_zarr_from_netcdf(
    input_dir,
    pattern="gamb2le_depolarisation_lidar_ceilometer_aurora_*.nc",
    output_zarr="gamb2le_depolarisation_lidar_ceilometer_aurora.zarr",
    chunks=None,
    engine="h5netcdf",
    start_date=None,
):
    """
    Convert a collection of NetCDF files into a single Zarr store using xarray+zarr,
    filtering by calendar day if requested.

    Parameters
    ----------
    input_dir : str
        Directory containing the NetCDF files.
    pattern : str, optional
        Glob pattern for the NetCDF files.
    output_zarr : str, optional
        Path to the output Zarr store (directory).
    chunks : dict, "auto", or None, optional
        Dask chunking passed to xarray.open_mfdataset, e.g. {"time": 100} or "auto".
    engine : {"h5netcdf", "netcdf4", "scipy"}, optional
        Backend engine. h5netcdf avoids the thread-safety issues that can cause
        segfaults when opening many files in parallel with netCDF4.
    start_date : datetime.date or None, optional
        If provided, only files whose encoded date is >= start_date are included.
    """
    file_pattern = os.path.join(input_dir, pattern)
    all_files = sorted(glob.glob(file_pattern))

    if not all_files:
        raise FileNotFoundError(f"No files matched pattern: {file_pattern}")

    print(f"Found {len(all_files)} files matching pattern.")
    # Filter by calendar day if requested
    if start_date is not None:
        selected_files = []
        for f in all_files:
            d = extract_date_from_name(f)
            if d is None:
                raise ValueError(f"Could not parse date from filename: {f}")
            if d >= start_date:
                selected_files.append(f)
        print(f"Filtered to {len(selected_files)} files on or after {start_date}.")
    else:
        selected_files = all_files
        print("No date filtering applied.")

    if not selected_files:
        raise FileNotFoundError(
            f"No files found on or after {start_date} matching {file_pattern}"
        )

    print("Using files:")
    for f in selected_files:
        print("  ", os.path.basename(f))

    # Prefer h5netcdf to avoid netCDF4 thread-safety issues that can segfault.
    # If h5netcdf is missing, fall back to netCDF4 while keeping parallel=False.
    chosen_engine = choose_engine(engine)

    ds = xr.open_mfdataset(
        selected_files,
        combine="by_coords",
        data_vars="minimal",
        coords="minimal",
        compat="override",
        chunks=chunks,
        engine=chosen_engine,
        parallel=False,
    )

    print("Dataset opened.")
    print("Dimensions:", dict(ds.dims))

    # Write to Zarr
    print(f"Writing Zarr store to: {output_zarr}")
    ds.to_zarr(output_zarr, mode="w")

    print("Done.")


if __name__ == "__main__":
    # EDIT THESE VALUES FOR YOUR SETUP

    # Directory where your NetCDF files live
    INPUT_DIR = "/mnt/data/cl61"

    # Glob pattern for your files
    PATTERN = "gamb2le_depolarisation_lidar_ceilometer_aurora_*.nc"

    # Output Zarr store path
    OUTPUT_ZARR = "/mnt/data/cl61/gamb2le_depolarisation_lidar_ceilometer_aurora_20251201.zarr"

    # Calendar-day cutoff: only use files whose encoded date >= this
    # Example: 3 December 2025 → filenames with 20251203 or later
    START_DATE = date(2025, 12, 1)

    # Optional chunking for xarray / dask.
    # "auto" (or None) lets xarray pick chunking and avoids the misaligned
    # chunk warning during to_zarr.
    CHUNKS = "auto"
    # Engine selection; h5netcdf is safer when opening many files.
    ENGINE = "h5netcdf"

    make_zarr_from_netcdf(
        input_dir=INPUT_DIR,
        pattern=PATTERN,
        output_zarr=OUTPUT_ZARR,
        chunks=CHUNKS,
        engine=ENGINE,
        start_date=START_DATE,
    )
