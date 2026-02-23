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


def _filter_readable(files, engine):
    """
    Return (readable_files, skipped) after cheap header/size probes.
    """
    readable = []
    skipped = []
    for f in files:
        size = os.path.getsize(f)
        if size == 0:
            skipped.append((f, "empty file"))
            print(f"Skipping unreadable file {os.path.basename(f)}: empty file")
            continue
        try:
            with open(f, "rb") as fh:
                sig = fh.read(8)
            if not (sig.startswith(b"CDF") or sig == b"\x89HDF\r\n\x1a\n"):
                skipped.append((f, "unexpected header"))
                print(f"Skipping unreadable file {os.path.basename(f)}: unexpected header")
                continue
            readable.append(f)
        except Exception as exc:
            skipped.append((f, exc))
            print(f"Skipping unreadable file {os.path.basename(f)}: {exc}")
    return readable, skipped


def _drop_bad_files(files, engine):
    """
    Attempt to open each file with xarray; return (good, skipped).
    """
    good = []
    skipped = []
    for f in files:
        try:
            xr.open_dataset(f, engine=engine).close()
            good.append(f)
        except Exception as exc:
            skipped.append((f, exc))
            print(f"Skipping unreadable file {os.path.basename(f)}: {exc}")
    return good, skipped


def append_new_files(
    input_dir,
    pattern,
    zarr_path,
    chunks="auto",
    engine="h5netcdf",
    append_dim="time",
    max_backfill_days=11,
    batch_size=200,
):
    if not os.path.isdir(zarr_path):
        raise FileNotFoundError(f"Zarr store not found: {zarr_path}")

    engine = choose_engine(engine)

    ds_existing = xr.open_zarr(zarr_path, chunks={})
    if append_dim not in ds_existing:
        raise ValueError(f"Append dimension '{append_dim}' not found in Zarr store.")

    last_time = pd.to_datetime(ds_existing[append_dim].max().values).to_pydatetime()
    print(f"Latest {append_dim} in Zarr: {last_time}")

    start_cutoff = last_time
    if max_backfill_days is not None:
        cutoff = pd.Timestamp.utcnow().replace(tzinfo=None) - pd.Timedelta(
            days=max_backfill_days
        )
        if cutoff > start_cutoff:
            start_cutoff = cutoff
            print(
                f"Limiting backfill to files on/after {start_cutoff} "
                f"(max_backfill_days={max_backfill_days})"
            )

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
        if ts > start_cutoff:
            new_files.append((ts, f))

    if not new_files:
        print("No new files to append.")
        return

    new_files.sort(key=lambda x: x[0])
    ordered_files = [f for _, f in new_files]

    readable_files, skipped = _filter_readable(ordered_files, engine)
    if skipped:
        print(f"Skipped {len(skipped)} unreadable files.")
    if not readable_files:
        print("No readable new files to append.")
        return

    print(f"Appending {len(readable_files)} new files:")
    for f in readable_files:
        print("  ", os.path.basename(f))

    total = len(readable_files)
    batches = [
        readable_files[i : i + batch_size] for i in range(0, total, batch_size)
    ]
    for idx, batch in enumerate(batches, start=1):
        print(f"Writing batch {idx}/{len(batches)} ({len(batch)} files)...")
        batch_files = list(batch)
        while batch_files:
            try:
                ds_new = xr.open_mfdataset(
                    batch_files,
                    combine="by_coords",
                    data_vars="minimal",
                    coords="minimal",
                    compat="override",
                    chunks=chunks,
                    engine=engine,
                    parallel=False,
                )
                break
            except Exception:
                batch_files, newly_skipped = _drop_bad_files(batch_files, engine)
                skipped.extend(newly_skipped)
                if not batch_files:
                    print("Batch became empty after dropping unreadable files; skipping batch.")
                    ds_new = None
                    break
        if ds_new is None:
            continue
        ds_new = ds_new.sortby(append_dim)
        ds_new.to_zarr(zarr_path, mode="a", append_dim=append_dim)
        ds_new.close()
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
    MAX_BACKFILL_DAYS = 11

    append_new_files(
        input_dir=INPUT_DIR,
        pattern=PATTERN,
        zarr_path=ZARR_PATH,
        chunks=CHUNKS,
        engine=ENGINE,
        append_dim=APPEND_DIM,
        max_backfill_days=MAX_BACKFILL_DAYS,
    )
