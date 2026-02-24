#!/usr/bin/env python3
"""
Append new RPG FMCW 94 GHz cloud radar NetCDF files into an existing Zarr store.
- Scans the root directory recursively for *.NC files newer than the latest time in the Zarr.
- Uses the same conversion logic as cloud_radar_to_zarr.py (Chirp 1 Ze -> dBZ, SLDR).
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import xarray as xr

ROOT_DEFAULT = Path("/mnt/data/ass/rpgfmcw94")
ZARR_DEFAULT = Path("/mnt/data/ass/rpgfmcw94/cloud_radar.zarr")
TIME_ZERO = np.datetime64("2001-01-01T00:00:00")
NC_REGEX = re.compile(r"_(\d{6})_(\d{6})")  # yymmdd_hhmmss


def _parse_timestamp(path: Path) -> datetime | None:
    m = NC_REGEX.search(path.name)
    if not m:
        return None
    date_part, time_part = m.groups()
    try:
        return datetime.strptime(date_part + time_part, "%y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _list_new_files(root: Path, after: datetime) -> List[Path]:
    files: List[Path] = []
    for p in root.rglob("*.NC"):
        if not p.name.upper().endswith("LV1.NC"):
            continue
        ts = _parse_timestamp(p)
        if ts is None:
            continue
        if ts > after:
            files.append((ts, p))
    files.sort(key=lambda x: x[0])
    return [p for _, p in files]


def _load_nc(path: Path) -> xr.Dataset:
    raw = xr.open_dataset(path, decode_times=False)
    base = TIME_ZERO
    time = base + raw["Time"].astype("timedelta64[s]") + raw["Timems"].astype("timedelta64[ms]")
    time_vals = np.array(time.values)

    required = ["C1Range", "C2Range"]
    for r in required:
        if r not in raw:
            raise KeyError(f"Missing {r} in {path}")

    r1 = raw["C1Range"].values
    r2 = raw["C2Range"].values
    ranges = np.concatenate([r1, r2])
    t_len = raw["Time"].sizes["Time"]

    var_specs = [
        ("ZE_dBZ", "C1ZE", "C2ZE", "dbz"),
        ("ZE45_dBZ", "C1ZE45", "C2ZE45", "dbz"),
        ("MeanVel", "C1MeanVel", "C2MeanVel", "linear"),
        ("ZDR", "C1ZDR", "C2ZDR", "linear"),
        ("SRCX", "C1SRCX", "C2SRCX", "linear"),
        ("SpecWidth", "C1SpecWidth", "C2SpecWidth", "linear"),
        ("SLDR", "C1SLDR", "C2SLDR", "linear"),
        ("Skew", "C1Skew", "C2Skew", "linear"),
        ("RHV", "C1RHV", "C2RHV", "linear"),
        ("PhiDP", "C1PhiDP", "C2PhiDP", "linear"),
        ("Kurt", "C1Kurt", "C2Kurt", "linear"),
        ("KDP", "C1KDP", "C2KDP", "linear"),
        ("DiffAtt", "C1DiffAtt", "C2DiffAtt", "linear"),
    ]

    data_vars = {}
    for out_name, c1, c2, mode in var_specs:
        if c1 not in raw or c2 not in raw:
            continue
        arr = np.full((t_len, len(ranges)), np.nan, dtype=np.float32)
        arr[:, : len(r1)] = raw[c1].values
        arr[:, len(r1) :] = raw[c2].values
        arr = np.where(arr <= -900, np.nan, arr)
        if mode == "dbz":
            arr = np.where(arr > 0, arr, np.nan)
            with np.errstate(divide="ignore"):
                arr = 10.0 * np.log10(arr)
        data_vars[out_name] = (("time", "range"), arr.astype(np.float32))

    ds = xr.Dataset(data_vars, coords={"time": time_vals, "range": ranges})
    return ds.sortby("time")


def append_new(root: Path, zarr_path: Path, chunks: dict | str | None = None):
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr store not found: {zarr_path}")

    base = xr.open_zarr(zarr_path, chunks={})
    if "time" not in base:
        raise KeyError("Zarr store missing time coordinate")
    last_time = pd.to_datetime(base["time"].max().values).to_pydatetime().replace(tzinfo=timezone.utc)
    print(f"Latest time in Zarr: {last_time}")

    files = _list_new_files(root, last_time)
    if not files:
        print("No new .NC files to append.")
        return

    print(f"Appending {len(files)} files")
    datasets = []
    for f in files:
        print(f"  {f.name}")
        ds = _load_nc(f)
        datasets.append(ds)

    combined = xr.concat(datasets, dim="time").sortby("time")
    if chunks:
        combined = combined.chunk(chunks)
    # Ensure chunk alignment with existing store; disable strict chunk safety to avoid overlap errors.
    combined.to_zarr(zarr_path, mode="a", append_dim="time", safe_chunks=False)
    print("Append complete.")


def main():
    parser = argparse.ArgumentParser(description="Append new cloud radar NC files into existing Zarr.")
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument("--zarr", type=Path, default=ZARR_DEFAULT)
    parser.add_argument("--chunk-time", type=int, default=400)
    args = parser.parse_args()

    chunks = {"time": args.chunk_time} if args.chunk_time else None
    append_new(args.root, args.zarr, chunks=chunks)


if __name__ == "__main__":
    main()
