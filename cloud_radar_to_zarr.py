#!/usr/bin/env python3
"""
Build a Zarr store for the RPG FMCW 94 GHz cloud radar from NetCDF Level 1 files.

- Recursively scans /mnt/data/ass/rpgfmcw94 (year/month/day layout) for *.NC files.
- Uses only files at/after a provided start timestamp (default: 2026-02-23 13:00:00 UTC).
- Extracts Chirp 1 reflectivity (C1ZE) and slanted LDR (C1SLDR), converts to dBZ, and
  writes them as ze_dbz and sldr_db on coords (time, range).
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List

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


def _list_nc_files(root: Path, start_dt: datetime) -> List[Path]:
    files: List[Path] = []
    for p in root.rglob("*.NC"):
        if not p.name.upper().endswith("LV1.NC"):
            continue
        ts = _parse_timestamp(p)
        if ts is None:
            continue
        if ts >= start_dt:
            files.append((ts, p))
    files.sort(key=lambda x: x[0])
    return [p for _, p in files]


COMBINE_VARS = [
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


def _combine_chirps(raw: xr.Dataset):
    """Combine chirp 1 and chirp 2 along range, using C1 time axis."""
    required = ["C1Range", "C2Range"]
    for r in required:
        if r not in raw:
            raise KeyError(f"Missing {r} in file")

    r1 = raw["C1Range"].values
    r2 = raw["C2Range"].values
    ranges = np.concatenate([r1, r2])

    t_len = raw["Time"].sizes["Time"]
    r_len = len(ranges)

    combined = {}
    for out_name, c1_name, c2_name, mode in COMBINE_VARS:
        if c1_name not in raw or c2_name not in raw:
            continue
        arr = np.full((t_len, r_len), np.nan, dtype=np.float32)
        arr[:, : len(r1)] = raw[c1_name].values
        arr[:, len(r1) :] = raw[c2_name].values
        # mask fill
        arr = np.where(arr <= -900, np.nan, arr)
        if mode == "dbz":
            arr = np.where(arr > 0, arr, np.nan)
            with np.errstate(divide="ignore"):
                arr = 10.0 * np.log10(arr)
        combined[out_name] = arr.astype(np.float32)
    return ranges, combined


def _load_nc(path: Path) -> xr.Dataset:
    raw = xr.open_dataset(path, decode_times=False)
    if "Time" not in raw:
        raise KeyError(f"Missing Time in {path}")
    base = TIME_ZERO
    time = base + raw["Time"].astype("timedelta64[s]") + raw["Timems"].astype("timedelta64[ms]")
    time_vals = np.array(time.values)

    ranges, combined = _combine_chirps(raw)

    data_vars = {name: (("time", "range"), arr) for name, arr in combined.items()}
    ds = xr.Dataset(data_vars, coords={"time": time_vals, "range": ranges})
    return ds.sortby("time")


def build_zarr(root: Path, output: Path, start_dt: datetime, chunks: dict | str | None = None):
    files = _list_nc_files(root, start_dt)
    if not files:
        raise FileNotFoundError(f"No .NC files found at/after {start_dt} in {root}")
    print(f"Found {len(files)} files >= {start_dt.isoformat()}.")

    datasets: List[xr.Dataset] = []
    for f in files:
        print(f"Loading {f.name}")
        ds = _load_nc(f)
        datasets.append(ds)

    combined = xr.concat(datasets, dim="time")
    combined = combined.sortby("time")
    if chunks:
        combined = combined.chunk(chunks)

    print(f"Writing Zarr to {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_zarr(output, mode="w")
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Build cloud radar Zarr store from NC files.")
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT, help="Root directory containing Yxxxx/Mxx/Dxx folders")
    parser.add_argument("--output", type=Path, default=ZARR_DEFAULT, help="Output Zarr path")
    parser.add_argument(
        "--start",
        type=str,
        default="2026-02-23T13:00:00Z",
        help="Start timestamp (UTC, ISO format) to include files",
    )
    parser.add_argument("--chunk-time", type=int, default=400, help="Optional time chunk size")
    args = parser.parse_args()

    start_dt = datetime.fromisoformat(args.start.replace("Z", "+00:00"))
    chunks = {"time": args.chunk_time} if args.chunk_time else None
    build_zarr(args.root, args.output, start_dt, chunks=chunks)


if __name__ == "__main__":
    main()
