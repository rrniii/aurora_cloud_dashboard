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

    def _to_dbz(da: xr.DataArray) -> xr.DataArray:
        da = da.astype("float32").where(da > 0)
        return 10.0 * np.log10(da)

    range_da = raw["C1Range"].rename({"C1Range": "range"})
    ze_dbz = _to_dbz(raw["C1ZE"]).rename({"Time": "time", "C1Range": "range"})
    sldr_db = raw["C1SLDR"].astype("float32").rename({"Time": "time", "C1Range": "range"})

    ze_dbz = ze_dbz.assign_coords(time=("time", time_vals), range=("range", range_da.values))
    sldr_db = sldr_db.assign_coords(time=("time", time_vals), range=("range", range_da.values))
    ds = xr.Dataset({"ze_dbz": ze_dbz, "sldr_db": sldr_db})
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

    combined.to_zarr(zarr_path, mode="a", append_dim="time")
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
