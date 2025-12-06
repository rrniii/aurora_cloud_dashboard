#!/usr/bin/env python3
"""
Consolidate Zarr metadata for faster opens.

Usage:
    python consolidate_zarr_metadata.py /path/to/store.zarr

By default, uses the main Zarr from the app:
    /mnt/data/cl61/gamb2le_depolarisation_lidar_ceilometer_aurora_20251201.zarr
"""

import sys
import zarr


def consolidate(path):
    print(f"Consolidating metadata for: {path}")
    zarr.consolidate_metadata(path)
    print("Done.")


if __name__ == "__main__":
    default = "/mnt/data/cl61/gamb2le_depolarisation_lidar_ceilometer_aurora_20251201.zarr"
    target = sys.argv[1] if len(sys.argv) > 1 else default
    consolidate(target)
