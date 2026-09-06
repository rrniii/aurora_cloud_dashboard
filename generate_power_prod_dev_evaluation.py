#!/usr/bin/env python3
"""Materialise non-blocking, exact prod/dev power forecast evidence."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import xarray as xr

from power_prod_dev_evaluation import (
    build_exact_intersection_evidence,
    write_paired_products,
)


PROD_ARCHIVE = Path(
    os.environ.get(
        "AURORA_POWER_PROD_FORECAST_ARCHIVE_ZARR",
        "/data/aurora/products/power/power_soc_forecast_archive.zarr",
    )
)
DEV_ARCHIVE = Path(
    os.environ.get(
        "AURORA_POWER_DEV_FORECAST_ARCHIVE_ZARR",
        "/data/aurora/dev-products/power/power_soc_forecast_archive.zarr",
    )
)
POWER_ZARR = Path(os.environ.get("POWER_ZARR_PATH", "/data/aurora/products/power/power.zarr"))
EVALUATION_ROOT = Path(
    os.environ.get(
        "AURORA_POWER_PROD_DEV_EVALUATION_ROOT",
        "/data/aurora/dev-products/power/evaluation/prod-vs-dev",
    )
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build exact-intersection production/development power evidence"
    )
    parser.add_argument("--prod-archive-zarr", type=Path, default=PROD_ARCHIVE)
    parser.add_argument("--dev-archive-zarr", type=Path, default=DEV_ARCHIVE)
    parser.add_argument("--power-zarr", type=Path, default=POWER_ZARR)
    parser.add_argument(
        "--output-zarr",
        type=Path,
        default=EVALUATION_ROOT / "paired_evidence.zarr",
    )
    parser.add_argument(
        "--status-json",
        type=Path,
        default=EVALUATION_ROOT / "status.json",
    )
    parser.add_argument(
        "--history-jsonl",
        type=Path,
        default=EVALUATION_ROOT / "history.jsonl",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    args = parser.parse_args()
    with xr.open_zarr(args.prod_archive_zarr, chunks={}) as opened:
        production = opened.load()
    with xr.open_zarr(args.dev_archive_zarr, chunks={}) as opened:
        development = opened.load()
    power = None
    if args.power_zarr.exists():
        with xr.open_zarr(args.power_zarr, chunks={}) as opened:
            power = opened.load()
    evidence = build_exact_intersection_evidence(
        production,
        development,
        power=power,
    )
    summary = write_paired_products(
        evidence,
        output_zarr=args.output_zarr,
        status_json=args.status_json,
        history_jsonl=args.history_jsonl,
        bootstrap_samples=max(int(args.bootstrap_samples), 0),
    )
    print(
        "Paired prod/dev power evidence: "
        f"status={summary['status']} rows={summary['pairedRows']} "
        f"cycles={summary['pairedIndependentCycles']}"
    )


if __name__ == "__main__":
    main()
