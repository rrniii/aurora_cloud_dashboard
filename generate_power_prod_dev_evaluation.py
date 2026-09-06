#!/usr/bin/env python3
"""Materialise non-blocking, exact prod/dev power forecast evidence."""

from __future__ import annotations

import argparse
import os
import signal
from pathlib import Path
from types import FrameType
from typing import Any

import xarray as xr

from power_prod_dev_evaluation import (
    attach_paired_observations,
    build_exact_intersection_evidence,
    paired_observation_view,
    write_evaluation_status_event,
    write_paired_products,
)
from power_v12_hybrid import stable_json_digest, utc_now_iso


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


class EvaluationInterrupted(RuntimeError):
    """Raised when the evaluator receives a cooperative termination signal."""


def _raise_interrupted(signum: int, _frame: FrameType | None) -> None:
    try:
        signal_name = signal.Signals(signum).name
    except ValueError:
        signal_name = str(signum)
    raise EvaluationInterrupted(signal_name)


def _materialise_evidence(
    *,
    prod_archive_zarr: Path,
    dev_archive_zarr: Path,
    power_zarr: Path,
) -> xr.Dataset:
    """Build pairs first, loading only APS samples those pairs can score."""

    with xr.open_zarr(prod_archive_zarr, chunks={}) as opened:
        production = opened.load()
    with xr.open_zarr(dev_archive_zarr, chunks={}) as opened:
        development = opened.load()
    evidence = build_exact_intersection_evidence(production, development)
    if evidence.sizes.get("record", 0) == 0 or not power_zarr.exists():
        return evidence
    with xr.open_zarr(power_zarr, chunks={}) as opened:
        observation_view = paired_observation_view(opened, evidence)
        observations = observation_view.load()
    return attach_paired_observations(evidence, observations)


def run_evaluation(
    *,
    prod_archive_zarr: Path,
    dev_archive_zarr: Path,
    power_zarr: Path,
    output_zarr: Path,
    status_json: Path,
    history_jsonl: Path,
    bootstrap_samples: int = 500,
) -> dict[str, Any]:
    """Run one fail-closed evaluation with durable lifecycle evidence."""

    requested_at = utc_now_iso()
    run_id = "prod-dev-evaluation-" + stable_json_digest(
        {
            "requested_at_utc": requested_at,
            "prod_archive_zarr": str(prod_archive_zarr),
            "dev_archive_zarr": str(dev_archive_zarr),
        }
    )[:20]
    running = {
        "schemaVersion": 1,
        "environment": "development",
        "authority": "evaluation_only",
        "status": "running",
        "campaignReady": False,
        "runID": run_id,
        "requestedAt": requested_at,
        "dataUpdatedAt": requested_at,
        "reasonCode": "prod_dev_evaluation_started",
    }
    try:
        write_evaluation_status_event(
            status_json=status_json,
            history_jsonl=history_jsonl,
            event=running,
        )
        evidence = _materialise_evidence(
            prod_archive_zarr=prod_archive_zarr,
            dev_archive_zarr=dev_archive_zarr,
            power_zarr=power_zarr,
        )
        evidence_status = str(evidence.attrs.get("status", "complete"))
        return write_paired_products(
            evidence,
            output_zarr=output_zarr,
            status_json=status_json,
            history_jsonl=history_jsonl,
            bootstrap_samples=max(int(bootstrap_samples), 0),
            status_context={
                "environment": "development",
                "authority": "evaluation_only",
                "runID": run_id,
                "requestedAt": requested_at,
                "reasonCode": f"prod_dev_evaluation_{evidence_status}",
            },
        )
    except Exception as exc:
        interrupted = isinstance(exc, EvaluationInterrupted)
        updated_at = utc_now_iso()
        failed = {
            "schemaVersion": 1,
            "environment": "development",
            "authority": "evaluation_only",
            "status": "interrupted" if interrupted else "failed",
            "campaignReady": False,
            "runID": run_id,
            "requestedAt": requested_at,
            "dataUpdatedAt": updated_at,
            "failedAt": updated_at,
            "reasonCode": (
                f"prod_dev_evaluation_interrupted:{exc}"
                if interrupted
                else f"prod_dev_evaluation_failed:{type(exc).__name__}"
            ),
            "detail": str(exc),
        }
        write_evaluation_status_event(
            status_json=status_json,
            history_jsonl=history_jsonl,
            event=failed,
        )
        raise


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
    previous_handlers = {
        signum: signal.signal(signum, _raise_interrupted)
        for signum in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        summary = run_evaluation(
            prod_archive_zarr=args.prod_archive_zarr,
            dev_archive_zarr=args.dev_archive_zarr,
            power_zarr=args.power_zarr,
            output_zarr=args.output_zarr,
            status_json=args.status_json,
            history_jsonl=args.history_jsonl,
            bootstrap_samples=args.bootstrap_samples,
        )
    finally:
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)
    print(
        "Paired prod/dev power evidence: "
        f"status={summary['status']} rows={summary['pairedRows']} "
        f"cycles={summary['pairedIndependentCycles']}"
    )


if __name__ == "__main__":
    main()
