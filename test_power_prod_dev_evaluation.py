from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import xarray as xr

from power_prod_dev_evaluation import (
    build_exact_intersection_evidence,
    paired_score_surface,
    write_paired_products,
)


def _archive(
    soc_offset: float,
    *,
    source_cycle_set_id: str = "cycle-a",
    load_mode: str = "DC-Only",
    anchor: float = 72.0,
) -> xr.Dataset:
    issues = pd.DatetimeIndex(["2026-09-01T00:00:00", "2026-09-01T01:00:00"])
    valid = np.asarray(
        [
            ["2026-09-01T03:00:00", "2026-09-01T06:00:00"],
            ["2026-09-01T03:00:00", "2026-09-01T06:00:00"],
        ],
        dtype="datetime64[ns]",
    )
    observed = np.asarray([70.0, 68.0])
    soc = np.tile(observed + soc_offset, (2, 1))
    return xr.Dataset(
        {
            "ForecastValidTime": (("issue_time", "forecast_step"), valid),
            "ForecastLeadHours": (("issue_time", "forecast_step"), np.tile([3.0, 6.0], (2, 1))),
            "ECMWFCycleTime": (("issue_time",), np.asarray([issues[0], issues[0]], dtype="datetime64[ns]")),
            "SOCAuthoringAnchorPct": (("issue_time",), np.asarray([anchor, anchor]).astype(str)),
            "SourceCycleSetID": (("issue_time",), np.asarray([source_cycle_set_id] * 2)),
            "ForecastSystemVersion": (("issue_time",), np.asarray(["test"] * 2)),
            "ForecastModelContractID": (("issue_time",), np.asarray(["contract"] * 2)),
            "ForecastIdentityID": (("issue_time",), np.asarray(["identity"] * 2)),
            "LoadMode": (("issue_time",), np.asarray([load_mode] * 2)),
            "BatterySOCForecast": (("issue_time", "forecast_step"), soc),
            "ForecastSolarWatts": (("issue_time", "forecast_step"), np.zeros((2, 2))),
            "ForecastLoadWatts": (("issue_time", "forecast_step"), np.full((2, 2), 100.0)),
        },
        coords={"issue_time": issues, "forecast_step": np.arange(2)},
    )


def test_exact_intersection_deduplicates_cached_cycle_rows_and_scores() -> None:
    times = pd.DatetimeIndex(["2026-09-01T03:00:00", "2026-09-01T06:00:00"])
    power = xr.Dataset(
        {
            "BatterySOC": (("time",), [70.0, 68.0]),
            "SolarWatts_East": (("time",), [0.0, 0.0]),
            "SolarWatts_South": (("time",), [0.0, 0.0]),
            "SolarWatts_West": (("time",), [0.0, 0.0]),
            "ACOutputWatts": (("time",), [80.0, 80.0]),
            "DCInverterWatts": (("time",), [20.0, 20.0]),
        },
        coords={"time": times},
    )

    evidence = build_exact_intersection_evidence(_archive(2.0), _archive(1.0), power=power)
    summary = paired_score_surface(evidence, bootstrap_samples=20)

    assert evidence.sizes["record"] == 2
    assert summary["pairedIndependentCycles"] == 1
    soc = summary["leadBuckets"]["0_6h"]["soc"]
    assert soc["productionMAE"] == 2.0
    assert soc["developmentMAE"] == 1.0
    assert soc["developmentMAEImprovementFraction"] == 0.5


def test_paired_products_are_separate_and_append_history() -> None:
    evidence = build_exact_intersection_evidence(_archive(2.0), _archive(1.0))
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        first = write_paired_products(
            evidence,
            output_zarr=root / "evidence.zarr",
            status_json=root / "status.json",
            history_jsonl=root / "history.jsonl",
            bootstrap_samples=0,
        )
        write_paired_products(
            evidence,
            output_zarr=root / "evidence.zarr",
            status_json=root / "status.json",
            history_jsonl=root / "history.jsonl",
            bootstrap_samples=0,
        )
        history = (root / "history.jsonl").read_text(encoding="utf-8").splitlines()

    assert first["pairedRows"] == 2
    assert len(history) == 2


def test_exact_intersection_rejects_source_mode_and_unknown_anchor() -> None:
    source_mismatch = build_exact_intersection_evidence(
        _archive(2.0, source_cycle_set_id="cycle-a"),
        _archive(1.0, source_cycle_set_id="cycle-b"),
    )
    assert source_mismatch.sizes["record"] == 0
    assert source_mismatch.attrs["status"] == "all_candidate_pairs_rejected"
    assert '"source_cycle_mismatch": 4' in source_mismatch.attrs["rejection_counts_json"]

    mode_mismatch = build_exact_intersection_evidence(
        _archive(2.0, load_mode="DC-Only"),
        _archive(1.0, load_mode="CL61"),
    )
    assert mode_mismatch.sizes["record"] == 0
    assert '"load_mode_mismatch": 4' in mode_mismatch.attrs["rejection_counts_json"]

    unknown_anchor = build_exact_intersection_evidence(
        _archive(2.0, anchor=np.nan),
        _archive(1.0, anchor=np.nan),
    )
    assert unknown_anchor.sizes["record"] == 0
    assert '"soc_anchor_unknown": 4' in unknown_anchor.attrs["rejection_counts_json"]


def _campaign_evidence(*, include_cloud: bool = True, multiple_modes: bool = True) -> xr.Dataset:
    issues = pd.date_range("2026-06-01", periods=30, freq="8h")
    leads = np.asarray([3.0, 12.0, 30.0, 60.0])
    issue_values = np.repeat(issues.to_numpy(dtype="datetime64[ns]"), len(leads))
    count = len(issue_values)
    variables: dict[str, tuple[tuple[str], np.ndarray]] = {
        "issue_time": (("record",), issue_values),
        "cycle_time": (("record",), issue_values),
        "valid_time": (
            ("record",),
            np.asarray(
                [issue + pd.Timedelta(hours=float(lead)) for issue in issues for lead in leads],
                dtype="datetime64[ns]",
            ),
        ),
        "lead_hours": (("record",), np.tile(leads, len(issues))),
        "source_cycle_set_id": (
            ("record",),
            np.repeat(np.asarray([f"cycle-{index}" for index in range(len(issues))]), len(leads)),
        ),
        "load_mode": (
            ("record",),
            np.repeat(
                np.asarray(
                    [
                        "DC-Only"
                        if not multiple_modes or index % 2 == 0
                        else "CL61"
                        for index in range(len(issues))
                    ]
                ),
                len(leads),
            ),
        ),
        "evaluation_cohort_id": (("record",), np.asarray(["cohort-current"] * count)),
        "forecast_model_contract_id_production": (("record",), np.asarray(["prod-v1"] * count)),
        "forecast_model_contract_id_development": (("record",), np.asarray(["dev-v1"] * count)),
    }
    for name in ("soc", "solar", "load"):
        observed = np.full(count, 50.0)
        variables[f"observed_{name}"] = (("record",), observed)
        variables[f"production_{name}"] = (("record",), observed + 2.0)
        variables[f"development_{name}"] = (("record",), observed + 1.0)
    if include_cloud:
        variables["cloud_regime"] = (
            ("record",),
            np.repeat(
                np.asarray(["clear" if index % 2 == 0 else "cloudy" for index in range(len(issues))]),
                len(leads),
            ),
        )
    return xr.Dataset(variables, coords={"record": np.arange(count)})


def test_campaign_readiness_requires_cloud_and_operating_state_diversity() -> None:
    ready = paired_score_surface(_campaign_evidence(), bootstrap_samples=0)
    no_cloud = paired_score_surface(
        _campaign_evidence(include_cloud=False), bootstrap_samples=0
    )
    one_mode = paired_score_surface(
        _campaign_evidence(multiple_modes=False), bootstrap_samples=0
    )

    assert ready["campaignReady"] is True
    assert ready["campaignDiversity"]["clearCovered"] is True
    assert ready["campaignDiversity"]["cloudyCovered"] is True
    assert no_cloud["campaignReady"] is False
    assert no_cloud["status"] == "diagnostic_only"
    assert one_mode["campaignReady"] is False
