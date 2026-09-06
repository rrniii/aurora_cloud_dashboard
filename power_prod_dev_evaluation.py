"""Exact-intersection evidence for production versus development power forecasts.

The evaluator is deliberately read-only with respect to both forecast trees.
It pairs only identical issue/cycle/valid-time rows, rejects SOC-anchor or lead
mismatches, deduplicates cached/retried source cycles, and materialises a
separate development evidence product.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from generate_power_soc_forecast import _atomic_write_zarr, _write_state
from power_v12_hybrid import LEAD_BUCKETS, stable_json_digest, utc_now_iso


FORECAST_FIELDS = {
    "soc": "BatterySOCForecast",
    "solar": "ForecastSolarWatts",
    "load": "ForecastLoadWatts",
}

POWER_OBSERVATION_FIELDS = (
    "BatterySOC",
    "SolarWatts_East",
    "SolarWatts_South",
    "SolarWatts_West",
    "ACOutputWatts",
    "DCInverterWatts",
)
OBSERVATION_MATCH_TOLERANCE = pd.Timedelta(minutes=10)


def _issue_text(archive: xr.Dataset, name: str, issue_count: int) -> np.ndarray:
    if name not in archive:
        return np.full(issue_count, "", dtype=object)
    return np.asarray(archive[name].values, dtype=str).reshape(-1)


def _normalise_pair_key_text(values: pd.Series) -> pd.Series:
    """Return canonical non-null text without inventing pairing provenance."""

    normalised = values.astype(str).str.strip()
    return normalised.mask(normalised.str.lower().isin({"", "nan", "none", "nat"}), "")


def _empty_evidence(
    *,
    status: str = "no_exact_intersection",
    rejection_counts: dict[str, int] | None = None,
    candidate_rows: int = 0,
) -> xr.Dataset:
    counts = dict(rejection_counts or {})
    return xr.Dataset(
        coords={"record": np.array([], dtype=np.int64)},
        attrs={
            "power_prod_dev_paired_evidence": "true",
            "status": status,
            "generated_at_utc": utc_now_iso(),
            "candidate_base_intersection_rows": int(candidate_rows),
            "mismatched_rows_rejected": int(counts.get("total", 0)),
            "rejection_counts_json": json.dumps(counts, sort_keys=True),
        },
    )


def _archive_rows(archive: xr.Dataset, *, label: str) -> pd.DataFrame:
    required = {"ForecastValidTime", "ForecastLeadHours", "issue_time"}
    if not required.issubset(set(archive.variables) | set(archive.coords)):
        return pd.DataFrame()
    issue_count = int(archive.sizes.get("issue_time", 0))
    steps = int(archive.sizes.get("forecast_step", 0))
    if issue_count == 0 or steps == 0:
        return pd.DataFrame()
    issue_values = pd.DatetimeIndex(archive["issue_time"].values)
    issues = pd.DatetimeIndex(np.repeat(issue_values.values, steps))
    valid = pd.DatetimeIndex(np.asarray(archive["ForecastValidTime"].values).reshape(-1))
    leads = np.asarray(archive["ForecastLeadHours"].values, dtype=np.float64).reshape(-1)
    if "ECMWFCycleTime" in archive:
        cycle_issue = pd.DatetimeIndex(archive["ECMWFCycleTime"].values)
    else:
        cycle_issue = issue_values.floor("3h")
    cycles = pd.DatetimeIndex(np.repeat(cycle_issue.values, steps))
    source_cycles = np.repeat(_issue_text(archive, "SourceCycleSetID", issue_count), steps)
    source_cycles = np.asarray([str(value).strip() for value in source_cycles], dtype=object)
    anchors = np.full(issue_count, np.nan, dtype=np.float64)
    if "SOCAuthoringAnchorPct" in archive:
        anchors = np.asarray(
            pd.to_numeric(
                np.asarray(archive["SOCAuthoringAnchorPct"].values).reshape(-1),
                errors="coerce",
            ),
            dtype=np.float64,
        )
    frame: dict[str, Any] = {
        "issue_time": issues,
        "cycle_time": cycles,
        "source_cycle_set_id": source_cycles,
        "valid_time": valid,
        "lead_hours": leads,
        "soc_anchor_pct": np.repeat(anchors, steps),
        "forecast_system_version": np.repeat(
            _issue_text(archive, "ForecastSystemVersion", issue_count), steps
        ),
        "forecast_model_contract_id": np.repeat(
            _issue_text(archive, "ForecastModelContractID", issue_count), steps
        ),
        "forecast_identity_id": np.repeat(
            _issue_text(archive, "ForecastIdentityID", issue_count), steps
        ),
        "feature_set_version": np.repeat(
            _issue_text(archive, "FeatureSetVersion", issue_count), steps
        ),
        "forecast_code_revision": np.repeat(
            _issue_text(archive, "ForecastCodeRevision", issue_count), steps
        ),
        "load_mode": np.repeat(_issue_text(archive, "LoadMode", issue_count), steps),
        "clearness_index": (
            np.asarray(archive["ECMWFClearnessIndex"].values, dtype=np.float64).reshape(-1)
            if "ECMWFClearnessIndex" in archive
            else np.full(issue_count * steps, np.nan, dtype=np.float64)
        ),
    }
    for short, field in FORECAST_FIELDS.items():
        frame[f"{label}_{short}"] = (
            np.asarray(archive[field].values, dtype=np.float64).reshape(-1)
            if field in archive
            else np.full(issue_count * steps, np.nan, dtype=np.float64)
        )
    out = pd.DataFrame(frame)
    return out.loc[
        ~out["issue_time"].isna()
        & ~out["cycle_time"].isna()
        & ~out["valid_time"].isna()
        & np.isfinite(out["lead_hours"])
    ].copy()


def _observations(power: xr.Dataset | None) -> dict[str, pd.Series]:
    if power is None or "time" not in power.coords:
        return {}
    index = pd.DatetimeIndex(power["time"].values)
    values: dict[str, pd.Series] = {}
    if "BatterySOC" in power:
        values["soc"] = pd.Series(np.asarray(power["BatterySOC"].values, dtype=float), index=index)
    solar_fields = [
        name
        for name in ("SolarWatts_East", "SolarWatts_South", "SolarWatts_West")
        if name in power
    ]
    if solar_fields:
        values["solar"] = pd.DataFrame(
            {name: np.asarray(power[name].values, dtype=float) for name in solar_fields},
            index=index,
        ).sum(axis=1, min_count=1)
    load_fields = [name for name in ("ACOutputWatts", "DCInverterWatts") if name in power]
    if load_fields:
        values["load"] = pd.DataFrame(
            {name: np.asarray(power[name].values, dtype=float) for name in load_fields},
            index=index,
        ).sum(axis=1, min_count=1)
    return values


def paired_observation_view(
    power: xr.Dataset,
    evidence: xr.Dataset,
    *,
    tolerance: pd.Timedelta = OBSERVATION_MATCH_TOLERANCE,
) -> xr.Dataset:
    """Return the lazy APS samples sufficient to score every paired row.

    The raw APS store contains millions of one-second samples.  Paired scoring
    needs only the nearest sample (within the evaluator tolerance) for each
    exact valid time.  Resolve those positions from the time coordinate, then
    retain only the six physical observation fields before any data variable
    is loaded.  This is mathematically equivalent to reindexing the complete
    history because every globally nearest in-tolerance sample is retained.
    """

    if "time" not in power.coords:
        raise ValueError("APS power dataset has no time coordinate")
    fields = [name for name in POWER_OBSERVATION_FIELDS if name in power]
    source = power[fields]
    if evidence.sizes.get("record", 0) == 0 or "valid_time" not in evidence:
        return source.isel(time=slice(0, 0))
    times = pd.DatetimeIndex(power["time"].values)
    if times.hasnans:
        raise ValueError("APS power time coordinate contains invalid timestamps")
    if not times.is_monotonic_increasing or times.has_duplicates:
        raise ValueError("APS power time coordinate must be unique and monotonic")
    targets = pd.DatetimeIndex(evidence["valid_time"].values).dropna().unique().sort_values()
    if not len(targets):
        return source.isel(time=slice(0, 0))
    tolerance = pd.Timedelta(tolerance)
    if tolerance < pd.Timedelta(0):
        raise ValueError("Observation matching tolerance cannot be negative")
    positions = times.get_indexer(targets, method="nearest", tolerance=tolerance)
    positions = np.unique(positions[positions >= 0])
    selected = (
        source.isel(time=positions)
        if len(positions)
        else source.isel(time=slice(0, 0))
    )
    selected.attrs = dict(selected.attrs)
    selected.attrs.update(
        {
            "paired_observation_selection": "nearest_unique_valid_times",
            "paired_observation_match_tolerance_seconds": float(
                tolerance / pd.Timedelta(seconds=1)
            ),
            "paired_observation_target_start_utc": pd.Timestamp(targets.min()).isoformat(),
            "paired_observation_target_end_utc": pd.Timestamp(targets.max()).isoformat(),
            "paired_observation_target_count": int(len(targets)),
            "paired_observation_sample_count": int(len(positions)),
        }
    )
    return selected


def attach_paired_observations(
    evidence: xr.Dataset,
    power: xr.Dataset | None,
    *,
    tolerance: pd.Timedelta = OBSERVATION_MATCH_TOLERANCE,
) -> xr.Dataset:
    """Attach retrospective observations without changing the paired cohort."""

    if power is None or evidence.sizes.get("record", 0) == 0:
        return evidence
    paired = evidence.copy()
    paired.attrs.update(
        {
            name: value
            for name, value in power.attrs.items()
            if name.startswith("paired_observation_")
        }
    )
    targets = pd.DatetimeIndex(paired["valid_time"].values)
    for name, series in _observations(power).items():
        paired[f"observed_{name}"] = (
            ("record",),
            series.reindex(
                targets,
                method="nearest",
                tolerance=pd.Timedelta(tolerance),
            ).to_numpy(dtype=np.float64),
        )
    return paired


def build_exact_intersection_evidence(
    production: xr.Dataset,
    development: xr.Dataset,
    *,
    power: xr.Dataset | None = None,
    anchor_tolerance_pct: float = 0.05,
    lead_tolerance_hours: float = 1.0e-4,
) -> xr.Dataset:
    prod = _archive_rows(production, label="production")
    dev = _archive_rows(development, label="development")
    if prod.empty or dev.empty:
        return _empty_evidence()
    base_keys = ["issue_time", "cycle_time", "valid_time"]
    strict_keys = [*base_keys, "source_cycle_set_id", "load_mode"]
    for frame in (prod, dev):
        frame["source_cycle_set_id"] = _normalise_pair_key_text(
            frame["source_cycle_set_id"]
        )
        frame["load_mode"] = _normalise_pair_key_text(frame["load_mode"])
    prod = prod.sort_values(strict_keys).drop_duplicates(strict_keys, keep="last")
    dev = dev.sort_values(strict_keys).drop_duplicates(strict_keys, keep="last")
    paired = prod.merge(
        dev,
        on=base_keys,
        how="inner",
        suffixes=("_production", "_development"),
    )
    if paired.empty:
        return _empty_evidence(
            rejection_counts={"no_common_issue_cycle_valid_time": 1, "total": 0}
        )
    source_production = _normalise_pair_key_text(paired["source_cycle_set_id_production"])
    source_development = _normalise_pair_key_text(paired["source_cycle_set_id_development"])
    source_known = source_production.ne("") & source_development.ne("")
    source_ok = source_known & source_production.eq(source_development)
    mode_production = _normalise_pair_key_text(paired["load_mode_production"])
    mode_development = _normalise_pair_key_text(paired["load_mode_development"])
    mode_known = mode_production.ne("") & mode_development.ne("")
    mode_ok = mode_known & mode_production.eq(mode_development)
    lead_ok = np.isclose(
        paired["lead_hours_production"],
        paired["lead_hours_development"],
        atol=float(lead_tolerance_hours),
        rtol=0.0,
    )
    prod_anchor = paired["soc_anchor_pct_production"].to_numpy(dtype=float)
    dev_anchor = paired["soc_anchor_pct_development"].to_numpy(dtype=float)
    anchor_known = np.isfinite(prod_anchor) & np.isfinite(dev_anchor)
    anchor_ok = anchor_known & np.isclose(
        prod_anchor, dev_anchor, atol=float(anchor_tolerance_pct), rtol=0.0
    )
    accepted = source_ok.to_numpy(dtype=bool) & mode_ok.to_numpy(dtype=bool) & lead_ok & anchor_ok
    rejection_counts = {
        "source_cycle_missing": int((~source_known).sum()),
        "source_cycle_mismatch": int((source_known & ~source_ok).sum()),
        "load_mode_missing": int((~mode_known).sum()),
        "load_mode_mismatch": int((mode_known & ~mode_ok).sum()),
        "soc_anchor_unknown": int(np.count_nonzero(~anchor_known)),
        "soc_anchor_mismatch": int(np.count_nonzero(anchor_known & ~anchor_ok)),
        "lead_mismatch": int(np.count_nonzero(~lead_ok)),
        "total": int(np.count_nonzero(~accepted)),
    }
    candidate_rows = int(len(paired))
    paired = paired.loc[accepted].copy()
    if paired.empty:
        return _empty_evidence(
            status="all_candidate_pairs_rejected",
            rejection_counts=rejection_counts,
            candidate_rows=candidate_rows,
        )
    paired["source_cycle_set_id"] = source_production.loc[paired.index]
    paired["load_mode"] = mode_production.loc[paired.index]
    clearness_production = paired["clearness_index_production"].to_numpy(dtype=float)
    clearness_development = paired["clearness_index_development"].to_numpy(dtype=float)
    clearness_known = np.isfinite(clearness_production) & np.isfinite(clearness_development)
    clearness_consistent = clearness_known & np.isclose(
        clearness_production,
        clearness_development,
        atol=0.02,
        rtol=0.0,
    )
    clearness = np.where(
        clearness_consistent,
        0.5 * (clearness_production + clearness_development),
        np.nan,
    )
    paired["cloud_regime"] = np.where(
        clearness >= 0.65,
        "clear",
        np.where(clearness <= 0.35, "cloudy", "transitional"),
    )
    paired.loc[~np.isfinite(clearness), "cloud_regime"] = ""
    # One source cycle contributes only one row at each valid time, even if an
    # old archive contains multiple cached anchors for that meteorological run.
    pre_deduplication_rows = int(len(paired))
    paired = paired.sort_values("issue_time").drop_duplicates(
        ["source_cycle_set_id", "load_mode", "cycle_time", "valid_time"],
        keep="last",
    )
    duplicate_rows = pre_deduplication_rows - int(len(paired))
    cohort_columns = (
        "forecast_system_version_production",
        "forecast_model_contract_id_production",
        "feature_set_version_production",
        "forecast_code_revision_production",
        "forecast_system_version_development",
        "forecast_model_contract_id_development",
        "feature_set_version_development",
        "forecast_code_revision_development",
    )
    paired["evaluation_cohort_id"] = [
        "prod-dev-cohort-v1-"
        + stable_json_digest(
            {column: str(row[column] or "") for column in cohort_columns}
        )[:20]
        for _, row in paired.iterrows()
    ]
    paired["lead_hours"] = paired["lead_hours_development"]
    output_columns = [
        "issue_time",
        "cycle_time",
        "valid_time",
        "lead_hours",
        "soc_anchor_pct_production",
        "soc_anchor_pct_development",
        "source_cycle_set_id_production",
        "source_cycle_set_id_development",
        "source_cycle_set_id",
        "forecast_system_version_production",
        "forecast_system_version_development",
        "forecast_model_contract_id_production",
        "forecast_model_contract_id_development",
        "forecast_identity_id_production",
        "forecast_identity_id_development",
        "feature_set_version_production",
        "feature_set_version_development",
        "forecast_code_revision_production",
        "forecast_code_revision_development",
        "evaluation_cohort_id",
        "load_mode_production",
        "load_mode_development",
        "load_mode",
        "clearness_index_production",
        "clearness_index_development",
        "cloud_regime",
        *(f"production_{name}" for name in FORECAST_FIELDS),
        *(f"development_{name}" for name in FORECAST_FIELDS),
    ]
    variables: dict[str, tuple[tuple[str], np.ndarray]] = {}
    for name in output_columns:
        values = paired[name]
        if name.endswith("_time"):
            array = pd.DatetimeIndex(values).to_numpy(dtype="datetime64[ns]")
        elif pd.api.types.is_numeric_dtype(values):
            array = values.to_numpy(dtype=np.float64)
        else:
            array = values.fillna("").astype(str).to_numpy(dtype="U512")
        variables[name] = (("record",), array)
    evidence = xr.Dataset(
        variables,
        coords={"record": np.arange(len(paired), dtype=np.int64)},
        attrs={
            "power_prod_dev_paired_evidence": "true",
            "status": "complete" if len(paired) else "no_exact_intersection",
            "generated_at_utc": utc_now_iso(),
            "anchor_tolerance_pct": float(anchor_tolerance_pct),
            "lead_tolerance_hours": float(lead_tolerance_hours),
            "candidate_base_intersection_rows": candidate_rows,
            "mismatched_rows_rejected": rejection_counts["total"],
            "rejection_counts_json": json.dumps(rejection_counts, sort_keys=True),
            "duplicate_pair_rows_discarded": duplicate_rows,
            "pair_key": (
                "issue_time+ecmwf_cycle_time+valid_time+source_cycle_set_id+load_mode;"
                "known_equal_soc_anchor;dedupe=source_cycle_set_id+load_mode+"
                "ecmwf_cycle_time+valid_time"
            ),
        },
    )
    return attach_paired_observations(evidence, power)


def _paired_metric(
    frame: pd.DataFrame,
    name: str,
    *,
    bootstrap_samples: int,
    diversity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    columns = [f"production_{name}", f"development_{name}", f"observed_{name}"]
    if any(column not in frame for column in columns):
        return {
            "status": "observations_unavailable",
            "campaignReady": False,
            "samples": 0,
            "cycles": 0,
        }
    rows = frame.dropna(subset=columns).copy()
    if rows.empty:
        return {
            "status": "insufficient_evidence",
            "campaignReady": False,
            "samples": 0,
            "cycles": 0,
        }
    prod_error = rows[columns[0]].to_numpy(dtype=float) - rows[columns[2]].to_numpy(dtype=float)
    dev_error = rows[columns[1]].to_numpy(dtype=float) - rows[columns[2]].to_numpy(dtype=float)
    prod_mae = float(np.mean(np.abs(prod_error)))
    dev_mae = float(np.mean(np.abs(dev_error)))
    cycles = _normalise_pair_key_text(rows["source_cycle_set_id"]).unique()
    improvements: list[float] = []
    if len(cycles) >= 2 and bootstrap_samples > 0:
        seed = int(stable_json_digest({"metric": name, "cycles": [str(v) for v in cycles]})[:16], 16)
        rng = np.random.default_rng(seed)
        for _ in range(int(bootstrap_samples)):
            sampled = rng.choice(cycles, size=len(cycles), replace=True)
            blocks = [rows.loc[rows["source_cycle_set_id"] == cycle] for cycle in sampled]
            sample = pd.concat(blocks, ignore_index=True)
            p = np.mean(np.abs(sample[columns[0]] - sample[columns[2]]))
            d = np.mean(np.abs(sample[columns[1]] - sample[columns[2]]))
            if p > 0:
                improvements.append(float(1.0 - d / p))
    utc_days = int(pd.DatetimeIndex(rows["issue_time"]).floor("D").nunique())
    sample_count_ready = len(cycles) >= 30 and utc_days >= 10
    diversity_ready = bool((diversity or {}).get("campaignReady", False))
    campaign_ready = sample_count_ready and diversity_ready
    return {
        "status": "campaign_evidence" if campaign_ready else "diagnostic_sparse",
        "campaignReady": campaign_ready,
        "requiredIndependentCycles": 30,
        "requiredUTCDays": 10,
        "sampleCountReady": sample_count_ready,
        "diversityReady": diversity_ready,
        "samples": int(len(rows)),
        "cycles": int(len(cycles)),
        "utcDays": utc_days,
        "productionMAE": prod_mae,
        "developmentMAE": dev_mae,
        "developmentBias": float(np.mean(dev_error)),
        "productionBias": float(np.mean(prod_error)),
        "developmentMAEImprovementFraction": float(1.0 - dev_mae / prod_mae)
        if prod_mae > 0
        else None,
        "improvement95CI": (
            [float(value) for value in np.quantile(improvements, (0.025, 0.975))]
            if improvements
            else None
        ),
    }


def _campaign_diversity(rows: pd.DataFrame) -> dict[str, Any]:
    """Require explicit clear/cloudy metadata and multiple operating states."""

    regimes = (
        set(_normalise_pair_key_text(rows["cloud_regime"])) - {""}
        if "cloud_regime" in rows
        else set()
    )
    modes = (
        set(_normalise_pair_key_text(rows["load_mode"])) - {""}
        if "load_mode" in rows
        else set()
    )
    modes = {
        mode
        for mode in modes
        if mode.lower() not in {"unknown", "unavailable", "unspecified"}
    }
    clear_covered = "clear" in regimes
    cloudy_covered = "cloudy" in regimes
    multiple_states = len(modes) >= 2
    ready = clear_covered and cloudy_covered and multiple_states
    return {
        "status": "campaign_diversity_met" if ready else "diagnostic_incomplete_diversity",
        "campaignReady": ready,
        "cloudRegimeMetadataAvailable": bool(regimes),
        "clearCovered": clear_covered,
        "cloudyCovered": cloudy_covered,
        "requiredOperatingStateCount": 2,
        "operatingStateCount": len(modes),
        "operatingStates": sorted(modes),
        "cloudRegimes": sorted(regimes),
    }


def paired_score_surface(
    evidence: xr.Dataset,
    *,
    bootstrap_samples: int = 500,
) -> dict[str, Any]:
    if evidence.sizes.get("record", 0) == 0:
        return {
            "schemaVersion": 1,
            "status": "insufficient_evidence",
            "campaignReady": False,
            "dataUpdatedAt": utc_now_iso(),
            "pairedRows": 0,
            "pairedIndependentCycles": 0,
            "evidenceStatus": str(evidence.attrs.get("status", "no_exact_intersection")),
            "candidateBaseIntersectionRows": int(
                evidence.attrs.get("candidate_base_intersection_rows", 0)
            ),
            "campaignDiversity": _campaign_diversity(pd.DataFrame()),
            "leadBuckets": {},
            "rejectionCounts": json.loads(
                str(evidence.attrs.get("rejection_counts_json", "{}"))
            ),
            "mismatchedRowsRejected": int(
                evidence.attrs.get("mismatched_rows_rejected", 0)
            ),
        }
    frame = evidence.to_dataframe().reset_index(drop=True)
    frame["cycle_time"] = pd.to_datetime(frame["cycle_time"])
    frame["issue_time"] = pd.to_datetime(frame["issue_time"])
    if "evaluation_cohort_id" in frame:
        contract_columns = [
            column
            for column in (
                "forecast_model_contract_id_production",
                "forecast_model_contract_id_development",
            )
            if column in frame
        ]
        contract_known = pd.Series(False, index=frame.index)
        for column in contract_columns:
            contract_known |= _normalise_pair_key_text(frame[column]).ne("")
        active_candidates = frame.loc[contract_known]
        latest_row = (
            active_candidates if not active_candidates.empty else frame
        ).sort_values("issue_time").iloc[-1]
        active_cohort = str(latest_row["evaluation_cohort_id"])
        cohort_frame = frame.loc[frame["evaluation_cohort_id"].astype(str) == active_cohort].copy()
        cohort_inventory = {
            str(cohort): {
                "pairedRows": int(len(rows)),
                "pairedIndependentCycles": int(rows["source_cycle_set_id"].nunique()),
                "pairedUTCDays": int(rows["issue_time"].dt.floor("D").nunique()),
                "status": "active" if str(cohort) == active_cohort else "historical",
            }
            for cohort, rows in frame.groupby("evaluation_cohort_id", dropna=False)
        }
    else:
        active_cohort = "legacy-unpartitioned"
        cohort_frame = frame
        cohort_inventory = {
            active_cohort: {
                "pairedRows": int(len(frame)),
                "pairedIndependentCycles": int(frame["source_cycle_set_id"].nunique()),
                "pairedUTCDays": int(frame["issue_time"].dt.floor("D").nunique()),
                "status": "active",
            }
        }
    lead_buckets: dict[str, Any] = {}
    for label, start, end in LEAD_BUCKETS:
        rows = cohort_frame.loc[
            (cohort_frame["lead_hours"] >= start) & (cohort_frame["lead_hours"] < end)
        ]
        diversity = _campaign_diversity(rows)
        lead_buckets[label] = {
            **{
                name: _paired_metric(
                    rows,
                    name,
                    bootstrap_samples=bootstrap_samples,
                    diversity=diversity,
                )
                for name in FORECAST_FIELDS
            },
            "diversity": diversity,
            "campaignReady": False,
        }
        lead_buckets[label]["campaignReady"] = bool(
            lead_buckets[label]["soc"].get("campaignReady") is True
            and diversity["campaignReady"] is True
        )
    campaign_ready = all(
        bucket["campaignReady"] is True
        for bucket in lead_buckets.values()
    )
    return {
        "schemaVersion": 1,
        "status": "campaign_evidence" if campaign_ready else "diagnostic_only",
        "campaignReady": campaign_ready,
        "dataUpdatedAt": utc_now_iso(),
        "pairedRows": int(len(cohort_frame)),
        "pairedIndependentCycles": int(cohort_frame["source_cycle_set_id"].nunique()),
        "pairedUTCDays": int(cohort_frame["issue_time"].dt.floor("D").nunique()),
        "totalExactIntersectionRows": int(len(frame)),
        "evidenceStatus": str(evidence.attrs.get("status", "complete")),
        "candidateBaseIntersectionRows": int(
            evidence.attrs.get("candidate_base_intersection_rows", len(frame))
        ),
        "activeCohortID": active_cohort,
        "cohorts": cohort_inventory,
        "campaignDiversity": _campaign_diversity(cohort_frame),
        "leadBuckets": lead_buckets,
        "pairKey": str(evidence.attrs.get("pair_key", "")),
        "mismatchedRowsRejected": int(evidence.attrs.get("mismatched_rows_rejected", 0)),
        "rejectionCounts": json.loads(
            str(evidence.attrs.get("rejection_counts_json", "{}"))
        ),
        "duplicatePairRowsDiscarded": int(
            evidence.attrs.get("duplicate_pair_rows_discarded", 0)
        ),
    }


def write_paired_products(
    evidence: xr.Dataset,
    *,
    output_zarr: Path,
    status_json: Path,
    history_jsonl: Path,
    bootstrap_samples: int = 500,
    status_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _atomic_write_zarr(evidence, Path(output_zarr))
    summary = paired_score_surface(evidence, bootstrap_samples=bootstrap_samples)
    if status_context:
        summary = {**summary, **status_context}
    write_evaluation_status_event(
        status_json=Path(status_json),
        history_jsonl=Path(history_jsonl),
        event=summary,
    )
    return summary


def write_evaluation_status_event(
    *,
    status_json: Path,
    history_jsonl: Path,
    event: dict[str, Any],
) -> None:
    """Atomically replace current status and append an immutable run event."""

    status_json = Path(status_json)
    history_jsonl = Path(history_jsonl)
    _write_state(status_json, event)
    history_jsonl.parent.mkdir(parents=True, exist_ok=True)
    history_event = {**event, "eventDigest": stable_json_digest(event)}
    with history_jsonl.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(history_event, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
