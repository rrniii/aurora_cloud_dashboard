"""Issue-time-safe, source-separated public solar ablations for v12.

This is intentionally a consumer of already-normalised local site extracts.
It never downloads public grids and never creates a pooled ECMWF/GFS/ICON/GEFS
forecast.  Every accepted extract produces its own physical-PV SOC candidate,
immutable pair bundle and campaign-evidence surface under the v12 candidate
tree.
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Mapping

import numpy as np
import pandas as pd
import xarray as xr

from generate_power_soc_forecast import (
    _aggregate_physical_soc_forecast,
    _atomic_write_zarr,
    _write_state,
    append_forecast_archive,
    apply_forecast_identity,
    forecast_publication_signature,
    integrate_soc_from_available_solar,
)
from power_battery_model import BatteryModel
from power_issue_time_features import (
    PUBLIC_SOURCES,
    public_source_manifest_record,
    site_extract_sha256,
)
from power_solar_model import (
    PhysicalSolarConfig,
    build_physical_solar_forecast_frames,
    physical_solar_config_digest,
)
from power_v12_hybrid import (
    build_campaign_evidence,
    campaign_score_surfaces,
    evaluation_contract_from_forecast,
    promotion_gate_review,
    stable_json_digest,
)


PUBLIC_SOURCE_ABLATION_VERSION = "separate_site_physical_pv_v1"
PUBLIC_SOURCE_FEATURE_SET_VERSION = "issue_safe_public_source_ablation_v1"
_MAX_SITE_STEPS = 160
_MAX_SITE_MEMBERS = 100


def _utc_naive(value: object) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    if parsed.tz is not None:
        return parsed.tz_convert("UTC").tz_localize(None)
    return parsed


def _text(value: object, fallback: str = "") -> str:
    return str(value if value is not None else fallback).strip() or fallback


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, object]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("source manifest is not an object")
    return raw


def _manifest_digest(path: Path) -> str:
    raw = _read_json(path)
    return "sha256:" + stable_json_digest(raw)


def _extract_path(root: Path, record: Mapping[str, object]) -> Path:
    relative = Path(str(record["site_extract_path"]))
    return (Path(root) / relative).resolve()


def _open_site_extract(
    root: Path,
    record: Mapping[str, object],
    *,
    expected_times: pd.DatetimeIndex,
    expected_latitude: float,
    expected_longitude: float,
) -> tuple[pd.Series, str]:
    """Read one bounded deterministic or internal-member-mean site trace."""

    path = _extract_path(root, record)
    expected_checksum = str(record["site_extract_sha256"]).removeprefix("sha256:")
    if site_extract_sha256(path) != expected_checksum:
        raise ValueError("Public source extract changed after manifest validation")
    extract_format = str(record["site_extract_format"])
    if extract_format == "zarr":
        with xr.open_zarr(path, chunks={}) as opened:
            source = opened.load()
    elif extract_format == "netcdf":
        with xr.open_dataset(path) as opened:
            source = opened.load()
    else:  # Manifest validation protects this branch; retain fail-closed semantics.
        raise ValueError("Unsupported source extract format")
    variable = str(record["irradiance_variable"])
    if variable not in source or "time" not in source.coords:
        raise ValueError("Public source extract is missing irradiance or time")
    extra_dims = set(source.sizes) - {"time", "member"}
    if extra_dims:
        raise ValueError("Public source extract retains non-site dimensions")
    if source.sizes.get("time", 0) > _MAX_SITE_STEPS:
        raise ValueError("Public source extract exceeds bounded time horizon")
    if source.sizes.get("member", 1) > _MAX_SITE_MEMBERS:
        raise ValueError("Public source extract exceeds bounded member count")
    values = source[variable]
    if values.dims == ("time",):
        result = np.asarray(values.values, dtype=np.float64)
        member_handling = "deterministic_site_extract"
    elif values.dims == ("member", "time"):
        raw = np.asarray(values.values, dtype=np.float64)
        if raw.shape[0] < 2:
            raise ValueError("Public internal ensemble needs at least two members")
        result = np.nanmean(raw, axis=0)
        member_handling = "issue_time_available_internal_member_mean_not_pooled"
    elif values.dims == ("time", "member"):
        raw = np.asarray(values.transpose("member", "time").values, dtype=np.float64)
        if raw.shape[0] < 2:
            raise ValueError("Public internal ensemble needs at least two members")
        result = np.nanmean(raw, axis=0)
        member_handling = "issue_time_available_internal_member_mean_not_pooled"
    else:
        raise ValueError("Public source irradiance must be time or member/time only")
    times = pd.DatetimeIndex(source["time"].values)
    if times.tz is not None:
        times = times.tz_convert("UTC").tz_localize(None)
    if not np.array_equal(times.values, expected_times.values):
        raise ValueError("Public source extract time grid is not the exact paired baseline grid")
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError("Public source irradiance must be finite and non-negative")
    source_latitude = float(record["site_latitude"])
    source_longitude = float(record["site_longitude"])
    if not (
        np.isclose(source_latitude, expected_latitude, atol=1.0e-4)
        and np.isclose(source_longitude, expected_longitude, atol=1.0e-4)
    ):
        raise ValueError("Public source manifest coordinates do not match the baseline site")
    return pd.Series(result, index=times), member_handling


def _baseline_times(baseline: xr.Dataset) -> tuple[pd.DatetimeIndex, pd.Timestamp, float]:
    if "time" not in baseline.coords:
        raise ValueError("Baseline forecast is missing its time grid")
    times = pd.DatetimeIndex(baseline["time"].values)
    if len(times) < 2 or times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError("Baseline forecast time grid is invalid")
    try:
        initial_soc = float(baseline.attrs["initial_soc_pct"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Baseline forecast has no finite SOC anchor") from exc
    if not np.isfinite(initial_soc):
        raise ValueError("Baseline forecast has no finite SOC anchor")
    return times, _utc_naive(baseline.attrs["initial_soc_time"]), initial_soc


def _source_identity(
    *,
    baseline: xr.Dataset,
    source: str,
    record: Mapping[str, object],
    manifest_digest: str,
    code_revision: str,
) -> dict[str, str]:
    feature_digest = stable_json_digest(
        {
            "schema": 1,
            "version": PUBLIC_SOURCE_ABLATION_VERSION,
            "source": source,
            "physical_config": _text(baseline.attrs.get("solar_physical_config_sha256")),
            "manifest_digest": manifest_digest,
            "baseline_control_contract_id": _text(
                baseline.attrs.get("forecast_model_contract_id")
            ),
        }
    )
    source_cycle = _utc_naive(record["source_cycle_time_utc"])
    source_checksum = _text(record["site_extract_sha256"])
    return {
        "forecast_model_name": f"aps_soc_energy_balance_v12_{source.lower()}_ablation",
        "forecast_model_version": "12",
        "forecast_model_status": "candidate",
        "forecast_system_version": "power-v12-public-source-ablation",
        "feature_set_version": PUBLIC_SOURCE_FEATURE_SET_VERSION,
        "feature_set_digest": feature_digest,
        "training_cutoff_utc": _text(baseline.attrs.get("initial_soc_time")),
        "forecast_code_revision": code_revision,
        "source_cycle_set_id": f"{source.lower()}:{source_cycle.isoformat()}:sha256:{source_checksum[-20:]}",
        "source_manifest_digest": manifest_digest,
        "degraded_mode_code": "hardware_geometry_unverified+public_source_ablation",
        "candidate_lane": f"E_public_source_{source.lower()}",
        "local_feature_contract_id": "public-source-site-extract-v2",
        "baseline_control_contract_id": _text(baseline.attrs.get("forecast_model_contract_id")),
        "baseline_control_system_version": _text(
            baseline.attrs.get("forecast_system_version"), "unversioned_control"
        ),
        "source_availability_code": f"{source.lower()}=issue_time_site_extract_available_not_pooled",
    }


def build_public_source_ablation_forecast(
    baseline: xr.Dataset,
    irradiance: pd.Series,
    *,
    source: str,
    record: Mapping[str, object],
    manifest_digest: str,
    configuration: PhysicalSolarConfig,
    latitude: float,
    longitude: float,
    code_revision: str,
    member_handling: str,
) -> xr.Dataset:
    """Run one source independently through the physical PV/battery model."""

    times, issue_time, initial_soc = _baseline_times(baseline)
    if not np.array_equal(pd.DatetimeIndex(irradiance.index).values, times.values):
        raise ValueError("Public source forcing does not cover paired output grid")
    for name in ("ForecastLoadWatts", "BatterySOCForecast"):
        if name not in baseline:
            raise ValueError(f"Baseline forecast is missing {name}")
    model = BatteryModel.from_attrs(baseline.attrs)
    intervals, substeps, metadata = build_physical_solar_forecast_frames(
        irradiance,
        latitude=latitude,
        longitude=longitude,
        config=configuration,
        forecast_start_time=issue_time,
    )
    integrated = integrate_soc_from_available_solar(
        initial_soc=initial_soc,
        initial_time=issue_time,
        available_solar_w=substeps["ForecastPVAvailableWatts"],
        irradiance=substeps["ECMWFSolarIrradiance"],
        load_w=pd.Series(np.asarray(baseline["ForecastLoadWatts"].values, dtype=np.float64), index=times),
        capacity_kwh=model.usable_capacity_kwh,
        battery_model=model,
    )
    forecast = _aggregate_physical_soc_forecast(integrated, intervals, irradiance, times)
    if forecast["BatterySOCForecast"].isna().any():
        raise ValueError("Public source physical SOC does not cover paired output grid")
    output = baseline.copy(deep=True)
    for name in forecast.columns:
        output[name] = (("time",), forecast[name].to_numpy(dtype=np.float32))
    aligned_physical = intervals.reindex(times)
    for name in aligned_physical:
        if name not in output:
            output[name] = (("time",), aligned_physical[name].to_numpy(dtype=np.float32))
    output["ForecastInputGlobalHorizontalIrradiance"] = (
        ("time",), irradiance.to_numpy(dtype=np.float32)
    )
    output.attrs = {str(key): str(value) for key, value in baseline.attrs.items()}
    output.attrs.update(
        {
            "power_soc_forecast_product": "true",
            "baseline_publication_signature": _text(baseline.attrs.get("publication_signature")),
            "baseline_ecmwf_cycle_time": _text(baseline.attrs.get("ecmwf_cycle_time")),
            "ecmwf_cycle_time": _text(record["source_cycle_time_utc"]),
            "public_source": source,
            "public_source_delivery_time_utc": _text(record["delivery_time_utc"]),
            "public_source_member_handling": member_handling,
            "public_source_irradiance_variable": _text(record["irradiance_variable"]),
            "solar_forcing_mode": "public_source_site_physical_available_pv",
            "solar_power_semantics": "available_dc_before_battery_acceptance",
            "solar_physical_config_sha256": physical_solar_config_digest(configuration),
            "solar_model_name": "three_array_poa_pv_v2",
            "solar_model_version": "2",
            "solar_model_status": "candidate",
            "solar_degradation_codes": _text(metadata.get("solar_degradation_codes"), "none"),
            "input_snapshot_id": _text(record["site_extract_sha256"]),
        }
    )
    pair_id = "public-source-pair-v1-" + stable_json_digest(
        {
            "baseline": output.attrs["baseline_publication_signature"],
            "source": source,
            "source_cycle": output.attrs["ecmwf_cycle_time"],
            "source_manifest_digest": manifest_digest,
            "source_input": output.attrs["input_snapshot_id"],
            "issue": output.attrs["initial_soc_time"],
        }
    )[:20]
    output.attrs["evaluation_pair_id"] = pair_id
    apply_forecast_identity(
        output,
        _source_identity(
            baseline=baseline,
            source=source,
            record=record,
            manifest_digest=manifest_digest,
            code_revision=code_revision,
        ),
    )
    output.attrs["publication_signature"] = forecast_publication_signature(output)
    return output


def _write_pair_bundle(
    root: Path,
    *,
    pair_id: str,
    baseline: xr.Dataset,
    candidate: xr.Dataset,
    manifest: Mapping[str, object],
) -> Path:
    signature = _text(candidate.attrs.get("publication_signature"))
    if not signature:
        raise ValueError("Public source candidate has no publication signature")
    family = Path(root) / "pairs" / pair_id
    bundle = family / signature
    complete = {
        "pair_status": "complete",
        "candidate_publication_signature": signature,
        **{str(key): value for key, value in manifest.items()},
    }
    if bundle.exists():
        if _read_json(bundle / "pair_manifest.json") != complete:
            raise RuntimeError("Existing immutable public-source pair bundle does not match")
        return bundle
    family.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".public-source-pair-", dir=family) as temporary:
        staging = Path(temporary)
        _atomic_write_zarr(baseline, staging / "baseline_forecast.zarr")
        _atomic_write_zarr(candidate, staging / "candidate_forecast.zarr")
        _write_state(staging / "pair_manifest.json", complete)
        staging.replace(bundle)
    return bundle


def run_public_source_ablations(
    *,
    candidate_root: Path,
    baseline: xr.Dataset,
    power_for_evidence: xr.Dataset,
    source_manifest_root: Path,
    configuration: PhysicalSolarConfig,
    latitude: float,
    longitude: float,
    code_revision: str,
) -> dict[str, dict[str, object]]:
    """Generate/score each separately available public source without pooling."""

    times, issue_time, _ = _baseline_times(baseline)
    results: dict[str, dict[str, object]] = {}
    source_manifest_root = Path(source_manifest_root)
    for source in PUBLIC_SOURCES:
        record = public_source_manifest_record(
            source, root=source_manifest_root, cutoff=issue_time
        )
        status = _text(record.get("status"))
        if status != "issue_time_available_pending_independent_ablation":
            results[source.lower()] = {"status": status}
            continue
        try:
            manifest_path = source_manifest_root / f"{source.lower()}.json"
            manifest_digest = _manifest_digest(manifest_path)
            irradiance, member_handling = _open_site_extract(
                source_manifest_root,
                record,
                expected_times=times,
                expected_latitude=latitude,
                expected_longitude=longitude,
            )
            candidate = build_public_source_ablation_forecast(
                baseline,
                irradiance,
                source=source,
                record=record,
                manifest_digest=manifest_digest,
                configuration=configuration,
                latitude=latitude,
                longitude=longitude,
                code_revision=code_revision,
                member_handling=member_handling,
            )
            # Re-validate source content immediately before exposing a candidate
            # writer or immutable pair bundle.
            verified = public_source_manifest_record(
                source, root=source_manifest_root, cutoff=issue_time
            )
            if (
                verified.get("status") != status
                or verified.get("site_extract_sha256") != record.get("site_extract_sha256")
            ):
                raise RuntimeError("Public source extract changed during ablation")
            contract = _text(candidate.attrs.get("forecast_model_contract_id"))
            root = Path(candidate_root) / "public_model_ablations" / source.lower() / "contracts" / contract
            forecast_path = root / "power_soc_forecast.zarr"
            archive_path = root / "power_soc_forecast_archive.zarr"
            _atomic_write_zarr(candidate, forecast_path)
            append_forecast_archive(candidate, archive_path)
            pair_id = _text(candidate.attrs.get("evaluation_pair_id"))
            _write_pair_bundle(
                root.parent.parent,
                pair_id=pair_id,
                baseline=baseline,
                candidate=candidate,
                manifest={
                    "schema_version": 1,
                    "evaluation_pair_id": pair_id,
                    "source": source,
                    "source_cycle_set_id": candidate.attrs["source_cycle_set_id"],
                    "source_manifest_digest": manifest_digest,
                    "input_snapshot_id": candidate.attrs["input_snapshot_id"],
                    "baseline_publication_signature": baseline.attrs.get("publication_signature", ""),
                    "forecast_model_contract_id": contract,
                    "forecast_identity_id": candidate.attrs.get("forecast_identity_id", ""),
                },
            )
            evidence = build_campaign_evidence(
                root.parent.parent / "pairs",
                power_for_evidence,
                lane=_text(candidate.attrs.get("candidate_lane")),
                evaluation_contract=evaluation_contract_from_forecast(candidate),
            )
            _atomic_write_zarr(evidence, root / "campaign_evidence.zarr")
            summary = campaign_score_surfaces(evidence)
            summary["promotion_gates"] = promotion_gate_review(evidence)
            summary["public_source"] = source
            summary["member_handling"] = member_handling
            _atomic_json(root / "evaluation_summary.json", summary)
            results[source.lower()] = {
                "status": "complete",
                "candidate_lane": candidate.attrs["candidate_lane"],
                "forecast_model_contract_id": contract,
                "forecast_identity_id": candidate.attrs["forecast_identity_id"],
                "source_manifest_digest": manifest_digest,
                "path": str(forecast_path),
                "member_handling": member_handling,
            }
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            results[source.lower()] = {
                "status": f"blocked_public_source_ablation_error:{type(exc).__name__}",
                "reason": str(exc),
            }
    return results
