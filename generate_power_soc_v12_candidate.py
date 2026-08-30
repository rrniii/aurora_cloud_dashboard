#!/usr/bin/env python3
"""Generate the isolated, evaluation-first v12 hybrid power candidate.

The runner is deliberately conservative.  It accepts only a full, independent
ECMWF baseline issue, snapshots its forcing and SOC anchor, and writes solely
under the candidate root.  Operational v10/v11 products, archives and
adaptive state are read-only inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import xarray as xr

from generate_power_soc_forecast import (
    DEFAULT_PHYSICAL_SOLAR_CONFIG_PATH,
    LEGACY_SOLAR_MODEL_NAME,
    PHYSICAL_SOLAR_MODEL_NAME,
    POWER_PDU_ZARR_PATH,
    POWER_ZARR_PATH,
    _atomic_write_zarr,
    _paths_overlap,
    _write_state,
    generate,
    validate_paired_candidate,
    validate_provider,
)
from generate_power_soc_physical_candidate import _baseline_seed_state
from power_solar_model import (
    load_physical_solar_config,
    physical_solar_config_digest,
    physical_solar_contract_id,
)
from power_issue_time_features import (
    IssueTimeFeatureSnapshot,
    build_issue_time_feature_snapshot,
)
from power_public_source_ablation import run_public_source_ablations
from power_v12_ensemble import (
    append_candidate_ensemble_archive,
    baseline_ensemble_signature,
    build_candidate_memberwise_ensemble,
    build_campaign_ensemble_evidence,
    campaign_ensemble_score_surfaces,
    ensemble_evaluation_contract_from_forecast,
    ensemble_promotion_gate,
    write_immutable_ensemble_pair_bundle,
)
from power_v12_hybrid import (
    build_campaign_evidence,
    campaign_score_surfaces,
    evaluation_contract_from_forecast,
    fit_bounded_load_residual,
    promotion_gate_review,
    stable_json_digest,
    utc_now_iso,
    v12_forecast_identity,
    V12_FEATURE_SET_VERSION,
    V12_POWER_HISTORY_DAYS,
)


BASELINE_FORECAST_ZARR_PATH = Path(
    os.environ.get(
        "AURORA_POWER_BASELINE_FORECAST_ZARR",
        "/data/aurora/dev-products/power/power_soc_forecast.zarr",
    )
)
BASELINE_ARCHIVE_ZARR_PATH = Path(
    os.environ.get(
        "AURORA_POWER_BASELINE_FORECAST_ARCHIVE_ZARR",
        "/data/aurora/dev-products/power/power_soc_forecast_archive.zarr",
    )
)
BASELINE_ENSEMBLE_ZARR_PATH = Path(
    os.environ.get(
        "AURORA_POWER_BASELINE_ENSEMBLE_ZARR",
        "/data/aurora/dev-products/power/power_soc_ensemble_forecast.zarr",
    )
)
CANDIDATE_ROOT = Path(
    os.environ.get(
        "AURORA_POWER_V12_CANDIDATE_ROOT",
        "/data/aurora/dev-products/power/candidates/v12",
    )
)
ASFS_LOGGER_ZARR_PATH = Path(
    os.environ.get(
        "ASFS_LOGGER_ZARR_PATH",
        "/data/aurora/products/asfs_logger/asfs_logger.zarr",
    )
)
UAS_MQTT_LOG_PATH = Path(
    os.environ.get("UAS_MQTT_LOG_PATH", "/project/aurora/raw/menapia/menapia_mqtt.log")
)
PUBLIC_SOURCE_MANIFEST_ROOT = Path(
    os.environ.get(
        "AURORA_POWER_PUBLIC_SOURCE_MANIFEST_ROOT",
        "/data/aurora/dev-products/power/public_model_inputs",
    )
)

LANE_PHYSICAL_SOLAR = "B_physical_solar"
LANE_LOAD_RESIDUAL = "C_load_residual"
LANE_HYBRID = "D_physical_solar_load_residual"
LANES = (LANE_PHYSICAL_SOLAR, LANE_LOAD_RESIDUAL, LANE_HYBRID)
POWER_HISTORY_FIELDS = (
    "BatterySOC",
    "SolarWatts_East",
    "SolarWatts_South",
    "SolarWatts_West",
    "BatteryWatts",
    "ACOutputWatts",
    "DCInverterWatts",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _public_source_manifest_root_digest(root: Path) -> str:
    """Track compact source manifests so a newly enrolled source is not skipped."""
    root = Path(root)
    if not root.exists():
        return "missing"
    digest = hashlib.sha256()
    found = False
    for path in sorted(root.glob("*.json")):
        if not path.is_file():
            continue
        found = True
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest() if found else "empty"


def _utc_naive(value: object) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    if parsed.tz is not None:
        return parsed.tz_convert("UTC").tz_localize(None)
    return parsed


def _read_json(path: Path) -> dict[str, object] | None:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _append_history(path: Path, value: dict[str, object]) -> None:
    """Append a compact immutable event; its digest chains the previous event."""
    path.parent.mkdir(parents=True, exist_ok=True)
    previous_digest = ""
    if path.exists():
        try:
            last = path.read_text(encoding="utf-8").splitlines()[-1]
            previous_digest = str(json.loads(last).get("event_digest", ""))
        except (OSError, IndexError, json.JSONDecodeError):
            previous_digest = "invalid_prior_history"
    event = {"previous_event_digest": previous_digest, **value}
    event["event_digest"] = stable_json_digest(event)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def _code_revision() -> str:
    configured = os.environ.get("AURORA_FORECAST_CODE_REVISION", "").strip()
    if configured:
        return configured
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unversioned"


def _model_evaluation_active() -> bool:
    if os.environ.get("AURORA_POWER_CANDIDATE_DEFER_MODEL_EVALUATION", "true").lower() != "true":
        return False
    try:
        result = subprocess.run(
            ["systemctl", "is-active", "--quiet", "aurora-model-evaluation-daily.service"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return False
    return result.returncode == 0


def _baseline_archive_before(
    path: Path,
    issue_time: pd.Timestamp,
) -> xr.Dataset | None:
    if not path.exists():
        return None
    try:
        with xr.open_zarr(path, chunks={}) as opened:
            archive = opened.load()
    except Exception:
        return None
    if "issue_time" not in archive.coords:
        return archive
    # A candidate may use only earlier forecast issues, never the baseline row
    # it is about to compare against.
    times = pd.DatetimeIndex(archive["issue_time"].values)
    return archive.isel(issue_time=np.asarray(times < issue_time))


def _source_manifest(
    *,
    baseline_attrs: dict[str, object],
    baseline_signature: str,
    input_forecast: Path,
    input_digest: str,
    issue_time: pd.Timestamp,
    physical_config_digest: str,
    physical_contract_id: str,
    power_history_days: float,
    feature_snapshot: IssueTimeFeatureSnapshot,
    baseline_control_contract_id: str,
    baseline_control_system_version: str,
) -> tuple[dict[str, object], str, str]:
    cycle = str(baseline_attrs.get("ecmwf_cycle_time", ""))
    provider = str(
        baseline_attrs.get(
            "ecmwf_provider_effective",
            baseline_attrs.get("ecmwf_provider_requested", "legacy"),
        )
    )
    source_cycle_set = f"ecmwf:{provider}:{cycle}:sha256:{input_digest[:20]}"
    manifest = {
        "schema_version": 2,
        "baseline_publication_signature": baseline_signature,
        "initial_soc_time": issue_time.isoformat(),
        "ecmwf_cycle_time": cycle,
        "source_cycle_set_id": source_cycle_set,
        "ecmwf_provider_effective": provider,
        "ecmwf_input_file": str(input_forecast),
        "ecmwf_input_sha256": input_digest,
        "physical_solar_config_sha256": physical_config_digest,
        "physical_solar_contract_id": physical_contract_id,
        "site_latitude": str(baseline_attrs.get("site_latitude", "")),
        "site_longitude": str(baseline_attrs.get("site_longitude", "")),
        "forecast_horizon_hours": str(baseline_attrs.get("forecast_horizon_hours", "")),
        "observation_cutoff": issue_time.isoformat(),
        "input_power_history_days": float(power_history_days),
        "baseline_control_contract_id": str(baseline_control_contract_id),
        "baseline_control_system_version": str(baseline_control_system_version),
        "issue_time_feature_contract_id": feature_snapshot.contract_id,
        "issue_time_feature_snapshot_digest": f"sha256:{feature_snapshot.snapshot_digest}",
        "issue_time_features": feature_snapshot.manifest,
        "public_model_ablations": feature_snapshot.manifest["public_model_ablations"],
    }
    return manifest, stable_json_digest(manifest), source_cycle_set


def _baseline_control_identity(attrs: dict[str, object]) -> tuple[str, str]:
    """Return the exact control identity allowed to train the load residual."""
    contract = str(attrs.get("forecast_model_contract_id", "")).strip()
    if not contract or contract.lower() in {"nan", "none"}:
        raise ValueError("Baseline is missing its forecast model contract identity")
    system = str(attrs.get("forecast_system_version", "")).strip()
    if not system or system.lower() in {"nan", "none"}:
        system = "unversioned_control"
    return contract, system


def _write_immutable_manifest(root: Path, manifest: dict[str, object], digest: str) -> Path:
    path = root / "source_manifests" / f"sha256-{digest}.json"
    if path.exists():
        existing = _read_json(path)
        if existing != manifest:
            raise RuntimeError(f"Immutable source manifest does not match: {path}")
    else:
        _atomic_json(path, manifest)
    return path


def _fixed_bias_from_baseline(attrs: dict[str, object]) -> dict[str, float]:
    try:
        raw = json.loads(str(attrs.get("soc_bias_correction_pct_points_by_bucket", "{}")))
    except json.JSONDecodeError:
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in raw.items():
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(parsed):
            out[str(key)] = parsed
    return out


def _validate_load_change_pair(candidate: xr.Dataset, baseline: xr.Dataset, *, solar_must_match: bool) -> None:
    """Validate identical source/state while intentionally allowing load changes."""
    failures: list[str] = []
    for name in ("initial_soc_time", "ecmwf_cycle_time", "ecmwf_input_file", "forecast_horizon_hours"):
        if str(candidate.attrs.get(name, "")) != str(baseline.attrs.get(name, "")):
            failures.append(name)
    for name in ("initial_soc_pct", "site_latitude", "site_longitude"):
        try:
            matches = np.isclose(
                float(candidate.attrs.get(name, np.nan)),
                float(baseline.attrs.get(name, np.nan)),
                rtol=0.0,
                atol=1.0e-8,
                equal_nan=True,
            )
        except (TypeError, ValueError):
            matches = False
        if not matches:
            failures.append(name)
    if "time" not in candidate or "time" not in baseline or not np.array_equal(
        np.asarray(candidate["time"].values), np.asarray(baseline["time"].values)
    ):
        failures.append("forecast_time_grid")
    fields = ["ECMWFSolarIrradiance"]
    if solar_must_match:
        fields.append("ForecastSolarWatts")
    for name in fields:
        if name not in candidate or name not in baseline or not np.allclose(
            np.asarray(candidate[name].values, dtype=np.float64),
            np.asarray(baseline[name].values, dtype=np.float64),
            rtol=1.0e-6,
            atol=1.0e-5,
            equal_nan=True,
        ):
            failures.append(name)
    for name in (
        "load_model",
        "load_model_version",
        "load_state_contract",
        "battery_energy_model",
        "battery_usable_capacity_kwh",
        "battery_charge_efficiency",
        "battery_discharge_efficiency",
        "battery_parasitic_load_w",
        "battery_max_charge_w",
        "battery_max_discharge_w",
        "soc_bias_correction_pct_points_by_bucket",
    ):
        if str(candidate.attrs.get(name, "")) != str(baseline.attrs.get(name, "")):
            failures.append(name)
    if failures:
        raise ValueError("v12 candidate source/state pair validation failed: " + ", ".join(dict.fromkeys(failures)))


def _pair_id(
    *,
    lane: str,
    baseline_signature: str,
    input_digest: str,
    issue_time: pd.Timestamp,
    physical_contract_id: str,
    source_manifest_digest: str,
) -> str:
    payload = {
        "schema": 2,
        "lane": lane,
        "baseline_publication_signature": baseline_signature,
        "input_sha256": input_digest,
        "issue_time": issue_time.isoformat(),
        "physical_solar_contract_id": physical_contract_id,
        "source_manifest_digest": source_manifest_digest,
    }
    return "power-v12-pair-v1-" + stable_json_digest(payload)[:20]


def _write_pair_bundle(
    lane_root: Path,
    *,
    pair_id: str,
    baseline: xr.Dataset,
    candidate: xr.Dataset,
    manifest: dict[str, object],
) -> Path:
    family = lane_root / "pairs" / pair_id
    signature = str(candidate.attrs.get("publication_signature", "")).strip()
    if not signature:
        raise RuntimeError("Candidate publication has no signature")
    bundle = family / signature
    manifest = {**manifest, "pair_status": "complete", "candidate_publication_signature": signature}
    if bundle.exists():
        existing = _read_json(bundle / "pair_manifest.json")
        if existing != manifest:
            raise RuntimeError(f"Existing immutable pair bundle does not match: {bundle}")
        return bundle
    family.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".pair-staging-", dir=family) as temporary:
        staging = Path(temporary)
        _atomic_write_zarr(baseline, staging / "baseline_forecast.zarr")
        _atomic_write_zarr(candidate, staging / "candidate_forecast.zarr")
        _write_state(staging / "pair_manifest.json", manifest)
        staging.replace(bundle)
    return bundle


def _lane_result_path(root: Path, lane: str) -> Path:
    return root / "lanes" / lane / "power_soc_forecast.zarr"


def _candidate_ensemble_paths(lane_root: Path, contract_id: str) -> dict[str, Path]:
    """Keep each semantic member-wise contract in its own candidate directory."""
    root = lane_root / "ensemble_contracts" / contract_id
    return {
        "forecast": root / "power_soc_ensemble_forecast.zarr",
        "archive": root / "power_soc_ensemble_archive.zarr",
        "evidence": root / "campaign_ensemble_evidence.zarr",
        "summary": root / "campaign_ensemble_summary.json",
        "gate": root / "promotion_ensemble_gate.json",
    }


def _load_baseline_ensemble(path: Path | None) -> tuple[xr.Dataset | None, str, str]:
    """Load a read-only site ensemble without blocking deterministic evidence."""
    if path is None:
        return None, "", "blocked_baseline_ensemble_not_configured"
    path = Path(path)
    if not path.exists():
        return None, "", "blocked_baseline_ensemble_missing"
    try:
        with xr.open_zarr(path, chunks={}) as opened:
            ensemble = opened.load()
        return ensemble, baseline_ensemble_signature(ensemble), "available"
    except Exception as exc:
        return None, "", f"blocked_baseline_ensemble_invalid:{type(exc).__name__}"


def _verify_baseline_ensemble_unchanged(path: Path, expected_signature: str) -> None:
    """Detect a late baseline ensemble refresh before candidate publication."""
    refreshed, signature, status = _load_baseline_ensemble(path)
    del refreshed
    if status != "available" or signature != expected_signature:
        raise RuntimeError("Baseline ensemble changed during candidate member-wise generation")


def run_candidate(
    *,
    baseline_forecast_zarr: Path = BASELINE_FORECAST_ZARR_PATH,
    baseline_archive_zarr: Path = BASELINE_ARCHIVE_ZARR_PATH,
    baseline_ensemble_zarr: Path | None = BASELINE_ENSEMBLE_ZARR_PATH,
    candidate_root: Path = CANDIDATE_ROOT,
    power_zarr: Path = POWER_ZARR_PATH,
    pdu_zarr: Path = POWER_PDU_ZARR_PATH,
    physical_config: Path = DEFAULT_PHYSICAL_SOLAR_CONFIG_PATH,
    asfs_zarr: Path = ASFS_LOGGER_ZARR_PATH,
    menapia_mqtt_log: Path = UAS_MQTT_LOG_PATH,
    public_source_manifest_root: Path = PUBLIC_SOURCE_MANIFEST_ROOT,
) -> dict[str, Path]:
    """Generate lanes B/C/D for one verified v10/v11 ECMWF baseline issue."""
    baseline_forecast_zarr = Path(baseline_forecast_zarr)
    baseline_archive_zarr = Path(baseline_archive_zarr)
    baseline_ensemble_zarr = (
        Path(baseline_ensemble_zarr) if baseline_ensemble_zarr is not None else None
    )
    candidate_root = Path(candidate_root)
    asfs_zarr = Path(asfs_zarr)
    menapia_mqtt_log = Path(menapia_mqtt_log)
    public_source_manifest_root = Path(public_source_manifest_root)
    if any(path.suffix.lower() == ".zarr" for path in (candidate_root, *candidate_root.parents)):
        raise ValueError("v12 candidate root cannot be inside a Zarr store")
    protected = tuple(
        path
        for path in (
            baseline_forecast_zarr,
            baseline_archive_zarr,
            baseline_ensemble_zarr,
            power_zarr,
            pdu_zarr,
            asfs_zarr,
        )
        if path is not None
    )
    if any(_paths_overlap(candidate_root, path) for path in protected):
        raise ValueError("v12 candidate root overlaps a protected baseline or input product")
    if not baseline_forecast_zarr.exists():
        raise FileNotFoundError(f"Baseline forecast is missing: {baseline_forecast_zarr}")
    if _model_evaluation_active():
        status = {
            "schema_version": 1,
            "environment": "development",
            "status": "deferred_model_evaluation_active",
            "updated_at_utc": utc_now_iso(),
        }
        _atomic_json(candidate_root / "status.json", status)
        return {}
    with xr.open_zarr(baseline_forecast_zarr, chunks={}) as opened:
        baseline = opened.load()
    attrs = dict(baseline.attrs)
    if str(attrs.get("forecast_verification_eligible", "")).lower() != "true":
        raise ValueError("Baseline is not an archive-eligible independent forecast issue")
    if str(attrs.get("forecast_refresh_kind", "")) != "ecmwf_cycle":
        raise ValueError("Baseline refresh is not a full ECMWF cycle")
    baseline_signature = str(attrs.get("publication_signature", "")).strip()
    if not baseline_signature:
        raise ValueError("Baseline forecast does not have a publication signature")
    input_forecast = Path(str(attrs.get("ecmwf_input_file", "")).strip())
    if not input_forecast.is_file():
        raise FileNotFoundError("Baseline ECMWF forcing is no longer available")
    issue_time = _utc_naive(attrs.get("initial_soc_time", ""))
    try:
        latitude = float(attrs["site_latitude"])
        longitude = float(attrs["site_longitude"])
        horizon_hours = int(float(attrs["forecast_horizon_hours"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Baseline is missing site or horizon provenance") from exc
    configuration = load_physical_solar_config(physical_config)
    config_digest = physical_solar_config_digest(configuration)
    physical_contract = physical_solar_contract_id(configuration, latitude=latitude, longitude=longitude)
    input_digest = _sha256_file(input_forecast)
    baseline_control_contract_id, baseline_control_system_version = _baseline_control_identity(attrs)
    code_revision = _code_revision()
    baseline_ensemble, baseline_ensemble_signature_value, ensemble_input_status = (
        _load_baseline_ensemble(baseline_ensemble_zarr)
    )
    public_source_manifest_root_digest = _public_source_manifest_root_digest(
        public_source_manifest_root
    )
    existing_status = _read_json(candidate_root / "status.json")
    if (
        existing_status
        and existing_status.get("status") == "complete"
        and existing_status.get("baseline_publication_signature") == baseline_signature
        and existing_status.get("candidate_feature_set_version") == V12_FEATURE_SET_VERSION
        and existing_status.get("candidate_code_revision") == code_revision
        and existing_status.get("physical_solar_config_sha256") == config_digest
        and existing_status.get("baseline_control_contract_id") == baseline_control_contract_id
        and existing_status.get("baseline_control_system_version") == baseline_control_system_version
        and existing_status.get("baseline_ensemble_signature") == baseline_ensemble_signature_value
        and existing_status.get("memberwise_ensemble_input_status") == ensemble_input_status
        and existing_status.get("public_source_manifest_root_digest")
        == public_source_manifest_root_digest
    ):
        return {
            lane: _lane_result_path(candidate_root, lane)
            for lane in LANES
            if _lane_result_path(candidate_root, lane).exists()
        }
    try:
        provider = validate_provider(str(attrs.get("ecmwf_provider_effective", "legacy")))
    except ValueError:
        provider = "legacy"
    reference_archive = _baseline_archive_before(baseline_archive_zarr, issue_time)
    history_start = issue_time - pd.Timedelta(days=V12_POWER_HISTORY_DAYS)
    with xr.open_zarr(power_zarr, chunks={}) as opened:
        fields = [name for name in POWER_HISTORY_FIELDS if name in opened]
        if "BatterySOC" not in fields:
            raise ValueError("Power input is missing BatterySOC for candidate load/evidence features")
        power_for_fit = opened[fields].sel(
            time=slice(history_start.to_datetime64(), issue_time.to_datetime64())
        ).load()
    feature_snapshot = build_issue_time_feature_snapshot(
        issue_time=issue_time,
        power_history=power_for_fit,
        pdu_zarr=pdu_zarr,
        asfs_zarr=asfs_zarr,
        menapia_mqtt_log=menapia_mqtt_log,
        public_source_manifest_root=public_source_manifest_root,
    )
    source_manifest, source_manifest_digest, source_cycle_set_id = _source_manifest(
        baseline_attrs=attrs,
        baseline_signature=baseline_signature,
        input_forecast=input_forecast,
        input_digest=input_digest,
        issue_time=issue_time,
        physical_config_digest=config_digest,
        physical_contract_id=physical_contract,
        power_history_days=V12_POWER_HISTORY_DAYS,
        feature_snapshot=feature_snapshot,
        baseline_control_contract_id=baseline_control_contract_id,
        baseline_control_system_version=baseline_control_system_version,
    )
    _write_immutable_manifest(candidate_root, source_manifest, source_manifest_digest)
    residual = fit_bounded_load_residual(
        reference_archive,
        power_for_fit,
        issue_time=issue_time,
        forecast_times=pd.DatetimeIndex(baseline["time"].values),
        load_mode=str(attrs.get("load_mode", "unknown")),
        control_forecast_model_contract_id=baseline_control_contract_id,
        control_forecast_system_version=baseline_control_system_version,
    )
    seed_state = _baseline_seed_state(attrs)
    fixed_bias = _fixed_bias_from_baseline(attrs)
    baseline_legacy_solar_w = pd.Series(
        np.asarray(baseline["ForecastSolarWatts"].values, dtype=np.float64),
        index=pd.DatetimeIndex(baseline["time"].values),
    )
    results: dict[str, Path] = {}
    lane_signatures: dict[str, str] = {}
    lane_summaries: dict[str, dict[str, object]] = {}
    lane_promotion_gates: dict[str, dict[str, object]] = {}
    lane_ensemble_summaries: dict[str, dict[str, object]] = {}
    lane_specs = (
        (LANE_PHYSICAL_SOLAR, PHYSICAL_SOLAR_MODEL_NAME, None, True, False),
        (LANE_LOAD_RESIDUAL, LEGACY_SOLAR_MODEL_NAME, residual.as_profile(), False, True),
        (LANE_HYBRID, PHYSICAL_SOLAR_MODEL_NAME, residual.as_profile(), False, False),
    )
    for lane, solar_model, load_profile, exact_pair, solar_must_match in lane_specs:
        lane_root = candidate_root / "lanes" / lane
        pair_id = _pair_id(
            lane=lane,
            baseline_signature=baseline_signature,
            input_digest=input_digest,
            issue_time=issue_time,
            physical_contract_id=physical_contract,
            source_manifest_digest=source_manifest_digest,
        )
        identity = v12_forecast_identity(
            lane=lane,
            issue_time=issue_time,
            source_cycle_set_id=source_cycle_set_id,
            source_manifest_digest=source_manifest_digest,
            physical_config_digest=config_digest,
            load_residual=residual if load_profile is not None else None,
            code_revision=code_revision,
            power_history_days=V12_POWER_HISTORY_DAYS,
            issue_feature_contract_id=feature_snapshot.contract_id,
            baseline_control_contract_id=baseline_control_contract_id,
            baseline_control_system_version=baseline_control_system_version,
            source_availability_code=feature_snapshot.source_availability_code,
            feature_degradation_codes=feature_snapshot.degradation_codes,
        )
        output = generate(
            power_zarr=power_zarr,
            pdu_zarr=pdu_zarr,
            output_zarr=lane_root / "power_soc_forecast.zarr",
            input_forecast=input_forecast,
            state_path=lane_root / "power_soc_forecast_state.json",
            archive_zarr=lane_root / "power_soc_forecast_archive.zarr",
            skill_zarr=lane_root / "daily_diagnostic_skill.zarr",
            hindcast_zarr=lane_root / "power_soc_hindcast.zarr",
            latitude=latitude,
            longitude=longitude,
            horizon_hours=horizon_hours,
            provider=provider,
            shadow_report_path=lane_root / "ecmwf_provider_shadow.json",
            max_power_age_minutes=None,
            archive_forecast=True,
            solar_model=solar_model,
            physical_solar_config_path=physical_config,
            power_cutoff_time=issue_time,
            power_history_days=V12_POWER_HISTORY_DAYS,
            evaluation_pair_id=pair_id,
            input_snapshot_id=f"sha256:{input_digest}",
            expected_input_sha256=input_digest,
            expected_physical_config_sha256=config_digest if solar_model == PHYSICAL_SOLAR_MODEL_NAME else None,
            pair_reference=baseline if exact_pair else None,
            state_override=seed_state,
            forecast_identity=identity,
            load_residual_profile=load_profile,
            reference_forecast_archive=reference_archive,
            fixed_soc_bias_corrections_override=fixed_bias,
            fixed_legacy_solar_w=(
                baseline_legacy_solar_w
                if solar_model == LEGACY_SOLAR_MODEL_NAME
                else None
            ),
        )
        with xr.open_zarr(output, chunks={}) as opened:
            candidate = opened.load()
        if exact_pair:
            validate_paired_candidate(candidate, baseline)
        else:
            _validate_load_change_pair(candidate, baseline, solar_must_match=solar_must_match)
        if _sha256_file(input_forecast) != input_digest:
            raise RuntimeError("ECMWF forcing changed during candidate generation")
        pair_manifest = {
            "schema_version": 2,
            "evaluation_pair_id": pair_id,
            "candidate_lane": lane,
            "baseline_publication_signature": baseline_signature,
            "input_snapshot_id": f"sha256:{input_digest}",
            "source_manifest_digest": source_manifest_digest,
            "issue_time_feature_snapshot_digest": f"sha256:{feature_snapshot.snapshot_digest}",
            "local_feature_contract_id": feature_snapshot.contract_id,
            "baseline_control_contract_id": baseline_control_contract_id,
            "baseline_control_system_version": baseline_control_system_version,
            "source_availability_code": feature_snapshot.source_availability_code,
            "initial_soc_time": issue_time.isoformat(),
            "source_cycle_set_id": source_cycle_set_id,
            "forecast_model_contract_id": str(candidate.attrs.get("forecast_model_contract_id", "")),
            "forecast_identity_id": str(candidate.attrs.get("forecast_identity_id", "")),
            "solar_model_contract_id": str(candidate.attrs.get("solar_model_contract_id", "")),
            "solar_physical_config_sha256": config_digest,
            "baseline_snapshot": "baseline_forecast.zarr",
            "candidate_snapshot": "candidate_forecast.zarr",
        }
        _write_pair_bundle(
            lane_root,
            pair_id=pair_id,
            baseline=baseline,
            candidate=candidate,
            manifest=pair_manifest,
        )
        evaluation_contract = evaluation_contract_from_forecast(candidate)
        evidence = build_campaign_evidence(
            lane_root / "pairs",
            power_for_fit,
            lane=lane,
            evaluation_contract=evaluation_contract,
        )
        _atomic_write_zarr(evidence, lane_root / "campaign_evidence.zarr")
        summary = campaign_score_surfaces(evidence)
        ensemble_summary: dict[str, object] = {
            "status": ensemble_input_status,
            "reason": "Baseline ensemble is read-only and optional for deterministic candidate evidence.",
        }
        ensemble_gate: dict[str, object] | None = None
        if baseline_ensemble is not None and baseline_ensemble_zarr is not None:
            try:
                candidate_ensemble = build_candidate_memberwise_ensemble(
                    baseline,
                    candidate,
                    baseline_ensemble,
                    lane=lane,
                    physical_config=configuration,
                    latitude=latitude,
                    longitude=longitude,
                )
                ensemble_contract = str(
                    candidate_ensemble.attrs["candidate_ensemble_contract_id"]
                )
                ensemble_paths = _candidate_ensemble_paths(lane_root, ensemble_contract)
                # Re-read and hash the site-member input before any candidate
                # ensemble writer or completion manifest is exposed.
                _verify_baseline_ensemble_unchanged(
                    baseline_ensemble_zarr, baseline_ensemble_signature_value
                )
                _atomic_write_zarr(candidate_ensemble, ensemble_paths["forecast"])
                append_candidate_ensemble_archive(
                    candidate_ensemble, ensemble_paths["archive"]
                )
                write_immutable_ensemble_pair_bundle(
                    lane_root,
                    deterministic_pair_id=pair_id,
                    baseline_ensemble=baseline_ensemble,
                    candidate_ensemble=candidate_ensemble,
                    manifest_extra={
                        "baseline_publication_signature": baseline_signature,
                        "source_manifest_digest": source_manifest_digest,
                        "source_cycle_set_id": source_cycle_set_id,
                        "input_snapshot_id": f"sha256:{input_digest}",
                    },
                )
                ensemble_evidence = build_campaign_ensemble_evidence(
                    lane_root / "ensemble_pairs",
                    power_for_fit,
                    lane=lane,
                    evaluation_contract=ensemble_evaluation_contract_from_forecast(
                        candidate_ensemble
                    ),
                )
                _atomic_write_zarr(ensemble_evidence, ensemble_paths["evidence"])
                ensemble_summary = campaign_ensemble_score_surfaces(ensemble_evidence)
                ensemble_gate = ensemble_promotion_gate(ensemble_evidence)
                ensemble_summary["promotion_gate"] = ensemble_gate
                ensemble_summary["status"] = "complete"
                ensemble_summary["candidate_ensemble_contract_id"] = ensemble_contract
                _atomic_json(ensemble_paths["summary"], ensemble_summary)
                _atomic_json(ensemble_paths["gate"], ensemble_gate)
            except (OSError, RuntimeError, ValueError, KeyError) as exc:
                ensemble_summary = {
                    "status": f"blocked_memberwise_candidate_error:{type(exc).__name__}",
                    "reason": str(exc),
                }
                ensemble_gate = None
        gates = promotion_gate_review(evidence, ensemble_gate=ensemble_gate)
        summary["promotion_gates"] = gates
        summary["memberwise_ensemble"] = ensemble_summary
        _atomic_json(lane_root / "evaluation_summary.json", summary)
        results[lane] = output
        lane_signatures[lane] = str(candidate.attrs.get("publication_signature", ""))
        lane_summaries[lane] = summary
        lane_promotion_gates[lane] = gates
        lane_ensemble_summaries[lane] = ensemble_summary
    public_source_results = run_public_source_ablations(
        candidate_root=candidate_root,
        baseline=baseline,
        power_for_evidence=power_for_fit,
        source_manifest_root=public_source_manifest_root,
        configuration=configuration,
        latitude=latitude,
        longitude=longitude,
        code_revision=code_revision,
    )
    status = {
        "schema_version": 1,
        "environment": "development",
        "authority": "candidate",
        "status": "complete",
        "updated_at_utc": utc_now_iso(),
        "baseline_publication_signature": baseline_signature,
        "source_manifest_digest": source_manifest_digest,
        "source_cycle_set_id": source_cycle_set_id,
        "input_snapshot_id": f"sha256:{input_digest}",
        "training_cutoff_utc": issue_time.isoformat(),
        "input_power_history_days": V12_POWER_HISTORY_DAYS,
        "candidate_feature_set_version": V12_FEATURE_SET_VERSION,
        "candidate_code_revision": code_revision,
        "physical_solar_config_sha256": config_digest,
        "baseline_control_contract_id": baseline_control_contract_id,
        "baseline_control_system_version": baseline_control_system_version,
        "baseline_ensemble_path": str(baseline_ensemble_zarr or ""),
        "baseline_ensemble_signature": baseline_ensemble_signature_value,
        "memberwise_ensemble_input_status": ensemble_input_status,
        "public_source_manifest_root_digest": public_source_manifest_root_digest,
        "issue_time_feature_contract_id": feature_snapshot.contract_id,
        "issue_time_feature_snapshot_digest": f"sha256:{feature_snapshot.snapshot_digest}",
        "source_availability_code": feature_snapshot.source_availability_code,
        "public_model_ablations": feature_snapshot.manifest["public_model_ablations"],
        "public_model_ablation_results": public_source_results,
        "load_residual": {
            "status": residual.status,
            "contract_id": residual.contract_id,
            "training_samples": residual.training_samples,
            "training_cycles": residual.training_cycles,
            "training_days": residual.training_days,
        },
        "lanes": {
            lane: {
                "path": str(path),
                "publication_signature": lane_signatures[lane],
                "memberwise_ensemble": lane_ensemble_summaries.get(lane, {}),
            }
            for lane, path in results.items()
        },
        "promotion_status": "not_eligible_requires_campaign_evidence",
        "promotion_gates": lane_promotion_gates.get(LANE_HYBRID, {}),
    }
    _atomic_json(candidate_root / "status.json", status)
    _atomic_json(
        candidate_root / "acceptance_record.json",
        {
            "schema_version": 1,
            "environment": "development",
            "authority": "candidate",
            "status": "not_accepted",
            "decision": "retain_unpublished_candidate",
            "updated_at_utc": utc_now_iso(),
            "baseline_publication_signature": baseline_signature,
            "source_manifest_digest": source_manifest_digest,
            "source_cycle_set_id": source_cycle_set_id,
            "issue_time_feature_contract_id": feature_snapshot.contract_id,
            "issue_time_feature_snapshot_digest": f"sha256:{feature_snapshot.snapshot_digest}",
            "promotion_gates": lane_promotion_gates.get(LANE_HYBRID, {}),
            "reason": (
                "Promotion is manual and requires cumulative paired campaign evidence; "
                "this runner never changes an operational forecast product."
            ),
            "required_gates": {
                "paired_independent_cycles": "minimum 30 per lead bucket across 10 UTC days",
                "soc_skill": "review campaign evidence against v10 and persistence",
                "solar_and_load": "review only issue-time-safe, uncensored metrics",
                "ensemble": "memberwise candidate ensemble required; status is recorded per lane",
                "reserve_events": "insufficient_events unless an event sample is available",
                "operational_safety": "memory, runtime, reproducibility and API compatibility must pass",
            },
        },
    )
    _atomic_json(
        candidate_root / "review_summary.json",
        {
            "schema_version": 1,
            "environment": "development",
            "authority": "candidate",
            "status": "pending_campaign_review",
            "updated_at_utc": utc_now_iso(),
            "baseline_publication_signature": baseline_signature,
            "source_manifest_digest": source_manifest_digest,
            "source_cycle_set_id": source_cycle_set_id,
            "issue_time_features": {
                "contract_id": feature_snapshot.contract_id,
                "snapshot_digest": f"sha256:{feature_snapshot.snapshot_digest}",
                "source_availability_code": feature_snapshot.source_availability_code,
            "public_model_ablations": feature_snapshot.manifest["public_model_ablations"],
            "public_model_ablation_results": public_source_results,
            },
            "load_residual": status["load_residual"],
            "lanes": lane_summaries,
            "memberwise_ensemble_input_status": ensemble_input_status,
            "promotion_gates": lane_promotion_gates.get(LANE_HYBRID, {}),
            "next_action": "Accumulate paired independent ECMWF-cycle evidence; do not promote from rolling diagnostics.",
        },
    )
    _append_history(
        candidate_root / "evaluation_history.jsonl",
        {
            "event": "candidate_cycle_complete",
            "at_utc": utc_now_iso(),
            "baseline_publication_signature": baseline_signature,
            "source_manifest_digest": source_manifest_digest,
            "lanes": sorted(results),
            "load_residual_status": residual.status,
            "public_model_ablation_results": {
                source: result.get("status", "unknown")
                for source, result in public_source_results.items()
            },
        },
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate issue-time-paired v12 hybrid power candidates")
    parser.add_argument("--baseline-forecast-zarr", type=Path, default=BASELINE_FORECAST_ZARR_PATH)
    parser.add_argument("--baseline-archive-zarr", type=Path, default=BASELINE_ARCHIVE_ZARR_PATH)
    parser.add_argument("--baseline-ensemble-zarr", type=Path, default=BASELINE_ENSEMBLE_ZARR_PATH)
    parser.add_argument("--candidate-root", type=Path, default=CANDIDATE_ROOT)
    parser.add_argument("--power-zarr", type=Path, default=POWER_ZARR_PATH)
    parser.add_argument("--pdu-zarr", type=Path, default=POWER_PDU_ZARR_PATH)
    parser.add_argument("--physical-config", type=Path, default=DEFAULT_PHYSICAL_SOLAR_CONFIG_PATH)
    parser.add_argument("--asfs-zarr", type=Path, default=ASFS_LOGGER_ZARR_PATH)
    parser.add_argument("--menapia-mqtt-log", type=Path, default=UAS_MQTT_LOG_PATH)
    parser.add_argument(
        "--public-source-manifest-root",
        type=Path,
        default=PUBLIC_SOURCE_MANIFEST_ROOT,
    )
    args = parser.parse_args()
    results = run_candidate(
        baseline_forecast_zarr=args.baseline_forecast_zarr,
        baseline_archive_zarr=args.baseline_archive_zarr,
        baseline_ensemble_zarr=args.baseline_ensemble_zarr,
        candidate_root=args.candidate_root,
        power_zarr=args.power_zarr,
        pdu_zarr=args.pdu_zarr,
        physical_config=args.physical_config,
        asfs_zarr=args.asfs_zarr,
        menapia_mqtt_log=args.menapia_mqtt_log,
        public_source_manifest_root=args.public_source_manifest_root,
    )
    if results:
        print("Verified isolated v12 candidate lanes: " + ", ".join(sorted(results)))
    else:
        print("v12 candidate deferred; no operational product was changed")


if __name__ == "__main__":
    main()
