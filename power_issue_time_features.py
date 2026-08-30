"""Bounded, as-of provenance for local power-candidate features.

This module intentionally separates *availability* from *predictive use*.  A
local observation is useful only when it was available by the forecast issue
time; delayed science products are never silently promoted to predictors.
The v12 runner stores the resulting compact snapshot in its immutable source
manifest, rather than retaining whole instrument grids or logs.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from uas_mqtt import parse_uas_mqtt_line


ISSUE_TIME_FEATURE_SCHEMA_VERSION = "issue-time-local-features-v1"
PUBLIC_SOURCE_ABLATION_SCHEMA_VERSION = "site-extracted-public-source-v2"
MAX_PUBLIC_SITE_EXTRACT_BYTES = 64 * 1024 * 1024

APS_FIELDS = (
    "BatterySOC",
    "BatteryVolts",
    "BatteryWatts",
    "ACOutputWatts",
    "DCInverterWatts",
    "SolarWatts_East",
    "SolarWatts_South",
    "SolarWatts_West",
    # Victron register 791: 0=off, 1=voltage/current limited, 2=MPPT
    # active.  These are observational eligibility flags, not predictors.
    "SolarMPPMode_East",
    "SolarMPPMode_South",
    "SolarMPPMode_West",
)
ASFS_FIELDS = (
    "sr30_swd_Irr_Avg",
    "spn1_tot_Avg",
    "spn1_dif_Avg",
    "vaisala_T_Avg",
    "metek_x_out_Avg",
    "metek_y_out_Avg",
    "watts_on_48vdc_Avg",
    "amp_meter_48vdc_Avg",
    "batt_volt_Avg",
)
PDU_FIELDS = tuple(
    f"PDUOutlet{outlet}{metric}"
    for outlet in range(1, 9)
    for metric in ("Watts", "State")
)
PUBLIC_SOURCES = ("IFS", "GFS", "ICON", "GEFS")


def _digest(value: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _utc_naive(value: object) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    if parsed.tz is not None:
        return parsed.tz_convert("UTC").tz_localize(None)
    return parsed


def _iso(value: pd.Timestamp | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    return _utc_naive(value).isoformat() + "Z"


def _finite(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return round(numeric, 6) if np.isfinite(numeric) else None


def issue_time_feature_contract_id() -> str:
    """Return the stable contract for the local-source feature interface."""
    payload = {
        "schema_version": ISSUE_TIME_FEATURE_SCHEMA_VERSION,
        "aps_fields": APS_FIELDS,
        "asfs_fields": ASFS_FIELDS,
        "pdu_fields": PDU_FIELDS,
        "menapia": "raw_dock1_dock2_tier_tail_only",
        "recharge": "six_cycle_prior_not_delayed_flight_predictor",
        "public_sources": PUBLIC_SOURCES,
        "public_source_schema": PUBLIC_SOURCE_ABLATION_SCHEMA_VERSION,
    }
    return "issue-features-v1-" + _digest(payload)[:16]


def _component_from_dataset(
    dataset: xr.Dataset | None,
    *,
    fields: Sequence[str],
    cutoff: pd.Timestamp,
    lookback: pd.Timedelta,
    freshness: pd.Timedelta,
    role: str,
) -> dict[str, object]:
    """Summarise a tiny, strictly as-of data window without retaining it."""
    component: dict[str, object] = {
        "role": role,
        "observation_cutoff_utc": _iso(cutoff),
        "lookback_minutes": int(lookback / pd.Timedelta(minutes=1)),
        "freshness_minutes": int(freshness / pd.Timedelta(minutes=1)),
        "fields": {},
    }
    if dataset is None or "time" not in dataset.coords:
        component["status"] = "missing"
        return component
    available = [
        name
        for name in fields
        if name in dataset and dataset[name].dims == ("time",)
    ]
    if not available:
        component["status"] = "no_expected_fields"
        return component
    start = cutoff - lookback
    try:
        selected = dataset[available].sel(
            time=slice(start.to_datetime64(), cutoff.to_datetime64())
        ).load()
    except Exception:
        component["status"] = "read_error"
        return component
    if selected.sizes.get("time", 0) == 0:
        component["status"] = "no_asof_samples"
        return component
    times = pd.DatetimeIndex(selected["time"].values)
    # The slice is expected to be half-open at the caller cutoff in most input
    # stores; explicitly drop any malformed future timestamp nevertheless.
    allowed = np.asarray(times <= cutoff)
    if not np.any(allowed):
        component["status"] = "no_asof_samples"
        return component
    selected = selected.isel(time=allowed)
    times = pd.DatetimeIndex(selected["time"].values)
    latest_times: list[pd.Timestamp] = []
    field_values: dict[str, object] = {}
    for name in available:
        values = np.asarray(selected[name].values, dtype=np.float64)
        finite = np.isfinite(values)
        if not np.any(finite):
            field_values[name] = {"samples": 0, "latest": None, "median": None}
            continue
        last_index = int(np.flatnonzero(finite)[-1])
        latest = pd.Timestamp(times[last_index])
        latest_times.append(latest)
        field_values[name] = {
            "samples": int(np.count_nonzero(finite)),
            "latest": _finite(values[last_index]),
            "median": _finite(np.nanmedian(values[finite])),
            "latest_time_utc": _iso(latest),
        }
    component["fields"] = field_values
    if not latest_times:
        component["status"] = "no_finite_asof_values"
        return component
    latest = max(latest_times)
    age = max((cutoff - latest) / pd.Timedelta(minutes=1), 0.0)
    component["latest_observation_utc"] = _iso(latest)
    component["age_minutes"] = round(float(age), 6)
    component["status"] = "fresh" if age <= freshness / pd.Timedelta(minutes=1) else "stale"
    return component


def _open_component(
    path: Path | None,
    **kwargs: object,
) -> dict[str, object]:
    if path is None or not Path(path).exists():
        return _component_from_dataset(None, **kwargs)
    try:
        with xr.open_zarr(Path(path), chunks={}) as opened:
            # ``_component_from_dataset`` loads only the supplied time window.
            return _component_from_dataset(opened, **kwargs)
    except Exception:
        return {
            "role": str(kwargs["role"]),
            "observation_cutoff_utc": _iso(kwargs["cutoff"]),
            "status": "read_error",
            "fields": {},
        }


def _tail_lines(path: Path, *, maximum_bytes: int = 131_072) -> list[str]:
    """Read a bounded UTF-8 tail without loading a growing raw MQTT log."""
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            end = handle.tell()
            handle.seek(max(end - maximum_bytes, 0), os.SEEK_SET)
            raw = handle.read()
    except OSError:
        return []
    decoded = raw.decode("utf-8", errors="replace")
    lines = decoded.splitlines()
    if end > maximum_bytes and lines:
        lines = lines[1:]
    return lines


def site_extract_sha256(path: Path) -> str:
    """Hash a small immutable site extract without retaining any global grid.

    The source producer normalises public forecasts into a compact local Zarr
    or NetCDF extract.  A deterministic tree hash lets the power candidate
    reject changed forcing before it is admitted to an issue-time ablation.
    """

    path = Path(path)
    digest = hashlib.sha256()
    if path.is_file():
        if path.stat().st_size > MAX_PUBLIC_SITE_EXTRACT_BYTES:
            raise ValueError("site extract exceeds the bounded 64 MiB limit")
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    if not path.is_dir() or path.is_symlink():
        raise ValueError("site extract is neither a regular file nor directory")
    total = 0
    for item in sorted(path.rglob("*")):
        if item.is_symlink():
            raise ValueError("site extract cannot contain symbolic links")
        if not item.is_file():
            continue
        relative = item.relative_to(path).as_posix()
        size = item.stat().st_size
        total += size
        if total > MAX_PUBLIC_SITE_EXTRACT_BYTES:
            raise ValueError("site extract exceeds the bounded 64 MiB limit")
        digest.update(relative.encode("utf-8"))
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    if total == 0:
        raise ValueError("site extract is empty")
    return digest.hexdigest()


def public_source_manifest_record(
    source: str,
    *,
    root: Path | None,
    cutoff: pd.Timestamp,
) -> dict[str, object]:
    """Validate one issue-time-safe, site-only public-source manifest."""

    record: dict[str, object] = {
        "source": source,
        "role": "separate_ablation_only_not_pooled",
        "required_schema": PUBLIC_SOURCE_ABLATION_SCHEMA_VERSION,
    }
    if root is None:
        record["status"] = "not_enrolled_no_site_manifest_root"
        return record
    root = Path(root)
    path = root / f"{source.lower()}.json"
    if not path.is_file():
        record["status"] = "not_enrolled_no_site_manifest"
        return record
    try:
        if path.stat().st_size > 1_000_000:
            raise ValueError("manifest exceeds 1 MiB")
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("manifest is not an object")
        delivery = _utc_naive(raw["delivery_time_utc"])
        cycle = _utc_naive(raw["source_cycle_time_utc"])
        checksum = str(raw["site_extract_sha256"]).lower()
        relative = Path(str(raw["site_extract_path"]))
        extract_format = str(raw["site_extract_format"]).lower()
        irradiance_variable = str(raw["irradiance_variable"])
        if raw.get("schema_version") != PUBLIC_SOURCE_ABLATION_SCHEMA_VERSION:
            raise ValueError("schema mismatch")
        if str(raw.get("source", "")).upper() != source:
            raise ValueError("source mismatch")
        if raw.get("site_extract_only") is not True:
            raise ValueError("not site extracted")
        if raw.get("global_grid_retained") is not False:
            raise ValueError("global-grid retention not explicitly rejected")
        if relative.is_absolute() or ".." in relative.parts or str(relative) in {"", "."}:
            raise ValueError("invalid site extract path")
        extract = (root / relative).resolve()
        if os.path.commonpath((str(root.resolve()), str(extract))) != str(root.resolve()):
            raise ValueError("site extract escapes manifest root")
        if extract_format not in {"zarr", "netcdf"}:
            raise ValueError("unsupported compact site extract format")
        if not irradiance_variable:
            raise ValueError("missing irradiance variable")
        if len(checksum) != 64 or any(char not in "0123456789abcdef" for char in checksum):
            raise ValueError("invalid checksum")
        if site_extract_sha256(extract) != checksum:
            raise ValueError("site extract checksum mismatch")
        latitude = float(raw["site_latitude"])
        longitude = float(raw["site_longitude"])
        if not (np.isfinite(latitude) and np.isfinite(longitude)):
            raise ValueError("invalid site coordinates")
    except (KeyError, OSError, ValueError, TypeError, json.JSONDecodeError):
        record["status"] = "invalid_site_manifest"
        return record
    record.update(
        {
            "source_cycle_time_utc": _iso(cycle),
            "delivery_time_utc": _iso(delivery),
            "site_extract_sha256": f"sha256:{checksum}",
            "site_extract_path": relative.as_posix(),
            "site_extract_format": extract_format,
            "irradiance_variable": irradiance_variable,
            "site_latitude": latitude,
            "site_longitude": longitude,
        }
    )
    record["status"] = (
        "issue_time_available_pending_independent_ablation"
        if delivery <= cutoff and cycle <= cutoff
        else "late_delivery_excluded_from_issue_time_use"
    )
    return record


def _menapia_component(path: Path | None, *, cutoff: pd.Timestamp) -> dict[str, object]:
    component: dict[str, object] = {
        "role": "fresh raw dock-tier context; delayed flight products are retrospective only",
        "observation_cutoff_utc": _iso(cutoff),
        "recharge_prior": {
            "status": "planning_prior_only",
            "complete_episodes": 6,
            "energy_wh_p10_p50_p90": [160.65, 163.8, 179.85],
            "increment_w_p10_p50_p90": [181.8, 270.75, 340.65],
            "planning_energy_wh": 200.0,
            "source": "menapia_complete_recoveries_2026-08-26_to_2026-08-27_n6",
        },
        "flight_product_predictor_status": "excluded_delayed_or_quality_controlled_product",
    }
    if path is None or not Path(path).exists():
        component["status"] = "missing"
        return component
    records = []
    malformed = 0
    for line_number, line in enumerate(_tail_lines(Path(path)), start=1):
        parsed = parse_uas_mqtt_line(line, line_number=line_number)
        if parsed is None:
            malformed += 1
            continue
        timestamp = _utc_naive(parsed.timestamp)
        if timestamp <= cutoff:
            records.append((timestamp, parsed))
    if not records:
        component["status"] = "no_asof_records"
        component["malformed_tail_lines"] = malformed
        return component
    timestamp, latest = max(records, key=lambda item: item[0])
    age = max((cutoff - timestamp) / pd.Timedelta(minutes=1), 0.0)
    component.update(
        {
            "status": "fresh" if age <= 5.0 else "stale",
            "latest_observation_utc": _iso(timestamp),
            "age_minutes": round(float(age), 6),
            "dock1_tier": int(latest.dock1_tier),
            "dock2_tier": int(latest.dock2_tier),
            "shared_tier": latest.shared_tier,
            "dock_pair_state": latest.dock_pair_state,
            "malformed_tail_lines": malformed,
        }
    )
    return component


def _public_source_record(
    source: str,
    *,
    root: Path | None,
    cutoff: pd.Timestamp,
) -> dict[str, object]:
    """Validate a separately supplied, site-extracted public-source manifest.

    The power candidate never retrieves or reads a global public-model grid.
    A source can only become an ablation candidate after a separate producer
    has recorded a checksum, cycle and real delivery time in this compact
    manifest.  Even then it is *not* pooled into the ECMWF forecast.
    """
    return public_source_manifest_record(source, root=root, cutoff=cutoff)


@dataclass(frozen=True)
class IssueTimeFeatureSnapshot:
    contract_id: str
    snapshot_digest: str
    manifest: dict[str, object]
    source_availability_code: str
    degradation_codes: tuple[str, ...]


def build_issue_time_feature_snapshot(
    *,
    issue_time: pd.Timestamp | str,
    power_history: xr.Dataset,
    pdu_zarr: Path | None,
    asfs_zarr: Path | None,
    menapia_mqtt_log: Path | None,
    public_source_manifest_root: Path | None = None,
) -> IssueTimeFeatureSnapshot:
    """Create a small as-of snapshot for one forecast issue.

    APS/PDU are operational inputs.  ASFS irradiance, met and 48-V values are
    retained as local, issue-time-safe diagnostic/ablation features until a
    separate replay demonstrates their incremental skill.  Raw Menapia dock
    tiers are safe current context; delayed flight products stay excluded.
    """
    cutoff = _utc_naive(issue_time)
    aps = _component_from_dataset(
        power_history,
        fields=APS_FIELDS,
        cutoff=cutoff,
        lookback=pd.Timedelta(minutes=30),
        freshness=pd.Timedelta(minutes=20),
        role="APS SOC, battery and station-bus anchor",
    )
    pdu = _open_component(
        pdu_zarr,
        fields=PDU_FIELDS,
        cutoff=cutoff,
        lookback=pd.Timedelta(minutes=30),
        freshness=pd.Timedelta(minutes=15),
        role="PDU operating-state vector",
    )
    asfs = _open_component(
        asfs_zarr,
        fields=ASFS_FIELDS,
        cutoff=cutoff,
        lookback=pd.Timedelta(minutes=90),
        freshness=pd.Timedelta(minutes=20),
        role="ASFS irradiance, meteorology and 48-V local diagnostic/ablation features",
    )
    menapia = _menapia_component(menapia_mqtt_log, cutoff=cutoff)
    public = {
        source.lower(): _public_source_record(
            source, root=public_source_manifest_root, cutoff=cutoff
        )
        for source in PUBLIC_SOURCES
    }
    public_status = ";".join(f"{source}={public[source]['status']}" for source in sorted(public))
    source_availability_code = f"ecmwf_control=available;{public_status}"
    degradation: list[str] = []
    for key, component in (("aps", aps), ("pdu", pdu), ("asfs", asfs), ("menapia", menapia)):
        status = str(component.get("status", "unknown"))
        if status != "fresh":
            degradation.append(f"{key}_{status}")
    if not any(
        str(record.get("status")) == "issue_time_available_pending_independent_ablation"
        for record in public.values()
    ):
        degradation.append("public_model_ablations_not_enrolled")
    base: dict[str, object] = {
        "schema_version": ISSUE_TIME_FEATURE_SCHEMA_VERSION,
        "feature_contract_id": issue_time_feature_contract_id(),
        "observation_cutoff_utc": _iso(cutoff),
        "features": {
            "aps": aps,
            "pdu": pdu,
            "asfs": asfs,
            "menapia": menapia,
            "solar_geometry": {
                "role": "physical PV model inputs; configuration and site coordinates are separately checksummed",
                "status": "configured_in_physical_solar_contract",
            },
            "ecmwf": {
                "role": "active deterministic forcing; checksum and cycle are in source manifest",
                "status": "paired_control_forcing",
            },
            "delayed_diagnostics": {
                "cloudnet_hatpro_radar_ceilometer_camera": "retrospective_or_preissue_latency_replay_only_not_predictors",
            },
        },
        "public_model_ablations": public,
        "source_availability_code": source_availability_code,
        "degradation_codes": sorted(set(degradation)),
    }
    digest = _digest(base)
    manifest = {**base, "snapshot_digest": f"sha256:{digest}"}
    return IssueTimeFeatureSnapshot(
        contract_id=str(base["feature_contract_id"]),
        snapshot_digest=digest,
        manifest=manifest,
        source_availability_code=source_availability_code,
        degradation_codes=tuple(sorted(set(degradation))),
    )
