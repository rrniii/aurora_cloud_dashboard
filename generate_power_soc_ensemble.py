#!/usr/bin/env python3
"""Generate a 50-member ECMWF solar-driven APS SOC ensemble forecast."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from dataclasses import replace
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from generate_power_soc_forecast import (
    DEFAULT_BATTERY_CAPACITY_KWH,
    DEFAULT_HORIZON_HOURS,
    DEFAULT_LATITUDE,
    DEFAULT_LONGITUDE,
    DEFAULT_OPEN_DATA_SOURCE,
    SOLAR_MPP_MODE_FIELDS,
    _bounded_load_profile,
    _forecast_integration_times,
    _power_frame,
    build_historical_load_forecast,
    forecast_identity_id,
    integrate_soc_forecast,
    latest_finite,
    solar_irradiance_from_ssrd,
)
from power_load_contract import (
    CONTROLLED_LOAD_CONTRACT,
    STATE_HOLD_POLICY,
    ControlledLoadEstimate,
    validate_state_held_load,
)
from power_load_dynamics import (
    PHASE_CODES,
    StateLoadDynamics,
    controlled_load_member_profiles,
)
from power_soc_thresholds import (
    MINIMUM_OPERATIONAL_SOC_PCT,
    SOC_BELOW_THRESHOLD_BRIER_FIELD,
    SOC_BELOW_THRESHOLD_PROBABILITY_FIELD,
)
from power_battery_model import BatteryModel

POWER_ZARR_PATH = Path(os.environ.get("POWER_ZARR_PATH", "/data/aurora/products/power/power.zarr"))
POWER_SOC_FORECAST_ZARR_PATH = Path(
    os.environ.get("POWER_SOC_FORECAST_ZARR_PATH", "/data/aurora/products/power/power_soc_forecast.zarr")
)
POWER_SOC_ENSEMBLE_ZARR_PATH = Path(
    os.environ.get("POWER_SOC_ENSEMBLE_ZARR_PATH", "/data/aurora/products/power/power_soc_ensemble_forecast.zarr")
)
POWER_SOC_ENSEMBLE_ARCHIVE_ZARR_PATH = Path(
    os.environ.get(
        "POWER_SOC_ENSEMBLE_ARCHIVE_ZARR_PATH", "/data/aurora/products/power/power_soc_ensemble_archive.zarr"
    )
)
POWER_SOC_ENSEMBLE_SKILL_ZARR_PATH = Path(
    os.environ.get("POWER_SOC_ENSEMBLE_SKILL_ZARR_PATH", "/data/aurora/products/power/power_soc_ensemble_skill.zarr")
)
POWER_ECMWF_ENSEMBLE_TMP_DIR = Path(
    os.environ.get("POWER_ECMWF_ENSEMBLE_TMP_DIR", "/data/aurora/products/power/ecmwf_solar_ensemble_tmp")
)

ENSEMBLE_MEMBERS = tuple(range(1, 51))
LEAD_BUCKETS = (("0_6h", 0.0, 6.0), ("6_24h", 6.0, 24.0), ("24_48h", 24.0, 48.0), ("48_96h", 48.0, 96.0))
ENSEMBLE_SKILL_RETENTION_DAYS = 7.0
ENSEMBLE_SKILL_VERIFICATION_WINDOW_DAYS = 1.0
ENSEMBLE_MAX_LEAD_DAYS = max(stop for _, _, stop in LEAD_BUCKETS) / 24.0
ENSEMBLE_POWER_MIN_HISTORY_DAYS = (
    ENSEMBLE_SKILL_RETENTION_DAYS
    + ENSEMBLE_SKILL_VERIFICATION_WINDOW_DAYS
    + ENSEMBLE_MAX_LEAD_DAYS
)
ENSEMBLE_POWER_HISTORY_DAYS = float(
    os.environ.get(
        "AURORA_POWER_ENSEMBLE_HISTORY_DAYS",
        str(max(14.0, ENSEMBLE_POWER_MIN_HISTORY_DAYS)),
    )
)
ENSEMBLE_POWER_SOURCE_FIELDS = (
    "BatterySOC",
    "SolarWatts_East",
    "SolarWatts_South",
    "SolarWatts_West",
    *SOLAR_MPP_MODE_FIELDS,
    "BatteryWatts",
    "ACOutputWatts",
    "DCInverterWatts",
)
ENSEMBLE_FORECAST_PROVENANCE_ATTRS = (
    "forecast_model_name",
    "forecast_model_version",
    "forecast_model_contract_id",
    "forecast_identity_id",
    "forecast_system_version",
    "feature_set_version",
    "feature_set_digest",
    "training_cutoff_utc",
    "forecast_code_revision",
    "source_cycle_set_id",
    "source_manifest_digest",
    "degraded_mode_code",
    "candidate_lane",
    "local_feature_contract_id",
    "baseline_control_contract_id",
    "baseline_control_system_version",
    "source_availability_code",
    "observation_cutoff_utc",
    "soc_anchor_time_utc",
    "adaptive_calibration_state_id",
    "evaluation_pair_id",
    "input_snapshot_id",
    "baseline_publication_signature",
    "forecast_refresh_kind",
    "forecast_verification_eligible",
    "independent_cycle",
)
ENSEMBLE_SEMANTIC_IDENTITY_FIELDS = (
    "ForecastModelContractID",
    "ForecastSystemVersion",
    "FeatureSetVersion",
    "FeatureSetDigest",
    "ForecastCodeRevision",
    "CandidateLane",
    "LocalFeatureContractID",
    "BaselineControlContractID",
    "BaselineControlSystemVersion",
)
ENSEMBLE_INPUT_ATTRS = (
    "initial_soc_time",
    "initial_soc_pct",
    "solar_calibration_factor_w_per_wm2",
    "solar_mos_factor_by_lead_bucket",
    "solar_calibration_contract_id",
    "battery_capacity_kwh",
    "battery_usable_capacity_kwh",
    "battery_charge_efficiency",
    "battery_discharge_efficiency",
    "battery_parasitic_load_w",
    "battery_max_charge_w",
    "battery_max_discharge_w",
    "battery_calibration_sample_count",
    "battery_calibration_confidence",
    "battery_energy_model",
    "load_bias_correction_w",
    "forecast_load_w",
    "forecast_load_p10_w",
    "forecast_load_p50_w",
    "forecast_load_p90_w",
    "load_model",
    "load_model_version",
    "load_mode",
    "load_mode_source",
    "load_mode_active_kits",
    "load_mode_signature",
    "load_state_contract",
    "load_state_hold_policy",
    "load_state_uncertainty_source",
    "load_state_dynamics",
    "load_state_dynamics_signature",
    *ENSEMBLE_FORECAST_PROVENANCE_ATTRS,
)
NUMERIC_ENSEMBLE_INPUT_ATTRS = {
    "initial_soc_pct",
    "solar_calibration_factor_w_per_wm2",
    "battery_capacity_kwh",
    "battery_usable_capacity_kwh",
    "battery_charge_efficiency",
    "battery_discharge_efficiency",
    "battery_parasitic_load_w",
    "battery_max_charge_w",
    "battery_max_discharge_w",
    "load_bias_correction_w",
    "forecast_load_w",
    "forecast_load_p10_w",
    "forecast_load_p50_w",
    "forecast_load_p90_w",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _ensemble_refresh_reasons(
    current_attrs: Mapping[str, object],
    deterministic_attrs: Mapping[str, object],
) -> list[str]:
    """Return deterministic inputs that require a same-cycle re-anchoring run."""
    reasons: list[str] = []
    deterministic_aliases = {
        "forecast_identity_id": "deterministic_forecast_identity_id",
        "source_cycle_set_id": "deterministic_source_cycle_set_id",
        "source_manifest_digest": "deterministic_source_manifest_digest",
    }
    for name in ENSEMBLE_INPUT_ATTRS:
        desired = deterministic_attrs.get(name)
        if desired is None:
            continue
        alias = deterministic_aliases.get(name)
        current = current_attrs.get(alias, current_attrs.get(name)) if alias else current_attrs.get(name)
        if name in NUMERIC_ENSEMBLE_INPUT_ATTRS:
            try:
                matches = bool(np.isclose(float(current), float(desired), rtol=0.0, atol=1e-6))
            except (TypeError, ValueError):
                matches = False
        else:
            matches = str(current) == str(desired)
        if not matches:
            reasons.append(name)
    return reasons


def _normalise_utc_timestamp(value: object, *, field: str) -> pd.Timestamp:
    """Return a timezone-naive UTC timestamp suitable for xarray coordinates."""
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a valid timestamp") from exc
    if pd.isna(timestamp):
        raise ValueError(f"{field} must be a valid timestamp")
    if timestamp.tz is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp


def _validate_power_cutoff_anchor(
    deterministic_attrs: Mapping[str, object],
    cutoff: pd.Timestamp | None,
) -> None:
    """Bind an explicit power cutoff to the immutable deterministic SOC anchor."""
    if cutoff is None:
        return
    if not deterministic_attrs.get("initial_soc_time"):
        raise ValueError("Deterministic forecast is missing initial_soc_time for the power cutoff")
    deterministic_anchor = _normalise_utc_timestamp(
        deterministic_attrs["initial_soc_time"],
        field="deterministic initial_soc_time",
    )
    if deterministic_anchor != cutoff:
        raise ValueError(
            "Ensemble power cutoff must equal deterministic initial_soc_time "
            f"({cutoff.isoformat()} != {deterministic_anchor.isoformat()})"
        )


def _power_at_or_before_cutoff(
    power: xr.Dataset,
    cutoff: pd.Timestamp | None,
    *,
    history_days: float = ENSEMBLE_POWER_HISTORY_DAYS,
) -> xr.Dataset:
    """Return a lazy, bounded power view ending at the paired SOC anchor.

    The ensemble skill surface needs seven retained days, its trailing 24-hour
    verification window, and up to 96 hours of issue-time persistence anchors.
    Keeping that complete window prevents the millions of older one-second APS
    samples from being materialised by each downstream pandas conversion.
    """
    retained_days = float(history_days)
    if not math.isfinite(retained_days) or retained_days < ENSEMBLE_POWER_MIN_HISTORY_DAYS:
        raise ValueError(
            "Ensemble power history must retain at least "
            f"{ENSEMBLE_POWER_MIN_HISTORY_DAYS:g} days (7-day skill, 24-hour "
            "verification, and 96-hour issue anchors)"
        )
    if "time" not in power.coords:
        raise ValueError("Power dataset has no time coordinate for the ensemble cutoff")
    times = pd.to_datetime(np.asarray(power["time"].values), errors="coerce", utc=True).tz_localize(None)
    valid_times = times[~times.isna()]
    if not len(valid_times):
        if cutoff is None:
            raise ValueError("Power dataset has no valid timestamps")
        raise ValueError(f"No APS power data exist at or before ensemble cutoff {cutoff.isoformat()}")
    end = pd.Timestamp(valid_times.max()) if cutoff is None else _normalise_utc_timestamp(cutoff, field="power cutoff")
    eligible = (~times.isna()) & (times <= end)
    if not eligible.any():
        raise ValueError(f"No APS power data exist at or before ensemble cutoff {end.isoformat()}")
    start = (end - pd.Timedelta(days=retained_days)).normalize()
    fields = [name for name in ENSEMBLE_POWER_SOURCE_FIELDS if name in power]
    selected_source = power[fields]
    if times.is_monotonic_increasing:
        selected = selected_source.sel(time=slice(start, end))
    else:
        selected = selected_source.isel(
            time=eligible & (times >= start)
        )
    frame = _power_frame(selected[["BatterySOC"]] if "BatterySOC" in selected else selected)
    if frame.empty or "BatterySOC" not in frame:
        raise ValueError(f"No APS BatterySOC data exist at or before ensemble cutoff {end.isoformat()}")
    latest_time, _ = latest_finite(frame["BatterySOC"])
    if cutoff is not None and latest_time != end:
        raise ValueError(
            "APS power does not contain a finite BatterySOC sample at the exact ensemble cutoff "
            f"{end.isoformat()} (latest is {latest_time.isoformat()})"
        )
    selected.attrs = dict(selected.attrs)
    selected.attrs["ensemble_power_history_days"] = f"{retained_days:g}"
    selected.attrs["ensemble_power_window_start_utc"] = start.isoformat()
    selected.attrs["ensemble_power_window_end_utc"] = end.isoformat()
    return selected


def latest_ensemble_cycle(*, source: str = DEFAULT_OPEN_DATA_SOURCE) -> pd.Timestamp:
    from ecmwf.opendata import Client

    client = Client(source=source)
    candidates = []
    for hour in (0, 12):
        value = client.latest(stream="enfo", type="pf", time=hour)
        candidates.append(pd.Timestamp(value).tz_localize(None))
    return max(candidates)


def retrieve_ensemble_grib(
    target: Path,
    *,
    cycle: pd.Timestamp,
    horizon_hours: int = DEFAULT_HORIZON_HOURS,
    source: str = DEFAULT_OPEN_DATA_SOURCE,
) -> Path:
    from ecmwf.opendata import Client

    requested_horizon = int(horizon_hours) + 24
    steps = list(range(0, requested_horizon + 1, 3))
    target.parent.mkdir(parents=True, exist_ok=True)
    Client(source=source, preserve_request_order=True).retrieve(
        date=cycle.strftime("%Y%m%d"),
        time=int(cycle.hour),
        stream="enfo",
        type="pf",
        number=list(ENSEMBLE_MEMBERS),
        levtype="sfc",
        param="ssrd",
        step=steps,
        target=str(target),
    )
    return target


def open_ensemble_site(path: Path, *, latitude: float, longitude: float) -> xr.Dataset:
    if path.suffix.lower() in {".grib", ".grib2", ".grb", ".grb2"}:
        return _open_grib_ensemble_site(path, latitude=latitude, longitude=longitude)
    ds = xr.open_dataset(path)
    if "surface_solar_radiation_downwards" in ds and "ssrd" not in ds:
        ds = ds.rename({"surface_solar_radiation_downwards": "ssrd"})
    if "ssrd" not in ds:
        raise KeyError("ECMWF ensemble does not contain ssrd")
    lat_name = "latitude" if "latitude" in ds.coords else "lat" if "lat" in ds.coords else None
    lon_name = "longitude" if "longitude" in ds.coords else "lon" if "lon" in ds.coords else None
    if lat_name and lon_name:
        select_lon = longitude
        if float(ds[lon_name].min()) >= 0.0 and select_lon < 0.0:
            select_lon %= 360.0
        # cfgrib's label-based nearest selection can cause its backend to allocate
        # the complete member x step x global-grid array. Resolve the two scalar
        # indices from the small coordinate vectors and slice before loading.
        latitude_index = int(np.abs(np.asarray(ds[lat_name].values) - latitude).argmin())
        longitude_index = int(np.abs(np.asarray(ds[lon_name].values) - select_lon).argmin())
        ds = ds.isel({lat_name: latitude_index, lon_name: longitude_index})
    return ds.load()


def _open_grib_ensemble_site(path: Path, *, latitude: float, longitude: float) -> xr.Dataset:
    """Stream one site from a multi-member GRIB without allocating its global grid."""
    from eccodes import (
        codes_get,
        codes_grib_find_nearest,
        codes_grib_new_from_file,
        codes_release,
    )

    records: list[tuple[int, int, float]] = []
    cycle: pd.Timestamp | None = None
    selected_latitude: float | None = None
    selected_longitude: float | None = None
    with path.open("rb") as handle:
        while True:
            grib_id = codes_grib_new_from_file(handle)
            if grib_id is None:
                break
            try:
                nearest = codes_grib_find_nearest(grib_id, latitude, longitude)[0]
                member = int(codes_get(grib_id, "number"))
                step = int(codes_get(grib_id, "step"))
                records.append((member, step, float(nearest["value"])))
                selected_latitude = float(nearest["lat"])
                selected_longitude = float(nearest["lon"])
                if cycle is None:
                    date = int(codes_get(grib_id, "dataDate"))
                    time = int(codes_get(grib_id, "dataTime"))
                    cycle = pd.Timestamp(datetime.strptime(f"{date:08d}{time:04d}", "%Y%m%d%H%M"))
            finally:
                codes_release(grib_id)

    if not records or cycle is None:
        raise ValueError(f"No ECMWF ensemble messages found in {path}")
    frame = pd.DataFrame(records, columns=["number", "step", "ssrd"])
    frame = frame.drop_duplicates(["number", "step"], keep="last")
    values = frame.pivot(index="number", columns="step", values="ssrd").sort_index().sort_index(axis=1)
    steps = pd.to_timedelta(values.columns.to_numpy(dtype=np.int64), unit="h")
    return xr.Dataset(
        {"ssrd": (("number", "step"), values.to_numpy(dtype=np.float64))},
        coords={
            "number": values.index.to_numpy(dtype=np.int64),
            "step": steps.to_numpy(),
            "time": cycle.to_datetime64(),
            "valid_time": ("step", (cycle + steps).to_numpy()),
            "latitude": selected_latitude,
            "longitude": selected_longitude,
        },
    )


def _member_dimension(ds: xr.Dataset) -> str:
    for name in ("number", "member", "realization"):
        if name in ds["ssrd"].dims:
            return name
    raise ValueError("ECMWF ensemble ssrd has no member dimension")


def _ensemble_site_forcing_sha256(solar: xr.Dataset) -> str:
    """Hash the exact site-level irradiance series consumed by every member.

    The operational download is intentionally discarded, so provenance binds
    the compact site forcing rather than a retained global GRIB.  Hashing the
    converted time/irradiance series makes the identity stable across the
    in-memory extraction and its site-Zarr cache representation.
    """
    member_dim = _member_dimension(solar)
    members = np.asarray(solar[member_dim].values).reshape(-1)
    digest = hashlib.sha256()
    header = {
        "schema": 1,
        "representation": "ecmwf_ensemble_site_interval_mean_ghi",
        "member_dimension": member_dim,
        "member_count": int(members.size),
    }
    digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(b"\0")
    for member in members:
        member_value: int | float | str
        try:
            numeric = float(member)
            member_value = int(numeric) if numeric.is_integer() else numeric
        except (TypeError, ValueError):
            member_value = str(member)
        series = solar_irradiance_from_ssrd(solar.sel({member_dim: member}))
        if series.empty:
            raise ValueError(f"ECMWF ensemble member {member_value} has no site irradiance")
        times = pd.DatetimeIndex(series.index).to_numpy(dtype="datetime64[ns]").astype("<i8")
        values = series.to_numpy(dtype=np.float64)
        values = np.where(np.isnan(values), np.nan, values).astype("<f8", copy=False)
        digest.update(
            json.dumps(member_value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\0")
        digest.update(times.tobytes(order="C"))
        digest.update(b"\0")
        digest.update(values.tobytes(order="C"))
        digest.update(b"\0")
    return digest.hexdigest()


def _ensemble_composite_source_identity(
    solar: xr.Dataset,
    deterministic_attrs: Mapping[str, object],
    *,
    cycle_time: pd.Timestamp,
    source: str,
) -> dict[str, str]:
    """Bind deterministic inputs and exact ENS site forcing into one source set."""
    deterministic_cycle = str(deterministic_attrs.get("source_cycle_set_id", "")).strip()
    deterministic_manifest = str(
        deterministic_attrs.get("source_manifest_digest", "")
    ).strip()
    if not deterministic_cycle or not deterministic_manifest.startswith("sha256:"):
        raise ValueError(
            "Deterministic forecast lacks checksum-bound source-cycle provenance"
        )
    cycle = _normalise_utc_timestamp(cycle_time, field="ensemble cycle_time")
    site_digest = _ensemble_site_forcing_sha256(solar)
    cycle_payload = {
        "schema": 1,
        "deterministic_source_cycle_set_id": deterministic_cycle,
        "ensemble_cycle_time_utc": cycle.isoformat(),
        "ensemble_source_provider": str(source),
        "ensemble_site_forcing_sha256": f"sha256:{site_digest}",
    }
    manifest_payload = {
        **cycle_payload,
        "deterministic_source_manifest_digest": deterministic_manifest,
    }
    cycle_digest = hashlib.sha256(
        json.dumps(cycle_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest_digest = hashlib.sha256(
        json.dumps(manifest_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "source_cycle_set_id": f"power-ensemble-source-set-v1-{cycle_digest[:20]}",
        "source_manifest_digest": f"sha256:{manifest_digest}",
        "deterministic_source_cycle_set_id": deterministic_cycle,
        "deterministic_source_manifest_digest": deterministic_manifest,
        "deterministic_forecast_identity_id": str(
            deterministic_attrs.get("forecast_identity_id", "")
        ),
        "ensemble_source_cycle_time_utc": cycle.isoformat(),
        "ensemble_source_provider": str(source),
        "ensemble_site_forcing_sha256": f"sha256:{site_digest}",
        "ensemble_source_identity_contract": "deterministic-plus-ens-site-forcing-v1",
    }


def _deterministic_controlled_load(
    deterministic: xr.Dataset,
    fallback_w: float,
) -> ControlledLoadEstimate:
    """Read the finite-state load distribution published by the central run."""

    def value(name: str, fallback: float) -> float:
        try:
            candidate = float(deterministic.attrs.get(name, fallback))
        except (TypeError, ValueError):
            return float(fallback)
        return candidate if np.isfinite(candidate) else float(fallback)

    try:
        sample_count = int(float(deterministic.attrs.get("load_mode_learning_observations", 0) or 0))
    except (TypeError, ValueError):
        sample_count = 0

    p50 = value("forecast_load_p50_w", fallback_w)
    return ControlledLoadEstimate(
        p10_w=value("forecast_load_p10_w", p50),
        p50_w=p50,
        p90_w=value("forecast_load_p90_w", p50),
        source=str(deterministic.attrs.get("load_anchor_method", "legacy_central_load")),
        sample_count=sample_count,
    ).validated()


def apply_operational_soc_threshold(ds: xr.Dataset) -> xr.Dataset:
    """Refresh threshold-derived fields without rerunning the ECMWF ensemble."""
    if "BatterySOCForecastEnsemble" not in ds:
        raise KeyError("Ensemble forecast does not contain BatterySOCForecastEnsemble")
    out = ds.copy()
    obsolete = [
        name
        for name in out.data_vars
        if name.startswith("BatterySOCBelow")
        and name.endswith("Probability")
        and name != SOC_BELOW_THRESHOLD_PROBABILITY_FIELD
    ]
    if obsolete:
        out = out.drop_vars(obsolete)
    probability = np.mean(
        np.asarray(out["BatterySOCForecastEnsemble"].values, dtype=np.float64)
        < MINIMUM_OPERATIONAL_SOC_PCT,
        axis=0,
    ).astype(np.float32)
    out[SOC_BELOW_THRESHOLD_PROBABILITY_FIELD] = (("time",), probability)
    out[SOC_BELOW_THRESHOLD_PROBABILITY_FIELD].attrs["units"] = "1"
    out.attrs["minimum_operational_soc_pct"] = f"{MINIMUM_OPERATIONAL_SOC_PCT:g}"
    return out


def build_ensemble_dataset(
    power: xr.Dataset,
    deterministic: xr.Dataset,
    solar_ensemble: xr.Dataset,
    *,
    horizon_hours: int = DEFAULT_HORIZON_HOURS,
) -> xr.Dataset:
    soc_power = power[["BatterySOC"]] if "BatterySOC" in power else power
    frame = _power_frame(soc_power)
    latest_time, latest_soc = latest_finite(frame["BatterySOC"])
    member_dim = _member_dimension(solar_ensemble)
    member_values = np.asarray(solar_ensemble[member_dim].values)
    solar_factor = float(deterministic.attrs.get("solar_calibration_factor_w_per_wm2", 1.0))
    capacity_kwh = float(deterministic.attrs.get("battery_capacity_kwh", DEFAULT_BATTERY_CAPACITY_KWH))
    battery_model = (
        BatteryModel.from_attrs(deterministic.attrs, default_capacity_kwh=capacity_kwh)
        if "battery_energy_model" in deterministic.attrs
        else BatteryModel(
            usable_capacity_kwh=capacity_kwh,
            charge_efficiency=1.0,
            discharge_efficiency=1.0,
            max_charge_w=20_000.0,
            max_discharge_w=20_000.0,
        )
    )
    correction_w = float(deterministic.attrs.get("load_bias_correction_w", 0.0))

    member_irradiance: list[pd.Series] = []
    for member in member_values:
        series = solar_irradiance_from_ssrd(solar_ensemble.sel({member_dim: member}))
        series = series[(series.index >= latest_time) & (series.index <= latest_time + pd.Timedelta(hours=horizon_hours))]
        if len(series) >= 2:
            member_irradiance.append(series)
    if not member_irradiance:
        raise ValueError("No ECMWF ensemble members overlap the requested SOC horizon")
    common_times = member_irradiance[0].index
    common_times = common_times[(common_times >= latest_time) & (common_times <= latest_time + pd.Timedelta(hours=horizon_hours))]
    fallback_level = np.nan
    if "ForecastLoadWatts" in deterministic and "time" in deterministic:
        source_load = np.asarray(deterministic["ForecastLoadWatts"].values, dtype=np.float64)
        finite_load = source_load[np.isfinite(source_load)]
        fallback_level = float(np.nanmedian(finite_load)) if finite_load.size else np.nan
    if not np.isfinite(fallback_level):
        load_frame = _power_frame(power)
        raw_load = build_historical_load_forecast(
            load_frame,
            common_times,
            end=latest_time,
            calibration_days=7,
        )
        bounded_load, _ = _bounded_load_profile(raw_load, correction_w)
        fallback_level = float(bounded_load.median())
    # This operational ensemble represents the detected system as it is now.
    # Planned instrument schedules are evaluated by the separate operating-plan
    # product; only the ECMWF solar members may vary here.
    controlled_load = _deterministic_controlled_load(
        deterministic,
        fallback_level,
    )
    # Keep the deterministic lead-dependent solar calibration when propagating
    # ECMWF members. The remaining member spread then combines meteorological
    # forcing with observed, mode-conditioned station-load variability.
    if {"ForecastSolarWatts", "ECMWFSolarIrradiance"}.issubset(deterministic.data_vars) and "time" in deterministic:
        deterministic_times = pd.DatetimeIndex(deterministic["time"].values)
        deterministic_solar = pd.Series(np.asarray(deterministic["ForecastSolarWatts"].values, dtype=np.float64), index=deterministic_times)
        deterministic_irradiance = pd.Series(np.asarray(deterministic["ECMWFSolarIrradiance"].values, dtype=np.float64), index=deterministic_times)
        inferred_factor = (deterministic_solar / deterministic_irradiance.where(deterministic_irradiance > 1.0)).replace([np.inf, -np.inf], np.nan)
        solar_factor_profile = inferred_factor.reindex(common_times, method="nearest", tolerance=pd.Timedelta(hours=2))
        solar_factor_profile = solar_factor_profile.ffill().bfill().fillna(solar_factor).clip(lower=0.0)
    else:
        solar_factor_profile = pd.Series(float(solar_factor), index=common_times)
    rng = np.random.default_rng(int(latest_time.value % (2**32 - 1)))
    state_dynamics = None
    if str(deterministic.attrs.get("load_state_contract", "")) == CONTROLLED_LOAD_CONTRACT:
        try:
            raw_dynamics = json.loads(str(deterministic.attrs.get("load_state_dynamics", "{}")))
            if isinstance(raw_dynamics, Mapping) and raw_dynamics:
                state_dynamics = StateLoadDynamics.from_dict(raw_dynamics)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            state_dynamics = None
    load_times = _forecast_integration_times(latest_time, common_times)
    load_member_profiles, load_phase_codes = controlled_load_member_profiles(
        state_dynamics,
        load_times,
        controlled_load,
        len(member_irradiance),
        seed=int(latest_time.value % (2**32 - 1)),
    )
    permutation = rng.permutation(len(member_irradiance))
    load_member_profiles = load_member_profiles[permutation]
    load_phase_codes = load_phase_codes[permutation]
    validate_state_held_load(
        np.zeros(len(load_times), dtype=np.int16),
        load_member_profiles,
        phase_codes=load_phase_codes,
    )
    parameter_spread = 0.06 if battery_model.calibration_confidence == "calibrated" else 0.10
    capacity_rank = np.linspace(-1.0, 1.0, len(member_irradiance), dtype=np.float64)
    charge_rank = capacity_rank.copy()
    discharge_rank = capacity_rank.copy()
    rng.shuffle(capacity_rank)
    rng.shuffle(charge_rank)
    rng.shuffle(discharge_rank)

    soc_rows = []
    irr_rows = []
    solar_rows = []
    load_rows = []
    capacity_rows = []
    charge_efficiency_rows = []
    discharge_efficiency_rows = []
    output_times: pd.DatetimeIndex | None = None
    for member_index, irradiance in enumerate(member_irradiance):
        irradiance = irradiance.reindex(common_times).interpolate().ffill().bfill()
        member_load = pd.Series(load_member_profiles[member_index], index=load_times, dtype=np.float64)
        member_model = replace(
            battery_model,
            usable_capacity_kwh=battery_model.usable_capacity_kwh
            * (1.0 + parameter_spread * float(capacity_rank[member_index])),
            charge_efficiency=float(
                np.clip(
                    battery_model.charge_efficiency * (1.0 + 0.03 * float(charge_rank[member_index])),
                    0.65,
                    1.0,
                )
            ),
            discharge_efficiency=float(
                np.clip(
                    battery_model.discharge_efficiency
                    * (1.0 + 0.03 * float(discharge_rank[member_index])),
                    0.65,
                    1.0,
                )
            ),
        ).validated()
        result = integrate_soc_forecast(
            initial_soc=latest_soc,
            initial_time=latest_time,
            irradiance=irradiance,
            solar_factor=solar_factor_profile,
            load_w=member_load,
            capacity_kwh=member_model.usable_capacity_kwh,
            battery_model=member_model,
        )
        output_times = pd.DatetimeIndex(result.index)
        soc_rows.append(result["BatterySOCForecast"].to_numpy(dtype=np.float32))
        irr_rows.append(result["ECMWFSolarIrradiance"].to_numpy(dtype=np.float32))
        solar_rows.append(result["ForecastSolarWatts"].to_numpy(dtype=np.float32))
        load_rows.append(result["ForecastLoadWatts"].to_numpy(dtype=np.float32))
        capacity_rows.append(member_model.usable_capacity_kwh)
        charge_efficiency_rows.append(member_model.charge_efficiency)
        discharge_efficiency_rows.append(member_model.discharge_efficiency)
    assert output_times is not None
    soc = np.asarray(soc_rows, dtype=np.float32)
    out = xr.Dataset(
        {
            "BatterySOCForecastEnsemble": (("member", "time"), soc),
            "ECMWFSolarIrradianceEnsemble": (("member", "time"), np.asarray(irr_rows, dtype=np.float32)),
            "ForecastSolarWattsEnsemble": (("member", "time"), np.asarray(solar_rows, dtype=np.float32)),
            "ForecastLoadWattsEnsemble": (("member", "time"), np.asarray(load_rows, dtype=np.float32)),
            "ForecastLoadPhaseCodeEnsemble": (("member", "time"), np.asarray(load_phase_codes, dtype=np.int8)),
            "BatteryUsableCapacityKWhEnsemble": (("member",), np.asarray(capacity_rows, dtype=np.float32)),
            "BatteryChargeEfficiencyEnsemble": (("member",), np.asarray(charge_efficiency_rows, dtype=np.float32)),
            "BatteryDischargeEfficiencyEnsemble": (("member",), np.asarray(discharge_efficiency_rows, dtype=np.float32)),
            "BatterySOCForecastP10": (("time",), np.nanquantile(soc, 0.10, axis=0).astype(np.float32)),
            "BatterySOCForecastP50": (("time",), np.nanquantile(soc, 0.50, axis=0).astype(np.float32)),
            "BatterySOCForecastP90": (("time",), np.nanquantile(soc, 0.90, axis=0).astype(np.float32)),
            "BatterySOCForecastMinimum": (("time",), np.nanmin(soc, axis=0).astype(np.float32)),
            "BatterySOCForecastMaximum": (("time",), np.nanmax(soc, axis=0).astype(np.float32)),
        },
        coords={"member": np.arange(1, soc.shape[0] + 1, dtype=np.int16), "time": output_times.to_numpy(dtype="datetime64[ns]")},
        attrs={
            **{
                name: str(deterministic.attrs.get(name, ""))
                for name in ENSEMBLE_FORECAST_PROVENANCE_ATTRS
            },
            "power_soc_ensemble_forecast_product": "true",
            "generated_at_utc": _utc_now(),
            "initial_soc_time": latest_time.isoformat(),
            "initial_soc_pct": f"{latest_soc:.6g}",
            "forecast_horizon_hours": str(int(horizon_hours)),
            "ensemble_members": str(int(soc.shape[0])),
            "solar_calibration_factor_w_per_wm2": f"{solar_factor:.6g}",
            "solar_mos_factor_by_lead_bucket": str(
                deterministic.attrs.get("solar_mos_factor_by_lead_bucket", "{}")
            ),
            "solar_calibration_contract_id": str(
                deterministic.attrs.get("solar_calibration_contract_id", "")
            ),
            **battery_model.attrs(),
            "load_bias_correction_w": f"{correction_w:.6g}",
            "forecast_load_w": str(deterministic.attrs.get("forecast_load_w", "")),
            "forecast_load_p10_w": f"{controlled_load.p10_w:.6g}",
            "forecast_load_p50_w": f"{controlled_load.p50_w:.6g}",
            "forecast_load_p90_w": f"{controlled_load.p90_w:.6g}",
            "load_model": str(deterministic.attrs.get("load_model", "kit_mode_persistence_v4")),
            "load_model_version": str(deterministic.attrs.get("load_model_version", "4")),
            "load_mode": str(deterministic.attrs.get("load_mode", "unknown")),
            "load_mode_source": str(deterministic.attrs.get("load_mode_source", "unknown")),
            "load_mode_active_kits": str(deterministic.attrs.get("load_mode_active_kits", "")),
            "load_mode_signature": str(deterministic.attrs.get("load_mode_signature", "")),
            "load_state_contract": str(
                deterministic.attrs.get("load_state_contract", CONTROLLED_LOAD_CONTRACT)
            ),
            "load_state_hold_policy": str(
                deterministic.attrs.get("load_state_hold_policy", STATE_HOLD_POLICY)
            ),
            "load_state_uncertainty_source": str(
                deterministic.attrs.get("load_state_uncertainty_source", "exact_state_observations")
            ),
            "load_state_dynamics": str(deterministic.attrs.get("load_state_dynamics", "{}")),
            "load_state_dynamics_signature": str(
                deterministic.attrs.get("load_state_dynamics_signature", "")
            ),
            "load_uncertainty": (
                "exact-state startup duration and fan-phase uncertainty independently paired with ECMWF solar members"
                if state_dynamics is not None
                else "stationary exact-state load distribution independently paired with ECMWF solar members"
            ),
            "battery_parameter_uncertainty": f"member-wise usable-capacity spread plus or minus {100.0 * parameter_spread:.0f}% and efficiency spread plus or minus 3%",
            "scenario_scope": "current_system_only",
            "minimum_operational_soc_pct": f"{MINIMUM_OPERATIONAL_SOC_PCT:g}",
            "source": "ECMWF IFS perturbed ssrd members plus APS power history",
        },
    )
    out = apply_operational_soc_threshold(out)
    for name in out.data_vars:
        out[name].attrs["units"] = "1" if name.endswith("Probability") else "%" if "SOC" in name else "W m-2" if "Irradiance" in name else "W"
    out["ForecastLoadPhaseCodeEnsemble"].attrs.update(
        {
            "units": "1",
            "phase_mapping": json.dumps({str(code): name for name, code in PHASE_CODES.items()}, sort_keys=True),
        }
    )
    return out


def _write_forecast(ds: xr.Dataset, path: Path) -> None:
    tmp = path.with_name(f"{path.name}.tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.chunk({"member": min(ds.sizes["member"], 10), "time": min(ds.sizes["time"], 64)}).to_zarr(tmp, mode="w", consolidated=True)
    if path.exists():
        shutil.rmtree(path)
    tmp.rename(path)


def _ensemble_site_cache_path(cache_dir: Path, cycle: pd.Timestamp, horizon_hours: int) -> Path:
    stamp = cycle.strftime("%Y%m%dT%H%M%SZ")
    return cache_dir / f"ecmwf_ens_ssrd_{stamp}_h{int(horizon_hours)}.site.zarr"


def _write_ensemble_site_cache(ds: xr.Dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    ds.to_zarr(tmp, mode="w", consolidated=True)
    if path.exists():
        shutil.rmtree(path)
    tmp.rename(path)


def _prune_ensemble_site_cache(cache_dir: Path, *, keep: int = 4) -> None:
    paths = sorted(cache_dir.glob("ecmwf_ens_ssrd_*_h*.site.zarr"), key=lambda path: path.name)
    for path in paths[:-keep]:
        shutil.rmtree(path)


def _archive_row(forecast: xr.Dataset) -> xr.Dataset:
    issue = pd.Timestamp(forecast.attrs["initial_soc_time"])
    times = pd.DatetimeIndex(forecast["time"].values)
    cycle = pd.Timestamp(forecast.attrs.get("ecmwf_cycle_time", issue))
    if cycle.tz is not None:
        cycle = cycle.tz_convert("UTC").tz_localize(None)
    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {
            "ForecastValidTime": (("issue_time", "forecast_step"), times.to_numpy(dtype="datetime64[ns]")[None, :]),
            "ForecastLeadHours": (("issue_time", "forecast_step"), (((times - issue) / pd.Timedelta(hours=1)).to_numpy(dtype=np.float32))[None, :]),
            "BatterySOCForecastEnsemble": (("issue_time", "member", "forecast_step"), forecast["BatterySOCForecastEnsemble"].values[None, :, :]),
            "ECMWFCycleTime": (("issue_time",), np.array([cycle.to_datetime64()], dtype="datetime64[ns]")),
    }
    archive_fields = {
        "ForecastModelName": "forecast_model_name",
        "ForecastModelVersion": "forecast_model_version",
        "ForecastModelContractID": "forecast_model_contract_id",
        "ForecastIdentityID": "forecast_identity_id",
        "ForecastSystemVersion": "forecast_system_version",
        "FeatureSetVersion": "feature_set_version",
        "FeatureSetDigest": "feature_set_digest",
        "TrainingCutoffUTC": "training_cutoff_utc",
        "ForecastCodeRevision": "forecast_code_revision",
        "SourceCycleSetID": "source_cycle_set_id",
        "SourceManifestDigest": "source_manifest_digest",
        "DegradedModeCode": "degraded_mode_code",
        "CandidateLane": "candidate_lane",
        "LocalFeatureContractID": "local_feature_contract_id",
        "BaselineControlContractID": "baseline_control_contract_id",
        "BaselineControlSystemVersion": "baseline_control_system_version",
        "SourceAvailabilityCode": "source_availability_code",
        "ObservationCutoffUTC": "observation_cutoff_utc",
        "SOCAuthoringAnchorTimeUTC": "soc_anchor_time_utc",
        "AdaptiveCalibrationStateID": "adaptive_calibration_state_id",
        "EvaluationPairID": "evaluation_pair_id",
        "InputSnapshotID": "input_snapshot_id",
        "BaselinePublicationSignature": "baseline_publication_signature",
        "ForecastRefreshKind": "forecast_refresh_kind",
        "LoadMode": "load_mode",
        "LoadModelVersion": "load_model_version",
    }
    for archive_name, attr_name in archive_fields.items():
        data_vars[archive_name] = (
            ("issue_time",),
            np.asarray([str(forecast.attrs.get(attr_name, ""))], dtype="U512"),
        )
    default_independent = str(forecast.attrs.get("forecast_refresh_kind", "ecmwf_cycle")) != "cached_reanchor"
    verification_eligible = (
        str(forecast.attrs.get("forecast_verification_eligible", str(default_independent)))
        .strip()
        .lower()
        == "true"
    )
    independent_cycle = (
        str(forecast.attrs.get("independent_cycle", str(default_independent)))
        .strip()
        .lower()
        == "true"
    )
    data_vars["ForecastVerificationEligible"] = (
        ("issue_time",), np.asarray([verification_eligible], dtype=bool)
    )
    data_vars["IndependentCycle"] = (
        ("issue_time",), np.asarray([independent_cycle], dtype=bool)
    )
    return xr.Dataset(
        data_vars,
        coords={
            "issue_time": np.array([issue.to_datetime64()], dtype="datetime64[ns]"),
            "member": forecast["member"].values,
            "forecast_step": np.arange(len(times), dtype=np.int16),
        },
    )


def append_ensemble_archive(forecast: xr.Dataset, path: Path, *, retention_days: float = 21.0) -> xr.Dataset:
    row = _archive_row(forecast)
    previous = xr.open_zarr(path, chunks={}).load() if path.exists() else None
    if previous is not None:
        steps = np.arange(max(previous.sizes["forecast_step"], row.sizes["forecast_step"]), dtype=np.int16)
        combined = xr.concat([previous.reindex(forecast_step=steps), row.reindex(forecast_step=steps)], dim="issue_time")
        combined = combined.sortby("issue_time")
        combined = combined.isel(issue_time=~combined.indexes["issue_time"].duplicated(keep="last"))
    else:
        combined = row
    issue_times = pd.DatetimeIndex(combined.issue_time.values)
    cutoff = issue_times.max() - pd.Timedelta(days=retention_days)
    combined = combined.isel(issue_time=issue_times >= cutoff)
    for name in ("ForecastVerificationEligible", "IndependentCycle"):
        if name in combined:
            combined[name] = combined[name].fillna(True).astype(bool)
    for name, variable in list(combined.variables.items()):
        if variable.dtype.kind == "O":
            combined[name] = variable.fillna("").astype(str).load()
    tmp = path.with_name(f"{path.name}.tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    combined.chunk({"issue_time": 8, "member": 10, "forecast_step": 64}).to_zarr(tmp, mode="w", consolidated=True)
    if path.exists():
        shutil.rmtree(path)
    tmp.rename(path)
    return combined


def _crps_ensemble(members: np.ndarray, observation: float) -> float:
    members = members[np.isfinite(members)]
    if members.size < 2 or not np.isfinite(observation):
        return np.nan
    first = np.mean(np.abs(members - observation))
    second = 0.5 * np.mean(np.abs(members[:, None] - members[None, :]))
    return float(first - second)


def _independent_ensemble_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Keep one re-anchored ensemble per ECMWF cycle and valid time."""
    if rows.empty:
        return rows
    selected = rows.copy()
    cycle_key = selected["cycle_time"].astype(str)
    if "source_cycle_set_id" in selected:
        source = selected["source_cycle_set_id"].astype(str).str.strip()
        usable = ~source.str.lower().isin({"", "nan", "none"})
        cycle_key = cycle_key.where(~usable, source)
    selected["_independent_cycle_key"] = cycle_key
    return selected.sort_values("issue_time").drop_duplicates(
        ["_independent_cycle_key", "valid_time"], keep="last"
    )


def _ensemble_cycle_count(rows: pd.DataFrame) -> int:
    if rows.empty:
        return 0
    if "_independent_cycle_key" in rows:
        return int(rows["_independent_cycle_key"].nunique())
    if "source_cycle_set_id" in rows:
        values = rows["source_cycle_set_id"].astype(str).str.strip()
        values = values.loc[~values.str.lower().isin({"", "nan", "none"})]
        if not values.empty:
            return int(values.nunique())
    return int(rows["cycle_time"].nunique())


def _ensemble_semantic_issue_mask(archive: xr.Dataset) -> np.ndarray:
    """Select the newest complete semantic forecast identity in the archive."""
    issue_count = int(archive.sizes.get("issue_time", 0))
    if issue_count == 0:
        return np.zeros(0, dtype=bool)
    issue_times = pd.DatetimeIndex(archive["issue_time"].values)
    latest_index = int(np.argmax(issue_times.to_numpy(dtype="datetime64[ns]")))
    if "ForecastModelContractID" not in archive:
        return np.ones(issue_count, dtype=bool)
    contract_values = np.asarray(archive["ForecastModelContractID"].fillna("").values, dtype=str)
    target_contract = str(contract_values[latest_index]).strip()
    if not target_contract or target_contract.lower() in {"nan", "none"}:
        return np.ones(issue_count, dtype=bool)
    selected = contract_values == target_contract
    for name in ENSEMBLE_SEMANTIC_IDENTITY_FIELDS[1:]:
        if name not in archive:
            return np.zeros(issue_count, dtype=bool)
        values = np.asarray(archive[name].fillna("").values, dtype=str)
        target = str(values[latest_index]).strip()
        selected &= values == target
    return selected


def build_ensemble_skill_dataset(
    archive: xr.Dataset,
    power: xr.Dataset,
    *,
    retention_days: float = ENSEMBLE_SKILL_RETENTION_DAYS,
) -> xr.Dataset:
    soc_power = power[["BatterySOC"]] if "BatterySOC" in power else power
    frame = _power_frame(soc_power)
    observed = frame.get("BatterySOC", pd.Series(dtype=np.float64))
    end = pd.Timestamp(frame.index.max())
    times = pd.date_range((end - pd.Timedelta(days=retention_days)).floor("1h"), end.ceil("1h"), freq="1h")
    fields = [f"ForecastSOCCRPS_{bucket}" for bucket, _, _ in LEAD_BUCKETS]
    fields += [f"ForecastSOCCRPSSkill_{bucket}" for bucket, _, _ in LEAD_BUCKETS]
    fields += [f"ForecastSOCCRPS{suffix}_{bucket}" for bucket, _, _ in LEAD_BUCKETS for suffix in ("Samples", "Cycles")]
    fields += [
        "ForecastSOCIntervalCoverage80",
        "ForecastSOCIntervalCoverage80Samples",
        "ForecastSOCIntervalCoverage80Cycles",
        SOC_BELOW_THRESHOLD_BRIER_FIELD,
        f"{SOC_BELOW_THRESHOLD_BRIER_FIELD}Reference",
        f"{SOC_BELOW_THRESHOLD_BRIER_FIELD}Skill",
        f"{SOC_BELOW_THRESHOLD_BRIER_FIELD}Samples",
        f"{SOC_BELOW_THRESHOLD_BRIER_FIELD}Cycles",
        "ForecastEnsembleCycles",
    ]
    values = {name: np.full(len(times), np.nan, dtype=np.float32) for name in fields}
    valid_times = pd.DatetimeIndex(archive["ForecastValidTime"].values.reshape(-1))
    leads = archive["ForecastLeadHours"].values.reshape(-1)
    ensembles = archive["BatterySOCForecastEnsemble"].values.transpose(0, 2, 1).reshape(-1, archive.sizes["member"])
    issues = np.repeat(pd.DatetimeIndex(archive.issue_time.values), archive.sizes["forecast_step"])
    semantic_issue_mask = _ensemble_semantic_issue_mask(archive)
    semantic_rows = np.repeat(semantic_issue_mask, archive.sizes["forecast_step"])

    def repeated_text(name: str) -> np.ndarray:
        if name not in archive:
            return np.full(len(issues), "", dtype="U1")
        return np.repeat(
            np.asarray(archive[name].fillna("").values, dtype=str).reshape(-1),
            archive.sizes["forecast_step"],
        )

    def repeated_bool(name: str, *, legacy_default: bool) -> np.ndarray:
        if name not in archive:
            return np.full(len(issues), legacy_default, dtype=bool)
        return np.repeat(
            np.asarray(archive[name].fillna(legacy_default).values, dtype=bool).reshape(-1),
            archive.sizes["forecast_step"],
        )

    source_cycle_set_ids = repeated_text("SourceCycleSetID")
    verification_eligible = repeated_bool(
        "ForecastVerificationEligible", legacy_default=True
    )
    independent_cycles = repeated_bool("IndependentCycle", legacy_default=True)
    if "ECMWFCycleTime" in archive:
        cycle_times = np.repeat(pd.DatetimeIndex(archive["ECMWFCycleTime"].values), archive.sizes["forecast_step"])
        cycle_times = cycle_times.where(~cycle_times.isna(), issues)
    else:
        cycle_times = issues
    observed_values = observed.reindex(valid_times, method="nearest", tolerance=pd.Timedelta(minutes=10)).to_numpy(dtype=np.float64)
    reference_values = observed.reindex(issues, method="nearest", tolerance=pd.Timedelta(minutes=10)).to_numpy(dtype=np.float64)
    rows = pd.DataFrame(
        {
            "valid_time": valid_times,
            "issue_time": issues,
            "cycle_time": cycle_times,
            "lead": leads,
            "observed": observed_values,
            "reference": reference_values,
            "source_cycle_set_id": source_cycle_set_ids,
            "semantic_identity_match": semantic_rows,
            "forecast_verification_eligible": verification_eligible,
            "independent_cycle": independent_cycles,
        }
    )
    rows["sample_index"] = np.arange(len(rows))
    rows = rows[
        np.isfinite(rows["observed"])
        & np.isfinite(rows["reference"])
        & (rows["valid_time"] <= end)
        & rows["semantic_identity_match"].astype(bool)
        & rows["forecast_verification_eligible"].astype(bool)
        & rows["independent_cycle"].astype(bool)
    ]
    for index, now in enumerate(times):
        selected = _independent_ensemble_rows(rows[(rows.valid_time > now - pd.Timedelta(hours=24)) & (rows.valid_time <= now)])
        if selected.empty:
            continue
        values["ForecastEnsembleCycles"][index] = float(_ensemble_cycle_count(selected))
        coverage = []
        brier = []
        brier_outcomes = []
        brier_cycles = []
        for bucket, start, stop in LEAD_BUCKETS:
            bucket_rows = selected[(selected.lead >= start) & (selected.lead < stop)]
            scores = []
            reference_scores = []
            for row in bucket_rows.itertuples(index=False):
                member_values = ensembles[int(row.sample_index)]
                scores.append(_crps_ensemble(member_values, float(row.observed)))
                reference_scores.append(abs(float(row.reference) - float(row.observed)))
                coverage.append(float(np.nanquantile(member_values, 0.1) <= row.observed <= np.nanquantile(member_values, 0.9)))
                probability = float(np.mean(member_values < MINIMUM_OPERATIONAL_SOC_PCT))
                outcome = float(row.observed < MINIMUM_OPERATIONAL_SOC_PCT)
                brier.append((probability - outcome) ** 2)
                brier_outcomes.append(outcome)
                brier_cycles.append(row.cycle_time)
            if scores:
                values[f"ForecastSOCCRPS_{bucket}"][index] = float(np.nanmean(scores))
                values[f"ForecastSOCCRPSSamples_{bucket}"][index] = float(len(scores))
                values[f"ForecastSOCCRPSCycles_{bucket}"][index] = float(
                    _ensemble_cycle_count(bucket_rows)
                )
                reference_score = float(np.nanmean(reference_scores))
                if np.isfinite(reference_score) and reference_score > 0.0:
                    values[f"ForecastSOCCRPSSkill_{bucket}"][index] = float(1.0 - np.nanmean(scores) / reference_score)
        if coverage:
            values["ForecastSOCIntervalCoverage80"][index] = float(np.mean(coverage))
            values["ForecastSOCIntervalCoverage80Samples"][index] = float(len(coverage))
            values["ForecastSOCIntervalCoverage80Cycles"][index] = float(
                _ensemble_cycle_count(selected)
            )
        if brier:
            values[SOC_BELOW_THRESHOLD_BRIER_FIELD][index] = float(np.mean(brier))
            values[f"{SOC_BELOW_THRESHOLD_BRIER_FIELD}Samples"][index] = float(len(brier))
            values[f"{SOC_BELOW_THRESHOLD_BRIER_FIELD}Cycles"][index] = float(
                _ensemble_cycle_count(selected)
            )
            event_rate = float(np.mean(brier_outcomes))
            reference_brier = event_rate * (1.0 - event_rate)
            values[f"{SOC_BELOW_THRESHOLD_BRIER_FIELD}Reference"][index] = reference_brier
            if reference_brier > 0.0:
                values[f"{SOC_BELOW_THRESHOLD_BRIER_FIELD}Skill"][index] = float(1.0 - np.mean(brier) / reference_brier)
    out = xr.Dataset({name: (("time",), data) for name, data in values.items()}, coords={"time": times.values})
    out.attrs = {
        "power_soc_ensemble_skill_product": "true",
        "generated_at_utc": _utc_now(),
        "source": "archived ECMWF SOC ensemble forecasts verified against APS BatterySOC",
        "verification_window_hours": "24",
        "minimum_operational_soc_pct": f"{MINIMUM_OPERATIONAL_SOC_PCT:g}",
        "forecast_identity_filter": "latest_complete_semantic_forecast_identity",
        **{
            attr_name: str(
                np.asarray(archive[archive_name].fillna("").values, dtype=str)[-1]
            )
            for archive_name, attr_name in {
                "ForecastModelContractID": "forecast_model_contract_id",
                "ForecastSystemVersion": "forecast_system_version",
                "FeatureSetVersion": "feature_set_version",
                "FeatureSetDigest": "feature_set_digest",
                "ForecastCodeRevision": "forecast_code_revision",
                "CandidateLane": "candidate_lane",
                "LocalFeatureContractID": "local_feature_contract_id",
                "BaselineControlContractID": "baseline_control_contract_id",
                "BaselineControlSystemVersion": "baseline_control_system_version",
            }.items()
            if archive_name in archive
        },
    }
    for name in out.data_vars:
        out[name].attrs["units"] = (
            "samples"
            if name.endswith("Samples")
            else "cycles"
            if name.endswith("Cycles")
            else "1"
            if name.endswith(("Coverage80", "Brier", "Reference", "Skill"))
            else "percentage points"
        )
    return out


def _write_time_product(ds: xr.Dataset, path: Path) -> None:
    tmp = path.with_name(f"{path.name}.tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    ds.chunk({"time": min(ds.sizes.get("time", 1), 168)}).to_zarr(tmp, mode="w", consolidated=True)
    if path.exists():
        shutil.rmtree(path)
    tmp.rename(path)


def generate(
    *,
    power_zarr: Path = POWER_ZARR_PATH,
    deterministic_zarr: Path = POWER_SOC_FORECAST_ZARR_PATH,
    output_zarr: Path = POWER_SOC_ENSEMBLE_ZARR_PATH,
    archive_zarr: Path = POWER_SOC_ENSEMBLE_ARCHIVE_ZARR_PATH,
    skill_zarr: Path = POWER_SOC_ENSEMBLE_SKILL_ZARR_PATH,
    input_forecast: Path | None = None,
    latitude: float = DEFAULT_LATITUDE,
    longitude: float = DEFAULT_LONGITUDE,
    horizon_hours: int = DEFAULT_HORIZON_HOURS,
    source: str = DEFAULT_OPEN_DATA_SOURCE,
    cache_dir: Path = POWER_ECMWF_ENSEMBLE_TMP_DIR,
    power_cutoff_time: pd.Timestamp | str | None = None,
    power_history_days: float = ENSEMBLE_POWER_HISTORY_DAYS,
) -> Path:
    cutoff = (
        _normalise_utc_timestamp(power_cutoff_time, field="power_cutoff_time")
        if power_cutoff_time is not None
        else None
    )
    cycle = latest_ensemble_cycle(source=source) if input_forecast is None else None
    if cycle is not None and output_zarr.exists():
        deterministic_metadata = xr.open_zarr(deterministic_zarr, chunks={})
        deterministic_attrs = dict(deterministic_metadata.attrs)
        deterministic_metadata.close()
        _validate_power_cutoff_anchor(deterministic_attrs, cutoff)
        current = xr.open_zarr(output_zarr, chunks={})
        if current.attrs.get("ecmwf_cycle_time") == cycle.isoformat():
            refresh_reasons = _ensemble_refresh_reasons(current.attrs, deterministic_attrs)
            if cutoff is not None and current.attrs.get("power_input_cutoff_time_utc") != cutoff.isoformat():
                refresh_reasons.append("power_input_cutoff_time_utc")
            obsolete = [
                name
                for name in current.data_vars
                if name.startswith("BatterySOCBelow")
                and name.endswith("Probability")
                and name != SOC_BELOW_THRESHOLD_PROBABILITY_FIELD
            ]
            threshold_changed = (
                current.attrs.get("minimum_operational_soc_pct") != f"{MINIMUM_OPERATIONAL_SOC_PCT:g}"
                or SOC_BELOW_THRESHOLD_PROBABILITY_FIELD not in current
                or bool(obsolete)
            )
            if refresh_reasons:
                print(
                    f"Re-anchoring ECMWF ensemble cycle {cycle.isoformat()} for updated inputs: "
                    f"{', '.join(refresh_reasons)}"
                )
                current.close()
            elif threshold_changed:
                current.load()
                current = apply_operational_soc_threshold(current)
                current.attrs["threshold_metrics_updated_at_utc"] = _utc_now()
                _write_forecast(current, output_zarr)
                if archive_zarr.exists():
                    archive = xr.open_zarr(archive_zarr, chunks={}).load()
                    power = xr.open_zarr(power_zarr, chunks={})
                    cutoff_power = _power_at_or_before_cutoff(
                        power,
                        cutoff,
                        history_days=power_history_days,
                    )
                    skill = build_ensemble_skill_dataset(archive, cutoff_power)
                    _write_time_product(skill, skill_zarr)
                    power.close()
                print(
                    f"Refreshed ensemble diagnostics for the "
                    f"{MINIMUM_OPERATIONAL_SOC_PCT:g}% operational threshold"
                )
                current.close()
                return output_zarr
            else:
                current.close()
                print(f"ECMWF ensemble cycle {cycle.isoformat()} already matches current forecast inputs")
                return output_zarr
        else:
            current.close()

    downloaded_input: Path | None = None
    solar: xr.Dataset | None = None
    power: xr.Dataset | None = None
    deterministic: xr.Dataset | None = None
    try:
        if input_forecast is None:
            assert cycle is not None
            site_cache = _ensemble_site_cache_path(cache_dir, cycle, horizon_hours)
            if site_cache.exists():
                solar = xr.open_zarr(site_cache, chunks={}).load()
            else:
                stamp = cycle.strftime("%Y%m%dT%H%M%SZ")
                downloaded_input = cache_dir / f"ecmwf_ens_ssrd_{stamp}.grib2"
                retrieve_ensemble_grib(downloaded_input, cycle=cycle, horizon_hours=horizon_hours, source=source)
                solar = open_ensemble_site(downloaded_input, latitude=latitude, longitude=longitude)
                _write_ensemble_site_cache(solar, site_cache)
                _prune_ensemble_site_cache(cache_dir)
        else:
            solar = open_ensemble_site(input_forecast, latitude=latitude, longitude=longitude)
        assert solar is not None
        power = xr.open_zarr(power_zarr, chunks={})
        deterministic = xr.open_zarr(deterministic_zarr, chunks={})
        _validate_power_cutoff_anchor(deterministic.attrs, cutoff)
        cutoff_power = _power_at_or_before_cutoff(
            power,
            cutoff,
            history_days=power_history_days,
        )
        cycle_time = cycle or pd.Timestamp(np.asarray(solar["time"].values).reshape(-1)[0])
        composite_source = _ensemble_composite_source_identity(
            solar,
            deterministic.attrs,
            cycle_time=pd.Timestamp(cycle_time),
            source=source,
        )
        forecast = build_ensemble_dataset(cutoff_power, deterministic, solar, horizon_hours=horizon_hours)
        if cutoff is not None:
            forecast.attrs["power_input_cutoff_time_utc"] = cutoff.isoformat()
        forecast.attrs["power_input_history_days"] = cutoff_power.attrs[
            "ensemble_power_history_days"
        ]
        forecast.attrs["power_input_window_start_utc"] = cutoff_power.attrs[
            "ensemble_power_window_start_utc"
        ]
        forecast.attrs["power_input_window_end_utc"] = cutoff_power.attrs[
            "ensemble_power_window_end_utc"
        ]
        forecast.attrs["ecmwf_cycle_time"] = pd.Timestamp(cycle_time).isoformat()
        forecast.attrs.update(composite_source)
        forecast.attrs["forecast_identity_id"] = forecast_identity_id(forecast.attrs)
        _write_forecast(forecast, output_zarr)
        archive = append_ensemble_archive(forecast, archive_zarr)
        skill = build_ensemble_skill_dataset(archive, cutoff_power)
        _write_time_product(skill, skill_zarr)
    finally:
        if deterministic is not None:
            deterministic.close()
        if power is not None:
            power.close()
        if solar is not None:
            solar.close()
        if downloaded_input is not None and downloaded_input.exists():
            downloaded_input.unlink()
            for sidecar in downloaded_input.parent.glob(f"{downloaded_input.name}.*.idx"):
                sidecar.unlink(missing_ok=True)
    print(f"Wrote {output_zarr} with {forecast.sizes['member']} members")
    return output_zarr


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the ECMWF 50-member APS SOC ensemble forecast")
    parser.add_argument("--power-zarr", type=Path, default=POWER_ZARR_PATH)
    parser.add_argument("--deterministic-zarr", type=Path, default=POWER_SOC_FORECAST_ZARR_PATH)
    parser.add_argument("--output-zarr", type=Path, default=POWER_SOC_ENSEMBLE_ZARR_PATH)
    parser.add_argument("--archive-zarr", type=Path, default=POWER_SOC_ENSEMBLE_ARCHIVE_ZARR_PATH)
    parser.add_argument("--skill-zarr", type=Path, default=POWER_SOC_ENSEMBLE_SKILL_ZARR_PATH)
    parser.add_argument("--input-forecast", type=Path)
    parser.add_argument("--latitude", type=float, default=DEFAULT_LATITUDE)
    parser.add_argument("--longitude", type=float, default=DEFAULT_LONGITUDE)
    parser.add_argument("--horizon-hours", type=int, default=DEFAULT_HORIZON_HOURS)
    parser.add_argument("--source", default=DEFAULT_OPEN_DATA_SOURCE)
    parser.add_argument("--cache-dir", type=Path, default=POWER_ECMWF_ENSEMBLE_TMP_DIR)
    parser.add_argument(
        "--power-cutoff-time",
        help="Exact deterministic initial_soc_time; ignore newer live APS power samples",
    )
    parser.add_argument(
        "--power-history-days",
        type=float,
        default=ENSEMBLE_POWER_HISTORY_DAYS,
        help=(
            "APS history retained for ensemble load fallback and skill verification "
            f"(default: {ENSEMBLE_POWER_HISTORY_DAYS:g}; minimum: "
            f"{ENSEMBLE_POWER_MIN_HISTORY_DAYS:g})"
        ),
    )
    args = parser.parse_args()
    generate(
        power_zarr=args.power_zarr,
        deterministic_zarr=args.deterministic_zarr,
        output_zarr=args.output_zarr,
        archive_zarr=args.archive_zarr,
        skill_zarr=args.skill_zarr,
        input_forecast=args.input_forecast,
        latitude=args.latitude,
        longitude=args.longitude,
        horizon_hours=args.horizon_hours,
        source=args.source,
        cache_dir=args.cache_dir,
        power_cutoff_time=args.power_cutoff_time,
        power_history_days=args.power_history_days,
    )


if __name__ == "__main__":
    main()
