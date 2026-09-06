#!/usr/bin/env python3
"""Generate an ECMWF-informed Aurora Power Supply SOC forecast product."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Mapping

import numpy as np
import pandas as pd
import xarray as xr

from ecmwf_forecast_provider import (
    DEFAULT_PROVIDER,
    DEFAULT_SHADOW_REPORT_PATH,
    open_solar_forecast as open_provider_solar_forecast,
    retrieve_open_data_grib as retrieve_provider_open_data_grib,
    validate_provider,
)
from power_soc_thresholds import MINIMUM_OPERATIONAL_SOC_PCT
from power_battery_model import BatteryModel, fit_battery_model, soc_delta_percent
from power_load_contract import (
    CONTROLLED_LOAD_CONTRACT,
    STATE_HOLD_POLICY,
    estimate_controlled_load,
)
from power_load_dynamics import (
    ControlledLoadProfile,
    PHASE_CODES,
    build_controlled_load_profile,
    learn_state_load_dynamics,
)
from power_solar_model import (
    SOLAR_MODEL_NAME as PHYSICAL_SOLAR_MODEL_NAME,
    PhysicalSolarConfig,
    build_physical_solar_forecast_frames,
    load_physical_solar_config,
    physical_solar_config_digest,
)

POWER_ZARR_PATH = Path(os.environ.get("POWER_ZARR_PATH", "/data/aurora/products/power/power.zarr"))
POWER_SOC_FORECAST_ZARR_PATH = Path(
    os.environ.get("POWER_SOC_FORECAST_ZARR_PATH", "/data/aurora/products/power/power_soc_forecast.zarr")
)
POWER_ECMWF_FORECAST_CACHE_DIR = Path(
    os.environ.get("POWER_ECMWF_FORECAST_CACHE_DIR", "/data/aurora/products/power/ecmwf_solar_forecast")
)
POWER_SOC_FORECAST_STATE_PATH = Path(
    os.environ.get("POWER_SOC_FORECAST_STATE_PATH", "/data/aurora/products/power/power_soc_forecast_state.json")
)
POWER_SOC_FORECAST_ARCHIVE_ZARR_PATH = Path(
    os.environ.get("POWER_SOC_FORECAST_ARCHIVE_ZARR_PATH", "/data/aurora/products/power/power_soc_forecast_archive.zarr")
)
POWER_SOC_FORECAST_SKILL_ZARR_PATH = Path(
    os.environ.get("POWER_SOC_FORECAST_SKILL_ZARR_PATH", "/data/aurora/products/power/power_soc_forecast_skill.zarr")
)
POWER_SOC_HINDCAST_ZARR_PATH = Path(
    os.environ.get("POWER_SOC_HINDCAST_ZARR_PATH", "/data/aurora/products/power/power_soc_hindcast.zarr")
)
POWER_PDU_ZARR_PATH = Path(os.environ.get("POWER_PDU_ZARR_PATH", "/data/aurora/products/power/pdu.zarr"))

DEFAULT_LATITUDE = float(os.environ.get("AURORA_POWER_FORECAST_LATITUDE", "64.829694"))
DEFAULT_LONGITUDE = float(os.environ.get("AURORA_POWER_FORECAST_LONGITUDE", "-23.248139"))
DEFAULT_HORIZON_HOURS = int(os.environ.get("AURORA_POWER_SOC_FORECAST_HOURS", "96"))
DEFAULT_CALIBRATION_DAYS = float(os.environ.get("AURORA_POWER_SOC_FORECAST_CALIBRATION_DAYS", "7"))
DEFAULT_FALLBACK_CALIBRATION_HOURS = float(os.environ.get("AURORA_POWER_SOC_FORECAST_FALLBACK_CALIBRATION_HOURS", "48"))
DEFAULT_BATTERY_CAPACITY_KWH = float(os.environ.get("APS_BATTERY_CAPACITY_KWH", "26"))
DEFAULT_SOLAR_CALIBRATION_FACTOR = float(os.environ.get("AURORA_POWER_SOLAR_CALIBRATION_FACTOR", "1.0"))
DEFAULT_LOAD_W = float(os.environ.get("AURORA_POWER_FORECAST_DEFAULT_LOAD_W", "0"))
DEFAULT_ADAPTIVE_ALPHA = float(os.environ.get("AURORA_POWER_SOC_FORECAST_ADAPTIVE_ALPHA", "0.25"))
DEFAULT_MIN_ADAPTIVE_ALPHA = float(os.environ.get("AURORA_POWER_SOC_FORECAST_MIN_ADAPTIVE_ALPHA", "0.08"))
DEFAULT_MAX_ADAPTIVE_ALPHA = float(os.environ.get("AURORA_POWER_SOC_FORECAST_MAX_ADAPTIVE_ALPHA", "0.45"))
DEFAULT_LOAD_BIAS_CORRECTION_LIMIT_W = float(os.environ.get("AURORA_POWER_LOAD_BIAS_CORRECTION_LIMIT_W", "2000"))
DEFAULT_LOAD_NEGATIVE_CORRECTION_FRACTION_LIMIT = float(
    os.environ.get("AURORA_POWER_LOAD_NEGATIVE_CORRECTION_FRACTION_LIMIT", "0.25")
)
DEFAULT_LOAD_MODE_LEVEL_HOURS = float(os.environ.get("AURORA_POWER_LOAD_MODE_LEVEL_HOURS", "2"))
DEFAULT_LOAD_MODE_STATE_MINUTES = float(os.environ.get("AURORA_POWER_LOAD_MODE_STATE_MINUTES", "30"))
DEFAULT_LOAD_MODE_LEARN_INTERVAL_MINUTES = float(
    os.environ.get("AURORA_POWER_LOAD_MODE_LEARN_INTERVAL_MINUTES", "60")
)
DEFAULT_AC_MODE_THRESHOLD_W = float(os.environ.get("AURORA_POWER_AC_MODE_THRESHOLD_W", "25"))
DEFAULT_PDU_MODE_FRESHNESS_MINUTES = float(os.environ.get("AURORA_POWER_PDU_MODE_FRESHNESS_MINUTES", "60"))
DEFAULT_PDU_ACTIVE_W_THRESHOLD = float(os.environ.get("AURORA_POWER_PDU_ACTIVE_W_THRESHOLD", "5"))
DEFAULT_LOAD_MODE_MIN_STABLE_SAMPLES = int(os.environ.get("AURORA_POWER_LOAD_MODE_MIN_STABLE_SAMPLES", "2"))
DEFAULT_MODE_MIN_OBSERVATIONS = int(os.environ.get("AURORA_POWER_LOAD_MODE_MIN_OBSERVATIONS", "3"))
DEFAULT_DYNAMICS_MIN_SAMPLES = int(os.environ.get("AURORA_POWER_LOAD_DYNAMICS_MIN_SAMPLES", "8"))
FORECAST_TARGET_MAE_PCT_POINTS = float(os.environ.get("AURORA_POWER_FORECAST_TARGET_MAE_PCT_POINTS", "10"))
FORECAST_TARGET_MIN_CYCLES = int(os.environ.get("AURORA_POWER_FORECAST_TARGET_MIN_CYCLES", "30"))
FORECAST_TARGET_MIN_SAMPLES = int(os.environ.get("AURORA_POWER_FORECAST_TARGET_MIN_SAMPLES", "20"))
DEFAULT_ZERO_SOLAR_THRESHOLD_W = float(os.environ.get("AURORA_POWER_ZERO_SOLAR_THRESHOLD_W", "10"))
DEFAULT_DARK_LOAD_LOOKBACK_HOURS = float(os.environ.get("AURORA_POWER_DARK_LOAD_LOOKBACK_HOURS", "48"))
DEFAULT_SOC_BIAS_CORRECTION_LIMIT = float(os.environ.get("AURORA_POWER_SOC_BIAS_CORRECTION_LIMIT", "8"))
DEFAULT_ARCHIVE_RETENTION_DAYS = float(os.environ.get("AURORA_POWER_SOC_FORECAST_ARCHIVE_RETENTION_DAYS", "21"))
DEFAULT_ECMWF_LOOKAHEAD_BUFFER_HOURS = int(os.environ.get("AURORA_POWER_ECMWF_LOOKAHEAD_BUFFER_HOURS", "24"))
DEFAULT_OPEN_DATA_SOURCE = os.environ.get("AURORA_POWER_ECMWF_OPEN_DATA_SOURCE", "azure")
DEFAULT_SKILL_WINDOW_HOURS = float(os.environ.get("AURORA_POWER_SOC_FORECAST_SKILL_WINDOW_HOURS", "24"))
DEFAULT_SKILL_RETENTION_DAYS = float(os.environ.get("AURORA_POWER_SOC_FORECAST_SKILL_RETENTION_DAYS", "7"))
ECMWF_PARAM = "ssrd"
LEGACY_SOLAR_MODEL_NAME = "ssrd_scalar_lead_mos_v1"
LEGACY_SOLAR_MODEL_VERSION = 1
LEGACY_SOLAR_FEATURE_SET_VERSION = "ecmwf_ssrd_scalar_v1"
VALID_SOLAR_MODELS = (LEGACY_SOLAR_MODEL_NAME, PHYSICAL_SOLAR_MODEL_NAME)
DEFAULT_SOLAR_MODEL = os.environ.get("AURORA_POWER_SOLAR_MODEL", LEGACY_SOLAR_MODEL_NAME).strip()
DEFAULT_PHYSICAL_SOLAR_CONFIG_PATH = Path(
    os.environ.get(
        "AURORA_POWER_SOLAR_PHYSICAL_CONFIG",
        str(Path(__file__).with_name("config") / "power_solar_physical_candidate_v1.json"),
    )
)
LOAD_MODEL_NAME = "finite_controlled_state_phases_v10"
LOAD_MODEL_VERSION = 10
PDU_OUTLET_KIT_NAMES = {
    4: "UAS",
    5: "CL61",
    6: "Radar",
    8: "HATPRO",
}
MPP_ACTIVE_MODE = 2.0
SOLAR_MPP_MODE_FIELDS = (
    "SolarMPPMode_East",
    "SolarMPPMode_South",
    "SolarMPPMode_West",
)


def resolve_ecmwf_cycle_hour(value: int | str | None, *, now: datetime | None = None) -> int | None:
    if value is None or value == "":
        return None
    if str(value).lower() != "auto":
        selected = int(value)
        if selected not in {0, 12}:
            raise ValueError("ECMWF long-cycle hour must be 0, 12, or auto")
        return selected
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    hour = current.astimezone(timezone.utc).hour
    return 0 if 8 <= hour < 20 else 12


def validate_solar_model(value: str | None) -> str:
    selected = str(value or DEFAULT_SOLAR_MODEL).strip().lower()
    if selected not in VALID_SOLAR_MODELS:
        raise ValueError(
            f"Unknown solar model {selected!r}; expected one of {', '.join(VALID_SOLAR_MODELS)}"
        )
    return selected


def _paths_overlap(first: Path, second: Path) -> bool:
    """Return whether either resolved path contains the other."""
    first_resolved = Path(first).resolve()
    second_resolved = Path(second).resolve()
    return (
        first_resolved == second_resolved
        or first_resolved in second_resolved.parents
        or second_resolved in first_resolved.parents
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


PDU_KIT_OUTLETS = {name: outlet for outlet, name in PDU_OUTLET_KIT_NAMES.items()}
LEAD_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("0_6h", 0.0, 6.0),
    ("6_24h", 6.0, 24.0),
    ("24_48h", 24.0, 48.0),
    ("48_96h", 48.0, 96.0),
)
ARCHIVE_FORECAST_FIELDS = (
    "BatterySOCForecast",
    "ForecastSOCBiasCorrectionPctPoints",
    "ECMWFSolarIrradiance",
    "ECMWFSourceIntervalHours",
    "ForecastEffectiveGlobalHorizontalIrradiance",
    "ForecastSolarWatts",
    "ForecastPVAvailableWatts",
    "ForecastPVDeliveredWatts",
    "ForecastPVCurtailedWatts",
    "ForecastPVAvailableWattsEast",
    "ForecastPVAvailableWattsSouth",
    "ForecastPVAvailableWattsWest",
    "ForecastPVControllerClippingWatts",
    "ForecastPVControllerClippingWattsEast",
    "ForecastPVControllerClippingWattsSouth",
    "ForecastPVControllerClippingWattsWest",
    "ForecastPlaneOfArrayIrradianceEast",
    "ForecastPlaneOfArrayIrradianceSouth",
    "ForecastPlaneOfArrayIrradianceWest",
    "ForecastPlaneOfArrayDirectIrradianceEast",
    "ForecastPlaneOfArrayDirectIrradianceSouth",
    "ForecastPlaneOfArrayDirectIrradianceWest",
    "ForecastPlaneOfArrayDiffuseIrradianceEast",
    "ForecastPlaneOfArrayDiffuseIrradianceSouth",
    "ForecastPlaneOfArrayDiffuseIrradianceWest",
    "ForecastEffectivePlaneOfArrayIrradianceEast",
    "ForecastEffectivePlaneOfArrayIrradianceSouth",
    "ForecastEffectivePlaneOfArrayIrradianceWest",
    "ForecastEffectivePlaneOfArrayDirectIrradianceEast",
    "ForecastEffectivePlaneOfArrayDirectIrradianceSouth",
    "ForecastEffectivePlaneOfArrayDirectIrradianceWest",
    "ForecastPVCellTemperatureEast",
    "ForecastPVCellTemperatureSouth",
    "ForecastPVCellTemperatureWest",
    "ForecastBatteryChargeInputWatts",
    "ForecastBatteryDischargeOutputWatts",
    "ECMWFDirectNormalIrradiance",
    "ECMWFDirectHorizontalIrradiance",
    "ECMWFDiffuseHorizontalIrradiance",
    "ECMWFClearnessIndex",
    "SolarZenithDegrees",
    "SolarAzimuthDegrees",
    "SolarCosineZenith",
    "SolarIntervalHours",
    "SolarForcingConsistencyFlag",
    "ForecastLoadWatts",
    "ForecastLoadP10Watts",
    "ForecastLoadP50Watts",
    "ForecastLoadP90Watts",
    "ForecastLoadPhaseCode",
)
FORECAST_IDENTITY_ATTRS = (
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
)
# Stable, algorithm/configuration identity used to decide which archived rows
# can share one score surface.  Deliberately exclude per-issue provenance
# (source cycle/manifest, SOC anchor, observation cutoff) and learned numeric
# state: those remain in ForecastIdentityID/AdaptiveCalibrationStateID for exact
# reproducibility, but must not fragment a like-for-like evaluation campaign.
FORECAST_SEMANTIC_IDENTITY_ARCHIVE_FIELDS = (
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
SCENARIO_LOADS_W = (100, 200, 300, 400, 500, 600)
HINDCAST_LEAD_HOURS = (6, 24, 48, 72)
DEFAULT_HINDCAST_RETENTION_DAYS = float(os.environ.get("AURORA_POWER_SOC_HINDCAST_RETENTION_DAYS", "7"))
DEFAULT_MAX_POWER_INPUT_AGE_MINUTES = float(
    os.environ.get("AURORA_POWER_SOC_FORECAST_MAX_POWER_INPUT_AGE_MINUTES", "20")
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def solar_calibration_contract_id(base_factor: float, mos_by_bucket: dict[str, float]) -> str:
    """Return the identity of one learned solar calibration state.

    This identifier is intentionally *not* a forecast-model contract.  The
    fitted factor and lead MOS values change as observations mature, while the
    algorithm being evaluated remains the same.
    """
    payload = {
        "schema": 1,
        "base_factor": round(float(base_factor), 8),
        "lead_mos": {name: round(float(value), 8) for name, value in sorted(mos_by_bucket.items())},
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return f"solar-calibration-v1-{digest}"


def legacy_solar_model_contract_id() -> str:
    """Return a stable semantic contract for the legacy solar algorithm."""
    payload = {
        "schema": 1,
        "solar_model_name": LEGACY_SOLAR_MODEL_NAME,
        "solar_model_version": LEGACY_SOLAR_MODEL_VERSION,
        "solar_feature_set_version": LEGACY_SOLAR_FEATURE_SET_VERSION,
        "calibration_method": "adaptive_scalar_plus_independent_cycle_lead_mos",
        "power_semantics": "electrical_power_from_ssrd_scalar",
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return f"solar-model-v1-{digest}"


def adaptive_calibration_state_id(attrs: Mapping[str, object]) -> str:
    """Hash learned numeric state separately from the stable model contract."""

    payload = {
        "schema": 1,
        "solar_calibration_state_id": _normalise_identity_text(
            attrs.get("solar_calibration_contract_id")
        ),
        "solar_calibration_factor_w_per_wm2": _normalise_identity_text(
            attrs.get("solar_calibration_factor_w_per_wm2")
        ),
        "raw_solar_calibration_factor_w_per_wm2": _normalise_identity_text(
            attrs.get("raw_solar_calibration_factor_w_per_wm2")
        ),
        "solar_mos_factor_by_lead_bucket": _normalise_identity_text(
            attrs.get("solar_mos_factor_by_lead_bucket")
        ),
        "load_bias_correction_w": _normalise_identity_text(
            attrs.get("load_bias_correction_w")
        ),
        "soc_bias_correction_pct_points_by_bucket": _normalise_identity_text(
            attrs.get("soc_bias_correction_pct_points_by_bucket")
        ),
        "load_residual_model_contract_id": _normalise_identity_text(
            attrs.get("load_residual_model_contract_id")
        ),
        "load_residual_model_status": _normalise_identity_text(
            attrs.get("load_residual_model_status")
        ),
        "load_residual_state_digest": _normalise_identity_text(
            attrs.get("load_residual_state_digest")
        ),
        "load_mode_signature": _normalise_identity_text(attrs.get("load_mode_signature")),
        "load_state_dynamics_signature": _normalise_identity_text(
            attrs.get("load_state_dynamics_signature")
        ),
        "load_mode_registry_digest": hashlib.sha256(
            _normalise_identity_text(attrs.get("load_mode_registry")).encode("utf-8")
        ).hexdigest(),
        "forecast_load_w": _normalise_identity_text(attrs.get("forecast_load_w")),
        "forecast_load_p10_w": _normalise_identity_text(
            attrs.get("forecast_load_p10_w")
        ),
        "forecast_load_p50_w": _normalise_identity_text(
            attrs.get("forecast_load_p50_w")
        ),
        "forecast_load_p90_w": _normalise_identity_text(
            attrs.get("forecast_load_p90_w")
        ),
        "battery_calibration_sample_count": _normalise_identity_text(
            attrs.get("battery_calibration_sample_count")
        ),
        "battery_calibration_confidence": _normalise_identity_text(
            attrs.get("battery_calibration_confidence")
        ),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:20]
    return f"adaptive-calibration-v1-{digest}"


def _normalise_identity_text(value: object) -> str:
    """Return a compact, serialisable forecast-identity value."""
    return str(value if value is not None else "").strip()


def forecast_identity_id(attrs: Mapping[str, object]) -> str:
    """Return an issue-specific provenance key without changing model versioning.

    A model contract is deliberately stable across comparable candidate issues.
    This companion key includes the immutable input/cutoff provenance needed to
    reproduce one issue exactly and is used as the primary key of the campaign
    evidence product.
    """
    payload = {
        "schema": 1,
        "forecast_model_contract_id": _normalise_identity_text(
            attrs.get("forecast_model_contract_id")
        ),
        "forecast_system_version": _normalise_identity_text(
            attrs.get("forecast_system_version")
        ),
        "feature_set_version": _normalise_identity_text(attrs.get("feature_set_version")),
        "feature_set_digest": _normalise_identity_text(attrs.get("feature_set_digest")),
        "training_cutoff_utc": _normalise_identity_text(attrs.get("training_cutoff_utc")),
        "forecast_code_revision": _normalise_identity_text(attrs.get("forecast_code_revision")),
        "source_cycle_set_id": _normalise_identity_text(attrs.get("source_cycle_set_id")),
        "source_manifest_digest": _normalise_identity_text(
            attrs.get("source_manifest_digest")
        ),
        "degraded_mode_code": _normalise_identity_text(attrs.get("degraded_mode_code")),
        "candidate_lane": _normalise_identity_text(attrs.get("candidate_lane")),
        "local_feature_contract_id": _normalise_identity_text(
            attrs.get("local_feature_contract_id")
        ),
        "baseline_control_contract_id": _normalise_identity_text(
            attrs.get("baseline_control_contract_id")
        ),
        "baseline_control_system_version": _normalise_identity_text(
            attrs.get("baseline_control_system_version")
        ),
        "source_availability_code": _normalise_identity_text(
            attrs.get("source_availability_code")
        ),
        "observation_cutoff_utc": _normalise_identity_text(
            attrs.get("observation_cutoff_utc")
        ),
        "soc_anchor_time_utc": _normalise_identity_text(attrs.get("soc_anchor_time_utc")),
        "soc_anchor_pct": _normalise_identity_text(attrs.get("initial_soc_pct")),
        "adaptive_calibration_state_id": _normalise_identity_text(
            attrs.get("adaptive_calibration_state_id")
        ),
        "initial_soc_time": _normalise_identity_text(attrs.get("initial_soc_time")),
        "input_snapshot_id": _normalise_identity_text(attrs.get("input_snapshot_id")),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:20]
    return f"forecast-identity-v1-{digest}"


def apply_forecast_identity(
    forecast: xr.Dataset,
    identity: Mapping[str, object] | None,
) -> xr.Dataset:
    """Attach v12 candidate provenance and derive a stable model contract.

    `training_cutoff_utc` and source-cycle details are intentionally excluded
    from the model contract: they distinguish issues, not the algorithm under
    evaluation.  They remain in ``forecast_identity_id`` and every archive row.
    """
    if identity is None:
        return forecast
    attrs = forecast.attrs
    defaults = {
        "forecast_system_version": attrs.get("forecast_model_version", ""),
        "feature_set_version": "unspecified",
        "feature_set_digest": "",
        "training_cutoff_utc": attrs.get("initial_soc_time", ""),
        "forecast_code_revision": "unversioned",
        "source_cycle_set_id": attrs.get("ecmwf_cycle_time", ""),
        "source_manifest_digest": "",
        "degraded_mode_code": "none",
        "candidate_lane": "",
        "local_feature_contract_id": "",
        "baseline_control_contract_id": "",
        "baseline_control_system_version": "",
        "source_availability_code": "",
        "observation_cutoff_utc": attrs.get("initial_soc_time", ""),
        "soc_anchor_time_utc": attrs.get("initial_soc_time", ""),
        "adaptive_calibration_state_id": attrs.get("adaptive_calibration_state_id", ""),
    }
    for name, default in defaults.items():
        attrs[name] = _normalise_identity_text(identity.get(name, default))
    for name in ("forecast_model_name", "forecast_model_version", "forecast_model_status"):
        if name in identity:
            attrs[name] = _normalise_identity_text(identity[name])

    base_contract = _normalise_identity_text(attrs.get("forecast_model_contract_id"))
    contract_payload = {
        "schema": 2,
        "base_forecast_contract": base_contract,
        "forecast_model_name": _normalise_identity_text(attrs.get("forecast_model_name")),
        "forecast_model_version": _normalise_identity_text(attrs.get("forecast_model_version")),
        "forecast_system_version": attrs["forecast_system_version"],
        "feature_set_version": attrs["feature_set_version"],
        "feature_set_digest": attrs["feature_set_digest"],
        "forecast_code_revision": attrs["forecast_code_revision"],
        "candidate_lane": attrs["candidate_lane"],
        "local_feature_contract_id": attrs["local_feature_contract_id"],
        "baseline_control_contract_id": attrs["baseline_control_contract_id"],
        "baseline_control_system_version": attrs["baseline_control_system_version"],
    }
    attrs["base_forecast_model_contract_id"] = base_contract
    attrs["forecast_model_contract_id"] = "forecast-model-v2-" + hashlib.sha256(
        json.dumps(contract_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    attrs["forecast_identity_id"] = forecast_identity_id(attrs)
    return forecast


def forecast_publication_signature(forecast: xr.Dataset) -> str:
    """Hash operationally meaningful inputs while ignoring run-time metadata."""
    attrs = forecast.attrs
    anchor = pd.to_datetime(attrs.get("initial_soc_time"), errors="coerce")
    anchor_bucket = "" if pd.isna(anchor) else pd.Timestamp(anchor).floor("30min").isoformat()

    def stepped(name: str, step: float) -> float | None:
        try:
            value = float(attrs.get(name, np.nan))
        except (TypeError, ValueError):
            return None
        if not np.isfinite(value):
            return None
        return round(value / step) * step

    dynamics = str(attrs.get("load_state_dynamics", ""))
    dynamics_signature = hashlib.sha256(dynamics.encode("utf-8")).hexdigest()[:16] if dynamics else ""
    payload = {
        "schema": 4,
        "anchor_30min": anchor_bucket,
        "soc_pct": stepped("initial_soc_pct", 1.0),
        "load_w": stepped("forecast_load_w", 25.0),
        "load_mode": str(attrs.get("load_mode_signature", attrs.get("load_mode", ""))),
        "load_phase": str(attrs.get("load_current_phase", "")),
        "load_dynamics": dynamics_signature,
        "ecmwf_cycle": str(attrs.get("ecmwf_cycle_time", "")),
        "solar_contract": str(attrs.get("solar_calibration_contract_id", "")),
        "forecast_model_contract": str(attrs.get("forecast_model_contract_id", "")),
        "evaluation_pair_id": str(attrs.get("evaluation_pair_id", "")),
        "input_snapshot_id": str(attrs.get("input_snapshot_id", "")),
        "site_latitude": stepped("site_latitude", 0.000001),
        "site_longitude": stepped("site_longitude", 0.000001),
        "battery_capacity_kwh": stepped("battery_usable_capacity_kwh", 0.25),
        "battery_charge_efficiency": stepped("battery_charge_efficiency", 0.01),
        "battery_discharge_efficiency": stepped("battery_discharge_efficiency", 0.01),
        "battery_parasitic_load_w": stepped("battery_parasitic_load_w", 10.0),
        "battery_max_charge_w": stepped("battery_max_charge_w", 25.0),
        "battery_max_discharge_w": stepped("battery_max_discharge_w", 25.0),
        "model_version": str(attrs.get("load_model_version", "")),
        "forecast_system_version": str(attrs.get("forecast_system_version", "")),
        "feature_set_version": str(attrs.get("feature_set_version", "")),
        "feature_set_digest": str(attrs.get("feature_set_digest", "")),
        "forecast_code_revision": str(attrs.get("forecast_code_revision", "")),
        "source_cycle_set_id": str(attrs.get("source_cycle_set_id", "")),
        "source_manifest_digest": str(attrs.get("source_manifest_digest", "")),
        "degraded_mode_code": str(attrs.get("degraded_mode_code", "")),
        "candidate_lane": str(attrs.get("candidate_lane", "")),
        "local_feature_contract_id": str(attrs.get("local_feature_contract_id", "")),
        "baseline_control_contract_id": str(attrs.get("baseline_control_contract_id", "")),
        "baseline_control_system_version": str(
            attrs.get("baseline_control_system_version", "")
        ),
        "source_availability_code": str(attrs.get("source_availability_code", "")),
        "forecast_identity_id": str(attrs.get("forecast_identity_id", "")),
        "adaptive_calibration_state_id": str(
            attrs.get("adaptive_calibration_state_id", "")
        ),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:20]


def validate_paired_candidate(candidate: xr.Dataset, baseline: xr.Dataset) -> None:
    """Fail before publication unless non-solar baseline inputs are identical."""

    failures: list[str] = []
    baseline_signature = str(baseline.attrs.get("publication_signature", "")).strip()
    if not baseline_signature:
        failures.append("baseline publication signature is missing")
    if str(candidate.attrs.get("baseline_publication_signature", "")) != baseline_signature:
        failures.append("baseline publication signature")

    def timestamp_attr(name: str, product: xr.Dataset) -> pd.Timestamp | None:
        value = pd.to_datetime(product.attrs.get(name), errors="coerce")
        if pd.isna(value):
            return None
        parsed = pd.Timestamp(value)
        return parsed.tz_convert("UTC").tz_localize(None) if parsed.tz is not None else parsed

    for name in ("initial_soc_time", "ecmwf_cycle_time"):
        if timestamp_attr(name, candidate) != timestamp_attr(name, baseline):
            failures.append(name)
    for name, tolerance in (
        ("initial_soc_pct", 1.0e-6),
        ("site_latitude", 1.0e-8),
        ("site_longitude", 1.0e-8),
        ("selected_grid_latitude", 1.0e-8),
        ("selected_grid_longitude", 1.0e-8),
        ("selected_grid_distance_km", 1.0e-6),
    ):
        if name not in baseline.attrs:
            continue
        try:
            candidate_value = float(candidate.attrs.get(name, np.nan))
            baseline_value = float(baseline.attrs[name])
        except (TypeError, ValueError):
            failures.append(name)
            continue
        if not np.isclose(candidate_value, baseline_value, rtol=0.0, atol=tolerance, equal_nan=True):
            failures.append(name)
    if str(candidate.attrs.get("forecast_horizon_hours", "")) != str(
        baseline.attrs.get("forecast_horizon_hours", "")
    ):
        failures.append("forecast_horizon_hours")
    if str(candidate.attrs.get("solar_input_representation", "")) == "embedded_site_irradiance":
        if str(candidate.attrs.get("site_irradiance_source_manifest_digest", "")) != str(
            baseline.attrs.get("source_manifest_digest", "")
        ):
            failures.append("site irradiance source manifest")
        if str(candidate.attrs.get("site_irradiance_source_cycle_set_id", "")) != str(
            baseline.attrs.get("source_cycle_set_id", "")
        ):
            failures.append("site irradiance source cycle")
        if str(candidate.attrs.get("input_snapshot_id", "")) != str(
            candidate.attrs.get("site_irradiance_sha256", "")
        ):
            failures.append("site irradiance snapshot identity")
    elif str(candidate.attrs.get("ecmwf_input_file", "")) != str(
        baseline.attrs.get("ecmwf_input_file", "")
    ):
        failures.append("ecmwf_input_file")

    if "time" not in candidate.coords or "time" not in baseline.coords or not np.array_equal(
        np.asarray(candidate["time"].values, dtype="datetime64[ns]"),
        np.asarray(baseline["time"].values, dtype="datetime64[ns]"),
    ):
        failures.append("forecast time grid")
    shared_fields = (
        "ECMWFSolarIrradiance",
        "ForecastLoadWatts",
        "ForecastLoadP10Watts",
        "ForecastLoadP50Watts",
        "ForecastLoadP90Watts",
        "ForecastLoadPhaseCode",
    )
    for name in shared_fields:
        if name not in candidate or name not in baseline:
            failures.append(name)
            continue
        left = np.asarray(candidate[name].values, dtype=np.float64)
        right = np.asarray(baseline[name].values, dtype=np.float64)
        if left.shape != right.shape or not np.allclose(
            left,
            right,
            rtol=1.0e-6,
            atol=1.0e-5,
            equal_nan=True,
        ):
            failures.append(name)
    shared_attrs = (
        "load_model",
        "load_model_version",
        "load_state_contract",
        "load_state_hold_policy",
        "load_mode_signature",
        "load_state_dynamics_signature",
        "battery_energy_model",
        "battery_usable_capacity_kwh",
        "battery_charge_efficiency",
        "battery_discharge_efficiency",
        "battery_parasitic_load_w",
        "battery_max_charge_w",
        "battery_max_discharge_w",
        "soc_bias_correction_pct_points_by_bucket",
    )
    for name in shared_attrs:
        if str(candidate.attrs.get(name, "")) != str(baseline.attrs.get(name, "")):
            failures.append(name)
    if failures:
        raise ValueError(
            "Physical candidate is not an exact non-solar pair with the baseline: "
            + ", ".join(dict.fromkeys(failures))
        )


def _atomic_write_zarr(ds: xr.Dataset, output_zarr: Path) -> None:
    output_zarr.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_zarr.with_name(f"{output_zarr.name}.tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    chunk_dim = "time" if "time" in ds.sizes else next(iter(ds.sizes), None)
    chunked = (
        ds.chunk({chunk_dim: min(max(ds.sizes.get(chunk_dim, 1), 1), 288)})
        if chunk_dim is not None
        else ds
    )
    chunked.to_zarr(tmp, mode="w", consolidated=True)
    if output_zarr.exists():
        shutil.rmtree(output_zarr)
    tmp.rename(output_zarr)


ISSUE_SNAPSHOT_DIGEST_MARKER = ".aurora-snapshot-content-v1.json"
ISSUE_SNAPSHOT_DIGEST_ALGORITHM = "sha256-relative-path-nul-content-nul-v1"


def _issue_snapshot_tree_digest(path: Path) -> tuple[str, int, int]:
    """Hash exact persisted Zarr content using the bundle artifact contract."""

    root = Path(path)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Issue snapshot is not a direct Zarr directory")
    files: list[Path] = []
    for candidate in root.rglob("*"):
        if candidate.is_symlink():
            raise ValueError("Issue snapshot contains a symbolic link")
        if not candidate.is_file():
            continue
        if candidate.relative_to(root).as_posix() == ISSUE_SNAPSHOT_DIGEST_MARKER:
            continue
        files.append(candidate)
    files.sort(key=lambda candidate: candidate.relative_to(root).as_posix())
    digest = hashlib.sha256()
    byte_count = 0
    for candidate in files:
        relative = candidate.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with candidate.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                byte_count += len(block)
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest(), len(files), byte_count


def _verify_issue_snapshot(path: Path, *, expected_signature: str) -> str:
    """Verify the marker, persisted bytes, and readable dataset signature."""

    marker_path = Path(path) / ISSUE_SNAPSHOT_DIGEST_MARKER
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError("Issue snapshot content-digest marker is missing or invalid") from exc
    if (
        marker.get("schemaVersion") != 1
        or marker.get("digestAlgorithm") != ISSUE_SNAPSHOT_DIGEST_ALGORITHM
    ):
        raise ValueError("Issue snapshot content-digest marker has an unsupported contract")
    expected_digest = str(marker.get("contentDigest") or "")
    digest, file_count, byte_count = _issue_snapshot_tree_digest(Path(path))
    if expected_digest != f"sha256:{digest}":
        raise ValueError("Issue snapshot content digest does not match persisted bytes")
    if int(marker.get("fileCount", -1)) != file_count or int(
        marker.get("byteCount", -1)
    ) != byte_count:
        raise ValueError("Issue snapshot content-digest counts do not match persisted bytes")
    marker_signature = str(marker.get("publicationSignature") or "").strip()
    if marker_signature != expected_signature:
        raise ValueError("Issue snapshot marker has a different forecast signature")
    try:
        with xr.open_zarr(path, chunks={}, consolidated=True) as opened:
            stored = opened.load()
    except Exception as exc:
        raise ValueError("Issue snapshot is unreadable") from exc
    if str(stored.attrs.get("publication_signature", "")).strip() != expected_signature:
        raise ValueError("Issue snapshot dataset has a different forecast signature")
    return digest


def write_immutable_issue_snapshot(forecast: xr.Dataset, output_zarr: Path) -> Path:
    """Write one archive-eligible issue once and reject conflicting reuse.

    The caller chooses the cycle/signature-specific path.  A retry with the
    same publication signature is idempotent; reusing that path for different
    forecast content fails closed instead of replacing campaign evidence.
    """

    path = Path(output_zarr)
    if path.suffix.lower() != ".zarr":
        raise ValueError("Issue snapshot path must end in .zarr")
    if str(forecast.attrs.get("forecast_verification_eligible", "")).lower() != "true":
        raise ValueError("Only archive-eligible forecast issues may be snapshotted")
    if str(forecast.attrs.get("forecast_refresh_kind", "")) != "ecmwf_cycle":
        raise ValueError("Only full ECMWF-cycle forecasts may be snapshotted")
    signature = str(forecast.attrs.get("publication_signature", "")).strip()
    if not signature:
        raise ValueError("Issue snapshot requires a publication signature")
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{path.name}.staging-", dir=str(path.parent))
    )
    try:
        chunk_dim = "time" if "time" in forecast.sizes else next(iter(forecast.sizes), None)
        chunked = (
            forecast.chunk(
                {chunk_dim: min(max(forecast.sizes.get(chunk_dim, 1), 1), 288)}
            )
            if chunk_dim is not None
            else forecast
        )
        chunked.to_zarr(staging, mode="w", consolidated=True)
        digest, file_count, byte_count = _issue_snapshot_tree_digest(staging)
        marker = {
            "schemaVersion": 1,
            "digestAlgorithm": ISSUE_SNAPSHOT_DIGEST_ALGORITHM,
            "contentDigest": f"sha256:{digest}",
            "fileCount": file_count,
            "byteCount": byte_count,
            "publicationSignature": signature,
        }
        (staging / ISSUE_SNAPSHOT_DIGEST_MARKER).write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staged_digest = _verify_issue_snapshot(
            staging, expected_signature=signature
        )
        if not path.exists():
            try:
                staging.rename(path)
                return path
            except OSError:
                if not path.exists():
                    raise
        existing_digest = _verify_issue_snapshot(path, expected_signature=signature)
        if existing_digest != staged_digest:
            raise ValueError("Issue snapshot path already contains a different forecast")
        return path
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _load_state(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_state(path: Path, state: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    tmp.rename(path)


def _atomic_write_archive(ds: xr.Dataset, output_zarr: Path) -> None:
    # Mixed Python values can leave an object-typed archive variable even
    # after xarray concatenation. Zarr's VLenUTF8 codec then fails when an
    # integer appears in a nominally textual chunk. Materialize only those
    # small metadata variables as fixed-width Unicode before writing.
    object_variables = [name for name, var in ds.variables.items() if var.dtype.kind == "O"]
    if object_variables:
        ds = ds.copy()
        for name in object_variables:
            ds[name] = ds[name].astype(str).load()
    output_zarr.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_zarr.with_name(f"{output_zarr.name}.tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    chunk_spec = {}
    if "issue_time" in ds.sizes:
        chunk_spec["issue_time"] = min(max(ds.sizes.get("issue_time", 1), 1), 64)
    if "forecast_step" in ds.sizes:
        chunk_spec["forecast_step"] = min(max(ds.sizes.get("forecast_step", 1), 1), 64)
    if "ForecastValidTime" in ds:
        ds["ForecastValidTime"].encoding["units"] = "nanoseconds since 1970-01-01"
        ds["ForecastValidTime"].encoding["dtype"] = "int64"
    ds.chunk(chunk_spec).to_zarr(tmp, mode="w", consolidated=True)
    if output_zarr.exists():
        shutil.rmtree(output_zarr)
    tmp.rename(output_zarr)


def _atomic_write_skill(ds: xr.Dataset, output_zarr: Path) -> None:
    output_zarr.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_zarr.with_name(f"{output_zarr.name}.tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    chunk_spec = {}
    if "time" in ds.sizes:
        chunk_spec["time"] = min(max(ds.sizes.get("time", 1), 1), 288)
    ds.chunk(chunk_spec).to_zarr(tmp, mode="w", consolidated=True)
    if output_zarr.exists():
        shutil.rmtree(output_zarr)
    tmp.rename(output_zarr)


def _atomic_write_time_product(ds: xr.Dataset, output_zarr: Path) -> None:
    output_zarr.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_zarr.with_name(f"{output_zarr.name}.tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    chunk = min(max(ds.sizes.get("time", 1), 1), 672)
    ds.chunk({"time": chunk}).to_zarr(tmp, mode="w", consolidated=True)
    if output_zarr.exists():
        shutil.rmtree(output_zarr)
    tmp.rename(output_zarr)


def _balanced_alpha(sample_count: object, *, base: float = DEFAULT_ADAPTIVE_ALPHA) -> float:
    try:
        samples = max(float(sample_count), 0.0)
    except Exception:
        samples = 0.0
    if samples <= 0.0:
        return float(base)
    early = DEFAULT_MAX_ADAPTIVE_ALPHA / np.sqrt(samples)
    return float(np.clip(max(float(base), early), DEFAULT_MIN_ADAPTIVE_ALPHA, DEFAULT_MAX_ADAPTIVE_ALPHA))


def _latest_cached_forecast(cache_dir: Path, *, param: str = ECMWF_PARAM) -> Path:
    patterns = (f"*{param}*.grib2", f"*{param}*.grib", "*.grib2", "*.grib")
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(cache_dir.glob(pattern))
    candidates = [path for path in candidates if path.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No cached ECMWF GRIB files found in {cache_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def retrieve_open_data_grib(
    output_grib: Path,
    *,
    horizon_hours: int = DEFAULT_HORIZON_HOURS,
    lookahead_buffer_hours: int = DEFAULT_ECMWF_LOOKAHEAD_BUFFER_HOURS,
    param: str = ECMWF_PARAM,
    source: str = DEFAULT_OPEN_DATA_SOURCE,
    cycle_hour: int | None = None,
) -> Path:
    """Retrieve ECMWF open-data solar forecast GRIB for the requested horizon."""
    path, _diagnostics = retrieve_provider_open_data_grib(
        output_grib,
        provider="legacy",
        horizon_hours=horizon_hours,
        lookahead_buffer_hours=lookahead_buffer_hours,
        param=param,
        source=source,
        cycle_hour=cycle_hour,
    )
    return path


def open_solar_forecast(path: Path, *, latitude: float, longitude: float) -> xr.Dataset:
    """Backward-compatible legacy opener used by tests and external callers."""
    return open_provider_solar_forecast(
        path,
        provider="legacy",
        latitude=latitude,
        longitude=longitude,
        shadow_report_path=None,
    ).dataset


def _ecmwf_cycle_time(ds: xr.Dataset) -> pd.Timestamp | None:
    coord = "forecast_reference_time" if "forecast_reference_time" in ds.coords else "time"
    if coord not in ds.coords:
        return None
    values = np.asarray(ds[coord].values).reshape(-1)
    if values.size == 0:
        return None
    cycle = pd.Timestamp(values[0])
    if pd.isna(cycle):
        return None
    if cycle.tz is not None:
        cycle = cycle.tz_convert("UTC").tz_localize(None)
    return cycle


def _forecast_valid_time(ds: xr.Dataset, da: xr.DataArray) -> xr.DataArray:
    valid_time = ds["valid_time"] if "valid_time" in ds.coords else None
    if valid_time is None:
        if "forecast_reference_time" in da.coords and "lead_time" in da.coords:
            valid_time = da["forecast_reference_time"] + da["lead_time"]
        elif "time" in da.coords and "step" in da.coords:
            valid_time = da["time"] + da["step"]
        elif "time" in da.coords:
            valid_time = da["time"]
        else:
            raise KeyError("Solar forecast needs time or valid_time coordinates")
    return valid_time


def accumulated_irradiance_from_forecast(ds: xr.Dataset, variable: str) -> pd.Series:
    """Convert an accumulated forecast radiation field from J/m2 to interval W/m2."""
    if variable not in ds:
        return pd.Series(dtype=np.float64)
    da = ds[variable]
    valid_time = _forecast_valid_time(ds, da)

    values = np.asarray(da.values, dtype=np.float64).reshape(-1)
    times = pd.DatetimeIndex(np.asarray(valid_time.values).reshape(-1))
    frame = pd.DataFrame({variable: values}, index=times).sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    frame = frame[np.isfinite(frame[variable])]
    if len(frame) < 2:
        return pd.Series(dtype=np.float64)

    delta_j_m2 = frame[variable].diff()
    delta_seconds = frame.index.to_series().diff().dt.total_seconds()
    irradiance = delta_j_m2 / delta_seconds
    irradiance = irradiance.clip(lower=0.0)
    return irradiance.dropna()


def solar_irradiance_from_ssrd(ds: xr.Dataset) -> pd.Series:
    """Convert accumulated ECMWF SSRD J/m2 to interval W/m2."""
    return accumulated_irradiance_from_forecast(ds, "ssrd")


SITE_IRRADIANCE_DIGEST_ALGORITHM = "sha256-time-utc-ns-float64-hex-v1"


def _normalise_site_irradiance_override(irradiance: pd.Series) -> pd.Series:
    """Validate an exact site-extracted irradiance trace without resampling it.

    A leading NaN is allowed only for the SOC anchor, which has no preceding
    forecast interval by convention. Every actual interval endpoint must be
    finite and non-negative.
    """

    if not isinstance(irradiance, pd.Series):
        raise TypeError("site_irradiance_override must be a pandas Series")
    try:
        index = pd.DatetimeIndex(irradiance.index)
    except (TypeError, ValueError) as exc:
        raise ValueError("Embedded site irradiance has an invalid time grid") from exc
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    if len(index) < 2 or index.hasnans:
        raise ValueError("Embedded site irradiance needs at least two valid UTC times")
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("Embedded site irradiance time grid must be strictly increasing")
    values = np.asarray(irradiance.to_numpy(), dtype=np.float64).reshape(-1)
    if values.size != len(index):
        raise ValueError("Embedded site irradiance values do not match its time grid")
    nonfinite = ~np.isfinite(values)
    if np.any(nonfinite):
        allowed_anchor = bool(nonfinite[0]) and int(np.count_nonzero(nonfinite)) == 1
        if not allowed_anchor:
            raise ValueError(
                "Embedded site irradiance must be finite except for one leading SOC anchor"
            )
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        raise ValueError("Embedded site irradiance needs at least two forecast intervals")
    if np.any(finite < 0.0):
        raise ValueError("Embedded site irradiance cannot contain negative values")
    return pd.Series(values, index=index, name="ECMWFSolarIrradiance")


def site_irradiance_series_digest(irradiance: pd.Series) -> str:
    """Return a stable SHA256 for the exact site forcing values and UTC grid."""

    normalised = _normalise_site_irradiance_override(irradiance)
    payload = {
        "algorithm": SITE_IRRADIANCE_DIGEST_ALGORITHM,
        "samples": [
            [
                int(timestamp.value),
                None if not np.isfinite(value) else float(value).hex(),
            ]
            for timestamp, value in normalised.items()
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _instantaneous_forecast_series(ds: xr.Dataset, candidates: tuple[str, ...]) -> pd.Series:
    for name in candidates:
        if name not in ds:
            continue
        da = ds[name]
        valid_time = _forecast_valid_time(ds, da)
        values = np.asarray(da.values, dtype=np.float64).reshape(-1)
        times = pd.DatetimeIndex(np.asarray(valid_time.values).reshape(-1))
        if values.size != times.size:
            continue
        series = pd.Series(values, index=times).sort_index()
        return series[~series.index.duplicated(keep="last")].dropna()
    return pd.Series(dtype=np.float64)


def _physical_solar_meteorology(ds: xr.Dataset, irradiance: pd.Series) -> dict[str, pd.Series | None]:
    """Return issue-time forecast fields supported by the physical PV adapter."""
    direct = pd.Series(dtype=np.float64)
    for name in ("fdir", "surface_direct_solar_radiation"):
        direct = accumulated_irradiance_from_forecast(ds, name)
        if not direct.empty:
            break

    temperature = _instantaneous_forecast_series(ds, ("t2m", "2t", "temperature_2m"))
    if not temperature.empty and float(temperature.median()) > 150.0:
        temperature = temperature - 273.15
    u10 = _instantaneous_forecast_series(ds, ("u10", "10u"))
    v10 = _instantaneous_forecast_series(ds, ("v10", "10v"))
    wind = pd.Series(dtype=np.float64)
    if not u10.empty and not v10.empty:
        aligned_u, aligned_v = u10.align(v10, join="inner")
        wind = pd.Series(np.hypot(aligned_u.values, aligned_v.values), index=aligned_u.index)

    # ECMWF SSR is net downward shortwave.  Where both accumulated fields are
    # present, 1 - SSR/SSRD is the implied surface albedo.
    net_shortwave = accumulated_irradiance_from_forecast(ds, "ssr")
    albedo = pd.Series(dtype=np.float64)
    if not net_shortwave.empty:
        down, net = irradiance.align(net_shortwave, join="inner")
        values = 1.0 - np.divide(
            net.to_numpy(dtype=np.float64),
            down.to_numpy(dtype=np.float64),
            out=np.full(len(down), np.nan, dtype=np.float64),
            where=down.to_numpy(dtype=np.float64) > 10.0,
        )
        albedo = pd.Series(np.clip(values, 0.0, 0.95), index=down.index).dropna()
    return {
        "direct_horizontal_w_m2": direct if not direct.empty else None,
        "air_temperature_c": temperature if not temperature.empty else None,
        "wind_speed_m_s": wind if not wind.empty else None,
        "ground_albedo": albedo if not albedo.empty else None,
    }


def _power_frame(power: xr.Dataset) -> pd.DataFrame:
    fields = [
        name
        for name in (
            "BatterySOC",
            "SolarWatts_East",
            "SolarWatts_South",
            "SolarWatts_West",
            *SOLAR_MPP_MODE_FIELDS,
            "BatteryWatts",
            "ACOutputWatts",
            "DCInverterWatts",
        )
        if name in power and power[name].dims == ("time",)
    ]
    if "time" not in power or not fields:
        return pd.DataFrame()
    times = pd.DatetimeIndex(power["time"].values)
    frame = pd.DataFrame({name: np.asarray(power[name].values, dtype=np.float64) for name in fields}, index=times)
    frame = frame[~frame.index.isna()].sort_index()
    return frame[~frame.index.duplicated(keep="last")]


def latest_finite(series: pd.Series) -> tuple[pd.Timestamp, float]:
    finite = series[np.isfinite(series)]
    if finite.empty:
        raise ValueError(f"No finite samples available for {series.name or 'series'}")
    return pd.Timestamp(finite.index[-1]), float(finite.iloc[-1])


def validate_power_input_freshness(
    power: xr.Dataset,
    *,
    max_age_minutes: float | None,
    now: pd.Timestamp | None = None,
) -> tuple[pd.Timestamp, float]:
    """Return the latest SOC anchor and reject an operationally stale input."""
    frame = _power_frame(power)
    if frame.empty or "BatterySOC" not in frame:
        raise ValueError("Power dataset needs BatterySOC to initialize the SOC forecast")
    latest_time, latest_soc = latest_finite(frame["BatterySOC"])
    if max_age_minutes is None:
        return latest_time, latest_soc
    reference = pd.Timestamp(now if now is not None else datetime.now(timezone.utc))
    if reference.tzinfo is not None:
        reference = reference.tz_convert("UTC").tz_localize(None)
    if latest_time.tzinfo is not None:
        latest_time = latest_time.tz_convert("UTC").tz_localize(None)
    age_minutes = float((reference - latest_time) / pd.Timedelta(minutes=1))
    if age_minutes > float(max_age_minutes):
        raise ValueError(
            "Refusing to publish a forecast from stale SOC/load input: "
            f"latest BatterySOC is {age_minutes:.1f} minutes old "
            f"(limit {float(max_age_minutes):.1f} minutes)"
        )
    return latest_time, latest_soc


def estimate_load_w(frame: pd.DataFrame, *, end: pd.Timestamp, calibration_days: float) -> float:
    start = end - pd.Timedelta(days=float(calibration_days))
    window = frame.loc[frame.index >= start]
    load = _observed_load_w(window)
    finite = load[np.isfinite(load)]
    if finite.empty:
        return float(DEFAULT_LOAD_W)
    return float(finite.median())


def build_historical_load_forecast(
    frame: pd.DataFrame,
    forecast_times: pd.DatetimeIndex,
    *,
    end: pd.Timestamp,
    calibration_days: float,
    default_load_w: float = DEFAULT_LOAD_W,
) -> pd.Series:
    """Forecast total station load from the robust level of its current operating mode."""
    forecast_times = pd.DatetimeIndex(forecast_times)
    if len(forecast_times) == 0:
        return pd.Series(dtype=np.float64)
    start = end - pd.Timedelta(days=float(calibration_days))
    load = _observed_load_w(frame)
    load = load.loc[(load.index >= start) & (load.index <= end)]
    finite = load[np.isfinite(load)]
    if finite.empty:
        return pd.Series(np.full(len(forecast_times), float(default_load_w)), index=forecast_times)

    samples = finite.resample("15min").median().dropna()
    if samples.empty:
        samples = finite
    state_window_start = end - pd.Timedelta(minutes=float(DEFAULT_LOAD_MODE_STATE_MINUTES))
    state_window = samples[samples.index >= state_window_start]
    if state_window.empty:
        state_window = samples.iloc[-min(len(samples), 2) :]

    ac = frame.get("ACOutputWatts")
    if ac is not None:
        ac_samples = ac.loc[(ac.index >= start) & (ac.index <= end)].resample("15min").median()
        ac_samples = ac_samples.reindex(samples.index, method="nearest", tolerance=pd.Timedelta(minutes=10))
        recent_ac = ac_samples[ac_samples.index >= state_window_start].dropna()
        current_ac_w = float(recent_ac.median()) if not recent_ac.empty else float(ac_samples.dropna().iloc[-1])
        current_active = bool(current_ac_w > DEFAULT_AC_MODE_THRESHOLD_W)
        states = ac_samples.fillna(current_ac_w).to_numpy(dtype=np.float64) > DEFAULT_AC_MODE_THRESHOLD_W
        mode_state = "ac-active" if current_active else "dc-only"
        split_w = float(DEFAULT_AC_MODE_THRESHOLD_W)
    else:
        values = samples.to_numpy(dtype=np.float64)
        split_w = _load_regime_threshold(values)
        current_reference = float(state_window.median())
        current_active = bool(np.isfinite(split_w) and current_reference > split_w)
        states = values > split_w if np.isfinite(split_w) else np.zeros(len(values), dtype=bool)
        current_ac_w = np.nan
        mode_state = "unlabelled-active" if current_active else "dc-only"

    opposite = np.flatnonzero(states != current_active)
    run_start_index = int(opposite[-1] + 1) if opposite.size else 0
    current_run = samples.iloc[run_start_index:]
    same_mode = samples.iloc[np.flatnonzero(states == current_active)]

    level_start = end - pd.Timedelta(hours=float(DEFAULT_LOAD_MODE_LEVEL_HOURS))
    level_samples = current_run[current_run.index >= level_start]
    measurement = _load_measurement_name(frame)
    # Dark-period discharge is useful for learning the DC-only component, but
    # it is not the current whole-station load when powered instruments are on.
    # The operational forecast must remain anchored to the latest stable
    # solar-minus-battery balance for the detected mode.
    level_measurement = measurement
    if len(level_samples) < 4:
        level_samples = same_mode[same_mode.index >= level_start]
    if level_samples.empty:
        level_samples = state_window
    learned_level = float(level_samples.median()) if not level_samples.empty else float(default_load_w)
    recent_anchor = state_window.tail(3).dropna()
    level = float(recent_anchor.median()) if len(recent_anchor) >= 2 else learned_level
    level = max(level, 0.0) if np.isfinite(level) else float(default_load_w)
    run_hours = max(float((end - current_run.index[0]) / pd.Timedelta(hours=1)), 0.0) if len(current_run) else 0.0
    forecast = pd.Series(np.full(len(forecast_times), level, dtype=np.float64), index=forecast_times)
    mode_name = "AC Load (Unlabelled)" if current_active else "DC-Only"
    forecast.attrs.update(
        {
            "load_model": LOAD_MODEL_NAME,
            "load_model_version": LOAD_MODEL_VERSION,
            "load_mode": mode_name,
            "load_mode_state": mode_state,
            "load_mode_source": "ac_output" if ac is not None else "load_level",
            "load_measurement": level_measurement,
            "load_balance_measurement": measurement,
            "load_current_ac_w": current_ac_w,
            "load_regime": mode_name,
            "load_regime_threshold_w": split_w,
            "load_regime_level_w": level,
            "load_recent_anchor_w": level,
            "load_learned_reference_w": learned_level,
            "load_anchor_method": "median_latest_30_minutes_whole_station_balance",
            "load_regime_run_hours": run_hours,
            "load_regime_sample_count": int(len(level_samples)),
        }
    )
    return forecast


def _load_regime_threshold(values: np.ndarray) -> float:
    """Find a material low/high load split using the largest gap in log power."""
    finite = np.sort(np.asarray(values, dtype=np.float64))
    finite = finite[np.isfinite(finite) & (finite >= 0.0)]
    if finite.size < 20:
        return np.nan
    low_quantile, high_quantile = np.nanquantile(finite, [0.10, 0.90])
    candidates = np.unique(finite[(finite >= low_quantile) & (finite <= high_quantile)])
    if candidates.size < 2:
        return np.nan
    log_values = np.log1p(candidates)
    gap_index = int(np.argmax(np.diff(log_values)))
    lower = float(candidates[gap_index])
    upper = float(candidates[gap_index + 1])
    if (upper + 1.0) / (lower + 1.0) < 3.0:
        return np.nan
    return float(np.expm1((np.log1p(lower) + np.log1p(upper)) / 2.0))


def _normalise_load_mode_registry(value: object) -> dict[str, dict[str, object]]:
    if not isinstance(value, dict):
        return {}
    registry: dict[str, dict[str, object]] = {}
    for label, raw_entry in value.items():
        if not isinstance(label, str) or not label or not isinstance(raw_entry, dict):
            continue
        observations = []
        for raw_observation in raw_entry.get("observations", []):
            if not isinstance(raw_observation, dict):
                continue
            try:
                timestamp = pd.Timestamp(raw_observation["time"])
                level_w = float(raw_observation["level_w"])
            except Exception:
                continue
            if pd.isna(timestamp) or not np.isfinite(level_w) or level_w < 0.0:
                continue
            observations.append({"time": timestamp.isoformat(), "level_w": level_w})
        entry = dict(raw_entry)
        entry["observations"] = observations[-168:]
        registry[label] = entry
    return registry


def _pdu_active_kits(
    pdu: xr.Dataset | None,
    *,
    end: pd.Timestamp,
) -> tuple[list[str], pd.Timestamp | None, float]:
    if pdu is None or "time" not in pdu or pdu.sizes.get("time", 0) == 0:
        return [], None, np.nan
    times = pd.DatetimeIndex(pdu["time"].values)
    eligible = np.flatnonzero((~times.isna()) & (times <= end))
    if not eligible.size:
        return [], None, np.nan
    latest_index = int(eligible[np.argmax(times[eligible].to_numpy(dtype="datetime64[ns]"))])
    latest_time = pd.Timestamp(times[latest_index])
    if end - latest_time > pd.Timedelta(minutes=float(DEFAULT_PDU_MODE_FRESHNESS_MINUTES)):
        return [], latest_time, np.nan
    window_indices = eligible[times[eligible] >= latest_time - pd.Timedelta(minutes=30)]
    active: list[str] = []
    active_watts = 0.0
    for outlet in range(1, 9):
        is_active = False
        state_name = f"PDUOutlet{outlet}State"
        watts_name = f"PDUOutlet{outlet}Watts"
        watts_level = np.nan
        watts_available = False
        if watts_name in pdu:
            watts = np.asarray(pdu[watts_name].isel(time=window_indices).values, dtype=np.float64)
            watts = watts[np.isfinite(watts)]
            if watts.size:
                watts_level = float(np.nanmedian(watts))
                watts_available = True
                is_active = bool(watts_level >= DEFAULT_PDU_ACTIVE_W_THRESHOLD)
        if not watts_available and state_name in pdu:
            states = np.asarray(pdu[state_name].isel(time=window_indices).values, dtype=np.float64)
            states = states[np.isfinite(states)]
            is_active = bool(states.size and np.nanmedian(states) >= 0.5)
        if is_active:
            active.append(PDU_OUTLET_KIT_NAMES.get(outlet, f"AC Outlet {outlet}"))
            if np.isfinite(watts_level):
                active_watts += max(watts_level, 0.0)
    return active, latest_time, active_watts if active else np.nan


def _resolve_load_mode(
    frame: pd.DataFrame,
    pdu: xr.Dataset | None,
    *,
    end: pd.Timestamp,
    observed_level_w: float,
    raw_registry: object,
    previous_mode: object,
) -> tuple[str, str, list[str], pd.Timestamp | None, float]:
    state_start = end - pd.Timedelta(minutes=float(DEFAULT_LOAD_MODE_STATE_MINUTES))
    recent_ac = frame.loc[frame.index >= state_start, "ACOutputWatts"].dropna() if "ACOutputWatts" in frame else pd.Series(dtype=float)
    current_ac_w = float(recent_ac.median()) if not recent_ac.empty else 0.0
    latest_ac_w = float(recent_ac.iloc[-1]) if not recent_ac.empty else current_ac_w

    active_kits, pdu_time, pdu_active_watts = _pdu_active_kits(pdu, end=end)
    if active_kits:
        # Use the latest AC sample for immediate PDU classification. The
        # smoothed level remains responsible for sustained-mode learning.
        if "CL61" in active_kits and latest_ac_w > DEFAULT_AC_MODE_THRESHOLD_W:
            return "Ceilometer-on-AC", "pdu_ac_signature", active_kits, pdu_time, pdu_active_watts
        return f"DC-Only + {' + '.join(active_kits)}", "pdu_signature", active_kits, pdu_time, pdu_active_watts
    if current_ac_w <= DEFAULT_AC_MODE_THRESHOLD_W:
        return "DC-Only", "ac_output", [], pdu_time, np.nan

    registry = _normalise_load_mode_registry(raw_registry)
    candidates: list[tuple[float, str]] = []
    for label, entry in registry.items():
        if label == "DC-Only":
            continue
        try:
            learned_level = float(entry.get("learned_level_w", np.nan))
        except Exception:
            continue
        if np.isfinite(learned_level):
            candidates.append((abs(learned_level - observed_level_w), label))
    if candidates:
        difference, label = min(candidates)
        if difference <= max(75.0, 0.20 * max(observed_level_w, 1.0)):
            return label, "learned_power_match", [], pdu_time, np.nan
    if isinstance(previous_mode, str) and previous_mode != "DC-Only" and previous_mode in registry:
        return previous_mode, "persisted_mode", [], pdu_time, np.nan
    return "AC Load (Unlabelled)", "ac_output", [], pdu_time, np.nan


def _load_mode_signature(mode: str, mode_source: str, active_kits: list[str]) -> str:
    if mode == "Ceilometer-on-AC" and "CL61" in active_kits:
        return (
            f"PDUOutlet{PDU_KIT_OUTLETS['CL61']}Watts>={DEFAULT_PDU_ACTIVE_W_THRESHOLD:g}W+"
            f"ACOutputWatts>{DEFAULT_AC_MODE_THRESHOLD_W:g}W"
        )
    if mode_source == "pdu_signature" and active_kits:
        parts = []
        for kit in active_kits:
            outlet = PDU_KIT_OUTLETS.get(kit)
            if outlet is None:
                parts.append(kit)
            else:
                parts.append(f"PDUOutlet{outlet}Watts>={DEFAULT_PDU_ACTIVE_W_THRESHOLD:g}W")
        return "+".join(parts)
    if mode == "DC-Only":
        return f"ACOutputWatts<={DEFAULT_AC_MODE_THRESHOLD_W:g}W"
    if mode_source == "learned_power_match":
        return "learned_total_load_match"
    return f"ACOutputWatts>{DEFAULT_AC_MODE_THRESHOLD_W:g}W"


def _mode_learning_status(load_diagnostics: dict[str, object], mode: str) -> tuple[bool, str]:
    state = str(load_diagnostics.get("load_mode_state", ""))
    expects_ac = mode != "DC-Only"
    active_states = {"ac-active", "unlabelled-active"}
    if expects_ac and state not in active_states:
        return False, "waiting_for_ac_state"
    if not expects_ac and state != "dc-only":
        return False, "waiting_for_dc_state"
    run_hours = float(load_diagnostics.get("load_regime_run_hours", 0.0) or 0.0)
    if run_hours * 60.0 < float(DEFAULT_LOAD_MODE_STATE_MINUTES):
        return False, "waiting_for_stable_duration"
    sample_count = int(load_diagnostics.get("load_regime_sample_count", 0) or 0)
    if sample_count < DEFAULT_LOAD_MODE_MIN_STABLE_SAMPLES:
        return False, "waiting_for_stable_samples"
    return True, "stable"


def _update_load_mode_registry(
    raw_registry: object,
    *,
    mode: str,
    observed_level_w: float,
    observed_at: pd.Timestamp,
    active_kits: list[str] | None = None,
    mode_source: str | None = None,
    signature: str | None = None,
) -> tuple[dict[str, dict[str, object]], float]:
    registry = _normalise_load_mode_registry(raw_registry)
    entry = dict(registry.get(mode, {}))
    observations = list(entry.get("observations", []))
    observation = {"time": pd.Timestamp(observed_at).isoformat(), "level_w": float(observed_level_w)}
    if observations:
        last_time = pd.Timestamp(observations[-1]["time"])
        if observed_at - last_time < pd.Timedelta(minutes=float(DEFAULT_LOAD_MODE_LEARN_INTERVAL_MINUTES)):
            observations[-1] = observation
        else:
            observations.append(observation)
    else:
        observations.append(observation)
    observations = observations[-168:]
    levels = np.asarray([item["level_w"] for item in observations], dtype=np.float64)
    learned_level = float(np.nanmedian(levels))
    entry.update(
        {
            "learned_level_w": learned_level,
            "latest_observed_level_w": float(observed_level_w),
            "observation_count": int(len(observations)),
            "last_seen": pd.Timestamp(observed_at).isoformat(),
            "observations": observations,
        }
    )
    if active_kits is not None:
        entry["active_kits"] = list(active_kits)
    if mode_source:
        entry["mode_source"] = str(mode_source)
    if signature:
        entry["signature"] = str(signature)
    registry[mode] = entry
    return registry, learned_level


def _observed_solar_w(frame: pd.DataFrame) -> pd.Series:
    solar_fields = [name for name in ("SolarWatts_East", "SolarWatts_South", "SolarWatts_West") if name in frame]
    if not solar_fields:
        return pd.Series(dtype=np.float64)
    return frame[solar_fields].sum(axis=1, min_count=1).clip(lower=0.0)


def _mpp_active_available_power_mask(frame: pd.DataFrame) -> pd.Series:
    """Return times where every metered array can represent available PV.

    Victron charger output is delivered power, not necessarily available panel
    power.  It is valid as a physical-PV target only while each independently
    metered charger reports register 791 mode 2 (MPPT active).  A missing or
    non-finite register is deliberately treated as censored instead of guessed.
    """
    if frame.empty or not set(SOLAR_MPP_MODE_FIELDS).issubset(frame.columns):
        return pd.Series(False, index=frame.index, dtype=bool)
    modes = frame.loc[:, list(SOLAR_MPP_MODE_FIELDS)].apply(pd.to_numeric, errors="coerce")
    return modes.eq(MPP_ACTIVE_MODE).all(axis=1)


def _solar_product_is_available_power(product: xr.Dataset) -> bool:
    """Return whether solar output is potential PV, not metered delivery."""
    model_name = str(product.attrs.get("solar_model_name", "")).strip().lower()
    semantics = str(product.attrs.get("solar_power_semantics", "")).strip().lower()
    return model_name == PHYSICAL_SOLAR_MODEL_NAME or "available" in semantics


def _active_forecast_contract_id(archive: xr.Dataset) -> str | None:
    """Select the newest contract when an archive contains real revisions."""
    if "ForecastModelContractID" not in archive:
        return None
    values = [
        str(value).strip()
        for value in np.asarray(archive["ForecastModelContractID"].values).reshape(-1)
    ]
    nonempty = [value for value in values if value and value.lower() != "nan"]
    if not nonempty:
        return None
    # Blank legacy identities are never evidence for a newly identified
    # contract.  Select the newest real contract even when it is the first one
    # archived, so all downstream score surfaces exclude unidentified rows.
    return nonempty[-1]


def _active_forecast_system_version(archive: xr.Dataset, contract_id: str | None) -> str | None:
    """Return the version paired with the selected contract, if archived."""
    if contract_id is None or "ForecastSystemVersion" not in archive:
        return None
    contracts = np.asarray(archive.get("ForecastModelContractID", xr.DataArray([])).values, dtype=str).reshape(-1)
    systems = np.asarray(archive["ForecastSystemVersion"].values, dtype=str).reshape(-1)
    selected = [
        str(system).strip()
        for contract, system in zip(contracts, systems, strict=False)
        if str(contract).strip() == contract_id and str(system).strip() and str(system).lower() != "nan"
    ]
    return selected[-1] if selected else None


def _filter_active_forecast_contract(table: pd.DataFrame, archive: xr.Dataset) -> pd.DataFrame:
    """Keep rows with the newest complete stable semantic forecast identity.

    ``ForecastIdentityID`` and ``AdaptiveCalibrationStateID`` identify one
    reproducible issue and therefore vary normally across a campaign.  The
    fields below describe the stable implementation/configuration contract and
    must all match before deterministic, hindcast, mode-specific, or adaptive
    learning calculations may pool archived rows.
    """
    target = _active_forecast_contract_id(archive)
    if target is None or table.empty or "forecast_model_contract_id" not in table:
        return table
    filtered = table[table["forecast_model_contract_id"] == target]
    contracts = np.asarray(archive["ForecastModelContractID"].values, dtype=str).reshape(-1)
    target_indices = np.flatnonzero(contracts == target)
    if target_indices.size == 0:
        return filtered.iloc[0:0]
    target_index = int(target_indices[-1])
    table_names = {
        "ForecastSystemVersion": "forecast_system_version",
        "FeatureSetVersion": "feature_set_version",
        "FeatureSetDigest": "feature_set_digest",
        "ForecastCodeRevision": "forecast_code_revision",
        "CandidateLane": "candidate_lane",
        "LocalFeatureContractID": "local_feature_contract_id",
        "BaselineControlContractID": "baseline_control_contract_id",
        "BaselineControlSystemVersion": "baseline_control_system_version",
    }
    for archive_name in FORECAST_SEMANTIC_IDENTITY_ARCHIVE_FIELDS[1:]:
        table_name = table_names[archive_name]
        if archive_name not in archive or table_name not in filtered:
            # Once a real contract exists, unidentified/incomplete rows must
            # not silently enter evidence or adaptive state.
            return filtered.iloc[0:0]
        values = np.asarray(archive[archive_name].fillna("").values, dtype=str).reshape(-1)
        target_value = str(values[target_index])
        filtered = filtered[filtered[table_name].astype(str) == target_value]
    return filtered


def _filter_solar_verification_rows(
    table: pd.DataFrame,
    *,
    mpp_active_mask: pd.Series | None = None,
    tolerance: pd.Timedelta = pd.Timedelta(minutes=10),
) -> tuple[pd.DataFrame, int, int]:
    """Keep physical-PV rows only when all three chargers were MPPT-active.

    Legacy forecasts model delivered charger output and need no MPP filter.
    Physical forecasts model available PV, so limited output is censored and
    cannot be used as a truth target.  The returned counts are respectively
    censored physical rows and eligible physical rows.
    """
    if table.empty:
        return table, 0, 0
    model = table.get("solar_model_name", pd.Series("", index=table.index)).astype(str).str.lower()
    semantics = table.get("solar_power_semantics", pd.Series("", index=table.index)).astype(str).str.lower()
    physical = (model == PHYSICAL_SOLAR_MODEL_NAME) | semantics.str.contains("available", regex=False)
    eligible_physical = np.zeros(len(table), dtype=bool)
    if bool(physical.any()) and mpp_active_mask is not None and not mpp_active_mask.empty:
        valid_times = pd.DatetimeIndex(table["valid_time"])
        mpp_at_valid_time = mpp_active_mask.reindex(
            valid_times,
            method="nearest",
            tolerance=tolerance,
        )
        eligible_physical = mpp_at_valid_time.eq(True).to_numpy(dtype=bool)
    eligible = ~physical.to_numpy(dtype=bool) | eligible_physical
    physical_values = physical.to_numpy(dtype=bool)
    excluded = int(np.count_nonzero(physical_values & ~eligible_physical))
    accepted = int(np.count_nonzero(physical_values & eligible_physical))
    return table.loc[eligible], excluded, accepted


def _solar_verification_status(*, excluded: int, accepted_available_power: int) -> str | None:
    if accepted_available_power:
        return (
            "eligible_available_power_mpp_active"
            if not excluded
            else "partially_eligible_available_power_mpp_active"
        )
    if excluded:
        return "excluded_available_power_observations_are_censored"
    return None


def _observed_load_w(frame: pd.DataFrame) -> pd.Series:
    solar_fields = [name for name in ("SolarWatts_East", "SolarWatts_South", "SolarWatts_West") if name in frame]
    if "BatteryWatts" in frame and len(solar_fields) == 3:
        solar = frame[solar_fields].sum(axis=1, min_count=len(solar_fields))
        # APS BatteryWatts is positive while charging and negative while
        # discharging, so generation minus battery flow is total station load.
        balanced = (solar - frame["BatteryWatts"]).clip(lower=0.0)
        if np.isfinite(balanced.to_numpy(dtype=np.float64)).any():
            return balanced
    load_fields = [name for name in ("ACOutputWatts", "DCInverterWatts") if name in frame]
    if not load_fields:
        return pd.Series(dtype=np.float64)
    return frame[load_fields].sum(axis=1, min_count=1).clip(lower=0.0)


def _clean_dc_only_observation(
    frame: pd.DataFrame,
    *,
    end: pd.Timestamp,
) -> tuple[float, pd.Timestamp, int] | None:
    solar_fields = [name for name in ("SolarWatts_East", "SolarWatts_South", "SolarWatts_West") if name in frame]
    required = {"BatteryWatts", "ACOutputWatts"}
    if len(solar_fields) != 3 or not required.issubset(frame.columns):
        return None
    start = end - pd.Timedelta(hours=float(DEFAULT_DARK_LOAD_LOOKBACK_HOURS))
    samples = pd.DataFrame(
        {
            "load_w": _observed_load_w(frame),
            "solar_w": frame[solar_fields].sum(axis=1, min_count=3),
            "battery_w": frame["BatteryWatts"],
            "ac_w": frame["ACOutputWatts"],
        }
    ).loc[start:end]
    samples = samples.resample("15min").median()
    clean = samples.loc[
        (samples["solar_w"] <= DEFAULT_ZERO_SOLAR_THRESHOLD_W)
        & (samples["battery_w"] < 0.0)
        & (samples["ac_w"] <= DEFAULT_AC_MODE_THRESHOLD_W),
        "load_w",
    ].dropna()
    if len(clean) < 4:
        return None
    return float(clean.median()), pd.Timestamp(clean.index[-1]), int(len(clean))


def _repair_dc_only_registry(
    raw_registry: object,
    frame: pd.DataFrame,
    *,
    end: pd.Timestamp,
) -> tuple[dict[str, dict[str, object]], float | None]:
    registry = _normalise_load_mode_registry(raw_registry)
    clean = _clean_dc_only_observation(frame, end=end)
    if clean is None:
        return registry, None
    level_w, observed_at, sample_count = clean
    tolerance_w = max(75.0, 0.35 * max(level_w, 1.0))
    entry = dict(registry.get("DC-Only", {}))
    observations = [
        observation
        for observation in entry.get("observations", [])
        if abs(float(observation["level_w"]) - level_w) <= tolerance_w
    ]
    replacement = {"time": observed_at.isoformat(), "level_w": level_w}
    replaced = False
    for index, observation in enumerate(observations):
        if abs(pd.Timestamp(observation["time"]) - observed_at) < pd.Timedelta(
            minutes=float(DEFAULT_LOAD_MODE_LEARN_INTERVAL_MINUTES)
        ):
            observations[index] = replacement
            replaced = True
            break
    if not replaced:
        observations.append(replacement)
    observations.sort(key=lambda observation: pd.Timestamp(observation["time"]))
    observations = observations[-168:]
    levels = np.asarray([observation["level_w"] for observation in observations], dtype=np.float64)
    latest = observations[-1]
    entry.update(
        {
            "learned_level_w": float(np.nanmedian(levels)),
            "latest_observed_level_w": float(latest["level_w"]),
            "observation_count": int(len(observations)),
            "last_seen": str(latest["time"]),
            "observations": observations,
            "active_kits": [],
            "mode_source": "battery_discharge_when_solar_zero",
            "signature": (
                f"ACOutputWatts<={DEFAULT_AC_MODE_THRESHOLD_W:g}W+"
                f"SolarTotalWatts<={DEFAULT_ZERO_SOLAR_THRESHOLD_W:g}W"
            ),
            "clean_dark_sample_count": sample_count,
        }
    )
    registry["DC-Only"] = entry
    return registry, level_w


def _load_measurement_name(frame: pd.DataFrame) -> str:
    solar_fields = {"SolarWatts_East", "SolarWatts_South", "SolarWatts_West"}
    if "BatteryWatts" in frame and solar_fields.issubset(frame.columns):
        return "solar_generation_minus_battery_power"
    return "ac_plus_dc_output_fallback"


def evaluate_previous_forecast(previous: xr.Dataset | None, frame: pd.DataFrame) -> dict[str, float | int | str]:
    """Score the previous forecast against newly arrived APS observations."""
    if previous is None or "time" not in previous or previous.sizes.get("time", 0) == 0 or frame.empty:
        return {}
    forecast_times = pd.DatetimeIndex(previous["time"].values)
    observed_end = pd.Timestamp(frame.index.max())
    valid_forecast = forecast_times <= observed_end
    if not np.any(valid_forecast):
        return {}

    metrics: dict[str, float | int | str] = {}
    if "BatterySOCForecast" in previous and "BatterySOC" in frame:
        forecast_soc = pd.Series(np.asarray(previous["BatterySOCForecast"].values, dtype=np.float64), index=forecast_times)
        forecast_soc = forecast_soc.loc[valid_forecast]
        observed_soc = frame["BatterySOC"].reindex(forecast_soc.index, method="nearest", tolerance=pd.Timedelta(minutes=10))
        valid = np.isfinite(forecast_soc.to_numpy()) & np.isfinite(observed_soc.to_numpy())
        if np.count_nonzero(valid) >= 2:
            errors = forecast_soc.to_numpy()[valid] - observed_soc.to_numpy()[valid]
            metrics["soc_mae_pct_points"] = float(np.mean(np.abs(errors)))
            metrics["soc_bias_pct_points"] = float(np.mean(errors))
            metrics["soc_sample_count"] = int(np.count_nonzero(valid))

    if "ForecastSolarWatts" in previous:
        forecast_solar = pd.Series(np.asarray(previous["ForecastSolarWatts"].values, dtype=np.float64), index=forecast_times)
        forecast_solar = forecast_solar.loc[valid_forecast]
        observed_solar = _observed_solar_w(frame).reindex(forecast_solar.index, method="nearest", tolerance=pd.Timedelta(minutes=10))
        available_power = _solar_product_is_available_power(previous)
        if available_power:
            mpp_active = _mpp_active_available_power_mask(frame).reindex(
                forecast_solar.index,
                method="nearest",
                tolerance=pd.Timedelta(minutes=10),
            )
            mpp_eligible = mpp_active.eq(True).to_numpy(dtype=bool)
        else:
            mpp_eligible = np.ones(len(forecast_solar), dtype=bool)
        valid = (
            np.isfinite(forecast_solar.to_numpy())
            & np.isfinite(observed_solar.to_numpy())
            & mpp_eligible
        )
        if np.count_nonzero(valid) >= 2:
            errors = forecast_solar.to_numpy()[valid] - observed_solar.to_numpy()[valid]
            metrics["solar_mae_w"] = float(np.mean(np.abs(errors)))
            metrics["solar_bias_w"] = float(np.mean(errors))
            metrics["solar_sample_count"] = int(np.count_nonzero(valid))
        if available_power:
            metrics["solar_verification_status"] = _solar_verification_status(
                excluded=int(np.count_nonzero(~mpp_eligible)),
                accepted_available_power=int(np.count_nonzero(mpp_eligible)),
            ) or "excluded_available_power_observations_are_censored"

    if "ForecastLoadWatts" in previous:
        forecast_load = pd.Series(np.asarray(previous["ForecastLoadWatts"].values, dtype=np.float64), index=forecast_times)
        forecast_load = forecast_load.loc[valid_forecast]
        observed_load = _observed_load_w(frame).reindex(forecast_load.index, method="nearest", tolerance=pd.Timedelta(minutes=10))
        valid = np.isfinite(forecast_load.to_numpy()) & np.isfinite(observed_load.to_numpy())
        if np.count_nonzero(valid) >= 2:
            errors = forecast_load.to_numpy()[valid] - observed_load.to_numpy()[valid]
            metrics["load_mae_w"] = float(np.mean(np.abs(errors)))
            metrics["load_bias_w"] = float(np.mean(errors))
            metrics["load_sample_count"] = int(np.count_nonzero(valid))

    if metrics:
        metrics["evaluated_at_utc"] = _utc_now()
    return metrics


def _metric_bucket_name(metric: str, bucket: str) -> str:
    return f"{metric}_{bucket}"


def scenario_soc_field(load_w: int | float) -> str:
    return f"BatterySOCForecast_Load{int(load_w)}W"


def scenario_load_label(load_w: int | float) -> str:
    return f"{int(load_w)} W Load"


def _errors_for_archive_variable(
    archive: xr.Dataset,
    frame: pd.DataFrame,
    *,
    forecast_var: str,
    observed: pd.Series,
    tolerance: pd.Timedelta,
    load_model_version: int | None = None,
    forecast_model_contract_id: str | None = None,
    forecast_system_version: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if forecast_var not in archive or "ForecastValidTime" not in archive or "ForecastLeadHours" not in archive:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    forecast_values = np.asarray(archive[forecast_var].values, dtype=np.float64).reshape(-1)
    valid_times = pd.DatetimeIndex(np.asarray(archive["ForecastValidTime"].values).reshape(-1))
    lead_hours = np.asarray(archive["ForecastLeadHours"].values, dtype=np.float64).reshape(-1)
    model_versions = None
    if load_model_version is not None:
        if "LoadModelVersion" not in archive:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
        model_versions = np.repeat(
            np.asarray(archive["LoadModelVersion"].values, dtype=np.float64).reshape(-1),
            int(archive.sizes.get("forecast_step", 0)),
        )
    forecast_contracts = None
    if forecast_model_contract_id is not None:
        if "ForecastModelContractID" not in archive:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
        forecast_contracts = np.repeat(
            np.asarray(archive["ForecastModelContractID"].values, dtype=str).reshape(-1),
            int(archive.sizes.get("forecast_step", 0)),
        )
    forecast_systems = None
    if forecast_system_version is not None:
        if "ForecastSystemVersion" not in archive:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
        forecast_systems = np.repeat(
            np.asarray(archive["ForecastSystemVersion"].values, dtype=str).reshape(-1),
            int(archive.sizes.get("forecast_step", 0)),
        )
    observed_end = pd.Timestamp(frame.index.max())
    valid_mask = (
        np.isfinite(forecast_values)
        & np.isfinite(lead_hours)
        & ~valid_times.isna()
        & (valid_times <= observed_end)
    )
    if model_versions is not None:
        valid_mask &= np.isfinite(model_versions) & (model_versions == float(load_model_version))
    if forecast_contracts is not None:
        valid_mask &= forecast_contracts == str(forecast_model_contract_id)
    if forecast_systems is not None:
        valid_mask &= forecast_systems == str(forecast_system_version)
    if not np.any(valid_mask):
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    forecast_values = forecast_values[valid_mask]
    valid_times = valid_times[valid_mask]
    lead_hours = lead_hours[valid_mask]
    observed_values = observed.reindex(valid_times, method="nearest", tolerance=tolerance).to_numpy(dtype=np.float64)
    paired = np.isfinite(observed_values)
    if not np.any(paired):
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    return forecast_values[paired] - observed_values[paired], lead_hours[paired]


def _archive_verification_frame(
    archive: xr.Dataset,
    observed: pd.Series,
    *,
    forecast_var: str,
    tolerance: pd.Timedelta,
) -> pd.DataFrame:
    if forecast_var not in archive or "ForecastValidTime" not in archive or "ForecastLeadHours" not in archive:
        return pd.DataFrame()
    forecast_values = np.asarray(archive[forecast_var].values, dtype=np.float64).reshape(-1)
    valid_times = pd.DatetimeIndex(np.asarray(archive["ForecastValidTime"].values).reshape(-1))
    lead_hours = np.asarray(archive["ForecastLeadHours"].values, dtype=np.float64).reshape(-1)
    issue_grid = np.repeat(
        pd.DatetimeIndex(archive["issue_time"].values).to_numpy(dtype="datetime64[ns]"),
        int(archive.sizes.get("forecast_step", 0)),
    )
    issue_times = pd.DatetimeIndex(issue_grid)
    if "LoadModelVersion" in archive:
        model_versions = np.repeat(
            np.asarray(archive["LoadModelVersion"].values, dtype=np.float64).reshape(-1),
            int(archive.sizes.get("forecast_step", 0)),
        )
    else:
        model_versions = np.full(len(forecast_values), np.nan, dtype=np.float64)
    if "LoadMode" in archive:
        load_modes = np.repeat(
            np.asarray(archive["LoadMode"].values, dtype=str).reshape(-1),
            int(archive.sizes.get("forecast_step", 0)),
        )
    else:
        load_modes = np.full(len(forecast_values), "unknown", dtype="U16")
    if "LoadModeLearningReady" in archive:
        mode_learning_ready = np.repeat(
            np.asarray(archive["LoadModeLearningReady"].values, dtype=bool).reshape(-1),
            int(archive.sizes.get("forecast_step", 0)),
        )
    else:
        mode_learning_ready = np.zeros(len(forecast_values), dtype=bool)
    if "ForecastVerificationEligible" in archive:
        verification_eligible = np.repeat(
            np.asarray(archive["ForecastVerificationEligible"].fillna(False).values, dtype=bool).reshape(-1),
            int(archive.sizes.get("forecast_step", 0)),
        )
    else:
        # Archives predating the explicit marker contained independent-cycle
        # rows only, so retain their established evaluation behaviour.
        verification_eligible = np.ones(len(forecast_values), dtype=bool)
    if "IndependentCycle" in archive:
        independent_cycles = np.repeat(
            np.asarray(archive["IndependentCycle"].fillna(False).values, dtype=bool).reshape(-1),
            int(archive.sizes.get("forecast_step", 0)),
        )
    else:
        independent_cycles = verification_eligible.copy()

    def repeated_text(name: str, default: str = "") -> np.ndarray:
        if name not in archive:
            return np.full(len(forecast_values), default, dtype="U1")
        return np.repeat(
            np.asarray(archive[name].values, dtype=str).reshape(-1),
            int(archive.sizes.get("forecast_step", 0)),
        )

    forecast_contracts = repeated_text("ForecastModelContractID")
    forecast_identity_ids = repeated_text("ForecastIdentityID")
    forecast_system_versions = repeated_text("ForecastSystemVersion")
    feature_set_versions = repeated_text("FeatureSetVersion")
    feature_set_digests = repeated_text("FeatureSetDigest")
    forecast_code_revisions = repeated_text("ForecastCodeRevision")
    source_cycle_set_ids = repeated_text("SourceCycleSetID")
    degraded_mode_codes = repeated_text("DegradedModeCode")
    candidate_lanes = repeated_text("CandidateLane")
    local_feature_contract_ids = repeated_text("LocalFeatureContractID")
    baseline_control_contract_ids = repeated_text("BaselineControlContractID")
    baseline_control_system_versions = repeated_text("BaselineControlSystemVersion")
    solar_model_names = repeated_text("SolarModelName")
    solar_power_semantics = repeated_text("SolarPowerSemantics")
    if "ECMWFCycleTime" in archive:
        cycle_grid = np.repeat(
            pd.DatetimeIndex(archive["ECMWFCycleTime"].values).to_numpy(dtype="datetime64[ns]"),
            int(archive.sizes.get("forecast_step", 0)),
        )
        cycle_times = pd.DatetimeIndex(cycle_grid)
    else:
        cycle_times = issue_times.floor("3h")
    observed_end = pd.Timestamp(observed.index.max()) if not observed.empty else pd.NaT
    valid_mask = (
        np.isfinite(forecast_values)
        & np.isfinite(lead_hours)
        & ~valid_times.isna()
        & ~issue_times.isna()
        & (valid_times <= observed_end)
    )
    if not np.any(valid_mask):
        return pd.DataFrame()
    forecast_values = forecast_values[valid_mask]
    valid_times = valid_times[valid_mask]
    issue_times = issue_times[valid_mask]
    cycle_times = cycle_times[valid_mask]
    model_versions = model_versions[valid_mask]
    load_modes = load_modes[valid_mask]
    mode_learning_ready = mode_learning_ready[valid_mask]
    verification_eligible = verification_eligible[valid_mask]
    independent_cycles = independent_cycles[valid_mask]
    forecast_contracts = forecast_contracts[valid_mask]
    forecast_identity_ids = forecast_identity_ids[valid_mask]
    forecast_system_versions = forecast_system_versions[valid_mask]
    feature_set_versions = feature_set_versions[valid_mask]
    feature_set_digests = feature_set_digests[valid_mask]
    forecast_code_revisions = forecast_code_revisions[valid_mask]
    source_cycle_set_ids = source_cycle_set_ids[valid_mask]
    degraded_mode_codes = degraded_mode_codes[valid_mask]
    candidate_lanes = candidate_lanes[valid_mask]
    local_feature_contract_ids = local_feature_contract_ids[valid_mask]
    baseline_control_contract_ids = baseline_control_contract_ids[valid_mask]
    baseline_control_system_versions = baseline_control_system_versions[valid_mask]
    solar_model_names = solar_model_names[valid_mask]
    solar_power_semantics = solar_power_semantics[valid_mask]
    lead_hours = lead_hours[valid_mask]
    observed_valid = observed.reindex(valid_times, method="nearest", tolerance=tolerance).to_numpy(dtype=np.float64)
    reference_values = observed.reindex(issue_times, method="nearest", tolerance=tolerance).to_numpy(dtype=np.float64)
    paired = np.isfinite(observed_valid) & np.isfinite(reference_values)
    if not np.any(paired):
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "issue_time": issue_times[paired],
            "cycle_time": cycle_times[paired],
            "load_model_version": model_versions[paired],
            "load_mode": load_modes[paired],
            "load_mode_learning_ready": mode_learning_ready[paired],
            "forecast_verification_eligible": verification_eligible[paired],
            "independent_cycle": independent_cycles[paired],
            "forecast_model_contract_id": forecast_contracts[paired],
            "forecast_identity_id": forecast_identity_ids[paired],
            "forecast_system_version": forecast_system_versions[paired],
            "feature_set_version": feature_set_versions[paired],
            "feature_set_digest": feature_set_digests[paired],
            "forecast_code_revision": forecast_code_revisions[paired],
            "source_cycle_set_id": source_cycle_set_ids[paired],
            "degraded_mode_code": degraded_mode_codes[paired],
            "candidate_lane": candidate_lanes[paired],
            "local_feature_contract_id": local_feature_contract_ids[paired],
            "baseline_control_contract_id": baseline_control_contract_ids[paired],
            "baseline_control_system_version": baseline_control_system_versions[paired],
            "solar_model_name": solar_model_names[paired],
            "solar_power_semantics": solar_power_semantics[paired],
            "valid_time": valid_times[paired],
            "lead_hour": lead_hours[paired],
            "forecast_value": forecast_values[paired],
            "observed_value": observed_valid[paired],
            "reference_value": reference_values[paired],
            "error": forecast_values[paired] - observed_valid[paired],
            "reference_error": reference_values[paired] - observed_valid[paired],
        }
    ).sort_values("valid_time")


def _independent_verification_rows(table: pd.DataFrame) -> pd.DataFrame:
    """Keep one forecast per ECMWF cycle and valid time.

    Cached learning runs re-anchor the same ECMWF cycle every 15 minutes. They
    are operationally useful but are not independent evidence for skill.
    """
    if table.empty:
        return table
    if "forecast_verification_eligible" in table:
        table = table.loc[table["forecast_verification_eligible"].astype(bool)]
    if "independent_cycle" in table:
        table = table.loc[table["independent_cycle"].astype(bool)]
    if table.empty:
        return table
    if "valid_time" not in table:
        return table
    cycle_key = table["cycle_time"].astype(str)
    if "source_cycle_set_id" in table:
        source = table["source_cycle_set_id"].astype(str).str.strip()
        usable = ~source.str.lower().isin({"", "nan", "none"})
        cycle_key = cycle_key.where(~usable, source)
    selected = table.copy()
    selected["_independent_cycle_key"] = cycle_key
    return selected.sort_values("issue_time").drop_duplicates(
        ["_independent_cycle_key", "valid_time"], keep="last"
    )


def _verification_cycle_count(table: pd.DataFrame) -> int:
    """Count immutable source-cycle sets, falling back for legacy archives."""
    if table.empty:
        return 0
    if "_independent_cycle_key" in table:
        return int(table["_independent_cycle_key"].nunique())
    if "source_cycle_set_id" in table:
        values = table["source_cycle_set_id"].astype(str).str.strip()
        values = values.loc[~values.str.lower().isin({"", "nan", "none"})]
        if not values.empty:
            return int(values.nunique())
    return int(table["cycle_time"].nunique())


def _add_error_metrics(metrics: dict[str, float | int | str], prefix: str, errors: np.ndarray, lead_hours: np.ndarray) -> None:
    valid = np.isfinite(errors) & np.isfinite(lead_hours)
    if np.count_nonzero(valid) < 2:
        return
    errors = errors[valid]
    lead_hours = lead_hours[valid]
    metrics[f"{prefix}_mae"] = float(np.mean(np.abs(errors)))
    metrics[f"{prefix}_bias"] = float(np.mean(errors))
    metrics[f"{prefix}_sample_count"] = int(errors.size)
    for bucket, start, end in LEAD_BUCKETS:
        in_bucket = (lead_hours >= start) & (lead_hours < end)
        if np.count_nonzero(in_bucket) >= 2:
            bucket_errors = errors[in_bucket]
            metrics[_metric_bucket_name(f"{prefix}_mae", bucket)] = float(np.mean(np.abs(bucket_errors)))
            metrics[_metric_bucket_name(f"{prefix}_bias", bucket)] = float(np.mean(bucket_errors))
            metrics[_metric_bucket_name(f"{prefix}_sample_count", bucket)] = int(bucket_errors.size)


def evaluate_forecast_archive(archive: xr.Dataset | None, frame: pd.DataFrame) -> dict[str, float | int | str]:
    """Score archived forecast runs against APS observations by lead time."""
    if archive is None or frame.empty or "issue_time" not in archive.sizes or archive.sizes.get("issue_time", 0) == 0:
        return {}
    metrics: dict[str, float | int | str] = {}
    tolerance = pd.Timedelta(minutes=10)
    if "BatterySOC" in frame:
        soc_table = _archive_verification_frame(
            archive,
            frame["BatterySOC"],
            forecast_var="BatterySOCForecast",
            tolerance=tolerance,
        )
        soc_table = _filter_active_forecast_contract(soc_table, archive)
        if not soc_table.empty:
            _add_error_metrics(
                metrics,
                "soc",
                soc_table["error"].to_numpy(dtype=np.float64),
                soc_table["lead_hour"].to_numpy(dtype=np.float64),
            )
    observed_solar = _observed_solar_w(frame)
    if not observed_solar.empty:
        solar_table = _archive_verification_frame(
            archive,
            observed_solar,
            forecast_var="ForecastSolarWatts",
            tolerance=tolerance,
        )
        solar_table = _filter_active_forecast_contract(solar_table, archive)
        solar_table, excluded, accepted = _filter_solar_verification_rows(
            solar_table,
            mpp_active_mask=_mpp_active_available_power_mask(frame),
            tolerance=tolerance,
        )
        solar_status = _solar_verification_status(
            excluded=excluded,
            accepted_available_power=accepted,
        )
        if solar_status is not None:
            metrics["solar_verification_status"] = solar_status
        if not solar_table.empty:
            _add_error_metrics(
                metrics,
                "solar",
                solar_table["error"].to_numpy(dtype=np.float64),
                solar_table["lead_hour"].to_numpy(dtype=np.float64),
            )
    observed_load = _observed_load_w(frame)
    if not observed_load.empty:
        load_table = _archive_verification_frame(
            archive,
            observed_load,
            forecast_var="ForecastLoadWatts",
            tolerance=tolerance,
        )
        load_table = _filter_active_forecast_contract(load_table, archive)
        if not load_table.empty:
            load_table = load_table[
                load_table["load_model_version"] == float(LOAD_MODEL_VERSION)
            ]
        if not load_table.empty:
            _add_error_metrics(
                metrics,
                "load",
                load_table["error"].to_numpy(dtype=np.float64),
                load_table["lead_hour"].to_numpy(dtype=np.float64),
            )
    if metrics:
        aliases = {
            "soc_mae_pct_points": "soc_mae",
            "soc_bias_pct_points": "soc_bias",
            "soc_sample_count": "soc_sample_count",
            "solar_mae_w": "solar_mae",
            "solar_bias_w": "solar_bias",
            "solar_sample_count": "solar_sample_count",
            "load_mae_w": "load_mae",
            "load_bias_w": "load_bias",
            "load_sample_count": "load_sample_count",
        }
        for alias, source in aliases.items():
            if source in metrics:
                metrics[alias] = metrics[source]
        metrics["evaluated_at_utc"] = _utc_now()
    return metrics


def evaluate_independent_forecast_archive(archive: xr.Dataset | None, frame: pd.DataFrame) -> dict[str, float | int | str]:
    """Score one operational forecast per ECMWF cycle and valid time.

    The archive deliberately retains re-issued forecasts so operators can audit
    their decisions. Those re-issues share weather forcing and must not be
    treated as independent training examples for calibration.
    """
    if archive is None or frame.empty or "issue_time" not in archive.sizes or archive.sizes.get("issue_time", 0) == 0:
        return {}
    metrics: dict[str, float | int | str] = {}
    tolerance = pd.Timedelta(minutes=10)

    def score(
        name: str,
        forecast_var: str,
        observed: pd.Series,
        *,
        current_load_model: bool = False,
        filter_censored_solar: bool = False,
    ) -> None:
        if observed.empty:
            return
        table = _archive_verification_frame(
            archive,
            observed,
            forecast_var=forecast_var,
            tolerance=tolerance,
        )
        if table.empty:
            return
        table = _filter_active_forecast_contract(table, archive)
        if current_load_model:
            table = table[table["load_model_version"] == float(LOAD_MODEL_VERSION)]
        if filter_censored_solar:
            table, excluded, accepted = _filter_solar_verification_rows(
                table,
                mpp_active_mask=_mpp_active_available_power_mask(frame),
                tolerance=tolerance,
            )
            solar_status = _solar_verification_status(
                excluded=excluded,
                accepted_available_power=accepted,
            )
            if solar_status is not None:
                metrics["solar_verification_status"] = solar_status
        table = _independent_verification_rows(table)
        if table.empty:
            return
        _add_error_metrics(
            metrics,
            name,
            table["error"].to_numpy(dtype=np.float64),
            table["lead_hour"].to_numpy(dtype=np.float64),
        )
        metrics[f"{name}_independent_cycles"] = _verification_cycle_count(table)

    if "BatterySOC" in frame:
        score("soc", "BatterySOCForecast", frame["BatterySOC"])
    score(
        "solar",
        "ForecastSolarWatts",
        _observed_solar_w(frame),
        filter_censored_solar=True,
    )
    score("load", "ForecastLoadWatts", _observed_load_w(frame), current_load_model=True)
    if not metrics:
        return {}
    aliases = {
        "soc_mae_pct_points": "soc_mae",
        "soc_bias_pct_points": "soc_bias",
        "solar_mae_w": "solar_mae",
        "solar_bias_w": "solar_bias",
        "load_mae_w": "load_mae",
        "load_bias_w": "load_bias",
    }
    for alias, source in aliases.items():
        if source in metrics:
            metrics[alias] = metrics[source]
    metrics["sample_independence"] = "one forecast per ECMWF cycle and valid time"
    metrics["evaluated_at_utc"] = _utc_now()
    return metrics


def _rolling_error_stats(errors: np.ndarray, reference_errors: np.ndarray) -> tuple[float, float, float, float, float, float, int]:
    valid = np.isfinite(errors)
    if np.count_nonzero(valid) < 2:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, int(np.count_nonzero(valid)))
    errors = errors[valid]
    ref = reference_errors[valid & np.isfinite(reference_errors)] if reference_errors.shape == valid.shape else np.array([])
    mae = float(np.mean(np.abs(errors)))
    bias = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    ref_mae = float(np.mean(np.abs(ref))) if ref.size >= 2 else np.nan
    skill = float(1.0 - (mae / ref_mae)) if np.isfinite(ref_mae) and ref_mae > 0.0 else np.nan
    return (mae, bias, rmse, ref_mae, skill, float(errors.size), int(errors.size))


def _guarded_skill(mae: float, reference_mae: float, *, minimum_reference_mae: float) -> float:
    if not np.isfinite(mae) or not np.isfinite(reference_mae) or reference_mae < minimum_reference_mae:
        return np.nan
    return float(1.0 - mae / reference_mae)


def _empty_skill_dataset() -> xr.Dataset:
    empty_time = np.array([], dtype="datetime64[ns]")
    fields = [
        "ForecastVerificationSamples",
        "ForecastIndependentCycles",
        "ForecastSOCMAE_0_6h_Verified",
        "ForecastSOCMAE_6_24h_Verified",
        "ForecastSOCMAE_24_48h_Verified",
        "ForecastSOCMAE_48_96h_Verified",
        "ForecastSOCBias_0_6h_Verified",
        "ForecastSOCSkill_0_6h",
        "ForecastLoadMAE24h",
        "ForecastLoadBias24h",
        "ForecastLoadSkill24h",
        "ForecastSolarMAE24h",
        "ForecastSolarBias24h",
        "ForecastSolarSkill24h",
    ]
    for bucket, _, _ in LEAD_BUCKETS:
        fields.extend(
            (
                f"ForecastSOCBias_{bucket}_Verified",
                f"ForecastSOCMAESamples_{bucket}",
                f"ForecastSOCMAECycles_{bucket}",
                f"ForecastSOCSkill_{bucket}",
                f"ForecastSOCReadiness_{bucket}",
            )
        )
    fields.extend(
        (
            "ForecastLoadVerificationSamples",
            "ForecastLoadIndependentCycles",
            "ForecastSolarVerificationSamples",
            "ForecastSolarIndependentCycles",
        )
    )
    return xr.Dataset(
        {name: (("time",), np.array([], dtype=np.float32)) for name in fields},
        coords={"time": empty_time},
        attrs={
            "power_soc_forecast_skill_product": "true",
            "source": "verification of archived APS SOC forecasts against observed APS power history",
            "generated_at_utc": _utc_now(),
        }
    )


def build_forecast_skill_dataset(
    archive: xr.Dataset | None,
    power: xr.Dataset,
    *,
    window_hours: float = DEFAULT_SKILL_WINDOW_HOURS,
    retention_days: float = DEFAULT_SKILL_RETENTION_DAYS,
    freq: str = "1h",
) -> xr.Dataset:
    """Build a past-facing verification product from archived forecasts and observations."""
    frame = _power_frame(power)
    if archive is None or frame.empty or "issue_time" not in archive.sizes or archive.sizes.get("issue_time", 0) == 0:
        return _empty_skill_dataset()

    tolerance = pd.Timedelta(minutes=10)
    pieces: dict[str, pd.DataFrame] = {}
    excluded_solar_rows = 0
    accepted_available_solar_rows = 0
    if "BatterySOC" in frame:
        pieces["soc"] = _filter_active_forecast_contract(_archive_verification_frame(
            archive,
            frame["BatterySOC"],
            forecast_var="BatterySOCForecast",
            tolerance=tolerance,
        ), archive)
    observed_solar = _observed_solar_w(frame)
    if not observed_solar.empty:
        solar_table = _filter_active_forecast_contract(_archive_verification_frame(
            archive,
            observed_solar,
            forecast_var="ForecastSolarWatts",
            tolerance=tolerance,
        ), archive)
        solar_table, excluded_solar_rows, accepted_available_solar_rows = _filter_solar_verification_rows(
            solar_table,
            mpp_active_mask=_mpp_active_available_power_mask(frame),
            tolerance=tolerance,
        )
        pieces["solar"] = solar_table
    observed_load = _observed_load_w(frame)
    if not observed_load.empty:
        load_table = _archive_verification_frame(
            archive,
            observed_load,
            forecast_var="ForecastLoadWatts",
            tolerance=tolerance,
        )
        load_table = _filter_active_forecast_contract(load_table, archive)
        if not load_table.empty:
            load_table = load_table[load_table["load_model_version"] == float(LOAD_MODEL_VERSION)]
        pieces["load"] = load_table
    pieces = {name: table for name, table in pieces.items() if not table.empty}
    if not pieces:
        return _empty_skill_dataset()

    observed_end = pd.Timestamp(frame.index.max())
    start = observed_end - pd.Timedelta(days=float(retention_days))
    time_index = pd.date_range(start.floor(freq), observed_end.ceil(freq), freq=freq)
    if len(time_index) == 0:
        return _empty_skill_dataset()

    columns: dict[str, np.ndarray] = {}
    window = pd.Timedelta(hours=float(window_hours))
    metric_names = [
        "ForecastVerificationSamples",
        "ForecastIndependentCycles",
        "ForecastSOCMAE_0_6h_Verified",
        "ForecastSOCMAE_6_24h_Verified",
        "ForecastSOCMAE_24_48h_Verified",
        "ForecastSOCMAE_48_96h_Verified",
        "ForecastSOCBias_0_6h_Verified",
        "ForecastSOCSkill_0_6h",
        "ForecastLoadMAE24h",
        "ForecastLoadBias24h",
        "ForecastLoadSkill24h",
        "ForecastSolarMAE24h",
        "ForecastSolarBias24h",
        "ForecastSolarSkill24h",
    ]
    for bucket, _, _ in LEAD_BUCKETS:
        metric_names.extend(
            (
                f"ForecastSOCBias_{bucket}_Verified",
                f"ForecastSOCMAESamples_{bucket}",
                f"ForecastSOCMAECycles_{bucket}",
                f"ForecastSOCSkill_{bucket}",
                f"ForecastSOCReadiness_{bucket}",
            )
        )
    metric_names.extend(
        (
            "ForecastLoadVerificationSamples",
            "ForecastLoadIndependentCycles",
            "ForecastSolarVerificationSamples",
            "ForecastSolarIndependentCycles",
        )
    )
    for metric_name in metric_names:
        columns[metric_name] = np.full(len(time_index), np.nan, dtype=np.float32)

    soc_mode_table = pieces.get("soc", pd.DataFrame())
    mode_labels: list[str] = []
    if not soc_mode_table.empty and "load_mode" in soc_mode_table:
        learned_mask = (
            soc_mode_table["load_mode_learning_ready"].astype(bool)
            if "load_mode_learning_ready" in soc_mode_table
            else pd.Series(False, index=soc_mode_table.index)
        )
        mode_labels = sorted(
            {
                str(value)
                for value in soc_mode_table.loc[learned_mask, "load_mode"]
                if str(value).strip() and str(value).lower() not in {"unknown", "nan"}
            }
        )
    mode_shape = (len(time_index), len(mode_labels), len(LEAD_BUCKETS))
    mode_mae = np.full(mode_shape, np.nan, dtype=np.float32)
    mode_bias = np.full(mode_shape, np.nan, dtype=np.float32)
    mode_samples = np.full(mode_shape, np.nan, dtype=np.float32)
    mode_cycles = np.full(mode_shape, np.nan, dtype=np.float32)
    mode_ready = np.full(mode_shape, np.nan, dtype=np.float32)

    for idx, now in enumerate(time_index):
        window_start = now - window
        soc = pieces.get("soc")
        if soc is not None:
            selected = _independent_verification_rows(
                soc[(soc["valid_time"] > window_start) & (soc["valid_time"] <= now)]
            )
            columns["ForecastVerificationSamples"][idx] = float(len(selected))
            columns["ForecastIndependentCycles"][idx] = float(
                _verification_cycle_count(selected)
            )
            for bucket, start_hour, end_hour in LEAD_BUCKETS:
                bucketed = selected[(selected["lead_hour"] >= start_hour) & (selected["lead_hour"] < end_hour)]
                mae, bias, _rmse, ref_mae, _skill, _sample_float, sample_count = _rolling_error_stats(
                    bucketed["error"].to_numpy(dtype=np.float64),
                    bucketed["reference_error"].to_numpy(dtype=np.float64),
                )
                if sample_count >= 2:
                    columns[f"ForecastSOCMAE_{bucket}_Verified"][idx] = mae
                    columns[f"ForecastSOCBias_{bucket}_Verified"][idx] = bias
                    columns[f"ForecastSOCMAESamples_{bucket}"][idx] = float(sample_count)
                    columns[f"ForecastSOCMAECycles_{bucket}"][idx] = float(
                        _verification_cycle_count(bucketed)
                    )
                    columns[f"ForecastSOCSkill_{bucket}"][idx] = _guarded_skill(
                        mae, ref_mae, minimum_reference_mae=0.5
                    )
                    columns[f"ForecastSOCReadiness_{bucket}"][idx] = float(
                        mae <= FORECAST_TARGET_MAE_PCT_POINTS
                        and abs(bias) <= FORECAST_TARGET_MAE_PCT_POINTS
                        and sample_count >= FORECAST_TARGET_MIN_SAMPLES
                        and _verification_cycle_count(bucketed)
                        >= FORECAST_TARGET_MIN_CYCLES
                    )
                    if bucket == "0_6h":
                        columns["ForecastSOCBias_0_6h_Verified"][idx] = bias
                        columns["ForecastSOCSkill_0_6h"][idx] = columns[f"ForecastSOCSkill_{bucket}"][idx]
                for mode_index, mode in enumerate(mode_labels):
                    mode_rows = bucketed[
                        (bucketed["load_mode"] == mode)
                        & bucketed["load_mode_learning_ready"].astype(bool)
                    ]
                    mode_mae_value, mode_bias_value, _rmse, _reference, _skill, _count_float, mode_count = _rolling_error_stats(
                        mode_rows["error"].to_numpy(dtype=np.float64),
                        mode_rows["reference_error"].to_numpy(dtype=np.float64),
                    )
                    if mode_count < 2:
                        continue
                    cycles = _verification_cycle_count(mode_rows)
                    mode_mae[idx, mode_index, LEAD_BUCKETS.index((bucket, start_hour, end_hour))] = mode_mae_value
                    mode_bias[idx, mode_index, LEAD_BUCKETS.index((bucket, start_hour, end_hour))] = mode_bias_value
                    mode_samples[idx, mode_index, LEAD_BUCKETS.index((bucket, start_hour, end_hour))] = float(mode_count)
                    mode_cycles[idx, mode_index, LEAD_BUCKETS.index((bucket, start_hour, end_hour))] = float(cycles)
                    mode_ready[idx, mode_index, LEAD_BUCKETS.index((bucket, start_hour, end_hour))] = float(
                        mode_mae_value <= FORECAST_TARGET_MAE_PCT_POINTS
                        and abs(mode_bias_value) <= FORECAST_TARGET_MAE_PCT_POINTS
                        and mode_count >= FORECAST_TARGET_MIN_SAMPLES
                        and cycles >= FORECAST_TARGET_MIN_CYCLES
                    )
        load = pieces.get("load")
        if load is not None:
            selected = _independent_verification_rows(
                load[(load["valid_time"] > window_start) & (load["valid_time"] <= now)]
            )
            mae, bias, _rmse, ref_mae, _skill, _sample_float, sample_count = _rolling_error_stats(
                selected["error"].to_numpy(dtype=np.float64),
                selected["reference_error"].to_numpy(dtype=np.float64),
            )
            if sample_count >= 2:
                columns["ForecastLoadMAE24h"][idx] = mae
                columns["ForecastLoadBias24h"][idx] = bias
                columns["ForecastLoadVerificationSamples"][idx] = float(sample_count)
                columns["ForecastLoadIndependentCycles"][idx] = float(
                    _verification_cycle_count(selected)
                )
                columns["ForecastLoadSkill24h"][idx] = _guarded_skill(
                    mae, ref_mae, minimum_reference_mae=5.0
                )
        solar = pieces.get("solar")
        if solar is not None:
            selected = _independent_verification_rows(
                solar[(solar["valid_time"] > window_start) & (solar["valid_time"] <= now)]
            )
            mae, bias, _rmse, ref_mae, _skill, _sample_float, sample_count = _rolling_error_stats(
                selected["error"].to_numpy(dtype=np.float64),
                selected["reference_error"].to_numpy(dtype=np.float64),
            )
            if sample_count >= 2:
                columns["ForecastSolarMAE24h"][idx] = mae
                columns["ForecastSolarBias24h"][idx] = bias
                columns["ForecastSolarVerificationSamples"][idx] = float(sample_count)
                columns["ForecastSolarIndependentCycles"][idx] = float(
                    _verification_cycle_count(selected)
                )
                columns["ForecastSolarSkill24h"][idx] = _guarded_skill(
                    mae, ref_mae, minimum_reference_mae=5.0
                )

    out = xr.Dataset(
        {name: (("time",), values) for name, values in columns.items()},
        coords={"time": time_index.to_numpy(dtype="datetime64[ns]")},
        attrs={
            "power_soc_forecast_skill_product": "true",
            "source": "verification of archived APS SOC forecasts against observed APS power history",
            "generated_at_utc": _utc_now(),
            "verification_window_hours": str(float(window_hours)),
            "retention_days": str(float(retention_days)),
            "reference_model": "persistence from observed value at forecast issue time",
            "skill_score": "1 - forecast_mae / persistence_mae",
            "description": "Past-facing forecast verification history; future forecast curves remain in power_soc_forecast.zarr.",
            "sample_independence": "one forecast per ECMWF cycle and valid time",
            "load_model": LOAD_MODEL_NAME,
            "load_model_version": str(LOAD_MODEL_VERSION),
            "target_soc_mae_pct_points": str(float(FORECAST_TARGET_MAE_PCT_POINTS)),
            "target_soc_abs_bias_pct_points": str(float(FORECAST_TARGET_MAE_PCT_POINTS)),
            "target_minimum_independent_cycles": str(int(FORECAST_TARGET_MIN_CYCLES)),
            "target_minimum_verified_samples": str(int(FORECAST_TARGET_MIN_SAMPLES)),
            "solar_verification_status": (
                _solar_verification_status(
                    excluded=excluded_solar_rows,
                    accepted_available_power=accepted_available_solar_rows,
                )
                or "eligible_delivered_power_baseline"
            ),
            "solar_verification_exclusion_reason": (
                "Observed charger output was limited or MPP register 791 was unavailable; it cannot verify available PV"
                if excluded_solar_rows
                else ""
            ),
            "forecast_model_contract_id": str(_active_forecast_contract_id(archive) or "legacy_or_single_contract"),
        },
    )
    if mode_labels:
        out = out.assign_coords(
            load_mode=np.asarray(mode_labels, dtype="U96"),
            lead_bucket=np.asarray([bucket for bucket, _, _ in LEAD_BUCKETS], dtype="U16"),
        )
        out["ForecastSOCModeMAE"] = (("time", "load_mode", "lead_bucket"), mode_mae)
        out["ForecastSOCModeBias"] = (("time", "load_mode", "lead_bucket"), mode_bias)
        out["ForecastSOCModeSamples"] = (("time", "load_mode", "lead_bucket"), mode_samples)
        out["ForecastSOCModeCycles"] = (("time", "load_mode", "lead_bucket"), mode_cycles)
        out["ForecastSOCModeReadiness"] = (("time", "load_mode", "lead_bucket"), mode_ready)
    for name in out.data_vars:
        if name.endswith("Samples"):
            out[name].attrs["units"] = "samples"
        elif name.endswith("Cycles"):
            out[name].attrs["units"] = "cycles"
        elif "Skill" in name:
            out[name].attrs["units"] = "1"
        elif name.startswith("ForecastSOCReadiness_"):
            out[name].attrs["units"] = "1"
        elif name == "ForecastSOCModeReadiness":
            out[name].attrs["units"] = "1"
        elif name.endswith("Samples"):
            out[name].attrs["units"] = "samples"
        elif name.endswith("Cycles"):
            out[name].attrs["units"] = "cycles"
        elif name in {"ForecastSOCModeMAE", "ForecastSOCModeBias"}:
            out[name].attrs["units"] = "percentage points"
        elif "SOC" in name:
            out[name].attrs["units"] = "percentage points"
        else:
            out[name].attrs["units"] = "W"
    return out


def _archive_row_from_forecast(forecast: xr.Dataset) -> xr.Dataset:
    issue_time = pd.Timestamp(forecast.attrs.get("initial_soc_time", forecast.attrs.get("generated_at_utc", _utc_now())))
    if issue_time.tz is not None:
        issue_time = issue_time.tz_convert("UTC").tz_localize(None)
    times = pd.DatetimeIndex(forecast["time"].values)
    lead_hours = ((times - issue_time) / pd.Timedelta(hours=1)).to_numpy(dtype=np.float32)
    step = np.arange(len(times), dtype=np.int32)
    data_vars: dict[str, tuple[tuple[str, str], np.ndarray]] = {
        "ForecastValidTime": (("issue_time", "forecast_step"), times.to_numpy(dtype="datetime64[ns]")[None, :]),
        "ForecastLeadHours": (("issue_time", "forecast_step"), lead_hours[None, :]),
    }
    cycle_time = pd.Timestamp(forecast.attrs.get("ecmwf_cycle_time", issue_time))
    if cycle_time.tz is not None:
        cycle_time = cycle_time.tz_convert("UTC").tz_localize(None)
    data_vars["ECMWFCycleTime"] = (("issue_time",), np.array([cycle_time.to_datetime64()], dtype="datetime64[ns]"))
    data_vars["LoadModelVersion"] = (
        ("issue_time",),
        np.array([float(forecast.attrs.get("load_model_version", LOAD_MODEL_VERSION))], dtype=np.float32),
    )
    # Keep the detected state with the issue record. Older archive rows simply
    # have no mode label and are excluded from future per-mode certification.
    data_vars["LoadMode"] = (("issue_time",), np.array([str(forecast.attrs.get("load_mode", "unknown"))], dtype="U96"))
    data_vars["LoadModeLearningReady"] = (
        ("issue_time",),
        np.array([forecast.attrs.get("load_mode_learning_ready", "false") == "true"], dtype=bool),
    )
    data_vars["ForecastRefreshKind"] = (
        ("issue_time",),
        np.asarray([str(forecast.attrs.get("forecast_refresh_kind", "unknown"))], dtype="U32"),
    )
    default_independent = str(
        forecast.attrs.get("forecast_refresh_kind", "ecmwf_cycle")
    ) != "cached_reanchor"
    data_vars["ForecastVerificationEligible"] = (
        ("issue_time",),
        np.asarray(
            [
                str(
                    forecast.attrs.get(
                        "forecast_verification_eligible", str(default_independent)
                    )
                ).lower()
                == "true"
            ],
            dtype=bool,
        ),
    )
    data_vars["IndependentCycle"] = (
        ("issue_time",),
        np.asarray(
            [
                str(
                    forecast.attrs.get("independent_cycle", str(default_independent))
                ).lower()
                == "true"
            ],
            dtype=bool,
        ),
    )
    provenance_fields = {
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
        "SOCAuthoringAnchorPct": "initial_soc_pct",
        "AdaptiveCalibrationStateID": "adaptive_calibration_state_id",
        "SolarModelName": "solar_model_name",
        "SolarModelVersion": "solar_model_version",
        "SolarModelContractID": "solar_model_contract_id",
        "SolarFeatureSetVersion": "solar_feature_set_version",
        "SolarModelStatus": "solar_model_status",
        "SolarDegradationCodes": "solar_degradation_codes",
        "SolarPowerSemantics": "solar_power_semantics",
        "SolarPhysicalConfigSHA256": "solar_physical_config_sha256",
        "EvaluationPairID": "evaluation_pair_id",
        "InputSnapshotID": "input_snapshot_id",
        "BaselinePublicationSignature": "baseline_publication_signature",
    }
    for archive_name, attr_name in provenance_fields.items():
        value = str(forecast.attrs.get(attr_name, ""))
        data_vars[archive_name] = (("issue_time",), np.asarray([value], dtype="U512"))
    for name in ARCHIVE_FORECAST_FIELDS:
        if name in forecast:
            data_vars[name] = (("issue_time", "forecast_step"), np.asarray(forecast[name].values, dtype=np.float32)[None, :])
    return xr.Dataset(
        data_vars,
        coords={"issue_time": np.array([issue_time.to_datetime64()], dtype="datetime64[ns]"), "forecast_step": step},
        attrs={
            "power_soc_forecast_archive": "true",
            "source": "archived rows from latest ECMWF-informed APS SOC forecasts",
        },
    )


def build_soc_hindcast_dataset(
    archive: xr.Dataset | None,
    power: xr.Dataset,
    *,
    retention_days: float = DEFAULT_HINDCAST_RETENTION_DAYS,
    lead_hours: tuple[int, ...] = HINDCAST_LEAD_HOURS,
    lead_tolerance_hours: float = 1.5,
) -> xr.Dataset:
    """Build observed SOC plus fixed-lead archived forecasts for the dashboard."""
    frame = _power_frame(power)
    if archive is None or frame.empty or "BatterySOC" not in frame:
        return xr.Dataset(coords={"time": np.array([], dtype="datetime64[ns]")})
    observed_end = pd.Timestamp(frame.index.max())
    observed_start = observed_end - pd.Timedelta(days=float(retention_days))
    observed = frame.loc[frame.index >= observed_start, "BatterySOC"].resample("15min").last().dropna()
    records = _archive_verification_frame(
        archive,
        frame["BatterySOC"],
        forecast_var="BatterySOCForecast",
        tolerance=pd.Timedelta(minutes=10),
    )
    records = _filter_active_forecast_contract(records, archive)
    records = _independent_verification_rows(records)
    series: list[pd.Series] = [observed.rename("BatterySOCObservedHindcast")]
    if records.empty:
        records = pd.DataFrame(columns=("valid_time", "lead_hour", "issue_time", "forecast_value"))
    for target in lead_hours:
        selected = records[
            (records["valid_time"] >= observed_start)
            & ((records["lead_hour"] - float(target)).abs() <= float(lead_tolerance_hours))
        ].copy()
        if selected.empty:
            continue
        selected["lead_delta"] = (selected["lead_hour"] - float(target)).abs()
        selected = selected.sort_values(["valid_time", "lead_delta", "issue_time"])
        selected = selected.drop_duplicates("valid_time", keep="first")
        values = pd.Series(
            selected["forecast_value"].to_numpy(dtype=np.float64),
            index=pd.DatetimeIndex(selected["valid_time"]),
            name=f"BatterySOCHindcast_{int(target)}h",
        )
        series.append(values)
    merged = pd.concat(series, axis=1).sort_index().dropna(how="all")
    out = xr.Dataset(
        {name: (("time",), merged[name].to_numpy(dtype=np.float32)) for name in merged.columns},
        coords={"time": merged.index.to_numpy(dtype="datetime64[ns]")},
        attrs={
            "power_soc_hindcast_product": "true",
            "generated_at_utc": _utc_now(),
            "retention_days": str(float(retention_days)),
            "lead_tolerance_hours": str(float(lead_tolerance_hours)),
            "source": "archived operational SOC forecasts matched to later APS observations",
            "forecast_model_contract_id": str(
                _active_forecast_contract_id(archive) or "legacy_or_single_contract"
            ),
        },
    )
    for name in out.data_vars:
        out[name].attrs["units"] = "%"
    return out


def append_forecast_archive(
    forecast: xr.Dataset,
    archive_zarr: Path,
    *,
    retention_days: float = DEFAULT_ARCHIVE_RETENTION_DAYS,
) -> xr.Dataset:
    """Append a latest forecast run to the forecast-run archive."""
    row = _archive_row_from_forecast(forecast)
    archive = None
    if archive_zarr.exists():
        try:
            archive = xr.open_zarr(archive_zarr, chunks={}).load()
        except Exception:
            archive = None
    if archive is not None and archive.sizes.get("issue_time", 0):
        max_steps = max(int(archive.sizes.get("forecast_step", 0)), int(row.sizes.get("forecast_step", 0)))
        steps = np.arange(max_steps, dtype=np.int32)
        archive = archive.reindex(forecast_step=steps)
        row = row.reindex(forecast_step=steps)
        combined = xr.concat([archive, row], dim="issue_time")
        combined = combined.sortby("issue_time")
        combined = combined.isel(issue_time=~combined.indexes["issue_time"].duplicated(keep="last"))
    else:
        combined = row
    issue_times = pd.DatetimeIndex(combined["issue_time"].values)
    # Retention is relative to the newest archived issue, not wall-clock time.
    # This preserves delayed field-data backfills and keeps historical fixtures
    # usable while still bounding the archive to the requested rolling span.
    cutoff = issue_times.max() - pd.Timedelta(days=float(retention_days))
    combined = combined.isel(issue_time=np.asarray(issue_times >= cutoff))
    _atomic_write_archive(combined, archive_zarr)
    return combined


def _adaptive_value(raw_value: float, state_value: object, *, alpha: float = DEFAULT_ADAPTIVE_ALPHA) -> float:
    try:
        previous = float(state_value)
    except Exception:
        return float(raw_value)
    if not np.isfinite(previous):
        return float(raw_value)
    return float((1.0 - alpha) * previous + alpha * raw_value)


def _load_bias_correction(previous_correction: object, previous_bias: object, *, alpha: float = DEFAULT_ADAPTIVE_ALPHA) -> float:
    try:
        correction = float(previous_correction)
    except Exception:
        correction = 0.0
    if not np.isfinite(correction):
        correction = 0.0
    try:
        bias = float(previous_bias)
    except Exception:
        return float(correction)
    if not np.isfinite(bias):
        return float(correction)
    # Positive bias means the previous forecast load was too high, so reduce the
    # next load profile; negative bias means it was too low.
    updated = correction - float(alpha) * bias
    return float(np.clip(updated, -DEFAULT_LOAD_BIAS_CORRECTION_LIMIT_W, DEFAULT_LOAD_BIAS_CORRECTION_LIMIT_W))


def _bounded_load_profile(raw_load_profile: pd.Series, correction_w: float) -> tuple[pd.Series, float]:
    """Apply adaptive load bias without allowing it to erase a load profile."""
    if raw_load_profile.empty:
        return raw_load_profile, float(correction_w)
    raw_values = raw_load_profile.to_numpy(dtype=np.float64)
    finite_raw = raw_values[np.isfinite(raw_values)]
    if finite_raw.size == 0:
        return raw_load_profile.clip(lower=0.0), float(correction_w)
    fraction_limit = float(np.clip(DEFAULT_LOAD_NEGATIVE_CORRECTION_FRACTION_LIMIT, 0.0, 0.95))
    raw_median = float(np.nanmedian(finite_raw))
    bounded_correction = float(correction_w)
    if bounded_correction < 0.0 and raw_median > 0.0:
        bounded_correction = max(bounded_correction, -fraction_limit * raw_median)
    adjusted = (raw_load_profile + bounded_correction).clip(lower=0.0)
    if fraction_limit > 0.0:
        floor = raw_load_profile.clip(lower=0.0) * (1.0 - fraction_limit)
        adjusted = adjusted.where(adjusted >= floor, floor)
    return adjusted, bounded_correction


def _soc_bias_corrections(
    previous_corrections: object,
    metrics: dict[str, float | int | str],
    *,
    alpha: float,
) -> dict[str, float]:
    corrections: dict[str, float] = {}
    if isinstance(previous_corrections, dict):
        for key, value in previous_corrections.items():
            try:
                parsed = float(value)
            except Exception:
                continue
            if np.isfinite(parsed):
                corrections[str(key)] = parsed
    for bucket, _, _ in LEAD_BUCKETS:
        bias_key = _metric_bucket_name("soc_bias", bucket)
        try:
            bias = float(metrics.get(bias_key, np.nan))
        except Exception:
            continue
        if not np.isfinite(bias):
            continue
        previous = float(corrections.get(bucket, 0.0))
        # Positive SOC bias means the forecast was too high; subtract it from
        # the next forecast for that lead bucket.
        updated = previous - float(alpha) * bias
        corrections[bucket] = float(np.clip(updated, -DEFAULT_SOC_BIAS_CORRECTION_LIMIT, DEFAULT_SOC_BIAS_CORRECTION_LIMIT))
    return corrections


def _apply_soc_bias_corrections(
    forecast: pd.DataFrame,
    corrections: dict[str, float],
    *,
    issue_time: pd.Timestamp,
    parasitic_load_w: float = 0.0,
) -> pd.DataFrame:
    """Apply a continuous residual by attenuating physical SOC increments only.

    Historical code added one constant offset inside each lead bucket.  That
    created artificial jumps at 6, 24 and 48 hours, including SOC increases at
    night while the battery was supplying load.  Here each bucket value is a
    continuous target, but it may only reduce the magnitude of an independently
    integrated physical increment.  It cannot reverse energy direction, create
    a change during zero flow, or exceed the integrated increment magnitude.
    """
    if "BatterySOCForecast" not in forecast:
        return forecast
    out = forecast.copy()
    lead_hours = (pd.DatetimeIndex(out.index) - pd.Timestamp(issue_time)) / pd.Timedelta(hours=1)
    raw_soc = out["BatterySOCForecast"].to_numpy(dtype=np.float64).copy()
    if not len(raw_soc):
        return out
    knot_hours = [0.0]
    knot_values = [0.0]
    carried = 0.0
    for bucket, _start, end in LEAD_BUCKETS:
        try:
            value = float(corrections.get(bucket, carried))
        except (TypeError, ValueError):
            value = carried
        if not np.isfinite(value):
            value = carried
        carried = float(np.clip(value, -DEFAULT_SOC_BIAS_CORRECTION_LIMIT, DEFAULT_SOC_BIAS_CORRECTION_LIMIT))
        knot_hours.append(float(end))
        knot_values.append(carried)
    correction_curve = np.interp(
        np.asarray(lead_hours, dtype=np.float64),
        np.asarray(knot_hours, dtype=np.float64),
        np.asarray(knot_values, dtype=np.float64),
        left=0.0,
        right=carried,
    )
    tentative = np.clip(raw_soc + correction_curve, 0.0, 100.0)
    soc = np.empty_like(raw_soc)
    soc[0] = raw_soc[0]
    solar = (
        out["ForecastSolarWatts"].to_numpy(dtype=np.float64)
        if "ForecastSolarWatts" in out
        else np.full(len(out), np.nan)
    )
    load = (
        out["ForecastLoadWatts"].to_numpy(dtype=np.float64)
        if "ForecastLoadWatts" in out
        else np.full(len(out), np.nan)
    )
    charge = (
        out["ForecastBatteryChargeInputWatts"].to_numpy(dtype=np.float64)
        if "ForecastBatteryChargeInputWatts" in out
        else np.full(len(out), np.nan)
    )
    discharge = (
        out["ForecastBatteryDischargeOutputWatts"].to_numpy(dtype=np.float64)
        if "ForecastBatteryDischargeOutputWatts" in out
        else np.full(len(out), np.nan)
    )
    for index in range(1, len(soc)):
        direction = _battery_flow_direction(
            charge[index],
            discharge[index],
            solar[index],
            load[index],
            parasitic_load_w=parasitic_load_w,
        )
        raw_delta = float(raw_soc[index] - raw_soc[index - 1])
        if direction is None:
            direction = 1 if raw_delta > 1.0e-9 else -1 if raw_delta < -1.0e-9 else 0
        desired_delta = float(tentative[index] - soc[index - 1])
        if direction > 0 and raw_delta >= -1.0e-9:
            corrected_delta = float(np.clip(desired_delta, 0.0, max(raw_delta, 0.0)))
            soc[index] = soc[index - 1] + corrected_delta
        elif direction < 0 and raw_delta <= 1.0e-9:
            corrected_delta = float(np.clip(desired_delta, min(raw_delta, 0.0), 0.0))
            soc[index] = soc[index - 1] + corrected_delta
        elif direction == 0 and abs(raw_delta) <= 1.0e-9:
            soc[index] = soc[index - 1]
        else:
            # Preserve a contradictory integrated interval unchanged so the
            # validator below fails closed rather than concealing bad inputs.
            soc[index] = raw_soc[index]
        soc[index] = float(np.clip(soc[index], 0.0, 100.0))
    out["BatterySOCForecast"] = soc
    out["ForecastSOCBiasCorrectionPctPoints"] = soc - raw_soc
    return out


def _battery_flow_direction(
    charge: float,
    discharge: float,
    solar: float,
    load: float,
    *,
    parasitic_load_w: float = 0.0,
    tolerance: float = 1.0e-6,
) -> int | None:
    """Return +1 charge, -1 discharge, 0 balanced, or None if unknown."""

    if np.isfinite(charge) and np.isfinite(discharge):
        net = float(charge - discharge)
    elif np.isfinite(solar) and np.isfinite(load):
        parasitic = float(parasitic_load_w) if np.isfinite(parasitic_load_w) else 0.0
        net = float(solar - load - max(parasitic, 0.0))
    else:
        return None
    if net > tolerance:
        return 1
    if net < -tolerance:
        return -1
    return 0


def validate_soc_physical_consistency(
    forecast: pd.DataFrame,
    *,
    parasitic_load_w: float = 0.0,
) -> None:
    """Fail closed when SOC direction contradicts interval-average energy flow."""

    if "BatterySOCForecast" not in forecast or len(forecast) < 2:
        return
    soc = forecast["BatterySOCForecast"].to_numpy(dtype=np.float64)
    if not np.isfinite(soc).all() or np.any(soc < -1.0e-6) or np.any(soc > 100.0 + 1.0e-6):
        raise ValueError("SOC forecast is non-finite or outside physical bounds")
    solar = (
        forecast["ForecastSolarWatts"].to_numpy(dtype=np.float64)
        if "ForecastSolarWatts" in forecast
        else np.full(len(forecast), np.nan)
    )
    load = (
        forecast["ForecastLoadWatts"].to_numpy(dtype=np.float64)
        if "ForecastLoadWatts" in forecast
        else np.full(len(forecast), np.nan)
    )
    charge = (
        forecast["ForecastBatteryChargeInputWatts"].to_numpy(dtype=np.float64)
        if "ForecastBatteryChargeInputWatts" in forecast
        else np.full(len(forecast), np.nan)
    )
    discharge = (
        forecast["ForecastBatteryDischargeOutputWatts"].to_numpy(dtype=np.float64)
        if "ForecastBatteryDischargeOutputWatts" in forecast
        else np.full(len(forecast), np.nan)
    )
    deltas = np.diff(soc)
    directions = np.asarray(
        [
            _battery_flow_direction(
                charge[index],
                discharge[index],
                solar[index],
                load[index],
                parasitic_load_w=parasitic_load_w,
            )
            for index in range(1, len(soc))
        ],
        dtype=object,
    )
    known = np.asarray([value is not None for value in directions], dtype=bool)
    numeric_direction = np.asarray(
        [0 if value is None else int(value) for value in directions], dtype=np.int8
    )
    invalid_rise = known & (deltas > 1.0e-6) & (numeric_direction <= 0)
    invalid_fall = known & (deltas < -1.0e-6) & (numeric_direction >= 0)
    if np.any(invalid_rise):
        raise ValueError("SOC forecast rises without a net-charging interval")
    if np.any(invalid_fall):
        raise ValueError("SOC forecast falls without a net-discharging interval")
    if "ForecastSOCBiasCorrectionPctPoints" in forecast:
        correction = forecast["ForecastSOCBiasCorrectionPctPoints"].to_numpy(
            dtype=np.float64
        )
        physical_soc = soc - correction
        physical_deltas = np.diff(physical_soc)
        if np.any(np.abs(deltas) > np.abs(physical_deltas) + 1.0e-6):
            raise ValueError("SOC residual exceeds the physically integrated increment")


def calibrate_solar_factor(
    frame: pd.DataFrame,
    irradiance: pd.Series,
    *,
    end: pd.Timestamp,
    calibration_days: float = DEFAULT_CALIBRATION_DAYS,
    fallback_hours: float = DEFAULT_FALLBACK_CALIBRATION_HOURS,
) -> float:
    """Estimate APS solar watts per ECMWF W/m2 from recent observations."""
    solar_fields = [name for name in ("SolarWatts_East", "SolarWatts_South", "SolarWatts_West") if name in frame]
    if not solar_fields or irradiance.empty:
        return float(DEFAULT_SOLAR_CALIBRATION_FACTOR)

    for start in (end - pd.Timedelta(days=float(calibration_days)), end - pd.Timedelta(hours=float(fallback_hours))):
        observed = frame.loc[frame.index >= start, solar_fields].sum(axis=1, min_count=1).clip(lower=0.0)
        if observed.empty:
            continue
        model = irradiance.reindex(observed.index, method="nearest", tolerance=pd.Timedelta(hours=2))
        valid = np.isfinite(observed.to_numpy(dtype=np.float64)) & np.isfinite(model.to_numpy(dtype=np.float64)) & (model.to_numpy(dtype=np.float64) > 20.0)
        if np.count_nonzero(valid) >= 6:
            ratios = observed.to_numpy(dtype=np.float64)[valid] / model.to_numpy(dtype=np.float64)[valid]
            ratios = ratios[np.isfinite(ratios)]
            if ratios.size:
                return float(np.clip(np.nanmedian(ratios), 0.0, 20.0))
        observed_finite = observed.to_numpy(dtype=np.float64)
        observed_finite = observed_finite[np.isfinite(observed_finite) & (observed_finite > 0.0)]
        model_finite = irradiance.to_numpy(dtype=np.float64)
        model_finite = model_finite[np.isfinite(model_finite) & (model_finite > 20.0)]
        if observed_finite.size >= 6 and model_finite.size >= 2:
            observed_scale = np.nanpercentile(observed_finite, 95)
            model_scale = np.nanpercentile(model_finite, 95)
            if np.isfinite(observed_scale) and np.isfinite(model_scale) and model_scale > 0.0:
                return float(np.clip(observed_scale / model_scale, 0.0, 20.0))
    return float(DEFAULT_SOLAR_CALIBRATION_FACTOR)


def calibrated_solar_factor_profile(
    base_factor: float,
    archive: xr.Dataset | None,
    frame: pd.DataFrame,
    forecast_times: pd.DatetimeIndex,
    *,
    issue_time: pd.Timestamp,
) -> tuple[pd.Series, dict[str, float]]:
    """Apply conservative, lead-specific solar MOS corrections.

    A single local panel factor cannot correct a systematic ECMWF error that
    grows with lead time. The correction is trained only on independent cycles
    and only on meaningful daylight samples; sparse buckets retain the physical
    panel factor unchanged.
    """
    factors = {bucket: 1.0 for bucket, _, _ in LEAD_BUCKETS}
    observed = _observed_solar_w(frame)
    if archive is not None and not observed.empty:
        table = _archive_verification_frame(
            archive,
            observed,
            forecast_var="ForecastSolarWatts",
            tolerance=pd.Timedelta(minutes=10),
        )
        table = _filter_active_forecast_contract(table, archive)
        table = _independent_verification_rows(table)
        if not table.empty:
            for bucket, start, stop in LEAD_BUCKETS:
                subset = table[
                    (table["lead_hour"] >= start)
                    & (table["lead_hour"] < stop)
                    & (table["forecast_value"] >= 50.0)
                    & (table["observed_value"] >= 0.0)
                ]
                if len(subset) < 12:
                    continue
                ratios = (subset["observed_value"] / subset["forecast_value"]).to_numpy(dtype=np.float64)
                ratios = ratios[np.isfinite(ratios)]
                if ratios.size >= 12:
                    # Bound a young calibration so one cloudy cycle cannot erase
                    # physically plausible charging in the next forecast.
                    factors[bucket] = float(np.clip(np.nanmedian(ratios), 0.4, 1.4))
    lead_hours = (pd.DatetimeIndex(forecast_times) - pd.Timestamp(issue_time)) / pd.Timedelta(hours=1)
    values = np.full(len(forecast_times), float(base_factor), dtype=np.float64)
    for bucket, start, stop in LEAD_BUCKETS:
        values[(lead_hours >= start) & (lead_hours < stop)] *= factors[bucket]
    return pd.Series(values, index=pd.DatetimeIndex(forecast_times)), factors


def integrate_soc_forecast(
    *,
    initial_soc: float,
    initial_time: pd.Timestamp | None = None,
    irradiance: pd.Series,
    solar_factor: float | pd.Series,
    load_w: float | pd.Series,
    fixed_solar_w: pd.Series | None = None,
    capacity_kwh: float = DEFAULT_BATTERY_CAPACITY_KWH,
    battery_model: BatteryModel | None = None,
) -> pd.DataFrame:
    """Integrate SOC forward from ECMWF solar and expected load."""
    if irradiance.empty:
        return pd.DataFrame()
    forecast_times = pd.DatetimeIndex(irradiance.index)
    forecast_irradiance = irradiance.to_numpy(dtype=np.float64)
    if fixed_solar_w is not None:
        fixed = pd.Series(fixed_solar_w, copy=True)
        fixed.index = pd.DatetimeIndex(fixed.index)
        if fixed.index.tz is not None:
            fixed.index = fixed.index.tz_convert("UTC").tz_localize(None)
        fixed = fixed[~fixed.index.duplicated(keep="last")].sort_index()
        matched = fixed.reindex(forecast_times)
        # A leading initial-SOC anchor has no interval solar forcing by
        # convention. Every actual forecast endpoint must be supplied exactly;
        # nearest-neighbour matching would silently break a paired ablation.
        missing = matched.isna()
        if missing.any():
            allowed_anchor = (
                initial_time is not None
                and len(matched) > 0
                and bool(missing.iloc[0])
                and pd.Timestamp(forecast_times[0]) == pd.Timestamp(initial_time)
            )
            if not allowed_anchor or int(missing.sum()) != 1:
                raise ValueError("fixed_solar_w must cover every forecast interval endpoint")
        forecast_solar_w = np.clip(matched.to_numpy(dtype=np.float64), 0.0, None)
    else:
        if isinstance(solar_factor, pd.Series):
            factors = solar_factor.reindex(forecast_times, method="nearest", tolerance=pd.Timedelta(hours=2))
            factors = factors.ffill().bfill().fillna(DEFAULT_SOLAR_CALIBRATION_FACTOR).to_numpy(dtype=np.float64)
        else:
            factors = np.full(len(forecast_times), float(solar_factor), dtype=np.float64)
        forecast_solar_w = np.clip(forecast_irradiance * factors, 0.0, None)
    if initial_time is not None:
        initial_time = pd.Timestamp(initial_time)
        if initial_time.tz is not None:
            initial_time = initial_time.tz_convert("UTC").tz_localize(None)
        if initial_time < forecast_times[0]:
            times = pd.DatetimeIndex([initial_time]).append(forecast_times)
            irradiance_values = np.concatenate(([np.nan], forecast_irradiance))
            solar_w = np.concatenate(([np.nan], forecast_solar_w))
        else:
            times = forecast_times
            irradiance_values = forecast_irradiance
            solar_w = forecast_solar_w
    else:
        times = forecast_times
        irradiance_values = forecast_irradiance
        solar_w = forecast_solar_w
    if isinstance(load_w, pd.Series):
        load_series = load_w.reindex(times, method="nearest", tolerance=pd.Timedelta(hours=2))
        if load_series.isna().all():
            load_values = np.full(len(times), DEFAULT_LOAD_W, dtype=np.float64)
        else:
            load_values = load_series.ffill().bfill().to_numpy(dtype=np.float64)
        load = np.clip(load_values, 0.0, None)
    else:
        load = np.full(len(times), max(float(load_w), 0.0), dtype=np.float64)
    model = battery_model or BatteryModel(
        usable_capacity_kwh=capacity_kwh,
        charge_efficiency=1.0,
        discharge_efficiency=1.0,
        max_charge_w=20_000.0,
        max_discharge_w=20_000.0,
    )
    model = model.validated()
    soc = np.full(len(times), np.nan, dtype=np.float64)
    soc[0] = float(np.clip(initial_soc, 0.0, 100.0))
    for idx in range(1, len(times)):
        dt_hours = max((times[idx] - times[idx - 1]) / pd.Timedelta(hours=1), 0.0)
        interval_solar_w = solar_w[idx]
        if not np.isfinite(interval_solar_w):
            interval_solar_w = 0.0
        delta_soc = soc_delta_percent(interval_solar_w - load[idx], dt_hours, model)
        soc[idx] = np.clip(soc[idx - 1] + float(delta_soc), 0.0, 100.0)
    return pd.DataFrame(
        {
            "BatterySOCForecast": soc,
            "ECMWFSolarIrradiance": irradiance_values,
            "ForecastSolarWatts": solar_w,
            "ForecastLoadWatts": load,
        },
        index=times,
    )


def integrate_soc_from_available_solar(
    *,
    initial_soc: float,
    available_solar_w: pd.Series,
    load_w: float | pd.Series,
    initial_time: pd.Timestamp | None = None,
    irradiance: pd.Series | None = None,
    capacity_kwh: float = DEFAULT_BATTERY_CAPACITY_KWH,
    battery_model: BatteryModel | None = None,
) -> pd.DataFrame:
    """Integrate SOC from physical available PV with explicit curtailment.

    Available PV first serves the station and battery parasitic demand.  Only
    the remaining power that the battery can accept is delivered; surplus PV is
    reported as curtailed.  Discharge is bounded by both the electrical limit
    and the energy remaining in the battery.
    """

    if available_solar_w.empty:
        return pd.DataFrame()
    forecast_times = pd.DatetimeIndex(available_solar_w.index)
    available_series = available_solar_w.reindex(forecast_times).clip(lower=0.0)
    prepended_anchor = False
    if initial_time is not None:
        initial_time = pd.Timestamp(initial_time)
        if initial_time.tz is not None:
            initial_time = initial_time.tz_convert("UTC").tz_localize(None)
        if initial_time < forecast_times[0]:
            prepended_anchor = True
            times = pd.DatetimeIndex([initial_time]).append(forecast_times)
            available = np.concatenate(([np.nan], available_series.to_numpy(dtype=np.float64)))
        else:
            times = forecast_times
            available = available_series.to_numpy(dtype=np.float64)
    else:
        times = forecast_times
        available = available_series.to_numpy(dtype=np.float64)

    if isinstance(load_w, pd.Series):
        # Forecast load values are interval means labelled by their right-hand
        # endpoint.  Preserve the baseline convention at physical substeps so
        # a load-phase transition cannot be mistaken for a solar-model change.
        source_load = load_w.copy()
        source_load.index = pd.DatetimeIndex(source_load.index)
        source_load = source_load[~source_load.index.duplicated(keep="last")].sort_index()
        load_series = source_load.reindex(times, method="bfill")
        if load_series.isna().all():
            load = np.full(len(times), DEFAULT_LOAD_W, dtype=np.float64)
        else:
            load = load_series.ffill().bfill().fillna(DEFAULT_LOAD_W).to_numpy(dtype=np.float64)
    else:
        load = np.full(len(times), max(float(load_w), 0.0), dtype=np.float64)
    load = np.clip(load, 0.0, None)

    model = battery_model or BatteryModel(usable_capacity_kwh=capacity_kwh)
    model = model.validated()
    soc = np.full(len(times), np.nan, dtype=np.float64)
    delivered = np.full(len(times), np.nan, dtype=np.float64)
    curtailed = np.full(len(times), np.nan, dtype=np.float64)
    battery_charge_input = np.full(len(times), np.nan, dtype=np.float64)
    battery_discharge_output = np.full(len(times), np.nan, dtype=np.float64)
    soc[0] = float(np.clip(initial_soc, 0.0, 100.0))

    for idx in range(1, len(times)):
        dt_hours = max((times[idx] - times[idx - 1]) / pd.Timedelta(hours=1), 0.0)
        pv_available = float(available[idx]) if np.isfinite(available[idx]) else 0.0
        non_battery_demand = float(load[idx]) + model.parasitic_load_w
        solar_to_demand = min(pv_available, non_battery_demand)
        surplus = max(pv_available - non_battery_demand, 0.0)
        deficit = max(non_battery_demand - pv_available, 0.0)

        remaining_kwh = max((100.0 - soc[idx - 1]) / 100.0 * model.usable_capacity_kwh, 0.0)
        stored_kwh = max(soc[idx - 1] / 100.0 * model.usable_capacity_kwh, 0.0)
        if dt_hours > 0.0:
            capacity_charge_limit_w = remaining_kwh * 1000.0 / (
                dt_hours * max(model.charge_efficiency, 1.0e-9)
            )
            energy_discharge_limit_w = (
                stored_kwh * 1000.0 * model.discharge_efficiency / dt_hours
            )
        else:
            capacity_charge_limit_w = 0.0
            energy_discharge_limit_w = 0.0
        accepted_charge_w = min(surplus, model.max_charge_w, capacity_charge_limit_w)
        supplied_discharge_w = min(deficit, model.max_discharge_w, energy_discharge_limit_w)
        stored_delta_kwh = (
            accepted_charge_w * model.charge_efficiency
            - supplied_discharge_w / model.discharge_efficiency
        ) * dt_hours / 1000.0
        soc[idx] = np.clip(
            soc[idx - 1] + 100.0 * stored_delta_kwh / model.usable_capacity_kwh,
            0.0,
            100.0,
        )
        delivered[idx] = solar_to_demand + accepted_charge_w
        curtailed[idx] = max(pv_available - delivered[idx], 0.0)
        battery_charge_input[idx] = accepted_charge_w
        battery_discharge_output[idx] = supplied_discharge_w

    irradiance_values = np.full(len(times), np.nan, dtype=np.float64)
    if irradiance is not None and not irradiance.empty:
        aligned = irradiance.reindex(times, method="nearest", tolerance=pd.Timedelta(hours=2))
        irradiance_values = aligned.to_numpy(dtype=np.float64)
        if prepended_anchor:
            irradiance_values[0] = np.nan
    return pd.DataFrame(
        {
            "BatterySOCForecast": soc,
            "ECMWFSolarIrradiance": irradiance_values,
            # Compatibility: ForecastSolarWatts remains available PV forcing.
            "ForecastSolarWatts": available,
            "ForecastPVAvailableWatts": available,
            "ForecastPVDeliveredWatts": delivered,
            "ForecastPVCurtailedWatts": curtailed,
            "ForecastBatteryChargeInputWatts": battery_charge_input,
            "ForecastBatteryDischargeOutputWatts": battery_discharge_output,
            "ForecastLoadWatts": load,
        },
        index=times,
    )


def _aggregate_physical_soc_forecast(
    integrated_substeps: pd.DataFrame,
    interval_solar: pd.DataFrame,
    irradiance: pd.Series,
    output_times: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Aggregate substep battery physics back to baseline forecast endpoints."""

    output_times = pd.DatetimeIndex(output_times)
    out = pd.DataFrame(index=output_times)
    out["BatterySOCForecast"] = integrated_substeps["BatterySOCForecast"].reindex(output_times)
    out["ECMWFSolarIrradiance"] = irradiance.reindex(output_times)
    for name in ("ForecastSolarWatts", "ForecastPVAvailableWatts"):
        out[name] = interval_solar[name].reindex(output_times)
    out["ForecastLoadWatts"] = integrated_substeps["ForecastLoadWatts"].reindex(
        output_times,
        method="nearest",
        tolerance=pd.Timedelta(hours=2),
    )
    energy_fields = (
        "ForecastPVDeliveredWatts",
        "ForecastPVCurtailedWatts",
        "ForecastBatteryChargeInputWatts",
        "ForecastBatteryDischargeOutputWatts",
    )
    for name in energy_fields:
        out[name] = np.nan
    source_times = pd.DatetimeIndex(integrated_substeps.index)
    for output_index in range(1, len(output_times)):
        start = pd.Timestamp(output_times[output_index - 1])
        end = pd.Timestamp(output_times[output_index])
        selected_times = source_times[(source_times > start) & (source_times <= end)]
        if len(selected_times) == 0:
            continue
        previous_times = pd.DatetimeIndex([start]).append(selected_times[:-1])
        weights = np.asarray(
            [
                (right - left) / pd.Timedelta(hours=1)
                for left, right in zip(previous_times, selected_times)
            ],
            dtype=np.float64,
        )
        for name in energy_fields:
            values = integrated_substeps[name].reindex(selected_times).to_numpy(dtype=np.float64)
            valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
            if np.any(valid):
                out.loc[end, name] = float(np.average(values[valid], weights=weights[valid]))
    return out


def _forecast_integration_times(
    initial_time: pd.Timestamp,
    forecast_times: pd.DatetimeIndex,
) -> pd.DatetimeIndex:
    """Return the exact timeline used by :func:`integrate_soc_forecast`."""
    times = pd.DatetimeIndex(forecast_times)
    if not len(times):
        return times
    anchor = pd.Timestamp(initial_time)
    if anchor.tz is not None:
        anchor = anchor.tz_convert("UTC").tz_localize(None)
    if anchor < times[0]:
        return pd.DatetimeIndex([anchor]).append(times)
    return times


def _extend_irradiance_with_diurnal_persistence(
    irradiance: pd.Series,
    horizon_end: pd.Timestamp,
) -> tuple[pd.Series, float]:
    """Extend the cycle-to-now tail with the preceding forecast day's shape."""
    if irradiance.empty or irradiance.index[-1] >= horizon_end:
        return irradiance, 0.0
    native_end = pd.Timestamp(irradiance.index[-1])
    differences = irradiance.index.to_series().diff().dropna()
    step = differences.median() if not differences.empty else pd.Timedelta(hours=3)
    if not isinstance(step, pd.Timedelta) or step <= pd.Timedelta(0):
        step = pd.Timedelta(hours=3)
    extension_times = list(pd.date_range(native_end + step, horizon_end, freq=step))
    if not extension_times or extension_times[-1] != horizon_end:
        extension_times.append(horizon_end)
    extended = irradiance.copy().sort_index()
    for target in extension_times:
        source_time = pd.Timestamp(target) - pd.Timedelta(hours=24)
        source_index = extended.index.union(pd.DatetimeIndex([source_time])).sort_values()
        source = extended.reindex(source_index).interpolate("time").get(source_time, np.nan)
        value = float(source) if np.isfinite(source) else float(extended.iloc[-1])
        extended.loc[pd.Timestamp(target)] = max(value, 0.0)
    extended = extended[~extended.index.duplicated(keep="last")].sort_index()
    extension_hours = float((pd.Timestamp(extended.index[-1]) - native_end) / pd.Timedelta(hours=1))
    return extended, max(extension_hours, 0.0)


def _candidate_residual_series(
    profile: Mapping[str, object],
    name: str,
    times: pd.DatetimeIndex,
) -> np.ndarray:
    """Align one optional issue-time-safe residual path to forecast endpoints."""
    value = profile.get(name)
    if value is None:
        return np.zeros(len(times), dtype=np.float64)
    if isinstance(value, pd.Series):
        aligned = value.reindex(times)
        if aligned.isna().any():
            aligned = aligned.ffill().bfill()
        return np.nan_to_num(aligned.to_numpy(dtype=np.float64), nan=0.0)
    values = np.asarray(value, dtype=np.float64)
    if values.ndim == 0:
        return np.full(len(times), float(values), dtype=np.float64)
    if values.shape != (len(times),):
        raise ValueError(f"Candidate residual {name} does not match the forecast time grid")
    return np.nan_to_num(values, nan=0.0)


def _apply_candidate_load_residual(
    profile: ControlledLoadProfile,
    times: pd.DatetimeIndex,
    residual: Mapping[str, object] | None,
    *,
    physical_floor_w: float = 0.0,
    physical_floor_source: str = "unavailable",
) -> tuple[ControlledLoadProfile, dict[str, object]]:
    """Apply bounded residual quantiles without changing state/phase semantics."""
    floor = float(physical_floor_w)
    if not np.isfinite(floor) or floor < 0.0:
        floor = 0.0
    if residual is None:
        return profile, {
            "load_residual_model_status": "not_requested",
            "load_residual_model_contract_id": "",
            "load_residual_state_digest": "",
            "load_residual_training_samples": 0,
            "load_residual_physical_floor_w": floor,
            "load_residual_physical_floor_source": str(physical_floor_source),
        }
    p10_correction = _candidate_residual_series(residual, "p10_correction_w", times)
    p50_correction = _candidate_residual_series(residual, "p50_correction_w", times)
    p90_correction = _candidate_residual_series(residual, "p90_correction_w", times)
    p10 = np.maximum(np.asarray(profile.p10_w, dtype=np.float64) + p10_correction, floor)
    p50 = np.maximum(np.asarray(profile.p50_w, dtype=np.float64) + p50_correction, floor)
    p90 = np.maximum(np.asarray(profile.p90_w, dtype=np.float64) + p90_correction, floor)
    p10 = np.minimum(p10, p50)
    p90 = np.maximum(p90, p50)
    residual_state_payload = {
        "schema": 1,
        "p10_correction_w": np.round(p10_correction, 8).tolist(),
        "p50_correction_w": np.round(p50_correction, 8).tolist(),
        "p90_correction_w": np.round(p90_correction, 8).tolist(),
        "training_samples": int(residual.get("training_samples", 0) or 0),
        "training_cycles": int(residual.get("training_cycles", 0) or 0),
        "training_days": int(residual.get("training_days", 0) or 0),
        "selection": _normalise_identity_text(residual.get("selection", "")),
    }
    diagnostics = {
        "load_residual_model_status": _normalise_identity_text(residual.get("status", "unknown")),
        "load_residual_model_contract_id": _normalise_identity_text(residual.get("contract_id", "")),
        "load_residual_state_digest": "sha256:"
        + hashlib.sha256(
            json.dumps(residual_state_payload, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "load_residual_training_samples": int(residual.get("training_samples", 0) or 0),
        "load_residual_training_cycles": int(residual.get("training_cycles", 0) or 0),
        "load_residual_training_days": int(residual.get("training_days", 0) or 0),
        "load_residual_bound_w": float(residual.get("bound_w", np.nan)),
        "load_residual_selection": _normalise_identity_text(residual.get("selection", "")),
        "load_residual_physical_floor_w": floor,
        "load_residual_physical_floor_source": str(physical_floor_source),
    }
    return (
        ControlledLoadProfile(
            p10,
            p50,
            p90,
            np.asarray(profile.phase_codes, dtype=np.int8),
            f"{profile.source}+bounded_issue_time_residual",
        ),
        diagnostics,
    )


def build_forecast_dataset(
    power: xr.Dataset,
    solar: xr.Dataset,
    *,
    pdu: xr.Dataset | None = None,
    previous_forecast: xr.Dataset | None = None,
    forecast_archive: xr.Dataset | None = None,
    state: dict[str, object] | None = None,
    horizon_hours: int = DEFAULT_HORIZON_HOURS,
    calibration_days: float = DEFAULT_CALIBRATION_DAYS,
    fallback_calibration_hours: float = DEFAULT_FALLBACK_CALIBRATION_HOURS,
    capacity_kwh: float = DEFAULT_BATTERY_CAPACITY_KWH,
    solar_model: str = DEFAULT_SOLAR_MODEL,
    physical_solar_config: PhysicalSolarConfig | None = None,
    latitude: float = DEFAULT_LATITUDE,
    longitude: float = DEFAULT_LONGITUDE,
    fixed_soc_bias_corrections: dict[str, float] | None = None,
    load_residual_profile: Mapping[str, object] | None = None,
    fixed_legacy_solar_w: pd.Series | None = None,
    site_irradiance_override: pd.Series | None = None,
) -> xr.Dataset:
    selected_solar_model = validate_solar_model(solar_model)
    if fixed_legacy_solar_w is not None and selected_solar_model != LEGACY_SOLAR_MODEL_NAME:
        raise ValueError("fixed_legacy_solar_w is valid only with the legacy solar model")
    frame = _power_frame(power)
    if frame.empty or "BatterySOC" not in frame:
        raise ValueError("Power dataset needs BatterySOC to initialize the SOC forecast")
    latest_time, latest_soc = latest_finite(frame["BatterySOC"])
    battery_model = fit_battery_model(
        frame.assign(ObservedLoadWatts=_observed_load_w(frame)),
        nominal_capacity_kwh=capacity_kwh,
        lookback_days=calibration_days,
    )
    state = dict(state or {})
    # Only independent ECMWF cycles are allowed to tune the live model. The
    # full archive remains available for audit, but repeated same-cycle
    # re-anchors otherwise overweight a single weather realisation.
    previous_metrics = evaluate_independent_forecast_archive(forecast_archive, frame)
    if not previous_metrics:
        previous_metrics = evaluate_previous_forecast(previous_forecast, frame)
    embedded_site_forcing = site_irradiance_override is not None
    if embedded_site_forcing:
        irradiance = _normalise_site_irradiance_override(site_irradiance_override)
        if pd.Timestamp(irradiance.index[0]) != pd.Timestamp(latest_time):
            raise ValueError(
                "Embedded site irradiance must begin on the exact SOC anchor time"
            )
        if not np.isfinite(float(irradiance.iloc[0])):
            irradiance = irradiance.iloc[1:]
    else:
        irradiance = solar_irradiance_from_ssrd(solar)
        if irradiance.empty:
            raise ValueError("No ECMWF solar forecast samples could be converted from ssrd")
    horizon_end = latest_time + pd.Timedelta(hours=horizon_hours)
    irradiance = irradiance[(irradiance.index >= latest_time) & (irradiance.index <= horizon_end)]
    irradiance, solar_tail_extension_hours = _extend_irradiance_with_diurnal_persistence(irradiance, horizon_end)
    if embedded_site_forcing and solar_tail_extension_hours > 0.0:
        raise ValueError(
            "Embedded site irradiance does not cover the complete paired forecast horizon"
        )
    if len(irradiance) < 2:
        raise ValueError("ECMWF solar forecast does not overlap the requested SOC forecast horizon")

    adaptive_alpha = _balanced_alpha(
        max(
            int(previous_metrics.get("soc_sample_count", 0) or 0),
            int(previous_metrics.get("solar_sample_count", 0) or 0),
            int(previous_metrics.get("load_sample_count", 0) or 0),
        )
    )
    physical_solar_frame: pd.DataFrame | None = None
    physical_solar_substeps: pd.DataFrame | None = None
    if selected_solar_model == LEGACY_SOLAR_MODEL_NAME:
        factor_raw = calibrate_solar_factor(
            frame,
            irradiance,
            end=latest_time,
            calibration_days=calibration_days,
            fallback_hours=fallback_calibration_hours,
        )
        factor = _adaptive_value(
            factor_raw,
            state.get("solar_calibration_factor_w_per_wm2"),
            alpha=adaptive_alpha,
        )
        solar_factor_profile, solar_mos_by_bucket = calibrated_solar_factor_profile(
            factor,
            forecast_archive,
            frame,
            pd.DatetimeIndex(irradiance.index),
            issue_time=latest_time,
        )
        solar_calibration_id = solar_calibration_contract_id(factor, solar_mos_by_bucket)
        solar_contract_id = legacy_solar_model_contract_id()
        solar_metadata: dict[str, object] = {
            "solar_model_name": LEGACY_SOLAR_MODEL_NAME,
            "solar_model_version": str(LEGACY_SOLAR_MODEL_VERSION),
            "solar_feature_set_version": LEGACY_SOLAR_FEATURE_SET_VERSION,
            "solar_model_contract_id": solar_contract_id,
            "solar_model_status": "operational_baseline",
            "solar_degradation_codes": "none",
            "solar_power_semantics": "electrical_power_from_ssrd_scalar",
            "solar_residual_calibration": "independent_cycle_lead_specific_mos",
        }
    else:
        if physical_solar_config is None:
            raise ValueError(
                "The physical solar model requires an explicit PhysicalSolarConfig; "
                "no inferred configuration is loaded implicitly"
            )
        factor_raw = 1.0
        factor = 1.0
        solar_mos_by_bucket = {bucket: 1.0 for bucket, _, _ in LEAD_BUCKETS}
        solar_factor_profile = pd.Series(
            np.ones(len(irradiance), dtype=np.float64),
            index=pd.DatetimeIndex(irradiance.index),
        )
        physical_solar_frame, physical_solar_substeps, solar_metadata = (
            build_physical_solar_forecast_frames(
            irradiance,
            latitude=latitude,
            longitude=longitude,
            config=physical_solar_config,
            forecast_start_time=latest_time,
            **_physical_solar_meteorology(solar, irradiance),
            )
        )
        if physical_solar_frame.empty or physical_solar_substeps.empty:
            raise ValueError("Physical solar model produced no forecast samples")
        solar_contract_id = str(solar_metadata["solar_model_contract_id"])
        solar_calibration_id = "physical-residual-calibration-disabled"
        solar_metadata.update(
            {
                "solar_model_status": "candidate",
                # Register 791 (MPP active versus limited) is not archived yet.
                # Learning against censored PV output would bias available power low.
                "solar_residual_calibration": "disabled_until_uncurtailed_mppt_samples_exist",
                "solar_training_cutoff_utc": latest_time.isoformat(),
            }
        )
    load_forecast_times = _forecast_integration_times(
        latest_time,
        pd.DatetimeIndex(irradiance.index),
    )
    raw_load_profile = build_historical_load_forecast(
        frame,
        load_forecast_times,
        end=latest_time,
        calibration_days=calibration_days,
        default_load_w=DEFAULT_LOAD_W,
    )
    load_diagnostics = dict(raw_load_profile.attrs)
    raw_load_w = float(raw_load_profile.median()) if not raw_load_profile.empty else float(DEFAULT_LOAD_W)
    load_mode_registry, clean_dc_only_level_w = _repair_dc_only_registry(
        state.get("load_mode_registry"),
        frame,
        end=latest_time,
    )
    load_mode, mode_source, active_kits, pdu_time, pdu_active_watts = _resolve_load_mode(
        frame,
        pdu,
        end=latest_time,
        observed_level_w=raw_load_w,
        raw_registry=load_mode_registry,
        previous_mode=state.get("current_load_mode"),
    )
    mode_signature = _load_mode_signature(load_mode, mode_source, active_kits)
    learning_ready, learning_reason = _mode_learning_status(load_diagnostics, load_mode)
    if learning_ready:
        load_mode_registry, learned_load_w = _update_load_mode_registry(
            load_mode_registry,
            mode=load_mode,
            observed_level_w=raw_load_w,
            observed_at=latest_time,
            active_kits=active_kits,
            mode_source=mode_source,
            signature=mode_signature,
        )
    else:
        learned_load_w = np.nan
    component_load_w = (
        float(clean_dc_only_level_w) + float(pdu_active_watts)
        if mode_source in {"pdu_signature", "pdu_ac_signature"}
        and np.isfinite(pdu_active_watts)
        and clean_dc_only_level_w is not None
        else np.nan
    )
    mode_entry = load_mode_registry.get(load_mode, {})
    learning_observations = int(mode_entry.get("observation_count", 0) or 0)
    try:
        learned_reference_w = float(mode_entry.get("learned_level_w", learned_load_w))
    except (TypeError, ValueError):
        learned_reference_w = np.nan
    learned_levels = []
    for observation in mode_entry.get("observations", []):
        try:
            value = float(observation["level_w"])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(value) and value >= 0.0:
            learned_levels.append(value)
    controlled_load = estimate_controlled_load(
        mode=load_mode,
        measured_current_w=raw_load_w,
        learned_observations_w=learned_levels,
        learned_level_w=learned_reference_w,
        component_estimate_w=component_load_w,
        dc_only_estimate_w=clean_dc_only_level_w,
        minimum_observations=DEFAULT_MODE_MIN_OBSERVATIONS,
    )
    state_dynamics = None
    state_dynamics_reason = "no_exact_pdu_state"
    exact_state_id = ""
    if load_mode == "DC-Only" or active_kits:
        try:
            from power_operating_scenarios import MODE_DC_ONLY, build_observation_frame, mode_id

            exact_state_id = mode_id(active_kits) if active_kits else MODE_DC_ONLY
            state_observations = build_observation_frame(
                power,
                pdu,
                end=latest_time,
                lookback_days=max(float(calibration_days), 14.0),
            )
            if state_observations.empty or str(state_observations["direct_mode"].iloc[-1]) != exact_state_id:
                state_dynamics_reason = "latest_observation_state_mismatch"
            else:
                candidate = learn_state_load_dynamics(state_observations, exact_state_id)
                if candidate is None:
                    state_dynamics_reason = "no_phase_profile"
                elif candidate.sample_count < DEFAULT_DYNAMICS_MIN_SAMPLES:
                    state_dynamics_reason = "insufficient_exact_state_phase_samples"
                else:
                    state_dynamics = candidate
                    state_dynamics_reason = "learned_exact_state_phases"
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            state_dynamics_reason = f"phase_learning_unavailable:{type(exc).__name__}"
    unadjusted_controlled_profile = build_controlled_load_profile(
        state_dynamics,
        pd.DatetimeIndex(raw_load_profile.index),
        controlled_load,
    )
    residual_floor_w = 0.0
    residual_floor_source = "unavailable"
    if clean_dc_only_level_w is not None and np.isfinite(clean_dc_only_level_w) and clean_dc_only_level_w > 0.0:
        residual_floor_w = float(clean_dc_only_level_w)
        residual_floor_source = "fresh_clean_dark_measurement"
    else:
        dc_only_entry = load_mode_registry.get("DC-Only", {})
        try:
            registry_floor_w = float(dc_only_entry.get("learned_level_w", np.nan))
        except (TypeError, ValueError):
            registry_floor_w = np.nan
        registry_count = int(dc_only_entry.get("observation_count", 0) or 0)
        if np.isfinite(registry_floor_w) and registry_floor_w > 0.0 and registry_count >= DEFAULT_MODE_MIN_OBSERVATIONS:
            residual_floor_w = registry_floor_w
            residual_floor_source = "dc_only_state_registry"
    controlled_profile, load_residual_diagnostics = _apply_candidate_load_residual(
        unadjusted_controlled_profile,
        pd.DatetimeIndex(raw_load_profile.index),
        load_residual_profile,
        physical_floor_w=residual_floor_w,
        physical_floor_source=residual_floor_source,
    )
    load_w = float(controlled_profile.p50_w[0])
    load_anchor_disagreement_w = float(load_w - raw_load_w)
    load_diagnostics.update(
        {
            "load_mode": load_mode,
            "load_mode_source": mode_source,
            "load_mode_active_kits": active_kits,
            "load_mode_pdu_time": pdu_time,
            "load_mode_pdu_active_watts": pdu_active_watts,
            "load_mode_signature": mode_signature,
            "load_mode_learning_ready": learning_ready,
            "load_mode_learning_reason": learning_reason,
            "load_mode_learning_observations": learning_observations,
            "load_mode_minimum_observations": DEFAULT_MODE_MIN_OBSERVATIONS,
            "load_regime": load_mode,
            "load_regime_level_w": load_w,
            "load_component_estimate_w": component_load_w,
            "load_anchor_disagreement_w": load_anchor_disagreement_w,
            "load_learned_reference_w": learned_reference_w,
            "load_anchor_method": controlled_profile.source,
            "load_distribution_sample_count": controlled_load.sample_count,
            "load_exact_state_id": exact_state_id,
            "load_current_phase": state_dynamics.current_phase if state_dynamics is not None else "steady",
            "load_state_dynamics_reason": state_dynamics_reason,
            **load_residual_diagnostics,
        }
    )
    # Bias learned by retired load models is not transferable across operating
    # modes. Each named mode learns its own robust power-balance level.
    load_bias_correction = float(
        controlled_profile.p50_w[0] - unadjusted_controlled_profile.p50_w[0]
    )
    load_profile = pd.Series(controlled_profile.p50_w, index=raw_load_profile.index)
    if selected_solar_model == LEGACY_SOLAR_MODEL_NAME:
        forecast = integrate_soc_forecast(
            initial_soc=latest_soc,
            initial_time=latest_time,
            irradiance=irradiance,
            solar_factor=solar_factor_profile,
            load_w=load_profile,
            fixed_solar_w=fixed_legacy_solar_w,
            capacity_kwh=battery_model.usable_capacity_kwh,
            battery_model=battery_model,
        )
    else:
        assert physical_solar_frame is not None
        assert physical_solar_substeps is not None
        integrated_substeps = integrate_soc_from_available_solar(
            initial_soc=latest_soc,
            initial_time=latest_time,
            available_solar_w=physical_solar_substeps["ForecastPVAvailableWatts"],
            irradiance=physical_solar_substeps["ECMWFSolarIrradiance"],
            load_w=load_profile,
            capacity_kwh=battery_model.usable_capacity_kwh,
            battery_model=battery_model,
        )
        forecast = _aggregate_physical_soc_forecast(
            integrated_substeps,
            physical_solar_frame,
            irradiance,
            _forecast_integration_times(latest_time, pd.DatetimeIndex(physical_solar_frame.index)),
        )
        physical_aligned = physical_solar_frame.reindex(forecast.index)
        for name in physical_aligned.columns:
            if name not in forecast:
                forecast[name] = physical_aligned[name]
    forecast["ForecastLoadP10Watts"] = controlled_profile.p10_w
    forecast["ForecastLoadP50Watts"] = controlled_profile.p50_w
    forecast["ForecastLoadP90Watts"] = controlled_profile.p90_w
    forecast["ForecastLoadPhaseCode"] = controlled_profile.phase_codes
    for scenario_load_w in SCENARIO_LOADS_W:
        if selected_solar_model == LEGACY_SOLAR_MODEL_NAME:
            scenario = integrate_soc_forecast(
                initial_soc=latest_soc,
                initial_time=latest_time,
                irradiance=irradiance,
                solar_factor=solar_factor_profile,
                load_w=float(scenario_load_w),
                fixed_solar_w=fixed_legacy_solar_w,
                capacity_kwh=battery_model.usable_capacity_kwh,
                battery_model=battery_model,
            )
        else:
            assert physical_solar_frame is not None
            assert physical_solar_substeps is not None
            scenario_substeps = integrate_soc_from_available_solar(
                initial_soc=latest_soc,
                initial_time=latest_time,
                available_solar_w=physical_solar_substeps["ForecastPVAvailableWatts"],
                irradiance=physical_solar_substeps["ECMWFSolarIrradiance"],
                load_w=float(scenario_load_w),
                capacity_kwh=battery_model.usable_capacity_kwh,
                battery_model=battery_model,
            )
            scenario = _aggregate_physical_soc_forecast(
                scenario_substeps,
                physical_solar_frame,
                irradiance,
                pd.DatetimeIndex(forecast.index),
            )
        forecast[scenario_soc_field(scenario_load_w)] = scenario["BatterySOCForecast"]
        scenario_for_validation = pd.DataFrame(
            {
                "BatterySOCForecast": scenario["BatterySOCForecast"],
                "ForecastSolarWatts": scenario["ForecastSolarWatts"],
                "ForecastLoadWatts": scenario["ForecastLoadWatts"],
            },
            index=scenario.index,
        )
        if "ForecastBatteryDischargeOutputWatts" in scenario:
            scenario_for_validation["ForecastBatteryDischargeOutputWatts"] = scenario[
                "ForecastBatteryDischargeOutputWatts"
            ]
        if "ForecastBatteryChargeInputWatts" in scenario:
            scenario_for_validation["ForecastBatteryChargeInputWatts"] = scenario[
                "ForecastBatteryChargeInputWatts"
            ]
        validate_soc_physical_consistency(
            scenario_for_validation,
            parasitic_load_w=battery_model.parasitic_load_w,
        )
    if fixed_soc_bias_corrections is None:
        soc_bias_corrections = _soc_bias_corrections(
            state.get("soc_bias_correction_pct_points_by_bucket"),
            previous_metrics,
            alpha=adaptive_alpha,
        )
    else:
        soc_bias_corrections = _soc_bias_corrections(
            fixed_soc_bias_corrections,
            {},
            alpha=0.0,
        )
    forecast = _apply_soc_bias_corrections(
        forecast,
        soc_bias_corrections,
        issue_time=latest_time,
        parasitic_load_w=battery_model.parasitic_load_w,
    )
    validate_soc_physical_consistency(
        forecast,
        parasitic_load_w=battery_model.parasitic_load_w,
    )
    soc_mae = float(previous_metrics.get("soc_mae_pct_points", np.nan))
    solar_mae = float(previous_metrics.get("solar_mae_w", np.nan))
    load_mae = float(previous_metrics.get("load_mae_w", np.nan))
    load_bias = float(previous_metrics.get("load_bias_w", np.nan))
    evaluation_samples = max(
        int(previous_metrics.get("soc_sample_count", 0) or 0),
        int(previous_metrics.get("solar_sample_count", 0) or 0),
        int(previous_metrics.get("load_sample_count", 0) or 0),
    )
    forecast["ForecastSOCMAERecent"] = soc_mae
    for bucket, _, _ in LEAD_BUCKETS:
        forecast[f"ForecastSOCMAE_{bucket}"] = float(previous_metrics.get(_metric_bucket_name("soc_mae", bucket), np.nan))
    forecast["ForecastSolarMAERecent"] = solar_mae
    forecast["ForecastLoadMAERecent"] = load_mae
    forecast["ForecastLoadBiasRecent"] = load_bias
    forecast["ForecastEvaluationSamples"] = float(evaluation_samples)
    forecast["ForecastSkillSampleCount"] = float(evaluation_samples)
    state_dynamics_json = json.dumps(state_dynamics.to_dict(), sort_keys=True) if state_dynamics is not None else "{}"
    state_dynamics_signature = hashlib.sha256(state_dynamics_json.encode("utf-8")).hexdigest()[:16]
    physical_candidate = selected_solar_model == PHYSICAL_SOLAR_MODEL_NAME
    forecast_model_name = (
        "aps_soc_energy_balance_v11_candidate"
        if physical_candidate
        else "aps_soc_energy_balance_v10"
    )
    forecast_model_version = "11" if physical_candidate else "10"
    forecast_contract_payload = {
        "schema": 1,
        "forecast_model_name": forecast_model_name,
        "forecast_model_version": forecast_model_version,
        "solar_model_contract_id": solar_contract_id,
        "load_model": LOAD_MODEL_NAME,
        "load_model_version": LOAD_MODEL_VERSION,
        "load_state_contract": CONTROLLED_LOAD_CONTRACT,
        "battery_energy_model": "bounded_bidirectional_efficiency_v1",
    }
    forecast_model_contract = "forecast-model-v1-" + hashlib.sha256(
        json.dumps(forecast_contract_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    solar_metadata_attrs = {str(key): str(value) for key, value in solar_metadata.items()}
    out = xr.Dataset(
        {name: (("time",), forecast[name].to_numpy(dtype=np.float32)) for name in forecast.columns},
        coords={"time": forecast.index.to_numpy(dtype="datetime64[ns]")},
        attrs={
            "power_soc_forecast_product": "true",
            "source": (
                "derived from ECMWF surface solar radiation, three-array PV physics, and APS power history"
                if physical_candidate
                else "derived from ECMWF ssrd forecast and APS power history"
            ),
            "forecast_model_name": forecast_model_name,
            "forecast_model_version": forecast_model_version,
            "forecast_model_contract_id": forecast_model_contract,
            "forecast_model_status": "candidate" if physical_candidate else "operational_baseline",
            "ecmwf_param": ECMWF_PARAM,
            "generated_at_utc": _utc_now(),
            "initial_soc_pct": f"{latest_soc:.6g}",
            "initial_soc_time": latest_time.isoformat(),
            "forecast_horizon_hours": str(int(horizon_hours)),
            "solar_tail_extension_hours": f"{solar_tail_extension_hours:.6g}",
            "solar_tail_extension_method": "24h_diurnal_persistence" if solar_tail_extension_hours > 0 else "none",
            "calibration_days": str(float(calibration_days)),
            "solar_calibration_factor_w_per_wm2": f"{factor:.6g}",
            "raw_solar_calibration_factor_w_per_wm2": f"{factor_raw:.6g}",
            "solar_mos_factor_by_lead_bucket": json.dumps(solar_mos_by_bucket, sort_keys=True),
            "solar_calibration": (
                "fixed provisional physical configuration; residual calibration disabled until uncurtailed MPPT samples exist"
                if physical_candidate
                else "replayed baseline legacy solar trace for paired load ablation"
                if fixed_legacy_solar_w is not None
                else "physical panel factor plus independent-cycle lead-specific MOS"
            ),
            "solar_forcing_mode": (
                "baseline_legacy_trace_replayed"
                if fixed_legacy_solar_w is not None
                else "physical_available_pv"
                if physical_candidate
                else "calibrated_ssrd_scalar"
            ),
            # Compatibility: this historical attribute name now correctly
            # identifies learned calibration state, not the solar algorithm.
            "solar_calibration_contract_id": solar_calibration_id,
            "solar_observation_censoring_status": (
                "uncurtailed_samples_unavailable_register_791_not_archived"
                if physical_candidate
                else "not_explicitly_filtered"
            ),
            "solar_training_cutoff_utc": latest_time.isoformat(),
            "forecast_load_w": f"{load_w:.6g}",
            "forecast_load_p10_w": f"{float(controlled_profile.p10_w[0]):.6g}",
            "forecast_load_p50_w": f"{float(controlled_profile.p50_w[0]):.6g}",
            "forecast_load_p90_w": f"{float(controlled_profile.p90_w[0]):.6g}",
            "raw_forecast_load_w": f"{raw_load_w:.6g}",
            "load_bias_correction_w": f"{load_bias_correction:.6g}",
            "soc_bias_correction_pct_points_by_bucket": json.dumps(soc_bias_corrections, sort_keys=True),
            "soc_bias_correction_method": "continuous_energy_direction_preserving_attenuation_v3",
            "soc_physical_consistency_status": "passed",
            "load_model": LOAD_MODEL_NAME,
            "load_model_version": str(LOAD_MODEL_VERSION),
            "load_state_contract": CONTROLLED_LOAD_CONTRACT,
            "load_state_hold_policy": STATE_HOLD_POLICY,
            "load_state_uncertainty_source": (
                "elapsed_time_exact_state_phase_distribution"
                if state_dynamics is not None
                else "exact_state_observations"
                if controlled_load.sample_count >= DEFAULT_MODE_MIN_OBSERVATIONS
                else "no_cross_state_spread"
            ),
            "load_exact_state_id": exact_state_id,
            "load_current_phase": state_dynamics.current_phase if state_dynamics is not None else "steady",
            "load_state_dynamics_reason": state_dynamics_reason,
            "load_state_dynamics": state_dynamics_json,
            "load_state_dynamics_signature": state_dynamics_signature,
            "load_mode": load_mode,
            "load_mode_source": mode_source,
            "load_mode_active_kits": ",".join(active_kits),
            "load_mode_pdu_time": pdu_time.isoformat() if pdu_time is not None else "",
            "load_mode_pdu_active_watts": f"{float(pdu_active_watts):.6g}",
            "load_mode_signature": mode_signature,
            "load_mode_learning_ready": str(bool(learning_ready)).lower(),
            "load_mode_learning_reason": learning_reason,
            "load_mode_learning_observations": str(learning_observations),
            "load_measurement": str(load_diagnostics.get("load_measurement", "unknown")),
            "load_balance_measurement": str(load_diagnostics.get("load_balance_measurement", "unknown")),
            "load_mode_registry": json.dumps(load_mode_registry, sort_keys=True),
            "load_anchor_method": str(load_diagnostics.get("load_anchor_method", "unknown")),
            "load_residual_model_status": str(load_diagnostics.get("load_residual_model_status", "not_requested")),
            "load_residual_model_contract_id": str(load_diagnostics.get("load_residual_model_contract_id", "")),
            "load_residual_state_digest": str(
                load_diagnostics.get("load_residual_state_digest", "")
            ),
            "load_residual_training_samples": str(int(load_diagnostics.get("load_residual_training_samples", 0))),
            "load_residual_training_cycles": str(int(load_diagnostics.get("load_residual_training_cycles", 0))),
            "load_residual_training_days": str(int(load_diagnostics.get("load_residual_training_days", 0))),
            "load_residual_bound_w": f"{float(load_diagnostics.get('load_residual_bound_w', np.nan)):.6g}",
            "load_residual_selection": str(load_diagnostics.get("load_residual_selection", "")),
            "load_residual_physical_floor_w": f"{float(load_diagnostics.get('load_residual_physical_floor_w', 0.0)):.6g}",
            "load_residual_physical_floor_source": str(
                load_diagnostics.get("load_residual_physical_floor_source", "unavailable")
            ),
            "load_component_estimate_w": f"{float(load_diagnostics.get('load_component_estimate_w', np.nan)):.6g}",
            "load_anchor_disagreement_w": f"{float(load_diagnostics.get('load_anchor_disagreement_w', np.nan)):.6g}",
            "load_learned_reference_w": f"{float(load_diagnostics.get('load_learned_reference_w', np.nan)):.6g}",
            "load_regime": str(load_diagnostics.get("load_regime", "unknown")),
            "load_regime_threshold_w": f"{float(load_diagnostics.get('load_regime_threshold_w', np.nan)):.6g}",
            "load_regime_level_w": f"{float(load_diagnostics.get('load_regime_level_w', load_w)):.6g}",
            "load_regime_run_hours": f"{float(load_diagnostics.get('load_regime_run_hours', 0.0)):.6g}",
            "load_regime_sample_count": str(int(load_diagnostics.get("load_regime_sample_count", 0))),
            **battery_model.attrs(),
            "adaptive_alpha": f"{adaptive_alpha:.6g}",
            "previous_forecast_metrics": json.dumps(previous_metrics, sort_keys=True),
            "scenario_loads_w": ",".join(str(load_w) for load_w in SCENARIO_LOADS_W),
            "scenario_loads_role": "legacy_fixed_load_sensitivity_only",
            "scenario_solar_mode": "ecmwf_physical_pv" if physical_candidate else "ecmwf",
            "minimum_operational_soc_pct": f"{MINIMUM_OPERATIONAL_SOC_PCT:g}",
            **solar_metadata_attrs,
        },
    )
    out["BatterySOCForecast"].attrs["units"] = "%"
    if "ForecastSOCBiasCorrectionPctPoints" in out:
        out["ForecastSOCBiasCorrectionPctPoints"].attrs["units"] = "percentage points"
    for scenario_load_w in SCENARIO_LOADS_W:
        field = scenario_soc_field(scenario_load_w)
        out[field].attrs["units"] = "%"
        out[field].attrs["scenario_load_w"] = str(int(scenario_load_w))
        out[field].attrs["scenario_solar_mode"] = out.attrs["scenario_solar_mode"]
    out["ECMWFSolarIrradiance"].attrs["units"] = "W m-2"
    out["ForecastSolarWatts"].attrs["units"] = "W"
    out["ForecastSolarWatts"].attrs["semantics"] = str(
        solar_metadata.get("solar_power_semantics", "electrical_solar_power")
    )
    out["ForecastLoadWatts"].attrs["units"] = "W"
    irradiance_fields = {
        "ECMWFDirectNormalIrradiance",
        "ECMWFDirectHorizontalIrradiance",
        "ECMWFDiffuseHorizontalIrradiance",
        "SolarExtraterrestrialNormalIrradiance",
        "SolarExtraterrestrialHorizontalIrradiance",
    }
    for name in out.data_vars:
        if (
            name in irradiance_fields
            or name == "ForecastEffectiveGlobalHorizontalIrradiance"
            or name.startswith("ForecastPlaneOfArray")
            or name.startswith("ForecastEffectivePlaneOfArray")
        ):
            out[name].attrs["units"] = "W m-2"
        elif name in {"SolarZenithDegrees", "SolarAzimuthDegrees"}:
            out[name].attrs["units"] = "degree"
        elif name in {"SolarCosineZenith", "ECMWFClearnessIndex"}:
            out[name].attrs["units"] = "1"
        elif name == "SolarForcingConsistencyFlag":
            out[name].attrs["units"] = "1"
            out[name].attrs["flag_meanings"] = (
                "0 consistent; bit 1 positive GHI below geometric horizon; "
                "bit 2 direct horizontal clipped to GHI"
            )
        elif name in {"SolarIntervalHours", "ECMWFSourceIntervalHours"}:
            out[name].attrs["units"] = "h"
        elif name.startswith("ForecastPVCellTemperature"):
            out[name].attrs["units"] = "degree_Celsius"
        elif name.startswith("ForecastPV") or name in {
            "ForecastBatteryChargeInputWatts",
            "ForecastBatteryDischargeOutputWatts",
        }:
            out[name].attrs["units"] = "W"
    if "ForecastPVAvailableWatts" in out:
        out["ForecastPVAvailableWatts"].attrs["semantics"] = "available_dc_before_battery_acceptance"
    if "ForecastPVDeliveredWatts" in out:
        out["ForecastPVDeliveredWatts"].attrs["semantics"] = "pv_serving_station_load_or_accepted_by_battery"
    if "ForecastPVCurtailedWatts" in out:
        out["ForecastPVCurtailedWatts"].attrs["semantics"] = "available_pv_not_used_or_accepted"
    if "ECMWFSolarIrradiance" in out:
        out["ECMWFSolarIrradiance"].attrs["semantics"] = "raw_ecmwf_source_interval_mean_ghi"
    if "ECMWFSourceIntervalHours" in out:
        out["ECMWFSourceIntervalHours"].attrs["semantics"] = "raw_ecmwf_source_interval_duration"
    if "ForecastEffectiveGlobalHorizontalIrradiance" in out:
        out["ForecastEffectiveGlobalHorizontalIrradiance"].attrs["semantics"] = (
            "issue_truncated_interval_mean_ghi_used_by_physical_pv"
        )
    for array_name in ("East", "South", "West"):
        raw_name = f"ForecastPlaneOfArrayIrradiance{array_name}"
        effective_name = f"ForecastEffectivePlaneOfArrayIrradiance{array_name}"
        temperature_name = f"ForecastPVCellTemperature{array_name}"
        if raw_name in out:
            out[raw_name].attrs["semantics"] = "raw_geometric_plane_of_array_irradiance"
        if effective_name in out:
            out[effective_name].attrs["semantics"] = "iam_adjusted_irradiance_used_by_pvwatts"
        if temperature_name in out:
            out[temperature_name].attrs["semantics"] = "faiman_module_temperature_proxy"
    out["ForecastLoadPhaseCode"] = (
        ("time",),
        np.asarray(controlled_profile.phase_codes, dtype=np.int8),
    )
    for name in ("ForecastLoadP10Watts", "ForecastLoadP50Watts", "ForecastLoadP90Watts"):
        out[name].attrs["units"] = "W"
        out[name].attrs["load_state_contract"] = CONTROLLED_LOAD_CONTRACT
    out["ForecastLoadPhaseCode"].attrs["phase_mapping"] = json.dumps(
        {str(code): name for name, code in PHASE_CODES.items()}, sort_keys=True
    )
    out["ForecastSOCMAERecent"].attrs["units"] = "percentage points"
    for bucket, _, _ in LEAD_BUCKETS:
        out[f"ForecastSOCMAE_{bucket}"].attrs["units"] = "percentage points"
    out["ForecastSolarMAERecent"].attrs["units"] = "W"
    out["ForecastLoadMAERecent"].attrs["units"] = "W"
    out["ForecastLoadBiasRecent"].attrs["units"] = "W"
    out["ForecastEvaluationSamples"].attrs["units"] = "samples"
    out["ForecastSkillSampleCount"].attrs["units"] = "samples"
    out.attrs["adaptive_calibration_state_id"] = adaptive_calibration_state_id(out.attrs)
    return out


def generate(
    power_zarr: Path = POWER_ZARR_PATH,
    output_zarr: Path = POWER_SOC_FORECAST_ZARR_PATH,
    *,
    pdu_zarr: Path = POWER_PDU_ZARR_PATH,
    input_forecast: Path | None = None,
    cache_dir: Path = POWER_ECMWF_FORECAST_CACHE_DIR,
    state_path: Path = POWER_SOC_FORECAST_STATE_PATH,
    archive_zarr: Path = POWER_SOC_FORECAST_ARCHIVE_ZARR_PATH,
    skill_zarr: Path | None = POWER_SOC_FORECAST_SKILL_ZARR_PATH,
    hindcast_zarr: Path | None = POWER_SOC_HINDCAST_ZARR_PATH,
    latitude: float = DEFAULT_LATITUDE,
    longitude: float = DEFAULT_LONGITUDE,
    horizon_hours: int = DEFAULT_HORIZON_HOURS,
    refresh_from_cache: bool = False,
    provider: str = DEFAULT_PROVIDER,
    shadow_report_path: Path | None = DEFAULT_SHADOW_REPORT_PATH,
    ecmwf_cycle_hour: int | str | None = None,
    max_power_age_minutes: float | None = DEFAULT_MAX_POWER_INPUT_AGE_MINUTES,
    archive_forecast: bool | None = None,
    solar_model: str = DEFAULT_SOLAR_MODEL,
    physical_solar_config_path: Path = DEFAULT_PHYSICAL_SOLAR_CONFIG_PATH,
    power_cutoff_time: pd.Timestamp | str | None = None,
    power_history_days: float | None = None,
    evaluation_pair_id: str | None = None,
    input_snapshot_id: str | None = None,
    expected_input_sha256: str | None = None,
    expected_physical_config_sha256: str | None = None,
    pair_reference: xr.Dataset | None = None,
    state_override: dict[str, object] | None = None,
    forecast_identity: Mapping[str, object] | None = None,
    load_residual_profile: Mapping[str, object] | None = None,
    reference_forecast_archive: xr.Dataset | None = None,
    fixed_soc_bias_corrections_override: dict[str, float] | None = None,
    fixed_legacy_solar_w: pd.Series | None = None,
    issue_snapshot_zarr: Path | None = None,
    site_irradiance_override: pd.Series | None = None,
    site_irradiance_provenance: Mapping[str, object] | None = None,
) -> Path:
    provider = validate_provider(provider)
    selected_solar_model = validate_solar_model(solar_model)
    embedded_site_forcing = site_irradiance_override is not None
    if embedded_site_forcing != (site_irradiance_provenance is not None):
        raise ValueError(
            "site_irradiance_override and site_irradiance_provenance must be supplied together"
        )
    normalised_site_irradiance: pd.Series | None = None
    site_irradiance_digest = ""
    site_provenance: dict[str, object] = {}
    site_cycle_time: pd.Timestamp | None = None
    if embedded_site_forcing:
        if input_forecast is not None or refresh_from_cache:
            raise ValueError(
                "Embedded site irradiance cannot be combined with a GRIB input or cache refresh"
            )
        normalised_site_irradiance = _normalise_site_irradiance_override(
            site_irradiance_override
        )
        site_irradiance_digest = site_irradiance_series_digest(
            normalised_site_irradiance
        )
        site_provenance = {
            str(name): value for name, value in dict(site_irradiance_provenance or {}).items()
        }
        expected_site_digest = str(
            site_provenance.get("site_irradiance_sha256", "")
        ).strip().lower()
        if expected_site_digest != f"sha256:{site_irradiance_digest}":
            raise ValueError("Embedded site irradiance checksum does not match its provenance")
        baseline_source_digest = str(
            site_provenance.get("source_manifest_digest", "")
        ).strip().lower()
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", baseline_source_digest):
            raise ValueError("Embedded site irradiance lacks source-manifest digest provenance")
        source_cycle_set = str(site_provenance.get("source_cycle_set_id", "")).strip()
        if not source_cycle_set or source_cycle_set.lower() in {"nan", "none"}:
            raise ValueError("Embedded site irradiance lacks source-cycle provenance")
        site_cycle_time = pd.to_datetime(
            site_provenance.get("source_cycle_time_utc"), utc=True, errors="coerce"
        )
        if pd.isna(site_cycle_time):
            raise ValueError("Embedded site irradiance lacks a valid source-cycle time")
        site_cycle_time = pd.Timestamp(site_cycle_time).tz_convert("UTC").tz_localize(None)
        if str(input_snapshot_id or "").strip().lower() != f"sha256:{site_irradiance_digest}":
            raise ValueError("input_snapshot_id does not identify the embedded site irradiance")
    if selected_solar_model == PHYSICAL_SOLAR_MODEL_NAME:
        protected = {
            "latest forecast": POWER_SOC_FORECAST_ZARR_PATH,
            "adaptive state": POWER_SOC_FORECAST_STATE_PATH,
            "verification archive": POWER_SOC_FORECAST_ARCHIVE_ZARR_PATH,
            "skill product": POWER_SOC_FORECAST_SKILL_ZARR_PATH,
            "hindcast product": POWER_SOC_HINDCAST_ZARR_PATH,
        }
        requested = {
            "latest forecast": output_zarr,
            "adaptive state": state_path,
            "verification archive": archive_zarr,
            "skill product": skill_zarr,
            "hindcast product": hindcast_zarr,
        }
        overlaps = []
        for requested_label, target in requested.items():
            if target is None:
                continue
            for protected_label, protected_target in protected.items():
                if _paths_overlap(Path(target), Path(protected_target)):
                    overlaps.append(f"{requested_label}->{protected_label}")
        if overlaps:
            raise ValueError(
                "Physical solar candidate refuses baseline artifact paths "
                f"({', '.join(overlaps)}); use generate_power_soc_physical_candidate.py "
                "with an isolated candidate root"
            )
    physical_solar_config = (
        load_physical_solar_config(physical_solar_config_path)
        if selected_solar_model == PHYSICAL_SOLAR_MODEL_NAME
        else None
    )
    if expected_physical_config_sha256 is not None:
        if physical_solar_config is None:
            raise ValueError("expected_physical_config_sha256 requires the physical solar model")
        expected_config_digest = str(expected_physical_config_sha256).strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected_config_digest):
            raise ValueError(
                "expected_physical_config_sha256 must be a 64-character hexadecimal digest"
            )
        if physical_solar_config_digest(physical_solar_config) != expected_config_digest:
            raise ValueError("Physical solar configuration checksum changed before generation")
    retrieval_diagnostics: dict[str, object] = {}
    if input_forecast is None and not embedded_site_forcing:
        if refresh_from_cache:
            input_forecast = _latest_cached_forecast(cache_dir)
        else:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            input_forecast = cache_dir / f"ecmwf_ssrd_{stamp}.grib2"
            selected_cycle_hour = resolve_ecmwf_cycle_hour(ecmwf_cycle_hour)
            input_forecast, retrieval_diagnostics = retrieve_provider_open_data_grib(
                input_forecast,
                provider="earthkit" if provider == "earthkit" else "legacy",
                horizon_hours=horizon_hours,
                lookahead_buffer_hours=DEFAULT_ECMWF_LOOKAHEAD_BUFFER_HOURS,
                param=ECMWF_PARAM,
                source=DEFAULT_OPEN_DATA_SOURCE,
                cycle_hour=selected_cycle_hour,
            )
    power = xr.open_zarr(power_zarr, chunks={})
    selected_power_cutoff: pd.Timestamp | None = None
    selected_power_history_days: float | None = None
    selected_power_start: pd.Timestamp | None = None
    if power_cutoff_time is not None:
        selected_power_cutoff = pd.Timestamp(power_cutoff_time)
        if selected_power_cutoff.tz is not None:
            selected_power_cutoff = selected_power_cutoff.tz_convert("UTC").tz_localize(None)
    if power_history_days is not None:
        try:
            selected_power_history_days = float(power_history_days)
        except (TypeError, ValueError) as exc:
            raise ValueError("power_history_days must be a positive finite number") from exc
        if not np.isfinite(selected_power_history_days) or selected_power_history_days <= 0.0:
            raise ValueError("power_history_days must be a positive finite number")
        if selected_power_cutoff is None:
            raise ValueError("power_history_days requires an explicit power_cutoff_time")
        selected_power_start = selected_power_cutoff - pd.Timedelta(days=selected_power_history_days)
    if selected_power_cutoff is not None:
        power = power.sel(
            time=slice(
                selected_power_start.to_datetime64() if selected_power_start is not None else None,
                selected_power_cutoff.to_datetime64(),
            )
        )
        if power.sizes.get("time", 0) == 0:
            raise ValueError(f"No APS power data exist at or before paired cutoff {selected_power_cutoff}")
    latest_power_time, _ = validate_power_input_freshness(
        power,
        max_age_minutes=max_power_age_minutes,
    )
    pdu = None
    if pdu_zarr.exists():
        try:
            pdu = xr.open_zarr(pdu_zarr, chunks={})
            if selected_power_cutoff is not None and "time" in pdu.coords:
                pdu = pdu.sel(
                    time=slice(
                        selected_power_start.to_datetime64()
                        if selected_power_start is not None
                        else None,
                        selected_power_cutoff.to_datetime64(),
                    )
                )
        except Exception:
            pdu = None
    if embedded_site_forcing:
        assert site_cycle_time is not None
        solar = xr.Dataset(
            coords={"forecast_reference_time": site_cycle_time.to_datetime64()},
            attrs={
                name: site_provenance[name]
                for name in (
                    "selected_grid_latitude",
                    "selected_grid_longitude",
                    "selected_grid_distance_km",
                )
                if name in site_provenance
            },
        )
        provider_diagnostics: dict[str, object] = {
            "requested_provider": provider,
            "effective_provider": str(
                site_provenance.get("ecmwf_provider_effective", provider)
            ),
            "fallback_reason": "",
            "input_representation": "embedded_site_irradiance",
            "site_irradiance_sha256": f"sha256:{site_irradiance_digest}",
            "global_grid_opened": False,
        }
        input_label = "embedded:ECMWFSolarIrradiance"
        input_digest = site_irradiance_digest
        input_digest_available = True
    else:
        assert input_forecast is not None
        provider_result = open_provider_solar_forecast(
            input_forecast,
            provider=provider,
            latitude=latitude,
            longitude=longitude,
            shadow_report_path=shadow_report_path,
        )
        solar = provider_result.dataset
        provider_diagnostics = dict(provider_result.diagnostics)
        input_label = str(input_forecast)
        input_digest_available = input_forecast.is_file()
        input_digest = (
            _sha256_file(input_forecast)
            if input_digest_available
            else hashlib.sha256(f"unavailable:{input_forecast}".encode("utf-8")).hexdigest()
        )
    previous_forecast = None
    if output_zarr.exists():
        try:
            previous_forecast = xr.open_zarr(output_zarr, chunks={})
        except Exception:
            previous_forecast = None
    forecast_archive = None
    if archive_zarr.exists():
        try:
            forecast_archive = xr.open_zarr(archive_zarr, chunks={}).load()
        except Exception:
            forecast_archive = None
    state = dict(state_override) if state_override is not None else _load_state(state_path)
    fixed_soc_bias_corrections = fixed_soc_bias_corrections_override
    if fixed_soc_bias_corrections is None and pair_reference is not None:
        try:
            parsed_corrections = json.loads(
                str(pair_reference.attrs.get("soc_bias_correction_pct_points_by_bucket", "{}"))
            )
            if isinstance(parsed_corrections, dict):
                fixed_soc_bias_corrections = parsed_corrections
        except json.JSONDecodeError:
            fixed_soc_bias_corrections = {}
    calibration_archive = (
        reference_forecast_archive
        if reference_forecast_archive is not None
        else forecast_archive
    )
    forecast = build_forecast_dataset(
        power,
        solar,
        pdu=pdu,
        previous_forecast=previous_forecast,
        forecast_archive=calibration_archive,
        state=state,
        horizon_hours=horizon_hours,
        solar_model=selected_solar_model,
        physical_solar_config=physical_solar_config,
        latitude=latitude,
        longitude=longitude,
        fixed_soc_bias_corrections=fixed_soc_bias_corrections,
        load_residual_profile=load_residual_profile,
        fixed_legacy_solar_w=fixed_legacy_solar_w,
        site_irradiance_override=normalised_site_irradiance,
    )
    forecast.attrs["ecmwf_input_file"] = input_label
    forecast.attrs["solar_input_representation"] = (
        "embedded_site_irradiance" if embedded_site_forcing else "provider_forecast_file"
    )
    forecast.attrs["site_irradiance_sha256"] = (
        f"sha256:{site_irradiance_digest}" if embedded_site_forcing else ""
    )
    forecast.attrs["site_irradiance_source_manifest_digest"] = (
        str(site_provenance.get("source_manifest_digest", ""))
        if embedded_site_forcing
        else ""
    )
    forecast.attrs["site_irradiance_source_cycle_set_id"] = (
        str(site_provenance.get("source_cycle_set_id", ""))
        if embedded_site_forcing
        else ""
    )
    forecast.attrs["site_latitude"] = str(float(latitude))
    forecast.attrs["site_longitude"] = str(float(longitude))
    forecast.attrs["refresh_from_cache"] = str(bool(refresh_from_cache)).lower()
    forecast.attrs["power_input_cutoff_time"] = (
        selected_power_cutoff.isoformat() if selected_power_cutoff is not None else "latest_available"
    )
    forecast.attrs["power_input_history_days"] = (
        f"{selected_power_history_days:.6g}"
        if selected_power_history_days is not None
        else "full_available_history"
    )
    forecast.attrs["evaluation_pair_id"] = str(evaluation_pair_id or "")
    forecast.attrs["input_snapshot_id"] = str(input_snapshot_id or "")
    forecast.attrs["baseline_publication_signature"] = (
        str(pair_reference.attrs.get("publication_signature", ""))
        if pair_reference is not None
        else ""
    )
    # Cached ECMWF refreshes deliberately re-anchor live SOC/load after a mode
    # transition. They remain auditable archive rows but are explicitly marked
    # ineligible for independent skill, hindcast, or adaptive-learning inputs.
    # Archive retention and evaluation eligibility are separate contracts.
    # Cached re-anchors are operationally meaningful audit samples, but they
    # share weather forcing and must never tune or certify the forecast.
    if archive_forecast is None:
        archive_forecast = True
    independent_cycle = not refresh_from_cache
    if issue_snapshot_zarr is not None:
        issue_snapshot_zarr = Path(issue_snapshot_zarr)
        if not archive_forecast or not independent_cycle:
            raise ValueError(
                "Issue snapshots require an archive-eligible full ECMWF-cycle forecast"
            )
        protected_outputs = tuple(
            Path(path)
            for path in (output_zarr, archive_zarr, skill_zarr, hindcast_zarr, state_path)
            if path is not None
        )
        if any(_paths_overlap(issue_snapshot_zarr, path) for path in protected_outputs):
            raise ValueError("Issue snapshot path overlaps a mutable forecast product")
    input_age_minutes = float(
        (pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None) - latest_power_time)
        / pd.Timedelta(minutes=1)
    )
    forecast.attrs["forecast_refresh_kind"] = (
        "ecmwf_cycle" if independent_cycle else "cached_reanchor"
    )
    forecast.attrs["forecast_verification_eligible"] = str(independent_cycle).lower()
    forecast.attrs["independent_cycle"] = str(independent_cycle).lower()
    forecast.attrs["input_power_age_minutes"] = f"{max(input_age_minutes, 0.0):.6g}"
    forecast.attrs["ecmwf_provider_requested"] = provider
    forecast.attrs["ecmwf_provider_effective"] = str(provider_diagnostics["effective_provider"])
    forecast.attrs["ecmwf_provider_fallback_reason"] = str(provider_diagnostics.get("fallback_reason", ""))
    forecast.attrs["ecmwf_provider_diagnostics"] = json.dumps(
        {**retrieval_diagnostics, **provider_diagnostics}, sort_keys=True
    )
    for name in ("selected_grid_latitude", "selected_grid_longitude", "selected_grid_distance_km"):
        if name in solar.attrs:
            forecast.attrs[name] = str(solar.attrs[name])
    cycle_time = _ecmwf_cycle_time(solar)
    if cycle_time is not None:
        forecast.attrs["ecmwf_cycle_time"] = cycle_time.isoformat()
    effective_provider = str(provider_diagnostics["effective_provider"])
    cycle_text = cycle_time.isoformat() if cycle_time is not None else "unknown"
    stable_feature_payload = {
        "schema": 1,
        "solar_feature_set_version": str(
            forecast.attrs.get("solar_feature_set_version", "")
        ),
        "load_model": str(forecast.attrs.get("load_model", "")),
        "load_model_version": str(forecast.attrs.get("load_model_version", "")),
        "load_state_contract": str(forecast.attrs.get("load_state_contract", "")),
        "battery_energy_model": "bounded_bidirectional_efficiency_v1",
    }
    default_identity = {
        "forecast_system_version": f"power-v{forecast.attrs.get('forecast_model_version', 'unknown')}",
        "feature_set_version": "+".join(
            value
            for value in (
                str(forecast.attrs.get("solar_feature_set_version", "")),
                str(forecast.attrs.get("load_state_contract", "")),
            )
            if value
        ),
        "feature_set_digest": hashlib.sha256(
            json.dumps(stable_feature_payload, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "training_cutoff_utc": latest_power_time.isoformat(),
        "observation_cutoff_utc": latest_power_time.isoformat(),
        "soc_anchor_time_utc": str(forecast.attrs.get("initial_soc_time", "")),
        "forecast_code_revision": os.environ.get(
            "AURORA_FORECAST_CODE_REVISION", "unversioned"
        ).strip()
        or "unversioned",
        "source_cycle_set_id": (
            str(site_provenance["source_cycle_set_id"])
            if embedded_site_forcing
            else f"ecmwf:{effective_provider}:{cycle_text}:sha256:{input_digest[:20]}"
        ),
        "source_manifest_digest": (
            str(site_provenance["source_manifest_digest"])
            if embedded_site_forcing
            else f"sha256:{input_digest}"
        ),
        "degraded_mode_code": "+".join(
            (
                *(
                    ("ecmwf_provider_fallback",)
                    if str(provider_diagnostics.get("fallback_reason", "")).strip()
                    else ()
                ),
                *(("source_checksum_unavailable",) if not input_digest_available else ()),
            )
        )
        or "none",
        "adaptive_calibration_state_id": str(
            forecast.attrs.get("adaptive_calibration_state_id", "")
        ),
    }
    if forecast_identity is not None:
        default_identity.update(
            {str(name): value for name, value in forecast_identity.items()}
        )
    apply_forecast_identity(forecast, default_identity)
    forecast.attrs["publication_signature"] = forecast_publication_signature(forecast)
    if pair_reference is not None:
        if not forecast.attrs["evaluation_pair_id"] or not forecast.attrs["input_snapshot_id"]:
            raise ValueError("Paired physical candidate requires evaluation_pair_id and input_snapshot_id")
        validate_paired_candidate(forecast, pair_reference)
    if expected_input_sha256 is not None:
        if embedded_site_forcing:
            raise ValueError("expected_input_sha256 is valid only for a file-backed forecast input")
        expected_digest = str(expected_input_sha256).strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected_digest):
            raise ValueError("expected_input_sha256 must be a 64-character hexadecimal digest")
        actual_digest = _sha256_file(input_forecast)
        if actual_digest != expected_digest:
            raise ValueError("ECMWF input checksum changed before candidate publication")
        if forecast.attrs["input_snapshot_id"] != f"sha256:{expected_digest}":
            raise ValueError("input_snapshot_id does not match expected_input_sha256")
    unchanged_publication = bool(
        previous_forecast is not None
        and previous_forecast.attrs.get("publication_signature") == forecast.attrs["publication_signature"]
    )
    if not unchanged_publication:
        _atomic_write_zarr(forecast, output_zarr)
    if issue_snapshot_zarr is not None:
        write_immutable_issue_snapshot(forecast, issue_snapshot_zarr)
    if archive_forecast:
        updated_archive = (
            forecast_archive
            if unchanged_publication and forecast_archive is not None
            else append_forecast_archive(forecast, archive_zarr)
        )
        if skill_zarr is not None:
            skill = build_forecast_skill_dataset(updated_archive, power)
            _atomic_write_skill(skill, skill_zarr)
        if hindcast_zarr is not None:
            hindcast = build_soc_hindcast_dataset(updated_archive, power)
            _atomic_write_time_product(hindcast, hindcast_zarr)
    next_state = dict(state)
    next_state.update(
        {
            "updated_at_utc": _utc_now(),
            "solar_calibration_factor_w_per_wm2": float(forecast.attrs["solar_calibration_factor_w_per_wm2"]),
            "forecast_load_w": float(forecast.attrs["forecast_load_w"]),
            "load_bias_correction_w": float(forecast.attrs["load_bias_correction_w"]),
            "load_model": forecast.attrs["load_model"],
            "load_model_version": int(forecast.attrs["load_model_version"]),
            "load_regime": forecast.attrs["load_regime"],
            "load_regime_threshold_w": float(forecast.attrs["load_regime_threshold_w"]),
            "load_regime_level_w": float(forecast.attrs["load_regime_level_w"]),
            "load_regime_run_hours": float(forecast.attrs["load_regime_run_hours"]),
            "load_regime_sample_count": int(forecast.attrs["load_regime_sample_count"]),
            "current_load_mode": forecast.attrs["load_mode"],
            "current_load_mode_source": forecast.attrs["load_mode_source"],
            "current_load_mode_signature": forecast.attrs["load_mode_signature"],
            "current_load_mode_learning_ready": forecast.attrs["load_mode_learning_ready"] == "true",
            "current_load_mode_learning_reason": forecast.attrs["load_mode_learning_reason"],
            "minimum_operational_soc_pct": float(forecast.attrs["minimum_operational_soc_pct"]),
            "load_measurement": forecast.attrs["load_measurement"],
            "load_balance_measurement": forecast.attrs["load_balance_measurement"],
            "load_mode_registry": json.loads(forecast.attrs["load_mode_registry"]),
            "battery_model": {
                name: forecast.attrs[name]
                for name in (
                    "battery_usable_capacity_kwh",
                    "battery_charge_efficiency",
                    "battery_discharge_efficiency",
                    "battery_parasitic_load_w",
                    "battery_max_charge_w",
                    "battery_max_discharge_w",
                    "battery_calibration_sample_count",
                    "battery_calibration_confidence",
                )
            },
            "soc_bias_correction_pct_points_by_bucket": json.loads(
                forecast.attrs["soc_bias_correction_pct_points_by_bucket"]
            ),
            "latest_metrics": json.loads(forecast.attrs["previous_forecast_metrics"]),
            "latest_ecmwf_input_file": input_label,
            "latest_refresh_from_cache": bool(refresh_from_cache),
            "latest_forecast_refresh_kind": forecast.attrs["forecast_refresh_kind"],
            "latest_forecast_verification_eligible": bool(independent_cycle),
            "latest_independent_cycle": bool(independent_cycle),
            "latest_input_power_time": latest_power_time.isoformat(),
            "latest_input_power_age_minutes": max(input_age_minutes, 0.0),
            "latest_ecmwf_provider_requested": provider,
            "latest_ecmwf_provider_effective": provider_diagnostics["effective_provider"],
            "latest_ecmwf_provider_fallback_reason": provider_diagnostics.get("fallback_reason", ""),
            "solar_model_name": forecast.attrs["solar_model_name"],
            "solar_model_version": int(forecast.attrs["solar_model_version"]),
            "solar_model_contract_id": forecast.attrs["solar_model_contract_id"],
            "solar_model_status": forecast.attrs["solar_model_status"],
            "solar_degradation_codes": forecast.attrs["solar_degradation_codes"],
            "solar_physical_config_sha256": forecast.attrs.get("solar_physical_config_sha256", ""),
            "forecast_model_name": forecast.attrs["forecast_model_name"],
            "forecast_model_version": int(forecast.attrs["forecast_model_version"]),
            "forecast_model_contract_id": forecast.attrs["forecast_model_contract_id"],
            "evaluation_pair_id": forecast.attrs["evaluation_pair_id"],
            "input_snapshot_id": forecast.attrs["input_snapshot_id"],
        }
    )
    _write_state(state_path, next_state)
    if unchanged_publication:
        print(f"Skipped unchanged forecast publication for {output_zarr}")
    else:
        print(f"Wrote {output_zarr} with {forecast.sizes.get('time', 0)} forecast samples")
    return output_zarr


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the ECMWF-informed APS SOC forecast Zarr")
    parser.add_argument("--power-zarr", type=Path, default=POWER_ZARR_PATH)
    parser.add_argument("--pdu-zarr", type=Path, default=POWER_PDU_ZARR_PATH)
    parser.add_argument("--output-zarr", type=Path, default=POWER_SOC_FORECAST_ZARR_PATH)
    parser.add_argument("--input-forecast", type=Path, help="Existing ECMWF GRIB/NetCDF file to use instead of downloading")
    parser.add_argument("--cache-dir", type=Path, default=POWER_ECMWF_FORECAST_CACHE_DIR)
    parser.add_argument("--state", type=Path, default=POWER_SOC_FORECAST_STATE_PATH)
    parser.add_argument("--archive-zarr", type=Path, default=POWER_SOC_FORECAST_ARCHIVE_ZARR_PATH)
    parser.add_argument("--skill-zarr", type=Path, default=POWER_SOC_FORECAST_SKILL_ZARR_PATH)
    parser.add_argument("--hindcast-zarr", type=Path, default=POWER_SOC_HINDCAST_ZARR_PATH)
    parser.add_argument("--no-skill-zarr", action="store_true", help="Do not write the forecast verification skill Zarr")
    parser.add_argument("--no-hindcast-zarr", action="store_true", help="Do not write the fixed-lead SOC hindcast Zarr")
    parser.add_argument("--latitude", type=float, default=DEFAULT_LATITUDE)
    parser.add_argument("--longitude", type=float, default=DEFAULT_LONGITUDE)
    parser.add_argument("--horizon-hours", type=int, default=DEFAULT_HORIZON_HOURS)
    parser.add_argument(
        "--ecmwf-cycle-hour",
        choices=("auto", "0", "12"),
        help="Select a 00 or 12 UTC long-range deterministic cycle instead of the latest cycle",
    )
    parser.add_argument(
        "--refresh-from-cache",
        action="store_true",
        help="Reuse the latest cached ECMWF GRIB instead of downloading a new forecast",
    )
    parser.add_argument(
        "--no-archive-forecast",
        action="store_true",
        help="Publish without retaining an operational audit row in the forecast archive",
    )
    parser.add_argument(
        "--max-power-age-minutes",
        type=float,
        default=DEFAULT_MAX_POWER_INPUT_AGE_MINUTES,
        help="Reject publication when the latest BatterySOC/load sample is older than this limit",
    )
    parser.add_argument(
        "--provider",
        choices=("legacy", "earthkit", "shadow"),
        default=DEFAULT_PROVIDER,
        help="ECMWF retrieval/decoding provider; shadow compares Earthkit while publishing legacy output",
    )
    parser.add_argument(
        "--solar-model",
        choices=VALID_SOLAR_MODELS,
        default=DEFAULT_SOLAR_MODEL,
        help="Solar-to-electrical-power model; the operational default remains the scalar v10 baseline",
    )
    parser.add_argument(
        "--solar-physical-config",
        type=Path,
        default=DEFAULT_PHYSICAL_SOLAR_CONFIG_PATH,
        help="Versioned array geometry/electrical config used only by the physical candidate",
    )
    parser.add_argument("--shadow-report", type=Path, default=DEFAULT_SHADOW_REPORT_PATH)
    parser.add_argument(
        "--issue-snapshot-zarr",
        type=Path,
        help=(
            "Write this full archive-eligible issue once at a cycle/signature-specific "
            "path for downstream paired evaluation"
        ),
    )
    args = parser.parse_args()
    generate(
        power_zarr=args.power_zarr,
        pdu_zarr=args.pdu_zarr,
        output_zarr=args.output_zarr,
        input_forecast=args.input_forecast,
        cache_dir=args.cache_dir,
        state_path=args.state,
        archive_zarr=args.archive_zarr,
        skill_zarr=None if args.no_skill_zarr else args.skill_zarr,
        hindcast_zarr=None if args.no_hindcast_zarr else args.hindcast_zarr,
        latitude=args.latitude,
        longitude=args.longitude,
        horizon_hours=args.horizon_hours,
        refresh_from_cache=args.refresh_from_cache,
        provider=args.provider,
        shadow_report_path=args.shadow_report,
        ecmwf_cycle_hour=args.ecmwf_cycle_hour,
        max_power_age_minutes=args.max_power_age_minutes,
        archive_forecast=False if args.no_archive_forecast else None,
        solar_model=args.solar_model,
        physical_solar_config_path=args.solar_physical_config,
        issue_snapshot_zarr=args.issue_snapshot_zarr,
    )


if __name__ == "__main__":
    main()
