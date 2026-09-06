"""Isolated member-wise uncertainty evaluation for the v12 power candidate.

The operational ECMWF ensemble remains read-only.  This module makes a
candidate-only B/C/D ensemble from those same site-extracted members, writes
only under a candidate lane, and evaluates immutable baseline/candidate pair
bundles.  It deliberately does not retrieve global grids, pool public-model
sources, or update any operational adaptive state.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Mapping

import numpy as np
import pandas as pd
import xarray as xr

from generate_power_soc_ensemble import _crps_ensemble, apply_operational_soc_threshold
from generate_power_soc_forecast import (
    _aggregate_physical_soc_forecast,
    _atomic_write_zarr,
    _write_state,
    integrate_soc_forecast,
    integrate_soc_from_available_solar,
)
from power_battery_model import BatteryModel
from power_solar_model import PhysicalSolarConfig, build_physical_solar_forecast_frames
from power_soc_thresholds import MINIMUM_OPERATIONAL_SOC_PCT
from power_v12_hybrid import (
    LEAD_BUCKETS,
    PAIR_ARTIFACT_DIGEST_ALGORITHM,
    immutable_artifact_record,
    pair_artifacts_valid,
    stable_json_digest,
    utc_now_iso,
)


CANDIDATE_ENSEMBLE_VERSION = "memberwise_physical_pv_load_delta_v1"
MIN_RESERVE_EVENTS = 10
ENSEMBLE_EVALUATION_CONTRACT_ATTRS = (
    "forecast_model_contract_id",
    "forecast_system_version",
    "feature_set_version",
    "feature_set_digest",
    "forecast_code_revision",
    "candidate_lane",
    "baseline_control_contract_id",
    "baseline_control_system_version",
    "local_feature_contract_id",
    "candidate_ensemble_contract_id",
    "candidate_ensemble_version",
)
_MEMBER_SERIES = (
    "ECMWFSolarIrradianceEnsemble",
    "ForecastSolarWattsEnsemble",
    "ForecastLoadWattsEnsemble",
)
_BATTERY_MEMBER_FIELDS = (
    "BatteryUsableCapacityKWhEnsemble",
    "BatteryChargeEfficiencyEnsemble",
    "BatteryDischargeEfficiencyEnsemble",
)


def _utc_naive(value: object) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    if parsed.tz is not None:
        return parsed.tz_convert("UTC").tz_localize(None)
    return parsed


def _text(value: object, fallback: str = "") -> str:
    return str(value if value is not None else fallback).strip() or fallback


def _time_index(dataset: xr.Dataset) -> pd.DatetimeIndex:
    if "time" not in dataset.coords:
        raise ValueError("Ensemble product is missing a time coordinate")
    times = pd.DatetimeIndex(dataset["time"].values)
    if times.tz is not None:
        times = times.tz_convert("UTC").tz_localize(None)
    if len(times) < 2 or times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError("Ensemble time grid must be strictly increasing with at least two rows")
    return times


def _member_values(dataset: xr.Dataset) -> np.ndarray:
    if "member" not in dataset.coords:
        raise ValueError("Ensemble product is missing member coordinates")
    members = np.asarray(dataset["member"].values)
    if len(members) < 2 or len(np.unique(members)) != len(members):
        raise ValueError("Ensemble product needs at least two unique members")
    return members


def _array_digest(value: np.ndarray) -> bytes:
    array = np.asarray(value)
    header = json.dumps(
        {"dtype": str(array.dtype), "shape": list(array.shape)}, sort_keys=True
    ).encode("utf-8")
    if array.dtype.kind == "O":
        body = json.dumps(array.astype(str).tolist(), separators=(",", ":")).encode("utf-8")
    else:
        body = np.ascontiguousarray(array).tobytes()
    return header + body


def baseline_ensemble_signature(dataset: xr.Dataset) -> str:
    """Return a compact content signature for the immutable baseline input.

    The candidate uses a fully loaded, site-only ensemble.  Hashing the fields
    actually propagated into SOC prevents a late refresh or a different member
    set being silently represented as the same paired baseline.
    """

    required = ("BatterySOCForecastEnsemble", *_MEMBER_SERIES, *_BATTERY_MEMBER_FIELDS)
    missing = [name for name in required if name not in dataset]
    if missing:
        raise ValueError("Baseline ensemble is missing: " + ", ".join(missing))
    digest = hashlib.sha256()
    digest.update(CANDIDATE_ENSEMBLE_VERSION.encode("utf-8"))
    for name in sorted(required):
        digest.update(name.encode("utf-8"))
        digest.update(_array_digest(np.asarray(dataset[name].values)))
    for name in ("initial_soc_time", "initial_soc_pct", "ecmwf_cycle_time", "forecast_horizon_hours"):
        digest.update(name.encode("utf-8"))
        digest.update(_text(dataset.attrs.get(name)).encode("utf-8"))
    return "baseline-ensemble-v1-" + digest.hexdigest()[:20]


def _same_timestamp(left: object, right: object, *, name: str) -> None:
    try:
        left_time = _utc_naive(left)
        right_time = _utc_naive(right)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unable to parse paired {name}") from exc
    if left_time != right_time:
        raise ValueError(f"Candidate ensemble {name} does not match the baseline pair")


def _finite_attr(dataset: xr.Dataset, name: str) -> float:
    try:
        value = float(dataset.attrs[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Ensemble product is missing finite {name}") from exc
    if not np.isfinite(value):
        raise ValueError(f"Ensemble product has non-finite {name}")
    return value


def validate_memberwise_candidate_inputs(
    baseline: xr.Dataset,
    candidate: xr.Dataset,
    baseline_ensemble: xr.Dataset,
) -> tuple[pd.DatetimeIndex, np.ndarray, pd.Timestamp, float]:
    """Fail closed unless the deterministic and member-wise pair is exact."""

    baseline_times = _time_index(baseline)
    candidate_times = _time_index(candidate)
    ensemble_times = _time_index(baseline_ensemble)
    if not np.array_equal(candidate_times.values, baseline_times.values):
        raise ValueError("Candidate deterministic time grid does not match baseline")
    if not np.array_equal(ensemble_times.values, baseline_times.values):
        raise ValueError("Baseline ensemble time grid does not match deterministic pair")
    for source in (baseline, candidate, baseline_ensemble):
        _member_values(baseline_ensemble)
        _finite_attr(source, "initial_soc_pct")
    _same_timestamp(
        baseline.attrs.get("initial_soc_time"),
        candidate.attrs.get("initial_soc_time"),
        name="SOC anchor",
    )
    _same_timestamp(
        baseline.attrs.get("initial_soc_time"),
        baseline_ensemble.attrs.get("initial_soc_time"),
        name="ensemble SOC anchor",
    )
    _same_timestamp(
        baseline.attrs.get("ecmwf_cycle_time"),
        candidate.attrs.get("ecmwf_cycle_time"),
        name="ECMWF cycle",
    )
    _same_timestamp(
        baseline.attrs.get("ecmwf_cycle_time"),
        baseline_ensemble.attrs.get("ecmwf_cycle_time"),
        name="ensemble ECMWF cycle",
    )
    initial_soc = _finite_attr(candidate, "initial_soc_pct")
    if not np.isclose(initial_soc, _finite_attr(baseline, "initial_soc_pct"), atol=1.0e-6):
        raise ValueError("Candidate SOC anchor differs from baseline")
    if not np.isclose(initial_soc, _finite_attr(baseline_ensemble, "initial_soc_pct"), atol=1.0e-6):
        raise ValueError("Candidate SOC anchor differs from baseline ensemble")
    for name in _MEMBER_SERIES:
        value = baseline_ensemble[name]
        if value.dims != ("member", "time"):
            raise ValueError(f"Baseline ensemble {name} must use (member, time)")
        values = np.asarray(value.values, dtype=np.float64)
        finite = np.isfinite(values)
        # The right-endpoint convention permits an undefined source forcing at
        # the SOC anchor itself.  It has no future interval duration and is
        # normalised to a harmless zero only when physical PV is reconstructed.
        allowed_anchor_gap = (
            name in {"ECMWFSolarIrradianceEnsemble", "ForecastSolarWattsEnsemble"}
            and np.all(finite[:, 1:])
        )
        if not np.all(finite) and not allowed_anchor_gap:
            raise ValueError(f"Baseline ensemble {name} has non-finite values")
    for name in _BATTERY_MEMBER_FIELDS:
        value = baseline_ensemble[name]
        if value.dims != ("member",):
            raise ValueError(f"Baseline ensemble {name} must use (member,)")
        if not np.all(np.isfinite(np.asarray(value.values, dtype=np.float64))):
            raise ValueError(f"Baseline ensemble {name} has non-finite values")
    for name in ("ForecastLoadWatts", "ForecastSolarWatts", "BatterySOCForecast"):
        if name not in baseline or name not in candidate:
            raise ValueError(f"Deterministic pair is missing {name}")
    return baseline_times, _member_values(baseline_ensemble), _utc_naive(candidate.attrs["initial_soc_time"]), initial_soc


def _validate_memberwise_ensemble_pair(
    baseline: xr.Dataset,
    candidate: xr.Dataset,
) -> tuple[pd.DatetimeIndex, np.ndarray, pd.Timestamp, float]:
    """Validate an already-materialised ensemble pair for evidence extraction."""

    baseline_times = _time_index(baseline)
    candidate_times = _time_index(candidate)
    if not np.array_equal(baseline_times.values, candidate_times.values):
        raise ValueError("Candidate ensemble time grid does not match baseline ensemble")
    members = _member_values(baseline)
    if not np.array_equal(members, _member_values(candidate)):
        raise ValueError("Candidate ensemble member set does not match baseline ensemble")
    _same_timestamp(
        baseline.attrs.get("initial_soc_time"),
        candidate.attrs.get("initial_soc_time"),
        name="ensemble SOC anchor",
    )
    _same_timestamp(
        baseline.attrs.get("ecmwf_cycle_time"),
        candidate.attrs.get("ecmwf_cycle_time"),
        name="ensemble ECMWF cycle",
    )
    initial_soc = _finite_attr(candidate, "initial_soc_pct")
    if not np.isclose(initial_soc, _finite_attr(baseline, "initial_soc_pct"), atol=1.0e-6):
        raise ValueError("Candidate ensemble SOC anchor differs from baseline ensemble")
    for product in (baseline, candidate):
        value = product.get("BatterySOCForecastEnsemble")
        if value is None or value.dims != ("member", "time"):
            raise ValueError("Ensemble pair is missing member-wise SOC")
        if not np.all(np.isfinite(np.asarray(value.values, dtype=np.float64))):
            raise ValueError("Ensemble pair has non-finite member-wise SOC")
    return baseline_times, members, _utc_naive(candidate.attrs["initial_soc_time"]), initial_soc


def _lane_modes(lane: str) -> tuple[bool, bool]:
    physical_solar = lane in {"B_physical_solar", "D_physical_solar_load_residual"}
    load_residual = lane in {"C_load_residual", "D_physical_solar_load_residual"}
    if lane not in {"B_physical_solar", "C_load_residual", "D_physical_solar_load_residual"}:
        raise ValueError(f"Unsupported v12 candidate lane: {lane}")
    return physical_solar, load_residual


def _member_scalar(dataset: xr.Dataset, name: str, index: int, fallback: float) -> float:
    if name not in dataset:
        return float(fallback)
    try:
        value = float(np.asarray(dataset[name].values, dtype=np.float64)[index])
    except (IndexError, TypeError, ValueError):
        return float(fallback)
    return value if np.isfinite(value) else float(fallback)


def candidate_ensemble_contract_id(
    candidate: xr.Dataset,
    baseline_ensemble: xr.Dataset,
) -> str:
    """Return the complete semantic contract for one candidate ensemble lane."""

    payload = {
        "schema": 1,
        "algorithm": CANDIDATE_ENSEMBLE_VERSION,
        "candidate_forecast_model_contract_id": _text(candidate.attrs.get("forecast_model_contract_id")),
        "candidate_forecast_identity_id": _text(candidate.attrs.get("forecast_identity_id")),
        "candidate_lane": _text(candidate.attrs.get("candidate_lane")),
        "solar_model_contract_id": _text(candidate.attrs.get("solar_model_contract_id")),
        "physical_config": _text(candidate.attrs.get("solar_physical_config_sha256")),
        "battery_contract": {
            name: _text(candidate.attrs.get(name))
            for name in (
                "battery_energy_model",
                "battery_parasitic_load_w",
                "battery_max_charge_w",
                "battery_max_discharge_w",
            )
        },
        "baseline_ensemble_signature": baseline_ensemble_signature(baseline_ensemble),
        "member_load_policy": "baseline_member_spread_plus_candidate_deterministic_delta",
    }
    return "candidate-ensemble-v1-" + stable_json_digest(payload)[:20]


def _series(dataset: xr.Dataset, name: str, times: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(np.asarray(dataset[name].values, dtype=np.float64), index=times)


def _member_series(
    dataset: xr.Dataset,
    name: str,
    index: int,
    times: pd.DatetimeIndex,
    *,
    normalise_anchor_gap: bool = False,
) -> pd.Series:
    values = np.asarray(dataset[name].isel(member=index).values, dtype=np.float64)
    if (
        normalise_anchor_gap
        and name in {"ECMWFSolarIrradianceEnsemble", "ForecastSolarWattsEnsemble"}
        and len(values)
        and not np.isfinite(values[0])
    ):
        values = values.copy()
        values[0] = 0.0
    return pd.Series(values, index=times)


def _candidate_member_load(
    baseline: xr.Dataset,
    candidate: xr.Dataset,
    baseline_ensemble: xr.Dataset,
    *,
    member_index: int,
    times: pd.DatetimeIndex,
    apply_load_residual: bool,
) -> pd.Series:
    base = _member_series(baseline_ensemble, "ForecastLoadWattsEnsemble", member_index, times)
    if not apply_load_residual:
        return base.clip(lower=0.0)
    delta = _series(candidate, "ForecastLoadWatts", times) - _series(baseline, "ForecastLoadWatts", times)
    try:
        floor = float(candidate.attrs.get("load_residual_physical_floor_w", 0.0))
    except (TypeError, ValueError):
        floor = 0.0
    if not np.isfinite(floor):
        floor = 0.0
    return (base + delta).clip(lower=max(floor, 0.0))


def _member_battery_model(
    candidate: xr.Dataset,
    baseline_ensemble: xr.Dataset,
    member_index: int,
) -> BatteryModel:
    template = BatteryModel.from_attrs(candidate.attrs)
    return replace(
        template,
        usable_capacity_kwh=_member_scalar(
            baseline_ensemble,
            "BatteryUsableCapacityKWhEnsemble",
            member_index,
            template.usable_capacity_kwh,
        ),
        charge_efficiency=_member_scalar(
            baseline_ensemble,
            "BatteryChargeEfficiencyEnsemble",
            member_index,
            template.charge_efficiency,
        ),
        discharge_efficiency=_member_scalar(
            baseline_ensemble,
            "BatteryDischargeEfficiencyEnsemble",
            member_index,
            template.discharge_efficiency,
        ),
    ).validated()


def _physical_member_result(
    irradiance: pd.Series,
    load_w: pd.Series,
    *,
    initial_soc: float,
    initial_time: pd.Timestamp,
    output_times: pd.DatetimeIndex,
    battery_model: BatteryModel,
    config: PhysicalSolarConfig,
    latitude: float,
    longitude: float,
) -> pd.DataFrame:
    intervals, substeps, _ = build_physical_solar_forecast_frames(
        irradiance,
        latitude=latitude,
        longitude=longitude,
        config=config,
        forecast_start_time=initial_time,
    )
    if intervals.empty or substeps.empty:
        raise ValueError("Physical member solar forcing has no future substeps")
    integrated = integrate_soc_from_available_solar(
        initial_soc=initial_soc,
        initial_time=initial_time,
        available_solar_w=substeps["ForecastPVAvailableWatts"],
        irradiance=substeps["ECMWFSolarIrradiance"],
        load_w=load_w,
        capacity_kwh=battery_model.usable_capacity_kwh,
        battery_model=battery_model,
    )
    result = _aggregate_physical_soc_forecast(integrated, intervals, irradiance, output_times)
    result = result.reindex(output_times)
    if result["BatterySOCForecast"].isna().any():
        raise ValueError("Physical member SOC does not cover every paired output time")
    return result


def _legacy_member_result(
    irradiance: pd.Series,
    solar_w: pd.Series,
    load_w: pd.Series,
    *,
    initial_soc: float,
    initial_time: pd.Timestamp,
    battery_model: BatteryModel,
) -> pd.DataFrame:
    result = integrate_soc_forecast(
        initial_soc=initial_soc,
        initial_time=initial_time,
        irradiance=irradiance,
        solar_factor=1.0,
        fixed_solar_w=solar_w,
        load_w=load_w,
        capacity_kwh=battery_model.usable_capacity_kwh,
        battery_model=battery_model,
    )
    if result["BatterySOCForecast"].isna().any():
        raise ValueError("Legacy member SOC does not cover every paired output time")
    return result


def _candidate_ensemble_publication_signature(
    candidate: xr.Dataset,
    baseline_ensemble: xr.Dataset,
    contract_id: str,
) -> str:
    payload = {
        "schema": 1,
        "candidate_ensemble_contract_id": contract_id,
        "candidate_forecast_publication_signature": _text(
            candidate.attrs.get("publication_signature")
        ),
        "candidate_forecast_identity_id": _text(candidate.attrs.get("forecast_identity_id")),
        "baseline_ensemble_signature": baseline_ensemble_signature(baseline_ensemble),
        "initial_soc_time": _text(candidate.attrs.get("initial_soc_time")),
        "source_cycle_set_id": _text(candidate.attrs.get("source_cycle_set_id")),
    }
    return "candidate-ensemble-publication-v1-" + stable_json_digest(payload)[:20]


def _validate_member_soc_physics(
    soc: np.ndarray,
    solar: np.ndarray,
    load: np.ndarray,
    charge: np.ndarray,
    discharge: np.ndarray,
    parasitic_load_w: np.ndarray,
) -> None:
    """Validate every independently integrated member before publication."""

    if soc.ndim != 2 or any(
        values.shape != soc.shape for values in (solar, load, charge, discharge)
    ):
        raise ValueError("Candidate member physics arrays do not share member/time shape")
    parasitic = np.asarray(parasitic_load_w, dtype=np.float64).reshape(-1)
    if parasitic.shape != (soc.shape[0],) or not np.isfinite(parasitic).all() or np.any(parasitic < 0.0):
        raise ValueError("Candidate member parasitic loads are invalid")
    parasitic = parasitic[:, None]
    if not np.isfinite(soc).all() or np.any(soc < -1.0e-5) or np.any(soc > 100.0 + 1.0e-5):
        raise ValueError("Candidate member SOC is non-finite or outside physical bounds")
    rises = np.diff(soc, axis=1) > 1.0e-5
    falls = np.diff(soc, axis=1) < -1.0e-5
    flow_known = np.isfinite(charge[:, 1:]) & np.isfinite(discharge[:, 1:])
    net_charge = (
        (flow_known & (charge[:, 1:] > discharge[:, 1:] + 1.0e-5))
        | (
            ~flow_known
            & np.isfinite(solar[:, 1:])
            & np.isfinite(load[:, 1:])
            & (solar[:, 1:] > load[:, 1:] + parasitic + 1.0e-5)
        )
    )
    net_discharge = (
        (flow_known & (discharge[:, 1:] > charge[:, 1:] + 1.0e-5))
        | (
            ~flow_known
            &
            np.isfinite(solar[:, 1:])
            & np.isfinite(load[:, 1:])
            & (solar[:, 1:] + 1.0e-5 < load[:, 1:] + parasitic)
        )
    )
    direction_known = flow_known | (
        np.isfinite(solar[:, 1:]) & np.isfinite(load[:, 1:])
    )
    if np.any(rises & direction_known & ~net_charge):
        raise ValueError("Candidate member SOC rises without net charging")
    if np.any(falls & direction_known & ~net_discharge):
        raise ValueError("Candidate member SOC falls without net discharging")


def build_candidate_memberwise_ensemble(
    baseline: xr.Dataset,
    candidate: xr.Dataset,
    baseline_ensemble: xr.Dataset,
    *,
    lane: str,
    physical_config: PhysicalSolarConfig,
    latitude: float,
    longitude: float,
) -> xr.Dataset:
    """Propagate the exact baseline members through one isolated candidate lane."""

    times, members, issue_time, initial_soc = validate_memberwise_candidate_inputs(
        baseline, candidate, baseline_ensemble
    )
    physical_solar, load_residual = _lane_modes(lane)
    contract_id = candidate_ensemble_contract_id(candidate, baseline_ensemble)
    soc_rows: list[np.ndarray] = []
    irradiance_rows: list[np.ndarray] = []
    solar_rows: list[np.ndarray] = []
    load_rows: list[np.ndarray] = []
    phase_rows: list[np.ndarray] = []
    available_rows: list[np.ndarray] = []
    delivered_rows: list[np.ndarray] = []
    curtailed_rows: list[np.ndarray] = []
    charge_rows: list[np.ndarray] = []
    discharge_rows: list[np.ndarray] = []
    capacities: list[float] = []
    charge_efficiencies: list[float] = []
    discharge_efficiencies: list[float] = []
    parasitic_loads: list[float] = []
    phase_source = baseline_ensemble.get("ForecastLoadPhaseCodeEnsemble")

    for member_index, _member in enumerate(members):
        irradiance = _member_series(
            baseline_ensemble,
            "ECMWFSolarIrradianceEnsemble",
            member_index,
            times,
            normalise_anchor_gap=physical_solar,
        )
        solar = _member_series(baseline_ensemble, "ForecastSolarWattsEnsemble", member_index, times)
        load = _candidate_member_load(
            baseline,
            candidate,
            baseline_ensemble,
            member_index=member_index,
            times=times,
            apply_load_residual=load_residual,
        )
        model = _member_battery_model(candidate, baseline_ensemble, member_index)
        result = (
            _physical_member_result(
                irradiance,
                load,
                initial_soc=initial_soc,
                initial_time=issue_time,
                output_times=times,
                battery_model=model,
                config=physical_config,
                latitude=latitude,
                longitude=longitude,
            )
            if physical_solar
            else _legacy_member_result(
                irradiance,
                solar,
                load,
                initial_soc=initial_soc,
                initial_time=issue_time,
                battery_model=model,
            )
        )
        result = result.reindex(times)
        soc_rows.append(result["BatterySOCForecast"].to_numpy(dtype=np.float32))
        irradiance_rows.append(result["ECMWFSolarIrradiance"].to_numpy(dtype=np.float32))
        solar_rows.append(result["ForecastSolarWatts"].to_numpy(dtype=np.float32))
        load_rows.append(result["ForecastLoadWatts"].to_numpy(dtype=np.float32))
        if phase_source is not None and phase_source.dims == ("member", "time"):
            phase_rows.append(np.asarray(phase_source.isel(member=member_index).values, dtype=np.int8))
        else:
            phase_rows.append(np.zeros(len(times), dtype=np.int8))
        for rows, name in (
            (available_rows, "ForecastPVAvailableWatts"),
            (delivered_rows, "ForecastPVDeliveredWatts"),
            (curtailed_rows, "ForecastPVCurtailedWatts"),
            (charge_rows, "ForecastBatteryChargeInputWatts"),
            (discharge_rows, "ForecastBatteryDischargeOutputWatts"),
        ):
            rows.append(
                np.asarray(
                    result.get(name, pd.Series(np.full(len(times), np.nan), index=times)),
                    dtype=np.float32,
                )
            )
        capacities.append(model.usable_capacity_kwh)
        charge_efficiencies.append(model.charge_efficiency)
        discharge_efficiencies.append(model.discharge_efficiency)
        parasitic_loads.append(model.parasitic_load_w)

    soc = np.asarray(soc_rows, dtype=np.float32)
    solar_members = np.asarray(solar_rows, dtype=np.float32)
    load_members = np.asarray(load_rows, dtype=np.float32)
    discharge_members = np.asarray(discharge_rows, dtype=np.float32)
    charge_members = np.asarray(charge_rows, dtype=np.float32)
    _validate_member_soc_physics(
        soc,
        solar_members,
        load_members,
        charge_members,
        discharge_members,
        np.asarray(parasitic_loads, dtype=np.float64),
    )
    quantiles = np.nanquantile(soc, (0.10, 0.50, 0.90), axis=0).astype(np.float32)
    if (
        not np.isfinite(quantiles).all()
        or np.any(quantiles < -1.0e-5)
        or np.any(quantiles > 100.0 + 1.0e-5)
        or np.any(quantiles[0] > quantiles[1] + 1.0e-5)
        or np.any(quantiles[1] > quantiles[2] + 1.0e-5)
    ):
        raise ValueError("Candidate SOC quantiles violate bounds or ordering")
    output = xr.Dataset(
        {
            "BatterySOCForecastEnsemble": (("member", "time"), soc),
            "ECMWFSolarIrradianceEnsemble": (
                ("member", "time"), np.asarray(irradiance_rows, dtype=np.float32)
            ),
            "ForecastSolarWattsEnsemble": (("member", "time"), solar_members),
            "ForecastPVAvailableWattsEnsemble": (("member", "time"), np.asarray(available_rows, dtype=np.float32)),
            "ForecastPVDeliveredWattsEnsemble": (("member", "time"), np.asarray(delivered_rows, dtype=np.float32)),
            "ForecastPVCurtailedWattsEnsemble": (("member", "time"), np.asarray(curtailed_rows, dtype=np.float32)),
            "ForecastBatteryChargeInputWattsEnsemble": (("member", "time"), np.asarray(charge_rows, dtype=np.float32)),
            "ForecastBatteryDischargeOutputWattsEnsemble": (("member", "time"), discharge_members),
            "ForecastLoadWattsEnsemble": (("member", "time"), load_members),
            "ForecastLoadPhaseCodeEnsemble": (("member", "time"), np.asarray(phase_rows, dtype=np.int8)),
            "BatteryUsableCapacityKWhEnsemble": (("member",), np.asarray(capacities, dtype=np.float32)),
            "BatteryChargeEfficiencyEnsemble": (("member",), np.asarray(charge_efficiencies, dtype=np.float32)),
            "BatteryDischargeEfficiencyEnsemble": (("member",), np.asarray(discharge_efficiencies, dtype=np.float32)),
            "BatterySOCForecastP10": (("time",), quantiles[0]),
            "BatterySOCForecastP50": (("time",), quantiles[1]),
            "BatterySOCForecastP90": (("time",), quantiles[2]),
            "BatterySOCForecastMinimum": (("time",), np.nanmin(soc, axis=0).astype(np.float32)),
            "BatterySOCForecastMaximum": (("time",), np.nanmax(soc, axis=0).astype(np.float32)),
        },
        coords={"member": members, "time": times.to_numpy(dtype="datetime64[ns]")},
        attrs={
            **{str(name): str(value) for name, value in candidate.attrs.items()},
            "power_soc_ensemble_forecast_product": "true",
            "candidate_memberwise_ensemble": "true",
            "candidate_ensemble_version": CANDIDATE_ENSEMBLE_VERSION,
            "candidate_ensemble_contract_id": contract_id,
            "candidate_ensemble_forecast_identity_id": "candidate-ensemble-identity-v1-"
            + stable_json_digest(
                {
                    "contract": contract_id,
                    "initial_soc_time": issue_time.isoformat(),
                    "source_cycle_set_id": _text(candidate.attrs.get("source_cycle_set_id")),
                }
            )[:20],
            "candidate_ensemble_publication_signature": _candidate_ensemble_publication_signature(
                candidate, baseline_ensemble, contract_id
            ),
            "baseline_ensemble_signature": baseline_ensemble_signature(baseline_ensemble),
            "memberwise_solar_forcing": "same_baseline_ecmwf_site_members",
            "memberwise_load_policy": (
                "baseline_member_spread_plus_candidate_deterministic_delta"
                if load_residual
                else "same_baseline_member_load_distribution"
            ),
            "solar_forcing_mode": (
                "memberwise_physical_available_pv" if physical_solar else "baseline_member_legacy_solar_replayed"
            ),
            "source": (
                "read-only ECMWF site members propagated through physical three-array PV and battery physics"
                if physical_solar
                else "read-only ECMWF site members with legacy solar trace and bounded candidate load residual"
            ),
            "generated_at_utc": utc_now_iso(),
            "soc_physical_consistency_status": "passed_member_bounds_net_discharge_and_quantile_order",
        },
    )
    output = apply_operational_soc_threshold(output)
    for name in output.data_vars:
        output[name].attrs["units"] = (
            "1"
            if name.endswith("Probability") or name.endswith("PhaseCodeEnsemble")
            else "%"
            if "SOC" in name
            else "kWh"
            if "Capacity" in name
            else "W m-2"
            if "Irradiance" in name
            else "W"
        )
    return output


def _ensemble_pair_id(
    *,
    deterministic_pair_id: str,
    lane: str,
    candidate_ensemble: xr.Dataset,
) -> str:
    return "power-v12-ensemble-pair-v1-" + stable_json_digest(
        {
            "schema": 1,
            "deterministic_pair_id": deterministic_pair_id,
            "lane": lane,
            "candidate_ensemble_contract_id": _text(
                candidate_ensemble.attrs.get("candidate_ensemble_contract_id")
            ),
            "baseline_ensemble_signature": _text(
                candidate_ensemble.attrs.get("baseline_ensemble_signature")
            ),
        }
    )[:20]


def write_immutable_ensemble_pair_bundle(
    lane_root: Path,
    *,
    deterministic_pair_id: str,
    baseline_ensemble: xr.Dataset,
    candidate_ensemble: xr.Dataset,
    manifest_extra: Mapping[str, object],
) -> Path:
    """Expose a completed member-wise pair atomically and never overwrite it."""

    signature = _text(candidate_ensemble.attrs.get("candidate_ensemble_publication_signature"))
    if not signature:
        raise ValueError("Candidate ensemble has no publication signature")
    pair_id = _ensemble_pair_id(
        deterministic_pair_id=deterministic_pair_id,
        lane=_text(candidate_ensemble.attrs.get("candidate_lane")),
        candidate_ensemble=candidate_ensemble,
    )
    family = Path(lane_root) / "ensemble_pairs" / pair_id
    bundle = family / signature
    manifest = {
        "schema_version": 1,
        "pair_status": "complete",
        "ensemble_evaluation_pair_id": pair_id,
        "deterministic_evaluation_pair_id": deterministic_pair_id,
        "candidate_lane": _text(candidate_ensemble.attrs.get("candidate_lane")),
        "candidate_ensemble_contract_id": _text(
            candidate_ensemble.attrs.get("candidate_ensemble_contract_id")
        ),
        "candidate_ensemble_forecast_identity_id": _text(
            candidate_ensemble.attrs.get("candidate_ensemble_forecast_identity_id")
        ),
        "candidate_ensemble_publication_signature": signature,
        "baseline_ensemble_signature": _text(
            candidate_ensemble.attrs.get("baseline_ensemble_signature")
        ),
        "baseline_snapshot": "baseline_ensemble.zarr",
        "candidate_snapshot": "candidate_ensemble.zarr",
        **{str(key): value for key, value in manifest_extra.items()},
    }
    required_artifacts = {
        "baseline": "baseline_ensemble.zarr",
        "candidate": "candidate_ensemble.zarr",
    }
    if bundle.exists():
        try:
            existing = json.loads((bundle / "pair_manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Existing ensemble pair bundle is invalid: {bundle}") from exc
        if any(existing.get(key) != value for key, value in manifest.items()):
            raise RuntimeError(f"Existing immutable ensemble pair bundle does not match: {bundle}")
        if not pair_artifacts_valid(existing, bundle, required_artifacts):
            raise RuntimeError(
                f"Existing immutable ensemble pair bundle failed content verification: {bundle}"
            )
        return bundle
    family.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".ensemble-pair-staging-", dir=family) as temporary:
        staging = Path(temporary)
        _atomic_write_zarr(baseline_ensemble, staging / "baseline_ensemble.zarr")
        _atomic_write_zarr(candidate_ensemble, staging / "candidate_ensemble.zarr")
        completed_manifest = {
            **manifest,
            "artifact_digest_algorithm": PAIR_ARTIFACT_DIGEST_ALGORITHM,
            "artifact_checksums": {
                logical_name: immutable_artifact_record(
                    staging / relative_path,
                    relative_path=relative_path,
                )
                for logical_name, relative_path in required_artifacts.items()
            },
        }
        _write_state(staging / "pair_manifest.json", completed_manifest)
        if not pair_artifacts_valid(completed_manifest, staging, required_artifacts):
            raise RuntimeError("Staged immutable ensemble pair failed content verification")
        staging.replace(bundle)
    return bundle


def completed_ensemble_pair_bundles(root: Path) -> list[tuple[dict[str, object], Path]]:
    bundles: list[tuple[dict[str, object], Path]] = []
    root = Path(root)
    if not root.exists():
        return bundles
    for family in sorted(root.iterdir()):
        if not family.is_dir() or family.name.startswith("."):
            continue
        for bundle in sorted(family.iterdir()):
            if not bundle.is_dir() or bundle.name.startswith("."):
                continue
            try:
                manifest = json.loads((bundle / "pair_manifest.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if manifest.get("pair_status") != "complete":
                continue
            if not pair_artifacts_valid(
                manifest,
                bundle,
                {
                    "baseline": "baseline_ensemble.zarr",
                    "candidate": "candidate_ensemble.zarr",
                },
            ):
                continue
            bundles.append((manifest, bundle))
    return bundles


def _ensemble_archive_row(forecast: xr.Dataset) -> xr.Dataset:
    times = _time_index(forecast)
    issue = _utc_naive(forecast.attrs["initial_soc_time"])
    members = _member_values(forecast)
    fields = (
        "BatterySOCForecastEnsemble",
        "ECMWFSolarIrradianceEnsemble",
        "ForecastSolarWattsEnsemble",
        "ForecastLoadWattsEnsemble",
        "ForecastPVAvailableWattsEnsemble",
        "ForecastPVDeliveredWattsEnsemble",
        "ForecastPVCurtailedWattsEnsemble",
    )
    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {
        "ForecastValidTime": (
            ("issue_time", "forecast_step"),
            times.to_numpy(dtype="datetime64[ns]")[None, :],
        ),
        "ForecastLeadHours": (
            ("issue_time", "forecast_step"),
            (((times - issue) / pd.Timedelta(hours=1)).to_numpy(dtype=np.float32))[None, :],
        ),
        "ECMWFCycleTime": (
            ("issue_time",),
            np.asarray([_utc_naive(forecast.attrs["ecmwf_cycle_time"]).to_datetime64()]),
        ),
    }
    for name in fields:
        if name in forecast:
            data_vars[name] = (
                ("issue_time", "member", "forecast_step"),
                np.asarray(forecast[name].values)[None, :, :],
            )
    for name in (
        "CandidateEnsembleContractID",
        "CandidateEnsembleForecastIdentityID",
        "CandidateLane",
        "SourceCycleSetID",
        "ForecastModelContractID",
        "ForecastSystemVersion",
        "FeatureSetVersion",
        "FeatureSetDigest",
        "ForecastCodeRevision",
        "TrainingCutoffUTC",
        "SourceManifestDigest",
        "DegradedModeCode",
        "InputSnapshotID",
        "BaselinePublicationSignature",
        "BaselineEnsembleSignature",
    ):
        attr_name = {
            "CandidateEnsembleContractID": "candidate_ensemble_contract_id",
            "CandidateEnsembleForecastIdentityID": "candidate_ensemble_forecast_identity_id",
            "CandidateLane": "candidate_lane",
            "SourceCycleSetID": "source_cycle_set_id",
            "ForecastModelContractID": "forecast_model_contract_id",
            "ForecastSystemVersion": "forecast_system_version",
            "FeatureSetVersion": "feature_set_version",
            "FeatureSetDigest": "feature_set_digest",
            "ForecastCodeRevision": "forecast_code_revision",
            "TrainingCutoffUTC": "training_cutoff_utc",
            "SourceManifestDigest": "source_manifest_digest",
            "DegradedModeCode": "degraded_mode_code",
            "InputSnapshotID": "input_snapshot_id",
            "BaselinePublicationSignature": "baseline_publication_signature",
            "BaselineEnsembleSignature": "baseline_ensemble_signature",
        }[name]
        data_vars[name] = (("issue_time",), np.asarray([_text(forecast.attrs.get(attr_name))], dtype="U256"))
    return xr.Dataset(
        data_vars,
        coords={
            "issue_time": np.asarray([issue.to_datetime64()]),
            "member": members,
            "forecast_step": np.arange(len(times), dtype=np.int16),
        },
        attrs={
            "power_v12_candidate_ensemble_archive": "true",
            "candidate_ensemble_contract_id": _text(
                forecast.attrs.get("candidate_ensemble_contract_id")
            ),
        },
    )


def append_candidate_ensemble_archive(
    forecast: xr.Dataset,
    path: Path,
    *,
    retention_days: float = 21.0,
) -> xr.Dataset:
    """Append within one semantic ensemble-contract directory only."""

    row = _ensemble_archive_row(forecast)
    previous = xr.open_zarr(path, chunks={}).load() if Path(path).exists() else None
    if previous is not None:
        expected = _text(forecast.attrs.get("candidate_ensemble_contract_id"))
        existing_contracts = previous.get("CandidateEnsembleContractID")
        seen = (
            {_text(value) for value in np.asarray(existing_contracts.values)}
            if existing_contracts is not None
            else set()
        )
        if seen and seen != {expected}:
            raise ValueError("Candidate ensemble archive mixes semantic contracts")
        if not np.array_equal(previous["member"].values, row["member"].values):
            raise ValueError("Candidate ensemble archive member set changed")
        steps = np.arange(
            max(previous.sizes["forecast_step"], row.sizes["forecast_step"]), dtype=np.int16
        )
        combined = xr.concat(
            [previous.reindex(forecast_step=steps), row.reindex(forecast_step=steps)],
            dim="issue_time",
        ).sortby("issue_time")
        combined = combined.isel(
            issue_time=~combined.indexes["issue_time"].duplicated(keep="last")
        )
    else:
        combined = row
    issues = pd.DatetimeIndex(combined["issue_time"].values)
    cutoff = issues.max() - pd.Timedelta(days=float(retention_days))
    combined = combined.isel(issue_time=issues >= cutoff)
    combined.attrs.update(row.attrs)
    _atomic_write_zarr(combined, Path(path))
    return combined


def ensemble_evaluation_contract_from_forecast(forecast: xr.Dataset) -> dict[str, str]:
    return {name: _text(forecast.attrs.get(name)) for name in ENSEMBLE_EVALUATION_CONTRACT_ATTRS}


def _matches_ensemble_evaluation_contract(
    forecast: xr.Dataset, contract: Mapping[str, object] | None
) -> bool:
    if contract is None:
        return True
    return all(
        _text(forecast.attrs.get(name)) == _text(contract.get(name))
        for name in ENSEMBLE_EVALUATION_CONTRACT_ATTRS
    )


def _power_soc(power: xr.Dataset) -> pd.Series:
    if "BatterySOC" not in power or "time" not in power.coords:
        return pd.Series(dtype=np.float64)
    return pd.Series(
        np.asarray(power["BatterySOC"].values, dtype=np.float64),
        index=pd.DatetimeIndex(power["time"].values),
    )


def build_campaign_ensemble_evidence(
    pairs_root: Path,
    power: xr.Dataset,
    *,
    lane: str,
    evaluation_contract: Mapping[str, object] | None = None,
) -> xr.Dataset:
    """Materialise complete, contract-filtered candidate ensemble pair rows."""

    observed = _power_soc(power)
    records: list[dict[str, object]] = []
    candidate_rows: list[np.ndarray] = []
    baseline_rows: list[np.ndarray] = []
    member_values: np.ndarray | None = None
    incompatible = 0
    for manifest, bundle in completed_ensemble_pair_bundles(pairs_root):
        try:
            with xr.open_zarr(bundle / "baseline_ensemble.zarr", chunks={}) as opened:
                baseline = opened.load()
            with xr.open_zarr(bundle / "candidate_ensemble.zarr", chunks={}) as opened:
                candidate = opened.load()
            times, members, issue, initial_soc = _validate_memberwise_ensemble_pair(
                baseline, candidate
            )
        except Exception:
            continue
        if not _matches_ensemble_evaluation_contract(candidate, evaluation_contract):
            incompatible += 1
            continue
        candidate_times = _time_index(candidate)
        if not np.array_equal(candidate_times.values, times.values):
            continue
        if member_values is None:
            member_values = members
        elif not np.array_equal(member_values, members):
            incompatible += 1
            continue
        candidate_soc = np.asarray(candidate["BatterySOCForecastEnsemble"].values, dtype=np.float64)
        baseline_soc = np.asarray(baseline["BatterySOCForecastEnsemble"].values, dtype=np.float64)
        if candidate_soc.shape != baseline_soc.shape or candidate_soc.shape != (len(members), len(times)):
            continue
        observations = observed.reindex(
            times, method="nearest", tolerance=pd.Timedelta(minutes=10)
        ).to_numpy(dtype=np.float64)
        lead_hours = (times - issue) / pd.Timedelta(hours=1)
        for index, valid_time in enumerate(times):
            records.append(
                {
                    "IssueTime": issue.to_datetime64(),
                    "ValidTime": valid_time.to_datetime64(),
                    "LeadHours": float(lead_hours[index]),
                    "SOCAuthoringAnchor": float(initial_soc),
                    "CandidateLane": str(lane),
                    "EnsembleEvaluationPairID": _text(manifest.get("ensemble_evaluation_pair_id")),
                    "CandidateEnsembleContractID": _text(
                        candidate.attrs.get("candidate_ensemble_contract_id")
                    ),
                    "CandidateEnsembleIdentityID": _text(
                        candidate.attrs.get("candidate_ensemble_forecast_identity_id")
                    ),
                    "SourceCycleSetID": _text(candidate.attrs.get("source_cycle_set_id")),
                    "LoadMode": _text(candidate.attrs.get("load_mode"), "unknown"),
                    "DegradedModeCode": _text(candidate.attrs.get("degraded_mode_code"), "none"),
                    "ObservedSOC": float(observations[index]),
                    "EvaluationAvailable": bool(np.isfinite(observations[index])),
                }
            )
            candidate_rows.append(candidate_soc[:, index])
            baseline_rows.append(baseline_soc[:, index])
    if not records:
        return xr.Dataset(
            coords={"record": np.array([], dtype=np.int64), "member": np.array([], dtype=np.int16)},
            attrs={
                "power_v12_candidate_ensemble_evidence": "true",
                "candidate_lane": lane,
                "generated_at_utc": utc_now_iso(),
                "evidence_status": "no_complete_ensemble_pair_bundles",
                "evaluation_contract": json.dumps(dict(evaluation_contract or {}), sort_keys=True),
                "incompatible_pair_count": int(incompatible),
            },
        )
    assert member_values is not None
    # Preserve every issue-time pair for audit/operational diagnostics, while
    # marking exactly one latest row per source cycle and valid time for
    # independent campaign evidence and promotion.
    order = sorted(
        range(len(records)),
        key=lambda index: (
            np.datetime64(records[index]["IssueTime"]),
            np.datetime64(records[index]["ValidTime"]),
            str(records[index]["EnsembleEvaluationPairID"]),
        ),
    )
    records = [records[index] for index in order]
    candidate_rows = [candidate_rows[index] for index in order]
    baseline_rows = [baseline_rows[index] for index in order]
    selected_by_cycle_valid: dict[tuple[str, int], int] = {}
    for index, record in enumerate(records):
        source_cycle = str(record.get("SourceCycleSetID", "")).strip()
        if not source_cycle:
            source_cycle = "issue:" + pd.Timestamp(record["IssueTime"]).isoformat()
            record["SourceCycleSetID"] = source_cycle
        valid_ns = int(np.datetime64(record["ValidTime"], "ns").astype(np.int64))
        selected_by_cycle_valid[(source_cycle, valid_ns)] = index
    selected_indices = set(selected_by_cycle_valid.values())
    duplicate_cycle_valid_rows = len(records) - len(selected_indices)
    for index, record in enumerate(records):
        record["IndependentEvaluationSample"] = index in selected_indices
        record["SOCAnchorTime"] = record["IssueTime"]
    scalar_vars: dict[str, tuple[tuple[str], np.ndarray]] = {}
    for name, values in {name: [record[name] for record in records] for name in records[0]}.items():
        first = values[0]
        if isinstance(first, np.datetime64):
            array = np.asarray(values, dtype="datetime64[ns]")
        elif isinstance(first, (bool, np.bool_)):
            array = np.asarray(values, dtype=bool)
        elif isinstance(first, (float, np.floating)):
            array = np.asarray(values, dtype=np.float64)
        else:
            array = np.asarray([str(value) for value in values], dtype="U512")
        scalar_vars[name] = (("record",), array)
    scalar_vars["CandidateSOCMembers"] = (
        ("record", "member"), np.asarray(candidate_rows, dtype=np.float32)
    )
    scalar_vars["BaselineSOCMembers"] = (
        ("record", "member"), np.asarray(baseline_rows, dtype=np.float32)
    )
    return xr.Dataset(
        scalar_vars,
        coords={"record": np.arange(len(records), dtype=np.int64), "member": member_values},
        attrs={
            "power_v12_candidate_ensemble_evidence": "true",
            "candidate_lane": lane,
            "generated_at_utc": utc_now_iso(),
            "evidence_status": "complete_ensemble_pair_bundles_materialised",
            "evaluation_contract": json.dumps(dict(evaluation_contract or {}), sort_keys=True),
            "incompatible_pair_count": int(incompatible),
            "duplicate_cycle_valid_rows_discarded": 0,
            "duplicate_cycle_valid_rows_retained": int(duplicate_cycle_valid_rows),
            "reserve_threshold_soc_pct": f"{MINIMUM_OPERATIONAL_SOC_PCT:g}",
        },
    )


def _ensemble_metric_summary(evidence: xr.Dataset, mask: np.ndarray) -> dict[str, object]:
    if evidence.sizes.get("record", 0) == 0 or "EvaluationAvailable" not in evidence:
        return {"status": "insufficient_evidence", "samples": 0, "cycles": 0, "utc_days": 0}
    available = np.asarray(evidence["EvaluationAvailable"].values, dtype=bool)
    selected = np.asarray(mask, dtype=bool) & available
    if not np.any(selected):
        return {"status": "insufficient_evidence", "samples": 0, "cycles": 0, "utc_days": 0}
    observations = np.asarray(evidence["ObservedSOC"].values, dtype=np.float64)[selected]
    anchors = np.asarray(evidence["SOCAuthoringAnchor"].values, dtype=np.float64)[selected]
    candidate = np.asarray(evidence["CandidateSOCMembers"].values, dtype=np.float64)[selected]
    baseline = np.asarray(evidence["BaselineSOCMembers"].values, dtype=np.float64)[selected]
    valid = np.isfinite(observations) & np.isfinite(anchors)
    valid &= np.array(
        [np.count_nonzero(np.isfinite(row)) >= 2 for row in candidate], dtype=bool
    )
    valid &= np.array(
        [np.count_nonzero(np.isfinite(row)) >= 2 for row in baseline], dtype=bool
    )
    observations, anchors, candidate, baseline = (
        values[valid] for values in (observations, anchors, candidate, baseline)
    )
    issue_values = pd.DatetimeIndex(evidence["IssueTime"].values)[selected][valid]
    source_cycles = (
        np.asarray(evidence["SourceCycleSetID"].values, dtype=str)[selected][valid]
        if "SourceCycleSetID" in evidence
        else np.asarray([], dtype=str)
    )
    if len(observations) == 0:
        return {"status": "insufficient_evidence", "samples": 0, "cycles": 0, "utc_days": 0}
    candidate_crps = np.asarray(
        [_crps_ensemble(row, observation) for row, observation in zip(candidate, observations)],
        dtype=np.float64,
    )
    baseline_crps = np.asarray(
        [_crps_ensemble(row, observation) for row, observation in zip(baseline, observations)],
        dtype=np.float64,
    )
    persistence_crps = np.abs(anchors - observations)
    coverage = np.asarray(
        [
            float(np.nanquantile(row, 0.10) <= observation <= np.nanquantile(row, 0.90))
            for row, observation in zip(candidate, observations)
        ],
        dtype=np.float64,
    )
    candidate_probability = np.mean(candidate < MINIMUM_OPERATIONAL_SOC_PCT, axis=1)
    baseline_probability = np.mean(baseline < MINIMUM_OPERATIONAL_SOC_PCT, axis=1)
    persistence_probability = (anchors < MINIMUM_OPERATIONAL_SOC_PCT).astype(np.float64)
    outcomes = (observations < MINIMUM_OPERATIONAL_SOC_PCT).astype(np.float64)
    candidate_brier = np.square(candidate_probability - outcomes)
    baseline_brier = np.square(baseline_probability - outcomes)
    persistence_brier = np.square(persistence_probability - outcomes)
    usable_cycles = [
        value.strip()
        for value in source_cycles
        if value.strip() and value.strip().lower() not in {"nan", "none"}
    ]
    cycles = len(set(usable_cycles)) if usable_cycles else int(issue_values.nunique())
    days = int(issue_values.floor("D").nunique())
    events = int(np.count_nonzero(outcomes))
    summary: dict[str, object] = {
        "status": "evidence" if len(observations) >= 2 else "diagnostic_sparse",
        "samples": int(len(observations)),
        "cycles": cycles,
        "utc_days": days,
        "candidate_crps": float(np.nanmean(candidate_crps)),
        "baseline_crps": float(np.nanmean(baseline_crps)),
        "persistence_crps": float(np.nanmean(persistence_crps)),
        "candidate_coverage_p10_p90": float(np.nanmean(coverage)),
        "baseline_coverage_p10_p90": float(
            np.mean(
                [
                    float(np.nanquantile(row, 0.10) <= observation <= np.nanquantile(row, 0.90))
                    for row, observation in zip(baseline, observations)
                ]
            )
        ),
        "reserve_events": events,
        "reserve_samples": int(len(outcomes)),
        "candidate_brier": float(np.mean(candidate_brier)),
        "baseline_brier": float(np.mean(baseline_brier)),
        "persistence_brier": float(np.mean(persistence_brier)),
    }
    for prefix, reference in (
        ("baseline", float(summary["baseline_crps"])),
        ("persistence", float(summary["persistence_crps"])),
    ):
        summary[f"candidate_crps_skill_vs_{prefix}"] = (
            float(1.0 - float(summary["candidate_crps"]) / reference)
            if np.isfinite(reference) and reference > 0.0
            else np.nan
        )
    for prefix, reference in (
        ("baseline", float(summary["baseline_brier"])),
        ("persistence", float(summary["persistence_brier"])),
    ):
        summary[f"candidate_brier_skill_vs_{prefix}"] = (
            float(1.0 - float(summary["candidate_brier"]) / reference)
            if np.isfinite(reference) and reference > 0.0
            else np.nan
        )
    summary["reserve_event_status"] = (
        "insufficient_events"
        if events < MIN_RESERVE_EVENTS
        else "pass"
        if float(summary["candidate_brier"]) < float(summary["baseline_brier"])
        and float(summary["candidate_brier"]) < float(summary["persistence_brier"])
        else "fail"
    )
    return summary


def campaign_ensemble_score_surfaces(evidence: xr.Dataset) -> dict[str, object]:
    """Return cumulative and 24-hour ensemble metrics without mixing contracts."""

    try:
        contract = json.loads(str(evidence.attrs.get("evaluation_contract", "{}")))
    except json.JSONDecodeError:
        contract = {}
    if evidence.sizes.get("record", 0) == 0:
        empty = _ensemble_metric_summary(evidence, np.array([], dtype=bool))
        return {
            "generated_at_utc": utc_now_iso(),
            "campaign_evidence": {"overall": empty, "lead_buckets": {label: empty for label, _, _ in LEAD_BUCKETS}},
            "daily_diagnostic": {"overall": empty, "lead_buckets": {label: empty for label, _, _ in LEAD_BUCKETS}},
            "evaluation_contract": contract,
            "incompatible_pair_count": int(evidence.attrs.get("incompatible_pair_count", 0) or 0),
        }
    leads = np.asarray(evidence["LeadHours"].values, dtype=np.float64)
    available = np.asarray(evidence["EvaluationAvailable"].values, dtype=bool)
    valid_times = pd.DatetimeIndex(evidence["ValidTime"].values)
    latest = valid_times[available].max() if np.any(available) else pd.NaT
    daily_mask = (
        np.asarray(valid_times > latest - pd.Timedelta(hours=24), dtype=bool)
        if not pd.isna(latest)
        else np.zeros(len(leads), dtype=bool)
    )

    def surface(mask: np.ndarray) -> dict[str, object]:
        return {
            "overall": _ensemble_metric_summary(evidence, mask),
            "lead_buckets": {
                label: _ensemble_metric_summary(evidence, mask & (leads >= start) & (leads < end))
                for label, start, end in LEAD_BUCKETS
            },
        }

    all_rows = (
        np.asarray(evidence["IndependentEvaluationSample"].values, dtype=bool)
        if "IndependentEvaluationSample" in evidence
        else np.ones(len(leads), dtype=bool)
    )
    return {
        "generated_at_utc": utc_now_iso(),
        "campaign_evidence": surface(all_rows),
        "daily_diagnostic": surface(daily_mask),
        "evaluation_contract": contract,
        "incompatible_pair_count": int(evidence.attrs.get("incompatible_pair_count", 0) or 0),
        "reserve_event_minimum": MIN_RESERVE_EVENTS,
    }


def ensemble_promotion_gate(evidence: xr.Dataset) -> dict[str, object]:
    """Review the CRPS, coverage and reserve-event gate; never accept automatically."""

    surfaces = campaign_ensemble_score_surfaces(evidence)
    campaign = surfaces["campaign_evidence"]
    buckets = campaign["lead_buckets"]
    evidence_ready = all(
        int(value.get("cycles", 0)) >= 30 and int(value.get("utc_days", 0)) >= 10
        for value in buckets.values()
    )
    overall = dict(campaign["overall"])
    if not evidence_ready:
        return {
            "status": "insufficient_evidence",
            "independent_evidence": {
                label: {
                    "cycles": int(value.get("cycles", 0)),
                    "utc_days": int(value.get("utc_days", 0)),
                    "status": "eligible"
                    if int(value.get("cycles", 0)) >= 30 and int(value.get("utc_days", 0)) >= 10
                    else "insufficient_evidence",
                }
                for label, value in buckets.items()
            },
            "overall": overall,
            "reserve_events": str(overall.get("reserve_event_status", "insufficient_events")),
            "surfaces": surfaces,
        }
    crps_pass = (
        float(overall.get("candidate_crps", np.nan)) < float(overall.get("baseline_crps", np.nan))
        and float(overall.get("candidate_crps", np.nan)) < float(overall.get("persistence_crps", np.nan))
    )
    coverage = float(overall.get("candidate_coverage_p10_p90", np.nan))
    coverage_pass = np.isfinite(coverage) and 0.75 <= coverage <= 0.90
    reserve = str(overall.get("reserve_event_status", "insufficient_events"))
    return {
        "status": "pass" if crps_pass and coverage_pass and reserve != "fail" else "fail",
        "crps_status": "pass" if crps_pass else "fail",
        "coverage_status": "pass" if coverage_pass else "fail",
        "coverage_required_range": [0.75, 0.90],
        "reserve_events": reserve,
        "reserve_event_minimum": MIN_RESERVE_EVENTS,
        "overall": overall,
        "independent_evidence": {
            label: {
                "cycles": int(value.get("cycles", 0)),
                "utc_days": int(value.get("utc_days", 0)),
                "status": "eligible",
            }
            for label, value in buckets.items()
        },
        "surfaces": surfaces,
    }
