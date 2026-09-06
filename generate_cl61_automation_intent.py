#!/usr/bin/env python3
"""Publish CL61 shadow-automation intents from an operating scenario product.

The publisher is intentionally read-only with respect to instruments.  It
creates diagnostic intents and a compact status document for the dashboard and
ASS shadow agent; it never opens a PDU connection.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from cl61_automation_contract import (
    AUTHORITY_DIAGNOSTIC,
    CL61_PDU_OUTLET,
    build_intent,
)
from power_operating_scenarios import SCENARIO_OPTIMIZED


DEFAULT_TTL_SECONDS = 900
DEFAULT_HISTORY_RETENTION = 4096


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _read_key(path: Path | None) -> bytes | None:
    if path is None or not path.is_file():
        return None
    try:
        value = path.read_bytes().strip()
    except OSError:
        return None
    return value or None


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _append_history(path: Path, value: dict[str, Any], *, retention: int = DEFAULT_HISTORY_RETENTION) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    previous: list[str] = []
    if path.exists():
        try:
            previous = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()][-max(int(retention) - 1, 0) :]
        except OSError:
            previous = []
    previous.append(json.dumps(value, sort_keys=True, separators=(",", ":")))
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text("\n".join(previous) + "\n", encoding="utf-8")
    temporary.replace(path)


def _attr_float(scenarios: xr.Dataset, name: str) -> float | None:
    try:
        value = float(scenarios.attrs.get(name, ""))
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _attr_bool(scenarios: xr.Dataset, name: str) -> bool:
    return str(scenarios.attrs.get(name, "")).strip().lower() == "true"


def _utc_string(value: object) -> str:
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return ""
    return pd.Timestamp(parsed).isoformat().replace("+00:00", "Z")


def build_diagnostic_intent(
    scenarios: xr.Dataset,
    *,
    environment: str = "development",
    generated_at: datetime | None = None,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    signing_key: bytes | None = None,
) -> dict[str, Any]:
    """Turn the CL61-first schedule into a non-executable diagnostic intent."""
    now = generated_at or _utc_now()
    scenario_coordinate = scenarios.coords.get("scenario")
    scenario_ids = (
        [str(value) for value in scenario_coordinate.values]
        if scenario_coordinate is not None
        else []
    )
    planning_ready = str(scenarios.attrs.get("planning_status", "ready")) == "ready"
    reason_codes = ["control_capability_disabled", "forecast_control_evidence_not_accepted"]
    desired = False
    proposed_action = "hold"
    current_on = False
    future_on = False
    mode_time_utc = ""
    if planning_ready and SCENARIO_OPTIMIZED in scenario_ids and "ScenarioModeCode" in scenarios:
        index = scenario_ids.index(SCENARIO_OPTIMIZED)
        mode_codes = np.asarray(scenarios["ScenarioModeCode"].isel(scenario=index).values, dtype=np.int64)
        time_values = pd.DatetimeIndex(scenarios["time"].values) if "time" in scenarios else pd.DatetimeIndex([])
        if mode_codes.size and len(time_values):
            current_on = bool(mode_codes[0] & 1)
            future_index = 1 if mode_codes.size > 1 else 0
            future_on = bool(mode_codes[future_index] & 1)
            desired = future_on
            mode_time_utc = _utc_string(time_values[future_index])
            if future_on and not current_on:
                proposed_action = "start"
            elif current_on and not future_on:
                proposed_action = "stop"
            if not _attr_bool(scenarios, "optimized_safe"):
                reason_codes.append(str(scenarios.attrs.get("optimized_reason_code", "schedule_not_safe")))
                proposed_action = "hold"
                desired = current_on
            else:
                reason_codes.append("cl61_first_schedule")
        else:
            reason_codes.append("missing_schedule_timeline")
    else:
        reason_codes.append("operating_scenarios_unavailable")

    forecast = {
        "forecast_system_version": str(scenarios.attrs.get("forecast_system_version", "")),
        "feature_set_version": str(scenarios.attrs.get("feature_set_version", "")),
        "feature_set_digest": str(scenarios.attrs.get("feature_set_digest", "")),
        "forecast_code_revision": str(scenarios.attrs.get("forecast_code_revision", "")),
        "source_cycle_set_id": str(scenarios.attrs.get("source_cycle_set_id", "")),
        "source_manifest_digest": str(scenarios.attrs.get("source_manifest_digest", "")),
        "forecast_identity_id": str(scenarios.attrs.get("forecast_identity_id", "")),
        "planning_forecast_generated_at_utc": str(scenarios.attrs.get("planning_forecast_generated_at_utc", "")),
        "planning_forecast_initial_soc_time": str(scenarios.attrs.get("planning_forecast_initial_soc_time", "")),
        "scenario_publication_signature": str(scenarios.attrs.get("publication_signature", "")),
        "schedule_policy": str(scenarios.attrs.get("optimized_schedule_policy", "")),
        "schedule_time_utc": mode_time_utc,
    }
    safety = {
        "control_eligible": False,
        "planning_ready": planning_ready,
        "schedule_safe": _attr_bool(scenarios, "optimized_safe"),
        "minimum_p10_soc_pct": _attr_float(scenarios, "optimized_minimum_p10_soc"),
        "operational_reserve_soc_pct": _attr_float(scenarios, "minimum_operational_soc_pct"),
        "minimum_run_hours": _attr_float(scenarios, "minimum_controlled_run_hours"),
        "current_cl61_on": current_on,
        "future_cl61_on": future_on,
    }
    input_ages_minutes = {
        "power": _attr_float(scenarios, "input_power_age_minutes"),
        "forecast": None,
        "pdu": None,
    }
    return build_intent(
        environment=environment,
        authority=AUTHORITY_DIAGNOSTIC,
        proposed_action=proposed_action,
        desired_outlet_state=desired,
        reason_codes=reason_codes,
        forecast=forecast,
        safety=safety,
        input_ages_minutes=input_ages_minutes,
        generated_at=now,
        ttl_seconds=ttl_seconds,
        signing_key=signing_key,
    )


def publish_diagnostic_intent(
    scenarios: xr.Dataset,
    *,
    intent_path: Path,
    status_path: Path,
    history_path: Path,
    environment: str = "development",
    signing_key_path: Path | None = None,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Publish the shadow intent, status and append-only diagnostic history."""
    key = _read_key(signing_key_path)
    intent = build_diagnostic_intent(
        scenarios,
        environment=environment,
        generated_at=generated_at,
        ttl_seconds=ttl_seconds,
        signing_key=key,
    )
    signature = dict(intent.get("signature", {}))
    status = {
        "schema_version": 1,
        "generated_at_utc": intent["generated_at_utc"],
        "environment": environment,
        "control_authority": "observe_only",
        "capability": False,
        "mode": "observe_only",
        "target": {"instrument": "CL61", "pdu_outlet": CL61_PDU_OUTLET},
        "last_intent_id": intent["intent_id"],
        "last_intent_digest": intent["content_digest"],
        "last_proposed_action": intent["proposed_action"],
        "desired_outlet_state": intent["desired_outlet_state"],
        "expires_at_utc": intent["expires_at_utc"],
        "reason_codes": intent["reason_codes"],
        "signature_algorithm": str(signature.get("algorithm", "none")),
        "signature_available": str(signature.get("algorithm", "none")) != "none",
        "safety": intent["safety"],
        "forecast": intent["forecast"],
    }
    _write_json_atomic(intent_path, intent)
    _write_json_atomic(status_path, status)
    _append_history(
        history_path,
        {
            "recorded_at_utc": intent["generated_at_utc"],
            "intent_id": intent["intent_id"],
            "content_digest": intent["content_digest"],
            "proposed_action": intent["proposed_action"],
            "desired_outlet_state": intent["desired_outlet_state"],
            "reason_codes": intent["reason_codes"],
            "authority": intent["authority"],
        },
    )
    return status


def configured_path(name: str, fallback: Path) -> Path:
    configured = os.environ.get(name, "").strip()
    return Path(configured) if configured else fallback
