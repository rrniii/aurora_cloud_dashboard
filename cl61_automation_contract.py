"""Immutable, fail-closed CL61 automation intent contract.

This module deliberately contains no PDU client and no control transport.  It
defines the small signed document exchanged between the forecast publisher and
the ASS-local controller.  A diagnostic intent is useful for shadow evaluation
but is never executable by the controller.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping


INTENT_SCHEMA_VERSION = 1
CL61_PDU_OUTLET = 5
AUTHORITY_DIAGNOSTIC = "diagnostic"
AUTHORITY_OPERATIONAL = "operational"
ALLOWED_AUTHORITIES = {AUTHORITY_DIAGNOSTIC, AUTHORITY_OPERATIONAL}
ALLOWED_ACTIONS = {"hold", "start", "stop"}
ALLOWED_ENVIRONMENTS = {"development", "production"}
HMAC_ALGORITHM = "hmac-sha256"
UNSIGNED_ALGORITHM = "none"


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def utc_iso(value: datetime) -> str:
    moment = value.astimezone(timezone.utc).replace(microsecond=0)
    return moment.isoformat().replace("+00:00", "Z")


def parse_utc(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _content_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = deepcopy(dict(value))
    for key in ("intent_id", "content_digest", "signature"):
        payload.pop(key, None)
    return payload


def content_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(_content_payload(value))).hexdigest()


def _signature_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = deepcopy(dict(value))
    payload.pop("signature", None)
    return payload


def sign_intent(value: Mapping[str, Any], signing_key: bytes | None) -> dict[str, Any]:
    """Return an HMAC-authenticated copy, or an explicitly unsigned diagnostic.

    An unsigned document can exist only for observer diagnostics.  The edge
    validator rejects it whenever operational execution is requested.
    """
    payload = deepcopy(dict(value))
    if signing_key:
        signature = hmac.new(signing_key, canonical_json(_signature_payload(payload)), hashlib.sha256).hexdigest()
        payload["signature"] = {"algorithm": HMAC_ALGORITHM, "value": signature}
    else:
        payload["signature"] = {"algorithm": UNSIGNED_ALGORITHM, "value": ""}
    return payload


def build_intent(
    *,
    environment: str,
    authority: str,
    proposed_action: str,
    desired_outlet_state: bool,
    reason_codes: list[str],
    forecast: Mapping[str, Any],
    safety: Mapping[str, Any],
    input_ages_minutes: Mapping[str, Any],
    generated_at: datetime | None = None,
    ttl_seconds: int = 900,
    signing_key: bytes | None = None,
) -> dict[str, Any]:
    """Create a canonical CL61 intent with a bounded validity window."""
    if environment not in ALLOWED_ENVIRONMENTS:
        raise ValueError(f"Unsupported automation environment: {environment}")
    if authority not in ALLOWED_AUTHORITIES:
        raise ValueError(f"Unsupported automation authority: {authority}")
    if proposed_action not in ALLOWED_ACTIONS:
        raise ValueError(f"Unsupported proposed action: {proposed_action}")
    now = (generated_at or utc_now()).astimezone(timezone.utc).replace(microsecond=0)
    expires = now + timedelta(seconds=max(int(ttl_seconds), 1))
    payload: dict[str, Any] = {
        "schema_version": INTENT_SCHEMA_VERSION,
        "environment": environment,
        "authority": authority,
        "generated_at_utc": utc_iso(now),
        "not_before_utc": utc_iso(now),
        "expires_at_utc": utc_iso(expires),
        "target": {"instrument": "CL61", "pdu_outlet": CL61_PDU_OUTLET},
        "proposed_action": proposed_action,
        "desired_outlet_state": bool(desired_outlet_state),
        "reason_codes": sorted({str(value) for value in reason_codes if str(value)}),
        "forecast": dict(forecast),
        "safety": dict(safety),
        "input_ages_minutes": dict(input_ages_minutes),
    }
    digest = content_digest(payload)
    payload["content_digest"] = digest
    payload["intent_id"] = digest[:24]
    return sign_intent(payload, signing_key)


def validate_intent(
    value: Mapping[str, Any],
    *,
    now: datetime | None = None,
    signing_key: bytes | None = None,
    require_operational: bool = False,
) -> tuple[bool, tuple[str, ...]]:
    """Validate structure, expiry, integrity and execution authority.

    Callers must treat every returned error as a fail-closed condition.
    """
    payload = dict(value)
    errors: list[str] = []
    if int(payload.get("schema_version", -1)) != INTENT_SCHEMA_VERSION:
        errors.append("unsupported_schema")
    if payload.get("environment") not in ALLOWED_ENVIRONMENTS:
        errors.append("invalid_environment")
    authority = str(payload.get("authority", ""))
    if authority not in ALLOWED_AUTHORITIES:
        errors.append("invalid_authority")
    if str(payload.get("proposed_action", "")) not in ALLOWED_ACTIONS:
        errors.append("invalid_action")
    desired_state = payload.get("desired_outlet_state")
    if not isinstance(desired_state, bool):
        errors.append("invalid_desired_state")
    elif payload.get("proposed_action") == "start" and not desired_state:
        errors.append("action_state_mismatch")
    elif payload.get("proposed_action") == "stop" and desired_state:
        errors.append("action_state_mismatch")
    target = payload.get("target")
    try:
        target_outlet = int(target.get("pdu_outlet", -1)) if isinstance(target, Mapping) else -1
    except (TypeError, ValueError):
        target_outlet = -1
    if not isinstance(target, Mapping) or target.get("instrument") != "CL61" or target_outlet != CL61_PDU_OUTLET:
        errors.append("invalid_target")
    computed_digest = content_digest(payload)
    if not hmac.compare_digest(str(payload.get("content_digest", "")), computed_digest):
        errors.append("content_digest_mismatch")
    if str(payload.get("intent_id", "")) != computed_digest[:24]:
        errors.append("intent_id_mismatch")
    check_time = (now or utc_now()).astimezone(timezone.utc)
    not_before = parse_utc(payload.get("not_before_utc"))
    expires = parse_utc(payload.get("expires_at_utc"))
    if not_before is None or expires is None:
        errors.append("invalid_validity_window")
    else:
        if expires <= not_before:
            errors.append("invalid_validity_window")
        if check_time < not_before:
            errors.append("not_yet_valid")
        if check_time >= expires:
            errors.append("expired")
    signature = payload.get("signature")
    algorithm = signature.get("algorithm") if isinstance(signature, Mapping) else ""
    signature_value = signature.get("value") if isinstance(signature, Mapping) else ""
    if algorithm == HMAC_ALGORITHM:
        if not signing_key:
            errors.append("signature_key_unavailable")
        else:
            expected = hmac.new(signing_key, canonical_json(_signature_payload(payload)), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(str(signature_value), expected):
                errors.append("signature_mismatch")
    elif algorithm != UNSIGNED_ALGORITHM:
        errors.append("unsupported_signature")
    if require_operational:
        if authority != AUTHORITY_OPERATIONAL:
            errors.append("diagnostic_authority")
        if algorithm != HMAC_ALGORITHM or not signing_key:
            errors.append("operational_signature_required")
        safety = payload.get("safety")
        if not isinstance(safety, Mapping) or not bool(safety.get("control_eligible", False)):
            errors.append("control_not_eligible")
        elif not bool(safety.get("planning_ready", False)) or not bool(safety.get("schedule_safe", False)):
            errors.append("schedule_not_safe")
        forecast = payload.get("forecast")
        if not isinstance(forecast, Mapping) or not all(
            str(forecast.get(name, "")).strip()
            for name in ("forecast_system_version", "source_cycle_set_id", "scenario_publication_signature")
        ):
            errors.append("forecast_provenance_missing")
    return (not errors, tuple(dict.fromkeys(errors)))
