"""Read-only product catalog helpers for the Aurora mobile API."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import re
import sqlite3
from time import monotonic
from typing import Any

from auroracam_catalog import AURORACAM_CAMERAS, available_days as auroracam_available_days, day_records as auroracam_day_records, latest_records as auroracam_latest_records
from display_artifact_manifest import load_manifest
from menapia_flight_status import summarize_menapia_flight
from uas_mqtt import load_uas_mqtt_log
from instrument_registry import (
    INSTRUMENTS,
    INSTRUMENT_BY_ID,
    InstrumentContract as Instrument,
    PDU_INSTRUMENTS as PDU_INSTRUMENT_CONTRACTS,
    SCIENCE_DC_INSTRUMENTS as SCIENCE_DC_INSTRUMENT_CONTRACTS,
)
from wxcam_catalog import parse_timestamp as parse_wxcam_timestamp


UTC = timezone.utc


APP_DIR = Path(__file__).resolve().parent
DATE_TOKEN_RE = re.compile(r"(20\d{6})")
WXCAM_DAY_RE = re.compile(r"20\d{2}-\d{2}-\d{2}")
AURORACAM_DAY_RE = re.compile(r"20\d{2}-\d{2}-\d{2}")
MENAPIA_FLIGHT_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")
MOBILE_POWER_MAX_POINTS = max(100, min(int(os.environ.get("AURORA_MOBILE_POWER_MAX_POINTS", "160")), 160))


WXCAM_STREAMS = {
    "fish_hdr": {"title": "FISH HDR", "systemImage": "camera.aperture"},
    "pano_hdr": {"title": "PANO HDR", "systemImage": "photo"},
}

PDU_INSTRUMENTS = tuple(
    (
        instrument.id,
        instrument.pdu_title or instrument.title,
        instrument.system_image,
        instrument.pdu_outlet,
    )
    for instrument in PDU_INSTRUMENT_CONTRACTS
)
PDU_INSTRUMENT_BY_ID = {instrument_id: (title, icon, outlet) for instrument_id, title, icon, outlet in PDU_INSTRUMENTS}
PDU_STATE_FRESHNESS_MINUTES = 30.0
AUTOMATIC_PHASE_FRESHNESS_MINUTES = 30.0
UAS_TIER_FRESHNESS_MINUTES = float(os.environ.get("UAS_STALE_AFTER_MINUTES", "5"))
SCIENCE_COLLECTION_FRESHNESS_MINUTES = 120.0
POWERED_COLLECTION_PRODUCTS = {
    "ceilometer": (
        "CEILOMETER_ZARR_PATH",
        "/data/aurora/products/cl61/gamb2le_depolarisation_lidar_ceilometer_aurora.zarr",
        90.0,
    ),
    "cloud-radar": (
        "CLOUD_RADAR_ZARR_PATH",
        "/data/aurora/products/rpgfmcw94/cloud_radar.zarr",
        90.0,
    ),
    "hatpro": (
        "HATPRO_ZARR_PATH",
        "/data/aurora/products/hatprog5/hatpro.zarr",
        180.0,
    ),
}
POWER_FRESH_MINUTES = 30.0
POWER_STALE_MINUTES = 120.0
OPERATIONS_TREND_WINDOW = timedelta(days=7)
OPERATIONS_TREND_CACHE_SECONDS = 60.0
OPERATIONS_TREND_FRESHNESS = timedelta(minutes=30)

OPERATIONS_TREND_STREAM_PREFIXES = (
    "cl61",
    "radar",
    "hatpro",
    "vaisalamet",
    "asfs_logger",
    "asfs_fast_sonic",
    "power",
    "wxcam",
)
OPERATIONS_PDU_STREAM_BY_OUTLET = {5: "cl61", 6: "radar", 8: "hatpro"}
OPERATIONS_STORAGE_KEYS = (
    "host_celine_source_used_pct",
    "host_celine_data_used_pct",
    "host_ass_data_used_pct",
    "host_ass_root_used_pct",
    "host_aps_data_used_pct",
    "host_aps_root_used_pct",
    "aurora_data_used_pct",
    "aurora_root_used_pct",
    "gws_storage_used_pct",
)
_OPERATIONS_TREND_CACHE: dict[str, Any] = {}

# These Science-tab products have no individual PDU outlet state. Their mobile
# status is therefore collection freshness, never an inferred power state.
SCIENCE_DC_INSTRUMENTS = tuple(
    (
        instrument.id,
        instrument.title,
        instrument.system_image,
        instrument.quicklook_subdir,
    )
    for instrument in SCIENCE_DC_INSTRUMENT_CONTRACTS
)

OPERATIONS_STREAMS = (
    {
        "id": "ceilometer",
        "title": "Ceilometer",
        "source": "cl61_source_sync_service_healthy_state",
        "services": ("ceilometer_append_service_healthy_state", "ceilometer_quicklooks_service_healthy_state"),
    },
    {
        "id": "cloud-radar",
        "title": "Cloud Radar",
        "source": "radar_source_sync_service_healthy_state",
        "services": ("radar_append_service_healthy_state", "radar_quicklooks_service_healthy_state"),
    },
    {
        "id": "hatpro",
        "title": "Scanning Microwave Radiometer",
        "source": "hatpro_source_sync_service_healthy_state",
        "services": ("hatpro_append_service_healthy_state", "hatpro_quicklooks_service_healthy_state"),
    },
    {
        "id": "vaisalamet",
        "title": "Meteorology",
        "source": "vaisalamet_source_sync_service_healthy_state",
        "services": ("vaisalamet_append_service_healthy_state", "vaisalamet_quicklooks_service_healthy_state"),
    },
    {
        "id": "asfs-logger",
        "title": "Radiation",
        "source": "asfs_logger_source_sync_service_healthy_state",
        "services": ("asfs_logger_append_service_healthy_state", "asfs_logger_quicklooks_service_healthy_state"),
    },
    {
        "id": "power",
        "title": "Aurora Power Supply",
        "source": "power_source_sync_service_healthy_state",
        "services": ("power_append_service_healthy_state", "power_quicklooks_service_healthy_state"),
    },
    {
        "id": "wxcam",
        "title": "WXcam",
        "source": "wxcam_source_sync_service_healthy_state",
        "services": (
            "wxcam_catalog_service_healthy_state",
            "wxcam_daily_videos_service_healthy_state",
        ),
    },
)


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def env_path(name: str, default: str | Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser()


def quicklook_root() -> Path:
    return env_path("AURORA_QUICKLOOK_ROOT", APP_DIR / "quicklooks")


def menapia_product_root() -> Path:
    """Return the read-only derived-flight product root.

    ``MENAPIA_FLIGHT_PRODUCT_ROOT`` is accepted as a compatibility alias for
    early development deployments, while ``MENAPIA_PRODUCT_ROOT`` is the
    public runtime contract.
    """
    configured = os.environ.get("MENAPIA_PRODUCT_ROOT") or os.environ.get("MENAPIA_FLIGHT_PRODUCT_ROOT")
    return Path(configured or "/data/aurora/products/menapia").expanduser()


def menapia_catalog_path() -> Path:
    return env_path("MENAPIA_CATALOG_PATH", menapia_product_root() / "catalog.json")


def menapia_product_status_path() -> Path:
    return env_path(
        "MENAPIA_PRODUCT_STATUS_PATH",
        "/data/aurora/internal/menapia-products/status.json",
    )


def uas_quicklook_root() -> Path:
    return env_path("UAS_QUICKLOOK_DIR", quicklook_root() / "uas")


def wxcam_daily_video_root() -> Path:
    return env_path("WXCAM_DAILY_VIDEO_DIR", "/data/aurora/products/wxcam/daily_videos")


def wxcam_hourly_thumbnail_root() -> Path:
    return env_path("WXCAM_HOURLY_THUMB_DIR", "/data/aurora/products/wxcam/hourly_thumbnails")


def wxcam_catalog_path() -> Path:
    return env_path("WXCAM_CATALOG_PATH", "/data/aurora/products/wxcam/wxcam_catalog.sqlite")


def auroracam_root() -> Path:
    return env_path("AURORACAM_RAW_ROOT", os.environ.get("AURORACAM_ROOT", "/project/aurora/raw/auroracam"))


def auroracam_preview_cache_root() -> Path:
    return env_path("AURORA_MOBILE_PREVIEW_CACHE", "/var/cache/aurora-mobile-api/auroracam")


def power_display_summary_path() -> Path:
    return env_path("POWER_DISPLAY_SUMMARY_ZARR_PATH", "/data/aurora/products/power/power_display_summary.zarr")


def power_display_section_path(section: str) -> Path:
    """Return the compact Power product used by one presentation section."""
    if section == "current":
        return env_path("POWER_CURRENT_DISPLAY_ZARR_PATH", "/data/aurora/products/power/power_current_display.zarr")
    if section == "forecast":
        return env_path("POWER_FORECAST_DISPLAY_ZARR_PATH", "/data/aurora/products/power/power_forecast_display.zarr")
    return power_display_summary_path()


def power_prewarm_path(section: str) -> Path | None:
    """Return the published Plotly JSON for one Power section, if valid."""
    if section not in {"current", "forecast"}:
        return None
    root = env_path("AURORA_INTERACTIVE_PREWARM_DIR", "/data/aurora/products/dashboard/prewarm")
    return root / f"power_{section}_latest_interactive.json"


def _representative_power_indices(values, maximum: int = MOBILE_POWER_MAX_POINTS):
    """Keep endpoints and local extrema without overloading native Charts."""
    import numpy as np

    count = len(values)
    if count <= maximum:
        return np.arange(count, dtype=int)
    # Two extrema per bucket retain peaks/dips that uniform stride sampling can
    # hide, while bounding every native trace to the mobile contract.
    bucket_count = max(1, (maximum - 2) // 2)
    edges = np.linspace(0, count, bucket_count + 1, dtype=int)
    selected = {0, count - 1}
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        if right <= left:
            continue
        bucket = values[left:right]
        selected.add(left + int(np.argmin(bucket)))
        selected.add(left + int(np.argmax(bucket)))
    indices = np.asarray(sorted(selected), dtype=int)
    if len(indices) <= maximum:
        return indices
    return indices[np.linspace(0, len(indices) - 1, maximum, dtype=int)]


def power_operating_scenario_paths() -> tuple[Path, ...]:
    """Locate the authoritative operating-plan product for native clients."""
    configured = env_path(
        "POWER_OPERATING_SCENARIOS_ZARR_PATH",
        "/data/aurora/products/power/power_operating_scenarios.zarr",
    )
    mirrored = Path("/data/aurora/products/power/power_operating_scenarios.zarr")
    return tuple(dict.fromkeys((configured, mirrored)))


def uas_mqtt_log_path() -> Path:
    return env_path("UAS_MQTT_LOG_PATH", "/project/aurora/raw/menapia/menapia_mqtt.log")


def operations_snapshot_path() -> Path:
    return env_path("OPS_MONITOR_LATEST_SNAPSHOT", "/project/aurora/raw/ops_monitor/latest.json")


def operations_health_path() -> Path:
    return env_path("OPS_MONITOR_LATEST_HEALTH", "/data/aurora/products/ops_monitor/health/latest_health.json")


def archive_health_path() -> Path:
    return env_path(
        "ARCHIVE_HEALTH_PATH",
        "/data/aurora/internal/archive_status/health-v1.json",
    )


def operations_alert_state_path() -> Path:
    return env_path("OPS_MONITOR_ALERT_STATE", "/data/aurora/products/ops_monitor/alerts/state.json")


def operations_zarr_path() -> Path:
    return env_path("OPS_MONITOR_ZARR_PATH", "/data/aurora/products/ops_monitor/ops_monitor.zarr")


def read_json_file(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        return {"_error": f"Invalid JSON: {exc}"}
    except OSError as exc:
        return {"_error": str(exc)}
    return value if isinstance(value, dict) else {"value": value}


def file_record(path: Path) -> dict[str, Any]:
    try:
        stat_result = path.stat()
    except OSError:
        return {"exists": False}
    return {
        "exists": True,
        "sizeBytes": stat_result.st_size,
        "modifiedAt": datetime.fromtimestamp(stat_result.st_mtime, UTC).isoformat().replace("+00:00", "Z"),
    }


def display_artifacts() -> dict[str, Any]:
    """Return the latest publishable dashboard-artifact manifest when present."""
    return load_manifest()


def normalize_level(value: Any) -> str:
    if isinstance(value, bool):
        return "green" if value else "red"
    if isinstance(value, int | float):
        if value == 1:
            return "green"
        if value == 0:
            return "red"
    text = str(value or "").strip().lower()
    if text in {"green", "ok", "healthy", "good", "1", "true"}:
        return "green"
    if text in {"amber", "yellow", "warning", "warn"}:
        return "amber"
    if text in {"red", "critical", "failed", "error", "0", "false"}:
        return "red"
    return "unknown"


def level_from_booleans(values: list[Any]) -> str:
    known = [value for value in values if value is not None]
    if not known:
        return "unknown"
    if any(str(value).strip().lower() in {"0", "false", "red", "failed", "error"} for value in known):
        return "red"
    return "green"


def media_url(*parts: str) -> str:
    return "/media/" + "/".join(part.strip("/") for part in parts)


def versioned_media_url(path: Path, *parts: str) -> str:
    """Return a stable media route that changes when the served bytes change."""
    url = media_url(*parts)
    try:
        stat_result = path.stat()
    except OSError:
        return url
    return f"{url}?v={stat_result.st_mtime_ns:x}-{stat_result.st_size:x}"


def dashboard_revision() -> str | None:
    configured = os.environ.get("AURORA_DASHBOARD_REVISION", "").strip()
    if configured:
        return configured

    head_path = APP_DIR / ".git" / "HEAD"
    try:
        head = head_path.read_text(encoding="utf-8").strip()
        if head.startswith("ref: "):
            ref_path = APP_DIR / ".git" / head.removeprefix("ref: ")
            head = ref_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return head[:12] if head else None


def deployment_descriptor() -> dict[str, Any]:
    domain = os.environ.get("AURORA_DOMAIN", "").strip() or "data-ocean.gamb2le.co.uk"
    environment = os.environ.get("AURORA_SITE_ENV", "").strip().lower()
    if not environment:
        environment = "development" if "data-ocean" in domain else "production"
    data_role = "live-mirror" if environment == "development" else "authoritative"
    return {
        "environment": environment,
        "domain": domain,
        "dashboardURL": f"https://{domain}/app",
        "dataRole": data_role,
        "revision": dashboard_revision(),
    }


def manifest() -> dict[str, Any]:
    return {
        "serverTime": utc_now_iso(),
        "schemaVersion": 3,
        "minimumRefreshIntervalSeconds": 60,
        "deployment": deployment_descriptor(),
        "sections": [
            {"id": "overview", "title": "Overview", "systemImage": "rectangle.3.group"},
            {"id": "power", "title": "Power", "systemImage": "bolt.batteryblock"},
            {"id": "plots", "title": "Plots", "systemImage": "chart.xyaxis.line"},
            {"id": "camera", "title": "Camera", "systemImage": "camera"},
            {"id": "ops", "title": "Ops", "systemImage": "gauge.with.dots.needle.bottom.50percent"},
        ],
        "instruments": [
            {
                "id": instrument.id,
                "title": instrument.title,
                "systemImage": instrument.system_image,
                "visible": instrument.visible,
                "supportsSummary": instrument.summary_supported,
                "supportsScienceQuicklooks": bool(instrument.science_prefixes),
                "supportsHousekeepingQuicklooks": bool(instrument.housekeeping_prefixes),
            }
            for instrument in INSTRUMENTS
        ],
        "wxcamStreams": [
            {"id": stream_id, "title": spec["title"], "systemImage": spec["systemImage"]}
            for stream_id, spec in WXCAM_STREAMS.items()
        ],
        # This is the cross-platform contract. Native clients can choose a
        # compact presentation, while the browser remains the full explorer.
        "capabilities": {
            "shared": [
                "power.current_system_ecmwf_p10_p90",
                "power.assigned_pdu_outlets",
                "operations.instrument_state",
                "operations.live_status",
                "quicklooks.science_housekeeping",
                "camera.auroracam_wxcam",
                "uas.flight_profiles",
            ],
            "browser": [
                "explore.arbitrary_variables_ranges",
                "plots.plotly_investigation",
                "uas.full_history_events",
                "camera.shareable_state",
            ],
            "native": [
                "overview.cached_snapshot",
                "overview.endpoint_failover",
                "overview.dynamic_type",
                "overview.pull_to_refresh",
            ],
        },
    }


def _archive_operator_status(archive_health: dict[str, Any]) -> dict[str, str]:
    """Translate the fail-closed archive contract into operator language.

    Infrastructure deliberately stays red whenever fresh verification evidence
    is unavailable so retention cannot delete anything.  When the only problem
    is a failed inventory listing, the previous complete report is clean, and
    all measured gap counters are zero, the operator-facing state is amber: the
    archive is not known to be incomplete, but new pruning remains paused.
    """
    supplied = archive_health.get("operator_status")
    if isinstance(supplied, dict):
        level = normalize_level(supplied.get("level"))
        title = str(supplied.get("title") or "").strip()
        detail = str(supplied.get("detail") or "").strip()
        if level in {"green", "amber", "red"} and title and detail:
            return {"level": level, "title": title, "detail": detail}

    contract_level = normalize_level(archive_health.get("overall_level"))
    failures = archive_health.get("failures")
    if not isinstance(failures, list):
        failures = []
    failure_text = [str(item) for item in failures]
    metrics = archive_health.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}
    evidence = archive_health.get("evidence")
    if not isinstance(evidence, dict):
        evidence = {}
    gate = evidence.get("object_store_gate")
    if not isinstance(gate, dict):
        gate = {}
    progress = evidence.get("object_store_inventory_progress")
    if not isinstance(progress, dict):
        progress = {}

    gap_keys = (
        "streams_gws_issue_count",
        "object_store_all_missing_count",
        "object_store_all_mismatch_count",
        "gws_all_missing_count",
        "gws_all_mismatch_count",
    )
    measured_gap_free = all(
        isinstance(metrics.get(key), int | float) and float(metrics[key]) == 0
        for key in gap_keys
    )
    last_complete_clean = bool(gate.get("clean")) and bool(
        gate.get("stable_parity")
    )
    verification_only = bool(failure_text) and all(
        item.startswith("object_store_evidence_stale_hours=")
        or item.startswith("object_store_inventory_progress_stale_minutes=")
        or item
        == "archive_service_unhealthy=aurora-object-store-inventory.service"
        for item in failure_text
    )
    inventory_failed = progress.get("state") == "failed"

    if (
        contract_level == "red"
        and measured_gap_free
        and last_complete_clean
        and verification_only
        and inventory_failed
    ):
        error = str(progress.get("error") or "")
        if "asfs_fast_gas" in error:
            target = "ASFS fast-gas products"
        else:
            target = "archive products"
        return {
            "level": "amber",
            "title": "Archive verification failed",
            "detail": (
                f"JASMIN object-store listing timed out for {target}. "
                "Last complete verification was clean. New pruning is paused "
                "until verification succeeds."
            ),
        }

    if contract_level == "green":
        return {
            "level": "green",
            "title": "Archive verification is healthy",
            "detail": "Archive verification is healthy",
        }
    return {
        "level": contract_level,
        "title": (
            "Archive health is red"
            if contract_level == "red"
            else "Archive verification is unavailable"
        ),
        "detail": "; ".join(failure_text) or "No failure detail was supplied.",
    }


def operations() -> dict[str, Any]:
    health = read_json_file(operations_health_path())
    snapshot = read_json_file(operations_snapshot_path())
    archive_health = read_json_file(archive_health_path())
    archive_status = _archive_operator_status(archive_health)
    archive_metrics = archive_health.get("metrics", {})
    if isinstance(archive_metrics, dict):
        snapshot.update(archive_metrics)
    snapshot["archive_contract_level"] = archive_health.get("overall_level")
    snapshot["archive_health_level"] = archive_status["level"]
    snapshot["archive_health_failures"] = archive_health.get("failures", [])
    snapshot["archive_health_detail"] = archive_status["detail"]
    alert_state = read_json_file(operations_alert_state_path())

    health_error = health.get("_error")
    snapshot_error = snapshot.get("_error")
    overall = normalize_level(
        archive_status["level"]
        if archive_health
        else health.get("overall_level") or snapshot.get("overall_level")
    )

    stream_states = [_stream_state(snapshot, spec) for spec in OPERATIONS_STREAMS]
    power_alert = _power_freshness_alert(snapshot)
    if power_alert:
        for stream in stream_states:
            if stream["id"] == "power":
                stream["level"] = _worst_level(
                    stream["level"], power_alert["level"]
                )
                stream["sourceHealthy"] = power_alert["level"]
                stream["detail"] = power_alert["detail"]
                break
    if any(stream["level"] == "red" for stream in stream_states):
        overall = "red"
    elif overall == "unknown" and any(
        stream["level"] == "green" for stream in stream_states
    ):
        overall = "green"

    updated_at = (
        archive_health.get("generated_at")
        or health.get("time_utc")
        or health.get("snapshot_time_utc")
        or snapshot.get("time_utc")
        or snapshot.get("snapshot_time_utc")
    )
    active_alerts = [
        alert
        for alert in _active_alerts(alert_state)
        if alert.get("id") not in {"archive:health_red", "archive:verification"}
    ]
    if power_alert:
        active_alerts = [
            alert for alert in active_alerts if alert.get("id") != "power:freshness"
        ]
        active_alerts = [power_alert, *active_alerts]
        overall = _worst_level(overall, power_alert["level"])
    if normalize_level(archive_status["level"]) in {"amber", "red"}:
        if archive_health:
            active_alerts = [
                {
                    "id": "archive:verification",
                    "title": archive_status["title"],
                    "level": archive_status["level"],
                    "detail": archive_status["detail"],
                },
                *active_alerts,
            ]

    check_counts = _check_counts(health, stream_states)
    archive_level = normalize_level(archive_status["level"])
    if archive_level in check_counts:
        check_counts[archive_level] = check_counts.get(archive_level, 0) + 1

    return {
        "serverTime": utc_now_iso(),
        "updatedAt": updated_at,
        "overallLevel": overall,
        "summary": _operations_summary(
            overall,
            stream_states,
            health_error,
            snapshot_error,
            archive_status,
            power_alert,
        ),
        "checkCounts": check_counts,
        "streamStates": stream_states,
        "rootCauseGroups": _root_cause_groups(snapshot, stream_states),
        "archiveDelivery": _archive_delivery(archive_health, snapshot),
        "archiveStatus": {
            "delivery": archive_health.get("delivery", {}),
            "durability": archive_health.get("durability", {}),
            "verification": archive_health.get("verification", {}),
            "retention": archive_health.get("retention", {}),
        },
        "alerts": active_alerts,
        "trendCards": _trend_cards(snapshot),
        "sources": {
            "health": {**file_record(operations_health_path()), "path": str(operations_health_path())},
            "snapshot": {**file_record(operations_snapshot_path()), "path": str(operations_snapshot_path())},
            "archiveHealth": {**file_record(archive_health_path()), "path": str(archive_health_path())},
        },
    }


def _stream_state(snapshot: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    source_value = snapshot.get(str(spec["source"]))
    service_values = [snapshot.get(str(key)) for key in spec["services"]]
    source_level = level_from_booleans([source_value])
    service_level = level_from_booleans(service_values)
    level = "red" if "red" in {source_level, service_level} else "green" if "green" in {source_level, service_level} else "unknown"

    failed_services = [
        key.removesuffix("_service_healthy_state").replace("_", " ")
        for key, value in zip(spec["services"], service_values, strict=False)
        if normalize_level(value) == "red"
    ]
    if normalize_level(source_value) == "red":
        detail = "Source sync is unhealthy"
    elif failed_services:
        detail = "Unhealthy: " + ", ".join(failed_services[:3])
    elif level == "green":
        detail = "Source and processing services healthy"
    else:
        detail = "No current status sample"

    return {
        "id": spec["id"],
        "title": spec["title"],
        "level": level,
        "detail": detail,
        "sourceHealthy": source_value,
        "serviceHealthyCount": sum(1 for value in service_values if normalize_level(value) == "green"),
        "serviceCount": len(service_values),
    }


def _operations_summary(
    overall: str,
    streams: list[dict[str, Any]],
    health_error: Any,
    snapshot_error: Any,
    archive_status: dict[str, str],
    power_alert: dict[str, Any] | None = None,
) -> str:
    if health_error:
        return f"Health JSON error: {health_error}"
    if snapshot_error:
        return f"Snapshot JSON error: {snapshot_error}"
    if power_alert and normalize_level(power_alert.get("level")) == "red":
        return str(power_alert["title"])
    red_count = sum(1 for stream in streams if stream["level"] == "red")
    unknown_count = sum(1 for stream in streams if stream["level"] == "unknown")
    if red_count:
        return f"{red_count} stream group{'s' if red_count != 1 else ''} need attention"
    if power_alert:
        return str(power_alert["title"])
    if unknown_count == len(streams):
        return "No operations snapshot available"
    if archive_status.get("level") == "amber":
        return "Archive verification is delayed; the last complete check was clean"
    if overall == "green":
        return "All visible stream groups are healthy"
    return "Operations status is partially available"


def _check_counts(health: dict[str, Any], streams: list[dict[str, Any]]) -> dict[str, int]:
    counts = health.get("check_counts")
    if isinstance(counts, dict):
        return {str(key): int(value) for key, value in counts.items() if isinstance(value, int | float)}
    return {
        "green": sum(1 for stream in streams if stream["level"] == "green"),
        "amber": sum(1 for stream in streams if stream["level"] == "amber"),
        "red": sum(1 for stream in streams if stream["level"] == "red"),
        "unknown": sum(1 for stream in streams if stream["level"] == "unknown"),
    }


def _root_cause_groups(snapshot: dict[str, Any], streams: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source_issues = [stream["title"] for stream in streams if normalize_level(stream.get("sourceHealthy")) == "red"]
    service_issues = [stream["title"] for stream in streams if stream["level"] == "red" and stream["title"] not in source_issues]
    storage_level = "red" if any(float(snapshot.get(key, 0) or 0) >= 80 for key in ("aurora_data_used_pct", "aurora_root_used_pct", "gws_used_pct")) else "green" if snapshot else "unknown"
    dashboard_level = level_from_booleans(
        [
            snapshot.get("dashboard_http_ok_state"),
            snapshot.get("failover_primary_dashboard_http_ok_state"),
            snapshot.get("failover_standby_dashboard_http_ok_state"),
        ]
    )
    archive_level = normalize_level(snapshot.get("archive_health_level"))
    archive_failures = snapshot.get("archive_health_failures")
    if not isinstance(archive_failures, list):
        archive_failures = []
    archive_detail = snapshot.get("archive_health_detail")
    delivery_pending = int(snapshot.get("archive_delivery_pending_count") or 0)
    delivery_gws = int(snapshot.get("archive_delivery_gws_pending_count") or 0)
    delivery_object = int(
        snapshot.get("archive_delivery_object_store_pending_count") or 0
    )
    delivery_oldest = float(
        snapshot.get("archive_delivery_oldest_pending_age_minutes") or 0
    )
    if delivery_pending:
        delivery_detail = (
            f"Newest-first live queue: {delivery_pending:,} files pending "
            f"(GWS {delivery_gws:,}, object store {delivery_object:,}); "
            f"oldest {delivery_oldest:.1f} min."
        )
    elif snapshot:
        delivery_detail = "Newest-first live delivery queue is clear."
    else:
        delivery_detail = "Live delivery queue is unavailable."
    archive_root_detail = " ".join(
        part
        for part in (
            delivery_detail,
            str(archive_detail or "").strip(),
        )
        if part
    )
    return [
        {"id": "source", "title": "Source freshness", "level": "red" if source_issues else "green" if snapshot else "unknown", "detail": ", ".join(source_issues[:4]) if source_issues else "No source freshness issues"},
        {"id": "processing", "title": "Local processing", "level": "red" if service_issues else "green" if snapshot else "unknown", "detail": ", ".join(service_issues[:4]) if service_issues else "Append, catalog, and quicklook services healthy"},
        {"id": "storage", "title": "Storage pressure", "level": storage_level, "detail": "Storage is below alert thresholds" if storage_level == "green" else "Storage needs attention"},
        {
            "id": "archive",
            "title": "GWS and object-store archive",
            "level": archive_level,
            "detail": (
                archive_root_detail
                if archive_root_detail
                else "; ".join(str(item) for item in archive_failures)
                if archive_failures
                else "Archive verification is healthy"
                if archive_level == "green"
                else "Archive verification evidence is unavailable"
            ),
        },
        {"id": "dashboard", "title": "Public dashboard", "level": dashboard_level, "detail": "Dashboard endpoint probes are healthy" if dashboard_level == "green" else "Dashboard endpoint probe needs attention"},
    ]


def _archive_delivery(
    archive_health: dict[str, Any], snapshot: dict[str, Any]
) -> dict[str, Any]:
    delivery = archive_health.get("delivery")
    if not isinstance(delivery, dict):
        delivery = {}
    verification = archive_health.get("verification")
    if not isinstance(verification, dict):
        verification = {}
    evidence = archive_health.get("evidence")
    if not isinstance(evidence, dict):
        evidence = {}
    progress = evidence.get("object_store_inventory_progress")
    if not isinstance(progress, dict):
        progress = {}
    return {
        "level": normalize_level(delivery.get("level") or "unknown"),
        "mode": str(delivery.get("mode") or "newest-first"),
        "pendingFiles": int(
            delivery.get("pending_files")
            or snapshot.get("archive_delivery_pending_count")
            or 0
        ),
        "pendingBytes": int(
            delivery.get("pending_bytes")
            or snapshot.get("archive_delivery_pending_bytes")
            or 0
        ),
        "gwsPendingFiles": int(
            delivery.get("gws_pending_files")
            or snapshot.get("archive_delivery_gws_pending_count")
            or 0
        ),
        "objectStorePendingFiles": int(
            delivery.get("object_store_pending_files")
            or snapshot.get("archive_delivery_object_store_pending_count")
            or 0
        ),
        "oldestPendingAgeMinutes": float(
            delivery.get("oldest_pending_age_minutes")
            or snapshot.get("archive_delivery_oldest_pending_age_minutes")
            or 0
        ),
        "lastSuccessAgeMinutes": delivery.get("last_success_age_minutes")
        if delivery.get("last_success_age_minutes") is not None
        else snapshot.get("archive_delivery_last_success_age_minutes"),
        "strictAudit": {
            "state": verification.get("state") or progress.get("state"),
            "completedJobs": verification.get("completed_jobs")
            or progress.get("completed_jobs", []),
            "totalJobs": int(
                verification.get("total_jobs") or progress.get("total_jobs") or 0
            ),
            "currentJobs": verification.get("current_jobs")
            or progress.get("current_jobs")
            or ([progress.get("current_job")] if progress.get("current_job") else []),
            "phase": verification.get("phase") or progress.get("phase"),
            "lastCompleteAt": verification.get("last_complete_at"),
            "lastCertifiedRawAt": verification.get("last_certified_raw_at"),
        },
    }


def _trend_cards(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    history = _operations_trend_values(_intentionally_paused_streams())
    specs = (
        ("storage", "Worst storage use", "%", OPERATIONS_STORAGE_KEYS),
        ("battery-soc", "APS state of charge", "%", ("aps_battery_soc_pct", "BatterySOC", "power_battery_soc")),
        ("battery-voltage", "APS battery voltage", "V", ("aps_battery_voltage_v", "DCInverterVolts", "power_battery_voltage")),
        ("source-lag", "Worst source lag", "min", ("worst_source_lag_min", "source_lag_max_min")),
        ("gws-lag", "Worst GWS lag", "min", ("worst_gws_lag_min", "gws_lag_max_min")),
    )
    cards: list[dict[str, Any]] = []
    for card_id, title, unit, keys in specs:
        values = [snapshot.get(key) for key in keys if isinstance(snapshot.get(key), int | float)]
        if values:
            value = max(values) if card_id in {"storage", "source-lag", "gws-lag"} else values[0]
        else:
            value = history.get(card_id)
        cards.append({"id": card_id, "title": title, "value": value, "unit": unit, "level": _trend_level(card_id, value)})
    return cards


def _intentionally_paused_streams() -> set[str]:
    states, _detail = _pdu_power_snapshot()
    return {
        prefix
        for outlet, prefix in OPERATIONS_PDU_STREAM_BY_OUTLET.items()
        if states.get(outlet) is False
    }


def _operations_trend_values(paused_prefixes: set[str]) -> dict[str, float]:
    """Read only the numeric tails needed by the native operations cards."""
    path = operations_zarr_path()
    try:
        metadata_mtime = (path / ".zmetadata").stat().st_mtime_ns
    except OSError:
        return {}

    cache_key = (str(path), metadata_mtime, tuple(sorted(paused_prefixes)))
    now_monotonic = monotonic()
    if (
        _OPERATIONS_TREND_CACHE.get("key") == cache_key
        and now_monotonic < float(_OPERATIONS_TREND_CACHE.get("expires_at", 0.0))
    ):
        return dict(_OPERATIONS_TREND_CACHE.get("values", {}))

    try:
        import numpy as np
        import zarr

        group = zarr.open_consolidated(str(path), mode="r")
        now = datetime.now(UTC)
        start = _operations_trend_start_index(group, now)
        time_array = group.get("time")
        latest_time = (
            _decode_cf_time(time_array[-1], str(time_array.attrs.get("units", "")))
            if time_array is not None and time_array.shape
            else None
        )
        if latest_time is None or now - latest_time > OPERATIONS_TREND_FRESHNESS:
            raise ValueError("latest operations trend sample is stale")

        def latest(name: str) -> float | None:
            if name not in group:
                return None
            latest_index = max(start, len(group[name]) - 1)
            values = np.asarray(
                group[name][latest_index:],
                dtype=np.float64,
            )
            if not values.size or not np.isfinite(values[-1]):
                return None
            return float(values[-1])

        def latest_max(names: tuple[str, ...]) -> float | None:
            values = [value for name in names if (value := latest(name)) is not None]
            return max(values) if values else None

        active_prefixes = tuple(
            prefix for prefix in OPERATIONS_TREND_STREAM_PREFIXES if prefix not in paused_prefixes
        )
        values = {
            "storage": latest_max(OPERATIONS_STORAGE_KEYS),
            "battery-soc": latest("aps_battery_soc_pct"),
            "battery-voltage": latest("aps_battery_voltage_v"),
            "source-lag": latest_max(tuple(f"{prefix}_source_age_min" for prefix in active_prefixes)),
            "gws-lag": latest_max(tuple(f"{prefix}_gws_lag_min" for prefix in active_prefixes)),
        }
        result = {key: value for key, value in values.items() if value is not None}
    except (ImportError, OSError, TypeError, ValueError, KeyError):
        result = {}

    _OPERATIONS_TREND_CACHE.update(
        {
            "key": cache_key,
            "expires_at": now_monotonic + OPERATIONS_TREND_CACHE_SECONDS,
            "values": result,
        }
    )
    return dict(result)


def _operations_trend_start_index(group: Any, now: datetime) -> int:
    """Locate the seven-day tail without decoding the full operations dataset."""
    import numpy as np

    if "time" not in group or not group["time"].shape:
        return 0
    time_values = np.asarray(group["time"][:])
    if time_values.size == 0:
        return 0

    parameters = _cf_time_parameters(str(group["time"].attrs.get("units", "")))
    if parameters:
        try:
            origin, seconds_per_unit = parameters
            cutoff = now.astimezone(UTC) - OPERATIONS_TREND_WINDOW
            cutoff_value = (cutoff - origin).total_seconds() / seconds_per_unit
            return int(np.searchsorted(time_values, cutoff_value, side="left"))
        except (OverflowError, TypeError, ValueError):
            pass

    # A conservative one-minute fallback keeps at least seven days while still
    # bounding reads when older stores use an unexpected CF-time encoding.
    return max(0, int(time_values.size) - 7 * 24 * 60)


def _cf_time_parameters(units: str) -> tuple[datetime, float] | None:
    match = re.fullmatch(
        r"(seconds|milliseconds|microseconds|nanoseconds|minutes|hours|days) since (.+)",
        units,
    )
    if not match:
        return None
    seconds_per_unit = {
        "nanoseconds": 1e-9,
        "microseconds": 1e-6,
        "milliseconds": 1e-3,
        "seconds": 1.0,
        "minutes": 60.0,
        "hours": 3600.0,
        "days": 86400.0,
    }
    try:
        origin = datetime.fromisoformat(match.group(2).replace("Z", "+00:00"))
    except ValueError:
        return None
    if origin.tzinfo is None:
        origin = origin.replace(tzinfo=UTC)
    return origin.astimezone(UTC), seconds_per_unit[match.group(1)]


def _decode_cf_time(value: Any, units: str) -> datetime | None:
    parameters = _cf_time_parameters(units)
    if parameters is None:
        return None
    origin, seconds_per_unit = parameters
    try:
        return origin + timedelta(seconds=float(value) * seconds_per_unit)
    except (OverflowError, TypeError, ValueError):
        return None


def _trend_level(card_id: str, value: Any) -> str:
    if not isinstance(value, int | float):
        return "unknown"
    if card_id == "storage":
        return "red" if value >= 90 else "amber" if value >= 80 else "green"
    if card_id == "battery-soc":
        return "red" if value <= 40 else "amber" if value < 50 else "green"
    if card_id == "battery-voltage":
        return "red" if value < 50 else "amber" if value < 52 else "green"
    return "red" if value >= 180 else "amber" if value >= 90 else "green"


def _active_alerts(alert_state: dict[str, Any]) -> list[dict[str, Any]]:
    active = alert_state.get("active") or alert_state.get("active_alerts") or alert_state.get("alerts") or []
    if isinstance(active, dict):
        iterator = active.items()
    elif isinstance(active, list):
        iterator = enumerate(active)
    else:
        return []
    alerts = []
    for key, value in iterator:
        if isinstance(value, dict):
            # The alert sender keeps recovered entries in state.json for
            # notification history.  Only entries that are still active
            # belong in the mobile operations payload.
            if value.get("active") is False:
                continue
            title = str(value.get("title") or value.get("kind") or key)
            level = normalize_level(value.get("level") or value.get("severity") or "red")
            detail = str(value.get("message") or value.get("detail") or "")
        else:
            title = str(value)
            level = "red"
            detail = ""
        alerts.append({"id": str(key), "title": title, "level": level, "detail": detail})
    return alerts


def overview() -> dict[str, Any]:
    """Return the lightweight operational cards shown first in the native app."""
    snapshot = read_json_file(operations_snapshot_path())
    status = operations()
    latest_cameras = auroracam("latest")
    camera_times = [record.get("timeUTC") for record in latest_cameras["frames"] if record.get("timeUTC")]
    latest_camera_time = max(camera_times) if camera_times else None
    # Use the cached measured timestamp that the operations collector publishes
    # with the battery metrics. Opening the wide display Zarr here added several
    # seconds to every Overview/API request. Keep the direct read only as a
    # compatibility fallback for older snapshots.
    latest_power_time = _snapshot_power_time(snapshot) or _latest_power_time()
    power_level = _age_level(
        latest_power_time,
        POWER_FRESH_MINUTES,
        POWER_STALE_MINUTES,
    )
    depletion_value, depletion_detail = _battery_depletion_text(
        snapshot,
        power_level=power_level,
        power_time=latest_power_time,
    )
    environmental_cards = _environmental_signal_cards()
    science_source_times = {
        "vaisalamet": next(
            (card.get("updatedAt") for card in environmental_cards if card.get("id") == "air-temperature"),
            None,
        ),
        "asfs-logger": next(
            (
                card.get("updatedAt")
                for card in environmental_cards
                if card.get("id") in {"shortwave-down", "wind-speed", "kt15"}
            ),
            None,
        ),
    }
    cards = [
        _overview_card("operations", "Operations", _operations_value(status["overallLevel"]), status["overallLevel"], status.get("updatedAt"), status["summary"]),
        _overview_card("battery-soc", "State of Charge", _metric_text(snapshot, ("aps_battery_soc_pct", "BatterySOC"), "%"), _freshness_guarded_level(_trend_level("battery-soc", _metric_value(snapshot, ("aps_battery_soc_pct", "BatterySOC"))), power_level), latest_power_time, _metric_age_detail(snapshot, "aps_battery_soc_age_min")),
        _overview_card("battery-voltage", "Battery Voltage", _metric_text(snapshot, ("aps_battery_voltage_v", "DCInverterVolts"), "V"), _freshness_guarded_level(_trend_level("battery-voltage", _metric_value(snapshot, ("aps_battery_voltage_v", "DCInverterVolts"))), power_level), latest_power_time, _metric_age_detail(snapshot, "aps_battery_voltage_age_min")),
        _overview_card("battery-depletion", "Time to Depleted", depletion_value, _freshness_guarded_level(_battery_depletion_level(snapshot), power_level), latest_power_time, depletion_detail),
        _overview_card("power", "Power Data", _power_time_text(latest_power_time), power_level, latest_power_time, _power_age_text(latest_power_time)),
        _overview_card("auroracam", "AURORACam", _age_text(latest_camera_time), _age_level(latest_camera_time, 30, 120), latest_camera_time, "Latest station camera frame"),
        *environmental_cards,
    ]
    return {
        "serverTime": utc_now_iso(),
        "cards": cards,
        "instrumentPower": _instrument_power_states(snapshot, science_source_times),
        "activeAlerts": status["alerts"],
    }


def _environmental_signal_cards() -> list[dict[str, Any]]:
    """Return the latest lightweight station-environment measurements.

    These cards intentionally read only the final sample of the existing
    meteorology and ASFS logger products.  They provide operational context on
    the overview without loading a chart or a full science data window.
    """
    meteorology = _latest_zarr_sample(
        env_path("VAISALAMET_ZARR_PATH", "/data/aurora/products/vaisalamet/vaisalamet.zarr"),
        ("t2_t",),
    )
    radiation = _latest_zarr_sample(
        env_path("ASFS_LOGGER_ZARR_PATH", "/data/aurora/products/asfs_logger/asfs_logger.zarr"),
        ("sr30_swd_Irr_Avg", "kt15_tem_Avg", "metek_x_out_Avg", "metek_y_out_Avg"),
    )

    cards: list[dict[str, Any]] = []
    if (sample := radiation.get("sr30_swd_Irr_Avg")) is not None:
        cards.append(_environmental_overview_card("shortwave-down", "Shortwave radiation down", sample, radiation["time"], "W/m2", "SR30 downwelling"))
    if (sample := radiation.get("metek_x_out_Avg")) is not None and (other := radiation.get("metek_y_out_Avg")) is not None:
        cards.append(_environmental_overview_card("wind-speed", "Wind speed", math.hypot(sample, other), radiation["time"], "m/s", "Metek horizontal wind"))
    if (sample := meteorology.get("t2_t")) is not None:
        cards.append(_environmental_overview_card("air-temperature", "2 m temperature", sample, meteorology["time"], "C", "Vaisala MET"))
    if (sample := radiation.get("kt15_tem_Avg")) is not None:
        cards.append(_environmental_overview_card("kt15", "KT15 surface temperature", sample, radiation["time"], "C", "KT15 surface sensor"))
    return cards


def environmental_signal_cards() -> list[dict[str, Any]]:
    """Return overview-ready station-environment cards for browser consumers."""
    return _environmental_signal_cards()


def _environmental_overview_card(card_id: str, title: str, value: float, sample_time: str, unit: str, source: str) -> dict[str, Any]:
    return _overview_card(
        card_id,
        title,
        f"{value:.1f} {unit}",
        _age_level(sample_time, 30, 120),
        sample_time,
        f"{source}; {_power_age_text(sample_time).removeprefix('Updated ')}",
    )


def _latest_zarr_sample(path: Path, variables: tuple[str, ...]) -> dict[str, Any]:
    """Read finite values from the final sample of a compact operational Zarr."""
    dataset = None
    try:
        import numpy as np
        import pandas as pd
        import xarray as xr

        if not path.exists():
            return {}
        dataset = xr.open_zarr(path, consolidated=True, chunks=None)
        if "time" not in dataset or dataset.sizes.get("time", 0) == 0:
            return {}
        sample_time = pd.Timestamp(dataset["time"].values[-1]).isoformat() + "Z"
        result: dict[str, Any] = {"time": sample_time}
        for variable in variables:
            if variable not in dataset or dataset[variable].dims != ("time",):
                continue
            value = float(np.asarray(dataset[variable].isel(time=-1).values))
            if math.isfinite(value):
                result[variable] = value
        return result
    except Exception:
        return {}
    finally:
        if dataset is not None:
            dataset.close()


def _instrument_power_states(
    snapshot: dict[str, Any],
    science_source_times: dict[str, str | None] | None = None,
) -> list[dict[str, Any]]:
    """Return PDU power states plus collection states for DC science streams."""
    states, detail = _pdu_power_snapshot()
    powered_labels = _powered_instrument_labels(states)
    science_source_times = science_source_times or {}

    pdu_rows = [
        _pdu_instrument_status(
            instrument_id,
            states,
            detail,
            powered_labels,
            collection_detail=(
                _powered_collection_detail(instrument_id)
                if states.get(outlet) is True
                else None
            ),
        )
        for instrument_id, _title, _icon, outlet in PDU_INSTRUMENTS
    ]
    science_rows = []
    for instrument_id, title, icon, prefix in SCIENCE_DC_INSTRUMENTS:
        source_age = _metric_value(snapshot, (f"{prefix}_source_age_min",))
        recent = snapshot.get(f"{prefix}_source_recent_state")
        if source_age is None:
            source_time = _parse_utc(science_source_times.get(instrument_id))
            if source_time is not None:
                source_age = max((datetime.now(UTC) - source_time).total_seconds() / 60, 0)
                recent = int(source_age <= SCIENCE_COLLECTION_FRESHNESS_MINUTES)
        if recent == 1:
            state, level = "Collecting", "green"
        elif recent == 0:
            state, level = "No recent data", "red"
        else:
            state, level = "Unknown", "amber"
        detail = (
            f"Source sample {_duration_text(source_age / 60)} old"
            if source_age is not None
            else "Source freshness unavailable"
        )
        science_rows.append(
            {
                "id": instrument_id,
                "title": title,
                "systemImage": icon,
                "state": state,
                "level": level,
                "detail": detail,
            }
        )
    # Collection freshness is the first operational signal on the mobile overview.
    # Keep it ahead of PDU outlet state regardless of changes to either inventory.
    return [*science_rows, *pdu_rows]


def _powered_collection_detail(instrument_id: str) -> str | None:
    """Return fresh sample evidence for a confirmed powered instrument."""
    product = POWERED_COLLECTION_PRODUCTS.get(instrument_id)
    if product is None:
        return None
    environment_name, default_path, freshness_minutes = product
    path = env_path(environment_name, default_path)
    try:
        import zarr

        group = zarr.open_group(str(path), mode="r")
        if "time" not in group or not group["time"].shape:
            return None
        sample_time = _decode_cf_time(
            group["time"][-1],
            str(group["time"].attrs.get("units", "")),
        )
        if sample_time is None:
            return None
        age_minutes = (datetime.now(UTC) - sample_time).total_seconds() / 60
        if age_minutes < -5 or age_minutes > freshness_minutes:
            return None
        return f"Source sample {_duration_text(max(age_minutes, 0) / 60)} old"
    except Exception:
        return None


def pdu_instrument_status(instrument_id: str) -> dict[str, Any] | None:
    """Return the current assigned PDU state for a powered instrument, if known."""
    if instrument_id not in PDU_INSTRUMENT_BY_ID:
        return None
    states, detail = _pdu_power_snapshot()
    outlet = PDU_INSTRUMENT_BY_ID[instrument_id][2]
    return _pdu_instrument_status(
        instrument_id,
        states,
        detail,
        _powered_instrument_labels(states),
        collection_detail=(
            _powered_collection_detail(instrument_id)
            if states.get(outlet) is True
            else None
        ),
    )


def pdu_outlet_states() -> dict[int, bool] | None:
    """Return fresh outlet states for dashboard health policy, if available."""
    states, _detail = _pdu_power_snapshot()
    return states or None


def _pdu_power_snapshot() -> tuple[dict[int, bool], str]:
    """Read one fresh PDU sample without inferring a state from stale data."""
    path = Path(os.environ.get("PDU_ZARR_PATH", "/data/aurora/products/power/pdu.zarr"))
    try:
        import zarr

        group = zarr.open_group(str(path), mode="r")
        if "time" not in group or not group["time"].shape:
            raise ValueError("no PDU samples")
        sample_time = _decode_cf_time(group["time"][-1], str(group["time"].attrs.get("units", "")))
        if sample_time is None:
            raise ValueError("unsupported PDU time encoding")
        age_minutes = max((datetime.now(UTC) - sample_time).total_seconds() / 60, 0)
        if age_minutes > PDU_STATE_FRESHNESS_MINUTES:
            raise ValueError("stale PDU sample")
        states = {
            outlet: float(group[f"PDUOutlet{outlet}State"][-1]) >= 0.5
            for _id, _title, _icon, outlet in PDU_INSTRUMENTS
            if f"PDUOutlet{outlet}State" in group
        }
        detail = f"PDU sample {_duration_text(age_minutes / 60)} old"
    except (ImportError, KeyError, OSError, TypeError, ValueError):
        states = {}
        detail = "PDU status unavailable"
    return states, detail


def _automatic_power_labels() -> dict[str, str]:
    """Return learned automatic subphase labels from a fresh operating-state product."""
    path = Path(
        os.environ.get(
            "POWER_OPERATING_STATE_ZARR_PATH",
            "/data/aurora/products/power/power_operating_state.zarr",
        )
    )
    try:
        import zarr

        group = zarr.open_group(str(path), mode="r")
        attrs = group.attrs
        observation_time = _parse_utc(
            attrs.get("last_observation_time_utc") or attrs.get("generated_at_utc")
        )
        if observation_time is None:
            raise ValueError("operating-state time unavailable")
        age_minutes = max((datetime.now(UTC) - observation_time).total_seconds() / 60, 0)
        if age_minutes > AUTOMATIC_PHASE_FRESHNESS_MINUTES:
            raise ValueError("stale operating-state phase")
        current_mode = str(attrs.get("current_mode", "")).strip().lower()
        raw_profiles = attrs.get("mode_load_profiles", {})
        profiles = (
            raw_profiles
            if isinstance(raw_profiles, dict)
            else json.loads(str(raw_profiles or "{}"))
        )
        current_profile = profiles.get(current_mode, {}) if isinstance(profiles, dict) else {}
        current_phase = str(current_profile.get("current_phase", ""))
        labels: dict[str, str] = {}
        mode_tokens = set(current_mode.split("_"))
        for instrument in PDU_INSTRUMENT_CONTRACTS:
            phase_label = dict(instrument.automatic_phase_labels).get(current_phase)
            mode_token = str(instrument.pdu_title or instrument.title).strip().lower().replace(" ", "-")
            if phase_label and mode_token in mode_tokens:
                labels[instrument.id] = phase_label
        return labels
    except (ImportError, json.JSONDecodeError, KeyError, OSError, TypeError, ValueError):
        return {}


def _powered_instrument_labels(states: dict[int, bool]) -> dict[str, str]:
    """Return current labels that add context to a confirmed powered state."""
    labels = _automatic_power_labels()
    uas_outlet = PDU_INSTRUMENT_BY_ID["uas"][2]
    if states.get(uas_outlet) is True:
        if uas_label := _fresh_uas_tier_label():
            labels["uas"] = uas_label
    return labels


def _fresh_uas_tier_label() -> str | None:
    """Return the latest effective UAS tier when its mirrored record is fresh."""
    result = load_uas_mqtt_log(uas_mqtt_log_path(), max_lines=200)
    if not result.records:
        return None
    latest = result.records[-1]
    age_minutes = max((datetime.now(UTC) - latest.timestamp).total_seconds() / 60, 0)
    if age_minutes > UAS_TIER_FRESHNESS_MINUTES or latest.effective_tier < 1:
        return None
    return f"On (Tier {latest.effective_tier})"


def _pdu_instrument_status(
    instrument_id: str,
    states: dict[int, bool],
    detail: str,
    automatic_labels: dict[str, str] | None = None,
    *,
    collection_detail: str | None = None,
) -> dict[str, Any]:
    title, icon, outlet = PDU_INSTRUMENT_BY_ID[instrument_id]
    powered = states.get(outlet)
    state = "On" if powered is True else "Off" if powered is False else "Unknown"
    if powered is True:
        if collection_detail is not None:
            state = "Collecting"
            detail = collection_detail
        else:
            state = (automatic_labels or {}).get(instrument_id, state)
    return {
        "id": instrument_id,
        "title": title,
        "systemImage": icon,
        "state": state,
        "level": "green" if powered is True else "unknown" if powered is False else "amber",
        "detail": detail,
    }


def _overview_card(card_id: str, title: str, value: str, level: str, updated_at: str | None, detail: str = "") -> dict[str, Any]:
    return {"id": card_id, "title": title, "value": value, "level": level, "updatedAt": updated_at, "detail": detail}


def _operations_value(level: str) -> str:
    return {"green": "Healthy", "amber": "Attention", "red": "Action"}.get(level, "Waiting")


def _metric_age_detail(snapshot: dict[str, Any], key: str) -> str:
    age = _metric_value(snapshot, (key,))
    return "Age unknown" if age is None else f"{_duration_text(age / 60)} old"


def _power_time_text(value: Any) -> str:
    moment = _parse_utc(str(value)) if value else None
    return moment.strftime("%H:%M UTC") if moment else "No data"


def _power_age_text(value: Any) -> str:
    moment = _parse_utc(str(value)) if value else None
    if moment is None:
        return "Latest measured power timestamp unavailable"
    age_minutes = max((datetime.now(UTC) - moment).total_seconds() / 60, 0)
    return f"Updated {_duration_text(age_minutes / 60)} ago"


def _latest_power_time() -> str | None:
    """Read the latest measured timestamp from the existing display summary."""
    path = power_display_summary_path()
    if not path.exists():
        return None
    dataset = None
    try:
        import numpy as np
        import pandas as pd
        import xarray as xr

        dataset = xr.open_zarr(path, consolidated=True, chunks=None)
        if "time" not in dataset:
            return None
        times = pd.DatetimeIndex(dataset["time"].values)
        now = pd.Timestamp(datetime.now(UTC)).tz_localize(None)
        latest = None
        for name in ("BatterySOC", "BatteryWatts", "DCInverterVolts", "ACOutputWatts"):
            if name not in dataset or dataset[name].dims != ("time",):
                continue
            values = np.asarray(dataset[name].values, dtype=np.float64)
            mask = np.isfinite(values) & (times <= now)
            if mask.any():
                candidate = times[mask].max()
                latest = candidate if latest is None or candidate > latest else latest
        return None if latest is None else pd.Timestamp(latest).isoformat() + "Z"
    except Exception:
        return None
    finally:
        if dataset is not None:
            dataset.close()


def _battery_depletion_text(
    snapshot: dict[str, Any],
    *,
    power_level: str = "green",
    power_time: str | None = None,
) -> tuple[str, str]:
    if normalize_level(power_level) in {"red", "unknown"}:
        return (
            "Unavailable",
            _stale_power_detail(power_time),
        )
    soc = _metric_value(snapshot, ("aps_battery_soc_pct", "BatterySOC"))
    power_w = _metric_value(snapshot, ("aps_battery_power_w", "BatteryWatts"))
    if soc is None or power_w is None:
        return "No data", "Needs battery state of charge and power"

    capacity_kwh = _metric_value(snapshot, ("aps_battery_capacity_kwh",)) or 26.0
    deadband_w = _metric_value(snapshot, ("aps_battery_depletion_deadband_w",)) or 25.0
    remaining_kwh = _metric_value(snapshot, ("aps_battery_remaining_kwh",))
    if remaining_kwh is None:
        remaining_kwh = max(soc, 0.0) / 100.0 * capacity_kwh
    energy_text = f"{remaining_kwh:.1f} kWh remaining from {capacity_kwh:.0f} kWh"

    if power_w < -deadband_w:
        hours = _metric_value(snapshot, ("aps_battery_depletion_hours",))
        if hours is None:
            hours = remaining_kwh / (abs(power_w) / 1000.0)
        return _duration_text(hours), f"{energy_text}; discharging at {abs(power_w):.0f} W"
    if power_w > deadband_w:
        return "Charging", f"{energy_text}; charging at {power_w:.0f} W"
    return "Flat", f"{energy_text}; battery power {power_w:.0f} W"


def _battery_depletion_level(snapshot: dict[str, Any]) -> str:
    soc = _metric_value(snapshot, ("aps_battery_soc_pct", "BatterySOC"))
    power_w = _metric_value(snapshot, ("aps_battery_power_w", "BatteryWatts"))
    if soc is None or power_w is None:
        return "unknown"
    deadband_w = _metric_value(snapshot, ("aps_battery_depletion_deadband_w",)) or 25.0
    if power_w >= -deadband_w:
        return "green"
    hours = _metric_value(snapshot, ("aps_battery_depletion_hours",))
    if hours is None:
        capacity_kwh = _metric_value(snapshot, ("aps_battery_capacity_kwh",)) or 26.0
        hours = (max(soc, 0.0) / 100.0 * capacity_kwh) / (abs(power_w) / 1000.0)
    return "green" if hours >= 24 else "amber" if hours >= 12 else "red"


def _duration_text(hours: float) -> str:
    total_minutes = max(int(round(hours * 60)), 0)
    days, remainder = divmod(total_minutes, 24 * 60)
    hour_count, minutes = divmod(remainder, 60)
    if days:
        return f"{days}d {hour_count}h"
    if hour_count:
        return f"{hour_count}h {minutes}m"
    return f"{minutes}m"


def _metric_value(snapshot: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = snapshot.get(key)
        if isinstance(value, int | float):
            return float(value)
    return None


def _metric_text(snapshot: dict[str, Any], keys: tuple[str, ...], unit: str) -> str:
    value = _metric_value(snapshot, keys)
    if value is not None:
        return f"{value:.1f}{unit}" if unit else f"{value:.1f}"
    for key in keys:
        value = snapshot.get(key)
        if value:
            return str(value)
    return "Unavailable"


def _age_level(value: str | None, green_minutes: float, amber_minutes: float) -> str:
    moment = _parse_utc(value)
    if moment is None:
        return "unknown"
    age_minutes = max((datetime.now(UTC) - moment).total_seconds() / 60, 0)
    return "green" if age_minutes < green_minutes else "amber" if age_minutes < amber_minutes else "red"


def _snapshot_power_time(snapshot: dict[str, Any]) -> str | None:
    return next(
        (
            str(snapshot[key])
            for key in (
                "power_latest_time_utc",
                "aps_battery_power_time_utc",
                "aps_battery_soc_time_utc",
                "aps_battery_voltage_time_utc",
            )
            if snapshot.get(key)
        ),
        None,
    )


def _worst_level(*levels: Any) -> str:
    normalized = [normalize_level(level) for level in levels]
    for candidate in ("red", "amber", "green"):
        if candidate in normalized:
            return candidate
    return "unknown"


def _freshness_guarded_level(value_level: Any, freshness_level: Any) -> str:
    if normalize_level(value_level) == "unknown":
        return "unknown"
    return _worst_level(value_level, freshness_level)


def _stale_power_detail(value: str | None) -> str:
    moment = _parse_utc(value)
    if moment is None:
        return "Latest APS power measurement is unavailable; depletion cannot be estimated"
    age_minutes = max((datetime.now(UTC) - moment).total_seconds() / 60, 0)
    return (
        f"APS power telemetry is {_duration_text(age_minutes / 60)} old; "
        "depletion cannot be estimated from stale data"
    )


def _power_freshness_alert(snapshot: dict[str, Any]) -> dict[str, Any] | None:
    latest_power_time = _snapshot_power_time(snapshot)
    if latest_power_time is None:
        return None
    level = _age_level(
        latest_power_time,
        POWER_FRESH_MINUTES,
        POWER_STALE_MINUTES,
    )
    if level not in {"amber", "red"}:
        return None
    moment = _parse_utc(latest_power_time)
    if moment is None:
        return None
    age_minutes = max((datetime.now(UTC) - moment).total_seconds() / 60, 0)
    return {
        "id": "power:freshness",
        "title": (
            "APS power telemetry is stale"
            if level == "red"
            else "APS power telemetry is delayed"
        ),
        "level": level,
        "detail": (
            f"No APS power sample since {moment.strftime('%H:%M UTC')} "
            f"({_duration_text(age_minutes / 60)} ago). Battery SOC, voltage "
            "and depletion are last-known values."
        ),
    }


def _age_text(value: str | None) -> str:
    moment = _parse_utc(value)
    if moment is None:
        return "No image"
    age_minutes = max((datetime.now(UTC) - moment).total_seconds() / 60, 0)
    return f"{age_minutes:.0f} min old"


def _parse_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return None


def auroracam(day: str = "latest", time_utc: str | None = None) -> dict[str, Any]:
    root = auroracam_root()
    days = sorted(auroracam_available_days(root), reverse=True)
    if day != "latest" and (not AURORACAM_DAY_RE.fullmatch(day) or day not in days):
        raise KeyError(f"Unknown AURORACam day: {day}")

    available_times: list[str] = []
    if day == "latest":
        selected_day = days[0] if days else None
        records = auroracam_latest_records(root)
    else:
        selected_day = day
        records = {}
        for camera_id in AURORACAM_CAMERAS:
            candidates = auroracam_day_records(root, camera_id, day)
            available_times.extend(record.time_utc for record in candidates)
            if time_utc:
                candidates = [record for record in candidates if record.time_utc == time_utc]
            if candidates:
                records[camera_id] = candidates[-1]

    frames = []

    for camera_id in AURORACAM_CAMERAS:
        record = records.get(camera_id)
        if record is None:
            continue
        frames.append(
            {
                "id": record.camera_id,
                "cameraID": record.camera_id,
                "title": record.label,
                "timeUTC": record.time_utc.replace(" ", "T") + "Z",
                "dayUTC": record.day_utc,
                "previewURL": media_url("auroracam", "preview", record.camera_id, record.day_utc, record.filename),
                "originalURL": media_url("auroracam", "original", record.camera_id, record.day_utc, record.filename),
                "sizeBytes": record.size_bytes,
                "modifiedAt": datetime.fromtimestamp(record.mtime_ns / 1_000_000_000, UTC).isoformat().replace("+00:00", "Z"),
            }
        )
    return {
        "serverTime": utc_now_iso(),
        "selectedDay": selected_day,
        "selectedTimeUTC": time_utc,
        "availableDays": days,
        "availableTimesUTC": sorted(set(available_times), reverse=True)[:288],
        "frames": frames,
    }


def resolve_auroracam_image_path(camera_id: str, day: str, filename: str) -> Path | None:
    if camera_id not in AURORACAM_CAMERAS or AURORACAM_DAY_RE.fullmatch(day) is None or Path(filename).name != filename:
        return None
    path = auroracam_root() / camera_id / day / filename
    return path if path.is_file() else None


def create_auroracam_preview(source: Path, max_dimension: int = 960, quality: int = 80) -> Path:
    """Create a bounded, on-demand preview; original camera JPEGs stay untouched."""
    from PIL import Image

    cache = auroracam_preview_cache_root()
    relative = source.resolve().relative_to(auroracam_root().resolve())
    target = cache / relative
    if target.is_file() and target.stat().st_mtime_ns >= source.stat().st_mtime_ns:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image = image.convert("RGB")
        image.thumbnail((max_dimension, max_dimension))
        image.save(target, format="JPEG", quality=quality, optimize=True)
    _prune_preview_cache(cache)
    return target


def _prune_preview_cache(cache: Path, max_bytes: int = 50 * 1024 * 1024) -> None:
    try:
        files = [path for path in cache.rglob("*.jpg") if path.is_file()]
        total = sum(path.stat().st_size for path in files)
        for path in sorted(files, key=lambda item: item.stat().st_mtime_ns):
            if total <= max_bytes:
                break
            size = path.stat().st_size
            path.unlink(missing_ok=True)
            total -= size
    except OSError:
        return


def power(window: str = "24h", group: str = "all") -> dict[str, Any]:
    """Return compact chart points without reading unrelated Power sections."""
    supported_groups = {
        "all",
        "current",
        "forecast",
        "observed",
        "forecast_24h",
        "forecast_96h",
        "verification",
    }
    if window not in {"24h", "96h"} or group not in supported_groups:
        raise KeyError("Unsupported Power window or group")
    if group in {"current", "observed"}:
        section = "current"
    elif group in {"forecast", "forecast_24h", "forecast_96h", "verification"}:
        section = "forecast"
    else:
        section = "all"
    path = power_display_section_path(section)
    if section != "all" and not path.exists():
        # A mixed-version deployment can temporarily lack the new product.
        # Retain the established combined store as a read-only fallback.
        path = power_display_summary_path()
    payload: dict[str, Any] = {
        "serverTime": utc_now_iso(),
        "window": window,
        "group": group,
        "source": {**file_record(path), "path": str(path)},
        "minimumOperationalSOCPct": 40,
        "panels": [],
    }
    if not path.exists():
        payload["warning"] = "Power display-summary product is unavailable"
        return payload
    try:
        import numpy as np
        import pandas as pd
        import xarray as xr
        from grouped_timeseries import (
            POWER_PANEL_TIME_GROUPS,
            POWER_PANEL_TIME_GROUP_BY_KEY,
            SUMMARY_LAYOUTS,
            build_power_forecast_info,
            build_power_verification_guidance,
            cl61_schedule_presentation,
            merge_operating_scenarios_into_display_summary,
            power_panel_label,
            power_trace_label,
        )

        now = pd.Timestamp(datetime.now(UTC)).tz_localize(None)
        start = now - pd.Timedelta(hours=24)
        horizon = 24 if window == "24h" else 96
        end = now + pd.Timedelta(hours=horizon)
        dataset = xr.open_zarr(path, chunks={"time": 1440}, consolidated=True)
        # The display summary can lag behind the fast planner.  Always replace
        # baked operating traces with the standalone contract, which rejects a
        # plan whose SOC anchor differs from the current ensemble forecast.
        scenarios = None
        if section in {"all", "forecast"}:
            for scenario_path in power_operating_scenario_paths():
                if not scenario_path.exists():
                    continue
                try:
                    candidate = xr.open_zarr(scenario_path, chunks={}, consolidated=True)
                except Exception:
                    continue
                scenarios = candidate
                break
            dataset = merge_operating_scenarios_into_display_summary(dataset, scenarios)
        # Xarray performs a coordinate slice rather than materialising every
        # variable in the historical display store. Only this bounded window is
        # converted to API chart points below.
        dataset = dataset.sel(time=slice(start, end))
        times = pd.DatetimeIndex(dataset["time"].values)
        if group == "all":
            selected_groups = tuple(POWER_PANEL_TIME_GROUPS)
        elif group == "current":
            selected_groups = ("observed",)
        elif group == "forecast":
            selected_groups = ("forecast_24h", "forecast_96h", "verification")
        else:
            selected_groups = (group,)
        panel_keys = {
            panel_key
            for selected_group in selected_groups
            for panel_key in POWER_PANEL_TIME_GROUPS[selected_group]
        }
        for panel in SUMMARY_LAYOUTS["power"]:
            if panel.key not in panel_keys:
                continue
            schedule_presentation = (
                cl61_schedule_presentation(dataset)
                if panel.key == "operating_plan_schedule"
                else None
            )
            forecast_panel = POWER_PANEL_TIME_GROUP_BY_KEY.get(panel.key) in {"forecast_24h", "forecast_96h"}
            panel_start = _forecast_panel_start(dataset, times, panel) if forecast_panel else start
            panel_end = (
                _power_forecast_panel_end(dataset, panel.key, default=end)
                if forecast_panel
                else now
            )
            traces = []
            for trace in panel.traces:
                if trace.var not in dataset or dataset[trace.var].dims != ("time",):
                    continue
                values = np.asarray(dataset[trace.var].values, dtype=np.float64)
                mask = np.isfinite(values) & (times >= panel_start) & (times <= panel_end)
                if trace.valid_min is not None:
                    mask &= values >= float(trace.valid_min)
                if trace.valid_max is not None:
                    mask &= values <= float(trace.valid_max)
                selected_times = times[mask]
                selected_values = values[mask] * float(trace.scale)
                if len(selected_times) > MOBILE_POWER_MAX_POINTS:
                    selected = _representative_power_indices(selected_values)
                    selected_times = selected_times[selected]
                    selected_values = selected_values[selected]
                if not len(selected_times):
                    continue
                segment_ids = [0]
                if len(selected_times) > 1:
                    gaps = np.diff(selected_times.asi8) / 1_000_000_000
                    typical_gap = float(np.median(gaps[gaps > 0])) if (gaps > 0).any() else 60.0
                    gap_threshold = max(typical_gap * 4, 300.0)
                    for gap in gaps:
                        segment_ids.append(segment_ids[-1] + (1 if gap > gap_threshold else 0))
                traces.append(
                    {
                        "id": trace.var,
                        "label": power_trace_label(dataset, trace),
                        "color": trace.color,
                        "axis": trace.axis,
                        "dash": trace.dash,
                        "step": trace.step,
                        "stepAlignment": (
                            "intervalEnd"
                            if trace.step and panel.key == "operating_plan_schedule"
                            else "intervalStart"
                        ),
                        "unit": _power_trace_display_unit(dataset, trace),
                        "points": [
                            {"time": pd.Timestamp(moment).isoformat() + "Z", "value": round(float(value), 5), "segment": segment}
                            for moment, value, segment in zip(selected_times, selected_values, segment_ids, strict=True)
                        ],
                    }
                )
            if traces:
                forecast_context = _power_forecast_context(dataset, panel.key, traces)
                payload["panels"].append(
                    {
                        "id": panel.key,
                        "title": power_panel_label(dataset, panel),
                        "explanation": (
                            schedule_presentation.explanation
                            if schedule_presentation is not None
                            else panel.description
                        ),
                        "info": build_power_forecast_info(panel.key, dataset),
                        "guidance": build_power_verification_guidance(panel.key, dataset),
                        "leftAxisLabel": panel.left_axis_label,
                        "rightAxisLabel": panel.right_axis_label,
                        "traces": traces,
                        **({"forecastContext": forecast_context} if forecast_context else {}),
                    }
                )
        dataset.close()
        if scenarios is not None:
            scenarios.close()
    except Exception as exc:
        payload["warning"] = f"Power display data unavailable: {exc}"
    return payload


def _forecast_panel_start(dataset, times, panel):
    """Return the first valid operational forecast time for a forecast-only panel."""
    import numpy as np
    import pandas as pd

    preferred_fields = {
        "soc_projection": ("BatterySOCForecast",),
        "soc_24h_forecast": ("SystemAsIsDecisionSOCP50",),
        "soc_ecmwf_forecast": (
            "SystemAsIsDecisionSOCP50",
            "BatterySOCForecastP50",
            "BatterySOCForecast",
        ),
        "ecmwf_solar_forecast": ("ForecastSolarWatts", "ECMWFSolarIrradiance"),
        "operating_plan_scenarios": ("OperatingCurrentSOCP50", "OperatingCL61OptimizedSOCP50"),
        "operating_plan_schedule": (
            "OperatingCL61OptimizedActiveCount",
            "OperatingCL61OptimizedCL61On",
            "OperatingCL61OptimizedRadarOn",
            "OperatingCL61OptimizedHATPROOn",
        ),
    }
    fields = preferred_fields.get(panel.key, tuple(trace.var for trace in panel.traces))
    for field in fields:
        if field not in dataset or dataset[field].dims != ("time",):
            continue
        values = np.asarray(dataset[field].values, dtype=np.float64)
        valid = np.isfinite(values)
        if valid.any():
            return pd.Timestamp(times[valid][0])
    return pd.Timestamp(datetime.now(UTC)).tz_localize(None)


def _power_forecast_basis(dataset, panel_key: str) -> tuple[str, str, str, int]:
    """Return label, SOC anchor, issue time, and horizon for one card."""
    operating_panel = panel_key.startswith("operating_plan")
    system_uses_operating = (
        panel_key in {"soc_24h_forecast", "soc_ecmwf_forecast"}
        and str(dataset.attrs.get("system_as_is_decision_source", "")) == "operating_scenario"
    )
    if operating_panel or system_uses_operating:
        horizon_value = dataset.attrs.get(
            "operating_decision_horizon_hours",
            dataset.attrs.get("operating_optimization_horizon_hours", 96),
        )
        try:
            horizon = max(int(float(horizon_value)), 1)
        except (TypeError, ValueError):
            horizon = 96
        if panel_key == "soc_24h_forecast":
            horizon = min(horizon, 24)
        kind = "Operating-plan forecast" if operating_panel else "System-as-is decision forecast"
        return (
            kind,
            str(dataset.attrs.get("operating_initial_soc_time", "")),
            str(dataset.attrs.get("operating_generated_at_utc", "")),
            horizon,
        )
    system_horizons = {
        "soc_24h_forecast": 24,
        "soc_ecmwf_forecast": 96,
        "ecmwf_solar_forecast": 96,
    }
    return (
        "System forecast",
        str(dataset.attrs.get("forecast_initial_soc_time", "")),
        str(dataset.attrs.get("forecast_generated_at_utc", "")),
        system_horizons.get(panel_key, 24),
    )


def _power_trace_display_unit(dataset, trace) -> str:
    """Return the unit after applying the trace's display scaling."""
    unit = str(dataset[trace.var].attrs.get("units", ""))
    if float(trace.scale) == 100.0 and unit.strip().lower() in {"", "1", "dimensionless"}:
        return "%"
    return unit


def _power_forecast_panel_end(dataset, panel_key: str, *, default):
    """Clamp decision panels to their declared horizon from the SOC anchor."""
    import pandas as pd

    if panel_key not in {
        "soc_24h_forecast",
        "soc_ecmwf_forecast",
        "ecmwf_solar_forecast",
        "operating_plan_scenarios",
        "operating_plan_schedule",
    }:
        return default
    _kind, anchor, _issued, horizon = _power_forecast_basis(dataset, panel_key)
    parsed = pd.to_datetime(anchor, errors="coerce", utc=True)
    if pd.isna(parsed):
        return default
    return pd.Timestamp(parsed).tz_convert("UTC").tz_localize(None) + pd.Timedelta(hours=horizon)


def _power_forecast_context(dataset, panel_key: str, traces: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return one anchor and one valid time for all values shown on a forecast card."""
    import pandas as pd

    forecast_panels = {
        "soc_24h_forecast",
        "soc_ecmwf_forecast",
        "ecmwf_solar_forecast",
        "operating_plan_scenarios",
        "operating_plan_schedule",
    }
    if panel_key not in forecast_panels:
        return None
    end_times: list[pd.Timestamp] = []
    for trace in traces:
        points = trace.get("points", [])
        if not points:
            continue
        value = pd.to_datetime(points[-1].get("time"), errors="coerce", utc=True)
        if pd.notna(value):
            end_times.append(pd.Timestamp(value).tz_convert("UTC").tz_localize(None))
    if not end_times:
        return None
    kind, anchor, issued, horizon = _power_forecast_basis(dataset, panel_key)
    return {
        "kind": kind,
        "anchorTime": str(anchor),
        "issuedTime": str(issued),
        # The minimum end time is the last valid point shared by every trace.
        "validTime": min(end_times).isoformat() + "Z",
        "horizonHours": horizon,
    }


def uas(window: str = "24h") -> dict[str, Any]:
    windows = {
        "24h": timedelta(hours=24),
        "7d": timedelta(days=7),
        "all": None,
    }
    if window not in windows:
        raise KeyError(f"Unknown UAS window: {window}")

    result = load_uas_mqtt_log(uas_mqtt_log_path())
    latest = result.records[-1] if result.records else None
    age_seconds = (datetime.now(UTC) - latest.timestamp).total_seconds() if latest else None
    level = "red" if result.missing or result.error else "amber" if age_seconds is None or age_seconds > 300 else "green"
    duration = windows[window]
    cutoff = datetime.now(UTC) - duration if duration is not None else None
    records = [record for record in result.records if cutoff is None or record.timestamp >= cutoff]
    # A corrupted or unexpectedly high-rate log must not make the mobile API
    # response unbounded. The newest records preserve the current state.
    records = records[-2_000:]
    flight_data = summarize_menapia_flight(read_json_file(archive_health_path()))
    return {
        "serverTime": utc_now_iso(),
        "window": window,
        "level": level,
        "latest": None if latest is None else {
            "timeUTC": latest.timestamp.isoformat().replace("+00:00", "Z"),
            "reportedTier": latest.reported_tier,
            "effectiveTier": latest.effective_tier,
            "eventType": latest.event_type,
        },
        "source": {**file_record(result.path), "path": str(result.path)},
        "malformedLineCount": len(result.malformed_lines),
        "flightData": flight_data,
        "records": [
            {
                "timeUTC": record.timestamp.isoformat().replace("+00:00", "Z"),
                "reportedTier": record.reported_tier,
                "effectiveTier": record.effective_tier,
                "eventType": record.event_type,
            }
            for record in records
        ],
    }


def _menapia_safe_flight_id(flight_id: str) -> str:
    value = str(flight_id or "")
    if not MENAPIA_FLIGHT_ID_RE.fullmatch(value):
        raise KeyError("Unknown UAS flight")
    return value


def _read_menapia_json(path: Path, *, maximum_bytes: int) -> dict[str, Any]:
    try:
        stat_result = path.stat()
    except FileNotFoundError as exc:
        raise KeyError("UAS flight product is not available") from exc
    except OSError as exc:
        raise KeyError(f"Could not read UAS flight product: {exc}") from exc
    if stat_result.st_size > maximum_bytes:
        raise KeyError("UAS flight product exceeds the safe response limit")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise KeyError(f"Invalid UAS flight product: {exc}") from exc
    if not isinstance(value, dict):
        raise KeyError("Invalid UAS flight product")
    return value


def _menapia_quality(value: Any) -> dict[str, Any]:
    quality = value if isinstance(value, dict) else {}
    level = str(quality.get("level") or "unknown").lower()
    if level not in {"green", "amber", "red"}:
        level = "unknown"
    warnings = quality.get("warnings")
    if not isinstance(warnings, list):
        warnings = []
    return {
        "level": level,
        "warnings": [str(item)[:500] for item in warnings if isinstance(item, str)][:50],
    }


def _menapia_flight_record(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    try:
        flight_id = _menapia_safe_flight_id(str(value.get("id") or ""))
    except KeyError:
        return None
    def text_field(key: str, maximum: int = 500) -> str | None:
        item = value.get(key)
        return str(item)[:maximum] if isinstance(item, str) else None

    def number_field(key: str) -> int | float | None:
        item = value.get(key)
        return item if isinstance(item, int | float) and not isinstance(item, bool) and math.isfinite(item) else None

    record = {
        "id": flight_id,
        "sourceFlightID": text_field("sourceFlightID"),
        "dayUTC": text_field("dayUTC", 10),
        "flightNumber": int(value["flightNumber"]) if isinstance(value.get("flightNumber"), int) else None,
        "title": text_field("title"),
        "startTimeUTC": text_field("startTimeUTC", 64),
        "endTimeUTC": text_field("endTimeUTC", 64),
        "durationSeconds": number_field("durationSeconds"),
        "samplePeriodSeconds": number_field("samplePeriodSeconds"),
        "modifiedAt": text_field("modifiedAt", 64),
    }
    record["quality"] = _menapia_quality(value.get("quality"))
    plot_path = menapia_product_root() / "plots" / f"{flight_id}.png"
    record["plotURL"] = versioned_media_url(plot_path, "uas", "flights", flight_id)
    return record


def _menapia_catalog() -> tuple[dict[str, Any], str | None]:
    """Load the bounded catalog and return ``(payload, error)``.

    Listing clients receive an explicit state for missing or malformed catalog
    products. Individual detail/media lookups still fail closed with 404.
    """
    path = menapia_catalog_path()
    try:
        payload = _read_menapia_json(path, maximum_bytes=5_000_000)
    except KeyError as exc:
        return {}, str(exc.args[0] if exc.args else exc)
    return payload, None


def _menapia_run_heartbeat() -> dict[str, Any]:
    """Return a validated optional run heartbeat.

    Older deployments have no heartbeat, so missing or malformed operational
    state deliberately falls back to the immutable catalog contract.
    """
    try:
        payload = _read_menapia_json(
            menapia_product_status_path(),
            maximum_bytes=100_000,
        )
    except KeyError:
        return {}
    state = str(payload.get("state") or "")
    completed_at = payload.get("completedAt")
    if state not in {"success", "partial_failure", "failed"}:
        return {}
    if _menapia_generated_age_seconds(completed_at) is None:
        return {}
    return {"state": state, "completedAt": str(completed_at)}


def _menapia_generated_age_seconds(generated_at: Any) -> float | None:
    try:
        parsed = datetime.fromisoformat(str(generated_at).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return max(0.0, (datetime.now(UTC) - parsed.astimezone(UTC)).total_seconds())


def _menapia_listing_state(
    *,
    error: str | None,
    last_run_state: str | None,
    generated_at: Any,
    flight_count: int,
) -> dict[str, Any]:
    age_seconds = _menapia_generated_age_seconds(generated_at)
    stale_after = max(60.0, float(os.environ.get("MENAPIA_PRODUCT_STALE_AFTER_MINUTES", "90")) * 60.0)
    if error:
        return {"state": "error", "level": "red", "title": "Flight products unavailable", "detail": error, "ageSeconds": None}
    if last_run_state == "failed":
        return {
            "state": "error",
            "level": "red",
            "title": "Flight product update failed",
            "detail": "The most recent product build failed; the last published flights remain selectable.",
            "ageSeconds": age_seconds,
        }
    if last_run_state == "partial_failure":
        return {
            "state": "partial",
            "level": "amber",
            "title": "Flight products partially updated",
            "detail": "Some source bundles could not be processed; available flights remain selectable.",
            "ageSeconds": age_seconds,
        }
    if not flight_count:
        return {
            "state": "empty",
            "level": "amber",
            "title": "No flights for this day",
            "detail": "The catalog is available but contains no flights for the selected UTC day.",
            "ageSeconds": age_seconds,
        }
    if age_seconds is None:
        return {
            "state": "unknown",
            "level": "amber",
            "title": "Flight product freshness unknown",
            "detail": "The catalog has no valid generation timestamp.",
            "ageSeconds": None,
        }
    if age_seconds > stale_after:
        return {
            "state": "stale",
            "level": "amber",
            "title": "Flight products are stale",
            "detail": f"The most recent product build completed {int(age_seconds // 60)} minutes ago.",
            "ageSeconds": age_seconds,
        }
    return {
        "state": "fresh",
        "level": "green",
        "title": "Flight products are current",
        "detail": "The most recent derived-product build completed within the expected refresh window.",
        "ageSeconds": age_seconds,
    }


def uas_flights(day: str = "latest") -> dict[str, Any]:
    """Return a bounded, path-free flight catalog for one UTC day."""
    if day != "latest" and not WXCAM_DAY_RE.fullmatch(day):
        raise KeyError(f"Unknown UAS flight day: {day}")
    payload, error = _menapia_catalog()
    raw_days = payload.get("availableDays") if isinstance(payload.get("availableDays"), list) else []
    available_days = [
        value for value in dict.fromkeys(str(item) for item in raw_days)
        if WXCAM_DAY_RE.fullmatch(value)
    ]
    selected_day = available_days[0] if day == "latest" and available_days else None if day == "latest" else day
    records = []
    raw_flights = payload.get("flights") if isinstance(payload.get("flights"), list) else []
    for raw_flight in raw_flights:
        record = _menapia_flight_record(raw_flight)
        if record and record.get("dayUTC") == selected_day:
            records.append(record)
    records.sort(key=lambda item: str(item.get("startTimeUTC") or ""), reverse=True)
    heartbeat = _menapia_run_heartbeat()
    last_run_state = str(
        heartbeat.get("state") or payload.get("lastRunState") or ""
    ) or None
    generated_at = payload.get("generatedAt")
    last_run_at = heartbeat.get("completedAt") or generated_at
    latest_flight_id = payload.get("latestFlightID")
    if not isinstance(latest_flight_id, str) or not MENAPIA_FLIGHT_ID_RE.fullmatch(latest_flight_id):
        latest_flight_id = None
    state = _menapia_listing_state(
        error=error,
        last_run_state=last_run_state,
        generated_at=last_run_at,
        flight_count=len(records),
    )
    daily_quicklook = None
    all_flights_plot_url = None
    if selected_day:
        token = selected_day.replace("-", "")
        path = resolve_quicklook_path("science", "uas", token)
        if path:
            all_flights_plot_url = versioned_media_url(
                path,
                "quicklook",
                "science",
                "uas",
                token,
            )
            daily_quicklook = {
                "token": token,
                "imageURL": all_flights_plot_url,
                **file_record(path),
            }
    return {
        "schemaVersion": int(payload.get("schemaVersion") or 1),
        "serverTime": utc_now_iso(),
        "generatedAt": generated_at,
        "lastRunAt": last_run_at,
        "lastRunState": last_run_state,
        "requestedDay": day,
        "selectedDay": selected_day,
        "latestFlightID": latest_flight_id,
        "availableDays": available_days,
        "status": state,
        "allFlightsPlotURL": all_flights_plot_url,
        "dailyQuicklook": daily_quicklook,
        "flights": records,
    }


def _menapia_catalog_flight(flight_id: str) -> dict[str, Any]:
    safe_id = _menapia_safe_flight_id(flight_id)
    payload, error = _menapia_catalog()
    if error:
        raise KeyError(error)
    raw_flights = payload.get("flights") if isinstance(payload.get("flights"), list) else []
    for raw_flight in raw_flights:
        record = _menapia_flight_record(raw_flight)
        if record and record["id"] == safe_id:
            return record
    raise KeyError("Unknown UAS flight")


def _menapia_series(payload: dict[str, Any]) -> dict[str, Any]:
    series = payload.get("series")
    if not isinstance(series, dict):
        raise KeyError("Invalid UAS flight series")
    time_values = series.get("timeUTC")
    if not isinstance(time_values, list):
        raise KeyError("Invalid UAS flight time series")
    maximum_points = max(1, min(int(os.environ.get("MENAPIA_MAX_PROFILE_POINTS", "20000")), 100_000))
    if len(time_values) > maximum_points:
        raise KeyError("UAS flight series exceeds the safe response limit")
    if any(not isinstance(value, str) or len(value) > 64 for value in time_values):
        raise KeyError("Invalid UAS flight timestamps")
    normalized: dict[str, Any] = {"timeUTC": list(time_values)}

    def numeric_values(values: Any, label: str) -> list[float | None]:
        if not isinstance(values, list) or len(values) != len(time_values):
            raise KeyError(f"Invalid UAS flight series length: {label}")
        result: list[float | None] = []
        for value in values:
            if value is None:
                result.append(None)
            elif isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value):
                result.append(float(value))
            else:
                raise KeyError(f"Invalid UAS flight series value: {label}")
        return result

    for key in ("temperatureC", "pressureHpa", "relativeHumidityPct"):
        group = series.get(key)
        if not isinstance(group, dict):
            raise KeyError(f"Invalid UAS flight series: {key}")
        normalized_group = {}
        for sensor in ("SN0122", "SN0123"):
            normalized_group[sensor] = numeric_values(group.get(sensor), f"{key}.{sensor}")
        normalized[key] = normalized_group
    normalized["altitudeM"] = numeric_values(series.get("altitudeM"), "altitudeM")
    return normalized


def uas_flight(flight_id: str) -> dict[str, Any]:
    """Return one validated profile without exposing product filesystem paths."""
    record = _menapia_catalog_flight(flight_id)
    safe_id = record["id"]
    path = menapia_product_root() / "flights" / f"{safe_id}.json"
    payload = _read_menapia_json(path, maximum_bytes=25_000_000)
    embedded = payload.get("flight")
    if isinstance(embedded, dict) and embedded.get("id") not in {None, safe_id}:
        raise KeyError("UAS flight detail does not match the catalog")
    return {
        "schemaVersion": int(payload.get("schemaVersion") or 1),
        "serverTime": utc_now_iso(),
        "flight": record,
        "plotURL": record["plotURL"],
        "series": _menapia_series(payload),
    }


def resolve_uas_flight_plot_path(flight_id: str) -> Path | None:
    """Resolve a catalog-listed per-flight plot using a fixed directory."""
    record = _menapia_catalog_flight(flight_id)
    path = menapia_product_root() / "plots" / f"{record['id']}.png"
    return path if path.is_file() else None


def instrument_summary(instrument_id: str, window: str = "24h") -> dict[str, Any]:
    instrument = _instrument_or_raise(instrument_id)
    latest = resolve_quicklook_path("science", instrument_id, "latest")
    entries = quicklooks("science", instrument_id).get("entries", [])[:8]
    panels = []
    if latest:
        panels.append(
            {
                "id": "latest-quicklook",
                "title": "Latest quicklook",
                "kind": "image",
                "imageURL": media_url("quicklook", "science", instrument_id, "latest"),
                "level": "green",
                "detail": "Latest generated quicklook",
            }
        )
    else:
        panels.append(
            {
                "id": "latest-quicklook",
                "title": "Latest quicklook",
                "kind": "empty",
                "level": "unknown",
                "detail": "No generated quicklook was found",
            }
        )
    return {
        "serverTime": utc_now_iso(),
        "instrument": {
            "id": instrument.id,
            "title": instrument.title,
            "systemImage": instrument.system_image,
            "supportsSummary": instrument.summary_supported,
        },
        "window": window,
        "updatedAt": file_record(latest).get("modifiedAt") if latest else None,
        "panels": panels,
        "recentQuicklooks": entries,
    }


def quicklooks(kind: str, instrument_id: str) -> dict[str, Any]:
    instrument = _instrument_or_raise(instrument_id)
    if kind not in {"science", "housekeeping"}:
        raise KeyError(f"Unknown quicklook kind: {kind}")
    prefixes = instrument.science_prefixes if kind == "science" else instrument.housekeeping_prefixes
    entries = _quicklook_entries(instrument, kind, prefixes)
    latest = next((entry for entry in entries if entry["token"] == "latest"), entries[0] if entries else None)
    return {
        "serverTime": utc_now_iso(),
        "kind": kind,
        "instrument": {"id": instrument.id, "title": instrument.title, "systemImage": instrument.system_image},
        "latest": latest,
        "entries": entries,
        "powerStatus": pdu_instrument_status(instrument_id),
    }


def _instrument_quicklook_root(instrument: Instrument) -> Path:
    return uas_quicklook_root() if instrument.id == "uas" else quicklook_root() / instrument.quicklook_subdir


def _quicklook_paths(instrument: Instrument, kind: str, prefixes: tuple[str, ...]) -> dict[str, Path]:
    directory = _instrument_quicklook_root(instrument)
    if not prefixes or not directory.exists():
        return {}
    paths: dict[str, Path] = {}
    for path in sorted(directory.glob("*")):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"} or not path.is_file():
            continue
        name = path.name
        is_latest = "latest" in name.lower()
        token_match = DATE_TOKEN_RE.search(name)
        matches_prefix = any(name.startswith(prefix) for prefix in prefixes)
        # ``latest.png`` is the legacy science alias. It has no instrument
        # prefix but is still a valid science quicklook inside this directory.
        is_science_latest_alias = kind == "science" and name.lower() == "latest.png"
        # Science product names may share a base prefix with their housekeeping
        # counterpart (for example ``cloud_radar`` and ``cloud_radar__hk_radar``).
        # Always reject the latter before accepting a science image.
        is_housekeeping_image = any(name.startswith(prefix) for prefix in instrument.housekeeping_prefixes)
        if not (matches_prefix or is_science_latest_alias) or (kind == "science" and is_housekeeping_image):
            continue
        if is_latest:
            paths.setdefault("latest", path)
        elif token_match:
            paths.setdefault(token_match.group(1), path)

    # The legacy ``latest.png`` aliases are maintained by separate jobs. If a
    # newer dated science quicklook exists, serve that data rather than an old
    # alias that still calls itself "Last 24 hours".
    if kind == "science":
        dated_paths = [path for token, path in paths.items() if token != "latest"]
        newest_dated = max(dated_paths, key=lambda path: path.name) if dated_paths else None
        latest = paths.get("latest")
        if newest_dated is not None and (latest is None or newest_dated.stat().st_mtime_ns > latest.stat().st_mtime_ns):
            paths["latest"] = newest_dated
    return paths


def _quicklook_entries(instrument: Instrument, kind: str, prefixes: tuple[str, ...]) -> list[dict[str, Any]]:
    paths = _quicklook_paths(instrument, kind, prefixes)

    def sort_key(item: tuple[str, Path]) -> tuple[int, str]:
        token, _path = item
        return (1 if token == "latest" else 0, token)

    entries = []
    for token, path in sorted(paths.items(), key=sort_key, reverse=True):
        dated_latest = token == "latest" and DATE_TOKEN_RE.search(path.name)
        entries.append(
            {
                "id": f"{instrument.id}-{kind}-{token}",
                "token": token,
                "title": (
                    f"Latest available ({_format_date_token(DATE_TOKEN_RE.search(path.name).group(1))})"
                    if dated_latest
                    else "Latest available" if token == "latest" and instrument.id == "uas"
                    else "Latest" if token == "latest"
                    else _format_date_token(token)
                ),
                "imageURL": (
                    versioned_media_url(path, "quicklook", kind, instrument.id, token)
                    if instrument.id == "uas"
                    else media_url("quicklook", kind, instrument.id, token)
                ),
                **file_record(path),
            }
        )
    return entries


def resolve_quicklook_path(kind: str, instrument_id: str, token: str) -> Path | None:
    instrument = _instrument_or_raise(instrument_id)
    paths = _quicklook_paths(
        instrument,
        kind,
        instrument.science_prefixes if kind == "science" else instrument.housekeeping_prefixes,
    )
    return paths.get(token)


def _find_quicklook_path_by_record(instrument: Instrument, kind: str, token: str) -> Path | None:
    prefixes = instrument.science_prefixes if kind == "science" else instrument.housekeeping_prefixes
    directory = _instrument_quicklook_root(instrument)
    if not directory.exists():
        return None
    for path in sorted(directory.glob("*")):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        if token == "latest" and "latest" not in path.name.lower():
            continue
        if token != "latest" and token not in path.name:
            continue
        is_housekeeping_image = any(path.name.startswith(prefix) for prefix in instrument.housekeeping_prefixes)
        is_science_latest_alias = kind == "science" and path.name.lower() == "latest.png"
        if (any(path.name.startswith(prefix) for prefix in prefixes) or is_science_latest_alias) and not (kind == "science" and is_housekeeping_image):
            return path
    return None


def wxcam(stream: str = "fish_hdr", day: str = "latest") -> dict[str, Any]:
    if stream not in WXCAM_STREAMS:
        raise KeyError(f"Unknown WXcam stream: {stream}")
    resolved_day = _resolve_wxcam_day(stream, day)
    video_path = resolve_wxcam_video_path(stream, resolved_day)
    thumbs = wxcam_thumbnail_records(stream, resolved_day)
    return {
        "serverTime": utc_now_iso(),
        "stream": {"id": stream, **WXCAM_STREAMS[stream]},
        "selectedDay": resolved_day,
        "availableDays": available_wxcam_days(stream),
        "video": {
            "url": media_url("wxcam", "video", stream, resolved_day),
            **(file_record(video_path) if video_path else {"exists": False}),
        },
        "posterURL": thumbs[0]["imageURL"] if thumbs else None,
        "thumbnails": thumbs,
    }


def resolve_wxcam_video_path(stream: str, day: str) -> Path | None:
    token = _wxcam_day_token(day)
    if not token:
        return None
    path = wxcam_daily_video_root() / stream / f"{token}.mp4"
    return path if path.exists() else None


def available_wxcam_days(stream: str) -> list[str]:
    directory = wxcam_daily_video_root() / stream
    if not directory.exists():
        return []
    days = []
    for path in sorted(directory.glob("*.mp4")):
        if path.stem == "latest":
            continue
        match = DATE_TOKEN_RE.fullmatch(path.stem)
        if match:
            days.append(_format_date_token(match.group(1)))
    return sorted(days, reverse=True)


def wxcam_thumbnail_records(stream: str, day: str) -> list[dict[str, Any]]:
    token = _wxcam_day_token(day)
    if not token:
        return []
    directory = wxcam_hourly_thumbnail_root() / stream / token
    if not directory.exists():
        return []
    selected: dict[int, tuple[tuple[int, int, str], Path]] = {}
    for path in sorted(directory.glob("*.jpg")):
        timestamp = parse_wxcam_timestamp(path)
        if timestamp is None or timestamp.strftime("%Y%m%d") != token:
            continue
        seconds_after_hour = timestamp.minute * 60 + timestamp.second
        score = (
            abs(seconds_after_hour - 30 * 60),
            seconds_after_hour,
            path.name,
        )
        current = selected.get(timestamp.hour)
        if current is None or score < current[0]:
            selected[timestamp.hour] = (score, path)

    records = []
    for hour, (_score, path) in sorted(selected.items()):
        records.append(
            {
                "id": path.stem,
                "title": path.stem,
                "hourUTC": hour,
                "imageURL": media_url("wxcam", "thumb", stream, token, path.name),
                **file_record(path),
            }
        )
    return records


def resolve_wxcam_thumbnail_path(stream: str, day_token: str, filename: str) -> Path | None:
    if stream not in WXCAM_STREAMS or DATE_TOKEN_RE.fullmatch(day_token) is None or Path(filename).name != filename:
        return None
    path = wxcam_hourly_thumbnail_root() / stream / day_token / filename
    return path if path.exists() else None


def latest_wxcam_catalog_record(stream: str, media_kind: str) -> dict[str, Any] | None:
    if stream not in WXCAM_STREAMS or media_kind not in {"image", "video"}:
        return None
    path = wxcam_catalog_path()
    if not path.exists():
        return None
    uri = f"file:{path}?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                """
                SELECT image_type, media_kind, time_utc, day_utc, filename, relative_path
                FROM images
                WHERE image_type = ? AND media_kind = ?
                ORDER BY time_epoch_ns DESC, raw_path DESC
                LIMIT 1
                """,
                (stream, media_kind),
            ).fetchone()
    except sqlite3.Error:
        return None
    return dict(row) if row else None


def _resolve_wxcam_day(stream: str, day: str) -> str:
    if day != "latest":
        return day if _wxcam_day_token(day) else "latest"
    record = latest_wxcam_catalog_record(stream, "video")
    if record and record.get("day_utc"):
        return str(record["day_utc"])
    days = available_wxcam_days(stream)
    return days[0] if days else "latest"


def _format_date_token(token: str) -> str:
    return f"{token[:4]}-{token[4:6]}-{token[6:8]}"


def _wxcam_day_token(day: str) -> str | None:
    if day == "latest":
        return "latest"
    if DATE_TOKEN_RE.fullmatch(day):
        return day
    if WXCAM_DAY_RE.fullmatch(day):
        return day.replace("-", "")
    return None


def _instrument_or_raise(instrument_id: str) -> Instrument:
    instrument = INSTRUMENT_BY_ID.get(instrument_id)
    if not instrument:
        raise KeyError(f"Unknown instrument: {instrument_id}")
    return instrument
