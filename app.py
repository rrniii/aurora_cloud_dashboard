"""Aurora dashboard application.

This module hosts the multi-instrument Panel and Plotly browser for atmospheric
curtains, station summaries, WXcam media, quicklooks, and operations monitoring.
It keeps per-instrument state warm and uses cached bounds, stale-render
protection, prewarmed latest summary figures, coarse-first rendering on heavier
2D plots, and instrument-specific trace-density controls to keep the UI
responsive during normal browsing. Quicklook PNGs are displayed through
responsive HTML wrappers so tall images do not reserve their original pixel
height as blank page space after browser scaling.
"""

import asyncio
from base64 import b64encode
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone, time
from functools import lru_cache, partial
import hashlib
from html import escape
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
from threading import RLock
from time import perf_counter
from urllib.parse import quote, urlencode

import numpy as np
import pandas as pd
import panel as pn
import param
from bokeh.core.json_encoder import serialize_json
from panel.io import hold
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import xarray as xr
from power_operating_scenarios import KIT_ORDER, MIN_RUN_HOURS, evaluate_custom_schedule
from power_soc_thresholds import (
    MINIMUM_OPERATIONAL_SOC_PCT,
    MINIMUM_OPERATIONAL_SOC_REFERENCE_LABEL,
    SOC_REFERENCE_PANEL_KEYS,
)
from power_scenario_catalog import SUGGESTED_OPERATING_SCENARIOS
try:
    from PIL import Image
except Exception:  # pragma: no cover - dashboard can still serve source images.
    Image = None
from grouped_timeseries import (
    POWER_PANEL_TIME_GROUP_BY_KEY,
    SUMMARY_LAYOUTS,
    build_power_forecast_info,
    build_summary_plotly,
    build_power_verification_guidance,
    calendar_date_tokens,
    combine_summary_datasets,
    default_calendar_label,
    default_interactive_label,
    display_name,
    housekeeping_label,
    housekeeping_daily_png,
    housekeeping_latest_png,
    is_summary_instrument,
    merge_operating_scenarios_into_display_summary,
    operating_mode_intervals,
    power_panel_label,
    power_trace_label,
    summary_daily_png,
    summary_latest_png,
    summary_source_instruments,
    SUMMARY_DISPLAY_END_ATTR,
    SUMMARY_DISPLAY_START_ATTR,
    POWER_CUMULATIVE_CONTEXT_DAYS,
    POWER_DISPLAY_SUMMARY_FIELDS,
    POWER_SOC_FORECAST_FIELDS,
    POWER_FUTURE_DISPLAY_FIELDS,
    prepare_summary_dataset,
    widget_group_options,
)
from data_services import (
    WindowRequest,
    coarsen_targets as _coarsen_targets,
    prepare_dataset_window,
)
from extra_housekeeping import (
    extra_housekeeping_daily_png,
    extra_housekeeping_label,
    extra_housekeeping_latest_png,
    extra_housekeeping_tokens,
)
from time_gap_breaks import insert_time_gap_breaks
from radar_colormaps import radar_plotly_colorscale
from auroracam_catalog import (
    AURORACAM_CAMERAS,
    available_days as auroracam_available_days,
    day_records as auroracam_day_records,
    latest_record as auroracam_latest_record,
)
from wxcam_catalog import (
    WXCAM_IMAGE_TYPES,
    available_days,
    catalog_time_bounds,
    latest_record,
    representative_hourly_records,
)
from menapia_flight_status import summarize_menapia_flight
from uas_mqtt import UASMqttParseResult, UASMqttRecord, load_uas_mqtt_log
import mobile_catalog
from browser_icons import instrument_icon_svg
from instrument_registry import browser_options, science_options
from presentation_models import empty_data_state
from request_context import (
    client_ip as _client_ip,
    live_session_count as _live_session_count,
    request_base_url as _request_base_url,
    request_header as _request_header,
    request_path as _request_path,
    request_query_args as _request_query_args,
    server_session_count as _server_session_count,
    session_id as _session_id,
    total_session_count as _total_session_count,
)

# Keep common artwork and CSS out of the server-rendered document. They are
# cacheable Panel/nginx assets rather than strings duplicated into every model.
DASHBOARD_ASSET_PREFIX = os.environ.get("AURORA_DASHBOARD_ASSET_PREFIX", "/dashboard-assets").rstrip("/")
DASHBOARD_STYLESHEET_PATH = Path(__file__).resolve().parent / "assets" / "dashboard.css"
try:
    DASHBOARD_STYLESHEET_VERSION = hashlib.sha256(DASHBOARD_STYLESHEET_PATH.read_bytes()).hexdigest()[:12]
except OSError:
    DASHBOARD_STYLESHEET_VERSION = "unavailable"
DASHBOARD_STYLESHEET = f"{DASHBOARD_ASSET_PREFIX}/dashboard.css?v={DASHBOARD_STYLESHEET_VERSION}"
# Panel 1.8 exposes external stylesheets through ``config.css_files``. This
# keeps the asset out of each Bokeh document while remaining compatible with
# the pinned runtime (``pn.extension(stylesheets=...)`` is not supported).
if DASHBOARD_STYLESHEET not in pn.config.css_files:
    pn.config.css_files.append(DASHBOARD_STYLESHEET)
pn.extension(
    "plotly",
    notifications=True,
    sizing_mode="stretch_width",
    defer_load=True,
)


# Keep the logo out of the server-rendered document.  It is deliberately served
# as a normal cacheable asset by Panel/nginx rather than base64-encoding a full
# resolution PNG into every new browser session.
DASHBOARD_LOGO = f"{DASHBOARD_ASSET_PREFIX}/logo.png"
DASHBOARD_FAVICON = "https://gamb2le.pages.dev/assets/logo.png"
# Dashboard palette. The supplied source values use #RRGGBBAA notation.
THEME_TEXT = "#1e2f50"
THEME_MUTED = "#52627d"
THEME_BORDER = "#b9c5d4"
THEME_LINE = "#8191a8"
THEME_GRID = "#dbe2ea"
THEME_PANEL = "#fffaf4"
THEME_ACCENT = "#36b9b2"
SITE_ENV_RAW = os.environ.get("AURORA_SITE_ENV", "").strip().lower()
SITE_ENV_EXPLICIT = bool(SITE_ENV_RAW)
SITE_ENV = SITE_ENV_RAW
SITE_DOMAIN = os.environ.get("AURORA_DOMAIN", "").strip()
if not SITE_ENV and SITE_DOMAIN:
    if "data-ocean" in SITE_DOMAIN:
        SITE_ENV = "development"
    elif SITE_DOMAIN == "data.gamb2le.co.uk":
        SITE_ENV = "production"
SITE_ENV = SITE_ENV or "unknown"
APS_INTERNAL_TEMP_LOW_AMBER_C = float(os.environ.get("APS_INTERNAL_TEMP_LOW_AMBER_C", "10"))
APS_INTERNAL_TEMP_LOW_RED_C = float(os.environ.get("APS_INTERNAL_TEMP_LOW_RED_C", "5"))
APS_INTERNAL_TEMP_HIGH_AMBER_C = float(os.environ.get("APS_INTERNAL_TEMP_HIGH_AMBER_C", "40"))
APS_INTERNAL_TEMP_HIGH_RED_C = float(os.environ.get("APS_INTERNAL_TEMP_HIGH_RED_C", "45"))
APS_DEWPOINT_RED_MARGIN_C = float(os.environ.get("APS_DEWPOINT_RED_MARGIN_C", "0"))


class WxcamVideoPlayer(pn.reactive.ReactiveHTML):
    src = param.String(default="")
    poster = param.String(default="")
    title = param.String(default="")
    subtitle = param.String(default="")
    mode_class = param.String(default="wxcam-player--wide")

    _template = """
    <div id="player_shell" class="wxcam-player ${mode_class}">
      <div id="meta_row" class="wxcam-player__meta">
        <div id="meta_text" class="wxcam-player__meta-text">
          <div id="title_text" class="wxcam-player__title">{{ title }}</div>
          <div id="subtitle_text" class="wxcam-player__subtitle">{{ subtitle }}</div>
        </div>
      </div>
      <div id="control_row" class="wxcam-player__controls">
        <button id="play_btn" type="button" onclick="${script('toggle_play')}">Play</button>
        <button id="back_btn" type="button" onclick="${script('jump_back')}">-0.5s</button>
        <button id="forward_btn" type="button" onclick="${script('jump_forward')}">+0.5s</button>
        <label id="speed_wrap" class="wxcam-player__inline-label">
          <span>Speed</span>
          <select id="speed_select" onchange="${script('change_speed')}">
            <option value="0.5">0.5x</option>
            <option value="1" selected>1x</option>
            <option value="1.5">1.5x</option>
            <option value="2">2x</option>
            <option value="4">4x</option>
          </select>
        </label>
        <label id="loop_wrap" class="wxcam-player__inline-label wxcam-player__checkbox">
          <input id="loop_toggle" type="checkbox" onchange="${script('toggle_loop')}"></input>
          <span>Loop</span>
        </label>
      </div>
      <div id="seek_row" class="wxcam-player__seek">
        <span id="current_time" class="wxcam-player__time">00:00</span>
        <input id="seek_slider" type="range" min="0" max="1000" value="0" step="1" oninput="${script('seek')}"></input>
        <span id="duration" class="wxcam-player__time">00:00</span>
      </div>
      <div id="video_frame" class="wxcam-player__frame">
        <video
          id="video_el"
          src="${src}"
          poster="${poster}"
          controls
          preload="metadata"
          playsinline
          onloadedmetadata="${script('sync_metadata')}"
          ontimeupdate="${script('sync_time')}"
          onplay="${script('sync_play_state')}"
          onpause="${script('sync_play_state')}"
          onended="${script('sync_play_state')}"
          onratechange="${script('sync_speed_state')}"
        ></video>
      </div>
    </div>
    """

    _scripts = {
        "src": """
          video_el.pause();
          video_el.load();
          seek_slider.value = 0;
          current_time.textContent = "00:00";
          duration.textContent = "00:00";
          play_btn.textContent = "Play";
          speed_select.value = "1";
        """,
        "toggle_play": """
          if (video_el.paused) {
            video_el.play();
          } else {
            video_el.pause();
          }
        """,
        "jump_back": """
          const nextTime = Math.max(0, (video_el.currentTime || 0) - 0.5);
          video_el.currentTime = nextTime;
          view.run_script('sync_time');
        """,
        "jump_forward": """
          const durationSeconds = Number.isFinite(video_el.duration) ? video_el.duration : 0;
          const candidateTime = (video_el.currentTime || 0) + 0.5;
          const nextTime = durationSeconds > 0 ? Math.min(durationSeconds, candidateTime) : candidateTime;
          video_el.currentTime = nextTime;
          view.run_script('sync_time');
        """,
        "change_speed": """
          video_el.playbackRate = Number(speed_select.value || 1);
          view.run_script('sync_speed_state');
        """,
        "toggle_loop": """
          video_el.loop = loop_toggle.checked;
        """,
        "seek": """
          const durationSeconds = Number.isFinite(video_el.duration) ? video_el.duration : 0;
          if (!durationSeconds) {
            return;
          }
          video_el.currentTime = (Number(seek_slider.value || 0) / 1000) * durationSeconds;
          view.run_script('sync_time');
        """,
        "sync_metadata": """
          const formatTime = (seconds) => {
            if (!Number.isFinite(seconds) || seconds < 0) {
              return "00:00";
            }
            const total = Math.floor(seconds);
            const hours = Math.floor(total / 3600);
            const minutes = Math.floor((total % 3600) / 60);
            const secs = total % 60;
            if (hours > 0) {
              return `${hours}:${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
            }
            return `${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
          };
          const durationSeconds = Number.isFinite(video_el.duration) ? video_el.duration : 0;
          current_time.textContent = formatTime(video_el.currentTime || 0);
          duration.textContent = formatTime(durationSeconds);
          seek_slider.value = 0;
          video_el.loop = loop_toggle.checked;
          video_el.playbackRate = Number(speed_select.value || 1);
          view.run_script('sync_play_state');
          view.run_script('sync_speed_state');
        """,
        "sync_time": """
          const formatTime = (seconds) => {
            if (!Number.isFinite(seconds) || seconds < 0) {
              return "00:00";
            }
            const total = Math.floor(seconds);
            const hours = Math.floor(total / 3600);
            const minutes = Math.floor((total % 3600) / 60);
            const secs = total % 60;
            if (hours > 0) {
              return `${hours}:${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
            }
            return `${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
          };
          const durationSeconds = Number.isFinite(video_el.duration) ? video_el.duration : 0;
          const currentSeconds = Number(video_el.currentTime || 0);
          current_time.textContent = formatTime(currentSeconds);
          duration.textContent = formatTime(durationSeconds);
          seek_slider.value = durationSeconds > 0 ? Math.round((currentSeconds / durationSeconds) * 1000) : 0;
        """,
        "sync_play_state": """
          play_btn.textContent = video_el.paused ? "Play" : "Pause";
        """,
        "sync_speed_state": """
          speed_select.value = String(video_el.playbackRate || 1);
        """,
    }

APP_DIR = Path(__file__).resolve().parent
QUICKLOOK_ROOT = Path(os.environ.get("AURORA_QUICKLOOK_ROOT", APP_DIR / "quicklooks"))
QUICKLOOK_DISPLAY_CACHE = Path(
    os.environ.get("AURORA_QUICKLOOK_DISPLAY_CACHE", "/data/aurora/products/dashboard/quicklook_display_cache")
)
QUICKLOOK_TRIM_THRESHOLD_PX = int(os.environ.get("AURORA_QUICKLOOK_TRIM_THRESHOLD_PX", "32"))
QUICKLOOK_TRIM_PADDING_PX = int(os.environ.get("AURORA_QUICKLOOK_TRIM_PADDING_PX", "24"))
QUICKLOOK_WHITE_THRESHOLD = int(os.environ.get("AURORA_QUICKLOOK_WHITE_THRESHOLD", "250"))
OPS_SNAPSHOT_PATH = Path(os.environ.get("OPS_MONITOR_SNAPSHOT_PATH", "/project/aurora/raw/ops_monitor/latest.json"))
ARCHIVE_HEALTH_PATH = Path(
    os.environ.get(
        "ARCHIVE_HEALTH_PATH",
        "/data/aurora/internal/archive_status/health-v1.json",
    )
)
PERF_LOG_ENABLED = os.environ.get("AURORA_DASHBOARD_PERF_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}
PERF_LOG_PATH = Path(os.environ.get("AURORA_DASHBOARD_PERF_LOG", "/data/aurora/products/dashboard/dashboard_perf.jsonl"))
PERF_LOG_MAX_BYTES = int(os.environ.get("AURORA_DASHBOARD_PERF_LOG_MAX_BYTES", str(10 * 1024 * 1024)))
PERF_LOG_BACKUP_COUNT = int(os.environ.get("AURORA_DASHBOARD_PERF_LOG_BACKUP_COUNT", "5"))
SESSION_HEARTBEAT_MS = int(os.environ.get("AURORA_DASHBOARD_SESSION_HEARTBEAT_MS", "0"))
_SESSION_BOOT_TS = datetime.now(timezone.utc)
OPS_TREND_CACHE_TTL = timedelta(minutes=int(os.environ.get("AURORA_OPS_TREND_CACHE_TTL_MINUTES", "5")))
OPS_TREND_WINDOW = timedelta(days=int(os.environ.get("AURORA_OPS_TREND_DAYS", "7")))
OPS_BATTERY_CAPACITY_KWH = float(os.environ.get("APS_BATTERY_CAPACITY_KWH", "26"))
OPS_BATTERY_DEPLETION_DEADBAND_W = float(os.environ.get("APS_BATTERY_DEPLETION_DEADBAND_W", "50"))


def _session_age_seconds() -> float:
    return round((datetime.now(timezone.utc) - _SESSION_BOOT_TS).total_seconds(), 3)


def _normalize_perf_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        dt = value.astimezone(timezone.utc) if value.tzinfo else value
        return dt.isoformat()
    if isinstance(value, timedelta):
        return round(value.total_seconds(), 6)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_normalize_perf_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize_perf_value(item) for key, item in value.items()}
    return str(value)


_PERF_LOGGER = logging.getLogger("aurora.dashboard.perf")
_PERF_LOGGER.setLevel(logging.INFO)
_PERF_LOGGER.propagate = False
_PERF_LOG_READY = False

if PERF_LOG_ENABLED:
    if _PERF_LOGGER.handlers:
        # Panel executes application modules for multiple sessions in one
        # process. A prior session may have installed the handler already.
        _PERF_LOG_READY = True
    else:
        try:
            PERF_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            _perf_handler = RotatingFileHandler(
                PERF_LOG_PATH,
                maxBytes=PERF_LOG_MAX_BYTES,
                backupCount=PERF_LOG_BACKUP_COUNT,
            )
            _perf_handler.setFormatter(logging.Formatter("%(message)s"))
            _PERF_LOGGER.addHandler(_perf_handler)
            _PERF_LOG_READY = True
        except Exception as exc:
            print(f"[perf] disabled: could not initialize {PERF_LOG_PATH}: {exc}")


def _perf_log(event: str, duration_ms: float | None = None, **fields) -> None:
    if not _PERF_LOG_READY:
        return
    instrument = fields.get("instrument", globals().get("CURRENT_INSTRUMENT"))
    session_id = fields.get("session_id", _session_id())
    record = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "duration_ms": None if duration_ms is None else round(float(duration_ms), 3),
        "session_id": session_id,
        "instrument": instrument,
        "live_sessions": fields.get("live_sessions", _live_session_count()),
        "server_sessions": fields.get("server_sessions", _server_session_count()),
        "total_sessions": fields.get("total_sessions", _total_session_count()),
        "session_age_s": fields.get("session_age_s", _session_age_seconds()),
        "busy": fields.get("busy", bool(getattr(pn.state, "busy", False))),
    }
    for key, value in fields.items():
        record[key] = _normalize_perf_value(value)
    try:
        _PERF_LOGGER.info(json.dumps(record, sort_keys=True))
    except Exception as exc:
        print(f"[perf] write failed for {event}: {exc}")


_BROWSER_PERF_EVENTS = {
    "browser_document_ready",
    "browser_first_power_plot",
    "browser_power_section_switch",
}


def _browser_rum_enabled() -> bool:
    """Return whether development browser measurements are enabled."""
    return os.environ.get("AURORA_BROWSER_RUM_ENABLED", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class BrowserPerformanceProbe(pn.custom.JSComponent):
    """Report real browser milestones to the development performance log."""

    message = param.Dict(default={})

    _esm = """
    export function render({ model }) {
      const marker = document.createElement("span");
      marker.hidden = true;
      marker.setAttribute("aria-hidden", "true");

      const roots = () => {
        const found = [document];
        for (let index = 0; index < found.length; index += 1) {
          const root = found[index];
          for (const element of root.querySelectorAll("*")) {
            if (element.shadowRoot && !found.includes(element.shadowRoot)) {
              found.push(element.shadowRoot);
            }
          }
        }
        return found;
      };
      const plots = () => roots().flatMap(
        (root) => Array.from(root.querySelectorAll(".js-plotly-plot"))
      ).filter((node) => node.getClientRects().length > 0);
      const plotSignature = () => {
        const nodes = plots();
        // Plotly stores trace metadata on the SVG renderer but not on the
        // WebGL canvas renderer. Use the visible axis/title text as a stable
        // render signature so Current and Forecast remain distinguishable.
        const labels = nodes.map((node) => Array.from(node.querySelectorAll("text"))
          .map((text) => text.textContent.trim())
          .filter(Boolean)
          .join("|"))
          .join("||");
        return `${nodes.length}:${labels}`;
      };
      const navigation = performance.getEntriesByType("navigation")[0] || {};
      const emit = (event, duration, extra = {}) => {
        const payload = {
          event,
          duration_ms: Math.round(Number(duration) * 1000) / 1000,
          path: `${location.pathname}${location.search}`,
          viewport_width: window.innerWidth,
          viewport_height: window.innerHeight,
          ...extra,
        };
        window.setTimeout(() => {
          model.message = { ...payload, nonce: Date.now() };
        }, 0);
      };

      // This component itself arrives with the first Bokeh document, so a
      // paint frame is sufficient to observe the shell. The former one-second
      // timeout inflated the reported shell latency without delaying the UI.
      const reportDocumentReady = () => emit("browser_document_ready", performance.now(), {
          response_start_ms: Number(navigation.responseStart || 0),
          dom_interactive_ms: Number(navigation.domInteractive || 0),
          load_event_end_ms: Number(navigation.loadEventEnd || 0),
          navigation_type: String(navigation.type || "unknown"),
        });
      if (typeof window.requestAnimationFrame === "function") {
        window.requestAnimationFrame(reportDocumentReady);
      } else {
        window.setTimeout(reportDocumentReady, 0);
      }

      let firstPlotSent = false;
      let firstPlotTimer = null;
      const checkFirstPlot = () => {
        if (firstPlotSent) return;
        if (firstPlotTimer === null) {
          const started = performance.now();
          firstPlotTimer = window.setInterval(() => {
          const nodes = plots();
          if (!nodes.length && performance.now() - started < 20000) return;
          window.clearInterval(firstPlotTimer);
          firstPlotTimer = null;
          if (!nodes.length) return;
          firstPlotSent = true;
          emit("browser_first_power_plot", performance.now(), {
            plot_count: nodes.length,
            renderer: nodes.some((node) => node.querySelector("canvas")) ? "webgl" : "svg",
          });
          }, 100);
        }
      };

      let switchTimer = null;
      const onClick = (event) => {
        // Panel's button group is rendered in a shadow root. ``closest`` on
        // the retargeted event node cannot see that button, whereas the
        // composed path preserves the original element for this measurement.
        const button = event.composedPath().find(
          (node) => node instanceof HTMLElement && node.matches("button")
        );
        if (!button) return;
        const label = button.textContent.trim();
        if (label !== "Current Conditions" && label !== "Forecast & Planning") return;
        if (switchTimer !== null) window.clearInterval(switchTimer);
        const started = performance.now();
        const original = plotSignature();
        switchTimer = window.setInterval(() => {
          const elapsed = performance.now() - started;
          const signature = plotSignature();
          if (signature === original && elapsed < 20000) return;
          window.clearInterval(switchTimer);
          switchTimer = null;
          emit("browser_power_section_switch", elapsed, {
            section: label,
            plot_signature_before: original,
            plot_signature_after: signature,
            timed_out: elapsed >= 20000,
          });
        }, 100);
      };

      document.addEventListener("click", onClick, true);
      // Panel nests figures inside several shadow roots. Poll briefly while
      // the initial document hydrates; a document-only MutationObserver does
      // not observe mutations made inside those nested component roots.
      checkFirstPlot();

      marker.remove = (() => {
        const remove = marker.remove.bind(marker);
        return () => {
          document.removeEventListener("click", onClick, true);
          if (firstPlotTimer !== null) window.clearTimeout(firstPlotTimer);
          if (switchTimer !== null) window.clearInterval(switchTimer);
          remove();
        };
      })();
      return marker;
    }
    """

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.param.watch(self._handle_param_message, "message")

    def _handle_param_message(self, event) -> None:
        self._handle_msg(event.new)

    def _handle_msg(self, data) -> None:
        if not isinstance(data, dict):
            return
        event = str(data.get("event", ""))
        if event not in _BROWSER_PERF_EVENTS:
            return
        try:
            duration_ms = float(data.get("duration_ms"))
        except (TypeError, ValueError):
            return
        fields = {
            str(key): value
            for key, value in data.items()
            if key not in {"event", "duration_ms"}
        }
        _perf_log(event, duration_ms=duration_ms, instrument="power", **fields)


def _browser_performance_probe() -> BrowserPerformanceProbe | None:
    if SITE_ENV != "development" or not _browser_rum_enabled():
        return None
    # A real 1 px box keeps Panel's global deferred loader from skipping the
    # probe while remaining visually imperceptible.
    return BrowserPerformanceProbe(width=1, height=1, margin=0, sizing_mode="fixed")


@contextmanager
def _timed_perf(event: str, **fields):
    details = dict(fields)
    start = perf_counter()
    try:
        yield details
    except Exception as exc:
        details.setdefault("status", "error")
        details.setdefault("error_type", type(exc).__name__)
        details.setdefault("error", str(exc))
        raise
    finally:
        duration_ms = (perf_counter() - start) * 1000.0
        _perf_log(event, duration_ms=duration_ms, **details)
        if event in {"interactive_view_update", "stacked_timeseries_render"} and duration_ms > INTERACTIVE_RENDER_BUDGET_MS:
            _perf_log(
                "interactive_render_budget_exceeded",
                instrument=details.get("instrument"),
                source_event=event,
                duration_ms=duration_ms,
                budget_ms=INTERACTIVE_RENDER_BUDGET_MS,
                status=details.get("status", "unknown"),
            )


def _path_from_env(env_name: str, default: Path) -> Path:
    return Path(os.environ.get(env_name, default))


def _quicklook_display_path(path: Path) -> Path:
    """Return a display copy with trailing blank PNG canvas trimmed when useful."""
    if path.suffix.lower() != ".png" or Image is None:
        return path
    try:
        stat = path.stat()
    except OSError:
        return path
    return _quicklook_display_path_cached(str(path), stat.st_size, stat.st_mtime_ns)


@lru_cache(maxsize=512)
def _quicklook_display_path_cached(path_str: str, size_bytes: int, mtime_ns: int) -> Path:
    source = Path(path_str)
    if Image is None:
        return source
    try:
        with Image.open(source) as image:
            if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
                rgba = image.convert("RGBA")
                white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                white.alpha_composite(rgba)
                rgb = white.convert("RGB")
            else:
                rgb = image.convert("RGB")
        arr = np.asarray(rgb)
        content_rows = np.where(np.any(arr < QUICKLOOK_WHITE_THRESHOLD, axis=2).any(axis=1))[0]
        if content_rows.size == 0:
            return source
        bottom = min(rgb.height, int(content_rows[-1]) + 1 + QUICKLOOK_TRIM_PADDING_PX)
        if rgb.height - bottom < QUICKLOOK_TRIM_THRESHOLD_PX:
            return source
        key = hashlib.sha1(f"{source.resolve()}:{size_bytes}:{mtime_ns}:{bottom}".encode("utf-8")).hexdigest()[:16]
        QUICKLOOK_DISPLAY_CACHE.mkdir(parents=True, exist_ok=True)
        cropped = QUICKLOOK_DISPLAY_CACHE / f"{source.stem}__trim_{key}.png"
        if not cropped.exists():
            rgb.crop((0, 0, rgb.width, bottom)).save(cropped, "PNG", optimize=True)
        return cropped
    except Exception as exc:
        logging.getLogger(__name__).debug("Could not trim quicklook PNG %s: %s", source, exc)
        return source


# --- Configuration ---
INSTRUMENTS = {
    "Ceilometer": {
        "zarr_env": "CEILOMETER_ZARR_PATH",
        "zarr_default": "/data/aurora/products/cl61/gamb2le_depolarisation_lidar_ceilometer_aurora.zarr",
        "chunk_spec": {"time": 600},
        "consolidated": True,
        "height_load_max": 10_000,
        "top_range_default": 8000,
        "vars": {
            "beta_att": {"label": "Attenuated Backscatter", "clim": (1e-7, 1e-4), "log": True, "colorscale": "Cividis"},
            "linear_depol_ratio": {"label": "Linear Depolarization Ratio", "clim": (0.0, 0.5), "log": False, "colorscale": "Viridis"},
        },
        "default_top": "beta_att",
        "default_bottom": "linear_depol_ratio",
        "quicklook_dir": _path_from_env("CEILOMETER_QUICKLOOK_DIR", QUICKLOOK_ROOT / "ceilometer"),
        "latest_image": _path_from_env("CEILOMETER_LATEST_IMAGE", QUICKLOOK_ROOT / "ceilometer" / "latest.png"),
    },
    "Cloud Radar": {
        "zarr_env": "CLOUD_RADAR_ZARR_PATH",
        "zarr_default": "/data/aurora/products/rpgfmcw94/cloud_radar.zarr",
        "chunk_spec": {"time": 400},
        "consolidated": True,
        "height_load_max": 9_000,
        "top_range_default": 9_000,
        "vars": {
            "ZE_dBZ": {"label": "ZE (dBZ)", "clim": (-30.0, 10.0), "log": False, "colorscale": radar_plotly_colorscale("ZE_dBZ")},
            "ZE45_dBZ": {"label": "ZE45 (dBZ)", "clim": (-30.0, 10.0), "log": False, "colorscale": radar_plotly_colorscale("ZE45_dBZ")},
            "MeanVel": {"label": "Mean Velocity (m/s)", "clim": (-5.0, 5.0), "log": False, "colorscale": radar_plotly_colorscale("MeanVel")},
            "ZDR": {"label": "ZDR (dB)", "clim": (-10.0, 6.0), "log": False, "colorscale": radar_plotly_colorscale("ZDR")},
            "SRCX": {"label": "SRCX", "clim": (0.0, 1.0), "log": False, "colorscale": radar_plotly_colorscale("SRCX")},
            "SpecWidth": {"label": "Spectrum Width (m/s)", "clim": (0.0, 3.0), "log": False, "colorscale": radar_plotly_colorscale("SpecWidth")},
            "SLDR": {"label": "SLDR (dB)", "clim": (-100.0, -10.0), "log": False, "colorscale": radar_plotly_colorscale("SLDR")},
            "Skew": {"label": "Skew", "clim": (-2.0, 2.0), "log": False, "colorscale": radar_plotly_colorscale("Skew")},
            "RHV": {"label": "RHV", "clim": (0.8, 1.0), "log": False, "colorscale": radar_plotly_colorscale("RHV")},
            "PhiDP": {"label": "PhiDP (rad)", "clim": (-2.0, 2.0), "log": False, "colorscale": radar_plotly_colorscale("PhiDP")},
            "Kurt": {"label": "Kurtosis", "clim": (0.0, 8.0), "log": False, "colorscale": radar_plotly_colorscale("Kurt")},
            "KDP": {"label": "KDP (rad/km)", "clim": (-4.0, 4.0), "log": False, "colorscale": radar_plotly_colorscale("KDP")},
            "DiffAtt": {"label": "Differential Attenuation (dB/km)", "clim": (-5.0, 5.0), "log": False, "colorscale": radar_plotly_colorscale("DiffAtt")},
        },
        "default_top": "ZE_dBZ",
        "default_bottom": "MeanVel",
        "quicklook_dir": _path_from_env("CLOUD_RADAR_QUICKLOOK_DIR", QUICKLOOK_ROOT / "cloud_radar"),
        "latest_image": _path_from_env("CLOUD_RADAR_LATEST_IMAGE", QUICKLOOK_ROOT / "cloud_radar" / "latest.png"),
    },
    "vaisalamet": {
        "zarr_env": "VAISALAMET_ZARR_PATH",
        "zarr_default": "/data/aurora/products/vaisalamet/vaisalamet.zarr",
        "chunk_spec": {"time": 1200},
        "consolidated": True,
        "height_load_max": 1,
        "top_range_default": 1,
        "vars": widget_group_options("vaisalamet"),
        "default_top": default_interactive_label("vaisalamet"),
        "default_bottom": default_interactive_label("vaisalamet"),
        "default_calendar": default_calendar_label("vaisalamet"),
        "quicklook_dir": _path_from_env("VAISALAMET_QUICKLOOK_DIR", QUICKLOOK_ROOT / "vaisalamet"),
        "latest_image": _path_from_env("VAISALAMET_LATEST_IMAGE", QUICKLOOK_ROOT / "vaisalamet" / "latest.png"),
    },
    "asfs-logger": {
        "zarr_env": "ASFS_LOGGER_ZARR_PATH",
        "zarr_default": "/data/aurora/products/asfs_logger/asfs_logger.zarr",
        "chunk_spec": {"time": 1200},
        "consolidated": True,
        "height_load_max": 1,
        "top_range_default": 1,
        "vars": widget_group_options("asfs-logger"),
        "default_top": default_interactive_label("asfs-logger"),
        "default_bottom": default_interactive_label("asfs-logger"),
        "default_calendar": default_calendar_label("asfs-logger"),
        "quicklook_dir": _path_from_env("ASFS_LOGGER_QUICKLOOK_DIR", QUICKLOOK_ROOT / "asfs_logger"),
        "latest_image": _path_from_env("ASFS_LOGGER_LATEST_IMAGE", QUICKLOOK_ROOT / "asfs_logger" / "latest.png"),
    },
    "asfs-fast-sonic": {
        "zarr_env": "ASFS_FAST_SONIC_ZARR_PATH",
        "zarr_default": "/data/aurora/products/asfs_fast_sonic/asfs_fast_sonic.zarr",
        "chunk_spec": {"time": 24000},
        "consolidated": True,
        "height_load_max": 1,
        "top_range_default": 1,
        "vars": widget_group_options("asfs-fast-sonic"),
        "default_top": default_interactive_label("asfs-fast-sonic"),
        "default_bottom": default_interactive_label("asfs-fast-sonic"),
        "default_calendar": default_calendar_label("asfs-fast-sonic"),
        "quicklook_dir": _path_from_env("ASFS_FAST_SONIC_QUICKLOOK_DIR", QUICKLOOK_ROOT / "asfs_fast_sonic"),
        "latest_image": _path_from_env("ASFS_FAST_SONIC_LATEST_IMAGE", QUICKLOOK_ROOT / "asfs_fast_sonic" / "latest.png"),
    },
    "power": {
        "zarr_env": "POWER_ZARR_PATH",
        "zarr_default": "/data/aurora/products/power/power.zarr",
        # The power source is high-frequency. Larger dashboard read chunks avoid
        # thousands of tiny Dask tasks when building the latest 24 h browser view.
        "chunk_spec": {"time": 24000},
        "consolidated": True,
        "height_load_max": 1,
        "top_range_default": 1,
        "vars": widget_group_options("power"),
        "default_top": default_interactive_label("power"),
        "default_bottom": default_interactive_label("power"),
        "default_calendar": default_calendar_label("power"),
        "quicklook_dir": _path_from_env("POWER_QUICKLOOK_DIR", QUICKLOOK_ROOT / "power"),
        "latest_image": _path_from_env("POWER_LATEST_IMAGE", QUICKLOOK_ROOT / "power" / "latest.png"),
    },
    "ops-monitor": {
        "zarr_env": "OPS_MONITOR_ZARR_PATH",
        "zarr_default": "/data/aurora/products/ops_monitor/ops_monitor.zarr",
        "chunk_spec": {"time": 720},
        "consolidated": True,
        "height_load_max": 1,
        "top_range_default": 1,
        "vars": widget_group_options("ops-monitor"),
        "default_top": default_interactive_label("ops-monitor"),
        "default_bottom": default_interactive_label("ops-monitor"),
        "default_calendar": default_calendar_label("ops-monitor"),
        "quicklook_dir": _path_from_env("OPS_MONITOR_QUICKLOOK_DIR", QUICKLOOK_ROOT / "ops_monitor"),
        "latest_image": _path_from_env("OPS_MONITOR_LATEST_IMAGE", QUICKLOOK_ROOT / "ops_monitor" / "latest.png"),
    },
    "wxcam": {
        "zarr_env": "WXCAM_ZARR_PATH",
        "zarr_default": "/data/aurora/products/wxcam/wxcam.zarr",
        "catalog_env": "WXCAM_CATALOG_PATH",
        "catalog_default": "/data/aurora/products/wxcam/wxcam_catalog.sqlite",
        "chunk_spec": {"time": 1},
        "consolidated": True,
        "height_load_max": 1,
        "top_range_default": 1,
        "vars": {
            spec["label"]: {
                "label": spec["label"],
                "image_type": image_type,
                "clim": (0.0, 1.0),
                "log": False,
                "colorscale": "Viridis",
            }
            for image_type, spec in WXCAM_IMAGE_TYPES.items()
        },
        "default_top": next((spec["label"] for spec in WXCAM_IMAGE_TYPES.values()), "FISH HDR"),
        "default_bottom": next((spec["label"] for spec in WXCAM_IMAGE_TYPES.values()), "FISH HDR"),
        "quicklook_dir": _path_from_env("WXCAM_QUICKLOOK_DIR", QUICKLOOK_ROOT / "wxcam"),
        "latest_image": _path_from_env("WXCAM_LATEST_IMAGE", QUICKLOOK_ROOT / "wxcam" / "latest.jpg"),
    },
    "Scanning Microwave Radiometer": {
        "zarr_env": "HATPRO_ZARR_PATH",
        "zarr_default": "/data/aurora/products/hatprog5/hatpro.zarr",
        "chunk_spec": {"time": 600},
        "consolidated": True,
        "height_load_max": 10_000,
        "top_range_default": 10_000,
        "vars": {
            "T_PROF": {"label": "Temperature Profile (K)", "clim": (210.0, 310.0), "log": False, "colorscale": "Inferno"},
        },
        "default_top": "T_PROF",
        "default_bottom": "T_PROF",
        "quicklook_dir": _path_from_env("HATPRO_QUICKLOOK_DIR", QUICKLOOK_ROOT / "hatpro"),
        "latest_image": _path_from_env("HATPRO_LATEST_IMAGE", QUICKLOOK_ROOT / "hatpro" / "latest.png"),
    },
}

INSTRUMENT_OPTIONS = browser_options()
SCIENCE_INSTRUMENT_OPTIONS = science_options()
HK_INSTRUMENT_OPTIONS = browser_options(housekeeping=True)

DEFAULT_WINDOW = timedelta(hours=24)
LIVE_REFRESH_MS = 60_000  # how often to snap to latest when live is on (ms)
TIME_SUBSAMPLE = 2  # slice time to lighten payloads
TIME_TARGET = 300  # target max time samples for plotting
HEIGHT_TARGET = 200  # target max height samples for plotting
DATA_REFRESH_MS = 300_000  # reload base dataset every 5 minutes
RENDER_DEBOUNCE_MS = int(os.environ.get("AURORA_RENDER_DEBOUNCE_MS", "150"))
INTERACTIVE_RENDER_BUDGET_MS = int(os.environ.get("AURORA_INTERACTIVE_RENDER_BUDGET_MS", "10000"))
INTERACTIVE_RENDER_CACHE_SIZE = int(os.environ.get("AURORA_INTERACTIVE_RENDER_CACHE_SIZE", "12"))
POWER_INTERACTIVE_MAX_TIME_SAMPLES = int(os.environ.get("AURORA_POWER_INTERACTIVE_MAX_TIME_SAMPLES", "700"))
POWER_LATEST_CACHE_ROUND_MINUTES = int(os.environ.get("AURORA_POWER_LATEST_CACHE_ROUND_MINUTES", "5"))
POWER_GENERAL_CACHE_ROUND_MINUTES = int(os.environ.get("AURORA_POWER_GENERAL_CACHE_ROUND_MINUTES", "1"))
POWER_LATEST_CACHE_TOLERANCE = timedelta(minutes=int(os.environ.get("AURORA_POWER_LATEST_CACHE_TOLERANCE_MINUTES", "10")))
PREWARM_LATEST_CACHE_TOLERANCE = timedelta(minutes=int(os.environ.get("AURORA_PREWARM_LATEST_CACHE_TOLERANCE_MINUTES", "30")))
SUMMARY_INTERACTIVE_MAX_TIME_SAMPLES = {
    "power": POWER_INTERACTIVE_MAX_TIME_SAMPLES,
    "vaisalamet": int(os.environ.get("AURORA_MET_INTERACTIVE_MAX_TIME_SAMPLES", "1200")),
    "asfs-logger": int(os.environ.get("AURORA_RADIATION_INTERACTIVE_MAX_TIME_SAMPLES", "1400")),
    "ops-monitor": int(os.environ.get("AURORA_OPS_INTERACTIVE_MAX_TIME_SAMPLES", "1000")),
}
SUMMARY_INTERACTIVE_COARSE_TIME_SAMPLES = int(os.environ.get("AURORA_SUMMARY_COARSE_TIME_SAMPLES", "700"))
UAS_MQTT_LOG_PATH = Path(os.environ.get("UAS_MQTT_LOG_PATH", "/project/aurora/raw/menapia/menapia_mqtt.log"))
UAS_QUICKLOOK_DIR = Path(os.environ.get("UAS_QUICKLOOK_DIR", QUICKLOOK_ROOT / "uas"))
UAS_STALE_AFTER = timedelta(minutes=int(os.environ.get("UAS_STALE_AFTER_MINUTES", "5")))
UAS_WINDOW_OPTIONS = {
    "Last 1 h": timedelta(hours=1),
    "Last 6 h": timedelta(hours=6),
    "Last 24 h": timedelta(hours=24),
    "Last 7 d": timedelta(days=7),
}
# A small future tolerance keeps normal clock skew harmless while protecting
# the dashboard from bogus outlier timestamps that can blank the latest window.
FUTURE_TIME_TOLERANCE = timedelta(days=2)
TIME_BOUNDS_CACHE_TTL = timedelta(seconds=45)
INTERACTIVE_PLACEHOLDER_HEIGHT = 540

_BASE_DS: dict[str, xr.Dataset | None] = {}
_TIME_BOUNDS_CACHE: dict[str, dict[str, object]] = {}
_INTERACTIVE_FIGURE_CACHE: dict[str, go.Figure] = {}
_INTERACTIVE_RENDER_CACHE: OrderedDict[tuple[object, ...], go.Figure] = OrderedDict()
_IN_FLIGHT_INTERACTIVE_RENDER_CACHE_KEYS: set[tuple[object, ...]] = set()
_INSTRUMENT_VIEW_STATE: dict[str, dict[str, object]] = {}
_DATASET_VERSION: dict[str, int] = {}
_DATASET_REFRESHED_AT: dict[str, datetime] = {}
CURRENT_INSTRUMENT = "power"
_RENDER_REQUEST_COUNTER = 0
_ACTIVE_RENDER_REQUEST_ID = 0
_BACKGROUND_RENDER_TASKS: dict[int, asyncio.Task] = {}
# Plot creation is CPU and IO bound, while Panel/Bokeh model changes must stay
# on the document callback.  Two workers keep custom Power requests responsive
# without multiplying the memory use of the dashboard process.
_BACKGROUND_PREPARATION_EXECUTOR = ThreadPoolExecutor(
    max_workers=max(1, min(int(os.environ.get("AURORA_DASHBOARD_PREP_WORKERS", "2")), 4)),
    thread_name_prefix="aurora-dashboard-prep",
)
_POWER_PREPARATION_LOCK = RLock()
_DISPLAYED_INTERACTIVE_INSTRUMENT: str | None = None
_PENDING_INTERACTIVE_RENDER_ARGS: tuple[object, ...] | None = None
_PENDING_INTERACTIVE_RENDER_CB = None
_APP_BOOTSTRAPPING = True
_INTERACTIVE_RENDER_ENABLED = False
_INTERACTIVE_FOOTER_LOADED = False
_POWER_DISPLAY_ENERGY_DS: xr.Dataset | None = None
_POWER_DISPLAY_ENERGY_REFRESHED_AT: datetime | None = None
_POWER_DISPLAY_SUMMARY_DS: xr.Dataset | None = None
_POWER_DISPLAY_SUMMARY_REFRESHED_AT: datetime | None = None
_POWER_DISPLAY_SUMMARY_SOURCE_PATH: Path | None = None
_POWER_DISPLAY_SECTION_DS: dict[str, xr.Dataset] = {}
_POWER_DISPLAY_SECTION_REFRESHED_AT: dict[str, datetime] = {}
_POWER_DISPLAY_SECTION_SOURCE_PATH: dict[str, Path] = {}
_POWER_OPERATING_SCENARIOS_DS: xr.Dataset | None = None
_POWER_OPERATING_SCENARIOS_REFRESHED_AT: datetime | None = None
_POWER_OPERATING_SCENARIOS_SOURCE_PATH: Path | None = None
_OPS_TREND_CACHE: dict[str, object] = {"updated_at": None, "markup": ""}
_SESSION_PERIODIC_CALLBACKS: list[object] = []


def _utcnow_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _safe_periodic_callback(callback, period: int, start: bool = True):
    """Register a Panel timer, but allow plain Python imports for smoke tests."""
    timer = pn.state.add_periodic_callback(callback, period=period, start=False)
    _SESSION_PERIODIC_CALLBACKS.append(timer)
    if start:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return timer
        try:
            timer.start()
        except RuntimeError as exc:
            # A normal `python -c "import app"` has no running asyncio loop.
            # `panel serve` provides one, so this only affects local smoke tests.
            if "no running event loop" not in str(exc):
                raise
    return timer


def _cfg(inst: str | None = None):
    return INSTRUMENTS[inst or CURRENT_INSTRUMENT]


def _zarr_path(inst: str | None = None):
    cfg = _cfg(inst)
    return os.environ.get(cfg["zarr_env"], cfg["zarr_default"])


def _power_display_energy_path() -> Path:
    return Path(os.environ.get("POWER_DISPLAY_ENERGY_ZARR_PATH", "/data/aurora/products/power/power_display_energy.zarr"))


def _power_display_summary_path() -> Path:
    return mobile_catalog.power_display_summary_path()


def _power_display_section_path(section: str) -> Path:
    """Return the compact display store for one Power browser section."""
    if section not in {"current", "forecast"}:
        raise ValueError(f"Unsupported Power display section: {section}")
    return mobile_catalog.power_display_section_path(section)


def _power_display_summary_metadata_path() -> Path:
    active = mobile_catalog.power_forecast_active_product_path("displayManifest")
    if active is not None:
        return active
    configured = os.environ.get("POWER_DISPLAY_SUMMARY_METADATA_PATH", "").strip()
    if configured:
        return Path(configured)
    return _power_display_summary_path().with_name("power_display_summary_metadata.json")


def _power_display_summary_time_bounds_metadata() -> tuple[datetime | None, datetime | None]:
    """Read initial Power bounds without opening the 147-variable display Zarr."""
    path = _power_display_summary_metadata_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        lower = pd.Timestamp(payload.get("time_start_utc")).to_pydatetime(warn=False)
        upper = pd.Timestamp(payload.get("time_end_utc")).to_pydatetime(warn=False)
    except (OSError, TypeError, ValueError, KeyError):
        return None, None
    return _ensure_utc(lower), _ensure_utc(upper)


def _power_operating_scenarios_path() -> Path:
    return mobile_catalog.power_operating_scenario_paths()[0]


def _power_operating_scenario_paths() -> tuple[Path, ...]:
    """Return the configured scenario store followed by the mirrored live store."""
    return mobile_catalog.power_operating_scenario_paths()


def _power_operating_recommendations_path() -> Path:
    configured = os.environ.get("POWER_OPERATING_RECOMMENDATION_ARCHIVE_PATH", "").strip()
    if configured:
        return Path(configured)
    return _power_operating_scenarios_path().with_name("power_operating_recommendations.json")


def _prewarmed_interactive_dir() -> Path:
    return Path(os.environ.get("AURORA_INTERACTIVE_PREWARM_DIR", "/data/aurora/products/dashboard/prewarm"))


def _prewarmed_interactive_path(inst: str) -> Path:
    if inst == "power":
        section = power_view_select.value if "power_view_select" in globals() else "current"
        active_forecast = (
            mobile_catalog.power_forecast_active_display_path()
            if section == "forecast"
            else None
        )
        if active_forecast is not None:
            # The legacy prewarm is not part of the immutable generation. Once
            # activated, render from the validated bundle instead of mixing it
            # with an older Plotly cache.
            return active_forecast.parent / ".forecast-prewarm-not-published.json"
        return _prewarmed_interactive_dir() / f"power_{section}_latest_interactive.json"
    safe = inst.replace(" ", "_").replace("-", "_").lower()
    return _prewarmed_interactive_dir() / f"{safe}_latest_interactive.json"


def _wxcam_catalog_path(inst: str | None = None) -> Path:
    cfg = _cfg(inst)
    return Path(os.environ.get(cfg["catalog_env"], cfg["catalog_default"]))


def _wxcam_daily_video_root() -> Path:
    return Path(os.environ.get("WXCAM_DAILY_VIDEO_DIR", "/data/aurora/products/wxcam/daily_videos"))


def _wxcam_hourly_thumbnail_root() -> Path:
    return Path(os.environ.get("WXCAM_HOURLY_THUMB_DIR", "/data/aurora/products/wxcam/hourly_thumbnails"))


def _wxcam_media_root() -> Path:
    return Path(os.environ.get("WXCAM_MEDIA_ROOT", "/data/aurora/products/wxcam"))


def _wxcam_media_url(path: Path) -> str:
    """Return a browser URL for WXcam media served by Panel's static route."""
    root = _wxcam_media_root().resolve()
    resolved = path.resolve()
    rel = resolved.relative_to(root)
    prefix = os.environ.get("WXCAM_MEDIA_URL_PREFIX", "/wxcam-media").strip() or "/wxcam-media"
    prefix = "/" + prefix.strip("/")
    version = path.stat().st_mtime_ns
    return f"{prefix}/{quote(rel.as_posix())}?{urlencode({'v': version})}"


def _auroracam_raw_root() -> Path:
    return Path(os.environ.get("AURORACAM_RAW_ROOT", os.environ.get("AURORACAM_ROOT", "/project/aurora/raw/auroracam")))


def _auroracam_media_url(path: Path) -> str:
    """Return a browser URL for AURORACam JPEGs served by Panel's static route."""
    root = _auroracam_raw_root().resolve()
    resolved = path.resolve()
    rel = resolved.relative_to(root)
    prefix = os.environ.get("AURORACAM_MEDIA_URL_PREFIX", "/auroracam-media").strip() or "/auroracam-media"
    prefix = "/" + prefix.strip("/")
    version = path.stat().st_mtime_ns
    return f"{prefix}/{quote(rel.as_posix())}?{urlencode({'v': version})}"


def _ensure_utc(dt):
    """Return a naive UTC datetime (or None) for consistent comparisons."""
    if dt is None:
        return None
    tz = getattr(dt, "tzinfo", None)
    if tz is None:
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def _get_base_dataset(inst: str | None = None):
    """Open the Zarr store (memoized per instrument) with configured chunks and consolidation."""
    inst = inst or CURRENT_INSTRUMENT
    if inst in _BASE_DS and _BASE_DS[inst] is not None:
        return _BASE_DS[inst]
    cfg = _cfg(inst)
    zarr_path = _zarr_path(inst)
    print(f"[base-ds] open {inst} -> {zarr_path}")
    with _timed_perf(
        "base_dataset_open",
        instrument=inst,
        zarr_path=zarr_path,
        requested_chunks=cfg["chunk_spec"],
        requested_consolidated=cfg["consolidated"],
    ) as perf:
        try:
            ds = xr.open_zarr(zarr_path, chunks=cfg["chunk_spec"], consolidated=cfg["consolidated"])
            perf["status"] = "configured"
        except Exception as first_exc:
            perf["status"] = "fallback"
            perf["configured_error"] = str(first_exc)
            try:
                ds = xr.open_zarr(zarr_path, chunks="auto", consolidated=False)
                perf["fallback_chunks"] = "auto"
                perf["fallback_consolidated"] = False
            except Exception as second_exc:
                perf["status"] = "unavailable"
                perf["fallback_error"] = str(second_exc)
                print(f"[base-ds] unavailable for {inst}: {first_exc}; fallback failed: {second_exc}")
                _BASE_DS[inst] = None
                return None
        perf["dims"] = dict(ds.sizes)
    _BASE_DS[inst] = ds
    _DATASET_REFRESHED_AT[inst] = datetime.now(timezone.utc)
    return ds


def _refresh_base_dataset(inst: str | None = None):
    """Drop the cached dataset so the next access reopens the Zarr (captures new data)."""
    inst = inst or CURRENT_INSTRUMENT
    ds = _BASE_DS.get(inst)
    if ds is not None:
        try:
            ds.close()
        except Exception:
            pass
    _BASE_DS[inst] = None
    _TIME_BOUNDS_CACHE.pop(inst, None)
    _DATASET_VERSION[inst] = _DATASET_VERSION.get(inst, 0) + 1
    _DATASET_REFRESHED_AT[inst] = datetime.now(timezone.utc)
    if inst == "power":
        _refresh_power_display_energy_dataset()


def _refresh_time_bounds_cache(inst: str | None = None):
    """Invalidate lightweight timestamp metadata without closing open Zarr datasets."""
    _TIME_BOUNDS_CACHE.pop(inst or CURRENT_INSTRUMENT, None)


def _dataset_cache_age(inst: str | None = None) -> timedelta:
    """Return age of the opened dataset handle; old handles are reopened during live refresh."""
    inst = inst or CURRENT_INSTRUMENT
    refreshed_at = _DATASET_REFRESHED_AT.get(inst)
    if refreshed_at is None or _BASE_DS.get(inst) is None:
        return timedelta.max
    return datetime.now(timezone.utc) - refreshed_at


def _get_power_display_energy_dataset() -> xr.Dataset | None:
    """Open the compact Power display-energy store when it is available."""
    global _POWER_DISPLAY_ENERGY_DS, _POWER_DISPLAY_ENERGY_REFRESHED_AT
    if _POWER_DISPLAY_ENERGY_DS is not None:
        return _POWER_DISPLAY_ENERGY_DS
    path = _power_display_energy_path()
    if not path.exists():
        return None
    with _timed_perf("power_display_energy_open", instrument="power", zarr_path=str(path)) as perf:
        try:
            ds = xr.open_zarr(path, chunks={"time": 1440}, consolidated=True)
        except Exception as exc:
            perf["status"] = "unavailable"
            perf["error"] = str(exc)
            return None
        perf["status"] = "ok"
        perf["dims"] = dict(ds.sizes)
    _POWER_DISPLAY_ENERGY_DS = ds
    _POWER_DISPLAY_ENERGY_REFRESHED_AT = datetime.now(timezone.utc)
    return ds


def _get_power_display_summary_dataset() -> xr.Dataset | None:
    """Open the legacy combined Power display store as a compatibility fallback."""
    global _POWER_DISPLAY_SUMMARY_DS, _POWER_DISPLAY_SUMMARY_REFRESHED_AT
    global _POWER_DISPLAY_SUMMARY_SOURCE_PATH
    publication = mobile_catalog.power_forecast_bundle_status(
        _power_display_section_path("forecast")
    )
    if publication.get("enabled") and publication.get("status") == "unavailable":
        return None
    path = _power_display_summary_path()
    if (
        _POWER_DISPLAY_SUMMARY_DS is not None
        and _POWER_DISPLAY_SUMMARY_SOURCE_PATH == path
    ):
        return _POWER_DISPLAY_SUMMARY_DS
    if _POWER_DISPLAY_SUMMARY_DS is not None:
        try:
            _POWER_DISPLAY_SUMMARY_DS.close()
        except Exception:
            pass
        _POWER_DISPLAY_SUMMARY_DS = None
        _POWER_DISPLAY_SUMMARY_REFRESHED_AT = None
        _POWER_DISPLAY_SUMMARY_SOURCE_PATH = None
    if not path.exists():
        return None
    with _timed_perf("power_display_summary_open", instrument="power", zarr_path=str(path)) as perf:
        try:
            ds = xr.open_zarr(path, chunks={"time": 1440}, consolidated=True)
            ds = merge_operating_scenarios_into_display_summary(ds, _get_power_operating_scenarios_dataset())
        except Exception as exc:
            perf["status"] = "unavailable"
            perf["error"] = str(exc)
            return None
        perf["status"] = "ok"
        perf["dims"] = dict(ds.sizes)
        perf["var_count"] = len(ds.data_vars)
    _POWER_DISPLAY_SUMMARY_DS = ds
    _POWER_DISPLAY_SUMMARY_REFRESHED_AT = datetime.now(timezone.utc)
    _POWER_DISPLAY_SUMMARY_SOURCE_PATH = path
    return ds


def _get_power_display_section_dataset(section: str) -> xr.Dataset | None:
    """Open only the display variables required by a Power page section.

    The combined store is retained for existing scripts and old deployments, but
    normal browser renders must not open all observed and forecast fields.
    """
    path = _power_display_section_path(section)
    if section in {"current", "forecast"}:
        publication = mobile_catalog.power_forecast_bundle_status(
            _power_display_section_path("forecast")
        )
        if publication.get("enabled") and publication.get("status") == "unavailable":
            return None
    cached = _POWER_DISPLAY_SECTION_DS.get(section)
    if cached is not None and _POWER_DISPLAY_SECTION_SOURCE_PATH.get(section) == path:
        return cached
    if cached is not None:
        try:
            cached.close()
        except Exception:
            pass
        _POWER_DISPLAY_SECTION_DS.pop(section, None)
        _POWER_DISPLAY_SECTION_REFRESHED_AT.pop(section, None)
        _POWER_DISPLAY_SECTION_SOURCE_PATH.pop(section, None)
    if not path.exists():
        return None
    with _timed_perf("power_display_section_open", instrument="power", section=section, zarr_path=str(path)) as perf:
        try:
            ds = xr.open_zarr(path, chunks={"time": 1440}, consolidated=True)
        except Exception as exc:
            perf["status"] = "unavailable"
            perf["error"] = str(exc)
            return None
        perf["status"] = "ok"
        perf["dims"] = dict(ds.sizes)
        perf["var_count"] = len(ds.data_vars)
    _POWER_DISPLAY_SECTION_DS[section] = ds
    _POWER_DISPLAY_SECTION_REFRESHED_AT[section] = datetime.now(timezone.utc)
    _POWER_DISPLAY_SECTION_SOURCE_PATH[section] = path
    return ds


def _get_power_operating_scenarios_dataset() -> xr.Dataset | None:
    """Open the compact learned operating-plan product when available."""
    global _POWER_OPERATING_SCENARIOS_DS, _POWER_OPERATING_SCENARIOS_REFRESHED_AT
    global _POWER_OPERATING_SCENARIOS_SOURCE_PATH
    required = {"component", "SolarEnsembleWatts", "ComponentLoadWatts"}
    paths = _power_operating_scenario_paths()
    if _POWER_OPERATING_SCENARIOS_DS is not None and required.issubset(
        set(_POWER_OPERATING_SCENARIOS_DS.variables)
    ) and _POWER_OPERATING_SCENARIOS_SOURCE_PATH in paths:
        return _POWER_OPERATING_SCENARIOS_DS
    if _POWER_OPERATING_SCENARIOS_DS is not None:
        try:
            _POWER_OPERATING_SCENARIOS_DS.close()
        except Exception:
            pass
        _POWER_OPERATING_SCENARIOS_DS = None
        _POWER_OPERATING_SCENARIOS_REFRESHED_AT = None
        _POWER_OPERATING_SCENARIOS_SOURCE_PATH = None
    for path in paths:
        if not path.exists():
            continue
        with _timed_perf("power_operating_scenarios_open", instrument="power", zarr_path=str(path)) as perf:
            try:
                ds = xr.open_zarr(path, chunks={}, consolidated=True)
            except Exception as exc:
                perf["status"] = "unavailable"
                perf["error"] = str(exc)
                continue
            missing = sorted(required.difference(ds.variables))
            if missing:
                perf["status"] = "incomplete"
                perf["missing"] = missing
                try:
                    ds.close()
                except Exception:
                    pass
                continue
            perf["status"] = "ok"
            perf["dims"] = dict(ds.sizes)
        _POWER_OPERATING_SCENARIOS_DS = ds
        _POWER_OPERATING_SCENARIOS_REFRESHED_AT = datetime.now(timezone.utc)
        _POWER_OPERATING_SCENARIOS_SOURCE_PATH = path
        return ds
    return None


def _refresh_power_display_energy_dataset() -> None:
    """Drop compact Power display-product handles so latest products reopen."""
    global _POWER_DISPLAY_ENERGY_DS, _POWER_DISPLAY_ENERGY_REFRESHED_AT
    global _POWER_DISPLAY_SUMMARY_DS, _POWER_DISPLAY_SUMMARY_REFRESHED_AT
    global _POWER_DISPLAY_SUMMARY_SOURCE_PATH
    global _POWER_OPERATING_SCENARIOS_DS, _POWER_OPERATING_SCENARIOS_REFRESHED_AT
    global _POWER_OPERATING_SCENARIOS_SOURCE_PATH
    if _POWER_DISPLAY_ENERGY_DS is not None:
        try:
            _POWER_DISPLAY_ENERGY_DS.close()
        except Exception:
            pass
    _POWER_DISPLAY_ENERGY_DS = None
    _POWER_DISPLAY_ENERGY_REFRESHED_AT = None
    if _POWER_DISPLAY_SUMMARY_DS is not None:
        try:
            _POWER_DISPLAY_SUMMARY_DS.close()
        except Exception:
            pass
    _POWER_DISPLAY_SUMMARY_DS = None
    _POWER_DISPLAY_SUMMARY_REFRESHED_AT = None
    _POWER_DISPLAY_SUMMARY_SOURCE_PATH = None
    for ds in _POWER_DISPLAY_SECTION_DS.values():
        try:
            ds.close()
        except Exception:
            pass
    _POWER_DISPLAY_SECTION_DS.clear()
    _POWER_DISPLAY_SECTION_REFRESHED_AT.clear()
    _POWER_DISPLAY_SECTION_SOURCE_PATH.clear()
    if _POWER_OPERATING_SCENARIOS_DS is not None:
        try:
            _POWER_OPERATING_SCENARIOS_DS.close()
        except Exception:
            pass
    _POWER_OPERATING_SCENARIOS_DS = None
    _POWER_OPERATING_SCENARIOS_REFRESHED_AT = None
    _POWER_OPERATING_SCENARIOS_SOURCE_PATH = None


def _open_power_display_energy_window(start, end) -> xr.Dataset | None:
    ds = _get_power_display_energy_dataset()
    if ds is None or "time" not in ds:
        return None
    start_dt = _as_naive_utc_datetime(start)
    end_dt = _as_naive_utc_datetime(end)
    if start_dt is None or end_dt is None:
        return None
    with _timed_perf("power_display_energy_window", instrument="power", start=start_dt, end=end_dt) as perf:
        window = ds.sel(time=slice(start_dt, end_dt))
        perf["matched_time_count"] = int(window.sizes.get("time", 0))
        if not window.sizes.get("time", 0):
            perf["status"] = "empty"
            return None
        perf["status"] = "ok"
        perf["output_time_count"] = int(window.sizes.get("time", 0))
        return window


def _open_power_display_summary_window(start, end, section: str | None = None) -> xr.Dataset | None:
    """Open a bounded Power display window without scanning every forecast field."""
    section = section or (power_view_select.value if "power_view_select" in globals() else "current")
    ds = _get_power_display_section_dataset(section)
    legacy = ds is None
    if ds is None:
        ds = _get_power_display_summary_dataset()
    if ds is None or "time" not in ds:
        return None
    start_dt = _as_naive_utc_datetime(start)
    end_dt = _as_naive_utc_datetime(end)
    if start_dt is None or end_dt is None:
        return None
    with _timed_perf("power_display_summary_window", instrument="power", start=start_dt, end=end_dt) as perf:
        window_end = end_dt
        if section == "forecast":
            window_end = end_dt + pd.Timedelta(hours=float(os.environ.get("AURORA_POWER_SOC_FORECAST_HOURS", "96")))
        window = ds.sel(time=slice(start_dt, window_end))
        matched = int(window.sizes.get("time", 0))
        perf["matched_time_count"] = matched
        perf["section"] = section
        perf["legacy_store"] = legacy
        if not matched:
            perf["status"] = "empty"
            return None
        perf["status"] = "ok"
        perf["output_time_count"] = int(window.sizes.get("time", 0))
        perf["var_count"] = len(window.data_vars)
        return window


def _remember_time_bounds(inst: str, lower: datetime | None, upper: datetime | None) -> tuple[datetime | None, datetime | None]:
    _TIME_BOUNDS_CACHE[inst] = {
        "captured_at": datetime.now(timezone.utc),
        "bounds": (lower, upper),
    }
    return lower, upper


def _time_bounds_from_dataset(ds: xr.Dataset | None) -> tuple[datetime | None, datetime | None, int, int]:
    if ds is None or "time" not in ds:
        return None, None, 0, 0
    times = np.asarray(ds["time"].values)
    raw_count = int(times.size)
    if times.size == 0:
        return None, None, raw_count, 0
    valid = _valid_time_mask(times)
    times = times[valid]
    valid_count = int(times.size)
    if times.size == 0:
        return None, None, raw_count, valid_count
    lower = pd.Timestamp(times.min()).to_pydatetime(warn=False)
    upper = pd.Timestamp(times.max()).to_pydatetime(warn=False)
    return lower, upper, raw_count, valid_count


def _time_bounds_from_power_display_dataset(
    ds: xr.Dataset | None,
) -> tuple[datetime | None, datetime | None, int, int]:
    """Return bounds from measured APS rows, excluding forecast-only timestamps."""
    if ds is None or "time" not in ds:
        return None, None, 0, 0
    times = np.asarray(ds["time"].values)
    raw_count = int(times.size)
    if times.size == 0:
        return None, None, raw_count, 0
    measured = np.zeros(times.shape, dtype=bool)
    for name in POWER_DISPLAY_SUMMARY_FIELDS:
        if name not in ds or ds[name].dims != ("time",):
            continue
        values = np.asarray(ds[name].values)
        if np.issubdtype(values.dtype, np.number):
            measured |= np.isfinite(values)
    valid = _valid_time_mask(times) & measured
    measured_times = times[valid]
    valid_count = int(measured_times.size)
    if measured_times.size == 0:
        return None, None, raw_count, valid_count
    lower = pd.Timestamp(measured_times.min()).to_pydatetime(warn=False)
    upper = pd.Timestamp(measured_times.max()).to_pydatetime(warn=False)
    return lower, upper, raw_count, valid_count


def _valid_time_mask(times: np.ndarray) -> np.ndarray:
    """Mask out NaT and clearly bogus future timestamps while preserving original indices."""
    if times.size == 0:
        return np.zeros(times.shape, dtype=bool)
    valid = ~np.isnat(times)
    cutoff = np.datetime64(_ensure_utc(datetime.now(timezone.utc) + FUTURE_TIME_TOLERANCE))
    valid &= times <= cutoff
    return valid if np.any(valid) else ~np.isnat(times)


def _dataset_time_bounds(inst: str | None = None):
    """Compute earliest and latest timestamps in the dataset (or None/None)."""
    inst = inst or CURRENT_INSTRUMENT
    cached = _TIME_BOUNDS_CACHE.get(inst)
    if cached:
        captured_at = cached.get("captured_at")
        bounds = cached.get("bounds")
        if isinstance(captured_at, datetime) and isinstance(bounds, tuple):
            if datetime.now(timezone.utc) - captured_at <= TIME_BOUNDS_CACHE_TTL:
                _perf_log("dataset_time_bounds_cache_hit", instrument=inst)
                return bounds
    with _timed_perf("dataset_time_bounds", instrument=inst) as perf:
        if inst == "wxcam":
            lower, upper = catalog_time_bounds(_wxcam_catalog_path(inst))
            perf["source"] = "wxcam_catalog"
            perf["time_start"] = lower
            perf["time_end"] = upper
            return _remember_time_bounds(inst, lower, upper)
        if inst == "power":
            lower, upper = _power_display_summary_time_bounds_metadata()
            if lower is not None and upper is not None:
                perf["source"] = "power_display_summary_metadata"
                perf["time_start"] = lower
                perf["time_end"] = upper
                return _remember_time_bounds(inst, lower, upper)
            for source_name, getter in (
                ("power_display_summary", _get_power_display_summary_dataset),
                ("power_display_energy", _get_power_display_energy_dataset),
            ):
                source = getter()
                if source_name == "power_display_summary":
                    lower, upper, raw_count, valid_count = _time_bounds_from_power_display_dataset(source)
                else:
                    lower, upper, raw_count, valid_count = _time_bounds_from_dataset(source)
                perf[f"{source_name}_time_count"] = raw_count
                perf[f"{source_name}_valid_time_count"] = valid_count
                if lower is not None and upper is not None:
                    perf["source"] = source_name
                    perf["time_start"] = lower
                    perf["time_end"] = upper
                    return _remember_time_bounds(inst, lower, upper)
        ds = _get_base_dataset(inst)
        if ds is None or "time" not in ds:
            perf["status"] = "no_dataset"
            return _remember_time_bounds(inst, None, None)
        lower, upper, raw_count, valid_count = _time_bounds_from_dataset(ds)
        perf["raw_time_count"] = raw_count
        if raw_count == 0:
            perf["status"] = "empty"
            return _remember_time_bounds(inst, None, None)
        perf["valid_time_count"] = valid_count
        if lower is None or upper is None:
            perf["status"] = "empty"
            return _remember_time_bounds(inst, None, None)
        perf["time_start"] = lower
        perf["time_end"] = upper
        return _remember_time_bounds(inst, lower, upper)


def _instrument_time_index(inst: str) -> pd.DatetimeIndex:
    if inst == "wxcam":
        return pd.DatetimeIndex([])
    ds = _get_base_dataset(inst)
    if ds is None or "time" not in ds:
        return pd.DatetimeIndex([])
    times = np.asarray(ds["time"].values)
    if times.size == 0:
        return pd.DatetimeIndex([])
    valid = _valid_time_mask(times)
    return pd.DatetimeIndex(times[valid])


def _power_display_observed_time_index() -> pd.DatetimeIndex:
    """Return observed Power timestamps from the compact display product.

    The Power footer only needs freshness and coverage. Opening the full raw
    Power Zarr just to build those small status widgets defeats the standard
    view prewarm, so prefer the compact display-summary product. A raw-store
    fallback keeps the existing degraded-data behaviour when that product has
    not yet been built.
    """
    try:
        ds = _get_power_display_summary_dataset()
    except Exception:
        ds = None
    if ds is None or "time" not in ds:
        return _instrument_time_index("power")

    times = pd.DatetimeIndex(ds["time"].values)
    if not len(times):
        return pd.DatetimeIndex([])
    observed = np.zeros(len(times), dtype=bool)
    for name in ("BatterySOC", "BatteryWatts", "DCInverterVolts", "ACOutputWatts"):
        if name not in ds:
            continue
        values = np.asarray(ds[name].values, dtype=np.float64)
        if values.shape[0] == len(times):
            observed |= np.isfinite(values)
    now = pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None)
    return times[observed & (times <= now)]


def _status_time_index(inst: str) -> pd.DatetimeIndex:
    """Use compact Power metadata for status widgets; use raw indexes elsewhere."""
    return _power_display_observed_time_index() if inst == "power" else _instrument_time_index(inst)


def _format_status_time(dt: datetime | None) -> str:
    if dt is None:
        return "No data"
    stamp = dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    return stamp.strftime("%H:%M UTC")


def _format_duration(delta: timedelta | None) -> str:
    if delta is None:
        return "n/a"
    total_seconds = max(int(delta.total_seconds()), 0)
    days, rem = divmod(total_seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, _seconds = divmod(rem, 60)
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def _hourly_coverage_summary(times: pd.DatetimeIndex, start: datetime | None, end: datetime | None) -> tuple[list[bool], int, int]:
    if start is None or end is None or end < start:
        return [], 0, 0
    start_ts = pd.Timestamp(start).floor("h")
    end_ts = pd.Timestamp(end).floor("h")
    expected = pd.date_range(start=start_ts, end=end_ts, freq="1h")
    if len(expected) == 0:
        return [], 0, 0
    covered = set(pd.DatetimeIndex(times).floor("h"))
    bits = [stamp in covered for stamp in expected]
    missing = sum(1 for bit in bits if not bit)
    return bits, missing, len(expected)


def _binned_time_coverage(times: pd.DatetimeIndex, start: datetime | None, end: datetime | None, segments: int = 64) -> list[bool]:
    if start is None or end is None or end <= start or len(times) == 0:
        return []
    start_ns = pd.Timestamp(start).value
    end_ns = pd.Timestamp(end).value
    if end_ns <= start_ns:
        return []
    time_ns = pd.DatetimeIndex(times).asi8
    mask = (time_ns >= start_ns) & (time_ns <= end_ns)
    if not np.any(mask):
        return [False] * segments
    window = time_ns[mask]
    edges = np.linspace(start_ns, end_ns, segments + 1)
    bits: list[bool] = []
    for idx in range(segments):
        lo = edges[idx]
        hi = edges[idx + 1]
        if idx == segments - 1:
            bits.append(bool(np.any((window >= lo) & (window <= hi))))
        else:
            bits.append(bool(np.any((window >= lo) & (window < hi))))
    return bits


def _availability_hour_titles(start: datetime | None, end: datetime | None) -> list[str]:
    if start is None or end is None or end < start:
        return []
    start_ts = pd.Timestamp(start).floor("h")
    end_ts = pd.Timestamp(end).floor("h")
    expected = pd.date_range(start=start_ts, end=end_ts, freq="1h")
    return [f"{stamp.strftime('%H:00')} UTC" for stamp in expected]


def _availability_binned_titles(start: datetime | None, end: datetime | None, segments: int) -> list[str]:
    if start is None or end is None or end <= start or segments <= 0:
        return []
    start_ns = pd.Timestamp(start).value
    end_ns = pd.Timestamp(end).value
    if end_ns <= start_ns:
        return []
    edges = np.linspace(start_ns, end_ns, segments + 1)
    labels: list[str] = []
    for idx in range(segments):
        lo = pd.to_datetime(int(edges[idx]), utc=True)
        hi = pd.to_datetime(int(edges[idx + 1]), utc=True)
        labels.append(f"{lo.strftime('%m-%d %H:%M')} to {hi.strftime('%m-%d %H:%M')} UTC")
    return labels


def _availability_bucket_text(start: datetime | None, end: datetime | None, count: int) -> str:
    if start is None or end is None or count <= 0 or end <= start:
        return "Each block shows one slice of the selected range."
    seconds = (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() / count
    if seconds >= 3600:
        hours = seconds / 3600
        if abs(hours - round(hours)) < 0.15:
            value = int(round(hours))
            unit = "hour" if value == 1 else "hours"
            return f"Each block represents {value} {unit} of the selected range."
        return f"Each block represents about {hours:.1f} hours of the selected range."
    minutes = max(int(round(seconds / 60)), 1)
    unit = "minute" if minutes == 1 else "minutes"
    return f"Each block represents about {minutes} {unit} of the selected range."


def _wxcam_hour_bits(selection: str, day_token: str) -> list[bool]:
    rows_by_hour = _wxcam_hourly_image_rows(selection, day_token)
    return [hour in rows_by_hour for hour in range(24)]


def _wxcam_combined_hour_states(day_token: str) -> list[int]:
    day_utc = _wxcam_day_token_to_utc(day_token)
    if not day_utc:
        return [0] * 24
    records_by_type = {
        image_type: representative_hourly_records(
            _wxcam_catalog_path("wxcam"), image_type, day_utc, media_kind="image"
        )
        for image_type in WXCAM_IMAGE_TYPES
    }
    required_types = max(len(records_by_type), 1)
    states: list[int] = []
    for hour in range(24):
        present_types = sum(1 for rows in records_by_type.values() if hour in rows)
        states.append(2 if present_types == required_types else 1 if present_types else 0)
    return states


def _availability_bar_markup(
    states: list[int | bool],
    start_label: str,
    end_label: str,
    caption: str,
    explainer: str,
    full_label: str = "Data present",
    partial_label: str = "Partial coverage",
    empty_label: str = "No data",
    segment_titles: list[str] | None = None,
) -> str:
    if not states:
        return "<div class='availability-shell'><div class='availability-empty'>No availability information</div></div>"
    parts = []
    for idx, state in enumerate(states):
        if state in (True, 2):
            cls = "availability-segment availability-segment--full"
            state_label = full_label
        elif state == 1:
            cls = "availability-segment availability-segment--partial"
            state_label = partial_label
        else:
            cls = "availability-segment availability-segment--empty"
            state_label = empty_label
        title = state_label
        if segment_titles and idx < len(segment_titles):
            title = f"{segment_titles[idx]}: {state_label}"
        parts.append(f"<span class='{cls}' title='{escape(title)}'></span>")

    legend_parts = [
        "<span class='availability-legend-item'>"
        "<span class='availability-legend-swatch availability-segment--full'></span>"
        f"{escape(full_label)}"
        "</span>"
    ]
    if any(state == 1 for state in states):
        legend_parts.append(
            "<span class='availability-legend-item'>"
            "<span class='availability-legend-swatch availability-segment--partial'></span>"
            f"{escape(partial_label)}"
            "</span>"
        )
    legend_parts.append(
        "<span class='availability-legend-item'>"
        "<span class='availability-legend-swatch availability-segment--empty'></span>"
        f"{escape(empty_label)}"
        "</span>"
    )
    return (
        "<div class='availability-shell'>"
        f"<div class='availability-caption'>{escape(caption)}</div>"
        f"<div class='availability-explainer'>{escape(explainer)}</div>"
        f"<div class='availability-bar'>{''.join(parts)}</div>"
        f"<div class='availability-scale'><span>{escape(start_label)}</span><span>{escape(end_label)}</span></div>"
        f"<div class='availability-legend'>{''.join(legend_parts)}</div>"
        "</div>"
    )


def _status_strip_markup(items: list[tuple[str, str, str]]) -> str:
    pills = []
    for label, value, tone in items:
        pills.append(
            f"<span class='status-pill status-pill--{escape(tone)}'><strong>{escape(label)}</strong> {escape(value)}</span>"
        )
    return f"<div class='status-strip'>{''.join(pills)}</div>" if pills else ""


OPS_STREAM_SPECS = (
    {
        "label": "Ceilometer",
        "stream_prefix": "cl61",
        "source_key": "cl61_source_sync_service_healthy_state",
        "processing_keys": (
            "ceilometer_append_service_healthy_state",
            "ceilometer_quicklooks_service_healthy_state",
        ),
    },
    {
        "label": "Cloud Radar",
        "stream_prefix": "radar",
        "source_key": "radar_source_sync_service_healthy_state",
        "processing_keys": (
            "radar_append_service_healthy_state",
            "radar_quicklooks_service_healthy_state",
            "radar_daily_quicklooks_service_healthy_state",
        ),
    },
    {
        "label": "HATPRO",
        "stream_prefix": "hatpro",
        "source_key": "hatpro_source_sync_service_healthy_state",
        "processing_keys": (
            "hatpro_append_service_healthy_state",
            "hatpro_quicklooks_service_healthy_state",
        ),
    },
    {
        "label": "Meteorology",
        "stream_prefix": "vaisalamet",
        "source_key": "vaisalamet_source_sync_service_healthy_state",
        "processing_keys": (
            "vaisalamet_append_service_healthy_state",
            "vaisalamet_quicklooks_service_healthy_state",
        ),
    },
    {
        "label": "Radiation",
        "stream_prefix": "asfs_logger",
        "source_key": "asfs_logger_source_sync_service_healthy_state",
        "processing_keys": (
            "asfs_logger_append_service_healthy_state",
            "asfs_logger_quicklooks_service_healthy_state",
        ),
    },
    {
        "label": "ASFS Fast Sonic",
        "stream_prefix": "asfs_fast_sonic",
        "source_key": "asfs_fast_sonic_source_sync_service_healthy_state",
        "processing_keys": (
            "asfs_fast_sonic_append_service_healthy_state",
            "asfs_fast_sonic_quicklooks_service_healthy_state",
        ),
    },
    {
        "label": "Aurora Power Supply",
        "stream_prefix": "power",
        "source_key": "power_source_sync_service_healthy_state",
        "processing_keys": (
            "power_append_service_healthy_state",
            "power_quicklooks_service_healthy_state",
        ),
    },
    {
        "label": "WXcam",
        "stream_prefix": "wxcam",
        "source_key": "wxcam_source_sync_service_healthy_state",
        "processing_keys": (
            "wxcam_append_service_healthy_state",
            "wxcam_catalog_service_healthy_state",
            "wxcam_daily_videos_service_healthy_state",
        ),
    },
)

# These are the operational-stream prefixes for the three data-producing PDU
# outlets. A fresh off state means missing source data is intentional.
OPS_STREAM_PDU_OUTLETS = {"cl61": 5, "radar": 6, "hatpro": 8}


def _ops_snapshot_path() -> Path:
    return OPS_SNAPSHOT_PATH


def _ops_read_snapshot() -> dict:
    path = _ops_snapshot_path()
    if not path.exists():
        snapshot = {"_missing": True}
    else:
        try:
            snapshot = json.loads(path.read_text())
        except Exception as exc:
            snapshot = {"_error": str(exc)}
    try:
        archive = json.loads(ARCHIVE_HEALTH_PATH.read_text(encoding="utf-8"))
        metrics = archive.get("metrics", {})
        if isinstance(metrics, dict):
            snapshot.update(metrics)
        snapshot["_archive_health"] = archive
    except FileNotFoundError:
        snapshot["_archive_health_missing"] = str(ARCHIVE_HEALTH_PATH)
    except Exception as exc:
        snapshot["_archive_health_error"] = str(exc)
    snapshot["_path"] = str(path)
    return snapshot


def _ops_float(value) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return None


def _ops_bool(value) -> bool | None:
    number = _ops_float(value)
    if number is None:
        return None
    return bool(number)


def _ops_timestamp(value) -> datetime | None:
    if not value:
        return None
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.to_pydatetime(warn=False)


def _ops_level_value(level: str) -> int:
    order = {"green": 0, "amber": 1, "red": 2, "gray": -1}
    return order.get(level, -1)


def _ops_worst_level(levels: list[str]) -> str:
    meaningful = [level for level in levels if level != "gray"]
    if not meaningful:
        return "gray"
    return max(meaningful, key=_ops_level_value)


def _ops_level_from_bool(value) -> str:
    state = _ops_bool(value)
    if state is None:
        return "gray"
    return "green" if state else "red"


def _ops_level_from_count(value, amber_at: float = 1.0) -> str:
    count = _ops_float(value)
    if count is None:
        return "gray"
    if count <= 0:
        return "green"
    if count <= amber_at:
        return "amber"
    return "red"


def _ops_level_from_used_pct(value) -> str:
    used_pct = _ops_float(value)
    if used_pct is None:
        return "gray"
    if used_pct < 75.0:
        return "green"
    if used_pct < 90.0:
        return "amber"
    return "red"


def _ops_level_from_age_minutes(value) -> str:
    age_min = _ops_float(value)
    if age_min is None:
        return "gray"
    if age_min <= 10.0:
        return "green"
    if age_min <= 30.0:
        return "amber"
    return "red"


def _ops_level_from_source_probes(fail_count_value, total_hosts: int = 3) -> str:
    fail_count = _ops_float(fail_count_value)
    if fail_count is None:
        return "gray"
    if fail_count <= 0:
        return "green"
    if fail_count < total_hosts:
        return "amber"
    return "red"


def _ops_level_from_battery_voltage(value) -> str:
    voltage = _ops_float(value)
    if voltage is None:
        return "gray"
    if voltage > 52.0:
        return "green"
    if voltage >= 50.0:
        return "amber"
    return "red"


def _ops_level_from_battery_soc(value) -> str:
    soc = _ops_float(value)
    if soc is None:
        return "gray"
    if soc >= 50.0:
        return "green"
    if soc > MINIMUM_OPERATIONAL_SOC_PCT:
        return "amber"
    return "red"


def _ops_level_from_battery_depletion(snapshot: dict) -> str:
    power_w = _ops_float(snapshot.get("aps_battery_power_w"))
    soc = _ops_float(snapshot.get("aps_battery_soc_pct"))
    if power_w is None or soc is None:
        return "gray"
    deadband_w = _ops_float(snapshot.get("aps_battery_depletion_deadband_w")) or OPS_BATTERY_DEPLETION_DEADBAND_W
    if power_w >= -deadband_w:
        return "green"
    hours = _ops_float(snapshot.get("aps_battery_depletion_hours"))
    if hours is None:
        capacity_kwh = _ops_float(snapshot.get("aps_battery_capacity_kwh")) or OPS_BATTERY_CAPACITY_KWH
        remaining_kwh = max(soc, 0.0) / 100.0 * capacity_kwh
        discharge_kw = abs(power_w) / 1000.0
        hours = remaining_kwh / discharge_kw if discharge_kw > 0 else None
    if hours is None:
        return "gray"
    if hours >= 24.0:
        return "green"
    if hours >= 12.0:
        return "amber"
    return "red"


def _ops_level_from_internal_temp(value) -> str:
    temperature = _ops_float(value)
    if temperature is None:
        return "gray"
    if temperature < APS_INTERNAL_TEMP_LOW_RED_C or temperature >= APS_INTERNAL_TEMP_HIGH_RED_C:
        return "red"
    if temperature < APS_INTERNAL_TEMP_LOW_AMBER_C or temperature >= APS_INTERNAL_TEMP_HIGH_AMBER_C:
        return "amber"
    return "green"


def _ops_level_from_dewpoint_margin(snapshot: dict) -> str:
    if _ops_bool(snapshot.get("aps_internal_humidity_available_state")) is False:
        return "gray"
    margin = _ops_float(snapshot.get("aps_internal_dewpoint_margin_c"))
    if margin is None:
        return "gray"
    if margin <= APS_DEWPOINT_RED_MARGIN_C:
        return "red"
    return "green"


def _ops_level_from_perf_log(snapshot: dict) -> str:
    exists = _ops_bool(snapshot.get("dashboard_perf_log_exists_state"))
    age_min = _ops_float(snapshot.get("dashboard_perf_log_age_min"))
    if exists is False:
        return "red"
    if age_min is None:
        return "gray"
    if age_min <= 30.0:
        return "green"
    if age_min <= 360.0:
        return "amber"
    return "red"


def _ops_expected_paused_prefixes() -> set[str]:
    """Return streams whose absence is explained by a fresh PDU-off state."""
    try:
        outlet_states = mobile_catalog.pdu_outlet_states()
    except Exception as exc:
        logging.getLogger(__name__).warning("Could not read PDU states for Operations health: %s", exc)
        return set()
    if outlet_states is None:
        return set()
    return {
        prefix
        for prefix, outlet in OPS_STREAM_PDU_OUTLETS.items()
        if outlet_states.get(outlet) is False
    }


def _ops_source_health(snapshot: dict, paused_prefixes: set[str]) -> tuple[int, int, int]:
    """Return active recent, active stale, and intentionally paused stream counts."""
    active_specs = [spec for spec in OPS_STREAM_SPECS if spec["stream_prefix"] not in paused_prefixes]
    recent = sum(_ops_bool(snapshot.get(f"{spec['stream_prefix']}_source_recent_state")) is True for spec in active_specs)
    stale = sum(_ops_bool(snapshot.get(f"{spec['stream_prefix']}_source_recent_state")) is False for spec in active_specs)
    paused = sum(spec["stream_prefix"] in paused_prefixes for spec in OPS_STREAM_SPECS)
    return recent, stale, paused


def _ops_source_freshness_text(snapshot: dict, prefix: str, *, intentionally_paused: bool = False) -> str:
    recent = _ops_bool(snapshot.get(f"{prefix}_source_recent_state"))
    age_min = _ops_float(snapshot.get(f"{prefix}_source_age_min"))
    if intentionally_paused:
        if age_min is None:
            return "Paused - PDU outlet off"
        return f"Paused - PDU outlet off (last data {_format_duration(timedelta(minutes=age_min))} ago)"
    if recent is None:
        return "No source timestamp"
    if age_min is None:
        return "Recent" if recent else "Stale"
    age_text = _format_duration(timedelta(minutes=age_min))
    return f"Recent ({age_text})" if recent else f"Stale ({age_text})"


def _ops_battery_text(snapshot: dict) -> tuple[str, str]:
    voltage = _ops_float(snapshot.get("aps_battery_voltage_v"))
    age_min = _ops_float(snapshot.get("aps_battery_voltage_age_min"))
    if voltage is None:
        return "No data", "Aurora Power Supply DC inverter voltage unavailable"
    value = f"{voltage:.2f} V"
    if age_min is None:
        return value, "Aurora Power Supply DC inverter voltage"
    age_text = _format_duration(timedelta(minutes=age_min))
    return value, f"Aurora Power Supply DC inverter voltage, {age_text} old"


def _ops_battery_soc_text(snapshot: dict) -> tuple[str, str]:
    soc = _ops_float(snapshot.get("aps_battery_soc_pct"))
    age_min = _ops_float(snapshot.get("aps_battery_soc_age_min"))
    if soc is None:
        return "No data", "Aurora Power Supply battery state of charge unavailable"
    value = f"{soc:.0f} %"
    if age_min is None:
        return value, "Aurora Power Supply battery state of charge"
    age_text = _format_duration(timedelta(minutes=age_min))
    return value, f"Aurora Power Supply battery state of charge, {age_text} old"


def _ops_battery_depletion_text(snapshot: dict) -> tuple[str, str]:
    soc = _ops_float(snapshot.get("aps_battery_soc_pct"))
    power_w = _ops_float(snapshot.get("aps_battery_power_w"))
    if soc is None or power_w is None:
        return "No data", "Needs BatterySOC and BatteryWatts from the Aurora Power Supply"

    capacity_kwh = _ops_float(snapshot.get("aps_battery_capacity_kwh")) or OPS_BATTERY_CAPACITY_KWH
    deadband_w = _ops_float(snapshot.get("aps_battery_depletion_deadband_w")) or OPS_BATTERY_DEPLETION_DEADBAND_W
    remaining_kwh = _ops_float(snapshot.get("aps_battery_remaining_kwh"))
    if remaining_kwh is None:
        remaining_kwh = max(soc, 0.0) / 100.0 * capacity_kwh

    age_min = _ops_float(snapshot.get("aps_battery_power_age_min"))
    age_text = "" if age_min is None else f", power sample {_format_duration(timedelta(minutes=age_min))} old"
    energy_text = f"{remaining_kwh:.1f} kWh remaining from {capacity_kwh:.0f} kWh"

    if power_w < -deadband_w:
        hours = _ops_float(snapshot.get("aps_battery_depletion_hours"))
        if hours is None:
            discharge_kw = abs(power_w) / 1000.0
            hours = remaining_kwh / discharge_kw if discharge_kw > 0 else None
        value = "No data" if hours is None else _format_duration(timedelta(hours=hours))
        return value, f"{energy_text}; discharging at {abs(power_w):.0f} W{age_text}"
    if power_w > deadband_w:
        return "Charging", f"{energy_text}; charging at {power_w:.0f} W{age_text}"
    return "Flat", f"{energy_text}; battery power {power_w:.0f} W within +/-{deadband_w:.0f} W deadband{age_text}"


def _ops_internal_temp_text(snapshot: dict) -> tuple[str, str]:
    temperature = _ops_float(snapshot.get("aps_internal_temp_c"))
    age_min = _ops_float(snapshot.get("aps_internal_temp_age_min"))
    if temperature is None:
        return "No data", "Aurora Power Supply internal temperature unavailable"
    value = f"{temperature:.1f} C"
    if age_min is None:
        return value, "Aurora Power Supply internal temperature"
    age_text = _format_duration(timedelta(minutes=age_min))
    return value, f"Aurora Power Supply internal temperature, {age_text} old"


def _ops_dewpoint_text(snapshot: dict) -> tuple[str, str]:
    humidity_available = _ops_bool(snapshot.get("aps_internal_humidity_available_state"))
    if humidity_available is False:
        return "No data", "APS InternalHumidity is not available; dew point is not calculated"

    humidity = _ops_float(snapshot.get("aps_internal_humidity_pct"))
    dewpoint = _ops_float(snapshot.get("aps_internal_dewpoint_c"))
    margin = _ops_float(snapshot.get("aps_internal_dewpoint_margin_c"))
    temperature = _ops_float(snapshot.get("aps_internal_dewpoint_temp_c"))
    age_min = _ops_float(snapshot.get("aps_internal_humidity_age_min"))
    if humidity is None or dewpoint is None or margin is None:
        return "No data", "Needs same-sample InternalTemperature and InternalHumidity from APS"

    age_text = "" if age_min is None else f", {_format_duration(timedelta(minutes=age_min))} old"
    temp_text = "unknown" if temperature is None else f"{temperature:.1f} C"
    return (
        f"{margin:.1f} C margin",
        f"T={temp_text}, RH={humidity:.0f} %, dew point={dewpoint:.1f} C{age_text}",
    )


def _ops_perf_log_text(snapshot: dict) -> tuple[str, str]:
    exists = _ops_bool(snapshot.get("dashboard_perf_log_exists_state"))
    age_min = _ops_float(snapshot.get("dashboard_perf_log_age_min"))
    size_mb = _ops_float(snapshot.get("dashboard_perf_log_size_mb"))
    path = snapshot.get("dashboard_perf_log_path") or "/data/aurora/products/dashboard/dashboard_perf.jsonl"
    if exists is False:
        return "Missing", f"Expected {path}"
    size_text = "" if size_mb is None else f", {size_mb:.1f} MB"
    if age_min is None:
        return "Unknown", f"{path}{size_text}"
    age_text = _format_duration(timedelta(minutes=age_min))
    return f"{age_text} old", f"{path}{size_text}"


def _ops_batch_resource_level(snapshot: dict) -> str:
    return _ops_worst_level(
        [
            _ops_level_from_used_pct(snapshot.get("aurora_batch_memory_pressure_pct")),
            _ops_level_from_count(snapshot.get("aurora_guard_skip_count_24h"), amber_at=5.0),
            _ops_level_from_count(snapshot.get("aurora_guard_stale_lock_count"), amber_at=0.0),
        ]
    )


def _ops_batch_resource_text(snapshot: dict) -> tuple[str, str]:
    active_jobs = int(_ops_float(snapshot.get("aurora_batch_active_heavy_job_count")) or 0)
    active_locks = int(_ops_float(snapshot.get("aurora_guard_lock_active_count")) or 0)
    guard_skips = int(_ops_float(snapshot.get("aurora_guard_skip_count_24h")) or 0)
    memory_pct = _ops_float(snapshot.get("aurora_batch_memory_pressure_pct"))
    memory_current = _ops_float(snapshot.get("aurora_batch_memory_current_mb"))
    memory_high = _ops_float(snapshot.get("aurora_batch_memory_high_mb"))
    active_names = str(snapshot.get("aurora_batch_active_heavy_jobs") or snapshot.get("aurora_guard_lock_active_units") or "").strip()
    value = f"{active_jobs} active jobs" if active_jobs else "No active jobs"
    if memory_pct is None:
        memory_text = "memory pressure unknown"
    elif memory_current is not None and memory_high is not None:
        memory_text = f"memory {memory_current:.0f}/{memory_high:.0f} MB ({memory_pct:.0f} %)"
    else:
        memory_text = f"memory {memory_pct:.0f} %"
    lock_text = f"{active_locks} active locks, {guard_skips} skips in 24 h"
    if active_names:
        lock_text = f"{lock_text}; {active_names}"
    return value, f"{memory_text}; {lock_text}"


def _ops_failover_endpoint_card(snapshot: dict, endpoint: str, title: str, expected_role: str) -> str:
    prefix = f"failover_{endpoint}_dashboard"
    ok = _ops_bool(snapshot.get(f"{prefix}_http_ok_state"))
    full_document = _ops_bool(snapshot.get(f"{prefix}_http_full_document_state"))
    status = _ops_float(snapshot.get(f"{prefix}_http_status_code"))
    response_ms = _ops_float(snapshot.get(f"{prefix}_http_response_ms"))
    content_bytes = _ops_float(snapshot.get(f"{prefix}_http_content_bytes"))
    page_title = str(snapshot.get(f"{prefix}_http_title") or "").strip()
    error = str(snapshot.get(f"{prefix}_http_error") or "").strip()
    url = str(snapshot.get(f"{prefix}_url") or "").strip()

    if full_document is True:
        level = "green"
        value = "Full app document"
    elif ok is True:
        level = "red"
        value = f"HTTP {int(status)} but incomplete" if status is not None else "Incomplete app document"
    elif ok is False:
        level = "red"
        value = f"HTTP {int(status)}" if status is not None else "Unreachable"
    else:
        level = "gray"
        value = "No probe data"

    details: list[str] = [expected_role]
    if response_ms is not None:
        details.append(f"{response_ms:.0f} ms")
    if content_bytes is not None:
        details.append(f"{content_bytes / 1024.0:.0f} KB")
    if page_title:
        details.append(f"title={page_title}")
    if url:
        details.append(url)
    if error:
        details.append(error[:140])
    return _ops_card_markup(title, level, value, "; ".join(details))


def _ops_site_env_card(snapshot: dict) -> str:
    site_env = str(snapshot.get("site_env") or SITE_ENV or "unknown").strip().lower()
    domain = str(snapshot.get("failover_collector_domain") or SITE_DOMAIN or "").strip()
    role = str(snapshot.get("failover_collector_role") or "").strip()
    level = "green" if site_env in {"production", "development"} else "amber"
    value = site_env.title() if site_env else "Unknown"
    details = [item for item in (domain, f"role={role}" if role else "") if item]
    return _ops_card_markup("Site environment", level, value, "; ".join(details))


def _ops_dev_mirror_level(snapshot: dict) -> str:
    site_env = str(snapshot.get("site_env") or SITE_ENV or "").strip().lower()
    if site_env != "development":
        return "gray"
    recent = _ops_bool(snapshot.get("dev_live_mirror_recent_state"))
    if recent is not None:
        return "green" if recent else "red"
    age_min = _ops_float(snapshot.get("dev_live_mirror_age_min"))
    threshold = _ops_float(snapshot.get("dev_live_mirror_recent_threshold_min")) or 7.5
    if age_min is None:
        return "gray"
    if age_min <= threshold:
        return "green"
    if age_min <= threshold * 2:
        return "amber"
    return "red"


def _ops_dev_mirror_card(snapshot: dict) -> str:
    site_env = str(snapshot.get("site_env") or SITE_ENV or "").strip().lower()
    if site_env != "development":
        return _ops_card_markup("Mirror lag", "gray", "Not applicable", "Production owns the authoritative writers")
    age_min = _ops_float(snapshot.get("dev_live_mirror_age_min"))
    threshold = _ops_float(snapshot.get("dev_live_mirror_recent_threshold_min")) or 7.5
    last_success = str(snapshot.get("dev_live_mirror_last_success_utc") or "").strip()
    stamp = str(snapshot.get("dev_live_mirror_stamp_path") or "").strip()
    error = str(snapshot.get("dev_live_mirror_error") or "").strip()
    value = "Unknown" if age_min is None else _format_duration(timedelta(minutes=age_min))
    details = [f"threshold {threshold:.1f} min"]
    if last_success:
        details.append(f"last success {last_success}")
    if stamp:
        details.append(stamp)
    if error:
        details.append(error[:120])
    return _ops_card_markup("Mirror lag", _ops_dev_mirror_level(snapshot), value, "; ".join(details))


def _ops_code_state_card(snapshot: dict, prefix: str, title: str) -> str:
    exists = _ops_bool(snapshot.get(f"{prefix}_repo_exists_state"))
    dirty_count = int(_ops_float(snapshot.get(f"{prefix}_git_dirty_count")) or 0)
    behind_count = int(_ops_float(snapshot.get(f"{prefix}_git_behind_count")) or 0)
    ahead_count = int(_ops_float(snapshot.get(f"{prefix}_git_ahead_count")) or 0)
    if exists is False:
        level = "red"
        value = "Missing"
    elif exists is None:
        level = "gray"
        value = "No probe data"
    else:
        level = _ops_worst_level(
            [
                "green",
                _ops_level_from_count(dirty_count, amber_at=10.0),
                "amber" if behind_count > 0 else "green",
                "amber" if ahead_count > 0 else "green",
            ]
        )
        value = str(snapshot.get(f"{prefix}_git_describe") or snapshot.get(f"{prefix}_git_commit") or "Unknown")
    branch = str(snapshot.get(f"{prefix}_git_branch") or "").strip()
    commit = str(snapshot.get(f"{prefix}_git_commit") or "").strip()
    tag = str(snapshot.get(f"{prefix}_git_tag") or "").strip()
    details = [
        f"branch={branch}" if branch else "",
        f"commit={commit}" if commit else "",
        f"tag={tag}" if tag else "tag=none",
        f"dirty={dirty_count}",
        f"behind={behind_count}",
        f"ahead={ahead_count}",
    ]
    return _ops_card_markup(title, level, value, "; ".join(item for item in details if item))


def _ops_perf_summary(path: Path, hours: float = 24.0, max_rows: int = 5000) -> dict:
    if not path.exists():
        return {"level": "red", "value": "Missing", "meta": f"Expected {path}"}
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    rows: list[dict] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            tail = deque(handle, maxlen=max_rows)
        for line in tail:
            try:
                row = json.loads(line)
            except Exception:
                continue
            ts = _ops_timestamp(row.get("ts_utc"))
            if ts is None or ts < cutoff:
                continue
            rows.append(row)
    except Exception as exc:
        return {"level": "red", "value": "Unreadable", "meta": str(exc)}

    durations = sorted(float(row["duration_ms"]) for row in rows if isinstance(row.get("duration_ms"), (int, float)))
    if not durations:
        return {"level": "gray", "value": "No timing samples", "meta": f"No timed events in the last {hours:g} h"}

    def quantile(q: float) -> float:
        if len(durations) == 1:
            return durations[0]
        position = (len(durations) - 1) * q
        low = int(np.floor(position))
        high = int(np.ceil(position))
        if low == high:
            return durations[low]
        return durations[low] * (high - position) + durations[high] * (position - low)

    p50 = quantile(0.50)
    p95 = quantile(0.95)
    max_duration = durations[-1]
    max_live = max((int(row["live_sessions"]) for row in rows if row.get("live_sessions") is not None), default=0)
    slowest = max((row for row in rows if isinstance(row.get("duration_ms"), (int, float))), key=lambda row: float(row["duration_ms"]))
    slow_label = str(slowest.get("event") or "event")
    slow_instrument = str(slowest.get("instrument") or "").strip()
    if slow_instrument:
        slow_label = f"{slow_label} / {slow_instrument}"

    if p95 <= 1000.0:
        level = "green"
    elif p95 <= 3000.0:
        level = "amber"
    else:
        level = "red"
    return {
        "level": level,
        "value": f"p95 {p95 / 1000.0:.1f}s",
        "meta": (
            f"{len(durations)} timed events in {hours:g} h; "
            f"p50 {p50 / 1000.0:.1f}s; max {max_duration / 1000.0:.1f}s ({slow_label}); "
            f"max live sessions {max_live}"
        ),
    }


def _ops_storage_text(snapshot: dict, key_prefix: str) -> str:
    used = _ops_float(snapshot.get(f"{key_prefix}_used_gb"))
    total = _ops_float(snapshot.get(f"{key_prefix}_total_gb"))
    used_pct = _ops_float(snapshot.get(f"{key_prefix}_used_pct"))
    if used is None or total is None:
        return "No data"
    if used_pct is None:
        return f"{used:.1f} / {total:.1f} GB"
    return f"{used:.1f} / {total:.1f} GB ({used_pct:.0f}%)"


def _ops_storage_location(snapshot: dict, key_prefix: str, host_label: str, fallback_path: str) -> str:
    resolved = snapshot.get(f"{key_prefix}_resolved_path")
    if resolved:
        return f"{host_label} {resolved}"
    return f"{host_label} {fallback_path}"


def _ops_manifest_ready(snapshot: dict) -> bool:
    for spec in OPS_STREAM_SPECS:
        prefix = spec["stream_prefix"]
        if _ops_float(snapshot.get(f"{prefix}_source_count")) not in (None, 0.0):
            return True
        if _ops_float(snapshot.get(f"{prefix}_local_count")) not in (None, 0.0):
            return True
        if _ops_float(snapshot.get(f"{prefix}_gws_count")) not in (None, 0.0):
            return True
        if _ops_float(snapshot.get(f"{prefix}_local_coverage_pct")) is not None:
            return True
        if _ops_float(snapshot.get(f"{prefix}_gws_coverage_pct")) is not None:
            return True
    return False


def _ops_archive_level(snapshot: dict, prefix: str) -> str:
    if _ops_bool(snapshot.get(f"{prefix}_backfill_pending_state")):
        return "amber"
    gws_missing = int(_ops_float(snapshot.get(f"{prefix}_gws_missing_count")) or 0)
    gws_mismatch = int(_ops_float(snapshot.get(f"{prefix}_gws_mismatch_count")) or 0)
    gws_coverage = _ops_float(snapshot.get(f"{prefix}_gws_coverage_pct"))
    if gws_coverage is None:
        if _ops_bool(snapshot.get("gws_probe_ok_state")):
            return "amber"
        return "gray"
    if gws_missing == 0 and gws_mismatch == 0:
        return "green"
    if gws_coverage >= 99.9:
        return "green"
    if gws_coverage >= 95.0:
        return "amber"
    return "red"


def _ops_archive_text(snapshot: dict, prefix: str) -> str:
    if _ops_bool(snapshot.get(f"{prefix}_backfill_pending_state")):
        local_coverage = _ops_float(snapshot.get(f"{prefix}_local_coverage_pct"))
        gws_coverage = _ops_float(snapshot.get(f"{prefix}_gws_coverage_pct"))
        local_text = "?" if local_coverage is None else f"{local_coverage:.0f}%"
        gws_text = "?" if gws_coverage is None else f"{gws_coverage:.0f}%"
        return f"Backfill {local_text} local / {gws_text} GWS"
    gws_missing = int(_ops_float(snapshot.get(f"{prefix}_gws_missing_count")) or 0)
    gws_mismatch = int(_ops_float(snapshot.get(f"{prefix}_gws_mismatch_count")) or 0)
    gws_coverage = _ops_float(snapshot.get(f"{prefix}_gws_coverage_pct"))
    if gws_coverage is None:
        return "Pending manifest sync"
    if gws_missing == 0 and gws_mismatch == 0 and gws_coverage < 99.9:
        return f"{gws_coverage:.0f}% mirrored, settled OK"
    return f"{gws_coverage:.0f}% mirrored"


def _ops_prune_level(snapshot: dict, prefix: str, manifest_ready: bool) -> str:
    if _ops_bool(snapshot.get(f"{prefix}_backfill_pending_state")):
        return "amber"
    if not manifest_ready:
        return "amber"
    prune_ready = _ops_bool(snapshot.get(f"{prefix}_prune_ready_state"))
    if prune_ready is None:
        return "gray"
    return "green" if prune_ready else "red"


def _ops_prune_text(snapshot: dict, prefix: str, manifest_ready: bool) -> str:
    if _ops_bool(snapshot.get(f"{prefix}_backfill_pending_state")):
        return "Backfill"
    if not manifest_ready:
        return "Pending verification"
    prune_ready = _ops_bool(snapshot.get(f"{prefix}_prune_ready_state"))
    if prune_ready is None:
        return "Unknown"
    return "Ready" if prune_ready else "Hold"


def _ops_failed_service_names(snapshot: dict) -> list[str]:
    names: list[str] = []
    for key, value in sorted(snapshot.items()):
        if not key.endswith("_service_healthy_state"):
            continue
        if _ops_bool(value) is not False:
            continue
        label = key.removesuffix("_service_healthy_state").replace("_", " ")
        names.append(label)
    return names


def _ops_light_markup(level: str, text: str) -> str:
    return (
        f"<span class='ops-light ops-light--{escape(level)}' aria-hidden='true'></span>"
        f"<span class='ops-light-text'>{escape(text)}</span>"
    )


def _ops_card_markup(title: str, level: str, value: str, meta: str = "") -> str:
    meta_markup = f"<div class='ops-card__meta'>{escape(meta)}</div>" if meta else ""
    return (
        "<div class='ops-card'>"
        f"<div class='ops-card__head'>{_ops_light_markup(level, title)}</div>"
        f"<div class='ops-card__value'>{escape(value)}</div>"
        f"{meta_markup}"
        "</div>"
    )


def _ops_table_cell(level: str, label: str, detail: str = "") -> str:
    detail_markup = f"<div class='ops-table__detail'>{escape(detail)}</div>" if detail else ""
    return (
        "<td class='ops-table__cell'>"
        f"<div class='ops-table__state'>{_ops_light_markup(level, label)}</div>"
        f"{detail_markup}"
        "</td>"
    )


def _ops_value_from_series(values: np.ndarray, mode: str = "latest") -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    if mode == "max":
        return float(np.nanmax(finite))
    if mode == "min":
        return float(np.nanmin(finite))
    return float(finite[-1])


def _ops_combined_series(ds: xr.Dataset, names: tuple[str, ...], mode: str = "max") -> np.ndarray | None:
    arrays = []
    for name in names:
        if name not in ds:
            continue
        arrays.append(np.asarray(ds[name].values, dtype=np.float64))
    if not arrays:
        return None
    stack = np.vstack(arrays)
    finite = np.isfinite(stack)
    valid_columns = np.any(finite, axis=0)
    combined = np.full(stack.shape[1], np.nan, dtype=np.float64)
    if not np.any(valid_columns):
        return combined
    selected = stack[:, valid_columns]
    if mode == "min":
        combined[valid_columns] = np.min(np.where(np.isfinite(selected), selected, np.inf), axis=0)
    else:
        combined[valid_columns] = np.max(np.where(np.isfinite(selected), selected, -np.inf), axis=0)
    return combined


def _ops_sparkline_svg(values: np.ndarray | None, level: str, width: int = 150, height: int = 34) -> str:
    if values is None:
        return "<div class='ops-sparkline ops-sparkline--empty'>No trend</div>"
    arr = np.asarray(values, dtype=np.float64)
    finite_mask = np.isfinite(arr)
    if np.count_nonzero(finite_mask) < 2:
        return "<div class='ops-sparkline ops-sparkline--empty'>No trend</div>"
    indices = np.flatnonzero(finite_mask)
    if indices.size > 60:
        keep = np.unique(np.linspace(0, indices.size - 1, 60, dtype=int))
        indices = indices[keep]
    y_values = arr[indices]
    x_values = np.linspace(0, width, len(indices))
    y_min = float(np.nanmin(y_values))
    y_max = float(np.nanmax(y_values))
    if not np.isfinite(y_min) or not np.isfinite(y_max):
        return "<div class='ops-sparkline ops-sparkline--empty'>No trend</div>"
    if y_max == y_min:
        y_min -= 1.0
        y_max += 1.0
    y_scaled = height - 4.0 - ((y_values - y_min) / (y_max - y_min)) * (height - 8.0)
    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(x_values, y_scaled, strict=False))
    color = {
        "green": "#2a9d8f",
        "amber": "#b7791f",
        "red": "#c05647",
        "gray": "#718195",
    }.get(level, "#0b7285")
    return (
        f"<svg class='ops-sparkline' viewBox='0 0 {width} {height}' role='img' aria-label='7 day trend'>"
        f"<polyline points='{points}' fill='none' stroke='{color}' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'></polyline>"
        "</svg>"
    )


def _ops_trend_card_markup(title: str, level: str, value: str, meta: str, values: np.ndarray | None) -> str:
    return (
        "<div class='ops-card ops-card--trend'>"
        f"<div class='ops-card__head'>{_ops_light_markup(level, title)}</div>"
        f"<div class='ops-card__value'>{escape(value)}</div>"
        f"{_ops_sparkline_svg(values, level)}"
        f"<div class='ops-card__meta'>{escape(meta)}</div>"
        "</div>"
    )


def _ops_trend_cards_markup(paused_prefixes: set[str] | None = None) -> str:
    """Return compact seven-day trend cards from the operations Zarr."""
    paused_prefixes = paused_prefixes or set()
    paused_key = tuple(sorted(paused_prefixes))
    now = datetime.now(timezone.utc)
    cached_at = _OPS_TREND_CACHE.get("updated_at")
    cached_markup = str(_OPS_TREND_CACHE.get("markup") or "")
    if (
        isinstance(cached_at, datetime)
        and now - cached_at < OPS_TREND_CACHE_TTL
        and cached_markup
        and _OPS_TREND_CACHE.get("paused_prefixes") == paused_key
    ):
        return cached_markup

    path = Path(_zarr_path("ops-monitor"))
    if not path.exists():
        markup = _ops_card_markup("Trends", "gray", "No data", f"Expected {path}")
        _OPS_TREND_CACHE.update({"updated_at": now, "markup": markup})
        return markup
    try:
        ds = xr.open_zarr(path, chunks={}, consolidated=bool(_cfg("ops-monitor").get("consolidated", True)))
        if "time" not in ds or ds.sizes.get("time", 0) == 0:
            raise ValueError("operations Zarr has no time samples")
        times = pd.DatetimeIndex(pd.to_datetime(ds["time"].values))
        cutoff = pd.Timestamp(now - OPS_TREND_WINDOW).tz_localize(None)
        mask = times >= cutoff
        if np.any(mask):
            ds = ds.isel(time=np.flatnonzero(mask))
    except Exception as exc:
        markup = _ops_card_markup("Trends", "gray", "Unavailable", str(exc))
        _OPS_TREND_CACHE.update({"updated_at": now, "markup": markup})
        return markup

    disk_series = _ops_combined_series(
        ds,
        (
            "host_celine_source_used_pct",
            "host_celine_data_used_pct",
            "host_ass_data_used_pct",
            "host_ass_root_used_pct",
            "host_aps_data_used_pct",
            "host_aps_root_used_pct",
            "aurora_data_used_pct",
            "aurora_root_used_pct",
            "gws_storage_used_pct",
        ),
        mode="max",
    )
    soc_series = np.asarray(ds["aps_battery_soc_pct"].values, dtype=np.float64) if "aps_battery_soc_pct" in ds else None
    voltage_series = np.asarray(ds["aps_battery_voltage_v"].values, dtype=np.float64) if "aps_battery_voltage_v" in ds else None
    source_lag_series = _ops_combined_series(
        ds,
        tuple(
            f"{spec['stream_prefix']}_source_age_min"
            for spec in OPS_STREAM_SPECS
            if spec["stream_prefix"] not in paused_prefixes
        ),
        mode="max",
    )
    gws_lag_series = _ops_combined_series(
        ds,
        tuple(f"{spec['stream_prefix']}_gws_lag_min" for spec in OPS_STREAM_SPECS),
        mode="max",
    )

    disk_latest = _ops_value_from_series(disk_series) if disk_series is not None else None
    soc_latest = _ops_value_from_series(soc_series) if soc_series is not None else None
    voltage_latest = _ops_value_from_series(voltage_series) if voltage_series is not None else None
    source_lag_latest = _ops_value_from_series(source_lag_series) if source_lag_series is not None else None
    gws_lag_latest = _ops_value_from_series(gws_lag_series) if gws_lag_series is not None else None

    cards = [
        _ops_trend_card_markup(
            "Disk pressure",
            _ops_level_from_used_pct(disk_latest),
            "No data" if disk_latest is None else f"{disk_latest:.0f} %",
            "Worst current storage use across source hosts, AURORA Cloud, and GWS",
            disk_series,
        ),
        _ops_trend_card_markup(
            "Battery SOC",
            _ops_level_from_battery_soc(soc_latest),
            "No data" if soc_latest is None else f"{soc_latest:.0f} %",
            "Aurora Power Supply state of charge",
            soc_series,
        ),
        _ops_trend_card_markup(
            "Battery voltage",
            _ops_level_from_battery_voltage(voltage_latest),
            "No data" if voltage_latest is None else f"{voltage_latest:.2f} V",
            "Aurora Power Supply DC inverter voltage",
            voltage_series,
        ),
        _ops_trend_card_markup(
            "Source lag",
            _ops_level_from_age_minutes(source_lag_latest),
            "All paused" if paused_prefixes and source_lag_latest is None else ("No data" if source_lag_latest is None else _format_duration(timedelta(minutes=source_lag_latest))),
            "Worst active stream source age" + (f"; {len(paused_prefixes)} PDU stream(s) paused" if paused_prefixes else ""),
            source_lag_series,
        ),
        _ops_trend_card_markup(
            "GWS lag",
            _ops_level_from_age_minutes(gws_lag_latest),
            "No data" if gws_lag_latest is None else _format_duration(timedelta(minutes=gws_lag_latest)),
            "Worst raw mirror lag on JASMIN GWS",
            gws_lag_series,
        ),
    ]
    markup = "".join(cards)
    _OPS_TREND_CACHE.update({"updated_at": now, "markup": markup, "paused_prefixes": paused_key})
    return markup


def _ops_root_cause_cards_markup(
    snapshot: dict,
    perf_summary: dict,
    manifest_ready: bool,
    source_level: str,
    source_freshness_level: str,
    processing_level: str,
    transfer_level: str,
    mirror_level: str,
    perf_log_level: str,
    source_stale: int,
    source_paused: int,
) -> str:
    """Group the traffic lights by where a user should look first."""
    source_probe_failures = int(_ops_float(snapshot.get("source_host_probe_fail_count")) or 0)
    source_sync_failures = int(_ops_float(snapshot.get("failed_source_sync_unit_count")) or 0)
    processing_failures = int(_ops_float(snapshot.get("failed_processing_unit_count")) or 0)
    transfer_failures = int(_ops_float(snapshot.get("failed_transfer_unit_count")) or 0)
    gws_issues = int(_ops_float(snapshot.get("streams_gws_issue_count")) or 0)
    local_issues = int(_ops_float(snapshot.get("streams_local_issue_count")) or 0)
    batch_value, batch_meta = _ops_batch_resource_text(snapshot)

    cards = [
        _ops_card_markup(
            "Source computers",
            _ops_worst_level([source_level, source_freshness_level]),
            f"{source_probe_failures} host failures, {source_stale} active stale streams"
            + (f", {source_paused} paused" if source_paused else ""),
            "Remote source reachability and whether each stream has produced data within 1.5 hours",
        ),
        _ops_card_markup(
            "Network and source sync",
            _ops_level_from_count(source_sync_failures, amber_at=1.0),
            f"{source_sync_failures} sync failures",
            "Systemd source-mirror jobs from source hosts into /project/aurora",
        ),
        _ops_card_markup(
            "Local processing",
            _ops_worst_level([processing_level, _ops_level_from_count(local_issues, amber_at=1.0)]),
            f"{processing_failures} service failures, {local_issues} local mirror issues",
            "Append, quicklook, catalog, video, and local manifest/product gates",
        ),
        _ops_card_markup(
            "GWS transfer",
            _ops_worst_level([transfer_level, mirror_level]),
            f"{transfer_failures} transfer failures, {gws_issues} GWS issues",
            "JASMIN reachability, raw/product rsync jobs, and GWS manifest agreement",
        ),
        _ops_card_markup(
            "Dashboard and render",
            _ops_worst_level([perf_log_level, str(perf_summary.get("level", "gray"))]),
            str(perf_summary.get("value", "No samples")),
            "Diagnostic only; this does not drive Overall action state",
        ),
        _ops_card_markup(
            "Background load",
            _ops_batch_resource_level(snapshot),
            batch_value,
            f"{batch_meta}; diagnostic only, not part of Overall",
        ),
    ]
    if not manifest_ready:
        cards.append(_ops_card_markup("Manifest seed", "amber", "Pending", "Archive/prune checks are still waiting for manifest history"))
    return "".join(cards)


def _ops_operations_markup() -> str:
    snapshot = _ops_read_snapshot()
    with _timed_perf("operations_dashboard_render", instrument="ops-monitor", snapshot_path=snapshot.get("_path")) as perf:
        if snapshot.get("_missing"):
            perf["status"] = "missing_snapshot"
            missing_meta = f"Expected {snapshot.get('_path', 'unknown path')}"
            return (
                "<div class='ops-shell'>"
                "<div class='ops-section-title'>Operations Dashboard</div>"
                f"{_ops_card_markup('Operations snapshot', 'red', 'Missing', missing_meta)}"
                "</div>"
            )
        if snapshot.get("_error"):
            perf["status"] = "snapshot_error"
            return (
                "<div class='ops-shell'>"
                "<div class='ops-section-title'>Operations Dashboard</div>"
                f"{_ops_card_markup('Operations snapshot', 'red', 'Unreadable', snapshot.get('_error', 'Unknown error'))}"
                "</div>"
            )

        updated_at = _ops_timestamp(snapshot.get("time_utc"))
        snapshot_age_min = None
        if updated_at is not None:
            snapshot_age_min = max((datetime.now(timezone.utc) - updated_at).total_seconds() / 60.0, 0.0)
        manifest_ready = _ops_manifest_ready(snapshot)
        failed_services = _ops_failed_service_names(snapshot)
        target_stream_count = int(_ops_float(snapshot.get("streams_target_count")) or len(OPS_STREAM_SPECS))
        backfill_pending_count = int(_ops_float(snapshot.get("streams_backfill_pending_count")) or 0)

        snapshot_level = _ops_level_from_age_minutes(snapshot_age_min)
        source_level = _ops_level_from_source_probes(snapshot.get("source_host_probe_fail_count"))
        paused_prefixes = _ops_expected_paused_prefixes()
        source_recent_count, source_stale_count, source_paused_count = _ops_source_health(snapshot, paused_prefixes)
        source_freshness_level = _ops_level_from_count(source_stale_count, amber_at=0.0)
        battery_level = _ops_level_from_battery_voltage(snapshot.get("aps_battery_voltage_v"))
        battery_soc_level = _ops_level_from_battery_soc(snapshot.get("aps_battery_soc_pct"))
        battery_depletion_level = _ops_level_from_battery_depletion(snapshot)
        internal_temp_level = _ops_level_from_internal_temp(snapshot.get("aps_internal_temp_c"))
        dewpoint_level = _ops_level_from_dewpoint_margin(snapshot)
        perf_log_level = _ops_level_from_perf_log(snapshot)
        processing_level = _ops_level_from_count(snapshot.get("failed_processing_unit_count"), amber_at=1.0)
        site_env = str(snapshot.get("site_env") or SITE_ENV or "").strip().lower()
        dev_mirror_level = _ops_dev_mirror_level(snapshot)
        transfer_level = _ops_worst_level(
            [
                _ops_level_from_count(snapshot.get("failed_transfer_unit_count"), amber_at=1.0),
                _ops_level_from_bool(snapshot.get("gws_probe_ok_state")),
            ]
        )
        if not manifest_ready:
            mirror_level = "amber"
        else:
            mirror_level = _ops_worst_level(
                [
                    _ops_level_from_bool(snapshot.get("mirror_verify_service_healthy_state")),
                    _ops_level_from_count(snapshot.get("streams_local_issue_count"), amber_at=1.0),
                    _ops_level_from_count(snapshot.get("streams_gws_issue_count"), amber_at=1.0),
                ]
            )
            if mirror_level == "green" and backfill_pending_count > 0:
                mirror_level = "amber"
        # Render/performance telemetry is diagnostic: it should stay visible,
        # but it should not turn the overall operations state into "Action
        # needed" when the data/transfer/power systems are otherwise healthy.
        overall_level = _ops_worst_level([
            snapshot_level,
            source_level,
            source_freshness_level,
            battery_level,
            battery_soc_level,
            battery_depletion_level,
            internal_temp_level,
            dewpoint_level,
            processing_level,
            transfer_level,
            mirror_level,
        ])
        if site_env == "development":
            overall_level = _ops_worst_level([overall_level, dev_mirror_level])

        overall_value = "Healthy"
        if overall_level == "amber":
            overall_value = "Attention needed"
        elif overall_level == "red":
            overall_value = "Action needed"
        elif overall_level == "gray":
            overall_value = "Waiting for data"

        updated_label = updated_at.strftime("%Y-%m-%d %H:%M UTC") if updated_at else "Unknown"
        age_label = f"{snapshot_age_min:.0f} min old" if snapshot_age_min is not None else "Age unknown"
        battery_value, battery_meta = _ops_battery_text(snapshot)
        battery_soc_value, battery_soc_meta = _ops_battery_soc_text(snapshot)
        battery_depletion_value, battery_depletion_meta = _ops_battery_depletion_text(snapshot)
        internal_temp_value, internal_temp_meta = _ops_internal_temp_text(snapshot)
        dewpoint_value, dewpoint_meta = _ops_dewpoint_text(snapshot)
        perf_log_value, perf_log_meta = _ops_perf_log_text(snapshot)
        batch_resource_value, batch_resource_meta = _ops_batch_resource_text(snapshot)
        batch_resource_level = _ops_batch_resource_level(snapshot)
        perf_summary = _ops_perf_summary(Path(snapshot.get("dashboard_perf_log_path") or PERF_LOG_PATH))
        root_cause_cards = _ops_root_cause_cards_markup(
            snapshot,
            perf_summary,
            manifest_ready,
            source_level,
            source_freshness_level,
            processing_level,
            transfer_level,
            mirror_level,
            perf_log_level,
            source_stale_count,
            source_paused_count,
        )
        trend_cards = _ops_trend_cards_markup(paused_prefixes)
        failover_endpoint_levels = [
            _ops_worst_level(
                [
                    _ops_level_from_bool(snapshot.get(f"failover_{endpoint}_dashboard_http_ok_state")),
                    _ops_level_from_bool(snapshot.get(f"failover_{endpoint}_dashboard_http_full_document_state")),
                ]
            )
            for endpoint in ("primary", "standby")
        ]
        failover_level = _ops_worst_level(failover_endpoint_levels)
        failover_cards = [
            _ops_failover_endpoint_card(snapshot, "primary", "Production endpoint", "data.gamb2le.co.uk"),
            _ops_failover_endpoint_card(snapshot, "standby", "Development endpoint", "data-ocean.gamb2le.co.uk"),
        ]
        deployment_cards = [
            _ops_site_env_card(snapshot),
            _ops_dev_mirror_card(snapshot),
            _ops_code_state_card(snapshot, "dashboard_code", "Dashboard code"),
            _ops_code_state_card(snapshot, "infra_code", "Infrastructure code"),
        ]

        summary_cards = [
            _ops_card_markup(
                "Overall",
                overall_level,
                overall_value,
                f"Snapshot {age_label}; {len(failed_services)} unhealthy services",
            ),
            _ops_card_markup(
                "Snapshot freshness",
                snapshot_level,
                age_label,
                f"Last operations sample {updated_label}",
            ),
            _ops_card_markup(
                "Source hosts",
                source_level,
                f"{max(0, 3 - int(_ops_float(snapshot.get('source_host_probe_fail_count')) or 0))}/3 reachable",
                f"{int(_ops_float(snapshot.get('source_host_probe_fail_count')) or 0)} probe failures",
            ),
            _ops_card_markup(
                "Source freshness",
                source_freshness_level,
                (
                    f"{source_stale_count} active stale streams"
                    if source_stale_count
                    else f"{source_recent_count}/{len(OPS_STREAM_SPECS) - source_paused_count} active streams recent"
                ),
                "Source data seen within the last 1.5 hours"
                + (f"; {source_paused_count} PDU stream(s) intentionally paused" if source_paused_count else ""),
            ),
            _ops_card_markup(
                "Battery voltage",
                battery_level,
                battery_value,
                f"{battery_meta}; green >52 V, amber 50-52 V, red <50 V",
            ),
            _ops_card_markup(
                "Battery SOC",
                battery_soc_level,
                battery_soc_value,
                f"{battery_soc_meta}; green >=50 %, amber >40-50 %, red <=40 %",
            ),
            _ops_card_markup(
                "Battery depletion",
                battery_depletion_level,
                battery_depletion_value,
                f"{battery_depletion_meta}; green >=24 h or not depleting, amber 12-24 h, red <12 h",
            ),
            _ops_card_markup(
                "APS internal temp",
                internal_temp_level,
                internal_temp_value,
                (
                    f"{internal_temp_meta}; green {APS_INTERNAL_TEMP_LOW_AMBER_C:.0f}-{APS_INTERNAL_TEMP_HIGH_AMBER_C:.0f} C, "
                    f"amber {APS_INTERNAL_TEMP_LOW_RED_C:.0f}-{APS_INTERNAL_TEMP_LOW_AMBER_C:.0f} C or "
                    f"{APS_INTERNAL_TEMP_HIGH_AMBER_C:.0f}-{APS_INTERNAL_TEMP_HIGH_RED_C:.0f} C, "
                    f"red <{APS_INTERNAL_TEMP_LOW_RED_C:.0f} C or >={APS_INTERNAL_TEMP_HIGH_RED_C:.0f} C"
                ),
            ),
            _ops_card_markup(
                "APS dew point",
                dewpoint_level,
                dewpoint_value,
                f"{dewpoint_meta}; red when temperature-dewpoint margin <= {APS_DEWPOINT_RED_MARGIN_C:.0f} C",
            ),
            _ops_card_markup(
                "Dashboard perf log",
                perf_log_level,
                perf_log_value,
                f"{perf_log_meta}; diagnostic only, not part of Overall",
            ),
            _ops_card_markup(
                "Render performance",
                perf_summary["level"],
                perf_summary["value"],
                f"{perf_summary['meta']}; diagnostic only, not part of Overall",
            ),
            _ops_card_markup(
                "Batch resources",
                batch_resource_level,
                batch_resource_value,
                f"{batch_resource_meta}; diagnostic only, not part of Overall",
            ),
            _ops_card_markup(
                "Failover endpoints",
                failover_level,
                "Both apps full"
                if failover_level == "green"
                else ("No probe data" if failover_level == "gray" else "Endpoint issue"),
                "Primary and standby public app probes; diagnostic only, not part of Overall",
            ),
            _ops_card_markup(
                "Processing pipeline",
                processing_level,
                f"{int(_ops_float(snapshot.get('failed_processing_unit_count')) or 0)} failed services",
                ", ".join(failed_services[:3]) if failed_services else "Append and quicklook services healthy",
            ),
            _ops_card_markup(
                "Transfers and GWS",
                transfer_level,
                "Reachable" if _ops_bool(snapshot.get("gws_probe_ok_state")) else "Unreachable",
                f"{int(_ops_float(snapshot.get('failed_transfer_unit_count')) or 0)} transfer failures",
            ),
            _ops_card_markup(
                "Mirror verification",
                mirror_level,
                "Pending manifest seed" if not manifest_ready else "Active",
                (
                    "Waiting for raw/GWS manifests to populate"
                    if not manifest_ready
                    else (
                        f"Local issues: {int(_ops_float(snapshot.get('streams_local_issue_count')) or 0)}, "
                        f"GWS issues: {int(_ops_float(snapshot.get('streams_gws_issue_count')) or 0)}, "
                        f"Backfills: {backfill_pending_count}"
                    )
                ),
            ),
        ]

        storage_cards = [
            _ops_card_markup(
                "CL61 root disk",
                _ops_worst_level(
                    [
                        _ops_level_from_bool(snapshot.get("host_celine_source_probe_ok_state")),
                        _ops_level_from_used_pct(snapshot.get("host_celine_source_used_pct")),
                    ]
                ),
                _ops_storage_text(snapshot, "host_celine_source"),
                _ops_storage_location(snapshot, "host_celine_source", "100.117.101.84", "/"),
            ),
            _ops_card_markup(
                "CL61 data disk",
                _ops_worst_level(
                    [
                        _ops_level_from_bool(snapshot.get("host_celine_data_probe_ok_state")),
                        _ops_level_from_used_pct(snapshot.get("host_celine_data_used_pct")),
                    ]
                ),
                _ops_storage_text(snapshot, "host_celine_data"),
                _ops_storage_location(snapshot, "host_celine_data", "100.117.101.84", "/home/aurora/data"),
            ),
            _ops_card_markup(
                "ASS data disk",
                _ops_worst_level(
                    [
                        _ops_level_from_bool(snapshot.get("host_ass_data_probe_ok_state")),
                        _ops_level_from_used_pct(snapshot.get("host_ass_data_used_pct")),
                    ]
                ),
                _ops_storage_text(snapshot, "host_ass_data"),
                _ops_storage_location(snapshot, "host_ass_data", "100.124.55.22", "/home/aurora/data"),
            ),
            _ops_card_markup(
                "ASS root disk",
                _ops_worst_level(
                    [
                        _ops_level_from_bool(snapshot.get("host_ass_root_probe_ok_state")),
                        _ops_level_from_used_pct(snapshot.get("host_ass_root_used_pct")),
                    ]
                ),
                _ops_storage_text(snapshot, "host_ass_root"),
                _ops_storage_location(snapshot, "host_ass_root", "100.124.55.22", "/"),
            ),
            _ops_card_markup(
                "APS data disk",
                _ops_worst_level(
                    [
                        _ops_level_from_bool(snapshot.get("host_aps_data_probe_ok_state")),
                        _ops_level_from_used_pct(snapshot.get("host_aps_data_used_pct")),
                    ]
                ),
                _ops_storage_text(snapshot, "host_aps_data"),
                _ops_storage_location(snapshot, "host_aps_data", "100.81.226.30", "/data"),
            ),
            _ops_card_markup(
                "APS root disk",
                _ops_worst_level(
                    [
                        _ops_level_from_bool(snapshot.get("host_aps_root_probe_ok_state")),
                        _ops_level_from_used_pct(snapshot.get("host_aps_root_used_pct")),
                    ]
                ),
                _ops_storage_text(snapshot, "host_aps_root"),
                _ops_storage_location(snapshot, "host_aps_root", "100.81.226.30", "/"),
            ),
            _ops_card_markup(
                "AURORA Cloud product disk",
                _ops_level_from_used_pct(snapshot.get("aurora_data_used_pct")),
                _ops_storage_text(snapshot, "aurora_data"),
                _ops_storage_location(snapshot, "aurora_data", "AURORA Cloud", "/data/aurora"),
            ),
            _ops_card_markup(
                "AURORA Cloud root disk",
                _ops_level_from_used_pct(snapshot.get("aurora_root_used_pct")),
                _ops_storage_text(snapshot, "aurora_root"),
                _ops_storage_location(snapshot, "aurora_root", "AURORA Cloud", "/"),
            ),
            _ops_card_markup(
                "JASMIN GWS",
                _ops_worst_level(
                    [
                        _ops_level_from_bool(snapshot.get("gws_probe_ok_state")),
                        _ops_level_from_used_pct(snapshot.get("gws_storage_used_pct")),
                    ]
                ),
                _ops_storage_text(snapshot, "gws_storage"),
                "/gws/ssde/j25b/gamb2le",
            ),
        ]

        manifest_cards = [
            _ops_card_markup(
                "Local raw mirror",
                (
                    "amber"
                    if not manifest_ready
                    else (
                        "amber"
                        if backfill_pending_count > 0 and int(_ops_float(snapshot.get("streams_local_issue_count")) or 0) == 0
                        else _ops_level_from_count(snapshot.get("streams_local_issue_count"), amber_at=1.0)
                    )
                ),
                (
                    "Pending seed"
                    if not manifest_ready
                    else (
                        f"{int(_ops_float(snapshot.get('streams_local_issue_count')) or 0)} issues"
                        if backfill_pending_count == 0
                        else f"{int(_ops_float(snapshot.get('streams_local_issue_count')) or 0)} issues, {backfill_pending_count} backfill"
                    )
                ),
                "Mirror against source-host manifests",
            ),
            _ops_card_markup(
                "GWS archive mirror",
                (
                    "amber"
                    if not manifest_ready
                    else (
                        "amber"
                        if backfill_pending_count > 0 and int(_ops_float(snapshot.get("streams_gws_issue_count")) or 0) == 0
                        else _ops_level_from_count(snapshot.get("streams_gws_issue_count"), amber_at=1.0)
                    )
                ),
                (
                    "Pending seed"
                    if not manifest_ready
                    else (
                        f"{int(_ops_float(snapshot.get('streams_gws_issue_count')) or 0)} issues"
                        if backfill_pending_count == 0
                        else f"{int(_ops_float(snapshot.get('streams_gws_issue_count')) or 0)} issues, {backfill_pending_count} backfill"
                    )
                ),
                "Mirror against JASMIN manifests",
            ),
            _ops_card_markup(
                "Product gates",
                "amber"
                if not manifest_ready
                else (
                    "green"
                    if int(_ops_float(snapshot.get("streams_product_gate_ok_count")) or 0) == target_stream_count and backfill_pending_count == 0
                    else (
                        "amber"
                        if int(_ops_float(snapshot.get("streams_product_gate_ok_count")) or 0) == target_stream_count
                        else "red"
                    )
                ),
                (
                    "Pending seed"
                    if not manifest_ready
                    else (
                        f"{int(_ops_float(snapshot.get('streams_product_gate_ok_count')) or 0)}/{target_stream_count} streams ready"
                        + (f", {backfill_pending_count} backfill" if backfill_pending_count else "")
                    )
                ),
                "Processing success through candidate prune windows",
            ),
            _ops_card_markup(
                "Prune readiness",
                "amber"
                if not manifest_ready
                else (
                    "green"
                    if int(_ops_float(snapshot.get("streams_prune_ready_count")) or 0) == target_stream_count and backfill_pending_count == 0
                    else (
                        "amber"
                        if int(_ops_float(snapshot.get("streams_prune_ready_count")) or 0) == target_stream_count
                        else "red"
                    )
                ),
                (
                    "Pending seed"
                    if not manifest_ready
                    else (
                        f"{int(_ops_float(snapshot.get('streams_prune_ready_count')) or 0)}/{target_stream_count} streams deletable"
                        + (f", {backfill_pending_count} backfill" if backfill_pending_count else "")
                    )
                ),
                "Only prune upstream when source, local, and GWS agree",
            ),
        ]

        table_rows = []
        for spec in OPS_STREAM_SPECS:
            intentionally_paused = spec["stream_prefix"] in paused_prefixes
            source_level_stream = (
                "gray"
                if intentionally_paused
                else _ops_worst_level(
                    [
                        _ops_level_from_bool(snapshot.get(spec["source_key"])),
                        _ops_level_from_bool(snapshot.get(f"{spec['stream_prefix']}_source_recent_state")),
                    ]
                )
            )
            processing_level_stream = _ops_worst_level([_ops_level_from_bool(snapshot.get(key)) for key in spec["processing_keys"]])
            processing_ok = sum(1 for key in spec["processing_keys"] if _ops_bool(snapshot.get(key)) is True)
            archive_level_stream = _ops_archive_level(snapshot, spec["stream_prefix"])
            prune_level_stream = _ops_prune_level(snapshot, spec["stream_prefix"], manifest_ready)
            processing_detail = f"{processing_ok}/{len(spec['processing_keys'])} healthy"
            table_rows.append(
                "<tr>"
                f"<th class='ops-table__rowlabel'>{escape(spec['label'])}</th>"
                f"{_ops_table_cell(source_level_stream, 'Paused' if intentionally_paused else 'Source', _ops_source_freshness_text(snapshot, spec['stream_prefix'], intentionally_paused=intentionally_paused))}"
                f"{_ops_table_cell(processing_level_stream, 'Processing', processing_detail)}"
                f"{_ops_table_cell(archive_level_stream, _ops_archive_text(snapshot, spec['stream_prefix']), 'Raw mirror to GWS')}"
                f"{_ops_table_cell(prune_level_stream, _ops_prune_text(snapshot, spec['stream_prefix'], manifest_ready), 'Deletion gate')}"
                "</tr>"
            )

        failed_markup = ""
        if failed_services:
            failed_items = "".join(f"<li>{escape(name)}</li>" for name in failed_services[:8])
            failed_markup = (
                "<div class='ops-section'>"
                "<div class='ops-section-title'>Current service issues</div>"
                f"<div class='ops-callout ops-callout--red'><ul>{failed_items}</ul></div>"
                "</div>"
            )

        perf["status"] = "ok"
        perf["failed_services"] = len(failed_services)
        perf["manifest_ready"] = manifest_ready
        perf["snapshot_age_min"] = snapshot_age_min

        return (
            "<div class='ops-shell'>"
            "<div class='ops-headline'>"
            "<div class='ops-headline__main'>"
            f"<div class='ops-section-title ops-section-title--headline'>{_ops_light_markup(overall_level, 'Operations Dashboard')}</div>"
            "<div class='ops-headline__text'>Traffic lights summarize service health, storage pressure, archive transfer status, and prune readiness from the latest operations snapshot.</div>"
            "</div>"
            "<div class='ops-legend'>"
            f"{_ops_light_markup('green', 'Healthy')}"
            f"{_ops_light_markup('amber', 'Attention or pending')}"
            f"{_ops_light_markup('red', 'Failing or blocked')}"
            f"{_ops_light_markup('gray', 'Unknown')}"
            "</div>"
            "</div>"
            "<div class='ops-section'>"
            "<div class='ops-section-title'>System summary</div>"
            f"<div class='ops-grid ops-grid--summary'>{''.join(summary_cards)}</div>"
            "</div>"
            "<div class='ops-section'>"
            "<div class='ops-section-title'>Root-cause groups</div>"
            f"<div class='ops-grid ops-grid--root-cause'>{root_cause_cards}</div>"
            "</div>"
            "<div class='ops-section'>"
            "<div class='ops-section-title'>Seven-day trends</div>"
            f"<div class='ops-grid ops-grid--trends'>{trend_cards}</div>"
            "</div>"
            "<div class='ops-section'>"
            "<div class='ops-section-title'>Deployment state</div>"
            f"<div class='ops-grid ops-grid--summary'>{''.join(deployment_cards)}</div>"
            "</div>"
            "<div class='ops-section'>"
            "<div class='ops-section-title'>Public endpoints</div>"
            f"<div class='ops-grid ops-grid--summary'>{''.join(failover_cards)}</div>"
            "<div class='ops-footnote'>These probes check both public app URLs and require the full AURORA document, so a small blank Bokeh shell does not count as healthy.</div>"
            "</div>"
            "<div class='ops-section'>"
            "<div class='ops-section-title'>Storage</div>"
            f"<div class='ops-grid ops-grid--storage'>{''.join(storage_cards)}</div>"
            "</div>"
            "<div class='ops-section'>"
            "<div class='ops-section-title'>Archive and pruning</div>"
            f"<div class='ops-grid ops-grid--summary'>{''.join(manifest_cards)}</div>"
            "</div>"
            f"{failed_markup}"
            "<div class='ops-section'>"
            "<div class='ops-section-title'>Per-stream health</div>"
            "<div class='ops-table-wrap'>"
            "<table class='ops-table'>"
            "<thead><tr><th>Stream</th><th>Source sync</th><th>Processing</th><th>Archive</th><th>Prune gate</th></tr></thead>"
            f"<tbody>{''.join(table_rows)}</tbody>"
            "</table>"
            "</div>"
            "<div class='ops-footnote'>Amber can mean an expected wait state, such as a manifest seed or verification window that has not completed yet. Red indicates a failing service, blocked gate, or critical storage pressure.</div>"
            "</div>"
            "</div>"
        )


def _selected_token_window(selected: str | None) -> tuple[datetime | None, datetime | None, str | None]:
    if not selected:
        return None, None, None
    if selected == "Today (latest)":
        day_token = datetime.now(timezone.utc).strftime("%Y%m%d")
    elif selected == "latest":
        day_token = datetime.now(timezone.utc).strftime("%Y%m%d")
    elif len(selected) == 8 and selected.isdigit():
        day_token = selected
    else:
        return None, None, None
    start = datetime.strptime(day_token, "%Y%m%d")
    if day_token == datetime.now(timezone.utc).strftime("%Y%m%d"):
        end = datetime.now(timezone.utc).replace(tzinfo=None)
    else:
        end = start + timedelta(days=1) - timedelta(seconds=1)
    return start, end, day_token


def open_window(t0, t1, bottom_m=None, top_m=None, instrument: str | None = None, render_quality: str = "full"):
    """Prepare a bounded dataset while retaining the historical app interface."""
    instrument = instrument or CURRENT_INSTRUMENT
    cfg = _cfg(instrument)
    t0 = _ensure_utc(t0)
    t1 = _ensure_utc(t1)
    with _timed_perf(
        "window_open",
        instrument=instrument,
        start=t0,
        end=t1,
        bottom_m=bottom_m,
        top_m=top_m,
        render_quality=render_quality,
    ) as perf:
        if t0 is None or t1 is None:
            perf["status"] = "invalid_window"
            return xr.Dataset()
        return prepare_dataset_window(
            _get_base_dataset(instrument),
            WindowRequest(
                start=t0,
                end=t1,
                bottom_m=bottom_m,
                top_m=top_m,
                height_load_max=cfg["height_load_max"],
                render_quality=render_quality,
            ),
            valid_time_mask=_valid_time_mask,
            perf=perf,
        )


def _make_plot(ds, var, clim, logz, coloraxis):
    """Build a Plotly heatmap trace for a variable with optional log10 scaling."""
    times = pd.to_datetime(ds["time"].values)
    heights = ds["range"].values
    data = np.array(ds[var].transpose("range", "time"))
    zmin, zmax = clim
    if logz:
        data = np.where(data > 0, data, np.nan)
        with np.errstate(divide="ignore"):
            data = np.log10(data)
        zmin, zmax = np.log10(clim[0]), np.log10(clim[1])
    times, data = insert_time_gap_breaks(times, data, time_axis=1)
    trace = go.Heatmap(
        x=pd.to_datetime(times),
        y=heights,
        z=data,
        connectgaps=False,
        hoverongaps=False,
        zmin=zmin,
        zmax=zmax,
        coloraxis=coloraxis,
        showscale=False,
    )
    return trace


def _is_stacked_timeseries_instrument(inst: str) -> bool:
    return is_summary_instrument(inst)


def _is_wxcam_instrument(inst: str) -> bool:
    return inst == "wxcam"


def _is_uas_instrument(inst: str) -> bool:
    return inst == "uas"


# Widgets / controls (Panel wires these into the view updater)
default_end = _utcnow_naive()
default_start = default_end - DEFAULT_WINDOW
range_start = pn.widgets.DatetimePicker(name="Start (UTC)", value=default_start)
range_end = pn.widgets.DatetimePicker(name="End (UTC)", value=default_end)
top_range_m = pn.widgets.IntInput(name="Top range (m)", value=_cfg()["top_range_default"], step=100, start=500)
bottom_range_m = pn.widgets.IntInput(name="Bottom range (m)", value=0, step=100, start=0)
var1_select = pn.widgets.Select(name="Top var", options=list(_cfg()["vars"].keys()), value=_cfg()["default_top"])
var2_select = pn.widgets.Select(name="Bottom var", options=list(_cfg()["vars"].keys()), value=_cfg()["default_bottom"])
beta_vmin = pn.widgets.FloatInput(name="Var1 min", value=_cfg()["vars"][var1_select.value]["clim"][0], step=0.1)
beta_vmax = pn.widgets.FloatInput(name="Var1 max", value=_cfg()["vars"][var1_select.value]["clim"][1], step=0.1)
ldr_vmin = pn.widgets.FloatInput(name="Var2 min", value=_cfg()["vars"][var2_select.value]["clim"][0], step=0.1)
ldr_vmax = pn.widgets.FloatInput(name="Var2 max", value=_cfg()["vars"][var2_select.value]["clim"][1], step=0.1)
lwp_ymin = pn.widgets.FloatInput(name="LWP min (g/m²)", value=0.0, step=10.0, visible=False)
lwp_ymax = pn.widgets.FloatInput(name="LWP max (g/m²)", value=400.0, step=10.0, visible=False)
iwv_ymin = pn.widgets.FloatInput(name="IWV min (kg/m²)", value=0.0, step=1.0, visible=False)
iwv_ymax = pn.widgets.FloatInput(name="IWV max (kg/m²)", value=40.0, step=1.0, visible=False)
irr_ymin = pn.widgets.FloatInput(name="IRR / SURF_T min (°C)", value=-20.0, step=1.0, visible=False)
irr_ymax = pn.widgets.FloatInput(name="IRR / SURF_T max (°C)", value=10.0, step=1.0, visible=False)
prev_btn = pn.widgets.Button(name="Previous Day", button_type="default")
next_btn = pn.widgets.Button(name="Next Day/Current Day", button_type="default")
live_toggle = pn.widgets.Toggle(name="Live Update (Last 24h)", button_type="primary", value=True)
reset_view_btn = pn.widgets.Button(name="Reset View Defaults", button_type="default")
instrument_select = pn.widgets.Select(name="Instrument", value=CURRENT_INSTRUMENT, options=INSTRUMENT_OPTIONS)
power_view_select = pn.widgets.RadioButtonGroup(
    name="Power view",
    options={
        "Current Conditions": "current",
        "Forecast & Planning": "forecast",
    },
    value="current",
    button_type="primary",
    sizing_mode="stretch_width",
    css_classes=["power-view-select"],
)
science_instrument = pn.widgets.Select(name="Instrument", value=CURRENT_INSTRUMENT, options=SCIENCE_INSTRUMENT_OPTIONS)
science_image_type = pn.widgets.Select(name="Image type", options=[], visible=False)
hk_instrument = pn.widgets.Select(name="Instrument", value=CURRENT_INSTRUMENT, options=HK_INSTRUMENT_OPTIONS)
uas_window = pn.widgets.Select(name="Window", value="Last 24 h", options=list(UAS_WINDOW_OPTIONS.keys()))
uas_flight_day = pn.widgets.Select(
    name="Flight day (UTC)",
    value="latest",
    options={"Latest available": "latest"},
)
uas_flight_select = pn.widgets.Select(
    name="Flight",
    value=None,
    options={"No flights available": None},
)
uas_refresh = pn.widgets.Button(name="Refresh", button_type="primary", width=110)

_live_guard = False
_instrument_guard = False
_instrument_change_origin = "interactive"
_uas_flight_widget_guard = False
_uas_flight_manual_selection = False
_uas_flight_listing: dict = {}
_UAS_PENDING_DAY_QUERY: str | None = None
_UAS_PENDING_FLIGHT_QUERY: str | None = None
_live_cb = None  # handle for periodic callback (used for live refresh)
_relayout_guard = False  # prevents loops when syncing zoom back to widgets
_base_dataset_timer = _safe_periodic_callback(_refresh_time_bounds_cache, period=DATA_REFRESH_MS, start=False)


def _last_24h_utc_window() -> tuple[datetime, datetime]:
    end = _utcnow_naive()
    return end - DEFAULT_WINDOW, end


def _set_float_input_default(inp, value: float, peer_value: float) -> None:
    inp.value = value
    span = abs(peer_value - value) or 1.0
    inp.step = span / 100.0
    inp.format = "0.000000e+00" if max(abs(value), abs(peer_value)) < 1e-3 else None


def _apply_plot_control_defaults(inst: str | None = None, reset_variables: bool = True) -> None:
    """Reset visible plot controls for the current instrument without changing data."""
    inst = inst or CURRENT_INSTRUMENT
    cfg = _cfg(inst)
    vars_cfg = cfg["vars"]
    is_hatpro = inst == "Scanning Microwave Radiometer"
    is_stacked_timeseries = _is_stacked_timeseries_instrument(inst)
    is_wxcam = _is_wxcam_instrument(inst)

    if reset_variables and not is_wxcam:
        var1_select.value = cfg["default_top"]
        var2_select.value = cfg["default_bottom"]

    bottom_range_m.value = 0
    top_range_m.value = cfg["top_range_default"]

    var1 = vars_cfg.get(var1_select.value) or vars_cfg.get(cfg["default_top"])
    var2 = vars_cfg.get(var2_select.value) or vars_cfg.get(cfg["default_bottom"])
    if var1:
        beta_vmin.name = f"{var1['label']} min"
        beta_vmax.name = f"{var1['label']} max"
        _set_float_input_default(beta_vmin, var1["clim"][0], var1["clim"][1])
        _set_float_input_default(beta_vmax, var1["clim"][1], var1["clim"][0])
    if var2 and not (is_hatpro or is_stacked_timeseries or is_wxcam):
        ldr_vmin.name = f"{var2['label']} min"
        ldr_vmax.name = f"{var2['label']} max"
        _set_float_input_default(ldr_vmin, var2["clim"][0], var2["clim"][1])
        _set_float_input_default(ldr_vmax, var2["clim"][1], var2["clim"][0])

    if is_hatpro:
        beta_vmin.name = "T_PROF min (K)"
        beta_vmax.name = "T_PROF max (K)"
        if var1:
            _set_float_input_default(beta_vmin, var1["clim"][0], var1["clim"][1])
            _set_float_input_default(beta_vmax, var1["clim"][1], var1["clim"][0])
        lwp_ymin.value, lwp_ymax.value = 0.0, 400.0
        iwv_ymin.value, iwv_ymax.value = 0.0, 60.0
        irr_ymin.value, irr_ymax.value = -20.0, 60.0


def _capture_current_instrument_state(inst: str | None = None) -> None:
    inst = inst or CURRENT_INSTRUMENT
    _INSTRUMENT_VIEW_STATE[inst] = {
        "range_start": _ensure_utc(range_start.value),
        "range_end": _ensure_utc(range_end.value),
        "top_range_m": top_range_m.value,
        "bottom_range_m": bottom_range_m.value,
        "var1_select": var1_select.value,
        "var2_select": var2_select.value,
        "beta_vmin": beta_vmin.value,
        "beta_vmax": beta_vmax.value,
        "ldr_vmin": ldr_vmin.value,
        "ldr_vmax": ldr_vmax.value,
        "lwp_ymin": lwp_ymin.value,
        "lwp_ymax": lwp_ymax.value,
        "iwv_ymin": iwv_ymin.value,
        "iwv_ymax": iwv_ymax.value,
        "irr_ymin": irr_ymin.value,
        "irr_ymax": irr_ymax.value,
        "live_toggle": live_toggle.value,
        "science_image_type": science_image_type.value,
        "wxcam_image_type": globals().get("wxcam_image_type").value if "wxcam_image_type" in globals() else None,
        "wxcam_date": globals().get("wxcam_date").value if "wxcam_date" in globals() else None,
    }


def _update_science_image_type_control(inst: str, saved_state: dict | None = None) -> None:
    """Show the WXcam image selector only while WXcam quicklooks are selected."""
    science_image_type.name = "Image type"
    is_wxcam = _is_wxcam_instrument(inst)
    science_image_type.visible = is_wxcam
    if not is_wxcam:
        science_image_type.options = []
        science_image_type.value = None
        return

    cfg = _cfg(inst)
    vars_cfg = cfg["vars"]
    saved_state = saved_state or {}
    saved_wxcam_type = saved_state.get("wxcam_image_type") or saved_state.get("science_image_type")
    selected = saved_wxcam_type if saved_wxcam_type in vars_cfg else cfg["default_top"]
    science_image_type.options = list(vars_cfg.keys())
    science_image_type.value = selected


def _apply_instrument_defaults(inst: str, reset_time: bool = True, sync_quicklooks: bool = False):
    """Switch instrument: refresh dataset cache, reset controls, and relabel color widgets."""
    global CURRENT_INSTRUMENT, _instrument_guard
    _instrument_guard = True
    previous_instrument = CURRENT_INSTRUMENT
    CURRENT_INSTRUMENT = inst
    if previous_instrument != inst or _BASE_DS.get(inst) is None:
        _refresh_base_dataset(inst)
    else:
        _refresh_time_bounds_cache(inst)
    cfg = _cfg(inst)
    saved_state = _INSTRUMENT_VIEW_STATE.get(inst, {})
    with hold():
        vars_cfg = cfg["vars"]
        is_hatpro = inst == "Scanning Microwave Radiometer"
        is_stacked_timeseries = _is_stacked_timeseries_instrument(inst)
        is_wxcam = _is_wxcam_instrument(inst)
        var1_name = cfg["default_top"]
        var2_name = cfg["default_bottom"]
        var1_select.options = list(vars_cfg.keys())
        var2_select.options = list(vars_cfg.keys())
        var1_select.value = var1_name
        var2_select.value = var2_name
        var1_select.name = "Overview" if is_stacked_timeseries else "Top var"
        var2_select.name = "Bottom var"
        var1 = vars_cfg[var1_name]
        var2 = vars_cfg[var2_name]
        beta_vmin.name = f"{var1['label']} min"
        beta_vmax.name = f"{var1['label']} max"
        ldr_vmin.name = f"{var2['label']} min"
        ldr_vmax.name = f"{var2['label']} max"

        def _set_input(inp, vmin, vmax):
            inp.value = vmin
            # heuristic step
            span = abs(vmax - vmin) or 1.0
            inp.step = span / 100.0
            inp.format = "0.000000e+00" if max(abs(vmin), abs(vmax)) < 1e-3 else None

        _set_input(beta_vmin, var1["clim"][0], var1["clim"][1])
        _set_input(beta_vmax, var1["clim"][1], var1["clim"][0])
        _set_input(ldr_vmin, var2["clim"][0], var2["clim"][1])
        _set_input(ldr_vmax, var2["clim"][1], var2["clim"][0])

        bottom_range_m.value = 0
        top_range_m.value = cfg["top_range_default"]

        if sync_quicklooks:
            science_instrument.value = inst
            hk_instrument.value = inst

        if reset_time:
            saved_live = bool(saved_state.get("live_toggle")) if (saved_state and not is_wxcam) else False
            saved_start = saved_state.get("range_start")
            saved_end = saved_state.get("range_end")
            if (
                not saved_live
                and isinstance(saved_start, datetime)
                and isinstance(saved_end, datetime)
                and saved_start < saved_end
            ):
                range_start.value = saved_start
                range_end.value = saved_end
            elif _APP_BOOTSTRAPPING:
                start, end = _last_24h_utc_window()
                range_start.value = start
                range_end.value = end
            else:
                start, end = _last_24h_utc_window()
                range_start.value = start
                range_end.value = end
            # WXcam is a manual browser: refresh when switching back into it,
            # but do not keep a hidden live timer running while it is selected.
            _set_live(saved_live if not is_wxcam else False)

        # Instrument-specific UI trimming
        range_start.visible = not is_wxcam
        range_end.visible = not is_wxcam
        live_toggle.visible = not is_wxcam
        var1_select.visible = not is_hatpro and not is_stacked_timeseries and not is_wxcam
        var2_select.visible = not is_hatpro and not is_stacked_timeseries and not is_wxcam
        bottom_range_m.visible = not is_stacked_timeseries and not is_wxcam
        top_range_m.visible = not is_stacked_timeseries and not is_wxcam
        ldr_vmin.visible = not (is_hatpro or is_stacked_timeseries or is_wxcam)
        ldr_vmax.visible = not (is_hatpro or is_stacked_timeseries or is_wxcam)
        beta_vmin.visible = not (is_stacked_timeseries or is_wxcam)
        beta_vmax.visible = not (is_stacked_timeseries or is_wxcam)
        prev_btn.visible = not is_wxcam
        next_btn.visible = not is_wxcam
        reset_view_btn.visible = not is_wxcam
        _update_science_image_type_control(inst, saved_state)
        if is_wxcam:
            saved_wxcam_type = saved_state.get("wxcam_image_type") or saved_state.get("science_image_type")
            wxcam_image_type.options = list(vars_cfg.keys())
            wxcam_image_type.value = saved_wxcam_type if saved_wxcam_type in vars_cfg else var1_name
            _refresh_wxcam_ql_options(preserve_current=False)
            saved_wxcam_date = saved_state.get("wxcam_date")
            if saved_wxcam_date in list(wxcam_date.options):
                wxcam_date.value = saved_wxcam_date
        if saved_state and not is_wxcam:
            if saved_state.get("var1_select") in vars_cfg:
                var1_select.value = saved_state["var1_select"]
            if saved_state.get("var2_select") in vars_cfg:
                var2_select.value = saved_state["var2_select"]
            if not is_stacked_timeseries:
                bottom_range_m.value = int(saved_state.get("bottom_range_m", bottom_range_m.value) or 0)
                top_range_m.value = int(saved_state.get("top_range_m", top_range_m.value) or top_range_m.value)
                beta_vmin.value = saved_state.get("beta_vmin", beta_vmin.value)
                beta_vmax.value = saved_state.get("beta_vmax", beta_vmax.value)
                ldr_vmin.value = saved_state.get("ldr_vmin", ldr_vmin.value)
                ldr_vmax.value = saved_state.get("ldr_vmax", ldr_vmax.value)
        if is_hatpro:
            beta_vmin.name = "T_PROF min (K)"
            beta_vmax.name = "T_PROF max (K)"
            _set_input(beta_vmin, var1["clim"][0], var1["clim"][1])
            _set_input(beta_vmax, var1["clim"][1], var1["clim"][0])
            # Show and reset timeseries y-range controls
            lwp_ymin.visible = lwp_ymax.visible = True
            iwv_ymin.visible = iwv_ymax.visible = True
            irr_ymin.visible = irr_ymax.visible = True
            lwp_ymin.value, lwp_ymax.value = 0.0, 400.0
            iwv_ymin.value, iwv_ymax.value = 0.0, 60.0
            irr_ymin.value, irr_ymax.value = -20.0, 60.0
            if saved_state:
                lwp_ymin.value = saved_state.get("lwp_ymin", lwp_ymin.value)
                lwp_ymax.value = saved_state.get("lwp_ymax", lwp_ymax.value)
                iwv_ymin.value = saved_state.get("iwv_ymin", iwv_ymin.value)
                iwv_ymax.value = saved_state.get("iwv_ymax", iwv_ymax.value)
                irr_ymin.value = saved_state.get("irr_ymin", irr_ymin.value)
                irr_ymax.value = saved_state.get("irr_ymax", irr_ymax.value)
        else:
            lwp_ymin.visible = lwp_ymax.visible = False
            iwv_ymin.visible = iwv_ymax.visible = False
            irr_ymin.visible = irr_ymax.visible = False

        if sync_quicklooks:
            _refresh_ql_options(preserve_current=False)
            _refresh_hk_options(preserve_current=False)
            # Force quicklook panes to refresh even if the selection string did not change.
            ql_date.param.trigger("value")
            hk_date.param.trigger("value")
    _instrument_guard = False
    # Run a single consolidated refresh after batching widget changes
    _update_view(
        range_start.value,
        range_end.value,
        bottom_range_m.value,
        top_range_m.value,
        var1_select.value,
        var2_select.value,
        beta_vmin.value,
        beta_vmax.value,
        ldr_vmin.value,
        ldr_vmax.value,
        lwp_ymin.value,
        lwp_ymax.value,
        iwv_ymin.value,
        iwv_ymax.value,
        irr_ymin.value,
        irr_ymax.value,
        instrument_select.value,
    )


def _refresh_to_latest(_event=None, reset_controls: bool = False):
    """Jump the interactive time controls to the latest 24 h window."""
    global _live_guard
    previous_guard = _live_guard
    _live_guard = True
    if not _is_wxcam_instrument(CURRENT_INSTRUMENT) and _dataset_cache_age(CURRENT_INSTRUMENT) >= timedelta(milliseconds=DATA_REFRESH_MS):
        _refresh_base_dataset(CURRENT_INSTRUMENT)
    start, end = _last_24h_utc_window()
    with hold():
        range_start.value = start
        range_end.value = end
        if reset_controls:
            _apply_plot_control_defaults(CURRENT_INSTRUMENT, reset_variables=True)
    _live_guard = previous_guard


def _set_live(state: bool):
    """Set live toggle state without re-triggering handlers."""
    global _live_guard
    previous_guard = _live_guard
    _live_guard = True
    live_toggle.value = state
    live_toggle.name = "Live Update (Last 24h)" if state else "Live Off"
    live_toggle.button_type = "primary" if state else "default"
    _live_guard = previous_guard


def _on_live_toggle(event):
    """Handle live toggle clicks."""
    if _live_guard:
        return
    if event.new:
        _set_live(True)
        _refresh_to_latest()
    else:
        _set_live(False)


live_toggle.param.watch(_on_live_toggle, "value")


def _reset_view_defaults(_event=None):
    """Reset time, height/range, and scale controls to the live default view."""
    if _is_wxcam_instrument(CURRENT_INSTRUMENT):
        return
    _set_live(True)
    _refresh_to_latest(reset_controls=True)


reset_view_btn.on_click(_reset_view_defaults)


def _on_instrument_change(event):
    """Switch datasets and reset controls when instrument dropdown changes."""
    if _instrument_guard:
        return
    if event.old:
        _capture_current_instrument_state(event.old)
    sync_quicklooks = _instrument_change_origin != "interactive"
    _apply_instrument_defaults(event.new, reset_time=True, sync_quicklooks=sync_quicklooks)
    if "power_plan_editor" in globals():
        _sync_power_section_visibility()
    if not _browser_tab_syncing and "desktop_tabs" in globals():
        active = _active_tab_slug()
        if event.new == "power" and active == "interactive":
            _set_active_tab("power")
        elif event.new != "power" and active == "power":
            _set_active_tab("interactive")


instrument_select.param.watch(_on_instrument_change, "value")


def _on_science_instrument_change(event):
    """Sync science quicklook instrument dropdown back to the main instrument selector."""
    if _instrument_guard:
        return
    _update_science_image_type_control(event.new)
    if _is_uas_instrument(event.new):
        # UAS has generated daily images, but no generic Zarr browser contract.
        # Refresh the Science view without widening the Interactive selector.
        _refresh_ql_options(preserve_current=False)
        ql_date.param.trigger("value")
        return
    global _instrument_change_origin
    _instrument_change_origin = "science"
    try:
        changed = instrument_select.value != event.new
        instrument_select.value = event.new
        if not changed:
            _refresh_ql_options(preserve_current=False)
            ql_date.param.trigger("value")
    finally:
        _instrument_change_origin = "interactive"


science_instrument.param.watch(_on_science_instrument_change, "value")


def _on_hk_instrument_change(event):
    """Sync housekeeping quicklook instrument dropdown back to the main selector."""
    if _instrument_guard:
        return
    global _instrument_change_origin
    _instrument_change_origin = "housekeeping"
    try:
        instrument_select.value = event.new
    finally:
        _instrument_change_origin = "interactive"


hk_instrument.param.watch(_on_hk_instrument_change, "value")


def _on_var_change(event):
    """Update limit widgets when variable selection changes."""
    cfg = _cfg()
    vars_cfg = cfg["vars"]
    if _is_wxcam_instrument(CURRENT_INSTRUMENT):
        if science_image_type.value != var1_select.value:
            science_image_type.value = var1_select.value
        return
    if _is_stacked_timeseries_instrument(CURRENT_INSTRUMENT):
        return
    var1 = vars_cfg.get(var1_select.value, None)
    var2 = vars_cfg.get(var2_select.value, None)
    if var1:
        beta_vmin.name = f"{var1['label']} min"
        beta_vmax.name = f"{var1['label']} max"
        beta_vmin.value, beta_vmax.value = var1["clim"]
    if var2:
        ldr_vmin.name = f"{var2['label']} min"
        ldr_vmax.name = f"{var2['label']} max"
        ldr_vmin.value, ldr_vmax.value = var2["clim"]


var1_select.param.watch(_on_var_change, "value")
var2_select.param.watch(_on_var_change, "value")


def _on_science_image_type_change(event):
    if not _is_wxcam_instrument(CURRENT_INSTRUMENT):
        return
    _refresh_ql_options(preserve_current=False)
    ql_date.param.trigger("value")


science_image_type.param.watch(_on_science_image_type_change, "value")


def _on_wxcam_image_type_change(event):
    if not _is_wxcam_instrument(CURRENT_INSTRUMENT):
        return
    _refresh_wxcam_ql_options(preserve_current=False)
    wxcam_date.param.trigger("value")

def _auto_refresh():
    """Periodic refresh when live mode is on."""
    if live_toggle.value:
        _refresh_to_latest()


# Kick off periodic live refresh.
_live_cb = _safe_periodic_callback(_auto_refresh, period=LIVE_REFRESH_MS, start=False)


def _shift_previous(_event=None):
    """Jump to the previous full UTC day (00:00–24:00), clamping to data start."""
    _set_live(False)
    tmin, tmax = _dataset_time_bounds()
    anchor_end = _ensure_utc(range_end.value) or _ensure_utc(range_start.value) or (tmax or _utcnow_naive())
    prev_day = (anchor_end - timedelta(days=1)).date()
    prev_start = datetime.combine(prev_day, datetime.min.time())
    prev_end = datetime.combine(prev_day, time(hour=23, minute=59))
    if tmin and prev_end < tmin:
        # no data that far back; keep current window
        return
    if tmin and prev_start < tmin:
        # Clamp to first day available, ending at that day's 23:59 (or tmax if earlier)
        prev_start = tmin
        prev_end = datetime.combine(tmin.date(), time(hour=23, minute=59))
        if tmax:
            prev_end = min(prev_end, tmax)
    range_start.value = prev_start
    range_end.value = prev_end


def _shift_next(_event=None):
    """Jump to the next full UTC day; if past data end, snap to latest 24h."""
    _set_live(False)
    tmin, tmax = _dataset_time_bounds()
    if range_end.value is None:
        _refresh_to_latest()
        return
    anchor = _ensure_utc(range_start.value) or _ensure_utc(range_end.value)
    if anchor is None:
        _refresh_to_latest()
        return
    next_start = datetime.combine(anchor.date() + timedelta(days=1), datetime.min.time())
    next_end = next_start + timedelta(days=1) - timedelta(minutes=1)
    if tmax and next_end > tmax:
        # Not enough data ahead; show latest 24h
        latest_start, latest_end = _last_24h_utc_window()
        if tmin:
            latest_start = max(latest_start, tmin)
        range_start.value = latest_start
        range_end.value = latest_end
    else:
        range_start.value = next_start
        range_end.value = next_end


prev_btn.on_click(_shift_previous)
next_btn.on_click(_shift_next)


def _on_manual_time_change(event):
    """Disable live mode if user edits time pickers manually."""
    if _live_guard or _relayout_guard:
        return
    if live_toggle.value:
        _set_live(False)


range_start.param.watch(_on_manual_time_change, "value")
range_end.param.watch(_on_manual_time_change, "value")

# Persistent plot pane so we can listen for zoom/pan events (relayout).
# Keep one stable shell for the interactive area so switching instruments
# reuses the same pane tree instead of rebuilding the whole section.
plot_pane = pn.pane.Plotly(config={"responsive": True}, sizing_mode="stretch_width", css_classes=["interactive-plot-pane"])
interactive_loading = pn.pane.HTML("", visible=False, sizing_mode="stretch_width", margin=(0, 0, 8, 0))
interactive_placeholder = pn.pane.HTML("", sizing_mode="stretch_width", margin=0)
interactive_body = pn.Column(plot_pane, sizing_mode="stretch_width", margin=0, css_classes=["interactive-plot-body"])
interactive_content = pn.Column(interactive_loading, interactive_body, sizing_mode="stretch_width", margin=0, css_classes=["interactive-content"])


def _interactive_placeholder_height(inst: str) -> int:
    if inst == "Scanning Microwave Radiometer":
        return 760
    if _is_stacked_timeseries_instrument(inst):
        return 620
    if _is_wxcam_instrument(inst):
        return 520
    return 760


def _interactive_loading_notice_markup(inst: str, message: str, phase: str) -> str:
    label = "WXcam" if inst == "wxcam" else display_name(inst)
    phase_label = "Refining detail" if phase == "refining" else "Loading"
    return (
        "<div class='interactive-loading-notice'>"
        f"<span class='interactive-loading-notice__badge'>{escape(phase_label)}</span>"
        f"<div class='interactive-loading-notice__text'><strong>{escape(label)}</strong> {escape(message)}</div>"
        "</div>"
    )


def _interactive_placeholder_markup(inst: str, message: str) -> str:
    label = "WXcam" if inst == "wxcam" else display_name(inst)
    return (
        "<div class='interactive-skeleton'>"
        f"<div class='interactive-skeleton__title'>{escape(label)}</div>"
        f"<div class='interactive-skeleton__subtitle'>{escape(message)}</div>"
        "<div class='interactive-skeleton__plot'></div>"
        "<div class='interactive-skeleton__plot interactive-skeleton__plot--secondary'></div>"
        "</div>"
    )


def _set_interactive_body(target) -> None:
    if len(interactive_body.objects) != 1 or interactive_body.objects[0] is not target:
        interactive_body[:] = [target]


def _show_interactive_placeholder(inst: str, message: str) -> None:
    placeholder_height = _interactive_placeholder_height(inst)
    interactive_placeholder.height = placeholder_height
    interactive_placeholder.min_height = placeholder_height
    interactive_placeholder.object = _interactive_placeholder_markup(inst, message)
    interactive_body.height = placeholder_height
    interactive_body.min_height = placeholder_height
    _set_interactive_body(interactive_placeholder)


def _set_interactive_loading(inst: str, message: str, phase: str = "loading", visible: bool = True) -> None:
    interactive_loading.object = _interactive_loading_notice_markup(inst, message, phase) if visible else ""
    interactive_loading.visible = visible


def _clear_interactive_loading() -> None:
    interactive_loading.object = ""
    interactive_loading.visible = False


def _viewport_int(attr: str) -> int | None:
    try:
        value = getattr(pn.state, attr, None)
    except Exception:
        return None
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _mobile_request_user_agent() -> bool:
    user_agent = (_request_header("User-Agent") or "").lower()
    return any(token in user_agent for token in ("iphone", "ipod", "android", "mobile"))


def _is_mobile_viewport() -> bool:
    viewport_width = _viewport_int("viewport_width")
    if viewport_width is not None:
        return viewport_width <= 768
    return _mobile_request_user_agent()


def _mobile_plot_width() -> int:
    viewport_width = _viewport_int("viewport_width")
    if viewport_width is None:
        viewport_width = 390 if _mobile_request_user_agent() else 768
    return max(320, min(760, viewport_width - 16))


def _figure_for_current_viewport(fig: go.Figure) -> tuple[go.Figure, int | None, int]:
    """Return a per-session display figure without mutating shared caches."""
    plot_height = int(getattr(fig.layout, "height", 900) or 900)
    if not _is_mobile_viewport():
        return fig, None, plot_height

    mobile_width = _mobile_plot_width()
    display_fig = go.Figure(fig)
    if plot_height > 900:
        mobile_height = max(900, min(plot_height, int(mobile_width * 4.2)))
    else:
        mobile_height = max(420, min(plot_height, int(mobile_width * 1.75)))

    display_fig.update_layout(
        autosize=False,
        width=mobile_width,
        height=mobile_height,
        margin=dict(l=44, r=14, t=42, b=56),
        showlegend=False,
        font=dict(size=10, color=THEME_TEXT),
        title=dict(font=dict(size=13, color=THEME_TEXT)),
    )
    display_fig.update_xaxes(
        automargin=False,
        domain=[0.0, 1.0],
        tickfont=dict(size=9),
        title_font=dict(size=10),
        title_standoff=14,
    )
    display_fig.update_yaxes(
        automargin=False,
        title_text="",
        tickfont=dict(size=9),
        title_font=dict(size=10),
        title_standoff=4,
    )
    for annotation in display_fig.layout.annotations or ():
        annotation.update(font=dict(size=10, color=THEME_TEXT), borderwidth=1)
    return display_fig, mobile_width, mobile_height


def _show_plot(fig: go.Figure, instrument: str | None = None, cache_figure: bool = True) -> None:
    global _DISPLAYED_INTERACTIVE_INSTRUMENT
    instrument = instrument or CURRENT_INSTRUMENT
    is_mobile = _is_mobile_viewport()
    display_fig, plot_width, plot_height = _figure_for_current_viewport(fig)
    plot_pane.config = {"responsive": True, "displayModeBar": not is_mobile, "displaylogo": False}
    plot_pane.width = plot_width
    plot_pane.height = plot_height
    plot_pane.min_height = plot_height
    interactive_body.width = plot_width
    interactive_body.height = plot_height
    interactive_body.min_height = plot_height
    _set_interactive_body(plot_pane)
    plot_pane.object = display_fig
    _DISPLAYED_INTERACTIVE_INSTRUMENT = instrument
    if cache_figure:
        _INTERACTIVE_FIGURE_CACHE[instrument] = go.Figure(fig)


def _show_interactive_panel(panel_obj, instrument: str | None = None) -> None:
    global _DISPLAYED_INTERACTIVE_INSTRUMENT
    plot_pane.width = None
    plot_pane.height = None
    plot_pane.min_height = None
    interactive_body.width = None
    interactive_body.height = None
    interactive_body.min_height = None
    _set_interactive_body(panel_obj)
    _DISPLAYED_INTERACTIVE_INSTRUMENT = instrument or CURRENT_INSTRUMENT


def _begin_render_request() -> int:
    global _RENDER_REQUEST_COUNTER, _ACTIVE_RENDER_REQUEST_ID
    _RENDER_REQUEST_COUNTER += 1
    _ACTIVE_RENDER_REQUEST_ID = _RENDER_REQUEST_COUNTER
    # Thread work cannot be forcibly interrupted safely, but cancelling its
    # awaiting task releases the event loop and ensures an obsolete result is
    # never published into the current Panel document.
    for pending_id, task in tuple(_BACKGROUND_RENDER_TASKS.items()):
        if pending_id == _ACTIVE_RENDER_REQUEST_ID:
            continue
        if not task.done():
            task.cancel()
        _BACKGROUND_RENDER_TASKS.pop(pending_id, None)
    return _ACTIVE_RENDER_REQUEST_ID


def _render_request_active(request_id: int | None) -> bool:
    return request_id is None or request_id == _ACTIVE_RENDER_REQUEST_ID


def _normalize_cache_value(value):
    if isinstance(value, datetime):
        return _ensure_utc(value).replace(microsecond=0).isoformat()
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime(warn=False).replace(microsecond=0).isoformat()
    if isinstance(value, float):
        return round(value, 4)
    return value


def _as_naive_utc_datetime(value):
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime(warn=False)
    return _ensure_utc(value)


def _floor_datetime(value, step: timedelta):
    value = _as_naive_utc_datetime(value)
    if value is None:
        return None
    step_seconds = max(int(step.total_seconds()), 1)
    timestamp = int(value.replace(tzinfo=timezone.utc).timestamp())
    floored = timestamp - (timestamp % step_seconds)
    return datetime.fromtimestamp(floored, tz=timezone.utc).replace(tzinfo=None)


def _is_power_latest_window(start, end, instrument: str) -> bool:
    if instrument != "power":
        return False
    start_dt = _as_naive_utc_datetime(start)
    end_dt = _as_naive_utc_datetime(end)
    if start_dt is None or end_dt is None or end_dt <= start_dt:
        return False
    if abs((end_dt - start_dt) - DEFAULT_WINDOW) > timedelta(minutes=2):
        return False
    # A fresh development prewarm is authoritative for the live browser
    # window. Avoid opening the raw Power Zarr merely to discover its latest
    # sample, because that read is exactly what the prewarm removes.
    prewarm_path = _prewarmed_interactive_path("power")
    try:
        prewarm_age = datetime.now(timezone.utc) - datetime.fromtimestamp(prewarm_path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        prewarm_age = None
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    if prewarm_age is not None and prewarm_age <= PREWARM_LATEST_CACHE_TOLERANCE:
        return abs(end_dt - now) <= PREWARM_LATEST_CACHE_TOLERANCE
    _lower, latest = _dataset_time_bounds(instrument)
    latest_dt = _as_naive_utc_datetime(latest)
    if latest_dt is None:
        return False
    return abs(end_dt - latest_dt) <= POWER_LATEST_CACHE_TOLERANCE


def _canonical_interactive_window(start, end, instrument: str):
    """Round Power's live latest window so small timestamp nudges can reuse cache."""
    if not _is_power_latest_window(start, end, instrument):
        return start, end, "exact"
    step = timedelta(minutes=max(1, POWER_LATEST_CACHE_ROUND_MINUTES))
    rounded_end = _floor_datetime(end, step)
    if rounded_end is None:
        return start, end, "exact"
    return rounded_end - DEFAULT_WINDOW, rounded_end, f"power_latest_{POWER_LATEST_CACHE_ROUND_MINUTES}min"


def _summary_context_start(start, instrument: str):
    """Include Power context only when compact display products are absent."""
    if instrument != "power":
        return start
    if _power_display_summary_path().exists() or _power_display_energy_path().exists():
        return start
    start_dt = _as_naive_utc_datetime(start)
    if start_dt is None:
        return start
    context_date = (start_dt - timedelta(days=max(0, POWER_CUMULATIVE_CONTEXT_DAYS))).date()
    return datetime.combine(context_date, time.min)


def _interactive_render_cache_key(
    start,
    end,
    bottom_val,
    top_val,
    var1_name,
    var2_name,
    bmin,
    bmax,
    lmin,
    lmax,
    lymin,
    lymax,
    iymin,
    iymax,
    rymin,
    rymax,
    instrument,
    render_quality: str = "full",
) -> tuple[object, ...]:
    start, end, window_cache_mode = _canonical_interactive_window(start, end, instrument)
    if instrument == "power" and window_cache_mode == "exact":
        step = timedelta(minutes=max(1, POWER_GENERAL_CACHE_ROUND_MINUTES))
        rounded_start = _floor_datetime(start, step)
        rounded_end = _floor_datetime(end, step)
        if rounded_start is not None and rounded_end is not None:
            start = rounded_start
            end = rounded_end
            window_cache_mode = f"power_{POWER_GENERAL_CACHE_ROUND_MINUTES}min"
    dataset_version: object = _DATASET_VERSION.get(instrument, 0)
    if window_cache_mode.startswith("power_"):
        dataset_version = window_cache_mode
    return (
        instrument,
        power_view_select.value if instrument == "power" else None,
        render_quality,
        dataset_version,
        window_cache_mode,
        _normalize_cache_value(start),
        _normalize_cache_value(end),
        _normalize_cache_value(bottom_val),
        _normalize_cache_value(top_val),
        var1_name,
        var2_name,
        _normalize_cache_value(bmin),
        _normalize_cache_value(bmax),
        _normalize_cache_value(lmin),
        _normalize_cache_value(lmax),
        _normalize_cache_value(lymin),
        _normalize_cache_value(lymax),
        _normalize_cache_value(iymin),
        _normalize_cache_value(iymax),
        _normalize_cache_value(rymin),
        _normalize_cache_value(rymax),
    )


def _store_interactive_render_cache(cache_key: tuple[object, ...] | None, fig: go.Figure) -> None:
    if cache_key is None or _is_wxcam_instrument(str(cache_key[0])):
        return
    _INTERACTIVE_RENDER_CACHE[cache_key] = go.Figure(fig)
    _INTERACTIVE_RENDER_CACHE.move_to_end(cache_key)
    while len(_INTERACTIVE_RENDER_CACHE) > INTERACTIVE_RENDER_CACHE_SIZE:
        _INTERACTIVE_RENDER_CACHE.popitem(last=False)


def _restore_exact_interactive_cache(cache_key: tuple[object, ...] | None, inst: str) -> bool:
    if cache_key is None:
        return False
    cached = _INTERACTIVE_RENDER_CACHE.get(cache_key)
    if cached is None:
        cached = _load_prewarmed_interactive_figure(cache_key, inst)
        if cached is None:
            return False
        _INTERACTIVE_RENDER_CACHE[cache_key] = go.Figure(cached)
        _INTERACTIVE_RENDER_CACHE.move_to_end(cache_key)
    _INTERACTIVE_RENDER_CACHE.move_to_end(cache_key)
    _show_plot(go.Figure(cached), instrument=inst, cache_figure=False)
    _perf_log("interactive_render_cache_hit", instrument=inst)
    return True


def _cache_key_targets_latest_prewarm(cache_key: tuple[object, ...], inst: str) -> bool:
    if len(cache_key) < 7 or cache_key[0] != inst:
        return False
    window_mode = str(cache_key[4])
    if inst not in {"power", "vaisalamet", "asfs-logger"}:
        return False
    if str(cache_key[2]) != _interactive_final_quality(inst):
        return False
    try:
        start_dt = _as_naive_utc_datetime(pd.Timestamp(cache_key[5]))
        end_dt = _as_naive_utc_datetime(pd.Timestamp(cache_key[6]))
    except Exception:
        return False
    if start_dt is None or end_dt is None or end_dt <= start_dt:
        return False
    if abs((end_dt - start_dt) - DEFAULT_WINDOW) > timedelta(minutes=5):
        return False
    if inst == "power":
        # Development prewarms represent the live browser window ending now,
        # rather than ending at the newest observed point. The latter hides a
        # collection gap and used to force an expensive synchronous Zarr read.
        prewarm_path = _prewarmed_interactive_path(inst)
        if not prewarm_path.exists():
            return False
        try:
            age = datetime.now(timezone.utc) - datetime.fromtimestamp(prewarm_path.stat().st_mtime, tz=timezone.utc)
        except OSError:
            return False
        if age > PREWARM_LATEST_CACHE_TOLERANCE:
            return False
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        return abs(end_dt - now) <= PREWARM_LATEST_CACHE_TOLERANCE
    else:
        _lower, latest = _dataset_time_bounds(inst)
    latest_dt = _as_naive_utc_datetime(latest)
    if latest_dt is None:
        return False
    return abs(end_dt - latest_dt) <= PREWARM_LATEST_CACHE_TOLERANCE


def _load_prewarmed_interactive_figure(cache_key: tuple[object, ...], inst: str) -> go.Figure | None:
    """Load a latest-view Plotly JSON produced by the quicklook pipeline."""
    if not _cache_key_targets_latest_prewarm(cache_key, inst):
        return None
    path = _prewarmed_interactive_path(inst)
    if not path.exists():
        return None
    with _timed_perf("interactive_prewarm_load", instrument=inst, path=str(path)) as perf:
        try:
            fig = _read_prewarmed_interactive_figure(path)
        except Exception as exc:
            perf["status"] = "error"
            perf["error"] = str(exc)
            return None
        perf["status"] = "ok"
        perf["trace_count"] = len(fig.data)
        return fig


def _read_prewarmed_interactive_figure(path: Path) -> go.Figure:
    """Read immutable prewarmed JSON without touching a Panel document."""
    return go.Figure(json.loads(path.read_text(encoding="utf-8")))


async def _load_prewarmed_interactive_figure_async(
    path: Path,
    cache_key: tuple[object, ...],
    inst: str,
    request_id: int,
    doc,
    render_args: tuple[object, ...],
) -> None:
    """Read cached JSON off the Tornado loop and publish only when still current."""
    try:
        with _timed_perf("interactive_prewarm_load_async", instrument=inst, path=str(path)) as perf:
            loop = asyncio.get_running_loop()
            fig = await loop.run_in_executor(
                _BACKGROUND_PREPARATION_EXECUTOR,
                _read_prewarmed_interactive_figure,
                path,
            )
            perf["status"] = "ok"
            perf["trace_count"] = len(fig.data)
    except asyncio.CancelledError:
        _perf_log(
            "interactive_prewarm_load_async",
            instrument=inst,
            request_id=request_id,
            status="cancelled",
        )
        raise
    except Exception as exc:
        _perf_log("interactive_prewarm_load_async", instrument=inst, status="error", error=str(exc))

        def recover() -> None:
            _IN_FLIGHT_INTERACTIVE_RENDER_CACHE_KEYS.discard(cache_key)
            if _render_request_active(request_id):
                _render_interactive_view(
                    *render_args,
                    request_id=request_id,
                    render_quality=_interactive_final_quality(inst),
                )

        doc.add_next_tick_callback(recover)
        return

    def publish() -> None:
        _IN_FLIGHT_INTERACTIVE_RENDER_CACHE_KEYS.discard(cache_key)
        if not _render_request_active(request_id):
            return
        _INTERACTIVE_RENDER_CACHE[cache_key] = go.Figure(fig)
        _INTERACTIVE_RENDER_CACHE.move_to_end(cache_key)
        _show_plot(fig, instrument=inst, cache_figure=False)
        _clear_interactive_loading()
        # Status/availability reads need the full source time index. Do not
        # contend with a Power prewarm before its first Plotly update reaches
        # the browser; the footer can follow shortly after first paint.
        if inst == "power":
            _schedule_timeout(_activate_interactive_footer_metrics, 1_500)

    doc.add_next_tick_callback(publish)


def _dataset_window_metrics(ds: xr.Dataset | None) -> dict[str, int]:
    if ds is None:
        return {"time_count": 0, "var_count": 0}
    return {
        "time_count": int(ds.sizes.get("time", 0)),
        "var_count": len(ds.data_vars),
    }


def _figure_metrics(fig: go.Figure) -> dict[str, int]:
    """Return lightweight render payload metrics without serializing by default."""
    point_counts: list[int] = []
    for trace in fig.data:
        x_values = getattr(trace, "x", None)
        y_values = getattr(trace, "y", None)
        if x_values is not None:
            try:
                point_counts.append(len(x_values))
                continue
            except TypeError:
                pass
        if y_values is not None:
            try:
                point_counts.append(len(y_values))
            except TypeError:
                point_counts.append(0)
        else:
            point_counts.append(0)
    metrics = {
        "plot_trace_count": len(fig.data),
        "plot_points_total": int(sum(point_counts)),
        "plot_points_max": int(max(point_counts)) if point_counts else 0,
        "plot_points_min": int(min(point_counts)) if point_counts else 0,
    }
    if os.environ.get("AURORA_MEASURE_PLOT_JSON_BYTES", "").lower() in {"1", "true", "yes"}:
        metrics["plot_json_bytes"] = len(fig.to_json().encode("utf-8"))
    return metrics


def _publish_plot_if_current(
    fig: go.Figure,
    instrument: str,
    request_id: int | None,
    cache_figure: bool = True,
    cache_key: tuple[object, ...] | None = None,
) -> bool:
    if not _render_request_active(request_id):
        return False
    _show_plot(fig, instrument=instrument, cache_figure=cache_figure)
    if cache_figure:
        _store_interactive_render_cache(cache_key, fig)
    return True


def _publish_panel_if_current(panel_obj, request_id: int | None) -> bool:
    if not _render_request_active(request_id):
        return False
    _show_interactive_panel(panel_obj)
    return True


def _restore_cached_interactive_view(inst: str) -> bool:
    if _is_wxcam_instrument(inst):
        if _DISPLAYED_INTERACTIVE_INSTRUMENT == inst and len(interactive_body.objects) == 1 and interactive_body.objects[0] is wxcam_interactive_browser:
            return True
        _show_interactive_panel(wxcam_interactive_browser, instrument=inst)
        return True
    cached = _INTERACTIVE_FIGURE_CACHE.get(inst)
    if cached is None:
        return False
    if _DISPLAYED_INTERACTIVE_INSTRUMENT == inst and len(interactive_body.objects) == 1 and interactive_body.objects[0] is plot_pane and plot_pane.object is not None:
        return True
    _show_plot(go.Figure(cached), instrument=inst, cache_figure=False)
    return True


def _science_quicklook_path_for_interactive(inst: str, start=None, end=None) -> Path | None:
    cfg = _cfg(inst)
    candidates: list[Path] = []
    if _is_stacked_timeseries_instrument(inst):
        quick_dir = Path(cfg["quicklook_dir"])
        start_dt = _as_naive_utc_datetime(start)
        end_dt = _as_naive_utc_datetime(end)
        if start_dt is not None and end_dt is not None:
            latest_path = summary_latest_png(quick_dir, inst)
            _lower, latest = _dataset_time_bounds(inst)
            latest_dt = _as_naive_utc_datetime(latest)
            if latest_dt is not None and abs(end_dt - latest_dt) <= POWER_LATEST_CACHE_TOLERANCE and latest_path.exists():
                candidates.append(latest_path)
            elif start_dt.date() == end_dt.date() and start_dt.hour == 0:
                candidates.append(summary_daily_png(quick_dir, inst, pd.Timestamp(start_dt)))
        candidates.append(summary_latest_png(quick_dir, inst))
    latest_image = cfg.get("latest_image")
    if latest_image:
        candidates.append(Path(latest_image))
    for path in candidates:
        if path.exists():
            return path
    return None


def _show_cached_quicklook_placeholder(inst: str, start=None, end=None) -> bool:
    path = _science_quicklook_path_for_interactive(inst, start=start, end=end)
    if path is None:
        return False
    panel_obj = pn.Column(
        pn.pane.HTML(
            "<div class='interactive-loading-note'>Showing a cached Science Quicklook while the interactive view loads.</div>",
            sizing_mode="stretch_width",
            margin=(0, 0, 8, 0),
        ),
        pn.pane.Image(str(path), sizing_mode="stretch_width"),
        sizing_mode="stretch_width",
    )
    _show_interactive_panel(panel_obj, instrument=inst)
    return True


def _interactive_supports_refine(inst: str) -> bool:
    return inst in {"Ceilometer", "Cloud Radar", "Scanning Microwave Radiometer"}


def _interactive_initial_quality(inst: str) -> str:
    """Choose the first render pass used to make an instrument feel responsive."""
    if inst == "power":
        return "downsampled"
    return "coarse" if _interactive_supports_refine(inst) else "full"


def _interactive_final_quality(inst: str) -> str:
    """Return the cache/render quality that represents the settled view."""
    return "downsampled" if inst == "power" else "full"


def _stacked_interactive_max_time_samples(instrument: str, render_quality: str) -> int:
    """Cap 1D interactive trace density by instrument and render pass."""
    if render_quality == "coarse":
        return SUMMARY_INTERACTIVE_COARSE_TIME_SAMPLES
    return SUMMARY_INTERACTIVE_MAX_TIME_SAMPLES.get(instrument, int(os.environ.get("AURORA_INTERACTIVE_MAX_TIME_SAMPLES", "1600")))


def _schedule_next_tick(callback) -> None:
    doc = pn.state.curdoc
    if doc is not None:
        doc.add_next_tick_callback(callback)
    else:
        callback()


def _schedule_timeout(callback, timeout_ms: int):
    doc = pn.state.curdoc
    if doc is not None:
        return doc.add_timeout_callback(callback, timeout_ms)
    callback()
    return None


def _image_type_from_selection(selection: str) -> str:
    cfg = _cfg("wxcam")
    return cfg["vars"][selection]["image_type"]


def _wxcam_today_token() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _empty_interactive_figure(instrument: str, reason: str, start=None, end=None, detail: str | None = None) -> go.Figure:
    start_dt = _as_naive_utc_datetime(start)
    end_dt = _as_naive_utc_datetime(end)
    contract = {
        "Ceilometer": "ceilometer",
        "Cloud Radar": "cloud-radar",
        "Scanning Microwave Radiometer": "hatpro",
    }.get(instrument, "")
    state = empty_data_state(
        instrument,
        reason,
        start=start_dt,
        end=end_dt,
        detail=detail,
        pdu_status=mobile_catalog.pdu_instrument_status(contract),
    )
    window = ""
    if state.start is not None and state.end is not None:
        window = f"<br><span style='font-size:12px;color:#647283'>Selected window: {state.start:%Y-%m-%d %H:%M} to {state.end:%Y-%m-%d %H:%M} UTC</span>"
    detail_markup = "" if not state.detail else f"<br><span style='font-size:12px;color:#647283'>{escape(state.detail)}</span>"
    eyebrow = (
        f"<span style='font-size:11px;color:{THEME_ACCENT};font-weight:600'>{state.eyebrow}</span><br>"
        if state.eyebrow
        else ""
    )
    fig = go.Figure()
    fig.add_annotation(
        text=f"{eyebrow}<b>{escape(state.instrument_title)}</b><br>{escape(state.reason)}{window}{detail_markup}",
        x=0.5,
        y=0.52,
        xref="paper",
        yref="paper",
        showarrow=False,
        align="center",
        font=dict(color=THEME_TEXT, size=16),
        bgcolor="#edf8f6" if state.intentionally_powered_off else None,
        bordercolor="#a9d8d0" if state.intentionally_powered_off else None,
        borderwidth=1 if state.intentionally_powered_off else 0,
        borderpad=16 if state.intentionally_powered_off else 0,
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        height=460,
        margin=dict(l=40, r=40, t=40, b=40),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color=THEME_TEXT),
    )
    return fig


def _wxcam_daily_video_dir(image_type: str) -> Path:
    return _wxcam_daily_video_root() / image_type


def _wxcam_row_time(row) -> datetime | None:
    if not row:
        return None
    try:
        return datetime.fromisoformat(str(row["time_utc"])).replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _humanize_age(moment: datetime | None) -> str:
    if moment is None:
        return "Updated time unavailable"
    delta = max(datetime.now(timezone.utc) - moment, timedelta(0))
    seconds = int(delta.total_seconds())
    if seconds < 90:
        return "Updated just now"
    minutes = seconds // 60
    if minutes < 60:
        return f"Updated {minutes} min ago"
    hours = minutes // 60
    if hours < 48:
        rem_minutes = minutes % 60
        return f"Updated {hours} h {rem_minutes:02d} min ago" if rem_minutes else f"Updated {hours} h ago"
    days = hours // 24
    return f"Updated {days} d ago"

def _wxcam_video_context(selection: str, selected_label: str | None) -> dict[str, object]:
    image_type = _image_type_from_selection(selection)
    path_str = _wxcam_interactive_video_options(selection).get(selected_label)
    if not path_str:
        return {}
    video_path = Path(path_str)
    latest_mode = selected_label == "Today (latest)"
    day_token = _wxcam_today_token() if latest_mode else (selected_label or "")
    headline = "Rolling latest 24 hours" if latest_mode else (
        pd.Timestamp(_wxcam_day_token_to_utc(day_token)).strftime("%Y-%m-%d UTC")
        if _wxcam_day_token_to_utc(day_token)
        else (selected_label or "Selected day")
    )
    last_row = latest_record(_wxcam_catalog_path("wxcam"), image_type, media_kind="video")
    last_time = _wxcam_row_time(last_row)
    last_clip_text = last_time.strftime("%H:%M UTC") if last_time else "n/a"
    subtitle_bits = [
        _humanize_age(last_time),
        f"Last clip {last_clip_text}",
    ]
    return {
        "path": video_path,
        "image_type": image_type,
        "selected_label": selected_label or "",
        "headline": headline,
        "day_token": day_token,
        "title": f"{selection} | {headline}",
        "subtitle": " | ".join(bit for bit in subtitle_bits if bit),
    }


def _wxcam_day_strip_markup(image_type: str, day_token: str) -> str:
    day_utc = _wxcam_day_token_to_utc(day_token)
    if not day_utc:
        return ""
    hourly_rows = representative_hourly_records(
        _wxcam_catalog_path("wxcam"),
        image_type,
        day_utc,
        media_kind="image",
    )
    cells: list[str] = []
    for hour in range(24):
        row = hourly_rows.get(hour)
        hour_label = f"{hour:02d}"
        if row is None:
            cells.append(
                "<div class='wxcam-hour-strip__tile wxcam-hour-strip__tile--empty'>"
                f"<div class='wxcam-hour-strip__placeholder'>{hour_label}</div>"
                "</div>"
            )
            continue
        thumb_path = _wxcam_hourly_thumbnail_path(image_type, day_token, str(row["filename"]))
        if thumb_path.exists():
            image_markup = f"<img class='wxcam-hour-strip__thumb' src='{_image_data_uri(thumb_path)}' alt='Hour {hour_label}'>"
        else:
            image_markup = f"<div class='wxcam-hour-strip__placeholder'>{hour_label}</div>"
        cells.append(
            "<div class='wxcam-hour-strip__tile wxcam-hour-strip__tile--day'>"
            f"{image_markup}"
            f"<div class='wxcam-hour-strip__hour'>{hour_label}</div>"
            "</div>"
        )
    return "".join(cells)


def _wxcam_hour_strip_markup(selection: str, selected_label: str | None) -> str:
    context = _wxcam_video_context(selection, selected_label)
    if not context:
        return ""
    if selected_label == "Today (latest)":
        return ""
    image_type = str(context["image_type"])
    body = _wxcam_day_strip_markup(image_type, str(context["day_token"]))
    if not body:
        return ""
    return (
        "<div class='wxcam-hour-strip'>"
        "<div class='wxcam-hour-strip__header'>"
        "<div class='wxcam-hour-strip__title'>Representative hourly stills (UTC)</div>"
        "<div class='wxcam-hour-strip__hint'>Historical days keep a small UTC hourly image strip for quick visual scanning.</div>"
        "</div>"
        f"<div class='wxcam-hour-strip__scroller'>{body}</div>"
        "</div>"
    )


def _wxcam_interactive_video_options(selection: str) -> dict[str, str | None]:
    image_type = _image_type_from_selection(selection)
    day_dir = _wxcam_daily_video_dir(image_type)
    with _timed_perf("wxcam_interactive_options", instrument="wxcam", image_type=image_type, day_dir=day_dir) as perf:
        if not day_dir.exists():
            perf["status"] = "missing_dir"
            return {"No videos available": None}
        opts: dict[str, str | None] = {}
        video_count = 0
        for video_path in sorted(day_dir.glob("*.mp4")):
            video_count += 1
            if video_path.name == "latest.mp4":
                continue
            opts[video_path.stem] = str(video_path)
        latest_path = day_dir / "latest.mp4"
        if latest_path.exists():
            opts["Today (latest)"] = str(latest_path)
        perf["video_count"] = video_count
        perf["option_count"] = len(opts)
        perf["status"] = "ok" if opts else "empty"
        return opts or {"No videos available": None}


def _wxcam_calendar_options(selection: str) -> dict[str, str | None]:
    """Return WXcam Science Quicklook day choices for the HDR thumbnail grid.

    Some internal state/event names still say "calendar" from the older tab
    label. The visible UI is now Science Quicklooks.
    """
    image_type = _image_type_from_selection(selection)
    with _timed_perf("wxcam_calendar_options", instrument="wxcam", image_type=image_type) as perf:
        day_values = available_days(_wxcam_catalog_path("wxcam"), image_type, media_kind="image")
        if not day_values:
            perf["status"] = "empty"
            return {"No images available": None}
        today_token = _wxcam_today_token()
        opts: dict[str, str | None] = {}
        for day_utc in day_values:
            day_token = day_utc.replace("-", "")
            if day_token == today_token:
                continue
            opts[day_token] = day_token
        if any(day_utc.replace("-", "") == today_token for day_utc in day_values):
            opts["Today (latest)"] = today_token
        perf["day_count"] = len(day_values)
        perf["option_count"] = len(opts)
        perf["status"] = "ok"
        return opts or {"No images available": None}


def _wxcam_day_token_to_utc(day_token: str) -> str | None:
    if len(day_token) != 8 or not day_token.isdigit():
        return None
    return f"{day_token[:4]}-{day_token[4:6]}-{day_token[6:8]}"


def _wxcam_calendar_day_token(selected_day: str | None) -> str | None:
    """Normalize the quicklook day selector's latest label to today's token."""
    if selected_day == "Today (latest)":
        return _wxcam_today_token()
    return selected_day


def _wxcam_hourly_image_rows(selection: str, day_token: str | None) -> dict[int, object]:
    day_utc = _wxcam_day_token_to_utc(day_token or "")
    if not day_utc:
        return {}
    image_type = _image_type_from_selection(selection)
    return representative_hourly_records(
        _wxcam_catalog_path("wxcam"),
        image_type,
        day_utc,
        media_kind="image",
    )


def _wxcam_hourly_thumbnail_path(image_type: str, day_token: str, source_name: str) -> Path:
    return _wxcam_hourly_thumbnail_root() / image_type / day_token / f"{Path(source_name).stem}.jpg"


def _wxcam_video_poster_data_uri(image_type: str, day_token: str, selected_label: str) -> str:
    day_utc = _wxcam_day_token_to_utc(day_token)
    if not day_utc:
        return ""
    rows = representative_hourly_records(
        _wxcam_catalog_path("wxcam"),
        image_type,
        day_utc,
        media_kind="image",
    )
    if not rows:
        return ""
    if selected_label == "Today (latest)":
        row = rows[max(rows)]
    else:
        row = rows.get(12) or rows[min(rows)]
    thumb_path = _wxcam_hourly_thumbnail_path(image_type, day_token, str(row["filename"]))
    if not thumb_path.exists():
        return ""
    try:
        return _image_data_uri(thumb_path)
    except Exception:
        return ""


@lru_cache(maxsize=24)
def _cached_quicklook_png_data_uri(path_str: str, size_bytes: int, mtime_ns: int) -> str:
    encoded = b64encode(Path(path_str).read_bytes()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _responsive_quicklook_png(path: Path) -> pn.pane.HTML:
    """Render a PNG without Panel's fixed intrinsic-height image box.

    Panel's native PNG pane stores the file's pixel height on the Bokeh model.
    That leaves a large blank area below tall quicklooks when the browser scales
    the image down to fit the available width. A plain HTML image lets the
    browser preserve the aspect ratio and collapse the following footer tightly.
    """
    display_path = _quicklook_display_path(path)
    stat_result = display_path.stat()
    data_uri = _cached_quicklook_png_data_uri(str(display_path), stat_result.st_size, stat_result.st_mtime_ns)
    alt = escape(display_path.stem.replace("_", " "))
    return pn.pane.HTML(
        f"<img class='quicklook-image__img' src='{data_uri}' alt='{alt}'>",
        sizing_mode="stretch_width",
        margin=0,
        css_classes=["quicklook-image"],
    )


def _media_pane(path: str):
    suffix = Path(path).suffix.lower()
    if suffix == ".mp4":
        return pn.pane.Video(path, sizing_mode="stretch_width", autoplay=False, loop=False, muted=True)
    if suffix == ".png":
        return _responsive_quicklook_png(Path(path))
    return pn.pane.Image(path, sizing_mode="stretch_width")


@lru_cache(maxsize=256)
def _cached_image_data_uri(path_str: str, size_bytes: int, mtime_ns: int) -> str:
    image_bytes = Path(path_str).read_bytes()
    encoded = b64encode(image_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def _image_data_uri(path: Path) -> str:
    stat_result = path.stat()
    return _cached_image_data_uri(str(path), stat_result.st_size, stat_result.st_mtime_ns)


wxcam_player = WxcamVideoPlayer(sizing_mode="stretch_width")
wxcam_player_shell = pn.Column(wxcam_player, sizing_mode="stretch_width")


def _build_wxcam_video_view(path: Path, selection: str, selected_label: str, context: dict[str, object]):
    image_type = _image_type_from_selection(selection)
    with _timed_perf("wxcam_video_view_build", instrument="wxcam", image_type=image_type, path=path, selected_label=selected_label) as perf:
        mode_class = "wxcam-player--vertical" if image_type == "fish_hdr" else "wxcam-player--wide"
        title = str(context.get("title", f"{selection} | {selected_label}"))
        subtitle = str(context.get("subtitle", ""))
        stat_result = path.stat()
        perf["size_bytes"] = int(stat_result.st_size)
        wxcam_player.src = _wxcam_media_url(path)
        perf["src_mode"] = "static_url"
        wxcam_player.title = title
        wxcam_player.subtitle = subtitle
        wxcam_player.mode_class = mode_class
        wxcam_player.poster = _wxcam_video_poster_data_uri(
            image_type,
            str(context.get("day_token", "")),
            selected_label,
        )
        return wxcam_player_shell


def _build_wxcam_image_view(path: Path, selection: str, selected_label: str):
    image_type = _image_type_from_selection(selection)
    with _timed_perf("wxcam_image_view_build", instrument="wxcam", image_type=image_type, path=path, selected_label=selected_label) as perf:
        mode_class = "wxcam-still--vertical" if image_type == "fish_hdr" else "wxcam-still--wide"
        title = escape(f"{selection} | {selected_label} | {path.name}")
        alt = escape(f"{selection} {selected_label}")
        stat_result = path.stat()
        perf["size_bytes"] = int(stat_result.st_size)
        return pn.Column(
            pn.pane.HTML(
                (
                    f"<div class='wxcam-still {mode_class}'>"
                    f"<div class='wxcam-player__meta'><div class='wxcam-player__title'>{title}</div></div>"
                    f"<div class='wxcam-still__frame'><img src='{_image_data_uri(path)}' alt='{alt}'></div>"
                    f"</div>"
                ),
                sizing_mode="stretch_width",
                margin=0,
            ),
            sizing_mode="stretch_width",
        )


wxcam_image_type = pn.widgets.Select(
    name="Image type",
    options=list(_cfg("wxcam")["vars"].keys()),
    value=_cfg("wxcam")["default_top"],
)
_wxcam_ql_options = _wxcam_interactive_video_options(wxcam_image_type.value)
wxcam_date = pn.widgets.Select(name="Date", options=list(_wxcam_ql_options.keys()))
if _wxcam_ql_options:
    wxcam_date.value = list(_wxcam_ql_options.keys())[-1]
wxcam_latest = pn.widgets.Button(name="Latest", button_type="primary")
wxcam_prev = pn.widgets.Button(name="Previous", button_type="default")
wxcam_next = pn.widgets.Button(name="Next", button_type="default")


def _refresh_wxcam_ql_options(preserve_current: bool = True):
    global _wxcam_ql_options
    current = wxcam_date.value if preserve_current else None
    _wxcam_ql_options = _wxcam_interactive_video_options(wxcam_image_type.value or _cfg("wxcam")["default_top"])
    opts = list(_wxcam_ql_options.keys())
    wxcam_date.options = opts
    if not opts:
        wxcam_date.value = None
        return
    if preserve_current and current in opts:
        wxcam_date.value = current
    else:
        wxcam_date.value = opts[-1]


def _shift_wxcam_ql(delta: int):
    _refresh_wxcam_ql_options(preserve_current=True)
    opts = list(wxcam_date.options)
    if not opts or wxcam_date.value not in opts:
        return
    idx = opts.index(wxcam_date.value)
    new_idx = max(0, min(len(opts) - 1, idx + delta))
    if new_idx != idx:
        wxcam_date.value = opts[new_idx]
    else:
        wxcam_date.param.trigger("value")


def _go_wxcam_latest(_event=None):
    _refresh_wxcam_ql_options(preserve_current=True)
    opts = list(wxcam_date.options)
    if "Today (latest)" in opts:
        if wxcam_date.value == "Today (latest)":
            wxcam_date.param.trigger("value")
        else:
            wxcam_date.value = "Today (latest)"
    elif opts:
        if wxcam_date.value == opts[-1]:
            wxcam_date.param.trigger("value")
        else:
            wxcam_date.value = opts[-1]


wxcam_latest.on_click(_go_wxcam_latest)
wxcam_prev.on_click(lambda _e: _shift_wxcam_ql(-1))
wxcam_next.on_click(lambda _e: _shift_wxcam_ql(1))


class WxcamSelectionState(param.Parameterized):
    selected_hour_path = param.String(default="")


wxcam_calendar_state = WxcamSelectionState()


def _refresh_wxcam_latest_if_needed():
    # WXcam should not auto-refresh while selected. We refresh its options when
    # the user switches away and comes back, or when they explicitly change the
    # WXcam controls.
    return


_wxcam_ql_timer = _safe_periodic_callback(_refresh_wxcam_latest_if_needed, period=300_000, start=False)


@pn.depends(wxcam_date.param.value, wxcam_image_type.param.value)
def _wxcam_interactive_media(selected, selection):
    selection = selection or _cfg("wxcam")["default_top"]
    with _timed_perf("wxcam_interactive_render", instrument="wxcam", selection=selection, selected=selected) as perf:
        context = _wxcam_video_context(selection, selected)
        path = context.get("path")
        if not path:
            perf["status"] = "missing_option"
            return pn.pane.Markdown("No media available for this selection.")
        video_path = Path(str(path))
        perf["path"] = video_path
        if not video_path.exists():
            perf["status"] = "missing_file"
            return pn.pane.Markdown("No media available for this selection.")
        perf["status"] = "ok"
        return _build_wxcam_video_view(video_path, selection, selected or "", context)


@pn.depends(wxcam_date.param.value, wxcam_image_type.param.value)
def _wxcam_interactive_hour_strip(selected, selection):
    selection = selection or _cfg("wxcam")["default_top"]
    return pn.pane.HTML(
        _wxcam_hour_strip_markup(selection, selected),
        sizing_mode="stretch_width",
        margin=0,
    )


# WXcam now uses the Interactive Data Browser as its primary browser/player so
# the controls and playback state live in one place instead of competing with
# the science-quicklook flow used by the other instruments.
wxcam_interactive_browser = pn.Column(
    pn.Card(
        pn.Row(wxcam_image_type, wxcam_latest, sizing_mode="stretch_width", css_classes=["mobile-stack"]),
        pn.Row(wxcam_prev, wxcam_date, wxcam_next, sizing_mode="stretch_width", css_classes=["mobile-stack"]),
        title="",
        collapsible=False,
        sizing_mode="stretch_width",
        css_classes=["small-card", "wxcam-browser__toolbar"],
    ),
    _wxcam_interactive_hour_strip,
    _wxcam_interactive_media,
    sizing_mode="stretch_width",
    css_classes=["wxcam-browser"],
)

wxcam_image_type.param.watch(_on_wxcam_image_type_change, "value")


class AuroracamCameraCard(pn.reactive.ReactiveHTML):
    camera_id = param.String(default="")
    title = param.String(default="")
    ip = param.String(default="")
    image_url = param.String(default="")
    image_class = param.String(default="auroracam-card__hidden")
    placeholder = param.String(default="No image")
    placeholder_class = param.String(default="")
    meta = param.String(default="")
    filename = param.String(default="")
    selected_class = param.String(default="")
    empty_class = param.String(default="auroracam-card--empty")
    clicked = param.Event()

    _template = """
    <button id="card" type="button" class="auroracam-card auroracam-card--button {{ selected_class }} {{ empty_class }}" onclick="${script('select')}">
      <div class="auroracam-card__head">
        <div class="auroracam-card__title">{{ title }}</div>
        <div class="auroracam-card__ip">{{ ip }}</div>
      </div>
      <img class="auroracam-card__img {{ image_class }}" src="{{ image_url }}" alt="{{ title }} selected image"></img>
      <div class="auroracam-card__placeholder {{ placeholder_class }}">{{ placeholder }}</div>
      <div class="auroracam-card__meta">{{ meta }}</div>
      <div class="auroracam-card__file">{{ filename }}</div>
    </button>
    """

    _scripts = {
        "select": "data.clicked = true",
    }


def _auroracam_today_token() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _auroracam_day_options() -> list[str]:
    with _timed_perf("auroracam_day_options", instrument="auroracam", root=_auroracam_raw_root()) as perf:
        days = auroracam_available_days(_auroracam_raw_root())
        if not days:
            perf["status"] = "empty"
            return ["No images available"]
        today = _auroracam_today_token()
        options = [day for day in days if day != today]
        if today in days:
            options.append("Today (latest)")
        perf["day_count"] = len(days)
        perf["option_count"] = len(options)
        perf["status"] = "ok"
        return options or ["No images available"]


def _auroracam_selected_day(selected_day: str | None) -> str | None:
    if selected_day == "Today (latest)":
        return _auroracam_today_token()
    if selected_day and len(selected_day) == 10 and selected_day[4] == "-" and selected_day[7] == "-":
        return selected_day
    return None


def _auroracam_record_time(record) -> datetime | None:
    if record is None:
        return None
    try:
        return datetime.fromisoformat(record.time_utc).replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _auroracam_record_url(record) -> str:
    return _auroracam_media_url(Path(record.raw_path))


def _auroracam_time_token(value: str | None) -> str | None:
    if not value or value == "Latest":
        return None
    token = str(value).replace(" UTC", "").strip()
    if len(token) == 5 and token[2] == ":":
        return token
    return None


def _auroracam_time_options(selected_day: str | None) -> list[str]:
    day = _auroracam_selected_day(selected_day)
    if not day:
        return ["Latest"]
    times: set[str] = set()
    root = _auroracam_raw_root()
    for camera_id in AURORACAM_CAMERAS:
        for record in auroracam_day_records(root, camera_id, day):
            moment = _auroracam_record_time(record)
            if moment is not None:
                times.add(moment.strftime("%H:%M UTC"))
    options = sorted(times)
    options.append("Latest")
    return options or ["Latest"]


def _auroracam_record_for_time(camera_id: str, selected_day: str | None, selected_time: str | None):
    day = _auroracam_selected_day(selected_day)
    if camera_id not in AURORACAM_CAMERAS:
        return None
    token = _auroracam_time_token(selected_time)
    if day is None or token is None:
        return auroracam_latest_record(_auroracam_raw_root(), camera_id, day)
    records = auroracam_day_records(_auroracam_raw_root(), camera_id, day)
    if not records:
        return None
    target = datetime.strptime(f"{day} {token}", "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    return min(
        records,
        key=lambda record: (
            abs((_auroracam_record_time(record) or target) - target),
            -int(record.time_epoch_ns),
        ),
    )


def _auroracam_card_params(camera_id: str, record, selected_camera_id: str | None) -> dict[str, str]:
    spec = AURORACAM_CAMERAS[camera_id]
    values = {
        "camera_id": camera_id,
        "title": spec["label"],
        "ip": spec["ip"],
        "image_url": "",
        "image_class": "auroracam-card__hidden",
        "placeholder": "No image",
        "placeholder_class": "",
        "meta": "",
        "filename": "",
        "selected_class": "auroracam-card--selected" if camera_id == selected_camera_id else "",
        "empty_class": "auroracam-card--empty",
    }
    if record is None:
        return values
    moment = _auroracam_record_time(record)
    time_label = moment.strftime("%Y-%m-%d %H:%M UTC") if moment else record.time_utc
    age = _humanize_age(moment)
    try:
        image_url = _auroracam_record_url(record)
    except Exception:
        image_url = ""
    values.update(
        {
            "image_url": image_url,
            "image_class": "" if image_url else "auroracam-card__hidden",
            "placeholder": "" if image_url else "Image unavailable",
            "placeholder_class": "auroracam-card__hidden" if image_url else "",
            "meta": f"{time_label} | {age}",
            "filename": record.filename,
            "empty_class": "",
        }
    )
    return values


def _auroracam_grid(selected_day: str | None, selected_time: str | None, selected_camera_id: str | None) -> pn.Column:
    day = _auroracam_selected_day(selected_day)
    label = day or "Latest available"
    time_label = selected_time or "Latest"
    for camera_id, card in auroracam_cards.items():
        record = _auroracam_record_for_time(camera_id, selected_day, selected_time)
        card.param.update(**_auroracam_card_params(camera_id, record, selected_camera_id))
    return pn.Column(
        pn.pane.HTML(
            "<div class='auroracam-section__head'>"
            "<div class='auroracam-section__title'>AURORACam frames</div>"
            f"<div class='auroracam-section__meta'>{escape(label)} | {escape(time_label)}</div>"
            "</div>",
            sizing_mode="stretch_width",
            margin=0,
        ),
        pn.GridBox(*auroracam_cards.values(), ncols=4, sizing_mode="stretch_width", css_classes=["auroracam-grid"]),
        sizing_mode="stretch_width",
        css_classes=["auroracam-section"],
    )


def _auroracam_viewer_markup(camera_id: str | None, selected_day: str | None, selected_time: str | None) -> str:
    day = _auroracam_selected_day(selected_day)
    if camera_id not in AURORACAM_CAMERAS:
        return "<div class='auroracam-viewer auroracam-viewer--empty'>No camera selected.</div>"
    record = _auroracam_record_for_time(str(camera_id), selected_day, selected_time)
    spec = AURORACAM_CAMERAS[str(camera_id)]
    title = escape(spec["label"])
    if record is None:
        day_label = escape(day or "selected day")
        return (
            "<div class='auroracam-viewer auroracam-viewer--empty'>"
            f"<div class='auroracam-viewer__title'>{title}</div>"
            f"<div class='auroracam-viewer__placeholder'>No image for {day_label}</div>"
            "</div>"
        )
    moment = _auroracam_record_time(record)
    time_label = moment.strftime("%Y-%m-%d %H:%M UTC") if moment else record.time_utc
    image_url = _auroracam_record_url(record)
    return (
        "<div class='auroracam-viewer'>"
        "<div class='auroracam-viewer__meta'>"
        f"<div class='auroracam-viewer__title'>{title}</div>"
        f"<div class='auroracam-viewer__subtitle'>{escape(time_label)} | {escape(record.filename)}</div>"
        "</div>"
        f"<div class='auroracam-viewer__frame'><img class='auroracam-viewer__img' src='{image_url}' alt='{title} selected image'></div>"
        "</div>"
    )


auroracam_camera = pn.widgets.Select(
    name="Camera",
    options={spec["label"]: camera_id for camera_id, spec in AURORACAM_CAMERAS.items()},
    value=next(iter(AURORACAM_CAMERAS)),
)
_auroracam_date_options = _auroracam_day_options()
auroracam_date = pn.widgets.Select(name="Date", options=_auroracam_date_options)
if _auroracam_date_options:
    auroracam_date.value = _auroracam_date_options[-1]
# Reading every camera record to populate historic frame times is expensive and
# irrelevant until AURORACam is opened. Populate the full list lazily, while
# retaining a pending share-link selection for the first camera activation.
_AURORACAM_PENDING_TIME_QUERY: str | None = None
auroracam_time = pn.widgets.Select(name="Time (UTC)", options=["Latest"], value="Latest")
auroracam_latest = pn.widgets.Button(name="Latest", button_type="primary")
auroracam_prev = pn.widgets.Button(name="Previous Day", button_type="default")
auroracam_next = pn.widgets.Button(name="Next Day", button_type="default")
auroracam_prev_time = pn.widgets.Button(name="Previous Frame", button_type="default")
auroracam_next_time = pn.widgets.Button(name="Next Frame", button_type="default")
auroracam_cards = {
    camera_id: AuroracamCameraCard(
        camera_id=camera_id,
        title=spec["label"],
        ip=spec["ip"],
        sizing_mode="stretch_width",
    )
    for camera_id, spec in AURORACAM_CAMERAS.items()
}
auroracam_detail = pn.pane.HTML(
    _auroracam_viewer_markup(auroracam_camera.value, auroracam_date.value, auroracam_time.value),
    sizing_mode="stretch_width",
    margin=0,
)
mobile_auroracam_detail = pn.pane.HTML(
    auroracam_detail.object,
    sizing_mode="stretch_width",
    margin=0,
)


def _refresh_auroracam_detail(_event=None) -> None:
    markup = _auroracam_viewer_markup(
        auroracam_camera.value,
        auroracam_date.value,
        auroracam_time.value,
    )
    auroracam_detail.object = markup
    mobile_auroracam_detail.object = markup


def _select_auroracam_camera(camera_id: str) -> None:
    if camera_id not in AURORACAM_CAMERAS:
        return
    if auroracam_camera.value == camera_id:
        auroracam_camera.param.trigger("value")
    else:
        auroracam_camera.value = camera_id


for _auroracam_card_id, _auroracam_card in auroracam_cards.items():
    _auroracam_card.param.watch(
        lambda _event, camera_id=_auroracam_card_id: _select_auroracam_camera(camera_id),
        "clicked",
    )

for _auroracam_control in (auroracam_camera, auroracam_date, auroracam_time):
    _auroracam_control.param.watch(_refresh_auroracam_detail, "value")


def _refresh_auroracam_time_options(preserve_current: bool = True) -> None:
    global _AURORACAM_PENDING_TIME_QUERY
    current = auroracam_time.value if preserve_current else None
    opts = _auroracam_time_options(auroracam_date.value)
    auroracam_time.options = opts
    if _AURORACAM_PENDING_TIME_QUERY in opts:
        auroracam_time.value = _AURORACAM_PENDING_TIME_QUERY
        _AURORACAM_PENDING_TIME_QUERY = None
    elif not opts:
        auroracam_time.value = None
    elif preserve_current and current in opts:
        auroracam_time.value = current
    else:
        auroracam_time.value = opts[-1]


def _refresh_auroracam_options(preserve_current: bool = True):
    current = auroracam_date.value if preserve_current else None
    opts = _auroracam_day_options()
    auroracam_date.options = opts
    if not opts:
        auroracam_date.value = None
    elif preserve_current and current in opts:
        auroracam_date.value = current
    else:
        auroracam_date.value = opts[-1]
    _refresh_auroracam_time_options(preserve_current=preserve_current)


def _shift_auroracam_day(delta: int):
    _refresh_auroracam_options(preserve_current=True)
    opts = list(auroracam_date.options)
    if not opts or auroracam_date.value not in opts:
        return
    idx = opts.index(auroracam_date.value)
    new_idx = max(0, min(len(opts) - 1, idx + delta))
    if new_idx != idx:
        auroracam_date.value = opts[new_idx]
    else:
        auroracam_date.param.trigger("value")


def _shift_auroracam_time(delta: int):
    _refresh_auroracam_time_options(preserve_current=True)
    opts = list(auroracam_time.options)
    if not opts or auroracam_time.value not in opts:
        return
    idx = opts.index(auroracam_time.value)
    new_idx = max(0, min(len(opts) - 1, idx + delta))
    if new_idx != idx:
        auroracam_time.value = opts[new_idx]
    else:
        auroracam_time.param.trigger("value")


def _go_auroracam_latest(_event=None):
    _refresh_auroracam_options(preserve_current=True)
    opts = list(auroracam_date.options)
    target = "Today (latest)" if "Today (latest)" in opts else (opts[-1] if opts else None)
    if target is None:
        return
    if auroracam_date.value == target:
        auroracam_date.param.trigger("value")
    else:
        auroracam_date.value = target
    if "Latest" in list(auroracam_time.options):
        if auroracam_time.value == "Latest":
            auroracam_time.param.trigger("value")
        else:
            auroracam_time.value = "Latest"


auroracam_latest.on_click(_go_auroracam_latest)
auroracam_prev.on_click(lambda _e: _shift_auroracam_day(-1))
auroracam_next.on_click(lambda _e: _shift_auroracam_day(1))
auroracam_prev_time.on_click(lambda _e: _shift_auroracam_time(-1))
auroracam_next_time.on_click(lambda _e: _shift_auroracam_time(1))


def _on_auroracam_date_change(_event) -> None:
    _refresh_auroracam_time_options(preserve_current=False)


auroracam_date.param.watch(_on_auroracam_date_change, "value")


def _refresh_auroracam_latest_if_needed() -> None:
    """Refresh live camera selections without changing an operator's history view."""
    if ACTIVE_TAB_SLUG != "auroracam":
        return
    if auroracam_date.value != "Today (latest)" or auroracam_time.value != "Latest":
        return
    _refresh_auroracam_options(preserve_current=True)
    auroracam_time.param.trigger("value")


@pn.depends(auroracam_date.param.value, auroracam_time.param.value, auroracam_camera.param.value)
def _auroracam_browser(selected_day, selected_time, camera_id):
    with _timed_perf(
        "auroracam_render",
        instrument="auroracam",
        selected_day=selected_day,
        selected_time=selected_time,
        camera_id=camera_id,
    ) as perf:
        root = _auroracam_raw_root()
        perf["root"] = root
        if not root.exists():
            perf["status"] = "missing_root"
            return pn.pane.HTML(
                "<div class='auroracam-empty'>AURORACam image root is not available.</div>",
                sizing_mode="stretch_width",
                margin=0,
            )
        perf["status"] = "ok"
        return pn.Column(
            _auroracam_grid(selected_day, selected_time, camera_id),
            sizing_mode="stretch_width",
            css_classes=["auroracam-browser"],
        )


def _update_wxcam_view(start, end, top_name: str, bottom_name: str, request_id: int | None = None) -> bool:
    return _publish_panel_if_current(wxcam_interactive_browser, request_id)


def _update_hatpro_view(
    start,
    end,
    bottom_val,
    top_val,
    lymin,
    lymax,
    iymin,
    iymax,
    rymin,
    rymax,
    request_id: int | None = None,
    render_quality: str = "full",
    cache_key: tuple[object, ...] | None = None,
):
    """Custom renderer for HATPRO radiometer: split LWP/IWV and IRR; T_PROF heatmap."""
    print(f"[hatpro] render window {start} -> {end}")
    with _timed_perf(
        "hatpro_render",
        instrument="Scanning Microwave Radiometer",
        start=start,
        end=end,
        render_quality=render_quality,
    ) as perf:
        if not _render_request_active(request_id):
            perf["status"] = "stale_before_start"
            return False
        bottom = max(float(bottom_val), 0.0)
        top = max(float(top_val), bottom + 100.0)
        ds = open_window(
            start,
            end,
            bottom_m=bottom,
            top_m=top,
            instrument="Scanning Microwave Radiometer",
            render_quality=render_quality,
        )
        cfg = _cfg("Scanning Microwave Radiometer")
        times = pd.to_datetime(ds["time"].values) if "time" in ds else None
        perf["time_count"] = 0 if times is None else int(len(times))
        if times is None or len(times) == 0:
            perf["status"] = "no_data"
            fig = _empty_interactive_figure(
                "Scanning Microwave Radiometer",
                "No samples were found for this selected time/range window.",
                start=start,
                end=end,
                detail="Try widening the time range or range limits, or check Operations for source freshness.",
            )
            return _publish_plot_if_current(fig, "Scanning Microwave Radiometer", request_id, cache_key=cache_key)

        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{}]],
            row_heights=[0.25, 0.25, 0.5],
            subplot_titles=("LWP / IWV", "IRR / SURF_T", cfg["vars"]["T_PROF"]["label"]),
        )
        # Color the subplot titles to match their traces.
        if len(fig.layout.annotations) >= 1:
            fig.layout.annotations[0].update(text='<span style="color:#1f77b4">LWP</span> / <span style="color:#2ca02c">IWV</span>', font=dict(size=14))
        if len(fig.layout.annotations) >= 2:
            fig.layout.annotations[1].update(
                text='<span style="color:#d62728">IRR</span> / <span style="color:#9467bd">SURF_T</span>',
                font=dict(size=14),
            )
        if len(fig.layout.annotations) >= 3:
            fig.layout.annotations[2].update(font=dict(size=14))

        # Row 1: LWP (left), IWV (right, kg/m²)
        if "LWP" in ds:
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=np.asarray(ds["LWP"]),
                    mode="lines",
                    name="LWP (g/m²)",
                    line=dict(color="#1f77b4", width=2),
                ),
                row=1,
                col=1,
                secondary_y=False,
            )
        if "IWV" in ds:
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=np.asarray(ds["IWV"]),
                    mode="lines",
                    name="IWV (kg/m²)",
                    line=dict(color="#2ca02c", width=2, dash="dot"),
                ),
                row=1,
                col=1,
                secondary_y=True,
            )

        # Row 2: IRR only
        if "IRR_Map" in ds:
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=np.asarray(ds["IRR_Map"]),
                    mode="lines",
                    name="IRR (°C)",
                    line=dict(color="#d62728", width=2),
                ),
                row=2,
                col=1,
            )
        if "SURF_T" in ds:
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=np.asarray(ds["SURF_T"]) - 273.15,  # convert K to °C for display
                    mode="lines",
                    name="SURF_T (°C)",
                    line=dict(color="#9467bd", width=2, dash="dot"),
                ),
                row=2,
                col=1,
            )

        # Row 3: Temperature profile heatmap + contours
        if "T_PROF" in ds:
            heights = ds["range"].values if "range" in ds else np.arange(ds["T_PROF"].shape[1])
            temps = np.array(ds["T_PROF"].transpose("range", "time"))
            profile_times, temps = insert_time_gap_breaks(times, temps, time_axis=1)
            profile_times = pd.to_datetime(profile_times)
            vmin, vmax = cfg["vars"]["T_PROF"]["clim"]
            fig.add_trace(
                go.Heatmap(
                    x=profile_times,
                    y=heights,
                    z=temps,
                    connectgaps=False,
                    hoverongaps=False,
                    zmin=vmin,
                    zmax=vmax,
                    coloraxis="coloraxis",
                    showscale=True,
                ),
                row=3,
                col=1,
            )
            fig.add_trace(
                go.Contour(
                    x=profile_times,
                    y=heights,
                    z=temps,
                    connectgaps=False,
                    showscale=False,
                    contours=dict(coloring="none", showlabels=True, labelfont=dict(color="white", size=10)),
                    line=dict(color="white", width=1),
                    hoverinfo="skip",
                ),
                row=3,
                col=1,
            )

        tickvals = []
        ticktext = []
        noon_annots = []
        if start and end:
            start_ts = pd.Timestamp(start)
            end_ts = pd.Timestamp(end)
            duration = end_ts - start_ts
            freq = "2h" if duration > pd.Timedelta(hours=24) else "h"
            hours = pd.date_range(start=start_ts.floor("h"), end=end_ts.ceil("h"), freq=freq)
            for t in hours:
                tickvals.append(t.to_pydatetime())
                ticktext.append(t.strftime("%H:%M"))
                if t.hour == 12:
                    noon_annots.append(
                        dict(
                            x=t.to_pydatetime(),
                            y=-0.06,
                            xref="x",
                            yref="paper",
                            text=t.strftime("%Y-%m-%d"),
                            showarrow=False,
                            xanchor="center",
                            yanchor="top",
                            font=dict(size=14, color=THEME_TEXT),
                        )
                    )

        # x-axes
        for row in (1, 2, 3):
            fig.update_xaxes(
                tickmode="array",
                tickvals=tickvals,
                ticktext=ticktext,
                range=[start_ts.to_pydatetime(), end_ts.to_pydatetime()] if start and end else None,
                tickangle=-45,
                showgrid=True,
                gridcolor=THEME_GRID,
                linecolor=THEME_LINE,
                tickfont=dict(color=THEME_TEXT, size=12),
                title_font=dict(color=THEME_TEXT, size=12),
                row=row,
                col=1,
            )
        fig.update_xaxes(title_text="Date and Time (UTC)", title_standoff=40, row=3, col=1)

        # y-axes
        fig.update_yaxes(title_text="LWP (g/m²)", range=[float(lymin), float(lymax)], row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text="IWV (kg/m²)", range=[float(iymin), float(iymax)], row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text="IRR / SURF_T (°C)", range=[float(rymin), float(rymax)], row=2, col=1)
        fig.update_yaxes(range=[bottom, top], title_text="Range (m)", row=3, col=1, showgrid=True, gridcolor=THEME_GRID, linecolor=THEME_LINE)

        fig.update_layout(
            showlegend=False,
            height=max(600, int(pn.state.viewport_height * 0.68)) if hasattr(pn.state, "viewport_height") else 760,
            margin=dict(l=60, r=80, t=30, b=110),
            coloraxis=dict(
                colorscale=cfg["vars"]["T_PROF"]["colorscale"],
                cmin=cfg["vars"]["T_PROF"]["clim"][0],
                cmax=cfg["vars"]["T_PROF"]["clim"][1],
                colorbar=dict(title=dict(text="T (K)", side="right"), x=1.02, y=0.18, len=0.3, tickfont=dict(color=THEME_TEXT, size=9)),
            ),
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(color=THEME_TEXT, size=13),
            annotations=tuple(list(fig.layout.annotations) + noon_annots),
        )
        perf["status"] = "ok"
        perf["trace_count"] = len(fig.data)
        return _publish_plot_if_current(fig, "Scanning Microwave Radiometer", request_id, cache_key=cache_key)


def _prepare_stacked_timeseries_figure(
    instrument: str,
    start,
    end,
    render_quality: str,
    *,
    power_section: str | None = None,
) -> tuple[go.Figure, dict[str, object]]:
    """Prepare a summary figure without touching Panel or Bokeh state."""
    metrics: dict[str, object] = {"source_instruments": list(summary_source_instruments(instrument))}
    source_instruments = tuple(metrics["source_instruments"])
    is_power = instrument == "power"
    power_section = power_section or "current"
    lock = _POWER_PREPARATION_LOCK if is_power else None

    def prepare() -> tuple[go.Figure, dict[str, object]]:
        power_display_summary_available = is_power and _power_display_summary_path().exists()
        power_display_energy_available = is_power and _power_display_energy_path().exists()
        source_render_quality = (
            "summary_full_time"
            if is_power and not (power_display_summary_available or power_display_energy_available)
            else render_quality
        )
        metrics.update(
            source_render_quality=source_render_quality,
            power_display_summary_available=bool(power_display_summary_available),
            power_display_energy_available=bool(power_display_energy_available),
        )
        context_start = _summary_context_start(start, instrument)
        metrics["context_start"] = context_start
        source_open_started = perf_counter()
        source_windows: list[xr.Dataset | None] = []
        if is_power:
            display_summary_window = _open_power_display_summary_window(start, end, section=power_section)
            if display_summary_window is not None:
                source_windows = [display_summary_window]
                metrics["power_display_summary"] = "used"
                metrics["power_display_energy"] = "embedded"
            else:
                metrics["power_display_summary"] = "missing"
        if not source_windows:
            source_windows = [
                open_window(context_start, end, instrument=source_inst, render_quality=source_render_quality)
                for source_inst in source_instruments
            ]
            if is_power:
                display_window = _open_power_display_energy_window(start, end)
                if display_window is not None:
                    source_windows.append(display_window)
                    metrics["power_display_energy"] = "used"
                else:
                    metrics["power_display_energy"] = "missing"
        metrics["source_open_ms"] = round((perf_counter() - source_open_started) * 1000, 3)
        source_metrics = [_dataset_window_metrics(window) for window in source_windows]
        metrics["source_window_count"] = len(source_metrics)
        metrics["source_window_time_counts"] = [item["time_count"] for item in source_metrics]
        metrics["source_window_var_counts"] = [item["var_count"] for item in source_metrics]
        combine_started = perf_counter()
        ds = combine_summary_datasets(instrument, *source_windows)
        metrics["combine_ms"] = round((perf_counter() - combine_started) * 1000, 3)
        metrics["combined_var_count"] = 0 if ds is None else len(ds.data_vars)
        if is_power and ds is not None:
            display_start = _as_naive_utc_datetime(start)
            display_end = _as_naive_utc_datetime(end)
            if display_start is not None:
                ds.attrs[SUMMARY_DISPLAY_START_ATTR] = display_start.isoformat()
            if display_end is not None:
                ds.attrs[SUMMARY_DISPLAY_END_ATTR] = display_end.isoformat()
        times = pd.to_datetime(ds["time"].values) if ds is not None and "time" in ds else None
        metrics["time_count"] = 0 if times is None else int(len(times))
        if times is None or len(times) == 0:
            metrics["status"] = "no_data"
            return _empty_interactive_figure(
                instrument,
                "No samples were found for this selected time window.",
                start=start,
                end=end,
                detail="Try a wider time range or check the Operations Dashboard source-freshness cards.",
            ), metrics
        try:
            max_time_samples = _stacked_interactive_max_time_samples(instrument, render_quality)
            metrics["max_time_samples"] = max_time_samples
            metrics["plot_density_mode"] = "per_trace_display_downsampled"
            fig_started = perf_counter()
            fig = build_summary_plotly(
                ds,
                instrument,
                title=display_name(instrument),
                max_time_samples=max_time_samples,
                x_limits=(start, end),
                panel_groups=(
                    {"observed"}
                    if is_power and power_section == "current"
                    else {"forecast_24h", "forecast_96h", "verification"}
                    if is_power
                    else None
                ),
            )
        except ValueError as exc:
            metrics["status"] = "no_data"
            return _empty_interactive_figure(
                instrument,
                "No plottable variables are available for this window.",
                start=start,
                end=end,
                detail=str(exc),
            ), metrics
        metrics["figure_build_ms"] = round((perf_counter() - fig_started) * 1000, 3)
        metrics.update(_figure_metrics(fig))
        metrics["status"] = "ok"
        metrics["trace_count"] = len(fig.data)
        return fig, metrics

    if lock is None:
        return prepare()
    with lock:
        return prepare()


def _update_stacked_timeseries_view(
    instrument: str,
    start,
    end,
    request_id: int | None = None,
    render_quality: str = "full",
    cache_key: tuple[object, ...] | None = None,
):
    """Render a 1D summary instrument with fixed multi-panel layouts."""
    print(f"[{instrument}] render window {start} -> {end}")
    with _timed_perf(
        "stacked_timeseries_render",
        instrument=instrument,
        start=start,
        end=end,
        render_quality=render_quality,
    ) as perf:
        if not _render_request_active(request_id):
            perf["status"] = "stale_before_start"
            return False
        fig, metrics = _prepare_stacked_timeseries_figure(
            instrument,
            start,
            end,
            render_quality,
            power_section=power_view_select.value if instrument == "power" else None,
        )
        perf.update(metrics)
        return _publish_plot_if_current(fig, instrument, request_id, cache_key=cache_key)


async def _prepare_power_view_async(
    start,
    end,
    request_id: int,
    render_quality: str,
    cache_key: tuple[object, ...],
    power_section: str,
    doc,
) -> None:
    """Build a custom Power figure off the event loop and publish it safely."""
    started = perf_counter()
    try:
        loop = asyncio.get_running_loop()
        prepare = partial(
            _prepare_stacked_timeseries_figure,
            "power",
            start,
            end,
            render_quality,
            power_section=power_section,
        )
        fig, metrics = await loop.run_in_executor(_BACKGROUND_PREPARATION_EXECUTOR, prepare)
    except asyncio.CancelledError:
        _perf_log(
            "power_background_prepare",
            instrument="power",
            request_id=request_id,
            status="cancelled",
            duration_ms=round((perf_counter() - started) * 1000, 3),
        )
        raise
    except Exception as exc:
        _perf_log(
            "power_background_prepare",
            instrument="power",
            request_id=request_id,
            status="error",
            error=str(exc),
            duration_ms=round((perf_counter() - started) * 1000, 3),
        )

        def publish_error() -> None:
            if _render_request_active(request_id):
                fallback = _empty_interactive_figure(
                    "power",
                    "The Power view could not be prepared.",
                    start=start,
                    end=end,
                    detail="The previous chart is retained while the next refresh is attempted.",
                )
                _publish_plot_if_current(fallback, "power", request_id, cache_key=cache_key)
                _clear_interactive_loading()

        doc.add_next_tick_callback(publish_error)
        return
    finally:
        _BACKGROUND_RENDER_TASKS.pop(request_id, None)

    log_metrics = {key: value for key, value in metrics.items() if key != "status"}
    _perf_log(
        "power_background_prepare",
        instrument="power",
        request_id=request_id,
        status=str(metrics.get("status", "ok")),
        duration_ms=round((perf_counter() - started) * 1000, 3),
        **log_metrics,
    )

    def publish() -> None:
        if not _render_request_active(request_id):
            return
        _publish_plot_if_current(fig, "power", request_id, cache_key=cache_key)
        _clear_interactive_loading()

    doc.add_next_tick_callback(publish)


def _start_background_power_prepare(
    start,
    end,
    request_id: int | None,
    render_quality: str,
    cache_key: tuple[object, ...],
) -> bool:
    """Schedule Power preparation when a live Panel event loop is available."""
    if request_id is None or pn.state.curdoc is None:
        return False
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return False
    task = loop.create_task(
        _prepare_power_view_async(
            start,
            end,
            request_id,
            render_quality,
            cache_key,
            power_view_select.value,
            pn.state.curdoc,
        )
    )
    _BACKGROUND_RENDER_TASKS[request_id] = task
    return True


def _render_interactive_view(
    start,
    end,
    bottom_val,
    top_val,
    var1_name,
    var2_name,
    bmin,
    bmax,
    lmin,
    lmax,
    lymin,
    lymax,
    iymin,
    iymax,
    rymin,
    rymax,
    instrument,
    request_id: int | None = None,
    render_quality: str = "full",
):
    """Render the current interactive view, dropping stale work if a newer request arrives."""
    if _instrument_guard:
        return
    global CURRENT_INSTRUMENT
    if instrument != CURRENT_INSTRUMENT:
        CURRENT_INSTRUMENT = instrument
    start, end, cache_window_mode = _canonical_interactive_window(start, end, instrument)
    print(f"[render-view] instrument={instrument} quality={render_quality} request={request_id}")
    cache_key = _interactive_render_cache_key(
        start, end, bottom_val, top_val, var1_name, var2_name, bmin, bmax,
        lmin, lmax, lymin, lymax, iymin, iymax, rymin, rymax, instrument,
        render_quality=render_quality,
    )
    if instrument == "power" and _start_background_power_prepare(
        start,
        end,
        request_id,
        render_quality,
        cache_key,
    ):
        _perf_log(
            "power_background_prepare_scheduled",
            instrument="power",
            request_id=request_id,
            render_quality=render_quality,
        )
        return
    try:
      with _timed_perf(
          "interactive_view_update",
          instrument=instrument,
          start=start,
          end=end,
          render_quality=render_quality,
          request_id=request_id,
      ) as perf:
        if not _render_request_active(request_id):
            perf["status"] = "stale_before_start"
            return
        perf["cache_window_mode"] = cache_window_mode
        perf["top_var"] = var1_name
        perf["bottom_var"] = var2_name
        perf["bottom_m"] = bottom_val
        perf["top_m"] = top_val
        if instrument == "Scanning Microwave Radiometer":
            perf["view_type"] = "hatpro"
            published = _update_hatpro_view(
                start,
                end,
                bottom_val,
                top_val,
                lymin,
                lymax,
                iymin,
                iymax,
                rymin,
                rymax,
                request_id=request_id,
                render_quality=render_quality,
                cache_key=cache_key,
            )
            if not published:
                perf["status"] = "stale"
                return
            perf["status"] = "ok"
        elif _is_wxcam_instrument(instrument):
            perf["view_type"] = "wxcam"
            published = _update_wxcam_view(start, end, var1_name, var2_name, request_id=request_id)
            if not published:
                perf["status"] = "stale"
                return
            perf["status"] = "ok"
        elif _is_stacked_timeseries_instrument(instrument):
            perf["view_type"] = "stacked_timeseries"
            published = _update_stacked_timeseries_view(
                instrument,
                start,
                end,
                request_id=request_id,
                render_quality=render_quality,
                cache_key=cache_key,
            )
            if not published:
                perf["status"] = "stale"
                return
            perf["status"] = "ok"
        else:
            perf["view_type"] = "heatmap"
            bottom = max(float(bottom_val), 0.0)
            top = max(float(top_val), bottom + 100.0)
            ds = open_window(start, end, bottom_m=bottom, top_m=top, instrument=instrument, render_quality=render_quality)
            cfg = _cfg()
            vars_cfg = cfg["vars"]
            var1 = vars_cfg.get(var1_name)
            var2 = vars_cfg.get(var2_name)
            bg = "white"
            fg = THEME_TEXT
            grid = THEME_GRID
            perf["time_count"] = int(ds.sizes.get("time", 0)) if ds is not None else 0
            perf["range_count"] = int(ds.sizes.get("range", 0)) if ds is not None else 0
            if ds is None or not ds.data_vars:
                perf["status"] = "no_data"
                fig = _empty_interactive_figure(
                    instrument,
                    "No samples were found for this selected time/range window.",
                    start=start,
                    end=end,
                    detail="Try widening the time range or range limits, or check Operations for source freshness.",
                )
                if not _publish_plot_if_current(fig, instrument, request_id, cache_key=cache_key):
                    perf["status"] = "stale"
                    return
            else:
                if var1 and var1.get("log"):
                    b_cmin = np.log10(bmin)
                    b_cmax = np.log10(bmax)
                    b_tickvals = list(range(int(np.floor(b_cmin)), int(np.ceil(b_cmax)) + 1))
                    sup = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
                    b_ticktext = [f"10{str(v).translate(sup)}" for v in b_tickvals]
                else:
                    b_cmin, b_cmax = bmin, bmax
                    b_tickvals = None
                    b_ticktext = None
                fig = make_subplots(
                    rows=2,
                    cols=1,
                    shared_xaxes=True,
                    shared_yaxes=False,
                    vertical_spacing=0.08,
                    subplot_titles=(var1["label"] if var1 else "", var2["label"] if var2 else ""),
                )
                if var1 and var1_name in ds:
                    fig.add_trace(_make_plot(ds, var1_name, (bmin, bmax), var1.get("log", False), coloraxis="coloraxis"), row=1, col=1)
                if var2 and var2_name in ds:
                    if var1_name == "beta_att" and var2_name == "linear_depol_ratio" and "beta_att" in ds:
                        times = pd.to_datetime(ds["time"].values)
                        heights = ds["range"].values
                        ldr = np.array(ds[var2_name].transpose("range", "time"))
                        beta_vals = np.array(ds["beta_att"].transpose("range", "time"))
                        ldr = np.where((ldr >= 0.0) & (ldr <= 1.0), ldr, np.nan)
                        mask_threshold = 10 ** -6.5
                        ldr = np.where(beta_vals >= mask_threshold, ldr, np.nan)
                        times, ldr = insert_time_gap_breaks(times, ldr, time_axis=1)
                        fig.add_trace(
                            go.Heatmap(
                                x=pd.to_datetime(times),
                                y=heights,
                                z=ldr,
                                connectgaps=False,
                                hoverongaps=False,
                                zmin=lmin,
                                zmax=lmax,
                                coloraxis="coloraxis2",
                                showscale=False,
                            ),
                            row=2,
                            col=1,
                        )
                    else:
                        times = pd.to_datetime(ds["time"].values)
                        heights = ds["range"].values
                        ldr = np.array(ds[var2_name].transpose("range", "time"))
                        times, ldr = insert_time_gap_breaks(times, ldr, time_axis=1)
                        fig.add_trace(
                            go.Heatmap(
                                x=pd.to_datetime(times),
                                y=heights,
                                z=ldr,
                                connectgaps=False,
                                hoverongaps=False,
                                zmin=lmin,
                                zmax=lmax,
                                coloraxis="coloraxis2",
                                showscale=False,
                            ),
                            row=2,
                            col=1,
                        )
                fig.update_yaxes(range=[bottom, top], title_text="Range (m)", row=1, col=1)
                fig.update_yaxes(range=[bottom, top], title_text="Range (m)", row=2, col=1)
                tickvals = []
                ticktext = []
                noon_annots = []
                if start and end:
                    start_ts = pd.Timestamp(start)
                    end_ts = pd.Timestamp(end)
                    duration = end_ts - start_ts
                    freq = "2h" if duration > pd.Timedelta(hours=24) else "h"
                    hours = pd.date_range(start=start_ts.floor("h"), end=end_ts.ceil("h"), freq=freq)
                    for t in hours:
                        tickvals.append(t.to_pydatetime())
                        ticktext.append(t.strftime("%H:%M"))
                        if t.hour == 12:
                            noon_annots.append(
                                dict(
                                    x=t.to_pydatetime(),
                                    y=-0.06,
                                    xref="x",
                                    yref="paper",
                                    text=t.strftime("%Y-%m-%d"),
                                    showarrow=False,
                                    xanchor="center",
                                    yanchor="top",
                                    font=dict(size=14, color=fg),
                                )
                            )
                fig.update_xaxes(
                    title_text="Date and Time (UTC)",
                    title_standoff=50,
                    tickmode="array",
                    tickvals=tickvals,
                    ticktext=ticktext,
                    range=[start_ts.to_pydatetime(), end_ts.to_pydatetime()] if start and end else None,
                    tickangle=-45,
                    showgrid=True,
                    gridcolor=grid,
                    linecolor=THEME_LINE,
                    tickfont=dict(color=fg, size=12),
                    title_font=dict(color=fg, size=12),
                    row=2,
                    col=1,
                )
                fig.update_xaxes(
                    tickmode="array",
                    tickvals=tickvals,
                    ticktext=ticktext,
                    range=[start_ts.to_pydatetime(), end_ts.to_pydatetime()] if start and end else None,
                    tickangle=-45,
                    showgrid=True,
                    gridcolor=grid,
                    linecolor=THEME_LINE,
                    tickfont=dict(color=fg, size=12),
                    title_font=dict(color=fg, size=12),
                    row=1,
                    col=1,
                )
                fig.update_yaxes(showgrid=True, gridcolor=grid, linecolor=THEME_LINE, tickfont=dict(color=fg, size=12), title_font=dict(color=fg, size=12), row=1, col=1)
                fig.update_yaxes(showgrid=True, gridcolor=grid, linecolor=THEME_LINE, tickfont=dict(color=fg, size=12), title_font=dict(color=fg, size=12), row=2, col=1)
                fig.update_yaxes(matches="y", row=2, col=1)
                fig.update_layout(
                    height=max(540, int(pn.state.viewport_height * 0.56)) if hasattr(pn.state, "viewport_height") else 680,
                    margin=dict(l=50, r=70, t=30, b=90),
                    coloraxis=dict(
                        colorscale=var1["colorscale"] if var1 else "Cividis",
                        cmin=b_cmin,
                        cmax=b_cmax,
                        colorbar=dict(
                            title=dict(text=var1["label"] if var1 else "", side="right"),
                            x=1.04,
                            y=0.77,
                            len=0.35,
                            tickvals=b_tickvals,
                            ticktext=b_ticktext,
                            tickfont=dict(color=fg, size=9),
                        ),
                    ),
                    coloraxis2=dict(
                        colorscale=var2["colorscale"] if var2 else "Viridis",
                        cmin=lmin,
                        cmax=lmax,
                        colorbar=dict(title=dict(text=var2["label"] if var2 else "", side="right"), x=1.04, y=0.27, len=0.35, tickfont=dict(color=fg, size=9)),
                    ),
                    paper_bgcolor=bg,
                    plot_bgcolor=bg,
                    font=dict(color=fg, size=13),
                    annotations=tuple(list(fig.layout.annotations) + noon_annots),
                )
                perf["status"] = "ok"
                perf["trace_count"] = len(fig.data)
                if not _publish_plot_if_current(fig, instrument, request_id, cache_key=cache_key):
                    perf["status"] = "stale"
                    return
        if (
            render_quality == "coarse"
            and _interactive_supports_refine(instrument)
            and _render_request_active(request_id)
            and perf.get("status") == "ok"
        ):
            _set_interactive_loading(instrument, "Refining higher-detail data for this view.", phase="refining", visible=True)
            _schedule_next_tick(
                lambda: _render_interactive_view(
                    start,
                    end,
                    bottom_val,
                    top_val,
                    var1_name,
                    var2_name,
                    bmin,
                    bmax,
                    lmin,
                    lmax,
                    lymin,
                    lymax,
                    iymin,
                    iymax,
                    rymin,
                    rymax,
                    instrument,
                    request_id=request_id,
                    render_quality="full",
                )
            )
        elif _render_request_active(request_id):
            _clear_interactive_loading()
    finally:
        _IN_FLIGHT_INTERACTIVE_RENDER_CACHE_KEYS.discard(cache_key)


def _start_interactive_render(
    start,
    end,
    bottom_val,
    top_val,
    var1_name,
    var2_name,
    bmin,
    bmax,
    lmin,
    lmax,
    lymin,
    lymax,
    iymin,
    iymax,
    rymin,
    rymax,
    instrument,
):
    if _instrument_guard:
        return
    global CURRENT_INSTRUMENT
    if instrument != CURRENT_INSTRUMENT:
        CURRENT_INSTRUMENT = instrument
    _capture_current_instrument_state(instrument)
    start, end, _cache_window_mode = _canonical_interactive_window(start, end, instrument)
    request_id = _begin_render_request()
    final_quality = _interactive_final_quality(instrument)
    first_quality = _interactive_initial_quality(instrument)
    full_cache_key = _interactive_render_cache_key(
        start,
        end,
        bottom_val,
        top_val,
        var1_name,
        var2_name,
        bmin,
        bmax,
        lmin,
        lmax,
        lymin,
        lymax,
        iymin,
        iymax,
        rymin,
        rymax,
        instrument,
        render_quality=final_quality,
    )
    # A fresh latest-view prewarm is immutable JSON. Read it in a worker thread
    # so a phone or browser session can keep handling controls while the file is
    # parsed. The only Panel mutation happens in the document callback above.
    prewarm_path = _prewarmed_interactive_path(instrument)
    cached_prewarm = (
        full_cache_key not in _INTERACTIVE_RENDER_CACHE
        and _cache_key_targets_latest_prewarm(full_cache_key, instrument)
        and prewarm_path.is_file()
        and pn.state.curdoc is not None
    )
    if cached_prewarm:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            _set_interactive_loading(instrument, "Loading the current cached view.", phase="loading", visible=True)
            if not _show_cached_quicklook_placeholder(instrument, start=start, end=end):
                _show_interactive_placeholder(instrument, "Loading the current cached view.")
            _IN_FLIGHT_INTERACTIVE_RENDER_CACHE_KEYS.add(full_cache_key)
            render_args = (
                start, end, bottom_val, top_val, var1_name, var2_name, bmin, bmax,
                lmin, lmax, lymin, lymax, iymin, iymax, rymin, rymax, instrument,
            )
            prewarm_task = loop.create_task(
                _load_prewarmed_interactive_figure_async(
                    prewarm_path,
                    full_cache_key,
                    instrument,
                    request_id,
                    pn.state.curdoc,
                    render_args,
                )
            )
            _BACKGROUND_RENDER_TASKS[request_id] = prewarm_task

            def finish_prewarm_task(task: asyncio.Task) -> None:
                if _BACKGROUND_RENDER_TASKS.get(request_id) is task:
                    _BACKGROUND_RENDER_TASKS.pop(request_id, None)
                if task.cancelled():
                    _IN_FLIGHT_INTERACTIVE_RENDER_CACHE_KEYS.discard(full_cache_key)

            prewarm_task.add_done_callback(finish_prewarm_task)
            return
    if _restore_exact_interactive_cache(full_cache_key, instrument):
        _clear_interactive_loading()
        return
    if full_cache_key in _IN_FLIGHT_INTERACTIVE_RENDER_CACHE_KEYS:
        _perf_log("interactive_render_deduplicated", instrument=instrument)
        return
    first_cache_key = _interactive_render_cache_key(
        start,
        end,
        bottom_val,
        top_val,
        var1_name,
        var2_name,
        bmin,
        bmax,
        lmin,
        lmax,
        lymin,
        lymax,
        iymin,
        iymax,
        rymin,
        rymax,
        instrument,
        render_quality=first_quality,
    )
    has_cached_view = _restore_exact_interactive_cache(first_cache_key, instrument)
    if has_cached_view and first_quality == "coarse":
        first_quality = "full"
    if not has_cached_view:
        has_cached_view = _restore_cached_interactive_view(instrument)
    loading_message = "Refreshing the current selection." if has_cached_view else "Preparing the latest view for this instrument."
    _set_interactive_loading(instrument, loading_message, phase="loading", visible=True)
    if not has_cached_view:
        if not _show_cached_quicklook_placeholder(instrument, start=start, end=end):
            _show_interactive_placeholder(instrument, "Loading the selected window and preparing the first pass of the plot.")

    _IN_FLIGHT_INTERACTIVE_RENDER_CACHE_KEYS.add(first_cache_key)
    _schedule_next_tick(
        lambda: _render_interactive_view(
            start,
            end,
            bottom_val,
            top_val,
            var1_name,
            var2_name,
            bmin,
            bmax,
            lmin,
            lmax,
            lymin,
            lymax,
            iymin,
            iymax,
            rymin,
            rymax,
            instrument,
            request_id=request_id,
            render_quality=first_quality,
        )
    )


def _schedule_interactive_render(
    start,
    end,
    bottom_val,
    top_val,
    var1_name,
    var2_name,
    bmin,
    bmax,
    lmin,
    lmax,
    lymin,
    lymax,
    iymin,
    iymax,
    rymin,
    rymax,
    instrument,
):
    if _instrument_guard:
        return
    global CURRENT_INSTRUMENT, _PENDING_INTERACTIVE_RENDER_ARGS, _PENDING_INTERACTIVE_RENDER_CB
    if instrument != CURRENT_INSTRUMENT:
        CURRENT_INSTRUMENT = instrument
    _PENDING_INTERACTIVE_RENDER_ARGS = (
        start,
        end,
        bottom_val,
        top_val,
        var1_name,
        var2_name,
        bmin,
        bmax,
        lmin,
        lmax,
        lymin,
        lymax,
        iymin,
        iymax,
        rymin,
        rymax,
        instrument,
    )
    if _APP_BOOTSTRAPPING or not _INTERACTIVE_RENDER_ENABLED:
        if _DISPLAYED_INTERACTIVE_INSTRUMENT is None:
            if not _show_cached_quicklook_placeholder(instrument, start=start, end=end):
                _show_interactive_placeholder(instrument, "Preparing the initial dashboard view.")
        _perf_log("interactive_render_deferred", instrument=instrument)
        return
    if _PENDING_INTERACTIVE_RENDER_CB is not None:
        _perf_log("interactive_render_debounced", instrument=instrument)
        return

    def _flush_pending_interactive_render() -> None:
        global _PENDING_INTERACTIVE_RENDER_ARGS, _PENDING_INTERACTIVE_RENDER_CB
        args = _PENDING_INTERACTIVE_RENDER_ARGS
        _PENDING_INTERACTIVE_RENDER_ARGS = None
        _PENDING_INTERACTIVE_RENDER_CB = None
        if args is not None:
            _start_interactive_render(*args)

    _PENDING_INTERACTIVE_RENDER_CB = _schedule_timeout(_flush_pending_interactive_render, RENDER_DEBOUNCE_MS)


@pn.depends(
    range_start.param.value,
    range_end.param.value,
    bottom_range_m.param.value,
    top_range_m.param.value,
    var1_select.param.value,
    var2_select.param.value,
    beta_vmin.param.value,
    beta_vmax.param.value,
    ldr_vmin.param.value,
    ldr_vmax.param.value,
    lwp_ymin.param.value,
    lwp_ymax.param.value,
    iwv_ymin.param.value,
    iwv_ymax.param.value,
    irr_ymin.param.value,
    irr_ymax.param.value,
    instrument_select.param.value,
    watch=True,
)
def _update_view(start, end, bottom_val, top_val, var1_name, var2_name, bmin, bmax, lmin, lmax, lymin, lymax, iymin, iymax, rymin, rymax, instrument):
    """Schedule an interactive render while keeping the current pane warm."""
    _schedule_interactive_render(
        start,
        end,
        bottom_val,
        top_val,
        var1_name,
        var2_name,
        bmin,
        bmax,
        lmin,
        lmax,
        lymin,
        lymax,
        iymin,
        iymax,
        rymin,
        rymax,
        instrument,
    )


def _schedule_current_interactive_render() -> None:
    _schedule_interactive_render(
        range_start.value,
        range_end.value,
        bottom_range_m.value,
        top_range_m.value,
        var1_select.value,
        var2_select.value,
        beta_vmin.value,
        beta_vmax.value,
        ldr_vmin.value,
        ldr_vmax.value,
        lwp_ymin.value,
        lwp_ymax.value,
        iwv_ymin.value,
        iwv_ymax.value,
        irr_ymin.value,
        irr_ymax.value,
        instrument_select.value,
    )


def _enable_browser_interactive_render() -> None:
    global _INTERACTIVE_RENDER_ENABLED
    if pn.state.curdoc is None:
        return
    _INTERACTIVE_RENDER_ENABLED = True
    # A latest Power view uses prewarmed JSON. Its status footer would open the
    # multi-million-sample raw store while that cached figure is parsing, so it
    # is activated by the prewarm publisher after first paint instead.
    if instrument_select.value != "power":
        _schedule_timeout(_activate_interactive_footer_metrics, 750)
    _schedule_timeout(_schedule_current_interactive_render, 250)


def _parse_relayout_time(val):
    """Parse plotly relayout timestamps safely."""
    try:
        return pd.Timestamp(pd.to_datetime(val)).to_pydatetime(warn=False)
    except Exception:
        return None


def _on_relayout(event):
    """When the user zooms/pans, sync controls (and disable live) so we reload data at higher detail."""
    global _relayout_guard
    if _relayout_guard:
        return
    data = event.new or {}
    x0 = data.get("xaxis.range[0]")
    x1 = data.get("xaxis.range[1]")
    start = _parse_relayout_time(x0) if x0 is not None else None
    end = _parse_relayout_time(x1) if x1 is not None else None
    if start is not None and end is not None and start > end:
        start, end = end, start
    y0 = None
    y1 = None
    if not _is_stacked_timeseries_instrument(CURRENT_INSTRUMENT) and not _is_wxcam_instrument(CURRENT_INSTRUMENT):
        y0 = data.get("yaxis.range[0]") or data.get("yaxis2.range[0]")
        y1 = data.get("yaxis.range[1]") or data.get("yaxis2.range[1]")
        if y0 is not None and y1 is not None and y0 > y1:
            y0, y1 = y1, y0
    if start is None and end is None and y0 is None and y1 is None:
        return
    _perf_log(
        "plot_relayout",
        instrument=CURRENT_INSTRUMENT,
        start=start,
        end=end,
        y0=y0,
        y1=y1,
        request_path=_request_path(),
        **_selection_snapshot(),
    )
    _relayout_guard = True
    try:
        _set_live(False)
        if start is not None and end is not None:
            range_start.value = start
            range_end.value = end
        if y0 is not None and y1 is not None:
            low = max(float(y0), 0.0)
            high = max(float(y1), low + 100.0)
            bottom_range_m.value = int(low)
            top_range_m.value = int(high)
    finally:
        _relayout_guard = False


plot_pane.param.watch(_on_relayout, "relayout_data")
# Initial render
_update_view(
    range_start.value,
    range_end.value,
    bottom_range_m.value,
    top_range_m.value,
    var1_select.value,
    var2_select.value,
    beta_vmin.value,
    beta_vmax.value,
    ldr_vmin.value,
    ldr_vmax.value,
    lwp_ymin.value,
    lwp_ymax.value,
    iwv_ymin.value,
    iwv_ymax.value,
    irr_ymin.value,
    irr_ymax.value,
    instrument_select.value,
)




# -------- Science / housekeeping quicklooks --------

def _empty_quicklook_options(mode: str) -> dict[str, None]:
    if mode == "housekeeping":
        return {"No housekeeping quicklooks available": None}
    return {"No images available": None}


def _housekeeping_quicklook_label(inst: str) -> str | None:
    return housekeeping_label(inst) if _is_stacked_timeseries_instrument(inst) else extra_housekeeping_label(inst)


def _housekeeping_latest_path(inst: str, quick_dir: Path) -> Path | None:
    return housekeeping_latest_png(quick_dir, inst) if _is_stacked_timeseries_instrument(inst) else extra_housekeeping_latest_png(quick_dir, inst)


def _housekeeping_daily_path(inst: str, quick_dir: Path, token: str) -> Path | None:
    return housekeeping_daily_png(quick_dir, inst, token) if _is_stacked_timeseries_instrument(inst) else extra_housekeeping_daily_png(quick_dir, inst, token)


def _housekeeping_tokens(inst: str, quick_dir: Path) -> list[str]:
    if _is_stacked_timeseries_instrument(inst):
        label = _housekeeping_quicklook_label(inst)
        if label is None:
            return []
        return [token for token in calendar_date_tokens(quick_dir, inst) if (_housekeeping_daily_path(inst, quick_dir, token) or Path()).exists()]
    return extra_housekeeping_tokens(quick_dir, inst)


def _legacy_science_quicklook_token(inst: str, png: Path) -> str | None:
    """Return YYYYMMDD for legacy daily science PNGs, excluding HK products."""
    prefixes = {
        "Ceilometer": "ceilometer",
        "Cloud Radar": "cloud_radar",
        "Scanning Microwave Radiometer": "hatpro",
    }
    prefix = prefixes.get(inst)
    if prefix is None or "__" in png.stem:
        return None
    expected_prefix = f"{prefix}_"
    if not png.stem.startswith(expected_prefix):
        return None
    token = png.stem.removeprefix(expected_prefix)
    if len(token) == 8 and token.isdigit():
        return token
    return None


def _quicklook_options(inst: str | None = None, wxcam_selection: str | None = None, mode: str = "science"):
    """Build a mapping of label -> quicklook asset token/path for a quicklook mode."""
    inst = inst or CURRENT_INSTRUMENT
    opts = {}
    if _is_uas_instrument(inst):
        if mode == "housekeeping":
            return _empty_quicklook_options(mode)
        if UAS_QUICKLOOK_DIR.exists():
            for path in sorted(UAS_QUICKLOOK_DIR.glob("uas__summary__*.png")):
                token = path.stem.removeprefix("uas__summary__")
                if len(token) == 8 and token.isdigit():
                    opts[token] = str(path)
            latest = UAS_QUICKLOOK_DIR / "uas__summary__latest.png"
            if latest.exists():
                opts["Latest available"] = str(latest)
        return opts or _empty_quicklook_options(mode)
    cfg = _cfg(inst)
    if _is_wxcam_instrument(inst):
        if mode == "housekeeping":
            return _empty_quicklook_options(mode)
        return _wxcam_calendar_options(wxcam_selection or _cfg("wxcam")["default_top"])
    if _is_stacked_timeseries_instrument(inst):
        quick_dir = cfg["quicklook_dir"]
        if mode == "science" and quick_dir.exists():
            for token in calendar_date_tokens(quick_dir, inst):
                if summary_daily_png(quick_dir, inst, token).exists():
                    opts[token] = token
            if summary_latest_png(quick_dir, inst).exists():
                opts["Today (latest)"] = "latest"
        elif mode == "housekeeping" and quick_dir.exists():
            hk_latest = _housekeeping_latest_path(inst, quick_dir)
            for token in _housekeeping_tokens(inst, quick_dir):
                hk_path = _housekeeping_daily_path(inst, quick_dir, token)
                if hk_path and hk_path.exists():
                    opts[token] = token
            if hk_latest and hk_latest.exists():
                opts["Today (latest)"] = "latest"
        return opts or _empty_quicklook_options(mode)
    if mode == "housekeeping":
        quick_dir = cfg["quicklook_dir"]
        hk_latest = _housekeeping_latest_path(inst, quick_dir)
        if quick_dir.exists():
            for token in _housekeeping_tokens(inst, quick_dir):
                hk_path = _housekeeping_daily_path(inst, quick_dir, token)
                if hk_path and hk_path.exists():
                    opts[token] = token
            if hk_latest and hk_latest.exists():
                opts["Today (latest)"] = "latest"
        return opts or _empty_quicklook_options(mode)
    quick_dir = cfg["quicklook_dir"]
    latest = cfg["latest_image"]
    # Collect dated quicklooks (sorted ascending), then append "Today" last.
    date_labels = []
    if quick_dir.exists():
        for png in sorted(quick_dir.glob("*.png")):
            if png.name == latest.name:
                continue
            label = _legacy_science_quicklook_token(inst, png)
            if label is not None:
                date_labels.append((label, str(png)))
    for label, path in date_labels:
        opts[label] = path
    if latest.exists():
        opts["Today (latest)"] = str(latest)
    return opts or _empty_quicklook_options(mode)


_ql_options = _quicklook_options(mode="science")
ql_date = pn.widgets.Select(name="Date", options=list(_ql_options.keys()))
if _ql_options:
    ql_date.value = list(_ql_options.keys())[-1]

_hk_options = _quicklook_options(mode="housekeeping")
hk_date = pn.widgets.Select(name="Date", options=list(_hk_options.keys()))
if _hk_options:
    hk_date.value = list(_hk_options.keys())[-1]


def _refresh_ql_options(preserve_current: bool = True):
    """Refresh available science quicklook options, optionally preserving current selection."""
    global _ql_options
    current = ql_date.value if preserve_current else None
    _ql_options = _quicklook_options(science_instrument.value, science_image_type.value, mode="science")
    opts = list(_ql_options.keys())
    ql_date.options = opts
    if not opts:
        ql_date.value = None
        return
    if preserve_current and current in opts:
        ql_date.value = current
    elif (
        _is_wxcam_instrument(science_instrument.value)
        or _is_stacked_timeseries_instrument(science_instrument.value)
        or _is_uas_instrument(science_instrument.value)
    ):
        if "Latest available" in opts:
            ql_date.value = "Latest available"
        else:
            ql_date.value = "Today (latest)" if "Today (latest)" in opts else opts[-1]
    else:
        ql_date.value = opts[-1]


def _refresh_hk_options(preserve_current: bool = True):
    """Refresh available housekeeping quicklook options, optionally preserving current selection."""
    global _hk_options
    current = hk_date.value if preserve_current else None
    _hk_options = _quicklook_options(hk_instrument.value, mode="housekeeping")
    opts = list(_hk_options.keys())
    hk_date.options = opts
    if not opts:
        hk_date.value = None
        return
    if preserve_current and current in opts:
        hk_date.value = current
    elif _is_stacked_timeseries_instrument(hk_instrument.value):
        hk_date.value = "Today (latest)" if "Today (latest)" in opts else opts[-1]
    else:
        hk_date.value = opts[-1]

# Quicklook navigation buttons
ql_prev = pn.widgets.Button(name="<<", button_type="default")
ql_next = pn.widgets.Button(name=">>", button_type="default")
hk_prev = pn.widgets.Button(name="<<", button_type="default")
hk_next = pn.widgets.Button(name=">>", button_type="default")


def _shift_ql(delta: int):
    """Move science quicklook day selection by delta steps."""
    _refresh_ql_options(preserve_current=True)
    opts = list(ql_date.options)
    if not opts or ql_date.value not in opts:
        return
    idx = opts.index(ql_date.value)
    new_idx = max(0, min(len(opts) - 1, idx + delta))
    if new_idx != idx:
        ql_date.value = opts[new_idx]
    else:
        # Force a refresh even if the value is unchanged (e.g., only one option)
        ql_date.param.trigger("value")


ql_prev.on_click(lambda _e: _shift_ql(-1))
ql_next.on_click(lambda _e: _shift_ql(1))


def _shift_hk(delta: int):
    """Move housekeeping selection by delta steps in the refreshed options list."""
    _refresh_hk_options(preserve_current=True)
    opts = list(hk_date.options)
    if not opts or hk_date.value not in opts:
        return
    idx = opts.index(hk_date.value)
    new_idx = max(0, min(len(opts) - 1, idx + delta))
    if new_idx != idx:
        hk_date.value = opts[new_idx]
    else:
        hk_date.param.trigger("value")


hk_prev.on_click(lambda _e: _shift_hk(-1))
hk_next.on_click(lambda _e: _shift_hk(1))

# Periodically refresh the "Today (latest)" selection to pick up new PNGs.
def _refresh_latest_if_needed():
    """If viewing the latest science quicklook, reload the mapping and redraw."""
    if _is_wxcam_instrument(science_instrument.value):
        return
    if ql_date.value in {"Today (latest)", "Latest available"}:
        global _ql_options
        _ql_options = _quicklook_options(science_instrument.value, science_image_type.value, mode="science")
        ql_date.param.trigger("value")


_ql_timer = _safe_periodic_callback(_refresh_latest_if_needed, period=300_000, start=False)


def _refresh_hk_latest_if_needed():
    """If viewing the latest housekeeping quicklook, reload the mapping and redraw."""
    if hk_date.value == "Today (latest)":
        global _hk_options
        _hk_options = _quicklook_options(hk_instrument.value, mode="housekeeping")
        hk_date.param.trigger("value")


_hk_timer = _safe_periodic_callback(_refresh_hk_latest_if_needed, period=300_000, start=False)

# Ensure initial map is fresh
_refresh_ql_options(preserve_current=True)
_refresh_hk_options(preserve_current=True)
_apply_instrument_defaults(CURRENT_INSTRUMENT, reset_time=True, sync_quicklooks=True)


def _safe_widget_value(widget_name: str):
    widget = globals().get(widget_name)
    return getattr(widget, "value", None) if widget is not None else None


def _selection_snapshot() -> dict[str, object]:
    return {
        "current_instrument": _safe_widget_value("instrument_select"),
        "science_instrument": _safe_widget_value("science_instrument"),
        "science_date": _safe_widget_value("ql_date"),
        "science_image_type": _safe_widget_value("science_image_type"),
        "hk_instrument": _safe_widget_value("hk_instrument"),
        "hk_date": _safe_widget_value("hk_date"),
        "wxcam_image_type": _safe_widget_value("wxcam_image_type"),
        "wxcam_date": _safe_widget_value("wxcam_date"),
        "wxcam_selected_hour_path": getattr(wxcam_calendar_state, "selected_hour_path", ""),
        "live_mode": _safe_widget_value("live_toggle"),
    }


def _selection_snapshot_safe() -> dict[str, object]:
    try:
        snapshot_fn = globals().get("_selection_snapshot")
        if callable(snapshot_fn):
            return snapshot_fn()
    except Exception:
        pass
    return {}


def _log_control_change(control: str, event, context: str, instrument: str | None = None) -> None:
    old = getattr(event, "old", None)
    new = getattr(event, "new", None)
    if old == new:
        return
    fields = _selection_snapshot()
    fields.update(
        {
            "control": control,
            "context": context,
            "old": old,
            "new": new,
            "request_path": _request_path(),
        }
    )
    _perf_log("ui_selection_change", instrument=instrument or CURRENT_INSTRUMENT, **fields)


def _log_session_loaded() -> None:
    fields = _selection_snapshot_safe()
    fields.update(
        {
            "request_path": _request_path(),
            "client_ip": _client_ip(),
            "user_agent": _request_header("User-Agent"),
            "status": "loaded",
            "app_boot_ms": round(_session_age_seconds() * 1000.0, 3),
        }
    )
    _perf_log("session_loaded", instrument=_safe_widget_value("instrument_select") or CURRENT_INSTRUMENT, **fields)


def _log_serialized_document_size() -> None:
    """Record the actual Bokeh document payload size after first render.

    Panel sends this model graph through its websocket.  Measure it only for
    development RUM sessions and after the initial chart is already visible,
    so the diagnostic cannot affect first-paint timing.
    """
    if SITE_ENV != "development" or not _browser_rum_enabled():
        return
    doc = pn.state.curdoc
    if doc is None:
        return
    started = perf_counter()
    try:
        # ``serialize_json`` is Bokeh's own protocol serializer. Unlike the
        # standard library encoder, it also handles NumPy and datetime values
        # used by richer forecast figures.
        payload_bytes = len(serialize_json(doc.to_json()).encode("utf-8"))
    except Exception as exc:
        _perf_log(
            "browser_document_model_size",
            instrument=_safe_widget_value("instrument_select") or CURRENT_INSTRUMENT,
            status="error",
            error_type=type(exc).__name__,
            duration_ms=round((perf_counter() - started) * 1000, 3),
        )
        return
    _perf_log(
        "browser_document_model_size",
        instrument=_safe_widget_value("instrument_select") or CURRENT_INSTRUMENT,
        status="ok",
        document_bytes=payload_bytes,
        duration_ms=round((perf_counter() - started) * 1000, 3),
        request_path=_request_path(),
    )


def _schedule_serialized_document_size() -> None:
    """Defer development-only document accounting until after first paint."""
    if SITE_ENV != "development" or not _browser_rum_enabled() or pn.state.curdoc is None:
        return
    pn.state.curdoc.add_timeout_callback(_log_serialized_document_size, 5_000)


def _log_session_heartbeat() -> None:
    fields = _selection_snapshot_safe()
    fields.update(
        {
            "request_path": _request_path(),
            "status": "alive",
        }
    )
    _perf_log("session_heartbeat", instrument=_safe_widget_value("instrument_select") or CURRENT_INSTRUMENT, **fields)


def _log_session_destroyed(session_context) -> None:
    perf_logger = globals().get("_perf_log")
    if not callable(perf_logger):
        return
    request = getattr(session_context, "request", None)
    path = str(getattr(request, "path", "")) or None
    server_context = getattr(session_context, "server_context", None)
    server_sessions = None
    if server_context is not None:
        try:
            server_sessions = int(len(server_context.sessions))
        except Exception:
            server_sessions = None
    fields = {}
    try:
        instrument_widget = globals().get("instrument_select")
        science_widget = globals().get("science_instrument")
        ql_widget = globals().get("ql_date")
        science_type_widget = globals().get("science_image_type")
        hk_widget = globals().get("hk_instrument")
        hk_date_widget = globals().get("hk_date")
        wxcam_type_widget = globals().get("wxcam_image_type")
        wxcam_date_widget = globals().get("wxcam_date")
        wxcam_state = globals().get("wxcam_calendar_state")
        live_widget = globals().get("live_toggle")
        fields.update(
            {
                "current_instrument": getattr(instrument_widget, "value", None),
                "science_instrument": getattr(science_widget, "value", None),
                "science_date": getattr(ql_widget, "value", None),
                "science_image_type": getattr(science_type_widget, "value", None),
                "hk_instrument": getattr(hk_widget, "value", None),
                "hk_date": getattr(hk_date_widget, "value", None),
                "wxcam_image_type": getattr(wxcam_type_widget, "value", None),
                "wxcam_date": getattr(wxcam_date_widget, "value", None),
                "wxcam_selected_hour_path": getattr(wxcam_state, "selected_hour_path", ""),
                "live_mode": getattr(live_widget, "value", None),
            }
        )
    except Exception:
        fields = {}
    fields.update(
        {
            "request_path": path,
            "status": "destroyed",
            "session_id": getattr(session_context, "id", None),
            "server_sessions": server_sessions,
        }
    )
    current_instrument = fields.get("current_instrument") or globals().get("CURRENT_INSTRUMENT")
    perf_logger("session_destroyed", instrument=current_instrument, **fields)
    for timer in list(globals().get("_SESSION_PERIODIC_CALLBACKS", ())):
        try:
            timer.stop()
        except Exception:
            pass
    globals().get("_SESSION_PERIODIC_CALLBACKS", []).clear()


instrument_select.param.watch(
    lambda event: _log_control_change("instrument_select", event, context="interactive", instrument=event.new),
    "value",
)
science_instrument.param.watch(
    lambda event: _log_control_change("science_instrument", event, context="science_quicklooks", instrument=event.new),
    "value",
)
ql_date.param.watch(
    lambda event: _log_control_change("ql_date", event, context="science_quicklooks", instrument=science_instrument.value),
    "value",
)
science_image_type.param.watch(
    lambda event: _log_control_change("science_image_type", event, context="science_quicklooks", instrument="wxcam"),
    "value",
)
hk_instrument.param.watch(
    lambda event: _log_control_change("hk_instrument", event, context="housekeeping_quicklooks", instrument=event.new),
    "value",
)
hk_date.param.watch(
    lambda event: _log_control_change("hk_date", event, context="housekeeping_quicklooks", instrument=hk_instrument.value),
    "value",
)
wxcam_image_type.param.watch(
    lambda event: _log_control_change("wxcam_image_type", event, context="wxcam_interactive", instrument="wxcam"),
    "value",
)
wxcam_date.param.watch(
    lambda event: _log_control_change("wxcam_date", event, context="wxcam_interactive", instrument="wxcam"),
    "value",
)
live_toggle.param.watch(
    lambda event: _log_control_change("live_toggle", event, context="interactive", instrument=instrument_select.value),
    "value",
)
wxcam_calendar_state.param.watch(
    lambda event: _log_control_change("wxcam_selected_hour_path", event, context="wxcam_calendar", instrument="wxcam"),
    "selected_hour_path",
)
pn.state.onload(_log_session_loaded)
pn.state.onload(_schedule_serialized_document_size)
pn.state.on_session_destroyed(_log_session_destroyed)
_session_heartbeat_cb = None
if SESSION_HEARTBEAT_MS > 0:
    _session_heartbeat_cb = _safe_periodic_callback(_log_session_heartbeat, period=SESSION_HEARTBEAT_MS, start=False)


def _start_session_heartbeat() -> None:
    if _session_heartbeat_cb is not None and not _session_heartbeat_cb.running:
        _session_heartbeat_cb.start()


pn.state.onload(_start_session_heartbeat)


def _sync_wxcam_calendar_hour(*_events):
    with _timed_perf(
        "wxcam_calendar_sync",
        instrument="wxcam",
        selected_day=ql_date.value,
        selection=science_image_type.value,
    ) as perf:
        if not _is_wxcam_instrument(science_instrument.value):
            perf["status"] = "non_wxcam"
            wxcam_calendar_state.selected_hour_path = ""
            return
        selected_day = ql_date.value
        selection = science_image_type.value or _cfg("wxcam")["default_top"]
        day_token = _wxcam_calendar_day_token(selected_day)
        day_utc = _wxcam_day_token_to_utc(day_token or "")
        if not day_utc:
            perf["status"] = "invalid_day"
            wxcam_calendar_state.selected_hour_path = ""
            return
        image_type = _image_type_from_selection(selection)
        rows_by_hour = representative_hourly_records(
            _wxcam_catalog_path("wxcam"),
            image_type,
            day_utc,
            media_kind="image",
        )
        available_paths = [str(row["raw_path"]) for row in rows_by_hour.values()]
        perf["hour_count"] = len(rows_by_hour)
        if not available_paths:
            perf["status"] = "empty"
            wxcam_calendar_state.selected_hour_path = ""
            return
        if wxcam_calendar_state.selected_hour_path not in available_paths:
            wxcam_calendar_state.selected_hour_path = ""
        perf["status"] = "ok"


science_instrument.param.watch(_sync_wxcam_calendar_hour, "value")
science_image_type.param.watch(_sync_wxcam_calendar_hour, "value")
ql_date.param.watch(_sync_wxcam_calendar_hour, "value")


def _build_wxcam_hour_tile(
    image_type: str,
    day_token: str,
    hour_index: int,
    row,
    selected_hour_path: str,
):
    hour_label = f"{hour_index:02d}:00"
    tile_classes = ["wxcam-hour-tile"]
    if image_type == "fish_hdr":
        tile_classes.append("wxcam-hour-tile--fish")
    if row is None:
        preview = pn.pane.HTML(
            f"<div class='wxcam-hour-tile__placeholder'>{hour_label}</div>",
            sizing_mode="stretch_width",
            margin=0,
        )
        button = pn.widgets.Button(name=f"{hour_label}", disabled=True, sizing_mode="stretch_width", margin=0)
        return pn.Column(
            preview,
            button,
            css_classes=tile_classes,
            sizing_mode="stretch_width",
            margin=0,
        )

    thumb_path = _wxcam_hourly_thumbnail_path(image_type, day_token, str(row["filename"]))
    if thumb_path.exists():
        preview = pn.pane.HTML(
            f"<img class='wxcam-hour-tile__img' src='{_image_data_uri(thumb_path)}' alt='{hour_label}'>",
            sizing_mode="stretch_width",
            margin=0,
        )
    else:
        preview = pn.pane.HTML(
            f"<div class='wxcam-hour-tile__placeholder'>{hour_label}</div>",
            sizing_mode="stretch_width",
            margin=0,
        )
    button = pn.widgets.Button(
        name=hour_label,
        button_type="primary" if str(row["raw_path"]) == selected_hour_path else "default",
        sizing_mode="stretch_width",
        margin=0,
    )
    button.on_click(lambda _event, path=str(row["raw_path"]): setattr(wxcam_calendar_state, "selected_hour_path", path))
    return pn.Column(
        preview,
        button,
        css_classes=tile_classes,
        sizing_mode="stretch_width",
        margin=0,
    )


def _wxcam_quicklook_header(selection: str, day_utc: str) -> pn.pane.HTML:
    """Return the UTC context banner shown above the WXcam science grid."""
    selection_text = escape(selection)
    day_text = escape(day_utc)
    return pn.pane.HTML(
        (
            "<div class='wxcam-quicklook-header'>"
            f"<div class='wxcam-quicklook-header__title'>{selection_text} science quicklook</div>"
            f"<div class='wxcam-quicklook-header__date'>{day_text} UTC</div>"
            "<div class='wxcam-quicklook-header__note'>"
            "All times are UTC. Each tile is one UTC hour and uses the retained HDR JPG nearest HH:30 UTC when available."
            "</div>"
            "</div>"
        ),
        sizing_mode="stretch_width",
        margin=(0, 0, 8, 0),
    )


def _build_wxcam_calendar_day_view(selection: str, day_token: str, selected_hour_path: str):
    """Build the WXcam Science Quicklook hourly HDR-image grid."""
    with _timed_perf(
        "wxcam_calendar_day_view",
        instrument="wxcam",
        selection=selection,
        day_token=day_token,
        selected_hour_path=selected_hour_path,
    ) as perf:
        day_utc = _wxcam_day_token_to_utc(day_token)
        if not day_utc:
            perf["status"] = "invalid_day"
            return pn.pane.Markdown("No hourly UTC images available for this selection.")
        image_type = _image_type_from_selection(selection)
        rows_by_hour = representative_hourly_records(
            _wxcam_catalog_path("wxcam"),
            image_type,
            day_utc,
            media_kind="image",
        )
        perf["hour_count"] = len(rows_by_hour)
        header = _wxcam_quicklook_header(selection, day_utc)
        if not rows_by_hour:
            perf["status"] = "empty"
            return pn.Column(header, pn.pane.Markdown("No hourly UTC images available for this selection."), sizing_mode="stretch_width")
        tiles = [
            _build_wxcam_hour_tile(image_type, day_token, hour_index, rows_by_hour.get(hour_index), selected_hour_path)
            for hour_index in range(24)
        ]
        grid = pn.GridBox(*tiles, ncols=8, sizing_mode="stretch_width")
        selected_row = next((row for row in rows_by_hour.values() if str(row["raw_path"]) == selected_hour_path), None)
        if selected_row is None:
            perf["status"] = "grid_only"
            return pn.Column(header, grid, sizing_mode="stretch_width")
        selected_hour_label = str(selected_row["time_utc"])[11:16] + " UTC"
        viewer = _build_wxcam_image_view(Path(str(selected_row["raw_path"])), selection, f"{day_utc} UTC | {selected_hour_label}")
        perf["status"] = "with_viewer"
        return pn.Column(header, grid, viewer, sizing_mode="stretch_width")


@pn.depends(
    ql_date.param.value,
    science_instrument.param.value,
    science_image_type.param.value,
    wxcam_calendar_state.param.selected_hour_path,
)
def _science_quicklook_image(selected, science_inst, wxcam_selection, selected_hour_path):
    """Show the selected science quicklook asset (or a message if missing)."""
    instrument = science_inst or CURRENT_INSTRUMENT
    with _timed_perf("science_quicklook_render", instrument=instrument, selected=selected) as perf:
        if _is_wxcam_instrument(instrument):
            selection = wxcam_selection or _cfg("wxcam")["default_top"]
            perf["selection"] = selection
            day_token = _wxcam_calendar_day_token(selected)
            return _build_wxcam_calendar_day_view(selection, day_token or "", selected_hour_path)
        if _is_stacked_timeseries_instrument(instrument):
            token = _quicklook_options(instrument, mode="science").get(selected)
            quick_dir = _cfg(instrument)["quicklook_dir"]
            if token is None:
                perf["status"] = "missing_file"
                return pn.pane.Markdown("No image available for this selection.")
            path = summary_latest_png(quick_dir, instrument) if token == "latest" else summary_daily_png(quick_dir, instrument, token)
            perf["path"] = path
            if not path.exists():
                perf["status"] = "missing_file"
                return pn.pane.Markdown("No image available for this selection.")
            perf["status"] = "ok"
            return _responsive_quicklook_png(path)
        path = _quicklook_options(instrument).get(selected)
        perf["path"] = path
        if path and Path(path).exists():
            perf["status"] = "ok"
            return _media_pane(path)
        perf["status"] = "missing_file"
        return pn.pane.Markdown("No image available for this selection.")


@pn.depends(hk_date.param.value, hk_instrument.param.value)
def _housekeeping_quicklook_image(selected, hk_inst):
    """Show the selected housekeeping quicklook asset (or a message if missing)."""
    instrument = hk_inst or CURRENT_INSTRUMENT
    with _timed_perf("housekeeping_quicklook_render", instrument=instrument, selected=selected) as perf:
        quick_dir = _cfg(instrument)["quicklook_dir"]
        hk_label = _housekeeping_quicklook_label(instrument)
        if hk_label is None:
            perf["status"] = "unsupported"
            return pn.pane.Markdown("No housekeeping quicklooks available for this instrument.")
        token = _quicklook_options(instrument, mode="housekeeping").get(selected)
        if token is None:
            perf["status"] = "missing_file"
            return pn.pane.Markdown("No housekeeping quicklooks available for this instrument.")
        path = _housekeeping_latest_path(instrument, quick_dir) if token == "latest" else _housekeeping_daily_path(instrument, quick_dir, token)
        perf["path"] = path
        if path and path.exists():
            perf["status"] = "ok"
            return _responsive_quicklook_png(path)
        perf["status"] = "missing_file"
        return pn.pane.Markdown("No housekeeping quicklooks available for this instrument.")


def _current_interactive_status_markup() -> str:
    inst = instrument_select.value
    if _is_wxcam_instrument(inst):
        selection = wxcam_image_type.value or _cfg("wxcam")["default_top"]
        selected = wxcam_date.value
        day_token = _wxcam_calendar_day_token(selected)
        bits = _wxcam_hour_bits(selection, day_token or "")
        missing = sum(1 for bit in bits if not bit)
        latest_row = latest_record(_wxcam_catalog_path("wxcam"), _image_type_from_selection(selection), media_kind="image")
        latest_dt = pd.Timestamp(str(latest_row["time_utc"])).to_pydatetime(warn=False) if latest_row else None
        lag = datetime.now() - latest_dt if latest_dt is not None else None
        items = [
            ("Last sample", _format_status_time(latest_dt), "info"),
            ("Hourly gaps", str(missing), "warn" if missing else "ok"),
            ("Lag", _format_duration(lag), "warn" if lag and lag > timedelta(hours=1) else "ok"),
        ]
        return _status_strip_markup(items)
    latest = _dataset_time_bounds(inst)[1]
    lag = datetime.now() - latest if latest is not None else None
    times = _status_time_index(inst)
    bits, missing, total = _hourly_coverage_summary(times, _ensure_utc(range_start.value), _ensure_utc(range_end.value))
    items = [
        ("Last sample", _format_status_time(latest), "info"),
        ("Hourly gaps", str(missing), "warn" if missing else "ok"),
        ("Lag", _format_duration(lag), "warn" if lag and lag > timedelta(hours=1) else "ok"),
    ]
    if total:
        items.insert(2, ("Coverage", f"{total - missing}/{total} h", "info"))
    return _status_strip_markup(items)


def _current_interactive_availability_markup() -> str:
    inst = instrument_select.value
    if _is_wxcam_instrument(inst):
        selection = wxcam_image_type.value or _cfg("wxcam")["default_top"]
        selected = wxcam_date.value
        day_token = _wxcam_calendar_day_token(selected)
        bits = _wxcam_hour_bits(selection, day_token or "")
        start_label = "00:00"
        end_label = "23:00"
        if day_token == datetime.now(timezone.utc).strftime("%Y%m%d"):
            end_label = datetime.now(timezone.utc).strftime("%H:00")
        return _availability_bar_markup(
            bits,
            f"{start_label} UTC",
            f"{end_label} UTC",
            "HDR image availability by UTC hour",
            "Each block represents one UTC hour. Teal means at least one HDR image exists for that hour.",
            full_label="HDR image present",
            empty_label="No HDR image",
            segment_titles=[f"{hour:02d}:00 UTC" for hour in range(24)],
        )
    start = _ensure_utc(range_start.value)
    end = _ensure_utc(range_end.value)
    bits = _binned_time_coverage(_status_time_index(inst), start, end, segments=72)
    start_label = start.strftime("%m-%d %H:%M") if start else "--"
    end_label = end.strftime("%m-%d %H:%M") if end else "--"
    return _availability_bar_markup(
        bits,
        start_label,
        end_label,
        "Data availability across the selected time window",
        _availability_bucket_text(start, end, len(bits)) + " Teal means at least one sample exists in that slice.",
        full_label="Samples present",
        empty_label="No samples",
        segment_titles=_availability_binned_titles(start, end, len(bits)),
    )


def _uas_science_day_query() -> str:
    if ql_date.value == "Latest available":
        return "latest"
    token = str(ql_date.value or "")
    if len(token) == 8 and token.isdigit():
        return f"{token[:4]}-{token[4:6]}-{token[6:8]}"
    return "latest"


def _uas_science_listing() -> dict:
    return mobile_catalog.uas_flights(day=_uas_science_day_query())


def _current_science_status_markup() -> str:
    inst = science_instrument.value
    if _is_uas_instrument(inst):
        listing = _uas_science_listing()
        status = listing.get("status") or {}
        last_run = pd.to_datetime(
            listing.get("lastRunAt") or listing.get("generatedAt"),
            errors="coerce",
            utc=True,
        )
        last_run_dt = None if pd.isna(last_run) else last_run.to_pydatetime(warn=False)
        flights = listing.get("flights") or []
        latest_start = pd.to_datetime(
            flights[0].get("startTimeUTC") if flights else None,
            errors="coerce",
            utc=True,
        )
        latest_dt = None if pd.isna(latest_start) else latest_start.to_pydatetime(warn=False)
        level = "ok" if status.get("level") == "green" else "warn"
        items = [
            ("Product state", str(status.get("title") or "Unknown"), level),
            ("Flights in image", str(len(flights)), "ok" if flights else "warn"),
            ("Latest flight", _format_status_time(latest_dt), "info"),
            ("Last product build", _format_status_time(last_run_dt), level),
        ]
        return _status_strip_markup(items)
    if _is_wxcam_instrument(inst):
        selection = science_image_type.value or _cfg("wxcam")["default_top"]
        day_token = _wxcam_calendar_day_token(ql_date.value)
        bits = _wxcam_hour_bits(selection, day_token or "")
        missing = sum(1 for bit in bits if not bit)
        latest_row = latest_record(_wxcam_catalog_path("wxcam"), _image_type_from_selection(selection), media_kind="image")
        latest_dt = pd.Timestamp(str(latest_row["time_utc"])).to_pydatetime(warn=False) if latest_row else None
        lag = datetime.now() - latest_dt if latest_dt is not None else None
        items = [
            ("Last sample", _format_status_time(latest_dt), "info"),
            ("Hourly gaps", str(missing), "warn" if missing else "ok"),
            ("Lag", _format_duration(lag), "warn" if lag and lag > timedelta(hours=1) else "ok"),
        ]
        return _status_strip_markup(items)
    start, end, _day_token = _selected_token_window(ql_date.value)
    times = _instrument_time_index(inst)
    if start is not None and end is not None:
        mask = (times >= pd.Timestamp(start)) & (times <= pd.Timestamp(end))
        window_times = times[mask]
    else:
        window_times = times
    latest_dt = window_times.max().to_pydatetime(warn=False) if len(window_times) else _dataset_time_bounds(inst)[1]
    bits, missing, total = _hourly_coverage_summary(window_times, start, end)
    now = _utcnow_naive()
    lag = now - latest_dt if latest_dt is not None and start and start.date() == now.date() else None
    items = [("Last sample", _format_status_time(latest_dt), "info")]
    if total:
        items.append(("Hourly gaps", str(missing), "warn" if missing else "ok"))
        items.append(("Coverage", f"{total - missing}/{total} h", "info"))
    if lag is not None:
        items.append(("Lag", _format_duration(lag), "warn" if lag > timedelta(hours=1) else "ok"))
    return _status_strip_markup(items)


def _current_science_availability_markup() -> str:
    inst = science_instrument.value
    if _is_uas_instrument(inst):
        listing = _uas_science_listing()
        flights = listing.get("flights") or []
        counts = [0] * 24
        for flight in flights:
            value = pd.to_datetime(flight.get("startTimeUTC"), errors="coerce", utc=True)
            if not pd.isna(value):
                counts[int(value.hour)] += 1
        selected_day = listing.get("selectedDay") or "--"
        return _availability_bar_markup(
            [count > 0 for count in counts],
            "00:00 UTC",
            "23:00 UTC",
            f"UAS flight starts on {selected_day}",
            "Each block represents one UTC hour. Teal means at least one flight starts in that hour.",
            full_label="Flight starts",
            empty_label="No flight start",
            segment_titles=[
                f"{hour:02d}:00 UTC - {counts[hour]} flight{'s' if counts[hour] != 1 else ''}"
                for hour in range(24)
            ],
        )
    if _is_wxcam_instrument(inst):
        selection = science_image_type.value or _cfg("wxcam")["default_top"]
        day_token = _wxcam_calendar_day_token(ql_date.value)
        bits = _wxcam_hour_bits(selection, day_token or "")
        end_label = datetime.now(timezone.utc).strftime("%H:00") if day_token == datetime.now(timezone.utc).strftime("%Y%m%d") else "23:00"
        return _availability_bar_markup(
            bits,
            "00:00 UTC",
            f"{end_label} UTC",
            "HDR image availability by UTC hour",
            "Each block represents one UTC hour. Teal means the science quicklook has an HDR image source for that hour.",
            full_label="Representative HDR image present",
            empty_label="No representative HDR image",
            segment_titles=[f"{hour:02d}:00 UTC" for hour in range(24)],
        )
    start, end, _day_token = _selected_token_window(ql_date.value)
    times = _instrument_time_index(inst)
    if start is None or end is None:
        return _availability_bar_markup([], "--", "--", "Data availability by hour", "Each block would represent one UTC hour in the selected day.")
    mask = (times >= pd.Timestamp(start)) & (times <= pd.Timestamp(end))
    bits, _missing, _total = _hourly_coverage_summary(times[mask], start, end)
    end_label = end.strftime("%H:%M")
    return _availability_bar_markup(
        bits,
        "00:00",
        end_label,
        "Data availability by hour",
        "Each block represents one UTC hour in the selected day. Teal means at least one sample exists in that hour.",
        full_label="Samples present",
        empty_label="No samples",
        segment_titles=_availability_hour_titles(start, end),
    )


def _current_hk_status_markup() -> str:
    inst = hk_instrument.value
    start, end, day_token = _selected_token_window(hk_date.value)
    if inst == "wxcam":
        states = _wxcam_combined_hour_states(day_token or "")
        missing = sum(1 for state in states if state != 2)
        latest_dt = _dataset_time_bounds("wxcam")[1]
        lag = datetime.now() - latest_dt if latest_dt is not None else None
        items = [
            ("Last sample", _format_status_time(latest_dt), "info"),
            ("Partial/Missing hours", str(missing), "warn" if missing else "ok"),
            ("Lag", _format_duration(lag), "warn" if lag and lag > timedelta(hours=1) else "ok"),
        ]
        return _status_strip_markup(items)
    times = _instrument_time_index(inst)
    if start is not None and end is not None:
        mask = (times >= pd.Timestamp(start)) & (times <= pd.Timestamp(end))
        window_times = times[mask]
    else:
        window_times = times
    latest_dt = window_times.max().to_pydatetime(warn=False) if len(window_times) else _dataset_time_bounds(inst)[1]
    bits, missing, total = _hourly_coverage_summary(window_times, start, end)
    now = _utcnow_naive()
    lag = now - latest_dt if latest_dt is not None and start and start.date() == now.date() else None
    items = [("Last sample", _format_status_time(latest_dt), "info")]
    if total:
        items.append(("Hourly gaps", str(missing), "warn" if missing else "ok"))
        items.append(("Coverage", f"{total - missing}/{total} h", "info"))
    if lag is not None:
        items.append(("Lag", _format_duration(lag), "warn" if lag > timedelta(hours=1) else "ok"))
    return _status_strip_markup(items)


def _current_hk_availability_markup() -> str:
    inst = hk_instrument.value
    start, end, day_token = _selected_token_window(hk_date.value)
    if inst == "wxcam":
        states = _wxcam_combined_hour_states(day_token or "")
        end_label = datetime.now(timezone.utc).strftime("%H:00") if day_token == datetime.now(timezone.utc).strftime("%Y%m%d") else "23:00"
        return _availability_bar_markup(
            states,
            "00:00 UTC",
            f"{end_label} UTC",
            "WXcam HDR availability by UTC hour",
            (
                "Each block represents one UTC hour. Teal means the retained "
                "WXcam HDR streams have an image for that hour; gold means "
                "only some retained streams are present; gray means no "
                "retained HDR image is available."
            ),
            full_label="All retained HDR streams present",
            partial_label="Partial retained HDR coverage",
            empty_label="No retained HDR image",
            segment_titles=[f"{hour:02d}:00 UTC" for hour in range(24)],
        )
    if start is None or end is None:
        return _availability_bar_markup(
            [],
            "--",
            "--",
            "Housekeeping data availability by hour",
            "Each block would represent one UTC hour in the selected day.",
        )
    times = _instrument_time_index(inst)
    mask = (times >= pd.Timestamp(start)) & (times <= pd.Timestamp(end))
    bits, _missing, _total = _hourly_coverage_summary(times[mask], start, end)
    return _availability_bar_markup(
        bits,
        "00:00",
        end.strftime("%H:%M"),
        "Housekeeping data availability by hour",
        "Each block represents one UTC hour in the selected day. Teal means at least one housekeeping sample exists in that hour.",
        full_label="Samples present",
        empty_label="No samples",
        segment_titles=_availability_hour_titles(start, end),
    )


def _uas_now() -> datetime:
    return datetime.now(timezone.utc)


def _uas_window_bounds(window_label: str | None = None) -> tuple[datetime, datetime]:
    end = _uas_now()
    delta = UAS_WINDOW_OPTIONS.get(window_label or uas_window.value, UAS_WINDOW_OPTIONS["Last 24 h"])
    return end - delta, end


def _uas_load_result() -> UASMqttParseResult:
    return load_uas_mqtt_log(UAS_MQTT_LOG_PATH)


def _uas_window_records(result: UASMqttParseResult, window_label: str | None = None) -> tuple[UASMqttRecord, ...]:
    start, end = _uas_window_bounds(window_label)
    return tuple(record for record in result.records if start <= record.timestamp <= end)


def _uas_latest_record(result: UASMqttParseResult) -> UASMqttRecord | None:
    return result.records[-1] if result.records else None


def _uas_level_for_age(age: timedelta | None) -> str:
    if age is None:
        return "warn"
    return "warn" if age > UAS_STALE_AFTER else "ok"


def _uas_summary_markup(result: UASMqttParseResult, records: tuple[UASMqttRecord, ...]) -> str:
    latest = _uas_latest_record(result)
    now = _uas_now()
    age = now - latest.timestamp if latest else None
    parse_level = "warn" if result.missing or result.error or result.malformed_lines else "ok"
    parse_label = "Missing log" if result.missing else "Read error" if result.error else "Parse warnings" if result.malformed_lines else "OK"
    current_level = _uas_level_for_age(age)
    try:
        archive_health = json.loads(ARCHIVE_HEALTH_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        archive_health = {}
    flight_data = summarize_menapia_flight(archive_health, now=now)
    flight_level = {
        "green": "ok",
        "amber": "warn",
        "red": "warn",
    }.get(flight_data["level"], "info")
    latest_flight = " / ".join(
        value
        for value in (
            flight_data.get("latestSourceDate"),
            flight_data.get("latestSourceFlight"),
        )
        if value
    ) or "--"
    credential_days = flight_data.get("credentialDaysRemaining")

    cards = [
        ("Current effective tier", str(latest.effective_tier) if latest else "--", current_level),
        ("Reported tier", str(latest.reported_tier) if latest else "--", current_level),
        ("Last update", _format_status_time(latest.timestamp if latest else None), current_level),
        ("Log age", _format_duration(age), current_level),
        ("Records in window", str(len(records)), "info"),
        ("Parse status", parse_label, parse_level),
        ("Flight-data ingest", str(flight_data["title"]), flight_level),
        ("Latest source flight", latest_flight, flight_level),
        ("Source objects", f"{flight_data['objectsExamined']:,}", "info"),
        ("GWS pending", f"{flight_data['gwsPendingFiles']:,}", flight_level),
        (
            "Object-store pending",
            f"{flight_data['objectStorePendingFiles']:,}",
            flight_level,
        ),
        (
            "Credential",
            "--" if credential_days is None else f"{credential_days} d remaining",
            flight_level,
        ),
    ]
    card_markup = "".join(
        (
            f"<div class='uas-card uas-card--{level}'>"
            f"<div class='uas-card__label'>{escape(label)}</div>"
            f"<div class='uas-card__value'>{escape(value)}</div>"
            "</div>"
        )
        for label, value, level in cards
    )
    detail = result.error or (result.malformed_lines[0] if result.malformed_lines else "")
    if result.missing:
        detail = f"Waiting for mirrored log at {result.path}"
    diagnostic_items = [
        ("Source", str(result.path), "info"),
        ("Window", uas_window.value, "info"),
        ("Malformed lines", str(len(result.malformed_lines)), "warn" if result.malformed_lines else "ok"),
        ("Flight data", str(flight_data["detail"]), flight_level),
    ]
    if detail:
        diagnostic_items.append(("Detail", detail[:160], "warn"))
    return f"<div class='uas-grid'>{card_markup}</div>{_status_strip_markup(diagnostic_items)}"


def _uas_history_figure(records: tuple[UASMqttRecord, ...], result: UASMqttParseResult) -> go.Figure:
    start, end = _uas_window_bounds()
    fig = go.Figure()
    if records:
        times = [record.timestamp for record in records]
        fig.add_trace(
            go.Scatter(
                x=times,
                y=[record.effective_tier for record in records],
                mode="lines+markers",
                line=dict(color=THEME_ACCENT, width=3, shape="hv"),
                marker=dict(size=6),
                name="Effective tier",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=times,
                y=[record.reported_tier for record in records],
                mode="lines",
                line=dict(color="#8aa0b4", width=2, dash="dot", shape="hv"),
                name="Reported tier",
            )
        )
        change_records = [record for record in records if record.event_type == "tier_change"]
        if change_records:
            fig.add_trace(
                go.Scatter(
                    x=[record.timestamp for record in change_records],
                    y=[record.effective_tier for record in change_records],
                    mode="markers",
                    marker=dict(color="#e0b15c", size=12, symbol="diamond"),
                    name="Tier change",
                )
            )
    else:
        reason = "No UAS records in the selected window."
        if result.missing:
            reason = "UAS MQTT log has not been mirrored yet."
        elif result.error:
            reason = f"Could not read UAS MQTT log: {result.error}"
        fig.add_annotation(text=escape(reason), x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(color=THEME_MUTED, size=14))
    fig.update_layout(
        title=dict(text="UAS Tier History", x=0.01, xanchor="left", font=dict(size=17, color=THEME_TEXT)),
        height=360,
        margin=dict(l=70, r=30, t=60, b=70),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color=THEME_TEXT, size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1.0),
    )
    fig.update_xaxes(title_text="Time (UTC)", range=[start, end], showgrid=True, gridcolor=THEME_GRID, linecolor=THEME_LINE)
    fig.update_yaxes(title_text="Tier", dtick=1, showgrid=True, gridcolor=THEME_GRID, linecolor=THEME_LINE)
    return fig


def _uas_log_table_markup(result: UASMqttParseResult) -> str:
    rows = []
    for record in reversed(result.records[-30:]):
        event_label = "Tier change" if record.event_type == "tier_change" else "Sample"
        rows.append(
            "<tr>"
            f"<td>{escape(record.timestamp.strftime('%Y-%m-%d %H:%M:%S'))}</td>"
            f"<td>{escape(event_label)}</td>"
            f"<td>{record.reported_tier}</td>"
            f"<td>{record.effective_tier}</td>"
            f"<td>{escape(record.raw)}</td>"
            "</tr>"
        )
    body = "".join(rows) if rows else "<tr><td colspan='5'>No parsed UAS log records are available.</td></tr>"
    return (
        "<div class='uas-table-wrap'>"
        "<div class='uas-section-title'>Latest Log Lines</div>"
        "<table class='uas-table'>"
        "<thead><tr><th>Timestamp UTC</th><th>Event</th><th>Reported</th><th>Effective</th><th>Raw line</th></tr></thead>"
        f"<tbody>{body}</tbody>"
        "</table>"
        "</div>"
    )


def _uas_flight_day_options(available_days: list[str]) -> dict[str, str]:
    options = {"Latest available": "latest"}
    for day in available_days:
        parsed = pd.to_datetime(day, errors="coerce")
        label = parsed.strftime("%d %B %Y") if not pd.isna(parsed) else day
        options[f"{label} (UTC)"] = day
    return options


def _uas_flight_label(flight: dict) -> str:
    start = pd.to_datetime(flight.get("startTimeUTC"), errors="coerce", utc=True)
    start_label = start.strftime("%H:%M UTC") if not pd.isna(start) else "Time unknown"
    duration = flight.get("durationSeconds")
    try:
        duration_label = f"{float(duration) / 60:.1f} min"
    except (TypeError, ValueError):
        duration_label = "duration unknown"
    title = str(flight.get("title") or f"Flight {flight.get('flightNumber') or ''}").strip()
    return f"{start_label} - {title} ({duration_label})"


def _uas_empty_flight_figure(message: str, *, title: str = "UAS Flight Profiles") -> go.Figure:
    figure = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.035)
    figure.add_annotation(
        text=escape(message),
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(color=THEME_MUTED, size=15),
    )
    figure.update_layout(
        title=dict(text=title, x=0.01, xanchor="left", font=dict(size=18, color=THEME_TEXT)),
        height=760,
        margin=dict(l=85, r=30, t=75, b=70),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color=THEME_TEXT, size=12),
    )
    for row in range(1, 5):
        figure.update_xaxes(showgrid=True, gridcolor=THEME_GRID, linecolor=THEME_LINE, row=row, col=1)
        figure.update_yaxes(showgrid=True, gridcolor=THEME_GRID, linecolor=THEME_LINE, row=row, col=1)
    return figure


def _uas_flight_figure(detail: dict) -> go.Figure:
    flight = detail.get("flight") or {}
    series = detail.get("series") or {}
    times = pd.to_datetime(series.get("timeUTC") or [], errors="coerce", utc=True)
    if not len(times):
        return _uas_empty_flight_figure("The selected flight contains no profile samples.")
    figure = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.035)
    sensor_styles = {
        "SN0122": dict(color="#2b6cb0", width=2.2),
        "SN0123": dict(color="#ed7d0b", width=2.2, dash="dash"),
    }
    groups = (
        ("temperatureC", "Temperature (°C)"),
        ("pressureHpa", "Pressure (hPa)"),
        ("relativeHumidityPct", "Relative humidity (%)"),
    )
    for row, (group_key, axis_title) in enumerate(groups, start=1):
        group = series.get(group_key) or {}
        for sensor, style in sensor_styles.items():
            figure.add_trace(
                go.Scatter(
                    x=times,
                    y=group.get(sensor) or [],
                    mode="lines+markers" if group_key == "pressureHpa" else "lines",
                    name=sensor,
                    legendgroup=sensor,
                    showlegend=row == 1,
                    line=style,
                    marker=dict(size=4, color=style["color"]),
                    connectgaps=False,
                    hovertemplate=f"%{{x|%Y-%m-%d %H:%M:%S}} UTC<br>{axis_title}: %{{y:.2f}}<extra>{sensor}</extra>",
                ),
                row=row,
                col=1,
            )
        figure.update_yaxes(title_text=axis_title, row=row, col=1)
    figure.add_trace(
        go.Scatter(
            x=times,
            y=series.get("altitudeM") or [],
            mode="lines",
            name="Fused altitude",
            line=dict(color="#37474f", width=2.3),
            connectgaps=False,
            hovertemplate="%{x|%Y-%m-%d %H:%M:%S} UTC<br>Fused altitude: %{y:.2f} m<extra></extra>",
        ),
        row=4,
        col=1,
    )
    figure.update_yaxes(title_text="Fused altitude (m)", row=4, col=1)
    for row in range(1, 5):
        figure.update_xaxes(showgrid=True, gridcolor=THEME_GRID, linecolor=THEME_LINE, row=row, col=1)
        figure.update_yaxes(showgrid=True, gridcolor=THEME_GRID, linecolor=THEME_LINE, row=row, col=1)
    figure.update_xaxes(title_text="Time (UTC)", row=4, col=1)
    start = pd.to_datetime(flight.get("startTimeUTC"), errors="coerce", utc=True)
    start_label = start.strftime("%d %B %Y, %H:%M:%S UTC") if not pd.isna(start) else "Start time unknown"
    duration = flight.get("durationSeconds")
    try:
        duration_label = f"{float(duration) / 60:.1f} min"
    except (TypeError, ValueError):
        duration_label = "duration unknown"
    title = str(flight.get("title") or "Menapia flight")
    figure.update_layout(
        title=dict(
            text=f"{escape(title)}<br><sup>{escape(start_label)} | {escape(duration_label)} | one-second medians</sup>",
            x=0.01,
            xanchor="left",
            font=dict(size=18, color=THEME_TEXT),
        ),
        height=920,
        margin=dict(l=95, r=30, t=90, b=70),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color=THEME_TEXT, size=12),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1.0),
        uirevision=f"uas-flight-{flight.get('id') or 'none'}",
    )
    return figure


def _uas_flight_status_markup(listing: dict, detail: dict | None = None, error: str | None = None) -> str:
    status = listing.get("status") or {}
    flights = listing.get("flights") or []
    flight = (detail or {}).get("flight") or next(
        (item for item in flights if item.get("id") == uas_flight_select.value),
        {},
    )
    generated = pd.to_datetime(listing.get("generatedAt"), errors="coerce", utc=True)
    generated_label = "--" if pd.isna(generated) else generated.strftime("%Y-%m-%d %H:%M UTC")
    last_run = pd.to_datetime(
        listing.get("lastRunAt") or listing.get("generatedAt"),
        errors="coerce",
        utc=True,
    )
    last_run_label = "--" if pd.isna(last_run) else last_run.strftime("%Y-%m-%d %H:%M UTC")
    start = pd.to_datetime(flight.get("startTimeUTC"), errors="coerce", utc=True)
    start_label = "--" if pd.isna(start) else start.strftime("%Y-%m-%d %H:%M:%S UTC")
    try:
        duration_label = f"{float(flight.get('durationSeconds')) / 60:.1f} min"
    except (TypeError, ValueError):
        duration_label = "--"
    quality = flight.get("quality") if isinstance(flight.get("quality"), dict) else {}
    quality_level = str(quality.get("level") or "unknown")
    level_class = "ok" if status.get("level") == "green" else "warn"
    quality_class = "ok" if quality_level == "green" else "warn" if quality_level in {"amber", "red"} else "info"
    cards = [
        ("Product state", str(status.get("title") or "Unknown"), level_class),
        ("Selected UTC day", str(listing.get("selectedDay") or "--"), "info"),
        ("Flights on day", str(len(flights)), "ok" if flights else "warn"),
        ("Selected flight", str(flight.get("title") or "--"), quality_class),
        ("Flight start", start_label, "info"),
        ("Duration", duration_label, "info"),
        ("Data quality", quality_level.title(), quality_class),
        ("Last product build", last_run_label, level_class),
        ("Product content", generated_label, "info"),
    ]
    card_markup = "".join(
        (
            f"<div class='uas-card uas-card--{level}'>"
            f"<div class='uas-card__label'>{escape(label)}</div>"
            f"<div class='uas-card__value'>{escape(value)}</div>"
            "</div>"
        )
        for label, value, level in cards
    )
    diagnostics = [
        ("Refresh state", str(status.get("state") or "unknown"), level_class),
        ("Aggregation", "One-second medians; no interpolation", "info"),
    ]
    detail_text = error or str(status.get("detail") or "")
    if detail_text:
        diagnostics.append(("Detail", detail_text[:300], "warn" if error or level_class == "warn" else "info"))
    warnings = quality.get("warnings") if isinstance(quality.get("warnings"), list) else []
    if warnings:
        diagnostics.append(("Quality note", "; ".join(str(item) for item in warnings)[:300], "warn"))
    return (
        "<div class='uas-section-title'>Menapia Flight Profiles</div>"
        f"<div class='uas-grid'>{card_markup}</div>"
        f"{_status_strip_markup(diagnostics)}"
    )


uas_status_pane = pn.pane.HTML("", sizing_mode="stretch_width", margin=0)
uas_plot_pane = pn.pane.Plotly(config={"responsive": True}, sizing_mode="stretch_width", margin=0)
uas_table_pane = pn.pane.HTML("", sizing_mode="stretch_width", margin=0)
uas_flight_status_pane = pn.pane.HTML("", sizing_mode="stretch_width", margin=0)
uas_flight_plot_pane = pn.pane.Plotly(
    config={"responsive": True, "scrollZoom": True, "displaylogo": False},
    sizing_mode="stretch_width",
    margin=0,
    css_classes=["uas-flight-plot"],
)


def _refresh_uas_selected_flight(listing: dict | None = None) -> None:
    listing = listing or _uas_flight_listing
    selected = uas_flight_select.value
    if not selected:
        status = listing.get("status") or {}
        message = str(status.get("detail") or "No flights are available for the selected UTC day.")
        uas_flight_status_pane.object = _uas_flight_status_markup(listing)
        uas_flight_plot_pane.object = _uas_empty_flight_figure(message)
        return
    try:
        detail = mobile_catalog.uas_flight(selected)
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        message = str(exc.args[0] if isinstance(exc, KeyError) and exc.args else exc)
        uas_flight_status_pane.object = _uas_flight_status_markup(listing, error=message)
        uas_flight_plot_pane.object = _uas_empty_flight_figure(message)
        return
    uas_flight_status_pane.object = _uas_flight_status_markup(listing, detail=detail)
    uas_flight_plot_pane.object = _uas_flight_figure(detail)


def _refresh_uas_flight_products(*, preserve_selection: bool = True) -> None:
    global _uas_flight_widget_guard, _uas_flight_listing, _uas_flight_manual_selection
    global _UAS_PENDING_DAY_QUERY, _UAS_PENDING_FLIGHT_QUERY
    if _uas_flight_widget_guard:
        return
    _uas_flight_widget_guard = True
    try:
        requested_day = uas_flight_day.value or "latest"
        listing = mobile_catalog.uas_flights(day=requested_day)
        day_options = _uas_flight_day_options(listing.get("availableDays") or [])
        allowed_days = set(day_options.values())
        pending_day = _UAS_PENDING_DAY_QUERY
        if pending_day in allowed_days and pending_day != requested_day:
            requested_day = pending_day
            listing = mobile_catalog.uas_flights(day=requested_day)
        elif requested_day not in allowed_days:
            requested_day = "latest"
            listing = mobile_catalog.uas_flights(day=requested_day)
        uas_flight_day.options = day_options
        uas_flight_day.value = requested_day
        _UAS_PENDING_DAY_QUERY = None

        flights = listing.get("flights") or []
        flight_options = {
            _uas_flight_label(flight): flight["id"]
            for flight in flights
            if flight.get("id")
        }
        if not flight_options:
            flight_options = {"No flights available": None}
        available_ids = set(flight_options.values())
        current = uas_flight_select.value
        pending_flight = _UAS_PENDING_FLIGHT_QUERY
        if pending_flight in available_ids:
            selected = pending_flight
            _uas_flight_manual_selection = True
        elif preserve_selection and _uas_flight_manual_selection and current in available_ids:
            selected = current
        elif requested_day == "latest" and listing.get("latestFlightID") in available_ids:
            selected = listing.get("latestFlightID")
            _uas_flight_manual_selection = False
        elif preserve_selection and current in available_ids:
            selected = current
        else:
            selected = next((value for value in flight_options.values() if value is not None), None)
            _uas_flight_manual_selection = False
        uas_flight_select.options = flight_options
        uas_flight_select.value = selected
        uas_flight_select.disabled = selected is None
        _UAS_PENDING_FLIGHT_QUERY = None
        _uas_flight_listing = listing
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        message = str(exc.args[0] if isinstance(exc, KeyError) and exc.args else exc)
        _uas_flight_listing = {
            "selectedDay": uas_flight_day.value,
            "flights": [],
            "status": {
                "state": "error",
                "level": "red",
                "title": "Flight products unavailable",
                "detail": message,
            },
        }
        uas_flight_select.options = {"Flight products unavailable": None}
        uas_flight_select.value = None
        uas_flight_select.disabled = True
        _uas_flight_manual_selection = False
    finally:
        _uas_flight_widget_guard = False
    _refresh_uas_selected_flight(_uas_flight_listing)


def _refresh_uas_dashboard(_event=None) -> None:
    result = _uas_load_result()
    records = _uas_window_records(result)
    uas_status_pane.object = _uas_summary_markup(result, records)
    uas_plot_pane.object = _uas_history_figure(records, result)
    uas_table_pane.object = _uas_log_table_markup(result)
    _refresh_uas_flight_products(preserve_selection=True)


uas_refresh.on_click(_refresh_uas_dashboard)
uas_window.param.watch(lambda _event: _refresh_uas_dashboard(), "value")


def _on_uas_flight_day_change(_event) -> None:
    global _uas_flight_manual_selection
    if _uas_flight_widget_guard:
        return
    _uas_flight_manual_selection = False
    _refresh_uas_flight_products(preserve_selection=False)


def _on_uas_flight_selection_change(_event) -> None:
    global _uas_flight_manual_selection
    if _uas_flight_widget_guard:
        return
    _uas_flight_manual_selection = True
    _refresh_uas_selected_flight()


uas_flight_day.param.watch(
    _on_uas_flight_day_change,
    "value",
)
uas_flight_select.param.watch(
    _on_uas_flight_selection_change,
    "value",
)


interactive_status = pn.bind(
    lambda *_deps: _current_interactive_status_markup(),
    instrument_select.param.value,
    range_start.param.value,
    range_end.param.value,
    live_toggle.param.value,
    wxcam_image_type.param.value,
    wxcam_date.param.value,
    var1_select.param.value,
    var2_select.param.value,
    bottom_range_m.param.value,
    top_range_m.param.value,
)
interactive_availability = pn.bind(
    lambda *_deps: _current_interactive_availability_markup(),
    instrument_select.param.value,
    range_start.param.value,
    range_end.param.value,
    live_toggle.param.value,
    wxcam_image_type.param.value,
    wxcam_date.param.value,
    var1_select.param.value,
    var2_select.param.value,
    bottom_range_m.param.value,
    top_range_m.param.value,
)
science_status = pn.bind(
    lambda *_deps: _current_science_status_markup(),
    science_instrument.param.value,
    ql_date.param.value,
    science_image_type.param.value,
    wxcam_calendar_state.param.selected_hour_path,
)
science_availability = pn.bind(
    lambda *_deps: _current_science_availability_markup(),
    science_instrument.param.value,
    ql_date.param.value,
    science_image_type.param.value,
    wxcam_calendar_state.param.selected_hour_path,
)
hk_status = pn.bind(
    lambda *_deps: _current_hk_status_markup(),
    hk_instrument.param.value,
    hk_date.param.value,
)
hk_availability = pn.bind(
    lambda *_deps: _current_hk_availability_markup(),
    hk_instrument.param.value,
    hk_date.param.value,
)


interactive_share_url = pn.widgets.TextInput(name="Share link", value="", sizing_mode="stretch_width")
science_share_url = pn.widgets.TextInput(name="Share link", value="", sizing_mode="stretch_width")
hk_share_url = pn.widgets.TextInput(name="Share link", value="", sizing_mode="stretch_width")
auroracam_share_url = pn.widgets.TextInput(name="Share link", value="", sizing_mode="stretch_width")
uas_share_url = pn.widgets.TextInput(name="Share link", value="", sizing_mode="stretch_width")
interactive_copy = pn.widgets.Button(name="Copy link", button_type="default", width=110)
science_copy = pn.widgets.Button(name="Copy link", button_type="default", width=110)
hk_copy = pn.widgets.Button(name="Copy link", button_type="default", width=110)
auroracam_copy = pn.widgets.Button(name="Copy link", button_type="default", width=110)
uas_copy = pn.widgets.Button(name="Copy link", button_type="default", width=110)
interactive_download = pn.widgets.Button(name="Download PNG", button_type="default", width=130)
science_download = pn.widgets.FileDownload(name="", label="Download PNG", button_type="default", auto=False, embed=False, width=130)
hk_download = pn.widgets.FileDownload(name="", label="Download PNG", button_type="default", auto=False, embed=False, width=130)


for button, widget in (
    (interactive_copy, interactive_share_url),
    (science_copy, science_share_url),
    (hk_copy, hk_share_url),
    (auroracam_copy, auroracam_share_url),
    (uas_copy, uas_share_url),
):
    button.js_on_click(
        args={"share": widget},
        code="""
        const text = share.value || '';
        if (!text) { return; }
        navigator.clipboard.writeText(text);
        """,
    )


interactive_download.js_on_click(
    args={"inst": instrument_select},
    code="""
    const plot = Array.from(document.querySelectorAll('.js-plotly-plot')).find((el) => el.offsetParent !== null);
    if (!plot || !window.Plotly) { return; }
    const base = (inst.value || 'aurora').toLowerCase().replace(/[^a-z0-9]+/g, '_');
    window.Plotly.downloadImage(plot, {format: 'png', filename: `${base}_interactive`});
    """,
)


def _science_download_path() -> Path | None:
    inst = science_instrument.value
    if _is_wxcam_instrument(inst):
        if wxcam_calendar_state.selected_hour_path:
            path = Path(wxcam_calendar_state.selected_hour_path)
            return path if path.exists() else None
        return None
    if _is_stacked_timeseries_instrument(inst):
        token = _quicklook_options(inst, mode="science").get(ql_date.value)
        if token is None:
            return None
        quick_dir = _cfg(inst)["quicklook_dir"]
        return summary_latest_png(quick_dir, inst) if token == "latest" else summary_daily_png(quick_dir, inst, token)
    raw_path = _quicklook_options(inst, mode="science").get(ql_date.value)
    if not raw_path:
        return None
    path = Path(raw_path)
    return path if path.exists() else None


def _hk_download_path() -> Path | None:
    inst = hk_instrument.value
    token = _quicklook_options(inst, mode="housekeeping").get(hk_date.value)
    if token is None:
        return None
    quick_dir = _cfg(inst)["quicklook_dir"]
    path = _housekeeping_latest_path(inst, quick_dir) if token == "latest" else _housekeeping_daily_path(inst, quick_dir, token)
    return path if path and path.exists() else None


def _view_query_params(tab_slug: str) -> dict[str, str]:
    params: dict[str, str] = {
        "tab": tab_slug,
        "instrument": instrument_select.value,
    }
    if tab_slug in {"interactive", "power"}:
        params["start"] = range_start.value.isoformat() if range_start.value else ""
        params["end"] = range_end.value.isoformat() if range_end.value else ""
        params["live"] = "1" if live_toggle.value else "0"
        if tab_slug == "power":
            params["power_view"] = power_view_select.value
        if _is_wxcam_instrument(instrument_select.value):
            params["wxcam_image_type"] = wxcam_image_type.value or ""
            params["wxcam_date"] = wxcam_date.value or ""
        else:
            params["top_var"] = var1_select.value or ""
            params["bottom_var"] = var2_select.value or ""
            params["bottom_m"] = str(bottom_range_m.value)
            params["top_m"] = str(top_range_m.value)
    elif tab_slug == "science":
        params["science_instrument"] = science_instrument.value
        params["science_date"] = ql_date.value or ""
        params["science_image_type"] = science_image_type.value or ""
        if wxcam_calendar_state.selected_hour_path:
            params["science_selected_hour"] = wxcam_calendar_state.selected_hour_path
    elif tab_slug == "housekeeping":
        params["hk_instrument"] = hk_instrument.value
        params["hk_date"] = hk_date.value or ""
    elif tab_slug == "auroracam":
        params["auroracam_camera"] = auroracam_camera.value or ""
        params["auroracam_date"] = auroracam_date.value or ""
        params["auroracam_time"] = auroracam_time.value or ""
    elif tab_slug == "uas":
        params["uas_window"] = uas_window.value or ""
        params["uas_day"] = _UAS_PENDING_DAY_QUERY or uas_flight_day.value or "latest"
        params["uas_flight"] = _UAS_PENDING_FLIGHT_QUERY or uas_flight_select.value or ""
    return {k: v for k, v in params.items() if v not in ("", None)}


def _build_share_url(tab_slug: str) -> str:
    query = urlencode(_view_query_params(tab_slug))
    return f"{_request_base_url()}?{query}" if query else _request_base_url()


def _active_tab_slug() -> str:
    return globals().get("ACTIVE_TAB_SLUG", "interactive")


def _update_browser_location() -> None:
    """Keep the address bar aligned with the current view for mobile reconnects."""
    try:
        location = pn.state.location
    except Exception:
        location = None
    if location is None:
        return
    search = "?" + urlencode(_view_query_params(_active_tab_slug()))
    if getattr(location, "search", None) != search:
        location.search = search


def _refresh_share_and_download_state(*_events) -> None:
    browser_tab = "power" if _active_tab_slug() == "power" else "interactive"
    interactive_share_url.value = _build_share_url(browser_tab)
    science_share_url.value = _build_share_url("science")
    hk_share_url.value = _build_share_url("housekeeping")
    auroracam_share_url.value = _build_share_url("auroracam")
    uas_share_url.value = _build_share_url("uas")
    _update_browser_location()

    interactive_download.visible = not _is_wxcam_instrument(instrument_select.value)

    science_path = _science_download_path()
    science_download.file = science_path
    science_download.filename = science_path.name if science_path else "science_quicklook.png"
    science_download.label = "Download PNG"
    science_download.disabled = science_path is None

    hk_path = _hk_download_path()
    hk_download.file = hk_path
    hk_download.filename = hk_path.name if hk_path else "housekeeping_quicklook.png"
    hk_download.label = "Download PNG"
    hk_download.disabled = hk_path is None


def _query_interactive_time_state(args: dict[str, str], instrument: str):
    """Resolve URL time state, treating live URLs as requests for the current window."""
    if _is_wxcam_instrument(instrument):
        return None
    live = args.get("live", "0") == "1"
    if live:
        start, end = _last_24h_utc_window()
        return start, end, True
    start_raw = args.get("start")
    end_raw = args.get("end")
    if not start_raw or not end_raw:
        return None
    try:
        start = pd.Timestamp(start_raw).to_pydatetime(warn=False)
        end = pd.Timestamp(end_raw).to_pydatetime(warn=False)
    except Exception:
        return None
    return start, end, False


def _apply_query_state() -> None:
    global _AURORACAM_PENDING_TIME_QUERY
    global _UAS_PENDING_DAY_QUERY, _UAS_PENDING_FLIGHT_QUERY
    args = _request_query_args()
    if not args:
        return
    visible_instruments = set(INSTRUMENT_OPTIONS.values())
    science_visible_instruments = set(SCIENCE_INSTRUMENT_OPTIONS.values())
    hk_visible_instruments = set(HK_INSTRUMENT_OPTIONS.values())
    # A direct Power URL owns the shared interactive component. Apply this
    # before its generic instrument parameter so tab switches cannot fall back
    # to the previous Ceilometer selection during initial callback wiring.
    instrument = "power" if args.get("tab") == "power" else args.get("instrument")
    if instrument in visible_instruments:
        instrument_select.value = instrument
    time_state = _query_interactive_time_state(args, instrument_select.value)
    if time_state is not None:
        start, end, live = time_state
        range_start.value = start
        range_end.value = end
        _set_live(live)
    if args.get("bottom_m"):
        try:
            bottom_range_m.value = int(args["bottom_m"])
        except Exception:
            pass
    if args.get("top_m"):
        try:
            top_range_m.value = int(args["top_m"])
        except Exception:
            pass
    if args.get("top_var") in _cfg(instrument_select.value)["vars"]:
        var1_select.value = args["top_var"]
    if args.get("bottom_var") in _cfg(instrument_select.value)["vars"]:
        var2_select.value = args["bottom_var"]
    if args.get("wxcam_image_type") in list(wxcam_image_type.options):
        wxcam_image_type.value = args["wxcam_image_type"]
    if args.get("wxcam_date") in list(wxcam_date.options):
        wxcam_date.value = args["wxcam_date"]
    if args.get("science_instrument") in science_visible_instruments:
        science_instrument.value = args["science_instrument"]
    if args.get("science_image_type") in list(science_image_type.options):
        science_image_type.value = args["science_image_type"]
    if args.get("science_date") in list(ql_date.options):
        ql_date.value = args["science_date"]
    if args.get("science_selected_hour"):
        wxcam_calendar_state.selected_hour_path = args["science_selected_hour"]
    if args.get("hk_instrument") in hk_visible_instruments:
        hk_instrument.value = args["hk_instrument"]
    if args.get("hk_date") in list(hk_date.options):
        hk_date.value = args["hk_date"]
    if args.get("auroracam_camera") in list(AURORACAM_CAMERAS):
        auroracam_camera.value = args["auroracam_camera"]
    if args.get("auroracam_date") in list(auroracam_date.options):
        auroracam_date.value = args["auroracam_date"]
    requested_auroracam_time = args.get("auroracam_time")
    if requested_auroracam_time in list(auroracam_time.options):
        auroracam_time.value = requested_auroracam_time
    elif requested_auroracam_time:
        # Historic times are loaded only when the camera tab is activated.
        # Hold this value so direct/share URLs still select their frame then.
        _AURORACAM_PENDING_TIME_QUERY = requested_auroracam_time
    if args.get("uas_window") in list(uas_window.options):
        uas_window.value = args["uas_window"]
    requested_uas_day = args.get("uas_day")
    if requested_uas_day == "latest" or (
        isinstance(requested_uas_day, str)
        and len(requested_uas_day) == 10
        and requested_uas_day[4:5] == "-"
        and requested_uas_day[7:8] == "-"
        and requested_uas_day.replace("-", "").isdigit()
    ):
        _UAS_PENDING_DAY_QUERY = requested_uas_day
    requested_uas_flight = args.get("uas_flight")
    if (
        isinstance(requested_uas_flight, str)
        and 1 <= len(requested_uas_flight) <= 128
        and all(character.isalnum() or character in "_-" for character in requested_uas_flight)
    ):
        _UAS_PENDING_FLIGHT_QUERY = requested_uas_flight
    if args.get("power_view") in {"current", "forecast"}:
        power_view_select.value = args["power_view"]


for widget, parameter in (
    (instrument_select, "value"),
    (power_view_select, "value"),
    (range_start, "value"),
    (range_end, "value"),
    (live_toggle, "value"),
    (var1_select, "value"),
    (var2_select, "value"),
    (bottom_range_m, "value"),
    (top_range_m, "value"),
    (wxcam_image_type, "value"),
    (wxcam_date, "value"),
    (science_instrument, "value"),
    (ql_date, "value"),
    (science_image_type, "value"),
    (hk_instrument, "value"),
    (hk_date, "value"),
    (auroracam_camera, "value"),
    (auroracam_date, "value"),
    (auroracam_time, "value"),
    (uas_window, "value"),
    (uas_flight_day, "value"),
    (uas_flight_select, "value"),
):
    widget.param.watch(_refresh_share_and_download_state, parameter)

wxcam_calendar_state.param.watch(_refresh_share_and_download_state, "selected_hour_path")


ACCENT = "#003155"  # stable MaterialTemplate header; gradients are applied inside views


_custom_cl61_start_default = (
    pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None).floor("h") + pd.Timedelta(hours=6)
).to_pydatetime()
custom_cl61_start = pn.widgets.DatetimePicker(
    name="Instrument start (UTC)",
    value=_custom_cl61_start_default,
    sizing_mode="stretch_width",
)
custom_cl61_duration = pn.widgets.IntSlider(
    name="Instrument run duration (hours)",
    start=1,
    end=96,
    step=1,
    value=MIN_RUN_HOURS,
    sizing_mode="stretch_width",
)
custom_plan_instrument = pn.widgets.Select(
    name="Instrument",
    options=list(KIT_ORDER),
    value="CL61",
    sizing_mode="stretch_width",
)


def _operating_plan_status(minimum_p10_soc: float) -> tuple[str, str]:
    if minimum_p10_soc >= MINIMUM_OPERATIONAL_SOC_PCT:
        return "Safe", "green"
    if minimum_p10_soc >= MINIMUM_OPERATIONAL_SOC_PCT - 5.0:
        return "Marginal", "amber"
    return "Unsafe", "red"


def _operating_plan_metric(label: str, value: str) -> str:
    return (
        "<div class='operating-plan-metric'>"
        f"<div class='operating-plan-metric__label'>{escape(label)}</div>"
        f"<div class='operating-plan-metric__value'>{escape(value)}</div>"
        "</div>"
    )


def _add_custom_schedule_bands(figure: go.Figure, result: dict) -> None:
    for start, end, label, color in operating_mode_intervals(
        pd.DatetimeIndex(result["time"]),
        np.asarray(result["mode_codes"]),
    ):
        figure.add_vrect(
            x0=start,
            x1=end,
            fillcolor=color,
            opacity=1.0,
            line_width=0,
            layer="below",
            annotation_text=f"{label} on",
            annotation_position="top left",
            annotation_font_size=9,
            annotation_font_color=THEME_TEXT,
            row="all",
            col=1,
        )


def _build_custom_cl61_plan_view(start_value, duration_value, kit_value):
    scenarios = _get_power_operating_scenarios_dataset()
    if scenarios is None:
        return pn.pane.Alert("Operating-plan forecast is not available.", alert_type="warning")
    try:
        result = evaluate_custom_schedule(
            scenarios,
            start_time=pd.Timestamp(start_value),
            duration_hours=max(int(duration_value), MIN_RUN_HOURS if str(kit_value) == "CL61" else 1),
            kit=str(kit_value),
        )
    except Exception as exc:
        return pn.pane.Alert(f"Could not calculate the custom operating plan: {exc}", alert_type="danger")

    minimum_p10 = float(result["minimum_p10_soc"])
    status_label, status_level = _operating_plan_status(minimum_p10)
    current_mode = str(scenarios.attrs.get("current_mode_label", "Unknown"))
    mode_maturity = str(scenarios.attrs.get("current_mode_maturity", "observed")).replace("_", " ").title()
    confidence = float(scenarios.attrs.get("current_mode_confidence", "nan"))
    confidence_text = f"{confidence * 100:.0f}%" if np.isfinite(confidence) else "Unknown"
    metrics = (
        f"<div class='operating-plan-status operating-plan-status--{status_level}'>{status_label} | Advisory only</div>"
        "<div class='operating-plan-metrics'>"
        + _operating_plan_metric("Detected mode", current_mode)
        + _operating_plan_metric("Mode evidence", mode_maturity)
        + _operating_plan_metric("Mode confidence", confidence_text)
        + _operating_plan_metric(f"{result['kit']} collection", f"{float(result['collection_hours']):.0f} h")
        + _operating_plan_metric("Minimum P10 SOC", f"{minimum_p10:.1f}%")
        + _operating_plan_metric("Final P10 SOC", f"{float(result['final_p10_soc']):.1f}%")
        + "</div>"
    )

    times = pd.DatetimeIndex(result["time"])
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        row_heights=(0.68, 0.32),
    )
    for name, label, color, dash in (
        ("soc_p10", "SOC P10", "#7fb6d6", "dot"),
        ("soc_p50", "SOC Median", "#4f8c63", "solid"),
        ("soc_p90", "SOC P90", "#7fb6d6", "dot"),
    ):
        figure.add_trace(
            go.Scatter(x=times, y=result[name], mode="lines", name=label, line=dict(color=color, width=2, dash=dash)),
            row=1,
            col=1,
        )
    figure.add_trace(
        go.Scatter(
            x=[times.min(), times.max()],
            y=[MINIMUM_OPERATIONAL_SOC_PCT, MINIMUM_OPERATIONAL_SOC_PCT],
            mode="lines",
            name=MINIMUM_OPERATIONAL_SOC_REFERENCE_LABEL,
            line=dict(color=THEME_TEXT, width=1.5, dash="dash"),
        ),
        row=1,
        col=1,
    )
    _add_custom_schedule_bands(figure, result)
    figure.add_trace(
        go.Scatter(x=times, y=result["load_p50_w"], mode="lines", name="Forecast Load", line=dict(color="#c05647", width=2)),
        row=2,
        col=1,
    )
    figure.update_yaxes(title_text="State of Charge [%]", range=[0, 100], row=1, col=1)
    figure.update_yaxes(title_text="Load [W]", rangemode="tozero", row=2, col=1)
    figure.update_xaxes(title_text="Date and Time (UTC)", row=2, col=1)
    figure.update_layout(
        height=520,
        margin=dict(l=70, r=30, t=20, b=55),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color=THEME_TEXT, size=12),
        hovermode="x unified",
    )
    figure.update_xaxes(showgrid=True, gridcolor=THEME_GRID)
    figure.update_yaxes(showgrid=True, gridcolor=THEME_GRID)
    return pn.Column(
        pn.pane.HTML(metrics, sizing_mode="stretch_width", margin=0),
        pn.pane.Plotly(figure, config={"responsive": True}, sizing_mode="stretch_width"),
        sizing_mode="stretch_width",
        margin=0,
    )


def _operating_decision_audit_view():
    """Show the latest auditable advisory plan and its accumulating hindcast."""
    path = _power_operating_recommendations_path()
    try:
        archive = json.loads(path.read_text(encoding="utf-8"))
        records = archive.get("recommendations", [])
        record = records[-1] if isinstance(records, list) and records else None
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        record = None
    if not isinstance(record, dict):
        return pn.pane.Alert("No archived 96 h operating decision is available yet.", alert_type="warning")

    verification = record.get("verification") if isinstance(record.get("verification"), dict) else {}
    status = str(verification.get("status", "awaiting measurements")).replace("_", " ").title()
    minimum_actual = verification.get("minimum_actual_soc_pct")
    mode_adherence = verification.get("mode_adherence_fraction")
    metrics = (
        "<div class='operating-plan-metrics'>"
        + _operating_plan_metric("Decision window", f"{int(record.get('decision_horizon_hours', 96))} h")
        + _operating_plan_metric("Planned CL61 collection", f"{float(record.get('collection_hours', 0.0)):.0f} h")
        + _operating_plan_metric("Forecast min P10 SOC", f"{float(record.get('minimum_p10_soc', float('nan'))):.1f}%")
        + _operating_plan_metric("Verification", status)
        + _operating_plan_metric(
            "SOC MAE so far",
            "Pending" if verification.get("soc_mae_pct") is None else f"{float(verification['soc_mae_pct']):.1f}%",
        )
        + _operating_plan_metric(
            "Actual min SOC",
            "Pending" if minimum_actual is None else f"{float(minimum_actual):.1f}%",
        )
        + _operating_plan_metric(
            "Mode adherence",
            "Pending" if mode_adherence is None else f"{100.0 * float(mode_adherence):.0f}%",
        )
        + "</div>"
    )
    windows = record.get("recommended_mode_windows", [])
    if isinstance(windows, list) and windows:
        schedule = " · ".join(
            f"{item.get('start_time_utc', '')[:13]}Z {item.get('mode_label', item.get('mode', 'code ' + str(item.get('mode_code', '?'))))}"
            for item in windows[:4]
            if isinstance(item, dict)
        )
    else:
        schedule = "No scheduled mode windows recorded"
    note = (
        "<div class='operating-plan-audit-note'>"
        f"Issued {escape(str(record.get('issued_at_utc', 'unknown')))}. {escape(schedule)}. "
        "The archive stores the hourly mode, load and SOC P10/P50/P90 trace for later comparison. Advisory only."
        "</div>"
    )
    return pn.Column(pn.pane.HTML(metrics + note, sizing_mode="stretch_width"), sizing_mode="stretch_width", margin=0)


@pn.depends(range_end.param.value)
def _operating_decision_audit_view_reactive(_live_refresh_anchor):
    return _operating_decision_audit_view()


@pn.depends(
    custom_cl61_start.param.value,
    custom_cl61_duration.param.value,
    custom_plan_instrument.param.value,
    range_end.param.value,
)
def _custom_cl61_plan_view(start_value, duration_value, kit_value, _live_refresh_anchor):
    return _build_custom_cl61_plan_view(start_value, duration_value, kit_value)


# Constructing this card evaluates the learned operating-plan product. It is a
# Forecast-only tool, so defer the work until the Forecast pane is requested.
power_plan_editor: pn.Card | None = None


def _get_power_plan_editor() -> pn.Card:
    global power_plan_editor
    if power_plan_editor is None:
        power_plan_editor = pn.Card(
            _operating_decision_audit_view_reactive,
            pn.Row(custom_plan_instrument, custom_cl61_start, custom_cl61_duration, sizing_mode="stretch_width", css_classes=["mobile-stack"]),
            _custom_cl61_plan_view,
            title="Custom Instrument Operating Plan",
            collapsible=True,
            collapsed=False,
            sizing_mode="stretch_width",
            css_classes=["small-card", "operating-plan-card"],
        )
    return power_plan_editor


power_plan_editor_container = pn.Column(sizing_mode="stretch_width", margin=0, visible=False)

mobile_custom_cl61_start = pn.widgets.DatetimePicker(
    name="Instrument start (UTC)",
    value=_custom_cl61_start_default,
    sizing_mode="stretch_width",
)
mobile_custom_cl61_duration = pn.widgets.IntSlider(
    name="Instrument run duration (hours)",
    start=1,
    end=96,
    step=1,
    value=MIN_RUN_HOURS,
    sizing_mode="stretch_width",
)
mobile_custom_plan_instrument = pn.widgets.Select(
    name="Instrument",
    options=list(KIT_ORDER),
    value="CL61",
    sizing_mode="stretch_width",
)


@pn.depends(
    mobile_custom_cl61_start.param.value,
    mobile_custom_cl61_duration.param.value,
    mobile_custom_plan_instrument.param.value,
    range_end.param.value,
)
def _mobile_custom_cl61_plan_view(start_value, duration_value, kit_value, _live_refresh_anchor):
    return _build_custom_cl61_plan_view(start_value, duration_value, kit_value)


mobile_power_plan_editor: pn.Card | None = None


def _get_mobile_power_plan_editor() -> pn.Card:
    global mobile_power_plan_editor
    if mobile_power_plan_editor is None:
        mobile_power_plan_editor = pn.Card(
            _operating_decision_audit_view_reactive,
            pn.Column(mobile_custom_plan_instrument, mobile_custom_cl61_start, mobile_custom_cl61_duration, sizing_mode="stretch_width"),
            _mobile_custom_cl61_plan_view,
            title="Custom Instrument Operating Plan",
            collapsible=True,
            collapsed=True,
            sizing_mode="stretch_width",
            css_classes=["operating-plan-card"],
        )
    return mobile_power_plan_editor



# The desktop browser keeps the compact row layout used by the stable site.
# Rows still wrap under the responsive CSS when a desktop link is forced on a
# narrow screen, while the dedicated mobile shell uses its own controls.
controls = pn.Card(
    pn.Column(
        pn.Row(
            instrument_select,
            range_start,
            range_end,
            live_toggle,
            sizing_mode="stretch_width",
            css_classes=["mobile-stack"],
        ),
        pn.Row(
            prev_btn,
            reset_view_btn,
            next_btn,
            sizing_mode="stretch_width",
            margin=(5, 0, 0, 0),
            css_classes=["mobile-stack"],
        ),
        pn.Row(var1_select, var2_select, sizing_mode="stretch_width", css_classes=["mobile-stack"]),
        pn.Row(bottom_range_m, top_range_m, sizing_mode="stretch_width", css_classes=["mobile-stack"]),
        pn.Row(beta_vmin, beta_vmax, ldr_vmin, ldr_vmax, sizing_mode="stretch_width", css_classes=["mobile-stack"]),
        pn.Row(
            lwp_ymin,
            lwp_ymax,
            iwv_ymin,
            iwv_ymax,
            irr_ymin,
            irr_ymax,
            sizing_mode="stretch_width",
            css_classes=["mobile-stack"],
        ),
        sizing_mode="stretch_width",
    ),
    title="Controls",
    collapsible=True,
    sizing_mode="stretch_width",
    css_classes=["small-card", "controls-card"],
)

# Styles are loaded once from /dashboard-assets/dashboard.css via pn.extension above.

# Template layout: header + tabs
template = pn.template.MaterialTemplate(
    title="AURORA Data Viewer",
    logo=DASHBOARD_LOGO,
    favicon=DASHBOARD_FAVICON,
    header_background=ACCENT,
    header_color="white",
    main_max_width="1800px",  # wide but keeps a valid string
    meta_viewport="width=device-width, initial-scale=1, viewport-fit=cover",
)


def _site_env_banner_pane() -> pn.pane.HTML | None:
    if SITE_ENV != "development" or not SITE_ENV_EXPLICIT:
        return None
    domain = SITE_DOMAIN or "data-ocean.gamb2le.co.uk"
    return pn.pane.HTML(
        "<div class='site-env-banner'>"
        "<span>Development site - live mirrored data</span>"
        f"<span class='site-env-banner__meta'>{escape(domain)}</span>"
        "</div>",
        sizing_mode="stretch_width",
        margin=(0, 0, 8, 0),
    )


def _lightweight_placeholder(label: str) -> pn.pane.HTML:
    return pn.pane.HTML(
        f"<div class='lazy-tab-placeholder'>{escape(label)} will load after the page opens.</div>",
        sizing_mode="stretch_width",
        margin=0,
    )


interactive_status_container = pn.Column(_lightweight_placeholder("Freshness and status"), sizing_mode="stretch_width")
interactive_availability_container = pn.Column(_lightweight_placeholder("Data availability"), sizing_mode="stretch_width")
science_status_container = pn.Column(_lightweight_placeholder("Science freshness and status"), sizing_mode="stretch_width")
science_availability_container = pn.Column(_lightweight_placeholder("Science data availability"), sizing_mode="stretch_width")
# These panes stay in the Bokeh document for the whole session. Updating a
# scalar ``object`` property is safe while a tab is attached; replacing the
# surrounding Column with newly created models during activation is not.
hk_status_pane = _lightweight_placeholder("Housekeeping freshness and status")
hk_availability_pane = _lightweight_placeholder("Housekeeping data availability")
hk_status_container = pn.Column(hk_status_pane, sizing_mode="stretch_width")
hk_availability_container = pn.Column(hk_availability_pane, sizing_mode="stretch_width")


def _activate_interactive_footer_metrics() -> None:
    global _INTERACTIVE_FOOTER_LOADED
    if _INTERACTIVE_FOOTER_LOADED:
        return
    interactive_status_container[:] = [interactive_status]
    interactive_availability_container[:] = [interactive_availability]
    _INTERACTIVE_FOOTER_LOADED = True


interactive_footer = pn.Card(
    pn.Row(
        interactive_copy,
        interactive_download,
        interactive_share_url,
        sizing_mode="stretch_width",
        css_classes=["mobile-stack", "action-row"],
    ),
    interactive_status_container,
    interactive_availability_container,
    title="",
    collapsible=False,
    sizing_mode="stretch_width",
    css_classes=["small-card"],
)

science_footer = pn.Card(
    pn.Row(
        science_copy,
        science_download,
        science_share_url,
        sizing_mode="stretch_width",
        css_classes=["mobile-stack", "action-row"],
    ),
    science_status_container,
    science_availability_container,
    title="",
    collapsible=False,
    sizing_mode="stretch_width",
    css_classes=["small-card"],
)

hk_footer = pn.Card(
    pn.Row(
        hk_copy,
        hk_download,
        hk_share_url,
        sizing_mode="stretch_width",
        css_classes=["mobile-stack", "action-row"],
    ),
    hk_status_container,
    hk_availability_container,
    title="",
    collapsible=False,
    sizing_mode="stretch_width",
    css_classes=["small-card"],
)

auroracam_footer = pn.Card(
    pn.Row(
        auroracam_copy,
        auroracam_share_url,
        sizing_mode="stretch_width",
        css_classes=["mobile-stack", "action-row"],
    ),
    title="",
    collapsible=False,
    sizing_mode="stretch_width",
    css_classes=["small-card"],
)

uas_footer = pn.Card(
    pn.Row(
        uas_copy,
        uas_share_url,
        sizing_mode="stretch_width",
        css_classes=["mobile-stack", "action-row"],
    ),
    title="",
    collapsible=False,
    sizing_mode="stretch_width",
    css_classes=["small-card"],
)

operations_dashboard = pn.pane.HTML("", sizing_mode="stretch_width", margin=0)


def _refresh_operations_dashboard() -> None:
    operations_dashboard.object = _ops_operations_markup()


_uas_timer = _safe_periodic_callback(_refresh_uas_dashboard, period=30_000, start=False)
_operations_timer = _safe_periodic_callback(_refresh_operations_dashboard, period=60_000, start=False)
_auroracam_timer = _safe_periodic_callback(_refresh_auroracam_latest_if_needed, period=60_000, start=False)


def _lazy_tab_placeholder(label: str) -> pn.pane.HTML:
    return pn.pane.HTML(
        f"<div class='lazy-tab-placeholder'>{escape(label)} will load when this tab is opened.</div>",
        sizing_mode="stretch_width",
        margin=0,
    )


science_quicklook_container = pn.Column(_lazy_tab_placeholder("Science quicklooks"), sizing_mode="stretch_width")
housekeeping_quicklook_pane = _lazy_tab_placeholder("House keeping quicklooks")
housekeeping_quicklook_container = pn.Column(housekeeping_quicklook_pane, sizing_mode="stretch_width")
uas_container = pn.Column(_lazy_tab_placeholder("UAS status"), sizing_mode="stretch_width")
operations_container = pn.Column(_lazy_tab_placeholder("Operations dashboard"), sizing_mode="stretch_width")
_LOADED_TABS: set[str] = set()


def _pane_html_object(pane) -> str:
    """Copy one simple quicklook pane into an already-attached HTML model."""
    if isinstance(pane, pn.pane.HTML):
        return str(pane.object or "")
    return f"<div class='lazy-tab-message'>{escape(str(getattr(pane, 'object', '') or ''))}</div>"


def _refresh_housekeeping_tab_content(*_events) -> None:
    """Refresh housekeeping without introducing new Bokeh model references."""
    rendered = _housekeeping_quicklook_image(hk_date.value, hk_instrument.value)
    housekeeping_quicklook_pane.object = _pane_html_object(rendered)
    hk_status_pane.object = _current_hk_status_markup()
    hk_availability_pane.object = _current_hk_availability_markup()


def _refresh_loaded_housekeeping_tab(_event=None) -> None:
    if "housekeeping" in _LOADED_TABS:
        _refresh_housekeeping_tab_content()


hk_instrument.param.watch(_refresh_loaded_housekeeping_tab, "value")
hk_date.param.watch(_refresh_loaded_housekeeping_tab, "value")
power_browser_guidance_container = pn.Column(
    sizing_mode="stretch_width",
    margin=0,
    visible=False,
)


@pn.depends(power_view_select.param.value)
def _power_section_intro(section: str):
    if section == "forecast":
        title = "Forecast & Planning"
        detail = "Forecasts, operating scenarios, recommended schedules, and verification."
    else:
        title = "Current Conditions"
        detail = "Observed station power, battery state, instrument load, voltage, and temperature."
    return pn.pane.HTML(
        "<div class='power-section-intro'>"
        f"<div class='power-section-intro__title'>{escape(title)}</div>"
        f"<div class='power-section-intro__detail'>{escape(detail)}</div>"
        "</div>",
        sizing_mode="stretch_width",
        margin=0,
    )


power_section_intro_container = pn.Column(
    _power_section_intro,
    sizing_mode="stretch_width",
    margin=0,
    visible=CURRENT_INSTRUMENT == "power",
)
power_view_select_container = pn.Column(
    power_view_select,
    sizing_mode="stretch_width",
    margin=0,
    visible=CURRENT_INSTRUMENT == "power",
)


def _sync_power_section_visibility() -> None:
    is_power = instrument_select.value == "power"
    is_forecast = is_power and power_view_select.value == "forecast"
    power_view_select_container.visible = is_power
    power_section_intro_container.visible = is_power
    power_browser_guidance_container.visible = is_forecast
    interactive_content.visible = not is_forecast
    power_plan_editor_container.visible = is_forecast
    if is_forecast:
        guidance = globals().get("_browser_power_briefing")
        if not power_browser_guidance_container.objects and callable(guidance):
            power_browser_guidance_container[:] = [guidance]
        if not power_plan_editor_container.objects:
            power_plan_editor_container[:] = [_get_power_plan_editor()]
    else:
        power_browser_guidance_container.clear()
        power_plan_editor_container.clear()


def _on_power_view_change(_event) -> None:
    _sync_power_section_visibility()
    if instrument_select.value == "power" and power_view_select.value != "forecast":
        # A section-specific prewarm is normally available. Start directly so
        # the tab can replace the visible figure without a placeholder flash.
        _start_interactive_render(
            range_start.value,
            range_end.value,
            bottom_range_m.value,
            top_range_m.value,
            var1_select.value,
            var2_select.value,
            beta_vmin.value,
            beta_vmax.value,
            ldr_vmin.value,
            ldr_vmax.value,
            lwp_ymin.value,
            lwp_ymax.value,
            iwv_ymin.value,
            iwv_ymax.value,
            irr_ymin.value,
            irr_ymax.value,
            "power",
        )
    _refresh_share_and_download_state()


power_view_select.param.watch(_on_power_view_change, "value")
_sync_power_section_visibility()

interactive_tab = pn.Column(
    controls,
    power_view_select_container,
    power_section_intro_container,
    power_browser_guidance_container,
    interactive_content,
    power_plan_editor_container,
    interactive_footer,
    sizing_mode="stretch_width",
)
science_quicklooks_tab = pn.Column(
    pn.Card(
        pn.Row(science_instrument, science_image_type, sizing_mode="stretch_width", css_classes=["mobile-stack"]),
        pn.Row(ql_prev, ql_date, ql_next, sizing_mode="stretch_width"),
        title="",
        collapsible=False,
        sizing_mode="stretch_width",
        css_classes=["small-card"],
    ),
    science_quicklook_container,
    science_footer,
    sizing_mode="stretch_width",
)
housekeeping_quicklooks_tab = pn.Column(
    pn.Card(
        pn.Row(hk_instrument, sizing_mode="stretch_width", css_classes=["mobile-stack"]),
        pn.Row(hk_prev, hk_date, hk_next, sizing_mode="stretch_width"),
        title="",
        collapsible=False,
        sizing_mode="stretch_width",
        css_classes=["small-card"],
    ),
    housekeeping_quicklook_container,
    hk_footer,
    sizing_mode="stretch_width",
)
auroracam_tab = pn.Column(
    pn.Card(
        pn.Row(auroracam_latest, auroracam_prev, auroracam_date, auroracam_next, sizing_mode="stretch_width", css_classes=["mobile-stack"]),
        pn.Row(auroracam_prev_time, auroracam_time, auroracam_next_time, sizing_mode="stretch_width", css_classes=["mobile-stack"]),
        title="",
        collapsible=False,
        sizing_mode="stretch_width",
        css_classes=["small-card", "auroracam-toolbar"],
    ),
    _auroracam_browser,
    auroracam_detail,
    auroracam_footer,
    sizing_mode="stretch_width",
)
uas_tab = pn.Column(
    pn.Card(
        pn.Row(
            uas_flight_day,
            uas_flight_select,
            uas_refresh,
            sizing_mode="stretch_width",
            css_classes=["mobile-stack"],
        ),
        title="Flight profiles",
        collapsible=False,
        sizing_mode="stretch_width",
        css_classes=["small-card", "uas-toolbar", "uas-flight-toolbar"],
    ),
    pn.Card(
        pn.Row(uas_window, sizing_mode="stretch_width", css_classes=["mobile-stack"]),
        title="UAS tier telemetry",
        collapsible=False,
        sizing_mode="stretch_width",
        css_classes=["small-card", "uas-toolbar"],
    ),
    uas_container,
    uas_footer,
    sizing_mode="stretch_width",
)
operations_tab = pn.Column(operations_container, sizing_mode="stretch_width")

MOBILE_TAB_OPTIONS = {
    "Overview": "overview",
    "Power": "power",
    "Plots": "plots",
    "Camera": "camera",
    "Ops": "ops",
}
MOBILE_TAB_LABEL_BY_SLUG = {value: key for key, value in MOBILE_TAB_OPTIONS.items()}


def _mobile_view_requested() -> bool:
    mode = (_request_query_args().get("view") or "auto").strip().lower()
    if mode == "desktop":
        return False
    if mode == "mobile":
        return True
    return _is_mobile_viewport()


def _mobile_query_params(tab_slug: str) -> dict[str, str]:
    params = _view_query_params("interactive")
    params["view"] = "mobile"
    params["mobile_tab"] = tab_slug
    if tab_slug == "power":
        params["power_view"] = power_view_select.value
    return {key: value for key, value in params.items() if value not in ("", None)}


def _mobile_level(level: str | None) -> str:
    text = str(level or "gray").lower()
    if text in {"green", "amber", "red"}:
        return text
    return "gray"


def _mobile_status_card_markup(label: str, value: str, meta: str = "", level: str | None = None) -> str:
    level_class = f" mobile-status-card--{_mobile_level(level)}" if level else ""
    return (
        f"<div class='mobile-status-card{level_class}'>"
        f"<div class='mobile-status-card__label'>{escape(label)}</div>"
        f"<div class='mobile-status-card__value'>{escape(value)}</div>"
        f"<div class='mobile-status-card__meta'>{escape(meta)}</div>"
        "</div>"
    )


def _browser_overview_instrument_markup() -> str:
    """Render one icon-led instrument-status list from the mobile contract."""
    try:
        rows = mobile_catalog.overview().get("instrumentPower", [])
    except Exception as exc:
        logging.getLogger(__name__).warning("Could not load browser overview instrument states: %s", exc)
        rows = []

    if not rows:
        return "<div class='mobile-section-note'>Instrument-state snapshot unavailable.</div>"

    rendered_rows = []
    for row in rows:
        level = _mobile_level(row.get("level"))
        rendered_rows.append(
            "<div class='overview-instrument-row overview-instrument-row--{level}' data-instrument-id='{instrument_id}'>"
            "<span class='overview-instrument-row__icon'>{icon}</span>"
            "<div><div class='overview-instrument-row__title'>{title}</div>"
            "<div class='overview-instrument-row__detail'>{detail}</div></div>"
            "<div class='overview-instrument-row__state'>{state}</div>"
            "</div>".format(
                level=escape(level),
                instrument_id=escape(str(row.get("id", "instrument"))),
                icon=instrument_icon_svg(row.get("systemImage")),
                title=escape(str(row.get("title", "Instrument"))),
                detail=escape(str(row.get("detail", "Status unavailable"))),
                state=escape(str(row.get("state", "Unknown"))),
            )
        )
    return (
        "<section class='overview-instrument-status'>"
        "<div class='overview-instrument-status__title'>Instrument status</div>"
        "<div class='overview-instrument-status__note'>Latest station status</div>"
        f"{''.join(rendered_rows)}</section>"
    )


def _mobile_auroracam_freshness() -> tuple[str, str, str]:
    latest_time: datetime | None = None
    latest_label = "No camera"
    for camera_id, spec in AURORACAM_CAMERAS.items():
        record = auroracam_latest_record(_auroracam_raw_root(), camera_id)
        moment = _auroracam_record_time(record)
        if moment is not None and (latest_time is None or moment > latest_time):
            latest_time = moment
            latest_label = str(spec["label"])
    if latest_time is None:
        return "No image", "AURORACam latest image unavailable", "amber"
    age_min = max((datetime.now(timezone.utc) - latest_time).total_seconds() / 60.0, 0.0)
    level = "green" if age_min < 30 else "amber" if age_min < 120 else "red"
    return latest_label, _humanize_age(latest_time), level


def _mobile_power_latest_measured_time() -> datetime | None:
    candidates = ("BatterySOC", "BatteryWatts", "DCInverterVolts", "ACOutputWatts")
    try:
        ds = _get_power_display_summary_dataset()
    except Exception:
        ds = None
    if ds is None or "time" not in ds:
        return _dataset_time_bounds("power")[1]
    times = pd.DatetimeIndex(ds["time"].values)
    latest: pd.Timestamp | None = None
    now = pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None)
    for name in candidates:
        if name not in ds:
            continue
        values = np.asarray(ds[name].values, dtype=np.float64)
        mask = np.isfinite(values) & (times <= now)
        if not mask.any():
            continue
        candidate = times[mask].max()
        if latest is None or candidate > latest:
            latest = candidate
    if latest is None:
        return _dataset_time_bounds("power")[1]
    return latest.to_pydatetime(warn=False)


def _mobile_overview_markup() -> str:
    snapshot = _ops_read_snapshot()
    if snapshot.get("_missing") or snapshot.get("_error"):
        ops_level = "red"
        ops_value = "No ops snapshot"
        ops_meta = snapshot.get("_error") or f"Missing {snapshot.get('_path', 'snapshot')}"
        snapshot_age = "Age unknown"
    else:
        updated_at = _ops_timestamp(snapshot.get("time_utc"))
        snapshot_age_min = None
        if updated_at is not None:
            snapshot_age_min = max((datetime.now(timezone.utc) - updated_at).total_seconds() / 60.0, 0.0)
        source_level = _ops_level_from_source_probes(snapshot.get("source_host_probe_fail_count"))
        paused_prefixes = _ops_expected_paused_prefixes()
        _source_recent, source_stale_count, _source_paused = _ops_source_health(snapshot, paused_prefixes)
        source_freshness_level = _ops_level_from_count(source_stale_count, amber_at=0.0)
        battery_level = _ops_level_from_battery_voltage(snapshot.get("aps_battery_voltage_v"))
        battery_soc_level = _ops_level_from_battery_soc(snapshot.get("aps_battery_soc_pct"))
        battery_depletion_level = _ops_level_from_battery_depletion(snapshot)
        processing_level = _ops_level_from_count(snapshot.get("failed_processing_unit_count"), amber_at=1.0)
        ops_level = _ops_worst_level([source_level, source_freshness_level, battery_level, battery_soc_level, battery_depletion_level, processing_level])
        ops_value = "Healthy" if ops_level == "green" else "Attention" if ops_level == "amber" else "Action" if ops_level == "red" else "Waiting"
        failed = len(_ops_failed_service_names(snapshot))
        snapshot_age = f"{snapshot_age_min:.0f} min old" if snapshot_age_min is not None else "Age unknown"
        ops_meta = f"Snapshot {snapshot_age}; {failed} unhealthy services"

    battery_value, battery_meta = _ops_battery_text(snapshot)
    battery_soc_value, battery_soc_meta = _ops_battery_soc_text(snapshot)
    depletion_value, depletion_meta = _ops_battery_depletion_text(snapshot)
    camera_value, camera_meta, camera_level = _mobile_auroracam_freshness()
    # The operations snapshot already records the latest measured power
    # timestamp. Opening the 271-variable display Zarr here made every
    # Overview session wait several seconds before its first paint.
    power_latest = next(
        (
            moment
            for key in (
                "power_latest_time_utc",
                "aps_battery_power_time_utc",
                "aps_battery_soc_time_utc",
                "aps_battery_voltage_time_utc",
            )
            if (moment := _ops_timestamp(snapshot.get(key))) is not None
        ),
        None,
    )
    if power_latest is None:
        power_latest = _mobile_power_latest_measured_time()
    power_latest_utc = power_latest.replace(tzinfo=timezone.utc) if power_latest and power_latest.tzinfo is None else power_latest
    power_meta = _humanize_age(power_latest_utc)
    power_level = _ops_level_from_age_minutes(max((datetime.now(timezone.utc) - power_latest_utc).total_seconds() / 60.0, 0.0) if power_latest_utc else None)
    cards = [
        _mobile_status_card_markup("Operations", ops_value, ops_meta, ops_level),
        _mobile_status_card_markup("State of Charge", battery_soc_value, battery_soc_meta, _ops_level_from_battery_soc(snapshot.get("aps_battery_soc_pct"))),
        _mobile_status_card_markup("Battery Voltage", battery_value, battery_meta, _ops_level_from_battery_voltage(snapshot.get("aps_battery_voltage_v"))),
        _mobile_status_card_markup("Time to Depleted", depletion_value, depletion_meta, _ops_level_from_battery_depletion(snapshot)),
        _mobile_status_card_markup("Power Data", power_latest.strftime("%H:%M UTC") if power_latest else "No data", power_meta, power_level),
        _mobile_status_card_markup("AURORACam", camera_value, camera_meta, camera_level),
    ]
    try:
        cards.extend(
            _mobile_status_card_markup(
                str(card["title"]),
                str(card["value"]),
                str(card.get("detail") or "Latest station measurement"),
                _mobile_level(card.get("level")),
            )
            for card in mobile_catalog.environmental_signal_cards()
        )
    except Exception as exc:
        logging.getLogger(__name__).warning("Could not load overview environmental signals: %s", exc)
    return (
        "<div class='mobile-shell'>"
        "<div><div class='mobile-section-title'>AURORA Overview</div>"
        "<div class='mobile-section-note'>Latest cached station snapshot. Detailed plots, cameras, and diagnostics remain in their dedicated views.</div></div>"
        f"<div class='mobile-card-grid'>{''.join(cards)}</div>"
        "</div>"
    )


def _mobile_overview() -> pn.Column:
    return pn.Column(pn.pane.HTML(_mobile_overview_markup(), sizing_mode="stretch_width", margin=0), sizing_mode="stretch_width", css_classes=["mobile-shell"])


_BROWSER_OVERVIEW_LOADED = False
browser_overview_summary_pane = pn.pane.HTML(
    "<div class='mobile-section-note'>Open Overview to load the latest station snapshot.</div>",
    sizing_mode="stretch_width",
    margin=0,
)
browser_overview_instrument_pane = pn.pane.HTML("", sizing_mode="stretch_width", margin=(8, 0, 0, 0))
browser_overview_container = pn.Column(
    browser_overview_summary_pane,
    browser_overview_instrument_pane,
    sizing_mode="stretch_width",
)
browser_overview_refresh = pn.widgets.Button(name="Refresh station snapshot", button_type="primary", icon="refresh")


def _refresh_browser_overview(_event=None, *, force: bool = False) -> None:
    global _BROWSER_OVERVIEW_LOADED
    # The initial tab selection happens before the document is attached and an
    # onload hook runs once the websocket opens. Replacing the same models in
    # both phases causes Bokeh to drop patches and can leave Overview blank.
    if _BROWSER_OVERVIEW_LOADED and _event is None and not force:
        return
    browser_overview_summary_pane.object = _mobile_overview_markup()
    browser_overview_instrument_pane.object = _browser_overview_instrument_markup()
    _BROWSER_OVERVIEW_LOADED = True


browser_overview_refresh.on_click(_refresh_browser_overview)
# Do not open the power/PDU products during module import. The active-tab
# loader refreshes this view when Overview is selected, including direct links.
browser_overview_tab = pn.Column(
    pn.Row(
        pn.pane.HTML("<div class='desktop-overview-heading'>Station overview</div>", sizing_mode="stretch_width", margin=0),
        browser_overview_refresh,
        sizing_mode="stretch_width",
    ),
    browser_overview_container,
    sizing_mode="stretch_width",
)


def _mobile_power_window() -> xr.Dataset | None:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    start = now - DEFAULT_WINDOW
    end = now + timedelta(hours=float(os.environ.get("AURORA_POWER_SOC_FORECAST_HOURS", "48")))
    section = power_view_select.value
    ds = _open_power_display_summary_window(start, end, section=section)
    if ds is None:
        ds = open_window(start, end, instrument="power", render_quality="coarse")
    if ds is not None and "time" in ds:
        ds.attrs[SUMMARY_DISPLAY_START_ATTR] = start.isoformat()
        ds.attrs[SUMMARY_DISPLAY_END_ATTR] = now.isoformat()
        ds = prepare_summary_dataset(ds, "power")
    return ds


def _mobile_trace_values(
    ds: xr.Dataset,
    trace,
    include_future: bool,
    *,
    start_at: pd.Timestamp | None = None,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    if trace.var not in ds or "time" not in ds:
        return pd.DatetimeIndex([]), np.asarray([], dtype=float)
    times = pd.DatetimeIndex(ds["time"].values)
    values = np.asarray(ds[trace.var].values, dtype=np.float64)
    if values.shape[0] != len(times):
        return pd.DatetimeIndex([]), np.asarray([], dtype=float)
    now = pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None)
    mask = np.isfinite(values)
    if trace.valid_min is not None:
        mask &= values >= float(trace.valid_min)
    if trace.valid_max is not None:
        mask &= values <= float(trace.valid_max)
    if not include_future:
        mask &= times <= now
    if start_at is not None:
        mask &= times >= start_at
    if not mask.any():
        return pd.DatetimeIndex([]), np.asarray([], dtype=float)
    out_times = times[mask]
    out_values = values[mask] * float(trace.scale)
    if trace.display_horizon_hours is not None and len(out_times):
        display_end = out_times.min() + pd.Timedelta(hours=float(trace.display_horizon_hours))
        display_mask = out_times <= display_end
        out_times = out_times[display_mask]
        out_values = out_values[display_mask]
    if trace.skip_if_all_zero and np.allclose(out_values[np.isfinite(out_values)], 0.0):
        return pd.DatetimeIndex([]), np.asarray([], dtype=float)
    max_points = 260
    if len(out_times) > max_points:
        stride = int(np.ceil(len(out_times) / max_points))
        out_times = out_times[::stride]
        out_values = out_values[::stride]
    return out_times, out_values


def _mobile_forecast_panel_start(ds: xr.Dataset, panel) -> pd.Timestamp | None:
    preferred_fields = {
        "soc_projection": ("BatterySOCForecast",),
        "soc_24h_forecast": ("SystemAsIsDecisionSOCP50",),
        "soc_ecmwf_forecast": (
            "SystemAsIsDecisionSOCP50",
            "BatterySOCForecastP50",
            "BatterySOCForecast",
        ),
        "ecmwf_solar_forecast": ("ForecastSolarWatts", "ECMWFSolarIrradiance"),
        "operating_plan_scenarios": ("OperatingCL61OptimizedSOCP50",),
        "operating_plan_schedule": (
            "OperatingCL61OptimizedActiveCount",
            "OperatingCL61OptimizedCL61On",
            "OperatingCL61OptimizedRadarOn",
            "OperatingCL61OptimizedHATPROOn",
        ),
    }
    times = pd.DatetimeIndex(ds["time"].values)
    for field in preferred_fields.get(panel.key, tuple(trace.var for trace in panel.traces)):
        if field not in ds or ds[field].dims != ("time",):
            continue
        values = np.asarray(ds[field].values, dtype=np.float64)
        valid = np.isfinite(values)
        if valid.any():
            return pd.Timestamp(times[valid][0])
    return None


def _verification_guidance_markup(guidance: dict[str, object] | None) -> str:
    if not guidance:
        return ""
    metrics = guidance.get("metrics", [])
    if not isinstance(metrics, list):
        return ""
    cards = []
    for metric in metrics:
        if not isinstance(metric, dict):
            continue
        level = escape(str(metric.get("level", "neutral")))
        cards.append(
            "<div class='verification-guidance__metric'>"
            f"<div class='verification-guidance__label'>{escape(str(metric.get('label', 'Metric')))}</div>"
            f"<div class='verification-guidance__value'>{escape(str(metric.get('valueText', 'Not available')))}</div>"
            f"<div class='verification-guidance__detail'>{escape(str(metric.get('direction', '')))}</div>"
            f"<div class='verification-guidance__detail'>{escape(str(metric.get('reference', '')))}</div>"
            f"<div class='verification-guidance__detail verification-guidance__status--{level}'>{escape(str(metric.get('status', '')))}</div>"
            f"<div class='verification-guidance__detail'>{escape(str(metric.get('evidence', '')))}</div>"
            "</div>"
        )
    if not cards:
        return ""
    return (
        "<div class='verification-guidance'>"
        f"<div class='verification-guidance__title'>{escape(str(guidance.get('title', 'How to read this plot')))}</div>"
        f"<div class='verification-guidance__summary'>{escape(str(guidance.get('summary', '')))}</div>"
        f"<div class='verification-guidance__metrics'>{''.join(cards)}</div>"
        "</div>"
    )


def _forecast_plot_info_markup(info: dict[str, object] | None) -> str:
    """Render shared forecast implementation notes without duplicating policy in views."""
    if not info:
        return ""
    metrics = info.get("metrics", [])
    rows = []
    if isinstance(metrics, list):
        for metric in metrics:
            if not isinstance(metric, dict):
                continue
            rows.append(
                "<li>"
                f"<strong>{escape(str(metric.get('label', 'Detail')))}</strong>: "
                f"{escape(str(metric.get('detail', '')))}"
                "</li>"
            )
    implementation = escape(str(info.get("implementation", "")))
    return (
        "<div class='forecast-plot-info'>"
        f"<div>{escape(str(info.get('summary', '')))}</div>"
        f"<div><strong>Implementation here:</strong> {implementation}</div>"
        f"<ul>{''.join(rows)}</ul>"
        "</div>"
    )


def _forecast_plot_info_control(panel, ds: xr.Dataset, *, mobile: bool = False):
    """Return a small per-plot info button with an on-demand explanation."""
    info = build_power_forecast_info(panel.key, ds)
    if not info:
        return None
    detail = pn.pane.HTML(_forecast_plot_info_markup(info), sizing_mode="stretch_width", visible=False, margin=(0, 0, 4, 0))
    button = pn.widgets.Button(
        name="Info",
        icon="info-circle",
        button_type="light",
        width=62,
        height=30,
        css_classes=["forecast-plot-info-button"],
    )

    def toggle(_event) -> None:
        detail.visible = not detail.visible
        button.name = "Hide" if detail.visible else "Info"

    button.on_click(toggle)
    title = pn.pane.HTML(
        f"<div class='{'mobile-plot-card__title' if mobile else 'forecast-plot-info__title'}'>{escape(power_panel_label(ds, panel))}</div>",
        sizing_mode="stretch_width",
        margin=0,
    )
    return pn.Column(
        pn.Row(title, pn.Spacer(), button, sizing_mode="stretch_width", margin=0),
        detail,
        sizing_mode="stretch_width",
        css_classes=["forecast-plot-info-control"],
    )


def _power_plot_card(ds: xr.Dataset, panel, *, mobile: bool) -> pn.Column | None:
    forecast_panel_keys = {
        "soc_24h_forecast",
        "soc_ecmwf_forecast",
        "operating_plan_scenarios",
        "operating_plan_schedule",
        "ecmwf_solar_forecast",
    }
    include_future = panel.key in forecast_panel_keys
    forecast_start = _mobile_forecast_panel_start(ds, panel) if include_future else None
    fig = go.Figure()
    has_right_axis = panel.right_axis_label is not None
    legend_items: list[str] = []
    for trace in panel.traces:
        if trace.projection_lookback_minutes is not None:
            continue
        times, values = _mobile_trace_values(ds, trace, include_future=include_future, start_at=forecast_start)
        if len(times) == 0:
            continue
        use_right = trace.axis == "right" and has_right_axis
        if trace.dash in {"dash", "dashdot", "longdash", "longdashdot"}:
            dash_class = " mobile-plot-card__legend-line--dash"
        elif trace.dash == "dot":
            dash_class = " mobile-plot-card__legend-line--dot"
        else:
            dash_class = ""
        trace_label = power_trace_label(ds, trace)
        legend_items.append(
            "<span class='mobile-plot-card__legend-item'>"
            f"<span class='mobile-plot-card__legend-line{dash_class}' style='color:{escape(trace.color)}'></span>"
            f"<span class='mobile-plot-card__legend-label'>{escape(trace_label)}</span>"
            "</span>"
        )
        fig.add_trace(
            go.Scatter(
                x=times,
                y=values,
                mode="lines",
                name=trace_label,
                yaxis="y2" if use_right else "y",
                line=dict(color=trace.color, width=trace.line_width, dash=trace.dash or "solid", shape="hv" if trace.step else "linear"),
                opacity=trace.opacity,
                hovertemplate=f"Time=%{{x}}<br>{trace_label}=%{{y:.4g}}<extra></extra>",
                connectgaps=False,
            )
        )
    if panel.key in {"operating_plan_scenarios", "ecmwf_solar_forecast"} and "OperatingCL61OptimizedModeCode" in ds:
        schedule_times = pd.DatetimeIndex(ds["time"].values)
        for start, end, label, color in operating_mode_intervals(
            schedule_times,
            np.asarray(ds["OperatingCL61OptimizedModeCode"].values),
        ):
            fig.add_vrect(
                x0=start,
                x1=end,
                fillcolor=color,
                opacity=1.0,
                line_width=0,
                layer="below",
                annotation_text=f"{label} on",
                annotation_position="top left",
                annotation_font_size=7,
                annotation_font_color=THEME_TEXT,
            )
    if panel.key in SOC_REFERENCE_PANEL_KEYS and fig.data:
        all_times = [pd.Timestamp(value) for trace in fig.data for value in trace.x]
        if all_times:
            fig.add_trace(
                go.Scatter(
                    x=[min(all_times), max(all_times)],
                    y=[MINIMUM_OPERATIONAL_SOC_PCT, MINIMUM_OPERATIONAL_SOC_PCT],
                    mode="lines",
                    name=MINIMUM_OPERATIONAL_SOC_REFERENCE_LABEL,
                    line=dict(color=THEME_TEXT, width=1.5, dash="dash"),
                    hovertemplate=f"{MINIMUM_OPERATIONAL_SOC_REFERENCE_LABEL}<extra></extra>",
                    showlegend=False,
                )
            )
            legend_items.append(
                "<span class='mobile-plot-card__legend-item'>"
                f"<span class='mobile-plot-card__legend-line mobile-plot-card__legend-line--dash' style='color:{escape(THEME_TEXT)}'></span>"
                f"<span class='mobile-plot-card__legend-label'>{escape(MINIMUM_OPERATIONAL_SOC_REFERENCE_LABEL)}</span>"
                "</span>"
            )
    if not fig.data:
        return None
    plot_height = (
        int(os.environ.get("AURORA_MOBILE_POWER_PLOT_HEIGHT", "110"))
        if mobile
        else int(os.environ.get("AURORA_DESKTOP_POWER_PLOT_HEIGHT", "300"))
    )
    layout = dict(
        height=plot_height,
        autosize=True,
        margin=(
            dict(l=27, r=27 if has_right_axis else 6, t=3, b=22)
            if mobile
            else dict(l=62, r=62 if has_right_axis else 24, t=12, b=54)
        ),
        showlegend=False,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(size=8 if mobile else 12, color=THEME_TEXT),
        xaxis=dict(
            showgrid=True,
            gridcolor=THEME_GRID,
            tickfont=dict(size=7 if mobile else 11),
            tickformat="%d %b<br>%H:%M UTC",
            title=None if mobile else "Forecast Time (UTC)",
            nticks=4 if mobile else 8,
        ),
        yaxis=dict(
            title=None if mobile else panel.left_axis_label,
            tickfont=dict(size=7 if mobile else 11),
            showgrid=True,
            gridcolor=THEME_GRID,
            nticks=4 if mobile else 6,
        ),
    )
    if has_right_axis:
        layout["yaxis2"] = dict(
            title=None if mobile else panel.right_axis_label,
            tickfont=dict(size=7 if mobile else 11),
            overlaying="y",
            side="right",
            showgrid=False,
            nticks=4 if mobile else 6,
        )
    fig.update_layout(**layout)
    guidance = _verification_guidance_markup(build_power_verification_guidance(panel.key, ds))
    info_control = _forecast_plot_info_control(panel, ds, mobile=mobile)
    card_class = "mobile-plot-card" if mobile else "desktop-power-plot-card"
    figure_class = "mobile-figure" if mobile else "desktop-power-figure"
    return pn.Column(
        *(
            [info_control]
            if info_control is not None
            else [pn.pane.HTML(f"<div class='{'mobile-plot-card__title' if mobile else 'forecast-plot-info__title'}'>{escape(panel.label)}</div>", margin=0)]
        ),
        *(
            [pn.pane.HTML(f"<div class='mobile-plot-card__note'>{escape(panel.description)}</div>", margin=0)]
            if panel.description
            else []
        ),
        *([pn.pane.HTML(guidance, margin=0)] if guidance else []),
        pn.pane.HTML(f"<div class='mobile-plot-card__legend'>{''.join(legend_items)}</div>", margin=0),
        pn.pane.Plotly(
            fig,
            config={"displayModeBar": not mobile, "displaylogo": False, "responsive": True},
            sizing_mode="stretch_width",
            height=plot_height + 8,
            css_classes=[figure_class],
        ),
        sizing_mode="stretch_width",
        css_classes=[card_class],
    )


def _mobile_power_card(ds: xr.Dataset, panel) -> pn.Column | None:
    return _power_plot_card(ds, panel, mobile=True)


def _desktop_power_forecast_cards(ds: xr.Dataset) -> list[pn.Column]:
    forecast_groups = {"forecast_24h", "forecast_96h", "verification"}
    panels = [
        panel
        for panel in SUMMARY_LAYOUTS["power"]
        if POWER_PANEL_TIME_GROUP_BY_KEY.get(panel.key) in forecast_groups
    ]
    return [
        card
        for card in (_power_plot_card(ds, panel, mobile=False) for panel in panels)
        if card is not None
    ]


def _power_forecast_status_markup(ds: xr.Dataset) -> str:
    """Expose planner provenance and prevent stale plans looking actionable."""
    planning_status = str(ds.attrs.get("operating_planning_status", "")).strip()
    planning_reason = str(ds.attrs.get("operating_planning_status_reason", "")).strip()
    if planning_status == "unavailable":
        detail = "Awaiting a planning forecast aligned with the current SOC measurement."
        if planning_reason:
            detail = f"{detail} {planning_reason}"
        return f"<div class='mobile-section-note'>{escape(detail)}</div>"
    kind = str(ds.attrs.get("operating_planning_forecast_refresh_kind", "")).strip()
    anchor = str(ds.attrs.get("operating_planning_forecast_initial_soc_time", "")).strip()
    issued = str(ds.attrs.get("operating_planning_forecast_generated_at_utc", "")).strip()
    if not kind:
        return "<div class='mobile-section-note'>Operating forecast is unavailable.</div>"
    if kind == "cached_reanchor":
        state = "Cached forecast - reduced confidence"
    else:
        state = "Fresh ECMWF forecast"
    detail = " | ".join(part for part in (f"Anchor {anchor}" if anchor else "", f"Issued {issued}" if issued else "") if part)
    return f"<div class='mobile-section-note'>{escape(state)}{': ' + escape(detail) if detail else ''}</div>"


def _browser_power_briefing_markup(ds: xr.Dataset) -> str:
    """Explain the Power forecast scenarios in the desktop browser."""
    status = str(ds.attrs.get("operating_planning_status", "")).strip()
    if status == "unavailable":
        reason = str(ds.attrs.get("operating_planning_status_reason", "")).strip()
        detail = "An operating recommendation will appear when its planning forecast is aligned with the current SOC measurement."
        if reason:
            detail = f"{detail} {reason}"
        return (
            "<div class='power-browser-briefing'>"
            "<div class='power-browser-briefing__title'>Operating scenarios</div>"
            f"<div class='power-browser-briefing__copy'>{escape(detail)}</div>"
            "</div>"
        )

    current_mode = str(ds.attrs.get("operating_current_mode_label", "Current system state")).strip()
    horizon = str(ds.attrs.get("operating_optimization_horizon_hours", "96")).strip()
    scenario_labels = ", ".join(definition.label for definition in SUGGESTED_OPERATING_SCENARIOS)
    return (
        "<div class='power-browser-briefing'>"
        "<div class='power-browser-briefing__title'>Forecast scenarios</div>"
        "<div class='power-browser-briefing__grid'>"
        "<div><strong>System as-is</strong><br>ECMWF ensemble forecast anchored to the latest confirmed finite instrument state and its detected sustained load phase. P10/P90 include weather, bounded battery parameters, and only recurrent startup or fan uncertainty within that same state.</div>"
        f"<div><strong>Instrument scenarios</strong><br>Current system mode: {escape(current_mode)}. Across {escape(horizon)} hours: {escape(scenario_labels)}. Each trace starts from the latest SOC and uses the learned load distribution for exactly its named state; states are never blended. UAS tier 3 remains provisional until repeated operating evidence is available.</div>"
        "<div><strong>Safety rule</strong><br>The recommended schedule is advisory only and aims to keep P10 SOC at or above the 40% operational minimum.</div>"
        "</div></div>"
    )


@pn.depends(instrument_select.param.value, range_start.param.value, range_end.param.value)
def _browser_power_briefing(instrument, start, end):
    if str(instrument) != "power":
        return pn.Spacer(height=0)
    ds = _open_power_display_summary_window(start, end, section="forecast")
    if ds is None or "time" not in ds:
        return pn.pane.Alert("Power forecast data is not available.", alert_type="warning")
    ds.attrs[SUMMARY_DISPLAY_START_ATTR] = _as_naive_utc_datetime(start).isoformat()
    ds.attrs[SUMMARY_DISPLAY_END_ATTR] = _as_naive_utc_datetime(end).isoformat()
    ds = prepare_summary_dataset(ds, "power")
    cards = _desktop_power_forecast_cards(ds)
    if not cards:
        return pn.pane.Alert("No plottable Power forecast panels are available.", alert_type="warning")
    return pn.Column(
        pn.pane.HTML(_browser_power_briefing_markup(ds), sizing_mode="stretch_width", margin=0),
        *cards,
        sizing_mode="stretch_width",
        css_classes=["power-browser-guidance"],
    )


_sync_power_section_visibility()


@pn.depends(power_view_select.param.value, range_end.param.value)
def _mobile_power_section(section: str, _live_refresh_anchor):
    ds = _mobile_power_window()
    if ds is None or "time" not in ds:
        return pn.Column(
            pn.pane.HTML("<div class='mobile-plot-card__empty'>Power display data is not available.</div>", sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
            css_classes=["mobile-shell"],
        )
    selected_groups = (
        {"observed"}
        if section == "current"
        else {"forecast_24h", "forecast_96h", "verification"}
    )
    panels = [
        panel
        for panel in SUMMARY_LAYOUTS["power"]
        if POWER_PANEL_TIME_GROUP_BY_KEY.get(panel.key, "observed") in selected_groups
    ]
    cards = [_mobile_power_card(ds, panel) for panel in panels]
    cards = [card for card in cards if card is not None]
    if not cards:
        cards = [pn.pane.HTML("<div class='mobile-plot-card__empty'>No plottable Power panels are available.</div>", sizing_mode="stretch_width")]
    forecast_content = []
    if section == "forecast":
        forecast_content = [
            pn.pane.HTML(_power_forecast_status_markup(ds), sizing_mode="stretch_width", margin=0),
            pn.pane.HTML(_browser_power_briefing_markup(ds), sizing_mode="stretch_width", margin=0),
            _get_mobile_power_plan_editor(),
        ]
    return pn.Column(
        *forecast_content,
        *cards,
        sizing_mode="stretch_width",
        css_classes=["mobile-shell", "mobile-shell--power"],
    )


def _mobile_power_tab() -> pn.Column:
    return pn.Column(
        pn.pane.HTML(
            "<div class='mobile-section-title mobile-power-title'>Power</div>"
            "<div class='mobile-section-note mobile-power-note'>Current station conditions and forecast planning are kept in separate views.</div>",
            sizing_mode="stretch_width",
            margin=0,
        ),
        power_view_select,
        _power_section_intro,
        _mobile_power_section,
        sizing_mode="stretch_width",
        css_classes=["mobile-shell", "mobile-shell--power"],
    )


def _mobile_plots_tab() -> pn.Column:
    _ensure_active_tab_loaded("science")
    return pn.Column(
        pn.pane.HTML(
            "<div class='mobile-section-title'>Plots</div>"
            "<div class='mobile-section-note'>Generated quicklooks load faster on a phone. Use desktop view for full interactive controls.</div>",
            sizing_mode="stretch_width",
            margin=0,
        ),
        science_quicklooks_tab,
        sizing_mode="stretch_width",
        css_classes=["mobile-shell"],
    )


def _mobile_auroracam_grid(selected_day: str | None, selected_time: str | None, selected_camera_id: str | None) -> pn.Column:
    day = _auroracam_selected_day(selected_day)
    label = day or "Latest available"
    time_label = selected_time or "Latest"
    for camera_id, card in auroracam_cards.items():
        record = _auroracam_record_for_time(camera_id, selected_day, selected_time)
        card.param.update(**_auroracam_card_params(camera_id, record, selected_camera_id))
    return pn.Column(
        pn.pane.HTML(
            "<div class='auroracam-section__head'>"
            "<div class='auroracam-section__title'>AURORACam frames</div>"
            f"<div class='auroracam-section__meta'>{escape(label)} | {escape(time_label)}</div>"
            "</div>",
            sizing_mode="stretch_width",
            margin=0,
        ),
        pn.GridBox(*auroracam_cards.values(), ncols=1, sizing_mode="stretch_width", css_classes=["auroracam-grid"]),
        sizing_mode="stretch_width",
        css_classes=["auroracam-section"],
    )


@pn.depends(auroracam_date.param.value, auroracam_time.param.value, auroracam_camera.param.value)
def _mobile_auroracam_browser(selected_day, selected_time, camera_id):
    root = _auroracam_raw_root()
    if not root.exists():
        return pn.pane.HTML(
            "<div class='auroracam-empty'>AURORACam image root is not available.</div>",
            sizing_mode="stretch_width",
            margin=0,
        )
    return pn.Column(
        _mobile_auroracam_grid(selected_day, selected_time, camera_id),
        sizing_mode="stretch_width",
        css_classes=["auroracam-browser"],
    )


def _mobile_camera_tab() -> pn.Column:
    return pn.Column(
        pn.pane.HTML("<div class='mobile-section-title'>Camera</div>", sizing_mode="stretch_width", margin=0),
        pn.Card(
            pn.Column(
                auroracam_latest,
                auroracam_date,
                auroracam_time,
                pn.Row(auroracam_prev_time, auroracam_next_time, sizing_mode="stretch_width", css_classes=["mobile-stack"]),
                pn.Row(auroracam_prev, auroracam_next, sizing_mode="stretch_width", css_classes=["mobile-stack"]),
                sizing_mode="stretch_width",
            ),
            title="",
            collapsible=False,
            sizing_mode="stretch_width",
            css_classes=["small-card", "auroracam-toolbar"],
        ),
        _mobile_auroracam_browser,
        mobile_auroracam_detail,
        sizing_mode="stretch_width",
        css_classes=["mobile-shell"],
    )


def _mobile_ops_tab() -> pn.Column:
    _ensure_active_tab_loaded("operations")
    return pn.Column(
        pn.pane.HTML("<div class='mobile-section-title'>Operations</div>", sizing_mode="stretch_width", margin=0),
        operations_tab,
        sizing_mode="stretch_width",
        css_classes=["mobile-shell"],
    )


def _mobile_initial_tab() -> str:
    args = _request_query_args()
    explicit = args.get("mobile_tab")
    if explicit in MOBILE_TAB_LABEL_BY_SLUG:
        return explicit
    legacy = args.get("tab")
    if legacy == "auroracam":
        return "camera"
    if legacy in {"science", "housekeeping"}:
        return "plots"
    if legacy in {"operations", "uas"}:
        return "ops"
    if legacy == "interactive" and args.get("instrument") not in (None, "", "power"):
        return "plots"
    return "overview"


def _mobile_app_nav_markup(active_slug: str | None = None) -> str:
    active = active_slug if active_slug in MOBILE_TAB_LABEL_BY_SLUG else "overview"
    items: list[str] = []
    for label, slug in MOBILE_TAB_OPTIONS.items():
        classes = "mobile-app-tabs__link"
        aria_current = ""
        if slug == active:
            classes += " mobile-app-tabs__link--active"
            aria_current = " aria-current='page'"
        href = "?" + escape(urlencode(_mobile_query_params(slug)), quote=True)
        items.append(
            f"<a class='{classes}' href='{href}'{aria_current}>"
            f"<span class='mobile-app-tabs__label'>{escape(label)}</span>"
            "</a>"
        )
    return "<nav class='mobile-app-tabs' aria-label='Mobile dashboard sections'>" + "".join(items) + "</nav>"


mobile_app_nav = pn.pane.HTML(
    _mobile_app_nav_markup(_mobile_initial_tab()),
    sizing_mode="stretch_width",
    margin=0,
    css_classes=["mobile-bottom-nav"],
)
mobile_app_active = pn.Column(sizing_mode="stretch_width", margin=0)
_MOBILE_LOADED_TABS: dict[str, object] = {}


def _mobile_panel_for_slug(slug: str):
    if slug not in _MOBILE_LOADED_TABS:
        if slug == "overview":
            _MOBILE_LOADED_TABS[slug] = _mobile_overview()
        elif slug == "power":
            _MOBILE_LOADED_TABS[slug] = _mobile_power_tab()
        elif slug == "plots":
            _MOBILE_LOADED_TABS[slug] = _mobile_plots_tab()
        elif slug == "camera":
            _MOBILE_LOADED_TABS[slug] = _mobile_camera_tab()
        elif slug == "ops":
            _MOBILE_LOADED_TABS[slug] = _mobile_ops_tab()
        else:
            _MOBILE_LOADED_TABS[slug] = _mobile_overview()
    return _MOBILE_LOADED_TABS[slug]


def _set_mobile_app_tab(slug: str | None) -> None:
    active = slug if slug in MOBILE_TAB_LABEL_BY_SLUG else "overview"
    panel = _mobile_panel_for_slug(active)
    if len(mobile_app_active.objects) != 1 or mobile_app_active.objects[0] is not panel:
        mobile_app_active[:] = [panel]
    markup = _mobile_app_nav_markup(active)
    if mobile_app_nav.object != markup:
        mobile_app_nav.object = markup
    try:
        location = pn.state.location
    except Exception:
        location = None
    if location is not None:
        search = "?" + urlencode(_mobile_query_params(active))
        if getattr(location, "search", None) != search:
            location.search = search


def _build_mobile_layout() -> pn.Column:
    _set_mobile_app_tab(_mobile_initial_tab())
    return pn.Column(mobile_app_nav, mobile_app_active, sizing_mode="stretch_width", margin=0, css_classes=["mobile-app"])

interactive_tab_host = pn.Column(
    interactive_tab,
    sizing_mode="stretch_width",
    margin=0,
    name="Interactive Data Browser",
)
power_tab_host = pn.Column(
    sizing_mode="stretch_width",
    margin=0,
    name="Power",
)

DESKTOP_TAB_SPECS = (
    ("Overview", "overview", browser_overview_tab),
    ("Interactive Data Browser", "interactive", interactive_tab_host),
    ("Power", "power", power_tab_host),
    ("Science Quicklooks", "science", science_quicklooks_tab),
    ("House Keeping Quicklooks", "housekeeping", housekeeping_quicklooks_tab),
    ("AURORACam", "auroracam", auroracam_tab),
    ("UAS", "uas", uas_tab),
    ("Operations Dashboard", "operations", operations_tab),
)
TAB_PANEL_BY_SLUG = {
    slug: panel for _label, slug, panel in DESKTOP_TAB_SPECS
}
TAB_INDEX_BY_SLUG = {
    slug: index for index, (_label, slug, _panel) in enumerate(DESKTOP_TAB_SPECS)
}
TAB_SLUG_BY_INDEX = {
    index: slug for slug, index in TAB_INDEX_BY_SLUG.items()
}
ACTIVE_TAB_SLUG = "interactive"
_browser_tab_syncing = False
desktop_tabs = pn.Tabs(
    *((label, panel) for label, _slug, panel in DESKTOP_TAB_SPECS),
    # Each expensive tab body already starts as a lightweight placeholder and
    # is populated on demand. Keeping every stable tab host attached avoids
    # Bokeh reference collisions when those placeholders are replaced.
    dynamic=False,
    sizing_mode="stretch_width",
    css_classes=["desktop-tabs"],
)
_desktop_tab_syncing = False
_ACTIVE_TAB_LOAD_GENERATION = 0


def _normalize_tab_slug(slug: str | None) -> str:
    return slug if slug in TAB_PANEL_BY_SLUG else "interactive"


def _sync_browser_tab_instrument(active: str) -> None:
    """Keep the shared browser component aligned with its top-level tab."""
    global _browser_tab_syncing
    if _browser_tab_syncing:
        return
    _browser_tab_syncing = True
    try:
        if active == "power":
            instrument_select.visible = False
            if instrument_select.value != "power":
                instrument_select.value = "power"
            interactive_tab_host.clear()
            power_tab_host[:] = [interactive_tab]
        elif active == "interactive":
            instrument_select.visible = True
            if instrument_select.value == "power":
                instrument_select.value = "Ceilometer"
            power_tab_host.clear()
            interactive_tab_host[:] = [interactive_tab]
    finally:
        _browser_tab_syncing = False


def _ensure_active_tab_loaded(slug: str | None = None) -> None:
    active = _normalize_tab_slug(slug or ACTIVE_TAB_SLUG)
    timer_groups = {
        "interactive": (_base_dataset_timer, _live_cb),
        "power": (_base_dataset_timer, _live_cb),
        "science": (_ql_timer, _wxcam_ql_timer),
        "housekeeping": (_hk_timer,),
        "auroracam": (_auroracam_timer,),
        "uas": (_uas_timer,),
        "operations": (_operations_timer,),
    }
    active_timers = set(timer_groups.get(active, ()))
    for timers in timer_groups.values():
        for timer in timers:
            try:
                if timer in active_timers:
                    if not getattr(timer, "running", False):
                        if _APP_BOOTSTRAPPING:
                            continue
                        try:
                            asyncio.get_running_loop()
                        except RuntimeError:
                            continue
                        timer.start()
                elif getattr(timer, "running", False):
                    timer.stop()
            except RuntimeError:
                # Plain imports and sessions shutting down do not have a live
                # event loop. The session-destroyed hook still stops all timers.
                pass
    if active == "overview":
        _refresh_browser_overview()
    elif active in {"interactive", "power"}:
        _sync_browser_tab_instrument(active)
        if not _APP_BOOTSTRAPPING and not _INTERACTIVE_RENDER_ENABLED:
            _enable_browser_interactive_render()
    elif active == "science" and "science" not in _LOADED_TABS:
        science_quicklook_container[:] = [_science_quicklook_image]
        science_status_container[:] = [science_status]
        science_availability_container[:] = [science_availability]
        _LOADED_TABS.add("science")
    elif active == "housekeeping" and "housekeeping" not in _LOADED_TABS:
        _refresh_housekeeping_tab_content()
        _LOADED_TABS.add("housekeeping")
    elif active == "auroracam":
        _refresh_auroracam_latest_if_needed()
    elif active == "uas" and "uas" not in _LOADED_TABS:
        uas_container[:] = [
            pn.Column(
                uas_flight_status_pane,
                uas_flight_plot_pane,
                uas_status_pane,
                uas_plot_pane,
                uas_table_pane,
                sizing_mode="stretch_width",
                css_classes=["uas-shell"],
            )
        ]
        _refresh_uas_dashboard()
        _LOADED_TABS.add("uas")
    elif active == "operations" and "operations" not in _LOADED_TABS:
        operations_container[:] = [operations_dashboard]
        _refresh_operations_dashboard()
        _LOADED_TABS.add("operations")


def _prepare_initial_stable_tab_content(slug: str | None) -> None:
    """Populate stable direct-link panes before the initial document is built."""
    active = _normalize_tab_slug(slug)
    if active == "overview":
        _refresh_browser_overview()
    elif active == "housekeeping" and "housekeeping" not in _LOADED_TABS:
        _refresh_housekeeping_tab_content()
        _LOADED_TABS.add("housekeeping")


def _schedule_active_tab_load(slug: str | None = None) -> None:
    """Load a tab only after its stable host is attached on the client."""
    global _ACTIVE_TAB_LOAD_GENERATION
    active = _normalize_tab_slug(slug or ACTIVE_TAB_SLUG)
    _ACTIVE_TAB_LOAD_GENERATION += 1
    generation = _ACTIVE_TAB_LOAD_GENERATION
    if _APP_BOOTSTRAPPING:
        return
    doc = pn.state.curdoc
    if doc is None:
        return

    def load_if_current() -> None:
        if generation != _ACTIVE_TAB_LOAD_GENERATION:
            return
        if active != ACTIVE_TAB_SLUG or desktop_tabs.active != TAB_INDEX_BY_SLUG[active]:
            return
        _ensure_active_tab_loaded(active)

    doc.add_next_tick_callback(load_if_current)


def _set_active_tab(slug: str | None) -> None:
    """Select one full-name desktop tab, then load it after attachment."""
    global ACTIVE_TAB_SLUG, _desktop_tab_syncing
    active = _normalize_tab_slug(slug)
    ACTIVE_TAB_SLUG = active
    tab_index = TAB_INDEX_BY_SLUG[active]
    if desktop_tabs.active != tab_index:
        _desktop_tab_syncing = True
        try:
            desktop_tabs.active = tab_index
        finally:
            _desktop_tab_syncing = False
    _schedule_active_tab_load(active)
    _refresh_share_and_download_state()


def _on_desktop_tab_change(event) -> None:
    """Lazy-load the selected desktop tab and keep its share URL current."""
    global ACTIVE_TAB_SLUG
    if _desktop_tab_syncing:
        return
    ACTIVE_TAB_SLUG = TAB_SLUG_BY_INDEX.get(event.new, "interactive")
    _schedule_active_tab_load(ACTIVE_TAB_SLUG)
    _refresh_share_and_download_state()


desktop_tabs.param.watch(_on_desktop_tab_change, "active")

SITE_FOOTER_HTML = """
<div class="site-footer">
  <div class="site-footer__title">More Aurora project information</div>
  <div class="site-footer__links">
    <div class="site-footer__link">
      <a href="https://gamb2le.pages.dev/" target="_blank" rel="noopener noreferrer">gamb2le.pages.dev</a>
      <div class="site-footer__desc">Project documentation portal with dashboard, infrastructure, and operational notes.</div>
    </div>
    <div class="site-footer__link">
      <a href="https://www.gamb2le.co.uk/" target="_blank" rel="noopener noreferrer">gamb2le.co.uk</a>
      <div class="site-footer__desc">Main Gamb2le website with broader project context and public-facing information.</div>
    </div>
  </div>
</div>
"""


def _site_footer_pane() -> pn.pane.HTML:
    return pn.pane.HTML(SITE_FOOTER_HTML, sizing_mode="stretch_width", margin=0)


interactive_tab.append(_site_footer_pane())
browser_overview_tab.append(_site_footer_pane())
science_quicklooks_tab.append(_site_footer_pane())
housekeeping_quicklooks_tab.append(_site_footer_pane())
auroracam_tab.append(_site_footer_pane())
uas_tab.append(_site_footer_pane())
operations_tab.append(_site_footer_pane())

_QUERY_TAB_SLUGS = set(TAB_PANEL_BY_SLUG)
_MOBILE_LAYOUT_ACTIVE = _mobile_view_requested()

_apply_query_state()
if _MOBILE_LAYOUT_ACTIVE:
    main_layout = _build_mobile_layout()
else:
    main_layout = pn.Column(desktop_tabs, sizing_mode="stretch_width", margin=0)
    requested_args = _request_query_args()
    requested_tab = requested_args.get("tab")
    # Preserve existing Power share links while moving the desktop Power view
    # out of the generic instrument browser.
    if requested_tab == "interactive" and requested_args.get("instrument") == "power":
        requested_tab = "power"
    _set_active_tab(requested_tab if requested_tab in _QUERY_TAB_SLUGS else "interactive")
    # Overview and housekeeping use stable HTML panes, so their final direct-
    # link state can be serialized into the initial document without adding
    # model references. The onload path then only enables the active timers.
    _prepare_initial_stable_tab_content(ACTIVE_TAB_SLUG)
    _refresh_share_and_download_state()
_APP_BOOTSTRAPPING = False
pn.state.onload(lambda: _schedule_active_tab_load(ACTIVE_TAB_SLUG))

site_env_banner = _site_env_banner_pane()
browser_performance_probe = _browser_performance_probe()
template.main[:] = (
    ([site_env_banner] if site_env_banner is not None else [])
    + ([browser_performance_probe] if browser_performance_probe is not None else [])
    + [main_layout]
)

# Serve the app. `location=True` installs Panel's Location model so the app can
# keep the browser URL aligned with the selected view.
template.servable(location=True)
