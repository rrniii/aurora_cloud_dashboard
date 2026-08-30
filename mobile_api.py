"""FastAPI wrapper for the Aurora Dashboard mobile API."""

from __future__ import annotations

import mimetypes
import os
import secrets
from pathlib import Path
from time import perf_counter
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, status
from fastapi.responses import FileResponse, Response

import mobile_catalog as catalog


app = FastAPI(
    title="Aurora Dashboard Mobile API",
    version="0.3.0",
    root_path=os.environ.get("AURORA_MOBILE_API_ROOT_PATH", ""),
)

_PUBLIC_READ_ONLY_MODE = "public_read_only"


def _auth_mode() -> str:
    """Return the configured access mode, failing closed on unknown values."""
    value = os.environ.get("AURORA_MOBILE_API_AUTH_MODE", "required").strip().lower()
    return _PUBLIC_READ_ONLY_MODE if value == _PUBLIC_READ_ONLY_MODE else "required"


def _is_public_read(request: Request) -> bool:
    return _auth_mode() == _PUBLIC_READ_ONLY_MODE and request.method in {"GET", "HEAD"}


@app.middleware("http")
async def cache_read_only_payloads(request: Request, call_next):
    """Permit short client caching for small authenticated JSON payloads.

    Media responses set their own cache policy and raw/Zarr paths are never
    exposed by this API.  A private response avoids sharing authenticated
    payloads in intermediary caches.
    """
    started = perf_counter()
    response = await call_next(request)
    response.headers.setdefault("Server-Timing", f"app;dur={(perf_counter() - started) * 1000:.1f}")
    if request.method == "GET" and not request.url.path.startswith("/media/"):
        visibility = "public" if _is_public_read(request) else "private"
        response.headers.setdefault("Cache-Control", f"{visibility}, max-age=30, stale-while-revalidate=60")
    return response


def _token() -> str | None:
    value = os.environ.get("AURORA_MOBILE_API_TOKEN")
    if value and value.strip():
        return value.strip()
    token_file = os.environ.get("AURORA_MOBILE_API_TOKEN_FILE")
    if token_file:
        try:
            value = Path(token_file).read_text(encoding="utf-8").strip()
        except OSError:
            value = ""
        if value:
            return value
    return None


def require_auth(authorization: Annotated[str | None, Header()] = None) -> None:
    expected = _token()
    if not expected:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Mobile API token is not configured")
    scheme, _, value = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not secrets.compare_digest(value, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid mobile API token")


def require_read_access(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    """Allow bounded read-only app data on explicitly public dashboard hosts."""
    if _is_public_read(request):
        return
    require_auth(authorization)


def _not_found(message: str = "Resource not found") -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=message)


def _file_response(path: Path, request: Request) -> Response:
    if not path.exists() or not path.is_file():
        raise _not_found()
    stat_result = path.stat()
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    headers = {
        "Cache-Control": "private, max-age=60",
        "ETag": f'W/"{stat_result.st_mtime_ns}-{stat_result.st_size}"',
        "Last-Modified": catalog.datetime.fromtimestamp(stat_result.st_mtime, catalog.UTC).strftime("%a, %d %b %Y %H:%M:%S GMT"),
    }
    if request.headers.get("if-none-match") == headers["ETag"]:
        return Response(status_code=status.HTTP_304_NOT_MODIFIED, headers=headers)
    return FileResponse(path, media_type=media_type, headers=headers)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "serverTime": catalog.utc_now_iso(),
        "authRequired": _auth_mode() != _PUBLIC_READ_ONLY_MODE,
        "accessMode": _auth_mode(),
    }


@app.get("/manifest", dependencies=[Depends(require_read_access)])
def manifest() -> dict:
    return catalog.manifest()


@app.get("/artifacts/manifest", dependencies=[Depends(require_auth)])
def display_artifacts() -> dict:
    """Expose derived browser artifacts without exposing raw or Zarr products."""
    return catalog.display_artifacts()


@app.get("/operations", dependencies=[Depends(require_read_access)])
def operations() -> dict:
    return catalog.operations()


@app.get("/overview", dependencies=[Depends(require_read_access)])
def overview() -> dict:
    return catalog.overview()


@app.get("/power", dependencies=[Depends(require_read_access)])
def power(
    window: str = Query("24h", pattern="^(24h|96h)$"),
    group: str = Query(
        "all",
        pattern="^(all|current|forecast|observed|forecast_24h|forecast_96h|verification)$",
    ),
) -> dict:
    try:
        return catalog.power(window=window, group=group)
    except KeyError as exc:
        raise _not_found(str(exc)) from exc


@app.get("/power/solar-evaluation", dependencies=[Depends(require_read_access)])
def power_solar_evaluation(
    lane: str = Query(
        "D_physical_solar_load_residual",
        pattern="^(B_physical_solar|C_load_residual|D_physical_solar_load_residual)$",
    ),
) -> dict:
    """Development-only, explicitly-labelled paired v12 candidate evidence."""
    try:
        return catalog.power_solar_evaluation(lane=lane)
    except KeyError as exc:
        raise _not_found(str(exc)) from exc


@app.get("/power/cl61-automation", dependencies=[Depends(require_read_access)])
def power_cl61_automation() -> dict:
    """Read-only CL61 automation readiness and shadow-decision status."""
    return catalog.cl61_automation_status()


@app.get("/media/power/figure/{section}", dependencies=[Depends(require_read_access)])
def power_figure(request: Request, section: str) -> Response:
    """Serve a cacheable prewarmed Plotly figure without exposing a Zarr store."""
    path = catalog.power_prewarm_path(section)
    if path is None:
        raise _not_found("Unsupported Power figure section")
    if not path.exists():
        raise _not_found("Power figure is not available yet")
    return _file_response(path, request)


@app.get("/auroracam", dependencies=[Depends(require_read_access)])
def auroracam(day: str = Query("latest"), time_utc: str | None = Query(None)) -> dict:
    try:
        return catalog.auroracam(day=day, time_utc=time_utc)
    except KeyError as exc:
        raise _not_found(str(exc)) from exc


@app.get("/uas", dependencies=[Depends(require_read_access)])
def uas(window: str = Query("24h", pattern="^(24h|7d|all)$")) -> dict:
    try:
        return catalog.uas(window=window)
    except KeyError as exc:
        raise _not_found(str(exc)) from exc


@app.get("/uas/flights", dependencies=[Depends(require_read_access)])
def uas_flights(day: str = Query("latest", pattern=r"^(latest|20\d{2}-\d{2}-\d{2})$")) -> dict:
    try:
        return catalog.uas_flights(day=day)
    except KeyError as exc:
        raise _not_found(str(exc)) from exc


@app.get("/uas/flights/{flight_id}", dependencies=[Depends(require_read_access)])
def uas_flight(flight_id: str) -> dict:
    try:
        return catalog.uas_flight(flight_id)
    except KeyError as exc:
        raise _not_found(str(exc)) from exc


@app.get("/instruments/{instrument_id}/summary", dependencies=[Depends(require_read_access)])
def instrument_summary(instrument_id: str, window: str = Query("24h", pattern="^(24h|7d)$")) -> dict:
    try:
        return catalog.instrument_summary(instrument_id, window)
    except KeyError as exc:
        raise _not_found(str(exc)) from exc


@app.get("/quicklooks", dependencies=[Depends(require_read_access)])
def quicklooks(kind: str = Query("science", pattern="^(science|housekeeping)$"), instrument: str = "power") -> dict:
    try:
        return catalog.quicklooks(kind, instrument)
    except KeyError as exc:
        raise _not_found(str(exc)) from exc


@app.get("/wxcam", dependencies=[Depends(require_read_access)])
def wxcam(stream: str = Query("fish_hdr"), day: str = Query("latest")) -> dict:
    try:
        return catalog.wxcam(stream, day)
    except KeyError as exc:
        raise _not_found(str(exc)) from exc


@app.get("/media/quicklook/{kind}/{instrument_id}/{token}", dependencies=[Depends(require_read_access)])
def quicklook_media(request: Request, kind: str, instrument_id: str, token: str) -> Response:
    try:
        path = catalog.resolve_quicklook_path(kind, instrument_id, token)
    except KeyError as exc:
        raise _not_found(str(exc)) from exc
    if not path:
        raise _not_found("Quicklook image not found")
    return _file_response(path, request)


@app.get("/media/uas/flights/{flight_id}", dependencies=[Depends(require_read_access)])
def uas_flight_plot(request: Request, flight_id: str) -> Response:
    try:
        path = catalog.resolve_uas_flight_plot_path(flight_id)
    except KeyError as exc:
        raise _not_found(str(exc)) from exc
    if not path:
        raise _not_found("UAS flight plot not found")
    return _file_response(path, request)


@app.get("/media/wxcam/video/{stream}/{day}", dependencies=[Depends(require_read_access)])
def wxcam_video(request: Request, stream: str, day: str) -> Response:
    try:
        resolved = catalog.wxcam(stream, day)["selectedDay"]
        path = catalog.resolve_wxcam_video_path(stream, resolved)
    except KeyError as exc:
        raise _not_found(str(exc)) from exc
    if not path:
        raise _not_found("WXcam video not found")
    return _file_response(path, request)


@app.get("/media/wxcam/thumb/{stream}/{day_token}/{filename}", dependencies=[Depends(require_read_access)])
def wxcam_thumbnail(request: Request, stream: str, day_token: str, filename: str) -> Response:
    path = catalog.resolve_wxcam_thumbnail_path(stream, day_token, filename)
    if not path:
        raise _not_found("WXcam thumbnail not found")
    return _file_response(path, request)


@app.get("/media/auroracam/original/{camera_id}/{day}/{filename}", dependencies=[Depends(require_read_access)])
def auroracam_original(request: Request, camera_id: str, day: str, filename: str) -> Response:
    path = catalog.resolve_auroracam_image_path(camera_id, day, filename)
    if not path:
        raise _not_found("AURORACam image not found")
    return _file_response(path, request)


@app.get("/media/auroracam/preview/{camera_id}/{day}/{filename}", dependencies=[Depends(require_read_access)])
def auroracam_preview(request: Request, camera_id: str, day: str, filename: str) -> Response:
    source = catalog.resolve_auroracam_image_path(camera_id, day, filename)
    if not source:
        raise _not_found("AURORACam image not found")
    try:
        preview = catalog.create_auroracam_preview(source)
    except OSError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"AURORACam preview unavailable: {exc}") from exc
    return _file_response(preview, request)
