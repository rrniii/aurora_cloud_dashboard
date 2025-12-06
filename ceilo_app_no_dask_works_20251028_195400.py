import os
import re
from datetime import datetime, timedelta
import numpy as np

import panel as pn
import holoviews as hv
import hvplot.xarray  # noqa: F401
import xarray as xr

hv.extension("bokeh")
pn.extension(notifications=True)

# =========================
# CONFIG
# =========================
DATA_DIR = os.environ.get("CEILOMETER_DIR", "/mnt/data/cl61")
FILE_REGEX = re.compile(r".*_(\d{8})_(\d{6})\.nc$")
GLOB_SUFFIX = ".nc"

VAR_BETA = "beta_att"
VAR_LDR = "linear_depol_ratio"

CANDIDATE_TIME = ["time", "Time", "datetime", "Datetime", "DATE_TIME"]
CANDIDATE_HEIGHT = ["height", "altitude", "range", "Height", "Altitude", "Range"]

DEFAULT_WINDOW = timedelta(hours=2)   # last 24h by default
REFRESH_MS = 60_000

# Scan only around the requested window to avoid trawling the entire directory
SCAN_PADDING = timedelta(hours=2)
FILES_CACHE_TTL = 60.0  # seconds

# =========================
# HELPERS
# =========================
def _is_numeric_dtype(da):
    try:
        return np.issubdtype(getattr(da, "dtype", None), np.number)
    except Exception:
        return False

def _parse_ts_from_name(fname: str):
    m = FILE_REGEX.match(fname)
    if not m:
        return None
    d, t = m.groups()
    try:
        return datetime.strptime(d + t, "%Y%m%d%H%M%S")  # naive UTC
    except Exception:
        return None

_FILES_CACHE = {"ts": 0.0, "paths": []}

def _list_files_sorted():
    now = datetime.utcnow().timestamp()
    if now - _FILES_CACHE["ts"] < FILES_CACHE_TTL:
        return _FILES_CACHE["paths"]
    entries = []
    try:
        for fn in os.listdir(DATA_DIR):
            if not fn.endswith(GLOB_SUFFIX):
                continue
            ts = _parse_ts_from_name(fn)
            if ts is not None:
                entries.append((ts, os.path.join(DATA_DIR, fn)))
    except FileNotFoundError:
        entries = []
    entries.sort(key=lambda x: x[0])
    paths = [p for _, p in entries]
    _FILES_CACHE["ts"] = now
    _FILES_CACHE["paths"] = paths
    return paths

def _time_height_names(ds: xr.Dataset):
    tname = hname = None
    for cand in CANDIDATE_TIME:
        if cand in ds.coords or cand in ds.variables:
            tname = cand
            break
    for cand in CANDIDATE_HEIGHT:
        if cand in ds.coords or cand in ds.variables:
            hname = cand
            break
    if tname is None or hname is None:
        raise ValueError(
            f"Could not find time/height coords. Found time={tname} height={hname}. "
            f"Available: {list(ds.coords) + list(ds.variables)}"
        )
    return tname, hname

def _files_for_timerange(t0: datetime, t1: datetime):
    scan_start = t0 - SCAN_PADDING
    scan_end = t1 + SCAN_PADDING
    outs = []
    for path in _list_files_sorted():
        ts = _parse_ts_from_name(os.path.basename(path))
        if ts and scan_start <= ts <= scan_end:
            outs.append(path)
    return outs

def _safe_open(path: str) -> xr.Dataset:
    # Avoid crashy stacks; prefer h5netcdf first
    try:
        return xr.open_dataset(path, engine="h5netcdf", decode_cf=True)
    except Exception:
        return xr.open_dataset(path, decode_cf=True)

def _infer_height_limit(hcoord: xr.DataArray) -> float:
    # If units say km → 8; else assume meters → 8000
    units = str(hcoord.attrs.get("units", "")).lower()
    if "km" in units:
        return 8.0
    # Heuristic: if typical values are < 40, probably km already
    vals = np.asarray(hcoord.values)
    finite = vals[np.isfinite(vals)]
    if finite.size and np.nanmedian(finite) < 40:
        return 8.0
    return 8000.0

def _limit_height(ds: xr.Dataset, hname: str) -> xr.Dataset:
    cutoff = _infer_height_limit(ds[hname])
    return ds.sel(**{hname: ds[hname] <= cutoff})

def _open_slice(t0: datetime, t1: datetime) -> xr.Dataset:
    files = _files_for_timerange(t0, t1)
    if not files:
        return xr.Dataset()

    parts = []
    for f in files:
        try:
            parts.append(_safe_open(f))
        except Exception:
            continue
    if not parts:
        return xr.Dataset()

    ds = xr.combine_by_coords(parts, combine_attrs="override")

    # Close promptly
    for p in parts:
        try:
            p.close()
        except Exception:
            pass

    tname, hname = _time_height_names(ds)
    ds = ds.sortby(tname)

    # Limit to requested window and 0–8 km
    ds = ds.sel(**{tname: slice(t0, t1)})
    ds = _limit_height(ds, hname)

    # Cast data vars if needed
    for v in (VAR_BETA, VAR_LDR):
        if v in ds and not _is_numeric_dtype(ds[v]):
            try:
                ds[v] = ds[v].astype("float32", copy=False)
            except Exception:
                try:
                    ds[v] = xr.DataArray(
                        np.asarray(ds[v].values, dtype="float32"),
                        coords=ds[v].coords, dims=ds[v].dims, attrs=ds[v].attrs
                    )
                except Exception:
                    pass
    return ds

def _plot_var(ds: xr.Dataset, var: str, tname: str, hname: str, title: str,
              clim=None, logz=False):
    if var not in ds:
        return hv.Text(0.5, 0.5, f"{var} not found").opts(title=title)

    # Ensure ascending height for intuitive axes
    h = ds[hname]
    if h.ndim == 1 and h.values[0] > h.values[-1]:
        ds = ds.assign_coords({hname: h[::-1]})
        ds[var] = ds[var].isel({hname: slice(None, None, -1)})

    # quadmesh handles irregular time spacing; rasterize keeps it snappy
    img = ds[var].hvplot.quadmesh(
        x=tname,
        y=hname,
        clim=clim,
        cnorm="linear",
        logz=logz,
        colorbar=True,
        title=title,
        width=1000,
        height=360,
        rasterize=True,
    )
    return img

# =========================
# WIDGETS
# =========================
now_naive_utc = datetime.utcnow()
range_end = pn.widgets.DatetimePicker(name="End", value=now_naive_utc)
range_start = pn.widgets.DatetimePicker(name="Start", value=now_naive_utc - DEFAULT_WINDOW)
live_toggle = pn.widgets.Toggle(name="Live", value=True)
refresh_btn = pn.widgets.Button(name="Refresh", button_type="primary")
status_alert = pn.pane.Alert("Initializing… loading selected window.", alert_type="info", sizing_mode="stretch_width")

# =========================
# DATA + VIEW
# =========================
@pn.cache(max_items=6)
def load_window(t0: datetime, t1: datetime) -> xr.Dataset:
    if t0 is None or t1 is None or t0 >= t1:
        return xr.Dataset()
    try:
        return _open_slice(t0, t1)
    except Exception as e:
        status_alert.object = f"Data load error: {e}"
        status_alert.alert_type = "danger"
        return xr.Dataset()

def build_plots(t0: datetime, t1: datetime):
    ds = load_window(t0, t1)
    if not ds.data_vars:
        return hv.Curve([]).opts(title="No data in window") + hv.Curve([])

    try:
        tname, hname = _time_height_names(ds)
    except Exception as e:
        return hv.Text(0.5, 0.5, str(e)).opts(title="Dataset issue") + hv.Curve([])

    p_beta = _plot_var(
        ds, VAR_BETA, tname, hname,
        "Attenuated Backscatter",
        clim=(1e-7, 1e-4),
        logz=True,          # log stretch for beta range
    )
    p_ldr = _plot_var(
        ds, VAR_LDR, tname, hname,
        "Linear Depolarization Ratio",
        clim=(0.0, 1.0),
        logz=False,
    )

    return (p_beta + p_ldr).opts(shared_axes=True).cols(1)

def view(t0: datetime, t1: datetime):
    return build_plots(t0, t1)

view_bound = pn.bind(view, range_start, range_end)

# =========================
# LIVE MODE / CALLBACKS
# =========================
def _tick():
    if not live_toggle.value:
        return
    end = datetime.utcnow()
    start = end - DEFAULT_WINDOW
    nudged = False
    if range_end.value is None or abs((end - range_end.value).total_seconds()) > 10:
        range_end.value = end
        nudged = True
    if range_start.value is None or abs((start - range_start.value).total_seconds()) > 10:
        range_start.value = start
        nudged = True
    if nudged:
        pass  # widget change triggers view refresh

def _manual_refresh(_=None):
    if range_end.value:
        range_end.value = range_end.value + timedelta(microseconds=1)

refresh_btn.on_click(_manual_refresh)
periodic = pn.state.add_periodic_callback(_tick, period=REFRESH_MS, start=True)

def _on_live_toggle(event):
    if event.new:
        status_alert.object = "Live mode: window follows current time."
        status_alert.alert_type = "success"
        _tick()
        periodic.start()
    else:
        status_alert.object = "Live mode off. Pan/zoom or set Start/End."
        status_alert.alert_type = "warning"
        periodic.stop()

live_toggle.param.watch(_on_live_toggle, "value")

# =========================
# LAYOUT
# =========================
controls = pn.Row(range_start, range_end, live_toggle, refresh_btn, sizing_mode="stretch_width")

dashboard = pn.Column(
    pn.pane.Markdown("### GAMB2LE Ceilometer"),
    controls,
    status_alert,
    view_bound,
    sizing_mode="stretch_both",
)

dashboard.servable()
