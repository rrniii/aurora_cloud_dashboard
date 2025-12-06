import os
import re
from datetime import datetime, timedelta
import numpy as np

import panel as pn
import holoviews as hv
import hvplot.xarray  # noqa: F401
import xarray as xr

# ----------------------------
# DASK: ON (2 workers, 2 threads each)
# ----------------------------
USE_DASK = os.environ.get("USE_DASK", "1") != "0"
dask_error = None
if USE_DASK:
    try:
        import dask
        from dask.distributed import Client, LocalCluster
        cluster = LocalCluster(
            processes=False,            # in-process to avoid HDF5 fights
            n_workers=2,                # <- per your request
            threads_per_worker=2,       # <- sensible concurrency
            dashboard_address=None,
        )
        client = Client(cluster)
        dask.config.set({
            "array.slicing.split_large_chunks": True,
            "optimization.fuse.active": True,
        })
    except Exception as e:
        dask_error = repr(e)
        USE_DASK = False

hv.extension("bokeh")
pn.extension(notifications=True)

# =========================
# CONFIG
# =========================
DATA_DIR = os.environ.get("CEILOMETER_DIR", "/mnt/data/cl61")
FILE_REGEX = re.compile(r".*_(\d{8})_(\d{6})\.nc$")
GLOB_SUFFIX = ".nc"

VAR_BETA = "beta_att"
VAR_LDR  = "linear_depol_ratio"
TARGET_VARS = (VAR_BETA, VAR_LDR)

CANDIDATE_TIME   = ["time", "Time", "datetime", "Datetime", "DATE_TIME"]
CANDIDATE_HEIGHT = ["height", "altitude", "range", "Height", "Altitude", "Range"]

DEFAULT_WINDOW  = timedelta(minutes=20)   # last 20 min
SCAN_PADDING    = timedelta(minutes=20)   # limit directory scan
FILES_CACHE_TTL = 30.0
REFRESH_MS      = 60_000

HEIGHT_MAX_M = 6000.0
BETA_CLIM = (1e-7, 1e-4)
LDR_CLIM  = (0.0, 0.7)

# =========================
# UTILS
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
        return datetime.strptime(d + t, "%Y%m%d%H%M%S")
    except Exception:
        return None

def _to_py_datetime_ms(epoch_ms):
    # Bokeh x_range uses ms since epoch (float). Be tolerant of None.
    if epoch_ms is None:
        return None
    return datetime.utcfromtimestamp(float(epoch_ms) / 1000.0)

# =========================
# DIR LIST CACHE
# =========================
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

# =========================
# IO + COMBINE
# =========================
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
    scan_end   = t1 + SCAN_PADDING
    outs = []
    for path in _list_files_sorted():
        ts = _parse_ts_from_name(os.path.basename(path))
        if ts and scan_start <= ts <= scan_end:
            outs.append(path)
    return outs

def _safe_open(path: str) -> xr.Dataset:
    try:
        return xr.open_dataset(path, engine="h5netcdf", decode_cf=True)
    except Exception:
        return xr.open_dataset(path, decode_cf=True)

def _normalize_vars(ds: xr.Dataset) -> xr.Dataset:
    for v in TARGET_VARS:
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

def _nonempty_slice(ds: xr.Dataset) -> bool:
    try:
        tname, hname = _time_height_names(ds)
        return (ds.sizes.get(tname, 0) > 0) and (ds.sizes.get(hname, 0) > 0)
    except Exception:
        return False

def _preprocess_factory(t0: datetime, t1: datetime):
    def _preprocess(ds: xr.Dataset) -> xr.Dataset:
        try:
            tname, hname = _time_height_names(ds)
        except Exception:
            return ds
        keep = [v for v in TARGET_VARS if v in ds]
        if keep:
            ds = ds[keep]
        try:
            ds = ds.sel(**{tname: slice(t0, t1)})
        except Exception:
            pass
        try:
            ds = ds.sel(**{hname: ds[hname] <= HEIGHT_MAX_M})
        except Exception:
            pass
        return ds
    return _preprocess

def _open_slice_nodask(t0: datetime, t1: datetime) -> xr.Dataset:
    files = _files_for_timerange(t0, t1)
    if not files:
        return xr.Dataset()
    parts = []
    for f in files:
        try:
            dsf = _safe_open(f)
            try:
                tname, hname = _time_height_names(dsf)
                dsf = dsf.sel(**{tname: slice(t0, t1)})
                dsf = dsf.sel(**{hname: dsf[hname] <= HEIGHT_MAX_M})
                keep = [v for v in TARGET_VARS if v in dsf]
                if keep:
                    dsf = dsf[keep]
            except Exception:
                pass
            if _nonempty_slice(dsf):
                parts.append(dsf)
            else:
                try: dsf.close()
                except Exception: pass
        except Exception:
            continue
    if not parts:
        return xr.Dataset()
    ds = xr.combine_by_coords(parts, combine_attrs="override")
    for p in parts:
        try: p.close()
        except Exception: pass
    ds = _normalize_vars(ds)
    try:
        tname, _ = _time_height_names(ds)
        ds = ds.sortby(tname)
    except Exception:
        pass
    return ds

def _open_slice_dask(t0: datetime, t1: datetime) -> xr.Dataset:
    files = _files_for_timerange(t0, t1)
    if not files:
        return xr.Dataset()
    preprocess = _preprocess_factory(t0, t1)
    try:
        ds = xr.open_mfdataset(
            files,
            engine="h5netcdf",
            combine="by_coords",
            parallel=True,
            preprocess=preprocess,
            chunks="auto",
            data_vars="minimal",
            coords="minimal",
            compat="override",
        )
    except Exception:
        return _open_slice_nodask(t0, t1)
    ds = _normalize_vars(ds)
    try:
        tname, _ = _time_height_names(ds)
        ds = ds.sortby(tname)
    except Exception:
        pass
    return ds

def _open_slice(t0: datetime, t1: datetime) -> xr.Dataset:
    return _open_slice_dask(t0, t1) if USE_DASK else _open_slice_nodask(t0, t1)

# =========================
# PLOTTING
# =========================
def _plot_var(ds: xr.Dataset, var: str, tname: str, hname: str, title: str,
              clim=None, logz=False, hooks=None):
    if var not in ds:
        return hv.Text(0.5, 0.5, f"{var} not found").opts(title=title)
    if ds.sizes.get(tname, 0) == 0 or ds.sizes.get(hname, 0) == 0:
        return hv.Text(0.5, 0.5, f"No {var} data in selected window").opts(title=title)
    # Ascending height
    h = ds[hname]
    if h.ndim == 1 and h.values[0] > h.values[-1]:
        ds = ds.assign_coords({hname: h[::-1]})
        ds[var] = ds[var].isel({hname: slice(None, None, -1)})
    img = ds[var].hvplot.quadmesh(
        x=tname,
        y=hname,
        ylabel="Range (m)",
        ylim=(0, HEIGHT_MAX_M),
        clim=clim,
        cnorm="linear",
        logz=logz,
        aggregator="mean",
        colorbar=True,
        title=title,
        width=950,
        height=340,
        rasterize=True,
        hooks=hooks or [],
    )
    return img

# =========================
# WIDGETS + STATUS
# =========================
now_utc = datetime.utcnow()
range_end   = pn.widgets.DatetimePicker(name="End",   value=now_utc)
range_start = pn.widgets.DatetimePicker(name="Start", value=now_utc - DEFAULT_WINDOW)
live_toggle = pn.widgets.Toggle(name="Live: On", value=True, button_type="success")
refresh_btn = pn.widgets.Button(name="Refresh", button_type="primary")

hint = f"**Dask:** {'ON' if USE_DASK else 'OFF'}"
if not USE_DASK and dask_error:
    hint += f" — {dask_error}"
dask_hint = pn.pane.Markdown(hint, width=420)

status_alert = pn.pane.Alert("", alert_type="info", sizing_mode="stretch_width", visible=False)

def _status(msg, level="info", transient=False, ms=1500):
    status_alert.object = msg
    status_alert.alert_type = level
    status_alert.visible = True
    if transient:
        def _hide():
            status_alert.visible = False
            status_alert.object = ""
        pn.state.add_periodic_callback(_hide, period=ms, count=1, start=True)

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
        _status(f"Data load error: {e}", "danger")
        return xr.Dataset()

# Bokeh hook: when user pans/zooms, update Start/End and force reload
def _attach_xrange_sync(plot, element):
    try:
        xrng = plot.state.x_range
    except Exception:
        return
    if getattr(plot.state, "_ceilometer_cb_attached", False):
        return
    plot.state._ceilometer_cb_attached = True

    def _on_change(attr, old, new):
        # If user interacts, turn Live off so the view sticks
        if live_toggle.value:
            live_toggle.value = False  # watcher updates label/color
        start = _to_py_datetime_ms(xrng.start)
        end   = _to_py_datetime_ms(xrng.end)
        if start is None or end is None:
            return
        # Update pickers
        range_start.value = start
        # Force a refresh even if Panel thinks nothing changed
        range_end.value   = end + timedelta(microseconds=1)
        _status("Updated from pan/zoom.", "info", transient=True, ms=900)

    xrng.on_change("start", _on_change)
    xrng.on_change("end",   _on_change)

def build_plots(t0: datetime, t1: datetime):
    ds = load_window(t0, t1)

    if not ds.data_vars:
        _status("No variables found in window.", "warning", transient=True)
        return hv.Text(0.5, 0.5, "No variables found").opts(title="Nothing to show") + hv.Curve([])

    try:
        tname, hname = _time_height_names(ds)
    except Exception as e:
        _status(str(e), "danger")
        return hv.Text(0.5, 0.5, str(e)).opts(title="Dataset issue") + hv.Curve([])

    if ds.sizes.get(tname, 0) == 0 or ds.sizes.get(hname, 0) == 0:
        _status("No data in selected Start/End range.", "warning", transient=True)
        return hv.Text(0.5, 0.5, "Empty window").opts(title="Empty") + hv.Curve([])

    hooks = [_attach_xrange_sync]

    p_beta = _plot_var(ds, VAR_BETA, tname, hname, "Attenuated Backscatter (beta_att)",
                       clim=BETA_CLIM, logz=True, hooks=hooks)
    p_ldr  = _plot_var(ds, VAR_LDR,  tname, hname, "Linear Depolarization Ratio",
                       clim=LDR_CLIM,  logz=False, hooks=hooks)

    _status("Data loaded successfully.", "success", transient=True)
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
    end   = datetime.utcnow()
    start = end - DEFAULT_WINDOW
    nudged = False
    if range_end.value is None or abs((end - range_end.value).total_seconds()) > 5:
        range_end.value = end; nudged = True
    if range_start.value is None or abs((start - range_start.value).total_seconds()) > 5:
        range_start.value = start; nudged = True
    if nudged:
        _status("Live window updated.", "info", transient=True, ms=800)

def _manual_refresh(_=None):
    if range_end.value:
        range_end.value = range_end.value + timedelta(microseconds=1)
        _status("Manual refresh.", "info", transient=True, ms=800)

refresh_btn.on_click(_manual_refresh)
periodic = pn.state.add_periodic_callback(_tick, period=REFRESH_MS, start=True)

def _on_live_toggle(event):
    if event.new:
        live_toggle.name = "Live: On"
        live_toggle.button_type = "success"
        periodic.start()
        _tick()
        _status("Live enabled.", "success", transient=True, ms=800)
    else:
        live_toggle.name = "Live: Off"
        live_toggle.button_type = "default"
        periodic.stop()
        _status("Live disabled.", "warning", transient=True, ms=800)

live_toggle.param.watch(_on_live_toggle, "value")

# =========================
# LAYOUT
# =========================
controls = pn.Row(
    range_start, range_end, live_toggle, refresh_btn, dask_hint,
    sizing_mode="stretch_width"
)

dashboard = pn.Column(
    pn.pane.Markdown("### Ceilometer: Time vs Height (0–6000 m)"),
    controls,
    status_alert,
    view_bound,
    sizing_mode="stretch_both",
)

dashboard.servable()
