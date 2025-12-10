# app.py
# Minimal Panel + Plotly viewer for ceilometer Zarr data.
# - Loads a Zarr dataset once and slices out a time window for plotting.
# - Two heatmaps: attenuated backscatter (log-scaled) and linear depol ratio.
# - Controls: instrument (placeholder), time window, range limits, color limits,
#   a “live” toggle to jump to the latest 24h, and previous/next day navigation.
# - Lightweight coarsening and subsampling to keep plots responsive.

import os
from datetime import datetime, timedelta, timezone, time
from pathlib import Path

import numpy as np
import pandas as pd
import panel as pn
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import xarray as xr

pn.extension("plotly", notifications=True, sizing_mode="stretch_width")

# --- Configuration ---
ZARR_PATH = os.environ.get(
    "CEILOMETER_ZARR_PATH",
    "/mnt/data/cl61/gamb2le_depolarisation_lidar_ceilometer_aurora_20251201.zarr",
)
DEFAULT_CHUNK_SPEC = {"time": 600}
ZARR_CONSOLIDATED = True
DEFAULT_WINDOW = timedelta(hours=24)
LIVE_REFRESH_MS = 60_000  # how often to snap to latest when live is on (ms)
HEIGHT_LOAD_MAX_M = 10_000  # drop range samples above this
TIME_SUBSAMPLE = 2  # slice time to lighten payloads
TIME_TARGET = 300  # target max time samples for plotting
HEIGHT_TARGET = 200  # target max height samples for plotting
TOP_RANGE_DEFAULT = 8000
BETA_CLIM = (1e-7, 1e-4)
LDR_CLIM = (0.0, 0.5)
LATEST_IMAGE = Path("/home/aurora/aurora_cloud_dashboard/last24h.png")
QUICKLOOK_DIR = Path("/home/aurora/aurora_cloud_dashboard/quicklooks/ceilometer")

_BASE_DS = None
DATA_REFRESH_MS = 300_000  # reload base dataset every 5 minutes


def _ensure_utc(dt):
    """Return a naive UTC datetime (or None) for consistent comparisons."""
    if dt is None:
        return None
    tz = getattr(dt, "tzinfo", None)
    if tz is None:
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def _get_base_dataset():
    """Open the Zarr store (memoized) with configured chunks and consolidation."""
    global _BASE_DS
    if _BASE_DS is not None:
        return _BASE_DS
    try:
        ds = xr.open_zarr(ZARR_PATH, chunks=DEFAULT_CHUNK_SPEC, consolidated=ZARR_CONSOLIDATED)
    except Exception:
        ds = xr.open_zarr(ZARR_PATH, chunks="auto", consolidated=False)
    _BASE_DS = ds
    return ds


def _refresh_base_dataset():
    """Drop the cached dataset so the next access reopens the Zarr (captures new data)."""
    global _BASE_DS
    _BASE_DS = None


def _median_filter_nan(arr, k=3):
    """Simple nan-aware median filter with square window k x k."""
    if arr.ndim != 2 or k < 2:
        return arr
    pad = k // 2
    padded = np.pad(arr, ((pad, pad), (pad, pad)), mode="constant", constant_values=np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(padded, (k, k))
    return np.nanmedian(windows, axis=(-2, -1))


def _dataset_time_bounds():
    """Compute earliest and latest timestamps in the dataset (or None/None)."""
    ds = _get_base_dataset()
    if ds is None or "time" not in ds:
        return None, None
    times = np.asarray(ds["time"].values)
    if times.size == 0:
        return None, None
    return pd.to_datetime(times.min()).to_pydatetime(), pd.to_datetime(times.max()).to_pydatetime()


def _coarsen_targets(duration: timedelta | None, height_span: float | None):
    """Return subsample/target counts that scale up for zoomed-in windows."""
    time_subsample = TIME_SUBSAMPLE
    time_target = TIME_TARGET
    height_target = HEIGHT_TARGET
    if duration is not None:
        hours = duration.total_seconds() / 3600.0
        if hours <= 2:
            time_subsample = 1
            time_target = 1200
        elif hours <= 6:
            time_subsample = 1
            time_target = 800
        elif hours <= 24:
            time_subsample = 1
            time_target = 400
    if height_span is not None:
        if height_span <= 1000:
            height_target = 400
        elif height_span <= 3000:
            height_target = 300
    return time_subsample, time_target, height_target


def open_window(t0, t1, bottom_m=None, top_m=None):
    """Slice the base dataset, adapt coarsening to window span, and filter height."""
    t0 = _ensure_utc(t0)
    t1 = _ensure_utc(t1)
    if t0 is None or t1 is None or t0 >= t1:
        return xr.Dataset()
    duration = t1 - t0
    height_span = None
    if bottom_m is not None or top_m is not None:
        b = max(bottom_m or 0.0, 0.0)
        t = top_m if top_m is not None else HEIGHT_LOAD_MAX_M
        height_span = max(t - b, 0.0)
    time_subsample, time_target, height_target = _coarsen_targets(duration, height_span)
    base = _get_base_dataset()
    if base is None:
        return xr.Dataset()
    try:
        tvals = base["time"].values
        mask = (tvals >= np.datetime64(t0)) & (tvals <= np.datetime64(t1))
        if not np.any(mask):
            return xr.Dataset()
        idx = np.nonzero(mask)[0]
        ds = base.isel(time=idx)
    except Exception:
        ds = base
    try:
        ds = ds.sel({"range": slice(0, HEIGHT_LOAD_MAX_M)})
    except Exception:
        ds = ds.where(ds["range"] <= HEIGHT_LOAD_MAX_M, drop=True)
    # If the user narrowed the plotted range, trim the data before coarsening so
    # we keep more vertical detail within the zoomed band.
    if bottom_m is not None or top_m is not None:
        low = max(bottom_m or 0.0, 0.0)
        high = min(top_m or HEIGHT_LOAD_MAX_M, HEIGHT_LOAD_MAX_M)
        try:
            ds = ds.sel({"range": slice(low, high)})
        except Exception:
            ds = ds.where((ds["range"] >= low) & (ds["range"] <= high), drop=True)
    if time_subsample > 1:
        ds = ds.isel(time=slice(None, None, time_subsample))
    # Coarsen to target sample counts to keep payloads small.
    try:
        if ds.sizes.get("range", 0) > height_target:
            fh = max(int(np.ceil(ds.sizes["range"] / height_target)), 1)
            ds = ds.coarsen({"range": fh}, boundary="trim").mean()
        if ds.sizes.get("time", 0) > time_target:
            ft = max(int(np.ceil(ds.sizes["time"] / time_target)), 1)
            ds = ds.coarsen({"time": ft}, boundary="trim").mean()
    except Exception:
        pass
    return ds


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
    trace = go.Heatmap(
        x=times,
        y=heights,
        z=data,
        zmin=zmin,
        zmax=zmax,
        coloraxis=coloraxis,
        showscale=False,
    )
    return trace


# Widgets / controls (Panel wires these into the view updater)
tmin, tmax = _dataset_time_bounds()
default_end = tmax or datetime.utcnow()
default_start = default_end - DEFAULT_WINDOW
range_start = pn.widgets.DatetimePicker(name="Start (UTC)", value=default_start)
range_end = pn.widgets.DatetimePicker(name="End (UTC)", value=default_end)
top_range_m = pn.widgets.IntInput(name="Top range (m)", value=TOP_RANGE_DEFAULT, step=100, start=500)
bottom_range_m = pn.widgets.IntInput(name="Bottom range (m)", value=0, step=100, start=0)
beta_vmin = pn.widgets.FloatInput(name="β min", value=BETA_CLIM[0], step=1e-7, format="0.000000e+00")
beta_vmax = pn.widgets.FloatInput(name="β max", value=BETA_CLIM[1], step=1e-6, format="0.000000e+00")
ldr_vmin = pn.widgets.FloatInput(name="LDR min", value=LDR_CLIM[0], step=0.05)
ldr_vmax = pn.widgets.FloatInput(name="LDR max", value=LDR_CLIM[1], step=0.05)
prev_btn = pn.widgets.Button(name="Previous Day", button_type="default")
next_btn = pn.widgets.Button(name="Next Day/Current Day", button_type="default")
live_toggle = pn.widgets.Toggle(name="Live Update (Last 24h)", button_type="primary", value=True)
instrument_select = pn.widgets.Select(name="Instrument", value="Ceilometer", options=["Ceilometer"])
calendar_instrument = pn.widgets.Select(name="Instrument", value="Ceilometer", options=["Ceilometer"])

_live_guard = False
_live_cb = None  # handle for periodic callback (used for live refresh)
_relayout_guard = False  # prevents loops when syncing zoom back to widgets
pn.state.add_periodic_callback(_refresh_base_dataset, period=DATA_REFRESH_MS, start=True)


def _refresh_to_latest(_event=None):
    """Jump window to latest 24h and update date pickers."""
    global _live_guard
    _live_guard = True
    tmin, tmax = _dataset_time_bounds()
    end = tmax or datetime.utcnow()
    start = end - DEFAULT_WINDOW
    range_start.value = start
    range_end.value = end
    bottom_range_m.value = 0
    top_range_m.value = TOP_RANGE_DEFAULT
    _live_guard = False


def _set_live(state: bool):
    """Set live toggle state without re-triggering handlers."""
    global _live_guard
    _live_guard = True
    live_toggle.value = state
    live_toggle.name = "Live Update (Last 24h)" if state else "Live Off"
    live_toggle.button_type = "primary" if state else "default"
    _live_guard = False


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
def _auto_refresh():
    """Periodic refresh when live mode is on."""
    if live_toggle.value:
        _refresh_to_latest()


# Kick off periodic live refresh.
_live_cb = pn.state.add_periodic_callback(_auto_refresh, period=LIVE_REFRESH_MS, start=True)


def _shift_previous(_event=None):
    """Jump to the previous full UTC day (00:00–24:00), clamping to data start."""
    _set_live(False)
    tmin, tmax = _dataset_time_bounds()
    anchor_end = _ensure_utc(range_end.value) or _ensure_utc(range_start.value) or (tmax or datetime.utcnow())
    prev_day = (anchor_end - timedelta(days=1)).date()
    prev_start = datetime.combine(prev_day, datetime.min.time())
    prev_end = datetime.combine(prev_day, time(hour=23, minute=59))
    if tmin and prev_end < tmin:
        # no data that far back; keep current window
        return
    if tmin and prev_start < tmin:
        # Clamp to first day available, ending at that day's 23:59 (or tmax if earlier)
        day_start = datetime.combine(tmin.date(), time.min)
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
        latest_end = tmax
        latest_start = tmax - DEFAULT_WINDOW
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
plot_pane = pn.pane.Plotly(config={"responsive": True}, sizing_mode="stretch_both")


@pn.depends(
    range_start.param.value,
    range_end.param.value,
    bottom_range_m.param.value,
    top_range_m.param.value,
    beta_vmin.param.value,
    beta_vmax.param.value,
    ldr_vmin.param.value,
    ldr_vmax.param.value,
    watch=True,
)
def _update_view(start, end, bottom_val, top_val, bmin, bmax, lmin, lmax):
    """Render both heatmaps for the current window and control values."""
    bottom = max(float(bottom_val), 0.0)
    top = max(float(top_val), bottom + 100.0)
    ds = open_window(start, end, bottom_m=bottom, top_m=top)
    # Simple light theme for plots
    bg = "white"
    fg = "#222222"
    grid = "#dddddd"
    if ds is None or not ds.data_vars:
        fig = go.Figure()
        fig.add_annotation(
            text="No data",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(color=fg, size=16),
        )
        fig.update_layout(height=600, paper_bgcolor=bg, plot_bgcolor=bg, margin=dict(l=40, r=40, t=40, b=40))
        plot_pane.object = fig
        return
    # Colorbar configs
    b_cmin = np.log10(bmin)
    b_cmax = np.log10(bmax)
    b_tickvals = list(range(int(np.floor(b_cmin)), int(np.ceil(b_cmax)) + 1))
    sup = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
    b_ticktext = [f"10{str(v).translate(sup)}" for v in b_tickvals]
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        shared_yaxes=False,
        vertical_spacing=0.05,
        subplot_titles=("Attenuated Backscatter", "Linear Depolarization Ratio"),
    )
    if "beta_att" in ds:
        fig.add_trace(_make_plot(ds, "beta_att", (bmin, bmax), True, coloraxis="coloraxis"), row=1, col=1)
    if "linear_depol_ratio" in ds and "beta_att" in ds:
        # Only plot depol where beta is above threshold.
        times = pd.to_datetime(ds["time"].values)
        heights = ds["range"].values
        ldr = np.array(ds["linear_depol_ratio"].transpose("range", "time"))
        beta_vals = np.array(ds["beta_att"].transpose("range", "time"))
        # Filter implausible depol values outside [0, 1].
        ldr = np.where((ldr >= 0.0) & (ldr <= 1.0), ldr, np.nan)
        mask_threshold = 10 ** -6.5  # ~3.2e-7; below this we consider depol unreliable
        ldr = np.where(beta_vals >= mask_threshold, ldr, np.nan)
        fig.add_trace(
            go.Heatmap(
                x=times,
                y=heights,
                z=ldr,
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
    # Hourly ticks; add horizontal date annotations at 12:00.
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
            if t.hour == 12:  # add a date label at local noon
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
        tickangle=-45,
        showgrid=True,
        gridcolor=grid,
        linecolor=fg,
        tickfont=dict(color=fg, size=12),
        title_font=dict(color=fg, size=12),
        row=2,
        col=1,
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
        tickangle=-45,
        showgrid=True,
        gridcolor=grid,
        linecolor=fg,
        tickfont=dict(color=fg, size=12),
        title_font=dict(color=fg, size=12),
        row=1,
        col=1,
    )
    fig.update_yaxes(showgrid=True, gridcolor=grid, linecolor=fg, tickfont=dict(color=fg, size=12), title_font=dict(color=fg, size=12), row=1, col=1)
    fig.update_yaxes(showgrid=True, gridcolor=grid, linecolor=fg, tickfont=dict(color=fg, size=12), title_font=dict(color=fg, size=12), row=2, col=1)
    # Keep both panels locked together when the user pans/zooms vertically.
    fig.update_yaxes(matches="y", row=2, col=1)
    fig.update_layout(
        height=600,
        margin=dict(l=60, r=90, t=40, b=120),
        coloraxis=dict(
            colorscale="Cividis",
            cmin=b_cmin,
            cmax=b_cmax,
            colorbar=dict(
                title="",
                x=1.04,
                y=0.77,
                len=0.35,
                tickvals=b_tickvals,
                ticktext=b_ticktext,
                tickfont=dict(color=fg, size=12),
            ),
        ),
        coloraxis2=dict(
            colorscale="Viridis",
            cmin=lmin,
            cmax=lmax,
            colorbar=dict(title="", x=1.04, y=0.27, len=0.35, tickfont=dict(color=fg, size=12)),
        ),
        paper_bgcolor=bg,
        plot_bgcolor=bg,
        font=dict(color=fg, size=13),
        annotations=tuple(list(fig.layout.annotations) + noon_annots),
    )
    plot_pane.object = fig


def _parse_relayout_time(val):
    """Parse plotly relayout timestamps safely."""
    try:
        return pd.to_datetime(val).to_pydatetime()
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
    y0 = data.get("yaxis.range[0]") or data.get("yaxis2.range[0]")
    y1 = data.get("yaxis.range[1]") or data.get("yaxis2.range[1]")
    if y0 is not None and y1 is not None and y0 > y1:
        y0, y1 = y1, y0
    if start is None and end is None and y0 is None and y1 is None:
        return
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
    beta_vmin.value,
    beta_vmax.value,
    ldr_vmin.value,
    ldr_vmax.value,
)


# -------- Calendar quicklooks --------

def _quicklook_options():
    """Build a mapping of label -> image path for available quicklook PNGs."""
    opts = {}
    # Collect dated quicklooks (sorted ascending), then append "Today" last.
    date_labels = []
    if QUICKLOOK_DIR.exists():
        for png in sorted(QUICKLOOK_DIR.glob("*.png")):
            label = png.stem.replace("ceilometer_", "")
            date_labels.append((label, str(png)))
    for label, path in date_labels:
        opts[label] = path
    if LATEST_IMAGE.exists():
        opts["Today (latest)"] = str(LATEST_IMAGE)
    return opts or {"No images available": None}


_ql_options = _quicklook_options()
ql_date = pn.widgets.Select(name="Date", options=list(_ql_options.keys()))
if _ql_options:
    ql_date.value = list(_ql_options.keys())[-1]


def _refresh_ql_options(preserve_current: bool = True):
    """Refresh available quicklook options, optionally preserving current selection."""
    global _ql_options
    current = ql_date.value if preserve_current else None
    _ql_options = _quicklook_options()
    opts = list(_ql_options.keys())
    ql_date.options = opts
    if not opts:
        ql_date.value = None
        return
    if preserve_current and current in opts:
        ql_date.value = current
    else:
        ql_date.value = opts[-1]

# Calendar navigation buttons
ql_prev = pn.widgets.Button(name="<<", button_type="default")
ql_next = pn.widgets.Button(name=">>", button_type="default")


def _shift_ql(delta: int):
    """Move calendar selection by delta steps in the refreshed options list."""
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

# Periodically refresh the "Today (latest)" selection to pick up new PNGs.
def _refresh_latest_if_needed():
    """If viewing the latest image, reload the mapping and redraw without changing selection."""
    if ql_date.value == "Today (latest)":
        # Update the cached map so _quicklook_image sees fresh file paths,
        # but do not touch the selector options to avoid snapping UI.
        global _ql_options
        _ql_options = _quicklook_options()
        ql_date.param.trigger("value")


_ql_timer = pn.state.add_periodic_callback(_refresh_latest_if_needed, period=60_000, start=True)

# Ensure initial map is fresh
_refresh_ql_options(preserve_current=True)

@pn.depends(ql_date.param.value)
def _quicklook_image(selected):
    """Show the selected quicklook image (or a message if missing)."""
    # Use the latest map in case files changed since last refresh.
    path = _quicklook_options().get(selected)
    if path and Path(path).exists():
        return pn.pane.PNG(path, sizing_mode="stretch_both")
    return pn.pane.Markdown("No image available for this selection.")


ACCENT = "#0b7285"  # header/accent color
css = """
# Global font override for a clean, consistent look.
body, .bk {
    font-family: "SF Pro Display","SF Pro","-apple-system","BlinkMacSystemFont","Segoe UI",sans-serif;
    font-size: 15px;
}
"""

# Controls card: group all widgets in a tidy stack.
controls = pn.Card(
    pn.Column(
        pn.Row(instrument_select, range_start, range_end, live_toggle, sizing_mode="stretch_width"),
        pn.Row(bottom_range_m, top_range_m, sizing_mode="stretch_width"),
        pn.Row(beta_vmin, beta_vmax, ldr_vmin, ldr_vmax, sizing_mode="stretch_width"),
        pn.Row(prev_btn, next_btn, sizing_mode="stretch_width", margin=(5, 0, 0, 0)),
        sizing_mode="stretch_width",
    ),
    title="",
    collapsible=False,
    sizing_mode="stretch_width",
)

pn.extension(raw_css=[css])

# Template layout: header + tabs (Interactive, Calendar placeholder)
template = pn.template.MaterialTemplate(
    title="AURORA Data Viewer",
    header_background=ACCENT,
    header_color="white",
)

interactive_tab = pn.Column(controls, plot_pane, sizing_mode="stretch_both")
tabs = pn.Tabs(
    ("Interactive", interactive_tab),
    (
        "Calendar",
        pn.Column(
            pn.Card(
                pn.Row(calendar_instrument, sizing_mode="stretch_width"),
                pn.Row(ql_prev, ql_date, ql_next, sizing_mode="stretch_width"),
                title="",
                collapsible=False,
                sizing_mode="stretch_width",
            ),
            _quicklook_image,
            sizing_mode="stretch_both",
        ),
    ),
    sizing_mode="stretch_both",
)

template.main[:] = [tabs]

def _apply_theme(dark: bool):
    """No-op placeholder (dark mode removed)."""
    return

# Serve the app
template.servable()
