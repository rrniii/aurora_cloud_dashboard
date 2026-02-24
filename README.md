# Aurora Cloud Dashboard

Lightweight Panel/Plotly dashboard to view ceilometer and cloud‑radar data from Zarr stores, plus helper scripts to build/append Zarr, generate quicklooks, and plot status.

## What's here
- `app.py` — Panel app with tabs for **Interactive** plots and **Calendar** quicklooks. Supports instrument switcher (Ceilometer / Cloud Radar), time window, range limits, color limits, day navigation, and live 24 h mode.
- Radar/ceilometer utilities:
  - `netcdf_to_zarr.py`, `append_new_netcdf_to_zarr.py` — build/append ceilometer Zarr from NetCDF.
  - `cloud_radar_to_zarr.py`, `append_new_cloud_radar_to_zarr.py` — build/append radar Zarr (stitches C1+C2 along range).
  - `plot_cloud_radar_last24h.py`, `plot_ceilometer_zarr_last24h.py` — produce last‑24h PNGs for the dashboard.
  - `generate_cloud_radar_quicklooks.py`, `generate_daily_quicklooks.py` — daily PNGs for calendar tab.
  - `consolidate_zarr_metadata.py`, `plot_status_timeseries.py`.
- Service units (user systemd) in `~/.config/systemd/user/`:
  - `ceilometer.service` — runs `panel serve app.py`.
  - `cloud_radar_append.service` + `.timer` — append new radar NetCDF files into the Zarr.
  - `cloud_radar_quicklooks.service` + `.timer` — refresh radar quicklook PNGs and last‑24h image.

## Data locations
- Ceilometer Zarr (default): `/mnt/data/cl61/gamb2le_depolarisation_lidar_ceilometer_aurora_20251201.zarr`
- Radar Zarr (default): `/mnt/data/ass/rpgfmcw94/cloud_radar.zarr`
- Quicklooks: `quicklooks/ceilometer/`, `quicklooks/cloud_radar/`
- Latest 24h PNGs: `last24h.png`, `last24h_cloudradar.png`

You can override Zarr paths with env vars: `CEILOMETER_ZARR_PATH`, `CLOUD_RADAR_ZARR_PATH`.

## Running locally
```bash
cd /home/aurora/aurora_cloud_dashboard
source aurora_cloud/bin/activate
panel serve app.py --address 0.0.0.0 --port 5006 --allow-websocket-origin=<host>
```

## Managing services (run as user `aurora`)
- Status: `systemctl --user status ceilometer.service`
- Restart dashboard: `systemctl --user restart ceilometer.service`
- Append radar files now: `systemctl --user start cloud_radar_append.service`
- Regenerate radar quicklooks now: `systemctl --user start cloud_radar_quicklooks.service`
- View logs: `journalctl --user -u <service> -f`

## Typical workflows
- Build radar Zarr from raw LV1 files (initial): `./cloud_radar_to_zarr.py --start 2026-02-23T13:00:00Z`
- Append latest radar files (manual): `./append_new_cloud_radar_to_zarr.py`
- Regenerate last‑24h radar PNG: `./plot_cloud_radar_last24h.py /mnt/data/ass/rpgfmcw94/cloud_radar.zarr last24h_cloudradar.png`

## Notes
- Radar variables stitched: ZE_dBZ, ZE45_dBZ, MeanVel, ZDR, SRCX, SpecWidth, SLDR, Skew, RHV, PhiDP, Kurt, KDP, DiffAtt (C1+C2 along range).
- Interactive plot defaults: 9 km range cap, configurable color limits per variable; plot height scales with viewport on desktop and is constrained on mobile.
- Calendar quicklooks use the same panels/limits as the radar 24 h plot.
