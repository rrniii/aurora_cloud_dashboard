# Services And Timers

Systemd services are installed system-wide under `/etc/systemd/system/`.

## Dashboard

- `aurora-dashboard.service`

The deployed dashboard service runs Panel with websocket keepalives and a
short, host-specific unused-session lifetime so mobile browsers do not leave
large stale documents behind on the single-process Panel server:

- `--keep-alive=15000`
- `--check-unused-sessions=15000`
- production `--unused-session-lifetime=120000` (2 minutes)
- development `--unused-session-lifetime=60000` (1 minute)
- `--session-token-expiration=86400`

These values are inventory settings in `aurora-cloud-infra`; this page records
the current contract rather than configuring the service.

This does not stop a mobile operating system from killing a background browser
tab. The app mirrors view state into the URL so a killed tab can reload into
the same tab, instrument, and key controls after the old server-side document
has been cleaned up.

The service also exposes camera media as static routes:

- `/wxcam-media` maps to `/data/aurora/products/wxcam`
- `/auroracam-media` maps to `/project/aurora/raw/auroracam`

WXcam MP4 playback and AURORACam JPEG display use those routes so media are
fetched by the browser over normal HTTP instead of being serialized into the
Panel websocket.

## Resource Tuning

The deployed host uses systemd drop-ins to keep the interactive dashboard
responsive when product builders and infrastructure-managed archive jobs
overlap.

- `aurora-dashboard.service` gets higher CPU and IO weights plus a modest
  scheduling priority increase.
- Background processing services run in `aurora-batch.slice`, which caps the
  batch pool at two CPU cores on the current four-vCPU droplet and gives it
  lower CPU/IO weights. The slice also has a soft `MemoryHigh=6G` pressure
  limit so large products do not squeeze the dashboard as aggressively on the
  current 8 GB droplet.
- Heavier dashboard jobs such as appenders, quicklook generation, and WXcam
  daily video builds get lower priority inside that batch slice. Archive-job
  resource policy is managed by `aurora-cloud-infra`.
- Append and quicklook timers have randomized delays so they are less likely
  to start in one burst.
- The guarded runner `/usr/local/bin/aurora-run-guarded` adds lightweight
  mutexes for the heaviest job classes. Quicklook-heavy and video-heavy jobs
  run one at a time; append/rsync IO jobs allow two concurrent jobs. If a slot
  is already busy the generated `ExecStart` wrapper records a clean skip
  instead of letting heavy jobs pile up.
- Guard events are written to `/run/aurora/guarded/events.jsonl` and mirrored
  to `/data/aurora/products/ops_monitor/health/guarded_jobs.jsonl` when that
  directory is writable. The Operations Dashboard shows active guarded jobs,
  lock skips in the last 24 h, and batch slice memory pressure as diagnostic
  resource telemetry.

Install or refresh these drop-ins from the deployed checkout with:

```bash
sudo /opt/aurora-cloud-dashboard/deployment/bin/aurora-install-resource-tuning
sudo systemctl restart aurora-dashboard.service
```

Useful verification commands:

```bash
systemctl show aurora-dashboard.service -p Slice -p Nice -p CPUWeight -p IOWeight
systemctl show aurora-batch.slice -p CPUQuotaPerSecUSec -p CPUWeight -p IOWeight
systemctl show aurora-power-quicklooks.service -p Slice -p Nice -p CPUWeight -p IOWeight
systemctl cat aurora-power-quicklooks.service aurora-wxcam-daily-videos.service
systemctl list-timers --all 'aurora-*'
tail -n 20 /data/aurora/products/ops_monitor/health/guarded_jobs.jsonl
```

The dashboard restart applies its service priority immediately. Existing
background services inherit the batch slice only after their current run exits
and the timer starts them again.

Optional compressed swap can be installed separately on small droplets:

```bash
sudo /opt/aurora-cloud-dashboard/deployment/bin/aurora-install-zram
systemctl status aurora-zram-swap.service
swapon --show
```

The default zram size is `4G` with `zstd` compression and priority `100`.
Override `AURORA_ZRAM_SIZE`, `AURORA_ZRAM_ALGORITHM`, or
`AURORA_ZRAM_PRIORITY` in the service environment before starting it if the VM
size changes.

## CL61

- `aurora-cl61-source-sync.timer`
- `aurora-ceilometer-append.timer`
- `aurora-ceilometer-last24h.timer`
- `aurora-ceilometer-quicklooks.timer`

## Cloud Radar

- `aurora-radar-source-sync.timer`
- `aurora-radar-append.timer`
- `aurora-radar-quicklooks.timer`
- `aurora-radar-daily-quicklooks.timer`

`aurora-radar-quicklooks.service` is overridden on the deployed host to update
only the rolling latest 24 h radar PNG. The heavier daily archive generator
runs from `aurora-radar-daily-quicklooks.timer` instead, so a frequent
quicklook refresh cannot spend several minutes backfilling daily radar products
while operators are using the interactive dashboard.

Radar PNG rendering also uses display-only thinning controlled by
`AURORA_RADAR_QUICKLOOK_MAX_TIME_SAMPLES` and
`AURORA_RADAR_QUICKLOOK_MAX_RANGE_SAMPLES`. This reduces memory use for static
quicklooks without changing the underlying radar Zarr.

## Scanning Microwave Radiometer

- `aurora-hatpro-source-sync.timer`
- `aurora-hatpro-append.timer`
- `aurora-hatpro-quicklooks.timer`

## Meteorology (VaisalaMET)

- `aurora-vaisalamet-source-sync.timer`
- `aurora-vaisalamet-append.timer`
- `aurora-vaisalamet-quicklooks.timer`

## ASFS Logger

- `aurora-asfs-logger-source-sync.timer`
- `aurora-asfs-logger-append.timer`
- `aurora-asfs-logger-quicklooks.timer`

## ASFS Fast Sonic

- `aurora-asfs-fast-sonic-source-sync.timer`
- `aurora-asfs-fast-sonic-append.timer`
- `aurora-asfs-fast-sonic-quicklooks.timer`

## ASFS Fast Gas

- `aurora-asfs-fast-gas-source-sync.timer`
- `aurora-asfs-fast-gas-append.timer`

Fast-gas is the high-rate LI-COR/gas file family from the ASFS logger CRD
area. It is stored in its own Zarr and is merged into `HK_ASFS` housekeeping
quicklooks for LI-COR continuity. It does not contain radiation variables.

## Power

- `aurora-power-source-sync.timer`
- `aurora-power-append.timer`
- `aurora-power-soc-forecast.timer`
- `aurora-power-soc-forecast-learn.timer`
- `aurora-power-soc-ensemble.timer`
- `aurora-power-soc-planning-forecast.timer`
- `aurora-power-operating-scenarios.timer`
- `aurora-power-quicklooks.timer`

`aurora-power-quicklooks.service` regenerates the compact APS display summary
and prewarmed Plotly JSON after the APS append cycle.
`aurora-power-soc-forecast.service` refreshes the ECMWF-informed SOC forecast
and adaptive forecast-skill state from a new ECMWF download every 3 hours.
`aurora-power-soc-forecast-learn.service` runs on a 15-minute timer, reuses the
latest cached ECMWF forecast, re-anchors to current SOC, scores archived
forecast runs, and updates skill/adaptive state faster than ECMWF is refreshed.
`aurora-power-soc-ensemble.service` checks hourly for a new ECMWF 00/12 UTC
ensemble cycle. New cycles retrieve all 50 perturbed `ssrd` members, write the
compact site ensemble and probabilistic verification products, then remove the
temporary global GRIB. It uses idle I/O scheduling and a two-hour timeout so it
does not block deterministic SOC learning.
`aurora-power-soc-planning-forecast.service` retrieves the eligible ECMWF 00/12
UTC deterministic cycle twice daily and writes a 240-hour forecast under
`/data/aurora/dev-products/power`.
`aurora-power-operating-scenarios.service` runs every five minutes. It learns
new mode/component evidence and regenerates named and optimized plans from
current SOC. It also aligns the mirrored UAS MQTT log so tier-specific loads
can be learned and the all-instruments/UAS-tier-3 scenario can be evaluated.
The app merges that compact product into the display summary at read time; it
does not rebuild the full Power summary every five minutes. Both environments
run these advisory products: production writes under `/data/aurora/products`,
while development writes independently under `/data/aurora/dev-products`.
On development only, the same run may publish a compact CL61
`observe_only` shadow status and append-only history. It has no PDU client and
does not create an actuator service; the separately managed ASS-local
controller remains disabled.

The deterministic and scenario jobs use semantic publication signatures.
When a timer run has the same physical SOC/load anchor, mode, ECMWF cycle,
solar contract, battery parameters, and model version, it advances state and
health without rewriting the public Zarr or duplicating an archive issue.

## ASS PDU

- `aurora-pdu-source-sync.timer`
- `aurora-pdu-append.timer`

The source sync pulls ASS Linux `/home/aurora/data/pdu/pdu_DDMMYYYY.csv` files
into `/project/aurora/raw/pdu`. The appender writes
`/data/aurora/products/power/pdu.zarr`, which is folded into the APS display
summary by the Power quicklook pipeline when available.

## WXcam

- `aurora-wxcam-source-sync.timer`
- `aurora-wxcam-catalog.timer`
- `aurora-wxcam-daily-videos.timer`
- `aurora-wxcam-append.timer`

## AURORACam

- `aurora-auroracam-source-sync.timer`
- `aurora-auroracam-index.timer`

## Operations

- `aurora-ops-monitor-collect.timer`
- `aurora-ops-monitor-append.timer`
- `aurora-ops-monitor-alerts.timer`
- `aurora-ops-monitor-quicklooks.timer`

`aurora-ops-monitor-collect.timer` is observe-only. It writes raw JSONL
snapshots under `/project/aurora/raw/ops_monitor` and compact health outputs
under `/data/aurora/products/ops_monitor/health`; it does not restart services,
delete files, rebuild data products, or change code.
It reads `/data/aurora/internal/archive_status/health-v1.json` as a versioned
contract. It does not SSH to JASMIN, inspect verifier manifests, or decide
whether source data may be pruned.

`aurora-ops-monitor-alerts.timer` evaluates the latest operations snapshot
after collection and sends threshold email alerts through `mailx` backed by an
outbound relay such as `msmtp`.

## Archive services

GWS and object-store writers, verification, archive health, and retention are
owned and documented by `aurora-cloud-infra`. The dashboard repository owns
only their read-only presentation. Use that repository's
`docs/ARCHIVE_SERVICES.md` for what is backed up, service names, settle windows,
pending-upload semantics, evidence paths, repair, and retention policy.

## Useful commands

```bash
sudo systemctl status aurora-dashboard.service
sudo systemctl list-timers --all | rg '^.*aurora-'
sudo journalctl -u aurora-dashboard.service -f
```

## Manual Product Regeneration

The deployed systemd services load `/etc/aurora-dashboard.env` before running
appenders and quicklook generators. That environment file points products at
the live dashboard tree, including:

- `AURORA_QUICKLOOK_ROOT=/data/aurora/products/quicklooks`
- `AURORA_INTERACTIVE_PREWARM_DIR=/data/aurora/products/dashboard/prewarm`

When running a generator manually on the deployed host, source that environment
first so the output lands where the dashboard reads it:

```bash
cd /opt/aurora-cloud-dashboard
set -a
source /etc/aurora-dashboard.env
set +a
source venv/bin/activate
./generate_asfs_logger_quicklooks.py
```

Without the environment file, several generators intentionally fall back to a
repo-local `quicklooks/` directory for development. That is useful for local
tests, but it does not update the live dashboard quicklook shown in the web
app.
