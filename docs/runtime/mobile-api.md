# Mobile API

The native iOS app uses a small read-only API instead of scraping the Panel
dashboard or reading Zarr/SQLite files directly from the phone.

The server-side API is released with the dashboard repository. The native
Swift client is maintained in the separate `aurora-dashboard-ios` repository,
so API and app changes can be reviewed and released independently. Keep the
two releases compatible through the tested `/mobile/v1` contract rather than
copying native code into this repository.

## Service

Install the Python dependencies into the dashboard virtual environment:

```bash
pip install -r requirements-mobile-api.txt
```

The API is served by `mobile_api.py`:

```bash
uvicorn mobile_api:app --host 127.0.0.1 --port 8010
```

Ansible installs the service and proxies it under both public dashboard
hostnames. It also creates a root-owned token file at
`/etc/aurora-mobile-api.token`, referenced through
`AURORA_MOBILE_API_TOKEN_FILE`. The token is never committed to the dashboard
or infrastructure repository.

```text
https://data.gamb2le.co.uk/mobile/v1
```

## Authentication

The service fails closed by default and requires:

```text
Authorization: Bearer <AURORA_MOBILE_API_TOKEN>
```

`GET /health` is intentionally unauthenticated so the iOS app can report
whether the service is reachable. Both current public dashboard hosts
explicitly use `AURORA_MOBILE_API_AUTH_MODE=public_read_only`; normal read-only
app use therefore does not require a token. That mode allows only `GET` and
`HEAD` requests to the bounded app payloads and media already displayed on the
public dashboard. The administrative derived-artifact inventory remains
token-protected, unknown mode values fail closed, and the legacy
`AURORA_MOBILE_API_ALLOW_PUBLIC` variable has no effect.

## Endpoints

- `GET /health` - service reachability and auth configuration.
- `GET /manifest` - deployment identity, tabs, instruments, WXcam streams,
  refresh defaults, and the cross-platform capability contract. The contract
  distinguishes shared live capabilities from browser-only scientific
  exploration and native-only resilience workflows. Deployment identity
  reports the current site environment, public domain, data role, dashboard
  URL, and checked-out revision.
- `GET /operations` - latest operations health, stream states, root-cause
  groups, active alerts, and compact seven-day context cards. Current APS and
  storage values come from the latest snapshot; source and GWS lag fall back to
  a cached, bounded read of the operations Zarr. Lag calculations exclude
  instruments with a fresh intentional PDU-off state.
- `GET /overview` - the small first-load status cards and active alerts. When
  the UAS PDU outlet is on and the mirrored Menapia record is fresh, its state
  includes the latest effective tier, for example `On (Tier 3)`. PDU `Off` and
  `Unknown` remain authoritative, and stale tier records are not displayed as
  current state. A powered CL61, Cloud Radar, or HATPRO reports `Collecting`
  only when its latest science-product sample is also fresh; power without
  fresh acquisition evidence remains `On`. The established collection windows
  are 90 minutes for CL61 and radar, and 180 minutes for hourly-batch HATPRO.
- `GET /power?window=24h|96h&group=...` - bounded native-chart traces from the
  section-specific Power display product when available (at most 260 points
  per trace). `current` reads observed variables only; `forecast` reads
  forecast, planning, and verification variables only. The legacy combined
  display product remains a fallback.
  `group=current` returns observed conditions; `group=forecast` returns the
  24-hour forecast, 96-hour forecast/planning, and verification panels. Each
  forecast panel includes an `info` object with its summary,
  implementation-specific assumptions, and metric definitions for the native
  per-plot Info sheet. Verification panels also retain their live `guidance`
  evidence and score cards. Decision panels also include a `forecastContext`
  with the physical SOC anchor, generation time, common decision endpoint, and
  `horizonHours`. The SOC 96 h and instrument-scenario cards use the same
  re-anchored system-as-is values and stop at `anchorTime + 96 h`; the longer
  internal reserve tail is not returned as a 96-hour decision value. The SOC
  next 24 h card is the first 24 hours of that same central system-as-is trace,
  with the same SOC anchor and load/state assumptions; it is not a separate
  deterministic forecast. The ECMWF solar/load input card is bounded to the
  same 96-hour planning horizon. Dimensionless probability fields that are
  scaled from fractions to percentages are returned with unit `%`, so clients
  do not have to infer display units. The
  legacy `all`, `observed`, `forecast_24h`, `forecast_96h`, and `verification`
  groups remain supported.
- `GET /media/power/figure/current|forecast` - an ETag-backed prewarmed Plotly
  JSON figure for a Power section. Access follows the host authentication mode.
  The response never exposes raw or Zarr paths and can be cached by native
  clients for one minute.
- `GET /auroracam?day=latest|YYYY-MM-DD&time_utc=...` - the latest four
  AURORACam records with separate preview and original URLs. Historical days
  also provide a bounded list of UTC frame times for native selection.
- `GET /uas?window=24h|7d|all` - latest UAS tier and a bounded history for
  the selected server-side window. `24h` is the default. `all` means all
  available records up to the newest 2,000, protecting mobile clients from an
  unbounded response.
- `GET /uas/flights?day=latest|YYYY-MM-DD` - path-free Menapia flight metadata
  for one UTC day, newest day/flight defaults, product freshness and quality,
  and archived-day choices. `allFlightsPlotURL` points to the daily composite
  Science Quicklook, while each flight has a same-origin `plotURL`. Both carry
  deterministic file-version queries so clients refresh when regenerated
  bytes replace a stable dated URL.
- `GET /uas/flights/{id}` - one catalog-listed flight and its equal-length,
  bounded one-second profile arrays: UTC time, SN0122/SN0123 temperature,
  pressure and relative humidity, plus fused altitude. Unknown or malformed
  IDs fail closed and product filesystem paths are never returned.
- `GET /media/uas/flights/{id}` - the ETag-backed static plot for one
  catalog-listed flight. Under the public proxy this is
  `/mobile/v1/media/uas/flights/{id}`.
- `GET /instruments/{id}/summary?window=24h|7d` - mobile instrument summary and
  latest generated quicklook references.
- `GET /quicklooks?kind=science|housekeeping&instrument={id}` - available
  quicklook dates and image URLs. For an assigned PDU instrument, the response
  also includes its current power state. Clients present an intentional
  power-off as an expected collection pause instead of a missing-data fault.
  Science listings for Meteorology, Radiation, Aurora Power Supply, and
  Operations use their explicit `__summary__latest.png` and
  `__summary__YYYYMMDD.png` products. The unqualified `latest.png` and legacy
  dated filenames for grouped instruments may contain housekeeping; they are
  never used as Science fallbacks. A missing summary remains unavailable,
  while an available dated summary can supply the latest entry. Housekeeping
  listings continue to use their separate `__hk_*__` products. Ceilometer,
  Cloud Radar, HATPRO, and WXcam retain their legacy science latest aliases.
  Quicklook image URLs include a file revision as the final path component,
  so native image caches refresh when a product is replaced or its selected
  source changes. Existing unversioned quicklook media URLs remain supported.
- `GET /wxcam?stream=fish_hdr|pano_hdr&day=latest|YYYY-MM-DD` - stitched MP4,
  day list, poster, and hourly thumbnails.
- `GET /media/...` - image/video responses with short cache headers and
  ETag/304 revalidation. Access follows the host authentication mode.
  AURORACam previews are generated only on demand, are capped at 960 pixels,
  and use a bounded 50 MB server cache.

The API reads existing deployed products only. It does not restart services,
write Zarr stores, mutate the WXcam catalog, or change Panel dashboard behavior.
The deployment fields in the manifest are derived from the existing service
environment and Git checkout; they do not create or refresh a data product.

Menapia consumers use `MENAPIA_PRODUCT_ROOT` (default
`/data/aurora/products/menapia`), optional `MENAPIA_CATALOG_PATH`, and
`UAS_QUICKLOOK_DIR` (default `/data/aurora/products/quicklooks/uas`). The
optional `MENAPIA_PRODUCT_STATUS_PATH` heartbeat defaults to
`/data/aurora/internal/menapia-products/status.json`; it records every builder
attempt so an unchanged, archive-stable catalog is not incorrectly reported as
stale. Missing or malformed heartbeat state falls back to `generatedAt` in the
catalog for compatibility with older deployments.
