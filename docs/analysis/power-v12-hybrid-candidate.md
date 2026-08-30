# Evaluation-first v12 hybrid power candidate

Status: development-only candidate.  It is not an operational forecast and it
does not replace the v11 development product or any production product.

## Purpose

v12 evaluates two bounded, issue-time-safe changes independently and together:

| Lane | Solar forcing | Load forcing |
| --- | --- | --- |
| B | Three-array physical PV model | Existing finite-state load profile |
| C | Existing solar model | Bounded load-residual correction |
| D | Three-array physical PV model | Bounded load-residual correction |

The physical PV lane resolves East, South and West arrays at ten-minute
substeps, conserves each ECMWF source-interval irradiance energy, and integrates
battery acceptance on that substep trace.  Its array configuration is explicitly
provisional until a surveyed bill of materials and uncurtailed MPPT evidence are
available.

The load lane fits a regularised residual only from forecast issues and observed
load available before the candidate issue time **and from the same frozen
baseline forecast contract**.  Rows that lack a complete control identity are
not eligible to train it. Candidate execution reads only the issue-time-safe
21-day APS/PDU history needed by the calibration and verification windows; it
never materialises the whole mirrored power store. It needs at least 48 valid
samples, three independent cycles and three UTC days; otherwise it emits a zero
correction with an `insufficient_issue_time_evidence` status.  Corrections are
shrunk and clipped to 500 W, and cannot reduce the forecast below a measured
or state-registry DC-only core-load floor.  This keeps a residual from erasing
the station's always-on electrical demand while preserving the finite-state
load model.

## Isolation and provenance

Each v12 lane writes only below:

```text
/data/aurora/dev-products/power/candidates/v12/lanes/<lane>/
```

The baseline latest forecast, archive, adaptive state, skill product and
hindcast remain read-only inputs.  A run accepts only an archive-eligible full
ECMWF-cycle baseline, then records the exact SOC anchor, cycle, forcing checksum,
physical configuration digest, source manifest, code revision and forecast
identity.  It fails closed if the baseline pairing, configured input checksum,
or protected path relationship is invalid.

Every source manifest now also contains a bounded `issue_time_features`
snapshot: APS/battery-bus values, full PDU context, ASFS SR30/SPN1 irradiance,
meteorology and 48-V diagnostics, and the raw two-dock Menapia tier tail. The
snapshot is cut at the issue time, hashed, and carries explicit source age and
availability. The six completed Menapia recovery episodes remain a documented
0.20-kWh planning prior; delayed flight products are not forecast predictors.

Cloudnet, HATPRO, radar, ceilometer and camera products are explicitly marked
retrospective or latency-replay-only. They are never read by the v12 forecast
writer as if they were live observations.

ECMWF remains the only active weather forcing in the B/C/D candidate. IFS, GFS,
ICON and GEFS each have a separate, fail-closed site-manifest slot under
`/data/aurora/dev-products/power/public_model_inputs`. A version-2 manifest
must identify an immutable local Zarr/NetCDF extract, its SHA-256, exact site
coordinates, source cycle, real delivery time, irradiance variable, and an
explicit `global_grid_retained: false` declaration. The consumer re-hashes it,
rejects any spatial dimension, requires the exact paired output grid, and
writes a separate physical-PV candidate under
`public_model_ablations/<source>/`. GEFS may use only its own on-time local
members (an explicitly labelled internal-member mean); sources are never pooled
with one another or with ECMWF, and nothing is imported from
`/data/aurora/model-evaluation`.

Completed baseline/candidate pairs are immutable directories at:

```text
pairs/<pair-id>/<candidate-publication-signature>/
```

Only a manifest with `pair_status: complete` is valid evaluation input.  The
runner additionally writes an append-only hash-chained evaluation history,
`campaign_evidence.zarr`, the rolling `daily_diagnostic_skill.zarr`, a
`review_summary.json`, and a `not_accepted` acceptance record.

Campaign evidence is partitioned by the complete semantic candidate contract:
candidate system/version/features/code, lane, local-feature interface and the
frozen baseline-control contract. Older pair bundles are retained as immutable
history but are excluded from a new contract's cumulative score surface.

## Evaluation rule

Rolling diagnostics are health signals, not promotion evidence.  The required
promotion surface is cumulative paired campaign evidence, stratified by lead,
load state, source availability, degraded mode and an issue-time-safe cloud
proxy.  A candidate is retained until it has at least 30 independent paired
cycles in every lead bucket across at least ten UTC days and has passed the
specified SOC, solar, load, calibration, reproducibility, resource and API
gates.  There is no automatic promotion path in v12.

Direct solar skill is calculated only for records where all three Victron MPP
mode-791 channels report `2` (MPPT active). A limited, off, missing, or stale
channel makes that delivered-power sample censored and excludes it; it never
appears as a physical-PV model miss. Until enough such records arrive, the
solar promotion gate stays explicitly blocked.

The baseline ECMWF site ensemble is also read-only. Every B/C/D lane propagates
the same members through its candidate calculation: B/D use ten-minute
physical-PV battery integration; C/D preserve the baseline member load spread
and apply only the deterministic bounded-residual delta. Candidate ensembles,
archives and pairs have their own semantic contract directory, content
signature and immutable `ensemble_pairs/<pair-id>/<signature>/` bundle. The
campaign product calculates CRPS against both v10/v11 baseline members and SOC
persistence, P10--P90 coverage, and reserve-event Brier skill. Coverage must
be 75--90%; reserve evidence with fewer than ten events is reported as
`insufficient_events`, never invented or silently passed.

`promotion_gates` in the candidate status/review record materialises the
quantitative 0--24 h, long-lead, load, solar, ensemble and reserve-event rules.
It is a manual review aid only: insufficient MPP-active solar data, an absent
or non-identical member-wise baseline ensemble, or insufficient independent
cycles are explicit blocks and cannot produce an automatic promotion.

## Development and iOS exposure

`aurora-power-v12-candidate.timer` is development-only and, after a successful
manual canary, enabled as a resource-limited evidence collector: 25% CPU, 1 GiB
soft memory and 1.5 GiB hard memory. It defers when the wider AURORA
model-evaluation service is active.  The mobile API exposes an additive,
read-only `/power/solar-evaluation` endpoint only when the development
candidate feature is enabled.  The iOS Development scheme is pinned to
`data-ocean` and renders it as “Development candidate — not operational”; the
operational `/power` schema and production app remain unchanged.
