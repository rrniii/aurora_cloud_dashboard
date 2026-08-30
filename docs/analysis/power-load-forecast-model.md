# APS load forecast model

## ECMWF data provider

The deterministic solar forecast has a provider boundary controlled by
`AURORA_ECMWF_PROVIDER` (or `--provider`):

- `legacy` uses `ecmwf-opendata` for retrieval and `cfgrib` for decoding.
- `earthkit` uses `earthkit-data` for retrieval and decoding, with an automatic
  legacy fallback that preserves a usable forecast.
- `shadow` publishes the legacy result and compares Earthkit against the same
  input file. The comparison is written to
  `/data/aurora/products/power/ecmwf_provider_shadow.json` by default. Each
  shadow run also appends an immutable JSONL observation. The development
  promotion gate requires seven days, at least 50 successful comparisons,
  matching valid times, and SSRD differences below `0.001 J m-2` before it can
  report `eligible_for_operator_review`. It never changes the active provider.

Both providers are normalized to the same site-level xarray contract before
the calibrated solar, load, and SOC model runs. Forecast Zarr variables and
paths therefore remain unchanged. Provider, cycle, selected grid point,
timings, parity, and fallback details are stored as forecast metadata and
propagated into the display summary metadata.

The 50-member ensemble continues to use direct ecCodes nearest-point streaming.
This avoids allocating the complete global member-by-step grid; it should only
move to Earthkit after an equivalent bounded-memory benchmark passes.

## Decision

The operational load model is `finite_controlled_state_phases_v10`. Its
observed target is the APS energy balance:

`load = SolarWatts_East + SolarWatts_South + SolarWatts_West - BatteryWatts`

APS `BatteryWatts` is positive during charging and negative during discharge.
This balance therefore measures total system consumption, including the 48 V
DC load. `ACOutputWatts + DCInverterWatts` is not used as the primary target:
in `DC-Only` it reports about `9 W` of inverter idle while the battery balance
shows about `220-226 W` of real consumption.

The **system-as-is** forecast first identifies the latest confirmed finite
operating state. That state is held until an explicit schedule changes it. The
load can change only through a detected phase learned inside that exact state:
startup, steady operation, or a low/high fan regime. It never extrapolates an
instrument switch from the clock, invents an intermediate configuration, or
borrows a load trajectory from another operating state.

For CL61, the high automatic phase represents its heater/blower. Operators
control whether CL61 has PDU power, but the instrument controls this subphase.
The dashboard therefore keeps the finite PDU state as `DC + CL61`, learns the
low/high load distributions inside that state, and reports `On with
Heater/Blower` while the fresh learned high phase is active. The scenario
product exposes `CL61` and `CL61 (heater on)` as separate forecast states, but
the heater remains an instrument-controlled phase rather than a separate PDU
command.

The load estimate follows a conservative priority order: fresh PDU component
power plus a clean DC-only baseline, a clean DC-only observation when no kit is
active, a mature distribution learned for that exact state, then the latest
whole-station balance only as a bootstrap for a newly seen state. The measured
whole-station value and its disagreement with the state estimate remain in the
metadata as diagnostics. Recent material changes are evaluated only after the
PDU state is fixed. Repeated episodes then supply phase-specific P10, median,
and P90 load, startup-duration quantiles, fan-phase occupancy, and fan-phase
dwell time. A startup or fan level must recur at least twice before it is
forecast as uncertainty: startup needs two state-entry episodes, while a fan
level needs two non-contiguous segments. A one-off historical level is
excluded, and the latest confirmed steady level is retained. These rules
prevent an old configuration from silently becoming an uncontrolled
historical-load trajectory. Detection does not wait for recurrence: a new
sustained latest level immediately becomes the current steady forecast, while
its repeatable phase uncertainty remains unlearned until the second occurrence.

The scenario learner is `hybrid_canonical_uas_cl61_states_v11`. Its discrete
part is an HMM-like finite-state classifier. Fresh PDU outlet watts provide direct
evidence for CL61, Radar, HATPRO, UAS, and their combinations; the APS AC
output and learned total-load level provide secondary evidence. Stale PDU data
cannot assert that a kit remains on. A transition prior prevents single noisy
observations from making the classification oscillate, while the posterior
probability supplies a dashboard confidence value.

The continuous part combines a robust Kalman learner over additive components
with an exact-state phase learner. The components are DC, CL61, Radar, HATPRO,
UAS, and Unknown AC. A whole-station power-balance observation can train a
controlled state only when UAS, CL61, Radar, and HATPRO each have direct PDU
evidence at that timestamp. Direct outlet watts can still constrain their own
instrument component. This prevents a high-load interval with a partial or
missing PDU vector from becoming a false `DC-Only` fan regime. Innovation
clipping prevents a transition spike or bad sample from moving the learned
level arbitrarily. Within each exact state, robust change detection requires at
least `20 W`, `8%` of load, and three robust noise scales before it accepts a
phase boundary. A phase needs at least four samples; an exact state needs at
least eight samples before its phase profile is used.

Recognition and durable learning remain separate. The latest state can change
as soon as fresh PDU and APS data identify it, but saved component parameters
advance only for timestamps newer than the previous training cursor. This makes
the five-minute process incremental and idempotent. Zero-solar battery
discharge with a complete PDU state vector remains the strongest evidence for
the DC component because it measures the complete battery-side load. The
learner uses the directly confirmed PDU mode, rather than the smoothed
classifier state, for that update. It does not shift the DC component to make
all components add up to a fixed target; that earlier adjustment could create
an impossible zero-watt DC baseline.

The UAS MQTT log is aligned to the same 15-minute operating-state timeline.
The canonical UAS states are Tier 1, Tier 1 + UAS Charging, Tier 2, Tier 2 +
UAS Charging, Tier 3, Tier 3 + UAS Charging, Tier 4 (12 V only), and Tier 5
(all off). Field Tier 11 is the only learning source for canonical Tier 1, and
field Tier 12 is the only learning source for canonical Tier 2; the raw value
is retained alongside the canonical value for audit. Proxy tiers become
reliable after two independent episodes and six observed hours. Direct tiers
require three episodes and six hours.

Charging is never inferred from an unexplained increase in watts. Until an
operator-annotated `UASCharge on`/`off` episode is available, Tiers 1-3 add an
estimated `300 W` for exactly three hours and then return to the base tier.
After at least one complete 2.5-hour annotated episode, the learner replaces
both the increment distribution and median duration with observed charge data.
Provisional base tiers continue to use the
documented conservative fallback distribution: P10 `55 W`, P50 `108 W`, P90
`302 W`.

SOC integration uses a calibrated battery model fitted only to usable,
non-saturated, stable telemetry. It estimates usable capacity, charge and
discharge efficiency, and observed charge/discharge power limits. Capacity is
bounded to `15-35 kWh`; efficiencies are validated before publication. The
separate parasitic term is currently zero because the measured whole-station
balance already contains system losses. Estimating another parasitic load from
the same signal would double count it.

The actionable 96-hour decision forecast preserves the current ensemble's
versioned solar-calibration contract. The older deterministic planning cycle is
used only to extend the reserve tail after native ensemble coverage ends. When
the two contracts differ, both IDs and the transition are recorded explicitly;
the current ensemble is not recalibrated with an older planning snapshot. The
ensemble varies ECMWF weather, calibrated battery
parameters, and only the learned uncertainty within the detected operating
state. If the state has a repeatable startup, members sample its learned
duration and phase-specific load. If it has low/high fan regimes, members
switch between only those learned regimes using their observed occupancy and
dwell time. The P10-P90 interval is therefore not solar-only, but it cannot
imply an unrequested instrument-state transition or use another state's load.

Future operator choices are represented explicitly instead of guessed. Named
plans include current mode, DC-Only, DC + CL61 continuously on, an optimized
CL61 schedule, a custom CL61 start/duration, and any other kit combinations the
model has learned. The fixed comparison set also includes CL61, CL61 + Radar,
CL61 + HATPRO, CL61 + HATPRO + Radar, HATPRO + Radar, Radar, HATPRO, and **all
instruments + UAS tier 3**. The old `100-600 W` plot is retained only as a
backwards-compatible data contract and is no longer the operating interface.
The dashboard derives canonical `SystemAsIsDecision*` traces from the freshly
re-anchored current-mode scenario. The SOC 96 h panel and the current line in
the scenario panel therefore share one SOC anchor, issue, weather basis, load
distribution, and endpoint. The standalone ensemble remains the explicit
fallback when a current operating-scenario product is unavailable.

## Evidence

The diagnostic run on 2026-07-16 used
`/data/aurora/products/power/power.zarr` and 15-minute median samples. In the
current `DC-Only` run, AC output was `0 W`, inverter idle was about `8.9 W`, and
55 zero-solar 15-minute battery-discharge samples gave a `223 W` baseline. The
independently measured ASFS 48 V trace was about `157 W`; it is the main 48 V
branch, while the battery-side balance also includes the rest of the DC system
and conversion losses.

Later on 2026-07-16, the Ceilometer transition supplied the first named AC-mode
case. Fresh PDU outlet 5 data reported `223 W`, median AC output was about
`174 W`, and the APS power balance measured about `455 W` total station load.
The prior `DC-Only` baseline was independently recovered as `223.7 W` from 59
zero-solar 15-minute samples (`220.0-225.5 W` 10th-90th percentile). The mode
therefore resolves as `DC-Only + CL61`, with outlet 5 as its primary signature.

A rolling-origin comparison used issue times every six hours during the latest
seven-day verification interval, up to 21 days of history per issue, and lead
times from 3 through 24 hours. The target for every model was power-balance
total load. Errors are in watts across 184 forecast-observation pairs.

| Model | MAE | Bias | RMSE |
|---|---:|---:|---:|
| UTC-hour median | 330.28 | 126.74 | 353.16 |
| Raw last-value persistence | 135.35 | 14.72 | 279.15 |
| `kit_mode_persistence_v2` | 131.53 | 15.62 | 275.97 |
| `kit_mode_persistence_v3` | 131.28 | -33.53 | 232.73 |

This table evaluates the version-3 predecessor and is retained as historical
context. On this limited backtest,
version 3 reduced RMSE by about 16% relative to
version 2 and slightly reduced MAE, but its negative bias shows that a single
`DC-Only` baseline cannot represent unlabelled higher-load operation. These are
historical results, not a guarantee of future skill. Named PDU kit modes
separate those periods as they are observed.

A 14-day phase audit on 2026-08-02 found repeatable load changes inside exact
states. Three CL61 starts initially used about `459-471 W` before settling near
`275-278 W`. CL61 + HATPRO started near `667 W` before settling around
`477-508 W`; CL61 + Radar + HATPRO started near `968 W` before settling around
`732-787 W`; and all instruments including UAS started near `1070 W` before
settling around `810-893 W`. These observations justify startup and fan phases,
but not a free-running time-of-day load forecast.

On 2026-08-03, outlet 5 supplied about `38 W` from 20:00 through 02:45 UTC and
then stepped to about `223 W` at 03:00 UTC while every controllable instrument
state remained unchanged. The exact-state learner detected `fan_high`; this is
the direct operational example used for the CL61 heater/blower presentation.

## Learning, planning, and verification

The operating-state job runs every five minutes, re-anchors SOC to the latest
actual observation, and consumes only newly arrived samples for parameter
learning. A separate planning job retrieves the latest eligible ECMWF 00 or 12
UTC deterministic cycle twice daily and forecasts 240 hours. Its native cycle
is reused by the faster state job, so a kit transition changes the load and SOC
plans without redownloading ECMWF.

The suggested-scenario `Current load / system as-is` trace always uses the
fresh operating-state job's exact state and automatic phase. It reuses the
planning product's solar cycle, not an older planning-cycle load trace. Thus,
when CL61 is the only powered instrument, the current trace and the fixed CL61
trace start from the same detected heater/blower phase.

The advisory scheduler reserves a feasible CL61 timetable first, then adds
Radar and HATPRO only from residual reserve. This makes the priority explicit
rather than trading a CL61 hour for a lower-priority instrument. The DC baseline
and current UAS state remain fixed; an already-on CL61, Radar, or HATPRO is also
held at its observed PDU state so the plan cannot assume that another outlet
has changed. Each newly proposed interval has a minimum 12-hour run and at most
one planned start per UTC day. Every candidate must keep P10 SOC at or above
40% across the full 240-hour horizon. Learned startup, fan, and heater/blower
phases are checked again before publication. The scheduler is advisory only and
never issues PDU commands; the optional development automation product is a
non-executable shadow receipt. The existing custom CL61 start/duration editor
remains an independent what-if calculation.

Archived deterministic forecasts carry `LoadModelVersion`. Load MAE, bias, and
skill only use rows from a matching model version, preventing retired model
errors from contaminating the improvement loop. Version 10 therefore starts a
fresh verification series. SOC MAE is reported by lead bucket; solar and load
MAE/bias diagnose the two principal error sources. Skill is measured against
persistence, and the fixed-lead hindcast shows what the dashboard would have
forecast 6, 24, 48, and 72 hours before each observation.

The 50-member SOC ensemble also re-anchors when the deterministic forecast's
SOC anchor, calibrated solar contract, finite-state load distribution, battery
model, model version, named mode, or exact-state phase profile changes, even
when ECMWF is still on the same 00/12 UTC ensemble cycle. The
cycle's accumulated SSRD values at the AURORA grid point are cached as a small
site Zarr, so hourly re-anchoring does not redownload or reparse the global
GRIB. This keeps the probabilistic forecast aligned with mode transitions such
as `DC-Only` to `DC-Only + CL61`.

When additional AC kit is switched on, its fresh PDU signature creates or
updates that named mode automatically. The dashboard shows the recognised mode,
confidence, component-aware plan curves, uncertainty, load axis, and the 40%
minimum operational reference so operators can inspect the classification and
risk directly.

Forecast jobs compute a semantic publication signature from the physical
anchor, load, mode, ECMWF cycle, solar contract, battery parameters, and model
version. Repeated runs with unchanged forecast meaning update state and health
but do not rewrite the public Zarr or create duplicate archive issues. This
keeps verification counts independent and reduces avoidable product churn.
