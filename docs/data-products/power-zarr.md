# Power Zarr

Path:

- `/data/aurora/products/power/power.zarr`

## Dataset shape

- dimension: `time`
- deployed shape when checked on `2026-05-21`:
  - `time=1118886`
- time coverage when checked: `2026-05-05 15:15:23.598658936` to
  `2026-05-21 20:15:04`
- sorted unique `time` coordinate

## Time coordinate

- `time` is parsed from the raw `aps_time` column

## Useful root attributes

- `instrument = "power"`
- `title = "Power level1 data"`
- `source = "power_data_YYYYMMDD.csv"`
- `wind_columns_excluded = "true"`

## Variable layout

- one `float32` `time` series per retained source column
- the deployed store currently contains `62` variables

Important ingest rules:

- raw column names are normalized by replacing `.` with `_`
- columns containing `wind` are excluded
- columns ending in `time` are excluded
- `InternalHumidity` is the reserved optional APS internal relative-humidity
  field in percent, used for operations dew-point monitoring when present

Examples include:

- `ACOutputAmps`
- `ACOutputHZ`
- `ACOutputVolts`
- `ACOutputWatts`
- `BatteryAmps`
- `BatterySOC`
- `BatteryState`
- `BatteryWatts`
- `DCInverterWatts`
- `InternalTemperature`
- `InternalHumidity`, when the APS logger provides a true internal RH signal
- `HeatsinkTemperature`
- `TempSensor1`
- `TempSensor2`
- `TempSensor3`
- `TempSensor4`
- `MaxSolarWatts_East`

## Schema note

- append runs keep the existing variable set fixed in the same way as
  `vaisalamet`
- `InternalHumidity` is the one allowed optional schema expansion; if it first
  appears in new raw APS CSV files, the appender adds it to the existing store
  with `NaN` backfill for older samples instead of silently dropping it
- append writes materialize only the already-filtered new sample block before
  writing, matching the cross-instrument Zarr append policy

## Chunking

- `time`-only variables are chunked `(1200,)`

## Dashboard performance note

The stored Zarr schema and chunking remain unchanged. The dashboard opens this
store with larger read chunks for interactive plotting, applies display-only
sanity limits for impossible APS values, uses per-trace time downsampling, and
rounds live latest windows into 5-minute cache buckets. Those choices are
presentation-layer optimizations only; they do not change ingest or storage.
The interactive APS summary prefers a compact derived display-summary Zarr
under `/data/aurora/products/power/power_display_summary.zarr`. The latest APS
interactive figure is also prewarmed as Plotly JSON by the quicklook pipeline
under `/data/aurora/products/dashboard/prewarm/`.

## ASS PDU Zarr

Path:

- `/data/aurora/products/power/pdu.zarr`

Raw source mirror:

- `/project/aurora/raw/pdu`

The ASS PDU source files are synced from ASS Linux
`/home/aurora/data/pdu/pdu_DDMMYYYY.csv`. The CSV rows contain
`datetime,outlet,name,state,amps,watts`; the appender pivots those rows into
one time-indexed variable per outlet metric:

- `PDUOutlet1Watts` through `PDUOutlet8Watts`
- `PDUOutlet1Amps` through `PDUOutlet8Amps`
- `PDUOutlet1State` through `PDUOutlet8State`

`State` is encoded as `1` for on and `0` for off. The store is derived from
the synced raw CSV files and can be rebuilt without modifying the source
mirror.

## Derived display-summary Zarr

Path:

- `/data/aurora/products/power/power_display_summary.zarr`

This is the primary dashboard-serving Power summary product. It is built from
`power.zarr` plus optional ASFS logger `watts_on_48vdc_Avg` and ASS PDU outlet
power by `generate_power_display_summary.py`. It keeps the raw Power, ASFS
logger, and PDU Zarrs authoritative, but stores only the curated one-minute
traces needed by the APS summary panels. That lets the interactive browser
switch Power windows without repeatedly opening and merging multi-day
one-second APS samples.

In a historical validation snapshot on `2026-05-21`, this derived store had
`time=21547`, 25 data variables, sorted unique timestamps, and coverage from
`2026-05-05 15:15:00` to `2026-05-21 20:10:00`.

Important variables include:

- `SolarWatts_East`, `SolarWatts_South`, `SolarWatts_West`
- `SolarVolts_East`, `SolarVolts_South`, `SolarVolts_West`
- `BatteryAmps`, `BatteryWatts`, `BatterySOC`
- `ACOutputWatts`, `DCInverterWatts`, `watts_on_48vdc_Avg`
- `PDUOutlet1Watts` through `PDUOutlet8Watts`, when PDU data are available
- `ACOutputVolts`, `DCInverterVolts`
- `InternalTemperature`, `HeatsinkTemperature`, `TempSensor1`-`TempSensor4`
- `PowerDisplaySolarYield_East`
- `PowerDisplaySolarYield_South`
- `PowerDisplaySolarYield_West`
- `PowerDisplayCumulativePowerGeneratedTotal`
- `PowerDisplayCumulativePowerUtilised`

The APS interactive summary also renders a display-only **SOC 24 h Forecast**
panel from `BatterySOC`. It fits the latest 30 minutes and latest 2 hours of SOC
with a low-degree polynomial and extrapolates both fits 24 hours forward. These
forecast traces are not stored in any Zarr product.

When `/data/aurora/products/power/power_soc_forecast.zarr` is available, the
display summary also includes the ECMWF-informed **SOC 96 h Forecast** traces:

- `BatterySOCForecast`
- `BatterySOCForecast_Load100W`
- `BatterySOCForecast_Load200W`
- `BatterySOCForecast_Load300W`
- `BatterySOCForecast_Load400W`
- `BatterySOCForecast_Load500W`
- `BatterySOCForecast_Load600W`
- `ECMWFSolarIrradiance`
- `ForecastSolarWatts`
- `ForecastLoadWatts`

The fixed-load fields remain in the deterministic product for backwards
compatibility, but the dashboard no longer presents them as operating plans.
The visible scenario panel is populated from the operating-mode product
described below. It always includes seven stable comparisons: CL61, CL61 +
Radar, CL61 + HATPRO, CL61 + HATPRO + Radar, HATPRO + Radar, Radar, and HATPRO.

The **SOC 96 h Forecast** is rendered from the canonical
`SystemAsIsDecisionSOCP10`, `SystemAsIsDecisionSOCP50`,
`SystemAsIsDecisionSOCP90`, and
`SystemAsIsDecisionBelow40Probability` display fields. When a fresh operating
scenario is available, those fields are aliases of its current-mode
distribution. Otherwise they fall back to one ensemble product. The panel
therefore never combines a deterministic central trace with separately issued
ensemble bounds.

These traces describe the current system as-is, not different future
instrument schedules. Version 10 varies ECMWF weather,
calibrated battery parameters, and the learned P10-P90 startup/fan behavior for
the one detected operating state. Every member keeps that exact state through
the horizon. Planned instrument combinations are shown separately in the
operating-scenario product.

When `/data/aurora/products/power/power_soc_forecast_skill.zarr` is available,
the display summary also includes past-facing forecast verification traces:

- `ForecastVerificationSamples`
- `ForecastSOCMAE_0_6h_Verified`
- `ForecastSOCMAE_6_24h_Verified`
- `ForecastSOCMAE_24_48h_Verified`
- `ForecastSOCMAE_48_96h_Verified`
- `ForecastSOCBias_0_6h_Verified`
- `ForecastSOCSkill_0_6h`
- `ForecastSolarMAE24h`
- `ForecastSolarBias24h`
- `ForecastSolarSkill24h`
- `ForecastLoadMAE24h`
- `ForecastLoadBias24h`
- `ForecastLoadSkill24h`

Root attributes include:

- `power_display_summary_product = "true"`
- `source = "derived from power.zarr plus optional asfs_logger.zarr ASS 48 V power, pdu.zarr outlet power, power_soc_forecast.zarr, and power_soc_forecast_skill.zarr"`
- `frequency = "1min"` unless overridden by
  `AURORA_POWER_DISPLAY_SUMMARY_FREQ`

The display-summary product is regenerated by the Power quicklook pipeline. It
is safe to delete and rebuild because it is derived from the raw Power, ASFS
logger, PDU, and Power SOC forecast Zarrs.

### Browser section products

The quicklook pipeline also writes two smaller, consolidated products beside
the compatibility store:

- `/data/aurora/products/power/power_current_display.zarr`
- `/data/aurora/products/power/power_forecast_display.zarr`
- `/data/aurora/products/power/power_display_manifest.json`

The Current Conditions browser reads only observed panels from the first store.
Forecast & Planning reads forecast, scenario, and verification panels from the
second. The manifest records their bounds and dimensions. This prevents an
interactive page from opening or scanning forecast fields when the user is
viewing current station conditions. The existing
`power_display_summary.zarr` remains a compatibility fallback.

## Derived SOC forecast Zarr

Path:

- `/data/aurora/products/power/power_soc_forecast.zarr`

This is a derived operational forecast product generated by
`generate_power_soc_forecast.py`. It retrieves ECMWF `ssrd` surface solar
radiation forecast data, converts accumulated `J m-2` values into interval
solar power in `W m-2`, calibrates expected APS solar charging from recent
`SolarWatts_East`, `SolarWatts_South`, and `SolarWatts_West`, builds a
named operating-mode station-load forecast from the APS power balance, and
integrates SOC forward from the latest valid `BatterySOC`. Observed total load
is `SolarWatts_East + SolarWatts_South + SolarWatts_West - BatteryWatts`, where
positive battery power is charging and negative power is discharge. This
captures the 48 V DC load that is absent from inverter idle power.

The operational deterministic product uses
`finite_controlled_state_phases_v10` under the
`finite_operating_state_phases_v2` contract. It identifies the latest confirmed
PDU/APS operating state and holds that exact state through the forecast horizon.
Load comes from fresh PDU components plus the clean DC-only baseline where
possible, otherwise from a mature distribution learned for that exact state.
Within the state, the model can detect and learn startup, steady, and low/high
fan phases. It publishes phase-specific load quantiles and learned startup
duration rather than forcing one wattage across a real transient. Startup needs
two state-entry episodes and a fan level needs two non-contiguous segments
before its uncertainty is published. One-off historical operating levels are
excluded in favour of the latest confirmed steady level. A new sustained latest
level therefore updates the central load immediately, but does not add a
recurring phase band until it occurs again. The latest
whole-station balance is used only to bootstrap a state that does not yet have
component or exact-state evidence. A load change without either an explicit
state transition or a learned within-state phase transition is rejected during
validation.

Phase-aware fields include:

- `ForecastLoadP10Watts`, `ForecastLoadP50Watts`, and
  `ForecastLoadP90Watts`
- `ForecastLoadPhaseCode`
- root attributes `load_exact_state_id`, `load_current_phase`,
  `load_state_dynamics`, and `load_state_dynamics_reason`

The ensemble product adds `ForecastLoadPhaseCodeEnsemble`. The operating
scenario product adds `ScenarioLoadPhaseCode` and `ScenarioLoadPhaseEpoch`, so
every accepted load change remains attributable to a controlled state or phase
boundary.

SOC is integrated with a fitted battery model containing usable capacity,
charge/discharge efficiencies, and observed power limits. The fit excludes
near-full/empty battery intervals, unstable load transitions, invalid timing,
and inconsistent power/SOC direction. The explicit parasitic term is zero
because the whole-station balance already includes those losses. Root
attributes state whether the fit is calibrated or provisional.

Before replacing the forecast, the generator scores archived forecast runs
against newly arrived APS observations and updates an adaptive calibration state
used by later forecast runs. The full forecast job downloads ECMWF data every 3
hours; the learning job can run every 15 minutes with `--refresh-from-cache` to
reuse the newest cached ECMWF GRIB while re-anchoring to the latest actual SOC.

The product still contains legacy fixed-load what-if fields from `100 W` through
`600 W` for API compatibility. They are excluded from archive scoring and
adaptive learning and are not shown as operational dashboard scenarios.

Variables:

- `BatterySOCForecast` in `%`
- `BatterySOCForecast_Load100W` in `%`
- `BatterySOCForecast_Load200W` in `%`
- `BatterySOCForecast_Load300W` in `%`
- `BatterySOCForecast_Load400W` in `%`
- `BatterySOCForecast_Load500W` in `%`
- `BatterySOCForecast_Load600W` in `%`
- `ECMWFSolarIrradiance` in `W m-2`
- `ForecastSolarWatts` in `W`
- `ForecastLoadWatts` in `W`
- `ForecastLoadP10Watts` in `W`
- `ForecastLoadP50Watts` in `W`
- `ForecastLoadP90Watts` in `W`

Root attributes include ECMWF input file, generation time, initial SOC time and
value, horizon, calibration window, solar calibration factor, forecast load,
load-bias correction, SOC lead-bucket correction, adaptive alpha, and battery
parameters. Load diagnostics include `load_model`, `load_model_version`,
`load_mode`, `load_mode_source`, `load_measurement`,
`load_balance_measurement`, `load_mode_registry`, `load_mode_signature`,
`load_mode_learning_ready`, `load_mode_learning_reason`,
`load_mode_pdu_active_watts`, `load_regime_level_w`, and
`load_regime_run_hours`. Version 10 also records `load_anchor_method`,
`load_state_contract`, `load_state_hold_policy`, the state P10/P50/P90 load,
the measured and learned-reference loads, their disagreement, calibrated
battery capacity/efficiencies/limits, `solar_calibration_contract_id`, and a semantic
`publication_signature`. It also records `load_state_dynamics_signature`, so a
new startup or fan profile republishes the deterministic and ensemble products
even when the named operating state is unchanged. `minimum_operational_soc_pct = "40"` identifies the
reference used by SOC risk plots and ensemble threshold verification. Scenario attributes include
`scenario_loads_w = "100,200,300,400,500,600"`
and `scenario_solar_mode = "ecmwf"`; these fixed loads are compatibility-only
sensitivity fields, not operating states. The product is separate from
model-evaluation ECMWF products.

The forecast-run archive is stored at:

- `/data/aurora/products/power/power_soc_forecast_archive.zarr`

It keeps recent forecast issue times, valid times, lead hours, load P10/P50/P90,
phase code, and selected forecast variables for skill scoring. The latest
forecast product remains the dashboard-facing product.

The forecast-verification product is stored at:

- `/data/aurora/products/power/power_soc_forecast_skill.zarr`

It compares archived forecasts to observed `BatterySOC`, summed solar charging,
and power-balance total load at matching valid times. Metrics are computed on past
timestamps over a rolling 24-hour verification window and include SOC MAE by
lead-time bucket, 0-6 h SOC bias, recent solar/load MAE and bias, sample count,
and deterministic skill scores relative to a persistence reference. Skill is
defined as `1 - forecast_mae / persistence_mae`, so positive values indicate
the operational forecast is beating the simple persistence baseline.

Cached 15-minute learning runs are grouped by ECMWF cycle and valid time before
verification, so the independent-cycle count is not inflated by highly
overlapping forecasts. Skill is withheld when the persistence error is too
small to provide a stable denominator. Load verification is also filtered by
`LoadModelVersion`; historical errors from retired load models are not mixed
with the current model's MAE, bias, or skill.

The fixed-lead hindcast product is stored at:

- `/data/aurora/products/power/power_soc_hindcast.zarr`

It retains seven days of observed SOC together with archived forecasts made 6,
24, 48, and 72 hours earlier. The dashboard draws these against the 40% minimum
operational SOC line.

The ECMWF ensemble products are stored at:

- `/data/aurora/products/power/power_soc_ensemble_forecast.zarr`
- `/data/aurora/products/power/power_soc_ensemble_archive.zarr`
- `/data/aurora/products/power/power_soc_ensemble_skill.zarr`

The ensemble generator retrieves the 50 IFS perturbed `ssrd` members for the
latest 00/12 UTC cycle. It streams the global GRIB messages through ecCodes and
extracts only the nearest AURORA grid point before constructing the compact
member-by-time array, stores that small site-level SSRD dataset for four cycles,
then deletes the temporary global GRIB. On later hourly runs within the same
ECMWF cycle, it rebuilds the ensemble whenever the latest actual SOC, calibrated
solar factor, learned load level, model version, or named operating mode has
changed. These same-cycle updates use the site cache rather than downloading
ECMWF again. Dashboard variables include SOC P10, P50, P90, minimum, maximum,
and probability below the 40% minimum operational SOC threshold. Ensemble
members vary ECMWF weather, calibrated battery capacity and efficiencies, and
the exact-state P10-P90 load distribution. A member's load remains stationary
until a modeled schedule changes the operating-state code. The deterministic, ensemble, and scenario products
must share the same `solar_calibration_contract_id`.
Verification includes CRPS by lead bucket, P10-P90 coverage, threshold Brier
score, and verified ensemble-cycle count.

## Learned operating-state and planning Zarrs

Development paths:

- `/data/aurora/dev-products/power/power_soc_planning_forecast.zarr`
- `/data/aurora/dev-products/power/power_operating_state.zarr`
- `/data/aurora/dev-products/power/power_operating_scenarios.zarr`
- `/data/aurora/dev-products/power/cl61_automation_status.json` (diagnostic only)

The 240-hour planning forecast is refreshed from the ECMWF 00 and 12 UTC cycles.
The first 96 actionable hours preserve the current ensemble and its calibration
contract. After native ensemble coverage ends, the reserve tail extends each
member against the deterministic planning solar curve rather than holding the
final irradiance value constant. Decision-ensemble and reserve-tail calibration
contract IDs are stored separately.

`generate_power_operating_scenarios.py` runs every five minutes. It re-anchors
all scenarios to the latest finite `BatterySOC`, derives observed total load as
summed APS solar power minus signed `BatteryWatts`, and classifies the current
kit configuration from fresh PDU outlet evidence. Stale PDU evidence is treated
as unknown rather than carried into a new mode. When the deterministic product
publishes the finite-state contract, the current-mode scenario reuses its exact
system-as-is distribution; alternative modes use the scenario component model.
The state product contains:

- `OperatingModeCode` and `OperatingModeProbability`
- `OperatingModeConfidence`
- `OperatingLoadState`, the composed PDU mode plus canonical UAS and CL61
  substate
- `DirectStateConfirmed`, which is `1` only when all four assigned PDU outlets
  have direct state or watt evidence for that observation
- `ObservedLoadWatts` and `EstimatedModeLoadWatts`
- `LoadInnovationWatts` and `LoadObservationOutlier`
- `UASEffectiveTier`, `UASCanonicalTier`, and `UASChargingState`
- `CL61StateCode`
- ten-entry `learned_state` load quantiles and maturity metadata

The persisted `hybrid_canonical_uas_cl61_states_v11` learner combines a finite set of
named operating modes with robust component and exact-state phase learning. Its
components are the DC baseline, CL61, Radar, HATPRO, UAS, and an unknown-AC
increment. Existing observations are reclassified on each run, but component
parameters are updated only from timestamps newer than the saved training
cursor. A total station load can train an exact mode or phase only when the
UAS, CL61, Radar, and HATPRO observations form a complete PDU state vector.
Incomplete PDU rows remain available for state recognition and diagnostics but
cannot contaminate a named state's load distribution. Re-running unchanged data
therefore does not double count evidence.
The UAS MQTT effective tier is aligned with these 15-minute observations and
the state stores tier-specific load quantiles, sample count, independent
episode count, and observed duration. Raw Tier 11 trains canonical Tier 1 and
raw Tier 12 trains canonical Tier 2; those proxy profiles require two episodes
and six hours. Tiers 3-5 require three episodes and six hours. Raw Tier 1/2 can
be displayed as the current state but cannot train the proxy-backed profiles.

The canonical learned-state catalogue is:

- Tier 1
- Tier 1 + UAS Charging
- Tier 2
- Tier 2 + UAS Charging
- Tier 3
- Tier 3 + UAS Charging
- Tier 4 (12 V only)
- Tier 5 (all off)
- CL61
- CL61 (heater on)

Charging states use an explicit `UASCharge on`/`off` event contract. Before
charge data are learned, Tiers 1-3 add `300 W` for the first three forecast
hours and then return to their base tier. At least one complete 2.5-hour event
is required before observed charging increments and median duration replace
that estimate.

The scenario product carries P10, P50, and P90 SOC and load for these plans:

- current recognised mode
- DC-Only
- DC + CL61 continuously on
- additive CL61, Radar, and HATPRO schedule
- P50 continuation scenario for controlled instruments that are already on
- CL61
- CL61 + Radar
- CL61 + HATPRO
- CL61 + HATPRO + Radar
- HATPRO + Radar
- Radar
- HATPRO
- all instruments + UAS tier 3
- the ten canonical learned-state scenarios listed above
- each additional learned kit combination

The all-instruments scenario keeps CL61, Radar, HATPRO, and UAS active and sets
the UAS effective tier to 3 for the complete horizon. Until tier-3 evidence is
mature it is labelled provisional and uses P10/P50/P90 fallback loads of
`55/108/302 W`; mature observed tier-3 quantiles replace that fallback.

The optimized plan is **CL61-first**. It reserves the feasible CL61 timetable
against the current finite-state baseline, then allows Radar and HATPRO only in
the remaining safe reserve. An existing CL61 is held through the planning
horizon. Existing Radar or HATPRO loads are also held at their observed state:
the plan must not assume a PDU change that the CL61-only controller cannot make.
Each newly proposed instrument interval has a minimum 12-hour run and at most
one planned start per UTC day. The first 96 hours are decision hours; the
complete 240-hour tail still enforces P10 SOC at or above 40%. Learned
exact-state startup, fan, and heater/blower phases are evaluated inside the
search and checked again before publication.

Recommendations remain advisory. The forecast service does not issue PDU
commands; optional development shadow products are explicitly non-executable.
The dashboard
clamps both the SOC 96 h card and the scenario comparison to exactly 96 hours
from the latest physical SOC anchor; the reserve tail is retained for safety
analysis but is not labelled as part of the 96-hour decision display. The dashboard
also evaluates a user-selected CL61 start and duration directly from the stored
solar and component ensembles, so edits react without another ECMWF download.

The P50 continuation rule is published as a separate comparison scenario; it
does not replace the conservative P10 plan. It can keep only CL61, Radar, or
HATPRO instruments that are already on in the recommendation trace until the
median SOC next reaches 95%, provided median SOC never falls below 40% first.
It never starts an instrument. If no 95% recovery exists in the 96-hour
decision window, or if the median path crosses 40% before recovery, the trace
falls back to the existing P10 priority plan. The product records eligibility,
held instruments, recovery time, minimum pre-recovery P50 SOC, and an explicit
`advisory_only` authority marker.

The scenario contract distinguishes a feasible CL61-first schedule, a safe
reserve-only plan, and an infeasible result. An infeasible trace is an unsafe
diagnostic fallback, not an instruction to operate the PDU. The product records
the schedule policy, any held observed instruments, per-instrument hours and
starts, UTC-day totals, reason, and `operator_action_required=true`; browser
and iOS clients label that result **No Feasible CL61-first Schedule**. The
schedule plot retains the three binary traces and adds
`ScenarioActiveInstrumentCount`, the hourly sum from zero to three.

Every hourly advisory cycle is also written to
`power_operating_recommendations.json`. A decision record includes the first
96-hour proposed operating-mode windows; hourly active-instrument count, P10,
P50, and P90 SOC and load traces; UTC-day instrument-hour and energy totals; the
forecast/model provenance; feasibility status and reason; fixed base mode;
collection objective; and the 40% P10 safety constraint. As later
telemetry arrives, the same record accumulates a
hindcast comparison against actual SOC and PDU-detected mode: SOC MAE and bias,
minimum actual SOC, 40% breaches, mode adherence, and coverage. These records
are evidence for assessing advisory-schedule quality; they never actuate a PDU.

The public deterministic and scenario products are not rewritten when a new
timer run has the same semantic publication signature. State and health still
advance, while forecast archives avoid duplicate issues that would inflate
verification counts.

The interactive APS summary presents `ACOutputWatts` and `DCInverterWatts` on
separate left/right axes in the **Output Power** panel. The optional
`watts_on_48vdc_Avg` context trace is presented as its own **ASS 48 V DC
Power** panel when ASFS logger data are available.
The optional PDU outlet watt traces are presented as their own **ASS PDU Outlet
Power** panel when `pdu.zarr` is available. It shows only the assigned UAS,
CL61, Radar, and HATPRO outlets; unassigned outlets remain available in the
underlying product but do not add noise to the dashboard legend. The panel
remains visible when every assigned outlet reports `0 W`, so powered-off kit is
distinguishable from missing PDU data.

## Derived display-energy Zarr

Path:

- `/data/aurora/products/power/power_display_energy.zarr`

This is a compact compatibility product containing only the cumulative kWh
traces used by the APS cumulative panel. It can be built directly from
`power.zarr` by `generate_power_display_energy.py`, but the normal quicklook
pipeline now refreshes it from the broader display-summary product.

`BatterySOC` is stored in the broader display-summary product, not in this
compatibility store. The generated and utilised energy traces from this compact
store are plotted on the cumulative panel's right axis when the broader summary
product is unavailable.

When checked on `2026-05-21`, this derived store had `time=21547`, 5 data
variables, sorted unique timestamps, and coverage from
`2026-05-05 15:15:00` to `2026-05-21 20:10:00`.

Variables:

- `PowerDisplaySolarYield_East`
- `PowerDisplaySolarYield_South`
- `PowerDisplaySolarYield_West`
- `PowerDisplayCumulativePowerGeneratedTotal`
- `PowerDisplayCumulativePowerUtilised`

Root attributes include:

- `power_display_energy_product = "true"`
- `source = "derived from power.zarr"`
- `frequency = "1min"` unless overridden by
  `AURORA_POWER_DISPLAY_ENERGY_FREQ`

The display-energy product is safe to delete and rebuild because it is derived
entirely from the dashboard display-summary logic.
