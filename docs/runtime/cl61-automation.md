# CL61-first scheduling and automation boundary

## Current state

The CL61 is the first instrument in the power plan and the only future
automation candidate. The shipped state is still **advisory / observe-only**:
no dashboard service, mobile endpoint, forecast job, or development product can
contact the ASS PDU.

Development may publish three small diagnostic products beside its isolated
operating-scenario Zarr:

- `cl61_automation_intent.json`
- `cl61_automation_status.json`
- `cl61_automation_history.jsonl`

They are enabled only by `AURORA_CL61_AUTOMATION_SHADOW_ENABLED=true`. Their
authority is always `diagnostic`, their safety field has
`control_eligible=false`, and the status has `mode=observe_only` and
`capability=false`. A missing or malformed product is represented as
unavailable; it is never interpreted as permission to act.

`GET /mobile/v1/power/cl61-automation` exposes only that status. It is a GET
endpoint with no command, arm, write, restart, or PDU route. The development
iOS build reads it only from data-ocean and rejects any status that claims a
capability or names an outlet other than CL61 / outlet 5.

## Schedule physics

The 96-hour plan is evaluated against the 240-hour P10 reserve tail. It first
finds a feasible CL61 timetable using the finite-state load model, three-array
solar forecast, battery ensemble, learned CL61 heater/blower phase, 12-hour
minimum runs, and at most one scheduled start per UTC day. Only then may Radar
and HATPRO use residual reserve.

An already-on CL61 is held through the planning horizon rather than treated as
an implicit switch-off. Existing Radar or HATPRO loads are also held at their
observed PDU state: the CL61 controller cannot silently change another outlet,
so the forecast must not borrow energy from a fictional stop. The product marks
those states as held observations. An unsafe diagnostic trace is not an
instruction to change equipment.

## Quantitative implementation plan

The work proceeds in capability lanes, with all forecast data and artifacts
kept separate from the operational baseline. No CL61 action is an acceptance
criterion for the early lanes.

| Stage | Deliverable | Evidence required before advancing |
| --- | --- | --- |
| 1. Frozen comparison | Immutable same-cycle baseline and candidate bundles, with full forecast identity and checksums | No candidate write to baseline products; every scored pair is complete and reproducible |
| 2. Physical solar | Three-array POA/PV v2, sub-hourly battery integration, controller-curtailment masking | Energy conservation; per-array/total daylight skill only on MPP-active data; no censored delivery scored as available PV |
| 3. Finite-state load | Exact PDU state and phase model, DC/AC-bus accounting, explicit UAS recovery branch | State-held episode skill; phase totals and component loads are never added together; mixed UAS docks remain explicit or abstained |
| 4. Development shadow | CL61-first schedule, status/history, and the additive development iOS view | All paths remain development-only and read-only; unknown/stale input gives an unavailable or hold result |
| 5. Future start-only trial | Edge-local controller capability, after a separate operator decision | The quantitative gates below plus independent local PDU/SNMP and readback review |

Run four immutable, same-input evaluation lanes: **A** frozen baseline;
**B** physical solar only; **C** revised finite-state load only; and **D** the
combined forecast. Solar-only pairs keep the baseline load byte-identical;
load-changing pairs share only the raw cutoff, source checksum, SOC anchor,
battery contract, and solar forcing. A model/configuration change starts a new
campaign and cannot inherit earlier evidence.

The current rolling view is a health diagnostic, not promotion evidence. The
cumulative campaign uses independent ECMWF cycles (not cached re-anchors),
half-open UTC windows, complete source manifests, and cycle-clustered paired
statistics. Sparse cloud/state/degraded-mode strata remain explicitly
diagnostic.

### Forecast gates

For a development forecast replacement, require at least 30 paired independent
cycles across 10 UTC days and all of the following:

- Combined 0–24-hour SOC MAE improves by at least 10%; neither 0–6 nor 6–24
  hours worsens by more than 2%.
- Both 0–6 and 6–24-hour SOC persistence skill are positive. A short-lead-only
  result is insufficient.
- Solar MAE and absolute load bias each improve by at least 10%; P10–P90
  coverage is 75–90%; quantile ordering and SOC limits hold.
- Each longer lead has non-negative persistence skill or at least 25% lower
  MAE than baseline without worse absolute bias.

Before considering an outlet-5 trial, raise that bar to at least 60 independent
cycles over 30 days, at least 95% successfully paired eligible cycles, 100%
complete scored bundles, and adequate clear/cloudy and important-state
coverage. The 6–24 and 24–48-hour SOC buckets must have positive persistence
skill, no reserve false-safe event may be accepted, and the paired confidence
interval for the primary SOC improvement must exclude zero. Physical solar
also needs surveyed array/controller facts or an explicitly bounded provisional
contract, plus MPP-active verification. UAS recovery remains a conservative
0.20-kWh planning allowance until at least 20 complete, mapped recoveries are
available.

## Safe rollout sequence

1. Release the dashboard and cloud changes only to development; publish the
   shadow files next to the isolated development power product and leave the
   operational mobile payload unchanged.
2. Build the explicit `AuroraDashboardDevelopment` app against data-ocean. Its
   CL61 card is observational and rejects any executable status.
3. Run the focused edge playbook in check mode, then (only with a separate
   deployment decision) install the no-network observer. The observer is a
   receipt/audit service, not a controller.
4. Accumulate and review the campaign evidence. A status, a dashboard plot, or
   a passing one-day rolling score never authorises an outlet action.
5. Only after a further explicit operator decision can the future start-only
   capability be considered. Automatic stop remains off until a subsequent,
   separately reviewed experiment.

## Future local controller

The companion edge implementation is intentionally separate from the cloud
forecast. It has two systemd lanes:

1. A root-owned observer with `IPAddressDeny=any`, which can record a shadow
   receipt but cannot reach any PDU.
2. A disabled controller timer, limited to the PDU address, requiring all of:
   a locally installed immutable release, deployment capability enabled,
   a local capability sentinel, a bounded arm lease, a signed operational
   intent, current local SOC/PDU health, outlet-5 readback, and no manual
   override or fault latch.

No component currently produces a signed operational intent. Even after a
future deployment capability change, automatic stops remain separately disabled
until they receive their own explicit approval. A failed command or readback
latches `fault`; retries stop until an operator clears it. Disarming never
changes the physical outlet state.

The promotion evidence and local preflight must be reviewed before the edge
activation runbook is used. This document is not an operational instruction.
