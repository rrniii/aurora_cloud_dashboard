# Physical APS solar forecast candidate

Status: candidate only. The operational v10 latest product, state, archive,
skill product, hindcast, ensemble and operating-scenario products remain on the
existing `ssrd_scalar_lead_mos_v1` solar model.

## What changes

The baseline converts ECMWF surface solar radiation downwards (`ssrd`) to PV
power with one fitted W per W m-2 multiplier and lead-time MOS factors. The
candidate instead:

1. Converts accumulated SSRD energy to interval-mean global horizontal
   irradiance (GHI).
2. Resolves NOAA-style solar zenith and azimuth at the Aurora site.
3. Disaggregates each forecast interval to ten-minute substeps using the
   extraterrestrial-horizontal solar shape, while conserving the ECMWF GHI
   energy in each interval exactly.
4. Splits GHI into direct and diffuse components. It uses forecast direct
   radiation when present and a bounded Erbs split otherwise.
5. Transposes direct, isotropic-sky diffuse and ground-reflected irradiance to
   the East, South and West array planes separately.
6. Applies an ASHRAE direct-beam incidence-angle modifier, optional Faiman
   module-temperature proxy, a PVWatts-style DC conversion, and an individual
   controller limit for each array.
7. Integrates SOC from available PV while reporting battery acceptance and
   curtailment explicitly. The ten-minute trace is retained through battery
   integration, and the first interval is split at the exact SOC issue time.

The implementation is in `power_solar_model.py`. The parameter contract is the
versioned JSON file `config/power_solar_physical_candidate_v1.json`; its SHA-256
digest is carried into every candidate product and archive issue.

## Output contract

Existing consumers retain:

- `ForecastSolarWatts`: available electrical PV used as the common scenario
  forcing;
- `ECMWFSolarIrradiance`: raw ECMWF source-interval mean GHI;
- `BatterySOCForecast` and the fixed-load sensitivity curves.

The candidate adds:

- `ForecastPVAvailableWatts`, `ForecastPVDeliveredWatts` and
  `ForecastPVCurtailedWatts`;
- available power, conventional raw plane-of-array irradiance, direct/diffuse
  plane-of-array components, IAM-adjusted effective irradiance, temperature
  proxy and controller clipping for East, South and West;
- source-interval duration, issue-truncated effective GHI/duration, solar
  geometry, DNI, DHI and clearness index;
- solar/forecast model names, versions, contract IDs, feature set, physical
  config digest, degradation codes, input snapshot and evaluation-pair ID.

Available power is deliberately distinct from observed Victron charger output.
A full battery or external charge control can limit the latter even when more
PV is physically available.

## Current provisional assumptions

The repository confirms three MPPT channels labelled East, South and West, but
does not contain a surveyed bill of materials, exact tilt/azimuth, installed Wp
or controller ratings. The candidate therefore marks its configuration
`provisional` and uses cardinal azimuths, a 64.83 degree tilt prior and explicit
electrical scale priors. These values must not be presented as surveyed
hardware facts.

The existing deterministic GRIB normally contains SSRD only. The adapter can
also consume forecast direct radiation, 2 m temperature, 10 m wind and net
shortwave-derived albedo, but it declares Erbs decomposition, disabled
temperature correction and fixed albedo as degradation codes when those fields
are absent.

Residual solar calibration is disabled for the candidate. The prepared
real-time acquisition change defines Victron register 791 as
`SolarMPPMode_East/South/West`; it must be released through the normal edge
path before new values reach the power archive. Historical archive rows do not
contain it. Recent chargers have been observed in voltage/current-limited
operation. Learning available PV from those censored measurements would bias
the physical model low.

## Isolated paired evaluation

The original focused physical-solar runner remains available for isolated
experiments.  The integrated B/C/D evaluation candidate is documented in
`power-v12-hybrid-candidate.md` and uses the separate v12 root below.

Run the focused physical-solar experiment:

```bash
python generate_power_soc_physical_candidate.py
```

The runner reads the already-published v10 ECMWF input file and exact APS SOC
anchor, hashes the GRIB, and writes only beneath:

```text
/data/aurora/dev-products/power/evaluations/solar_physical_v1/
```

The main generator refuses to run the physical model against any configured
baseline latest/state/archive/skill/hindcast path. Candidate and v10 issues can
therefore share a valid time and cycle without overwriting or mixing archive
rows. The current ensemble remains scalar-v10 and is not combined with this
candidate.

Each completed issue preserves an immutable baseline snapshot, candidate
snapshot and manifest under
`pairs/<evaluation_pair_id>/<candidate_publication_signature>/`. Pair identity
includes the baseline publication, forcing checksum, issue time and physical
model contract. The complete manifest is written in a hidden staging directory
before the whole bundle is atomically made visible; paired evaluators must
accept only manifests with `pair_status = complete`.

SOC and load can be evaluated from the isolated candidate archive. Direct PV
skill is scored only where register 791 reports MPPT-active operation for all
three chargers; all other delivered-power rows remain censored. Promotion
requires:

- a surveyed array/controller configuration or an issue-time-safe fit from
  MPP-active samples;
- archive coverage of register 791 for all three chargers;
- paired v10/candidate cycle scoring by lead time and season;
- East/South/West and total daylight-energy verification on uncurtailed data;
- end-to-end SOC improvement without worse reserve-risk calibration;
- the same physical transformation applied member-by-member to the ECMWF
  ensemble before any operating-scenario use.

Method references: [NOAA solar calculations](https://gml.noaa.gov/grad/solcalc/solareqns.PDF),
[Erbs decomposition in pvlib](https://pvlib-python.readthedocs.io/en/latest/reference/generated/pvlib.irradiance.erbs.html),
and [NREL PVWatts](https://pvwatts.nrel.gov/downloads/pvwattsv5.pdf).
