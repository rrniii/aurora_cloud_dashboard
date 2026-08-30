from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import xarray as xr

from power_issue_time_features import site_extract_sha256
from power_public_source_ablation import run_public_source_ablations
from power_solar_model import load_physical_solar_config


CONFIG_PATH = Path(__file__).with_name("config") / "power_solar_physical_candidate_v1.json"


def _baseline(issue: pd.Timestamp) -> xr.Dataset:
    times = pd.date_range(issue, periods=5, freq="3h")
    return xr.Dataset(
        {
            "BatterySOCForecast": (("time",), np.asarray([70.0, 69.0, 68.0, 67.0, 66.0])),
            "ForecastLoadWatts": (("time",), np.full(len(times), 180.0)),
            "ForecastSolarWatts": (("time",), np.asarray([0.0, 100.0, 400.0, 250.0, 0.0])),
            "ECMWFSolarIrradiance": (("time",), np.asarray([0.0, 50.0, 200.0, 120.0, 0.0])),
        },
        coords={"time": times},
        attrs={
            "initial_soc_time": issue.isoformat(),
            "initial_soc_pct": "70",
            "ecmwf_cycle_time": issue.isoformat(),
            "forecast_horizon_hours": "12",
            "publication_signature": "baseline-publication",
            "forecast_model_contract_id": "baseline-contract",
            "forecast_system_version": "v11-control",
            "load_model_version": "11",
            "battery_usable_capacity_kwh": "26",
            "battery_charge_efficiency": "0.92",
            "battery_discharge_efficiency": "0.92",
            "battery_max_charge_w": "3000",
            "battery_max_discharge_w": "3000",
        },
    )


def test_public_source_ablation_is_site_bounded_separate_and_paired() -> None:
    issue = pd.Timestamp("2026-06-21T06:00:00")
    baseline = _baseline(issue)
    power = xr.Dataset(
        {"BatterySOC": (("time",), baseline["BatterySOCForecast"].values)},
        coords={"time": baseline["time"].values},
    )
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        source_root = root / "source"
        source_root.mkdir()
        extract = source_root / "ifs.zarr"
        xr.Dataset(
            {"ghi_w_m2": (("time",), np.asarray([0.0, 80.0, 500.0, 280.0, 0.0]))},
            coords={"time": baseline["time"].values},
        ).to_zarr(extract, mode="w", consolidated=True)
        (source_root / "ifs.json").write_text(
            json.dumps(
                {
                    "schema_version": "site-extracted-public-source-v2",
                    "source": "IFS",
                    "site_extract_only": True,
                    "global_grid_retained": False,
                    "source_cycle_time_utc": issue.isoformat() + "Z",
                    "delivery_time_utc": issue.isoformat() + "Z",
                    "site_extract_path": "ifs.zarr",
                    "site_extract_format": "zarr",
                    "irradiance_variable": "ghi_w_m2",
                    "site_latitude": 64.829694,
                    "site_longitude": -23.248139,
                    "site_extract_sha256": site_extract_sha256(extract),
                }
            ),
            encoding="utf-8",
        )
        results = run_public_source_ablations(
            candidate_root=root / "candidate",
            baseline=baseline,
            power_for_evidence=power,
            source_manifest_root=source_root,
            configuration=load_physical_solar_config(CONFIG_PATH),
            latitude=64.829694,
            longitude=-23.248139,
            code_revision="test-revision",
        )
        assert results["ifs"]["status"] == "complete"
        assert results["gfs"]["status"] == "not_enrolled_no_site_manifest"
        output_path = Path(str(results["ifs"]["path"]))
        with xr.open_zarr(output_path, chunks={}) as candidate:
            assert candidate.attrs["public_source"] == "IFS"
            assert candidate.attrs["candidate_lane"] == "E_public_source_ifs"
            assert candidate.attrs["solar_forcing_mode"] == "public_source_site_physical_available_pv"
            np.testing.assert_allclose(
                candidate["ForecastInputGlobalHorizontalIrradiance"].values,
                np.asarray([0.0, 80.0, 500.0, 280.0, 0.0]),
            )
        pairs = list((root / "candidate" / "public_model_ablations" / "ifs" / "pairs").rglob("pair_manifest.json"))
        assert len(pairs) == 1
        assert json.loads(pairs[0].read_text(encoding="utf-8"))["pair_status"] == "complete"
    # The source lane never mutates the baseline in memory.
    np.testing.assert_allclose(baseline["ForecastLoadWatts"].values, 180.0)


def test_public_source_ablation_rejects_a_retained_spatial_dimension() -> None:
    issue = pd.Timestamp("2026-06-21T06:00:00")
    baseline = _baseline(issue)
    power = xr.Dataset(
        {"BatterySOC": (("time",), baseline["BatterySOCForecast"].values)},
        coords={"time": baseline["time"].values},
    )
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        source_root = root / "source"
        source_root.mkdir()
        extract = source_root / "icon.zarr"
        xr.Dataset(
            {
                "ghi_w_m2": (
                    ("time", "latitude"),
                    np.ones((baseline.sizes["time"], 2), dtype=np.float64),
                )
            },
            coords={"time": baseline["time"].values, "latitude": [64.8, 64.9]},
        ).to_zarr(extract, mode="w", consolidated=True)
        (source_root / "icon.json").write_text(
            json.dumps(
                {
                    "schema_version": "site-extracted-public-source-v2",
                    "source": "ICON",
                    "site_extract_only": True,
                    "global_grid_retained": False,
                    "source_cycle_time_utc": issue.isoformat() + "Z",
                    "delivery_time_utc": issue.isoformat() + "Z",
                    "site_extract_path": "icon.zarr",
                    "site_extract_format": "zarr",
                    "irradiance_variable": "ghi_w_m2",
                    "site_latitude": 64.829694,
                    "site_longitude": -23.248139,
                    "site_extract_sha256": site_extract_sha256(extract),
                }
            ),
            encoding="utf-8",
        )
        results = run_public_source_ablations(
            candidate_root=root / "candidate",
            baseline=baseline,
            power_for_evidence=power,
            source_manifest_root=source_root,
            configuration=load_physical_solar_config(CONFIG_PATH),
            latitude=64.829694,
            longitude=-23.248139,
            code_revision="test-revision",
        )
    assert results["icon"]["status"] == "blocked_public_source_ablation_error:ValueError"
    assert "non-site dimensions" in str(results["icon"]["reason"])
