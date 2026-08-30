from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import xarray as xr

from power_issue_time_features import build_issue_time_feature_snapshot, site_extract_sha256


def test_issue_time_snapshot_excludes_future_local_observations_and_late_public_source():
    cutoff = pd.Timestamp("2026-08-29T12:00:00")
    times = pd.to_datetime(["2026-08-29T11:55:00", "2026-08-29T12:05:00"])
    power = xr.Dataset(
        {
            "BatterySOC": (("time",), [70.0, 1.0]),
            "BatteryWatts": (("time",), [-120.0, -999.0]),
            "ACOutputWatts": (("time",), [200.0, 999.0]),
            "SolarMPPMode_East": (("time",), [2.0, 1.0]),
            "SolarMPPMode_South": (("time",), [2.0, 1.0]),
            "SolarMPPMode_West": (("time",), [2.0, 1.0]),
        },
        coords={"time": times},
    )
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        pdu_path = root / "pdu.zarr"
        asfs_path = root / "asfs.zarr"
        mqtt_path = root / "menapia_mqtt.log"
        source_root = root / "public"
        pdu = xr.Dataset(
            {
                "PDUOutlet1Watts": (("time",), [50.0, 999.0]),
                "PDUOutlet1State": (("time",), [1.0, 1.0]),
            },
            coords={"time": times},
        )
        asfs = xr.Dataset(
            {
                "sr30_swd_Irr_Avg": (("time",), [123.0, 999.0]),
                "spn1_tot_Avg": (("time",), [120.0, 999.0]),
                "spn1_dif_Avg": (("time",), [40.0, 999.0]),
                "watts_on_48vdc_Avg": (("time",), [210.0, 999.0]),
            },
            coords={"time": times},
        )
        pdu.to_zarr(pdu_path, mode="w", consolidated=True)
        asfs.to_zarr(asfs_path, mode="w", consolidated=True)
        mqtt_path.write_text(
            "2026-08-29 11:59:00: 5 5\n2026-08-29 12:05:00: 1 1\n",
            encoding="utf-8",
        )
        source_root.mkdir()
        extract = source_root / "gfs.zarr"
        xr.Dataset(
            {"ghi_w_m2": (("time",), np.asarray([10.0, 20.0]))}, coords={"time": times}
        ).to_zarr(extract, mode="w", consolidated=True)
        (source_root / "gfs.json").write_text(
            json.dumps(
                {
                    "schema_version": "site-extracted-public-source-v2",
                    "source": "GFS",
                    "site_extract_only": True,
                    "global_grid_retained": False,
                    "source_cycle_time_utc": "2026-08-29T06:00:00Z",
                    "delivery_time_utc": "2026-08-29T12:05:00Z",
                    "site_extract_path": "gfs.zarr",
                    "site_extract_format": "zarr",
                    "irradiance_variable": "ghi_w_m2",
                    "site_latitude": 64.829694,
                    "site_longitude": -23.248139,
                    "site_extract_sha256": site_extract_sha256(extract),
                }
            ),
            encoding="utf-8",
        )
        snapshot = build_issue_time_feature_snapshot(
            issue_time=cutoff,
            power_history=power,
            pdu_zarr=pdu_path,
            asfs_zarr=asfs_path,
            menapia_mqtt_log=mqtt_path,
            public_source_manifest_root=source_root,
        )
    features = snapshot.manifest["features"]
    assert features["aps"]["fields"]["BatterySOC"]["latest"] == 70.0
    assert features["aps"]["fields"]["SolarMPPMode_East"]["latest"] == 2.0
    assert features["asfs"]["fields"]["sr30_swd_Irr_Avg"]["latest"] == 123.0
    assert features["menapia"]["dock_pair_state"] == "dock1_5__dock2_5"
    assert snapshot.manifest["public_model_ablations"]["gfs"]["status"] == (
        "late_delivery_excluded_from_issue_time_use"
    )
    assert snapshot.manifest["features"]["delayed_diagnostics"][
        "cloudnet_hatpro_radar_ceilometer_camera"
    ].endswith("not_predictors")
    assert snapshot.snapshot_digest


def test_issue_time_snapshot_marks_on_time_public_source_ablation_without_pooling():
    times = pd.date_range("2026-08-29T11:30:00", periods=2, freq="10min")
    power = xr.Dataset(
        {"BatterySOC": (("time",), np.asarray([70.0, 70.0]))},
        coords={"time": times},
    )
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        extract = root / "ifs.zarr"
        xr.Dataset(
            {"ghi_w_m2": (("time",), np.asarray([10.0, 20.0]))}, coords={"time": times}
        ).to_zarr(extract, mode="w", consolidated=True)
        (root / "ifs.json").write_text(
            json.dumps(
                {
                    "schema_version": "site-extracted-public-source-v2",
                    "source": "IFS",
                    "site_extract_only": True,
                    "global_grid_retained": False,
                    "source_cycle_time_utc": "2026-08-29T06:00:00Z",
                    "delivery_time_utc": "2026-08-29T11:45:00Z",
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
        snapshot = build_issue_time_feature_snapshot(
            issue_time="2026-08-29T12:00:00",
            power_history=power,
            pdu_zarr=root / "missing-pdu.zarr",
            asfs_zarr=root / "missing-asfs.zarr",
            menapia_mqtt_log=root / "missing-mqtt.log",
            public_source_manifest_root=root,
        )
    ifs = snapshot.manifest["public_model_ablations"]["ifs"]
    assert ifs["status"] == "issue_time_available_pending_independent_ablation"
    assert ifs["role"] == "separate_ablation_only_not_pooled"
