from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr

import send_ops_alerts

try:
    from fastapi.testclient import TestClient
    import mobile_api
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without optional API deps
    TestClient = None
    mobile_api = None
    FASTAPI_IMPORT_ERROR = exc
else:
    FASTAPI_IMPORT_ERROR = None


@unittest.skipIf(TestClient is None, f"FastAPI test dependencies are not installed: {FASTAPI_IMPORT_ERROR}")
class MobileAPITests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(mobile_api.app)

    def test_health_is_public_without_disclosing_token_configuration(self) -> None:
        with patch.dict(
            os.environ,
            {"AURORA_MOBILE_API_TOKEN": "secret", "AURORA_MOBILE_API_AUTH_MODE": "required"},
            clear=False,
        ):
            response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
        self.assertTrue(response.json()["authRequired"])
        self.assertEqual(response.json()["accessMode"], "required")
        self.assertNotIn("tokenConfigured", response.json())

    def test_public_read_only_mode_allows_app_reads_with_a_stale_token(self) -> None:
        with patch.dict(
            os.environ,
            {
                "AURORA_MOBILE_API_TOKEN": "secret",
                "AURORA_MOBILE_API_AUTH_MODE": "public_read_only",
            },
            clear=False,
        ):
            health = self.client.get("/health")
            manifest = self.client.get(
                "/manifest",
                headers={"Authorization": "Bearer stale-testflight-token"},
            )

        self.assertFalse(health.json()["authRequired"])
        self.assertEqual(health.json()["accessMode"], "public_read_only")
        self.assertEqual(manifest.status_code, 200)
        self.assertEqual(
            manifest.headers["Cache-Control"],
            "public, max-age=30, stale-while-revalidate=60",
        )

    def test_cl61_automation_endpoint_is_read_only_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            status_path = Path(tmp) / "cl61_automation_status.json"
            status_path.write_text(
                json.dumps(
                    {
                        "mode": "observe_only",
                        "capability": False,
                        "control_authority": "observe_only",
                        "target": {"instrument": "CL61", "pdu_outlet": 5},
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {
                    "AURORA_MOBILE_API_AUTH_MODE": "public_read_only",
                    "CL61_AUTOMATION_STATUS_PATH": str(status_path),
                },
                clear=False,
            ):
                response = self.client.get("/power/cl61-automation")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["available"])
        self.assertEqual(response.json()["status"]["mode"], "observe_only")
        self.assertFalse(response.json()["status"]["capability"])

    def test_public_read_only_mode_keeps_artifact_inventory_protected(self) -> None:
        with patch.dict(
            os.environ,
            {
                "AURORA_MOBILE_API_TOKEN": "secret",
                "AURORA_MOBILE_API_AUTH_MODE": "public_read_only",
            },
            clear=False,
        ):
            unauthorized = self.client.get("/artifacts/manifest")
            authorized = self.client.get(
                "/artifacts/manifest",
                headers={"Authorization": "Bearer secret"},
            )

        self.assertEqual(unauthorized.status_code, 401)
        self.assertEqual(authorized.status_code, 200)

    def test_legacy_public_mode_does_not_bypass_authentication(self) -> None:
        with patch.dict(
            os.environ,
            {"AURORA_MOBILE_API_TOKEN": "secret", "AURORA_MOBILE_API_ALLOW_PUBLIC": "true"},
            clear=False,
        ):
            response = self.client.get("/manifest")

        self.assertEqual(response.status_code, 401)

    def test_manifest_requires_bearer_token(self) -> None:
        with patch.dict(os.environ, {"AURORA_MOBILE_API_TOKEN": "secret"}, clear=False):
            unauthorized = self.client.get("/manifest")
            authorized = self.client.get("/manifest", headers={"Authorization": "Bearer secret"})

        self.assertEqual(unauthorized.status_code, 401)
        self.assertEqual(authorized.status_code, 200)
        self.assertIn("power", {instrument["id"] for instrument in authorized.json()["instruments"]})
        self.assertIn("deployment", authorized.json())

    def test_read_only_payloads_are_short_term_cacheable(self) -> None:
        with patch.dict(os.environ, {"AURORA_MOBILE_API_TOKEN": "secret"}, clear=False):
            response = self.client.get("/manifest", headers={"Authorization": "Bearer secret"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["Cache-Control"], "private, max-age=30, stale-while-revalidate=60")

    def test_display_artifact_manifest_requires_bearer_token(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "latest.json"
            manifest.write_text(json.dumps({"schemaVersion": 1, "artifactCount": 2}), encoding="utf-8")
            with patch.dict(
                os.environ,
                {"AURORA_MOBILE_API_TOKEN": "secret", "AURORA_DISPLAY_ARTIFACT_MANIFEST": str(manifest)},
                clear=False,
            ):
                unauthorized = self.client.get("/artifacts/manifest")
                authorized = self.client.get("/artifacts/manifest", headers={"Authorization": "Bearer secret"})

        self.assertEqual(unauthorized.status_code, 401)
        self.assertEqual(authorized.status_code, 200)
        self.assertTrue(authorized.json()["available"])
        self.assertEqual(authorized.json()["artifactCount"], 2)

    def test_operations_endpoint_reads_fixture_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "latest.json"
            health = root / "latest_health.json"
            alerts = root / "state.json"
            snapshot.write_text(
                json.dumps(
                    {
                        "time_utc": "2026-07-05T07:30:00Z",
                        "cl61_source_sync_service_healthy_state": 1,
                        "ceilometer_append_service_healthy_state": 0,
                        "ceilometer_quicklooks_service_healthy_state": 1,
                    }
                ),
                encoding="utf-8",
            )
            health.write_text(json.dumps({"overall_level": "red"}), encoding="utf-8")
            alerts.write_text(json.dumps({"active": {"storage": {"title": "Storage high", "level": "red"}}}), encoding="utf-8")

            with patch.dict(
                os.environ,
                {
                    "AURORA_MOBILE_API_TOKEN": "secret",
                    "OPS_MONITOR_LATEST_SNAPSHOT": str(snapshot),
                    "OPS_MONITOR_LATEST_HEALTH": str(health),
                    "OPS_MONITOR_ALERT_STATE": str(alerts),
                },
                clear=False,
            ):
                response = self.client.get("/operations", headers={"Authorization": "Bearer secret"})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["overallLevel"], "red")
        self.assertEqual(body["alerts"][0]["title"], "Storage high")
        ceilometer = next(stream for stream in body["streamStates"] if stream["id"] == "ceilometer")
        self.assertEqual(ceilometer["level"], "red")

    def test_operations_endpoint_preserves_storage_alert_detail_and_severity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "latest.json"
            health = root / "latest_health.json"
            archive = root / "archive_health.json"
            alerts = root / "state.json"
            alert_log = root / "alerts.jsonl"
            snapshot_data = {
                "time_utc": "2026-08-29T08:00:00Z",
                "aurora_root_used_pct": 80.0,
                "aurora_root_free_gb": 19.5,
                "aurora_root_resolved_path": "/",
            }
            snapshot.write_text(json.dumps(snapshot_data), encoding="utf-8")
            health.write_text(json.dumps({"overall_level": "green"}), encoding="utf-8")
            archive.write_text("{}", encoding="utf-8")

            with patch.object(send_ops_alerts, "_recent_pdu_outlet_states", return_value=None), patch.object(
                send_ops_alerts, "_transport_configured", return_value=False
            ):
                send_ops_alerts.process_alerts(
                    snapshot_data,
                    state_path=alerts,
                    log_path=alert_log,
                )

            with patch.dict(
                os.environ,
                {
                    "AURORA_MOBILE_API_TOKEN": "secret",
                    "OPS_MONITOR_LATEST_SNAPSHOT": str(snapshot),
                    "OPS_MONITOR_LATEST_HEALTH": str(health),
                    "ARCHIVE_HEALTH_PATH": str(archive),
                    "OPS_MONITOR_ALERT_STATE": str(alerts),
                },
                clear=False,
            ):
                response = self.client.get(
                    "/operations", headers={"Authorization": "Bearer secret"}
                )

        self.assertEqual(response.status_code, 200)
        storage_alert = next(
            alert
            for alert in response.json()["alerts"]
            if alert["id"] == "storage:aurora_root"
        )
        self.assertEqual(storage_alert["level"], "amber")
        self.assertEqual(storage_alert["title"], "AURORA Cloud root disk storage at 80.0%")
        self.assertEqual(
            storage_alert["detail"],
            "AURORA Cloud root disk is using 80.0% of capacity, free=19.5 GB. Path: /.",
        )

    def test_overview_and_uas_endpoints_require_auth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log = root / "menapia_mqtt.log"
            log.write_text("2026-07-05 07:30:00: 4 3\n", encoding="utf-8")
            with patch.dict(
                os.environ,
                {"AURORA_MOBILE_API_TOKEN": "secret", "UAS_MQTT_LOG_PATH": str(log)},
                clear=False,
            ):
                self.assertEqual(self.client.get("/overview").status_code, 401)
                overview = self.client.get("/overview", headers={"Authorization": "Bearer secret"})
                uas = self.client.get("/uas", headers={"Authorization": "Bearer secret"})

        self.assertEqual(overview.status_code, 200)
        self.assertEqual(uas.status_code, 200)
        self.assertEqual(uas.json()["latest"]["effectiveTier"], 3)

    def test_power_candidate_endpoint_is_separate_and_explicitly_labelled(self) -> None:
        expected = {
            "environment": "development",
            "authority": "candidate",
            "status": "candidate",
            "lane": "D_physical_solar_load_residual",
            "comparison": [],
        }
        with patch.dict(os.environ, {"AURORA_MOBILE_API_TOKEN": "secret"}, clear=False), patch.object(
            mobile_api.catalog,
            "power_solar_evaluation",
            return_value=expected,
        ) as evaluation:
            response = self.client.get(
                "/power/solar-evaluation",
                headers={"Authorization": "Bearer secret"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["authority"], "candidate")
        evaluation.assert_called_once_with(lane="D_physical_solar_load_residual")

    def test_disabled_power_candidate_endpoint_is_not_found(self) -> None:
        with patch.dict(os.environ, {"AURORA_MOBILE_API_TOKEN": "secret"}, clear=False), patch.object(
            mobile_api.catalog,
            "power_solar_evaluation",
            side_effect=KeyError("Power candidate evaluation is disabled on this host"),
        ):
            response = self.client.get(
                "/power/solar-evaluation",
                headers={"Authorization": "Bearer secret"},
            )

        self.assertEqual(response.status_code, 404)
        self.assertNotIn("/data/aurora", response.text)

    def test_uas_flight_listing_detail_and_safe_plot_media(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "flights").mkdir()
            (root / "plots").mkdir()
            flight = {
                "id": "flight-1",
                "sourceFlightID": "source-1",
                "dayUTC": "2026-08-28",
                "flightNumber": 1,
                "title": "Flight 1",
                "startTimeUTC": "2026-08-28T10:00:00Z",
                "endTimeUTC": "2026-08-28T10:00:01Z",
                "durationSeconds": 1,
                "samplePeriodSeconds": 1,
                "modifiedAt": "2026-08-28T10:01:00Z",
                "quality": {"level": "green", "warnings": []},
                "detailPath": "flights/flight-1.json",
                "plotPath": "plots/flight-1.png",
            }
            (root / "catalog.json").write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                        "lastRunState": "success",
                        "latestFlightID": "flight-1",
                        "availableDays": ["2026-08-28"],
                        "flights": [flight],
                    }
                ),
                encoding="utf-8",
            )
            (root / "flights" / "flight-1.json").write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "flight": flight,
                        "series": {
                            "timeUTC": ["2026-08-28T10:00:00Z", "2026-08-28T10:00:01Z"],
                            "temperatureC": {"SN0122": [8.0, 8.1], "SN0123": [8.2, 8.3]},
                            "pressureHpa": {"SN0122": [1005.0, 1005.1], "SN0123": [1006.0, 1006.1]},
                            "relativeHumidityPct": {"SN0122": [75.0, 76.0], "SN0123": [74.0, 75.0]},
                            "altitudeM": [-60.0, 48.0],
                        },
                    }
                ),
                encoding="utf-8",
            )
            (root / "plots" / "flight-1.png").write_bytes(b"flight-png")
            env = {"AURORA_MOBILE_API_TOKEN": "secret", "MENAPIA_PRODUCT_ROOT": str(root)}
            headers = {"Authorization": "Bearer secret"}
            with patch.dict(os.environ, env, clear=False):
                listing = self.client.get("/uas/flights?day=latest", headers=headers)
                detail = self.client.get("/uas/flights/flight-1", headers=headers)
                media = self.client.get("/media/uas/flights/flight-1", headers=headers)
                traversal = self.client.get("/uas/flights/..%2Fcatalog", headers=headers)
                unknown_media = self.client.get("/media/uas/flights/unknown", headers=headers)

        self.assertEqual(listing.status_code, 200)
        self.assertEqual(listing.json()["selectedDay"], "2026-08-28")
        self.assertTrue(listing.json()["flights"][0]["plotURL"].startswith("/media/uas/flights/flight-1?v="))
        self.assertIsNone(listing.json()["allFlightsPlotURL"])
        self.assertNotIn("detailPath", listing.json()["flights"][0])
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.json()["series"]["altitudeM"], [-60.0, 48.0])
        self.assertEqual(media.status_code, 200)
        self.assertEqual(media.content, b"flight-png")
        self.assertEqual(traversal.status_code, 404)
        self.assertEqual(unknown_media.status_code, 404)

    def test_uas_flight_routes_require_read_access(self) -> None:
        with patch.dict(os.environ, {"AURORA_MOBILE_API_TOKEN": "secret"}, clear=False):
            self.assertEqual(self.client.get("/uas/flights").status_code, 401)
            self.assertEqual(self.client.get("/uas/flights/flight-1").status_code, 401)
            self.assertEqual(self.client.get("/media/uas/flights/flight-1").status_code, 401)

    def test_power_accepts_all_group_without_generating_a_product(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.zarr"
            with patch.dict(
                os.environ,
                {"AURORA_MOBILE_API_TOKEN": "secret", "POWER_DISPLAY_SUMMARY_ZARR_PATH": str(missing)},
                clear=False,
            ):
                response = self.client.get("/power?window=24h&group=all", headers={"Authorization": "Bearer secret"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["group"], "all")
        self.assertEqual(response.json()["panels"], [])

    def test_power_accepts_current_and_forecast_groups(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.zarr"
            with patch.dict(
                os.environ,
                {"AURORA_MOBILE_API_TOKEN": "secret", "POWER_DISPLAY_SUMMARY_ZARR_PATH": str(missing)},
                clear=False,
            ):
                current = self.client.get(
                    "/power?window=24h&group=current",
                    headers={"Authorization": "Bearer secret"},
                )
                forecast = self.client.get(
                    "/power?window=96h&group=forecast",
                    headers={"Authorization": "Bearer secret"},
                )

        self.assertEqual(current.status_code, 200)
        self.assertEqual(current.json()["group"], "current")
        self.assertEqual(forecast.status_code, 200)
        self.assertEqual(forecast.json()["group"], "forecast")

    def test_power_current_group_prefers_the_section_product(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            current_path = Path(tmp) / "power_current_display.zarr"
            now = pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None).floor("h")
            times = pd.date_range(now - pd.Timedelta(hours=2), periods=4, freq="1h")
            xr.Dataset(
                {"BatterySOC": ("time", np.asarray([70.0, 71.0, 72.0, 73.0]))},
                coords={"time": times},
            ).to_zarr(current_path, mode="w", consolidated=True)
            with patch.dict(
                os.environ,
                {
                    "AURORA_MOBILE_API_TOKEN": "secret",
                    "POWER_CURRENT_DISPLAY_ZARR_PATH": str(current_path),
                    "POWER_DISPLAY_SUMMARY_ZARR_PATH": str(Path(tmp) / "missing.zarr"),
                },
                clear=False,
            ):
                response = self.client.get("/power?window=24h&group=current", headers={"Authorization": "Bearer secret"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["source"]["path"], str(current_path))

    def test_power_prewarmed_figure_is_a_cacheable_media_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            current = root / "power_current_latest_interactive.json"
            current.write_text('{"data":[],"layout":{}}', encoding="utf-8")
            with patch.dict(
                os.environ,
                {"AURORA_MOBILE_API_TOKEN": "secret", "AURORA_INTERACTIVE_PREWARM_DIR": str(root)},
                clear=False,
            ):
                response = self.client.get("/media/power/figure/current", headers={"Authorization": "Bearer secret"})
                unavailable = self.client.get("/media/power/figure/unknown", headers={"Authorization": "Bearer secret"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["Cache-Control"], "private, max-age=60")
        self.assertEqual(response.json(), {"data": [], "layout": {}})
        self.assertEqual(unavailable.status_code, 404)

    def test_auroracam_listing_and_original_media_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "radar-cam" / "2026-07-05"
            source.mkdir(parents=True)
            image = source / "radar-cam_2026-07-05_12-30.jpg"
            image.write_bytes(b"jpeg")
            with patch.dict(
                os.environ,
                {"AURORA_MOBILE_API_TOKEN": "secret", "AURORACAM_RAW_ROOT": str(root)},
                clear=False,
            ):
                listing = self.client.get("/auroracam", headers={"Authorization": "Bearer secret"})
                media = self.client.get(
                    "/media/auroracam/original/radar-cam/2026-07-05/radar-cam_2026-07-05_12-30.jpg",
                    headers={"Authorization": "Bearer secret"},
                )

        self.assertEqual(listing.status_code, 200)
        self.assertEqual(listing.json()["frames"][0]["cameraID"], "radar-cam")
        self.assertEqual(media.status_code, 200)
        self.assertEqual(media.content, b"jpeg")

    def test_quicklook_listing_and_media_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            quicklook_dir = root / "power"
            quicklook_dir.mkdir()
            (quicklook_dir / "power__summary__latest.png").write_bytes(b"png")

            with patch.dict(
                os.environ,
                {
                    "AURORA_MOBILE_API_TOKEN": "secret",
                    "AURORA_QUICKLOOK_ROOT": str(root),
                },
                clear=False,
            ):
                listing = self.client.get(
                    "/quicklooks?kind=science&instrument=power",
                    headers={"Authorization": "Bearer secret"},
                )
                media = self.client.get(
                    "/media/quicklook/science/power/latest",
                    headers={"Authorization": "Bearer secret"},
                )
                versioned_media = self.client.get(
                    listing.json()["latest"]["imageURL"],
                    headers={"Authorization": "Bearer secret"},
                )
                not_modified = self.client.get(
                    "/media/quicklook/science/power/latest",
                    headers={"Authorization": "Bearer secret", "If-None-Match": media.headers["ETag"]},
                )

        self.assertEqual(listing.status_code, 200)
        self.assertTrue(listing.json()["latest"]["imageURL"].startswith("/media/quicklook/science/power/latest/"))
        self.assertEqual(versioned_media.status_code, 200)
        self.assertEqual(versioned_media.content, media.content)
        self.assertEqual(media.status_code, 200)
        self.assertEqual(media.content, b"png")
        self.assertIn("ETag", media.headers)

        self.assertEqual(not_modified.status_code, 304)

    def test_quicklook_refresh_changes_native_image_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "vaisalamet"
            directory.mkdir()
            summary = directory / "vaisalamet__summary__latest.png"
            summary.write_bytes(b"first science image")
            (directory / "latest.png").write_bytes(b"legacy housekeeping image")
            with patch.dict(os.environ, {"AURORA_MOBILE_API_TOKEN": "secret", "AURORA_QUICKLOOK_ROOT": str(root)}):
                headers = {"Authorization": "Bearer secret"}
                endpoint = "/quicklooks?kind=science&instrument=vaisalamet"
                first = self.client.get(endpoint, headers=headers).json()
                first_url = first["latest"]["imageURL"]
                first_media = self.client.get(first_url, headers=headers)
                summary.write_bytes(b"updated science image with new observations")
                second = self.client.get(endpoint, headers=headers).json()
                second_url = second["latest"]["imageURL"]
                second_media = self.client.get(second_url, headers=headers)
                panel = self.client.get("/instruments/vaisalamet/summary", headers=headers).json()["panels"][0]
                self.assertNotEqual(first_url, second_url)
                self.assertNotIn("?", second_url)
                self.assertEqual(first_media.content, b"first science image")
                self.assertEqual(second_media.content, b"updated science image with new observations")
                self.assertEqual(second["entries"][0]["imageURL"], second_url)
                self.assertEqual(panel["imageURL"], second_url)
                self.assertEqual(self.client.get(second_url).status_code, 401)
                self.assertEqual(self.client.get(second_url, headers={**headers, "If-None-Match": second_media.headers["ETag"]}).status_code, 304)
                # A listing can age while a scheduled generator updates the PNG.
                self.assertEqual(self.client.get(first_url, headers=headers).content, second_media.content)
                self.assertEqual(self.client.get(second_url.replace("/science/", "/invalid/"), headers=headers).status_code, 404)

    def test_wxcam_listing_and_media_responses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            videos = root / "videos" / "fish_hdr"
            thumbs = root / "thumbs" / "fish_hdr" / "20260705"
            videos.mkdir(parents=True)
            thumbs.mkdir(parents=True)
            (videos / "20260705.mp4").write_bytes(b"video")
            thumbnail_name = "HDR_20260705_123000.jpg"
            (thumbs / thumbnail_name).write_bytes(b"thumb")

            with patch.dict(
                os.environ,
                {
                    "AURORA_MOBILE_API_TOKEN": "secret",
                    "WXCAM_DAILY_VIDEO_DIR": str(root / "videos"),
                    "WXCAM_HOURLY_THUMB_DIR": str(root / "thumbs"),
                    "WXCAM_CATALOG_PATH": str(root / "missing.sqlite"),
                },
                clear=False,
            ):
                listing = self.client.get(
                    "/wxcam?stream=fish_hdr&day=2026-07-05",
                    headers={"Authorization": "Bearer secret"},
                )
                video = self.client.get(
                    "/media/wxcam/video/fish_hdr/2026-07-05",
                    headers={"Authorization": "Bearer secret"},
                )
                thumb = self.client.get(
                    f"/media/wxcam/thumb/fish_hdr/20260705/{thumbnail_name}",
                    headers={"Authorization": "Bearer secret"},
                )

        self.assertEqual(listing.status_code, 200)
        self.assertTrue(listing.json()["video"]["exists"])
        self.assertEqual(listing.json()["thumbnails"][0]["hourUTC"], 12)
        self.assertEqual(video.status_code, 200)
        self.assertEqual(video.content, b"video")
        self.assertEqual(thumb.status_code, 200)
        self.assertEqual(thumb.content, b"thumb")


if __name__ == "__main__":
    unittest.main()
