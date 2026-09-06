from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from datetime import datetime, timedelta, timezone

import mobile_catalog


class MobileCatalogTests(unittest.TestCase):
    def test_power_bundle_digest_has_fixed_cross_repo_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "nested").mkdir()
            (root / "a.txt").write_bytes(b"alpha")
            (root / "nested" / "b.bin").write_bytes(bytes([0, 255]))

            digest = mobile_catalog._power_bundle_artifact_digest(root)

        self.assertEqual(
            digest,
            "b30b81d6b3aaa8da809cf52efe5a39de3b52a097e49a66d5aa0a44bc896b5eb2",
        )

    def test_power_bundle_status_validates_digests_and_observes_failed_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generation = root / "generation-one"
            forecast = generation / "power_forecast_display.zarr"
            forecast.mkdir(parents=True)
            (forecast / "zarr.json").write_text("forecast", encoding="utf-8")
            digest = mobile_catalog._power_bundle_artifact_digest(forecast)
            sibling_products = {}
            for logical_name, relative in (
                ("currentDisplay", "power_current_display.zarr"),
                ("displaySummary", "power_display_summary.zarr"),
                ("operatingScenarios", "power_operating_scenarios.zarr"),
            ):
                product_path = generation / relative
                product_path.mkdir()
                (product_path / "zarr.json").write_text(
                    logical_name, encoding="utf-8"
                )
                sibling_products[logical_name] = {
                    "relativePath": relative,
                    "sha256": mobile_catalog._power_bundle_artifact_digest(
                        product_path
                    ),
                }
            manifest = generation / "generation.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "status": "complete",
                        "dataUpdatedAt": "2026-09-05T12:00:00Z",
                        "validUntil": "2026-09-09T12:00:00Z",
                        "forecastIdentityID": "identity-one",
                        "sourceCycleSetID": "cycle-one",
                        "forecastRefreshKind": "ecmwf_cycle",
                        "independentCycle": True,
                        "controlAuthority": "advisory_only",
                        "cl61ActuationEnabled": False,
                        "products": {
                            **sibling_products,
                            "forecastDisplay": {
                                "relativePath": "power_forecast_display.zarr",
                                "sha256": digest,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            status_path = root / "status.json"
            status_path.write_text(json.dumps({"status": "complete"}), encoding="utf-8")
            mobile_catalog._POWER_BUNDLE_VALIDATION_CACHE.clear()
            environment = {
                "AURORA_POWER_FORECAST_ORCHESTRATION_ENABLED": "true",
                "AURORA_POWER_FORECAST_PUBLICATION_ACTIVE": "true",
                "AURORA_POWER_FORECAST_BUNDLE_READY_PATH": str(manifest),
                "AURORA_POWER_FORECAST_BUNDLE_STATUS_PATH": str(status_path),
            }
            with patch.dict(os.environ, environment, clear=False):
                self.assertEqual(
                    mobile_catalog.power_display_section_path("forecast").resolve(),
                    forecast.resolve(),
                )
                self.assertEqual(
                    mobile_catalog.power_display_section_path("current").resolve(),
                    (generation / "power_current_display.zarr").resolve(),
                )
                self.assertEqual(
                    mobile_catalog.power_display_summary_path().resolve(),
                    (generation / "power_display_summary.zarr").resolve(),
                )
                self.assertEqual(
                    tuple(
                        path.resolve()
                        for path in mobile_catalog.power_operating_scenario_paths()
                    ),
                    ((generation / "power_operating_scenarios.zarr").resolve(),),
                )
                current = mobile_catalog.power_forecast_bundle_status(forecast)
                self.assertEqual(current["status"], "complete")
                self.assertEqual(
                    current["verifiedProducts"],
                    [
                        "currentDisplay",
                        "displaySummary",
                        "forecastDisplay",
                        "operatingScenarios",
                    ],
                )
                self.assertEqual(current["forecastRefreshKind"], "ecmwf_cycle")
                self.assertIs(current["independentCycle"], True)
                with patch.object(
                    mobile_catalog, "datetime", wraps=datetime
                ) as mocked_datetime:
                    mocked_datetime.now.return_value = datetime(
                        2026, 9, 10, tzinfo=timezone.utc
                    )
                    expired_from_cache = mobile_catalog.power_forecast_bundle_status(
                        forecast
                    )
                self.assertEqual(expired_from_cache["status"], "stale")
                self.assertEqual(
                    expired_from_cache["staleReason"], "bundle_validity_expired"
                )
                status_path.write_text(
                    json.dumps({"status": "failed", "failedStage": "ensemble"}),
                    encoding="utf-8",
                )
                failed_retry = mobile_catalog.power_forecast_bundle_status(forecast)

        self.assertEqual(failed_retry["status"], "complete")
        self.assertEqual(failed_retry["lastAttemptStatus"], "failed")
        self.assertEqual(failed_retry["lastAttemptReason"], "ensemble")

    def test_power_bundle_activation_pending_uses_explicit_legacy_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy = root / "legacy_forecast.zarr"
            ready = root / "forecast-bundle" / "active" / "generation.json"
            environment = {
                "AURORA_POWER_FORECAST_ORCHESTRATION_ENABLED": "true",
                "AURORA_POWER_FORECAST_PUBLICATION_ACTIVE": "true",
                "AURORA_POWER_FORECAST_BUNDLE_READY_PATH": str(ready),
                "POWER_FORECAST_DISPLAY_ZARR_PATH": str(legacy),
            }
            with patch.dict(os.environ, environment, clear=False):
                selected = mobile_catalog.power_display_section_path("forecast")
                publication = mobile_catalog.power_forecast_bundle_status(selected)

        self.assertEqual(selected, legacy)
        self.assertEqual(publication["status"], "activation_pending")
        self.assertEqual(publication["displaySource"], "legacy")

    def test_shadow_orchestration_does_not_change_legacy_power_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy = root / "legacy_forecast.zarr"
            ready = root / "forecast-bundle" / "active" / "generation.json"
            environment = {
                "AURORA_POWER_FORECAST_ORCHESTRATION_ENABLED": "true",
                "AURORA_POWER_FORECAST_PUBLICATION_ACTIVE": "false",
                "AURORA_POWER_FORECAST_BUNDLE_READY_PATH": str(ready),
                "POWER_FORECAST_DISPLAY_ZARR_PATH": str(legacy),
            }
            with patch.dict(os.environ, environment, clear=False):
                selected = mobile_catalog.power_display_section_path("forecast")
                publication = mobile_catalog.power_forecast_bundle_status(selected)

        self.assertEqual(selected, legacy)
        self.assertFalse(publication["enabled"])
        self.assertEqual(publication["status"], "shadow_unpublished")

    def test_power_current_fails_closed_after_invalid_bundle_activation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "forecast-bundle" / "active" / "generation.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(
                json.dumps({"schemaVersion": 1, "status": "incomplete"}),
                encoding="utf-8",
            )
            environment = {
                "AURORA_POWER_FORECAST_ORCHESTRATION_ENABLED": "true",
                "AURORA_POWER_FORECAST_PUBLICATION_ACTIVE": "true",
                "AURORA_POWER_FORECAST_BUNDLE_READY_PATH": str(manifest),
                "AURORA_POWER_FORECAST_BUNDLE_STATUS_PATH": str(root / "status.json"),
                "POWER_CURRENT_DISPLAY_ZARR_PATH": str(root / "legacy-current.zarr"),
                "POWER_FORECAST_DISPLAY_ZARR_PATH": str(root / "legacy-forecast.zarr"),
            }
            mobile_catalog._POWER_BUNDLE_VALIDATION_CACHE.clear()
            with patch.dict(os.environ, environment, clear=False):
                response = mobile_catalog.power(window="24h", group="current")

        self.assertEqual(response["status"], "unavailable")
        self.assertEqual(response["panels"], [])
        self.assertIn("failed validation", response["warning"])

    def test_cl61_automation_status_is_read_only_and_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cl61_automation_status.json"
            path.write_text(
                json.dumps(
                    {
                        "mode": "observe_only",
                        "capability": False,
                        "control_authority": "observe_only",
                        "target": {"instrument": "CL61", "pdu_outlet": 5},
                        "last_proposed_action": "start",
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"CL61_AUTOMATION_STATUS_PATH": str(path)}):
                payload = mobile_catalog.cl61_automation_status()

        self.assertTrue(payload["available"])
        self.assertEqual(payload["status"]["mode"], "observe_only")
        self.assertFalse(payload["status"]["capability"])
        self.assertEqual(payload["status"]["target"]["pdu_outlet"], 5)
        self.assertNotIn("path", payload["source"])

    def test_cl61_automation_status_rejects_control_capability(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cl61_automation_status.json"
            path.write_text(
                json.dumps(
                    {
                        "mode": "armed",
                        "capability": True,
                        "control_authority": "operational",
                        "target": {"instrument": "CL61", "pdu_outlet": 5},
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"CL61_AUTOMATION_STATUS_PATH": str(path)}):
                payload = mobile_catalog.cl61_automation_status()

        self.assertFalse(payload["available"])
        self.assertFalse(payload["status"]["capability"])
        self.assertEqual(payload["status"]["reason_codes"], ["non_observe_only_status_rejected"])

    def test_power_candidate_evaluation_is_explicitly_development_only(self) -> None:
        import numpy as np
        import pandas as pd
        import xarray as xr

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lane = "D_physical_solar_load_residual"
            signature = "candidate-signature"
            pair_id = "pair-id"
            bundle = root / "lanes" / lane / "pairs" / pair_id / signature
            bundle.mkdir(parents=True)
            times = pd.date_range("2026-06-21T12:00:00", periods=2, freq="3h")
            baseline = xr.Dataset(
                {
                    "BatterySOCForecast": (("time",), [70.0, 69.0]),
                    "ForecastLoadWatts": (("time",), [100.0, 100.0]),
                    "ForecastSolarWatts": (("time",), [0.0, 200.0]),
                },
                coords={"time": times},
            )
            candidate = baseline.copy(deep=True)
            candidate["BatterySOCForecast"][:] = np.asarray([70.0, 69.5])
            candidate.attrs.update(
                {
                    "forecast_system_version": "power-v12-hybrid-candidate",
                    "forecast_model_contract_id": "contract-id",
                    "feature_set_version": "features-v1",
                    "feature_set_digest": "feature-digest",
                    "source_manifest_digest": "manifest-digest",
                    "source_cycle_set_id": "cycle-set",
                    "source_availability_code": "ecmwf_control=available;gfs=not_enrolled",
                    "local_feature_contract_id": "issue-features-v1-test",
                    "baseline_control_contract_id": "baseline-control-test",
                    "degraded_mode_code": "candidate_only",
                    "load_residual_model_status": "active",
                    "generated_at_utc": "2026-06-21T12:05:00Z",
                    "observation_cutoff_utc": "2026-06-21T12:00:00Z",
                    "forecast_code_revision": "revision-test",
                    "adaptive_calibration_state_id": "calibration-state-test",
                    "soc_bias_correction_method": "continuous_piecewise_linear_with_net_discharge_guard_v2",
                    "soc_physical_consistency_status": "passed",
                }
            )
            baseline.to_zarr(bundle / "baseline_forecast.zarr", mode="w", consolidated=True)
            candidate.to_zarr(bundle / "candidate_forecast.zarr", mode="w", consolidated=True)
            (bundle / "pair_manifest.json").write_text(
                json.dumps(
                    {
                        "pair_status": "complete",
                        "evaluation_pair_id": pair_id,
                        "candidate_publication_signature": signature,
                        "baseline_publication_signature": "baseline-signature",
                    }
                )
            )
            (root / "lanes" / lane / "evaluation_summary.json").write_text("{}")
            (root / "status.json").write_text(
                json.dumps(
                    {
                        "environment": "development",
                        "authority": "candidate",
                        "status": "complete",
                        "data_updated_at_utc": "2026-06-21T12:05:00Z",
                        "valid_until_utc": "2026-06-21T15:00:00Z",
                        "promotion_status": "not_eligible_requires_campaign_evidence",
                        "lanes": {lane: {"publication_signature": signature}},
                        "public_model_ablation_results": {
                            "ifs": {
                                "status": "complete",
                                "candidate_lane": "E_public_source_ifs",
                                "path": "/data/aurora/dev-products/secret.zarr",
                            }
                        },
                    }
                )
            )
            with patch.dict(
                os.environ,
                {
                    "AURORA_POWER_CANDIDATE_API_ENABLED": "true",
                    "AURORA_POWER_V12_CANDIDATE_ROOT": str(root),
                },
            ):
                payload = mobile_catalog.power_solar_evaluation(lane)
                (root / "run_status.json").write_text(
                    json.dumps(
                        {
                            "status": "failed",
                            "reason_code": "candidate_generation_failed:ValueError",
                            "requested_at_utc": "2026-06-22T00:00:00Z",
                            "updated_at_utc": "2026-06-22T00:00:01Z",
                        }
                    ),
                    encoding="utf-8",
                )
                after_failed_attempt = mobile_catalog.power_solar_evaluation(lane)
            self.assertEqual(payload["environment"], "development")
            self.assertEqual(payload["authority"], "candidate")
            self.assertEqual(payload["status"], "stale")
            self.assertEqual(payload["staleReason"], "forecast_validity_ended")
            self.assertEqual(payload["generatedAt"], "2026-06-21T12:05:00Z")
            self.assertNotEqual(payload["generatedAt"], payload["requestedAt"])
            self.assertEqual(payload["validUntil"], "2026-06-21T15:00:00Z")
            self.assertEqual(payload["pairID"], pair_id)
            self.assertEqual(payload["comparison"][1]["candidateSOC"], 69.5)
            self.assertEqual(payload["localFeatureContractID"], "issue-features-v1-test")
            self.assertIn("gfs=not_enrolled", payload["sourceAvailabilityCode"])
            self.assertEqual(payload["forecastCodeRevision"], "revision-test")
            self.assertEqual(payload["calibrationStateID"], "calibration-state-test")
            self.assertEqual(payload["physicalDrivers"]["physicalConsistencyStatus"], "passed")
            self.assertEqual(after_failed_attempt["status"], "stale")
            self.assertEqual(
                after_failed_attempt["staleReason"],
                "forecast_validity_ended",
            )
            self.assertEqual(
                after_failed_attempt["lastAttemptReason"],
                "candidate_generation_failed:ValueError",
            )
            self.assertEqual(after_failed_attempt["comparison"], payload["comparison"])
            self.assertEqual(payload["publicSourceAblations"]["ifs"]["status"], "complete")
            self.assertNotIn("path", json.dumps(payload).lower())

    def test_float_health_states_are_normalized_and_counted(self) -> None:
        self.assertEqual(mobile_catalog.normalize_level(1.0), "green")
        self.assertEqual(mobile_catalog.normalize_level(0.0), "red")

        spec = mobile_catalog.OPERATIONS_STREAMS[0]
        snapshot = {str(spec["source"]): 1.0}
        snapshot.update({str(key): 1.0 for key in spec["services"]})
        state = mobile_catalog._stream_state(snapshot, spec)

        self.assertEqual(state["level"], "green")
        self.assertEqual(state["serviceHealthyCount"], state["serviceCount"])

    def test_public_dashboard_group_uses_published_probe_keys(self) -> None:
        snapshot = {
            "dashboard_http_ok_state": 1.0,
            "failover_primary_dashboard_http_ok_state": 1.0,
            "failover_standby_dashboard_http_ok_state": 1.0,
            "archive_health_level": "green",
        }

        groups = mobile_catalog._root_cause_groups(snapshot, [])
        dashboard = next(group for group in groups if group["id"] == "dashboard")

        self.assertEqual(dashboard["level"], "green")

    def test_operations_prefers_and_merges_archive_health(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "snapshot.json"
            health = root / "health.json"
            archive = root / "archive.json"
            alerts = root / "alerts.json"
            snapshot.write_text('{"overall_level":"green","cl61_gws_missing_count":9}')
            health.write_text('{"overall_level":"green"}')
            archive.write_text(
                json.dumps(
                    {
                        "generated_at": "2026-07-25T20:00:00Z",
                        "overall_level": "red",
                        "failures": ["object_store_raw_missing=3"],
                        "metrics": {"cl61_gws_missing_count": 0},
                    }
                )
            )
            alerts.write_text("{}")
            with patch.dict(
                os.environ,
                {
                    "OPS_MONITOR_LATEST_SNAPSHOT": str(snapshot),
                    "OPS_MONITOR_LATEST_HEALTH": str(health),
                    "ARCHIVE_HEALTH_PATH": str(archive),
                    "OPS_MONITOR_ALERT_STATE": str(alerts),
                },
            ):
                response = mobile_catalog.operations()

        self.assertEqual(response["overallLevel"], "red")
        self.assertEqual(response["updatedAt"], "2026-07-25T20:00:00Z")
        archive_alert = next(
            alert
            for alert in response["alerts"]
            if alert["id"] == "archive:verification"
        )
        self.assertIn("object_store_raw_missing=3", archive_alert["detail"])
        archive_group = next(
            group
            for group in response["rootCauseGroups"]
            if group["id"] == "archive"
        )
        self.assertEqual(archive_group["level"], "red")
        self.assertEqual(response["checkCounts"]["red"], 1)
        self.assertEqual(
            response["sources"]["archiveHealth"]["path"],
            str(archive),
        )

    def test_operations_presents_clean_inventory_timeout_as_amber(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "snapshot.json"
            health = root / "health.json"
            archive = root / "archive.json"
            alerts = root / "alerts.json"
            snapshot_payload: dict[str, int | str] = {
                "time_utc": "2026-08-02T07:09:00Z"
            }
            for spec in mobile_catalog.OPERATIONS_STREAMS:
                snapshot_payload[str(spec["source"])] = 1
                for service in spec["services"]:
                    snapshot_payload[str(service)] = 1
            snapshot.write_text(json.dumps(snapshot_payload), encoding="utf-8")
            health.write_text('{"overall_level":"red"}', encoding="utf-8")
            archive.write_text(
                json.dumps(
                    {
                        "generated_at": "2026-08-02T07:09:34Z",
                        "overall_level": "red",
                        "failures": [
                            "object_store_evidence_stale_hours=9.81",
                            "archive_service_unhealthy=aurora-object-store-inventory.service",
                        ],
                        "metrics": {
                            "streams_gws_issue_count": 0,
                            "object_store_all_missing_count": 0,
                            "object_store_all_mismatch_count": 0,
                            "gws_all_missing_count": 0,
                            "gws_all_mismatch_count": 0,
                        },
                        "evidence": {
                            "object_store_gate": {
                                "clean": True,
                                "stable_parity": True,
                            },
                            "object_store_inventory_progress": {
                                "state": "failed",
                                "current_job": "products",
                                "error": (
                                    "CalledProcessError: rclone lsjson "
                                    "gamb2le-o/data/output/aurora-cloud/products/"
                                    "asfs_fast_gas"
                                ),
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            alerts.write_text("{}", encoding="utf-8")
            with patch.dict(
                os.environ,
                {
                    "OPS_MONITOR_LATEST_SNAPSHOT": str(snapshot),
                    "OPS_MONITOR_LATEST_HEALTH": str(health),
                    "ARCHIVE_HEALTH_PATH": str(archive),
                    "OPS_MONITOR_ALERT_STATE": str(alerts),
                },
            ):
                response = mobile_catalog.operations()

        archive_alert = next(
            alert
            for alert in response["alerts"]
            if alert["id"] == "archive:verification"
        )
        self.assertEqual(response["overallLevel"], "amber")
        self.assertEqual(response["checkCounts"]["amber"], 1)
        self.assertEqual(archive_alert["level"], "amber")
        self.assertEqual(archive_alert["title"], "Archive verification failed")
        self.assertEqual(
            archive_alert["detail"],
            "JASMIN object-store listing timed out for ASFS fast-gas products. "
            "Last complete verification was clean. New pruning is paused until "
            "verification succeeds.",
        )
        archive_group = next(
            group
            for group in response["rootCauseGroups"]
            if group["id"] == "archive"
        )
        self.assertEqual(archive_group["level"], "amber")
        self.assertIn(archive_alert["detail"], archive_group["detail"])
        self.assertIn("Newest-first live delivery queue is clear", archive_group["detail"])
        self.assertIn("last complete check was clean", response["summary"])

    def test_operations_uses_infrastructure_operator_status_while_audit_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "snapshot.json"
            health = root / "health.json"
            archive = root / "archive.json"
            alerts = root / "alerts.json"
            snapshot.write_text('{"time_utc":"2026-08-02T12:00:00Z"}')
            health.write_text('{"overall_level":"green"}')
            archive.write_text(
                json.dumps(
                    {
                        "generated_at": "2026-08-02T12:00:00Z",
                        "overall_level": "amber",
                        "failures": ["object_store_evidence_stale_hours=14.5"],
                        "operator_status": {
                            "level": "amber",
                            "title": "Archive verification is running",
                            "detail": (
                                "Strict audit 4 of 5 families complete; checking manifests. "
                                "Last complete verification was clean. New pruning is paused."
                            ),
                            "pruning_paused": True,
                        },
                        "metrics": {
                            "archive_delivery_pending_count": 12,
                            "archive_delivery_gws_pending_count": 0,
                            "archive_delivery_object_store_pending_count": 12,
                        },
                    }
                )
            )
            alerts.write_text(
                json.dumps(
                    {
                        "active": {
                            "archive:health_red": {
                                "active": True,
                                "title": "Archive health is red",
                                "level": "red",
                            }
                        }
                    }
                )
            )
            with patch.dict(
                os.environ,
                {
                    "OPS_MONITOR_LATEST_SNAPSHOT": str(snapshot),
                    "OPS_MONITOR_LATEST_HEALTH": str(health),
                    "ARCHIVE_HEALTH_PATH": str(archive),
                    "OPS_MONITOR_ALERT_STATE": str(alerts),
                },
            ):
                response = mobile_catalog.operations()

        self.assertEqual(response["overallLevel"], "amber")
        archive_alerts = [
            alert for alert in response["alerts"] if alert["id"].startswith("archive:")
        ]
        self.assertEqual(len(archive_alerts), 1)
        self.assertEqual(archive_alerts[0]["id"], "archive:verification")
        self.assertEqual(archive_alerts[0]["title"], "Archive verification is running")

    def test_v2_clean_audit_activity_is_status_not_alert(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "snapshot.json"
            health = root / "health.json"
            archive = root / "archive.json"
            alerts = root / "alerts.json"
            snapshot.write_text('{"time_utc":"2026-08-16T12:00:00Z"}')
            health.write_text('{"overall_level":"green"}')
            archive.write_text(
                json.dumps(
                    {
                        "schema_version": "health-v2",
                        "generated_at": "2026-08-16T12:00:00Z",
                        "overall_level": "green",
                        "operator_status": {
                            "level": "green",
                            "title": "Archive copies are healthy",
                            "detail": (
                                "Newest files are being delivered; a routine strict "
                                "audit is running in the background."
                            ),
                            "pruning_paused": True,
                        },
                        "delivery": {
                            "level": "green",
                            "mode": "newest-first",
                            "pending_files": 12,
                            "gws_pending_files": 12,
                            "object_store_pending_files": 12,
                            "oldest_pending_age_minutes": 14,
                            "last_success_age_minutes": 2,
                        },
                        "verification": {
                            "state": "running",
                            "completed_jobs": ["raw"],
                            "total_jobs": 5,
                            "current_jobs": ["products"],
                            "last_certified_raw_at": "2026-08-16T11:00:00Z",
                        },
                        "durability": {
                            "raw_retention": {"clean": True},
                            "products": {"clean": True},
                        },
                        "retention": {"ready": False, "paused": True},
                    }
                )
            )
            alerts.write_text("{}")
            with patch.dict(
                os.environ,
                {
                    "OPS_MONITOR_LATEST_SNAPSHOT": str(snapshot),
                    "OPS_MONITOR_LATEST_HEALTH": str(health),
                    "ARCHIVE_HEALTH_PATH": str(archive),
                    "OPS_MONITOR_ALERT_STATE": str(alerts),
                },
            ):
                response = mobile_catalog.operations()

        self.assertEqual(response["overallLevel"], "green")
        self.assertFalse(
            any(alert["id"].startswith("archive:") for alert in response["alerts"])
        )
        self.assertEqual(response["archiveDelivery"]["strictAudit"]["state"], "running")
        self.assertEqual(response["archiveStatus"]["retention"]["paused"], True)

    def test_power_trace_sampling_is_bounded_and_preserves_extrema(self) -> None:
        import numpy as np

        values = np.zeros(1_000)
        values[333] = 99.0
        values[777] = -42.0

        indices = mobile_catalog._representative_power_indices(values)

        self.assertLessEqual(len(indices), mobile_catalog.MOBILE_POWER_MAX_POINTS)
        self.assertIn(0, indices)
        self.assertIn(len(values) - 1, indices)
        self.assertIn(333, indices)
        self.assertIn(777, indices)

    def test_manifest_contains_native_sections_and_visible_instruments(self) -> None:
        with patch.dict(
            os.environ,
            {
                "AURORA_SITE_ENV": "development",
                "AURORA_DOMAIN": "data-ocean.gamb2le.co.uk",
                "AURORA_DASHBOARD_REVISION": "abc123def456",
            },
        ):
            manifest = mobile_catalog.manifest()

        self.assertEqual([section["id"] for section in manifest["sections"]], ["overview", "power", "plots", "camera", "ops"])
        self.assertIn("power", {instrument["id"] for instrument in manifest["instruments"]})
        power = next(instrument for instrument in manifest["instruments"] if instrument["id"] == "power")
        self.assertTrue(power["supportsHousekeepingQuicklooks"])
        self.assertIn("fish_hdr", {stream["id"] for stream in manifest["wxcamStreams"]})
        self.assertEqual(manifest["schemaVersion"], 3)
        self.assertTrue(
            {
                "power.current_system_ecmwf_p10_p90",
                "power.assigned_pdu_outlets",
                "operations.instrument_state",
            }.issubset(manifest["capabilities"]["shared"])
        )
        self.assertIn("explore.arbitrary_variables_ranges", manifest["capabilities"]["browser"])
        self.assertEqual(
            manifest["deployment"],
            {
                "environment": "development",
                "domain": "data-ocean.gamb2le.co.uk",
                "dashboardURL": "https://data-ocean.gamb2le.co.uk/app",
                "dataRole": "live-mirror",
                "revision": "abc123def456",
            },
        )

    def test_auroracam_lists_day_times_for_native_time_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "radar-cam" / "2026-07-05"
            source.mkdir(parents=True)
            for stamp in ("12-00", "12-30"):
                (source / f"radar-cam_2026-07-05_{stamp}.jpg").write_bytes(b"jpeg")

            with patch.dict(os.environ, {"AURORACAM_RAW_ROOT": str(root)}):
                response = mobile_catalog.auroracam("2026-07-05")

        self.assertEqual(response["availableTimesUTC"], ["2026-07-05 12:30", "2026-07-05 12:00"])

    def test_uas_window_is_filtered_by_the_server(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "menapia_mqtt.log"
            recent = datetime.now(timezone.utc) - timedelta(minutes=5)
            path.write_text(
                "2026-07-01 12:00:00: Tier change 1 2\n"
                f"{recent:%Y-%m-%d %H:%M:%S}: Tier change 2 3\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"UAS_MQTT_LOG_PATH": str(path)}):
                response = mobile_catalog.uas("24h")

        self.assertEqual(response["window"], "24h")
        self.assertEqual([record["effectiveTier"] for record in response["records"]], [3])
        self.assertEqual([record["dock1Tier"] for record in response["records"]], [2])
        self.assertEqual([record["dock2Tier"] for record in response["records"]], [3])
        self.assertEqual(response["records"][0]["sharedTier"], None)

    def test_uas_flight_catalog_filters_day_and_hides_product_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "flights").mkdir()
            (root / "plots").mkdir()
            (root / "plots" / "flight-new.png").write_bytes(b"flight")
            quicklooks = root / "quicklooks"
            quicklooks.mkdir()
            (quicklooks / "uas__summary__20260828.png").write_bytes(b"daily")
            catalog = {
                "schemaVersion": 1,
                "generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "lastRunState": "success",
                "latestFlightID": "flight-new",
                "availableDays": ["2026-08-28", "2026-08-27"],
                "flights": [
                    {
                        "id": "flight-new",
                        "sourceFlightID": "source-new",
                        "dayUTC": "2026-08-28",
                        "flightNumber": 2,
                        "title": "Flight 2",
                        "startTimeUTC": "2026-08-28T12:00:00Z",
                        "endTimeUTC": "2026-08-28T12:30:00Z",
                        "durationSeconds": 1800,
                        "samplePeriodSeconds": 1,
                        "modifiedAt": "2026-08-28T12:31:00Z",
                        "quality": {"level": "green", "warnings": []},
                        "detailPath": "flights/flight-new.json",
                        "plotPath": "plots/flight-new.png",
                    },
                    {
                        "id": "flight-old",
                        "dayUTC": "2026-08-27",
                        "title": "Flight 1",
                        "startTimeUTC": "2026-08-27T10:00:00Z",
                        "quality": {"level": "amber", "warnings": ["short capture"]},
                    },
                ],
            }
            (root / "catalog.json").write_text(json.dumps(catalog), encoding="utf-8")
            with patch.dict(
                os.environ,
                {"MENAPIA_PRODUCT_ROOT": str(root), "UAS_QUICKLOOK_DIR": str(quicklooks)},
                clear=False,
            ):
                response = mobile_catalog.uas_flights("latest")

        self.assertEqual(response["selectedDay"], "2026-08-28")
        self.assertEqual(response["status"]["state"], "fresh")
        self.assertEqual([flight["id"] for flight in response["flights"]], ["flight-new"])
        self.assertTrue(response["flights"][0]["plotURL"].startswith("/media/uas/flights/flight-new?v="))
        self.assertNotIn("detailPath", response["flights"][0])
        self.assertNotIn("plotPath", response["flights"][0])
        self.assertEqual(response["dailyQuicklook"]["token"], "20260828")
        self.assertTrue(response["allFlightsPlotURL"].startswith("/media/quicklook/science/uas/20260828?v="))

    def test_uas_mutable_media_urls_change_with_file_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plots = root / "plots"
            quicklooks = root / "quicklooks"
            plots.mkdir()
            quicklooks.mkdir()
            plot = plots / "flight-1.png"
            daily = quicklooks / "uas__summary__20260828.png"
            plot.write_bytes(b"v1")
            daily.write_bytes(b"daily-v1")
            catalog = {
                "schemaVersion": 1,
                "generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "lastRunState": "success",
                "latestFlightID": "flight-1",
                "availableDays": ["2026-08-28"],
                "flights": [
                    {
                        "id": "flight-1",
                        "dayUTC": "2026-08-28",
                        "title": "Flight 1",
                        "startTimeUTC": "2026-08-28T10:00:00Z",
                        "quality": {"level": "green", "warnings": []},
                    }
                ],
            }
            (root / "catalog.json").write_text(json.dumps(catalog), encoding="utf-8")
            env = {"MENAPIA_PRODUCT_ROOT": str(root), "UAS_QUICKLOOK_DIR": str(quicklooks)}
            with patch.dict(os.environ, env, clear=False):
                first = mobile_catalog.uas_flights()
                plot.write_bytes(b"version-two")
                daily.write_bytes(b"daily-version-two")
                os.utime(plot, ns=(2_000_000_000, 2_000_000_000))
                os.utime(daily, ns=(2_000_000_000, 2_000_000_000))
                second = mobile_catalog.uas_flights()

        self.assertNotEqual(first["flights"][0]["plotURL"], second["flights"][0]["plotURL"])
        self.assertNotEqual(first["allFlightsPlotURL"], second["allFlightsPlotURL"])
        self.assertEqual(second["dailyQuicklook"]["imageURL"], second["allFlightsPlotURL"])

    def test_uas_flight_detail_validates_equal_bounded_series(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "flights").mkdir()
            (root / "plots").mkdir()
            metadata = {
                "id": "flight-1",
                "dayUTC": "2026-08-28",
                "title": "Flight 1",
                "startTimeUTC": "2026-08-28T10:00:00Z",
                "quality": {"level": "green", "warnings": []},
            }
            (root / "catalog.json").write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "generatedAt": "2026-08-28T10:30:00Z",
                        "lastRunState": "success",
                        "latestFlightID": "flight-1",
                        "availableDays": ["2026-08-28"],
                        "flights": [metadata],
                    }
                ),
                encoding="utf-8",
            )
            series = {
                "timeUTC": ["2026-08-28T10:00:00Z", "2026-08-28T10:00:01Z"],
                "temperatureC": {"SN0122": [8.0, 8.1], "SN0123": [8.2, None]},
                "pressureHpa": {"SN0122": [1005.0, 1005.1], "SN0123": [1006.0, 1006.1]},
                "relativeHumidityPct": {"SN0122": [75.0, 76.0], "SN0123": [74.0, 75.0]},
                "altitudeM": [-60.0, 48.0],
            }
            (root / "flights" / "flight-1.json").write_text(
                json.dumps({"schemaVersion": 1, "flight": metadata, "series": series}),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"MENAPIA_PRODUCT_ROOT": str(root)}, clear=False):
                response = mobile_catalog.uas_flight("flight-1")
                with self.assertRaises(KeyError):
                    mobile_catalog.uas_flight("../catalog")

        self.assertEqual(response["series"]["altitudeM"], [-60.0, 48.0])
        self.assertEqual(response["series"]["temperatureC"]["SN0123"], [8.2, None])
        self.assertNotIn("detailPath", response["flight"])

    def test_uas_flight_catalog_reports_missing_and_partial_states(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.dict(os.environ, {"MENAPIA_PRODUCT_ROOT": str(root)}, clear=False):
                missing = mobile_catalog.uas_flights()
            (root / "catalog.json").write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                        "lastRunState": "partial_failure",
                        "latestFlightID": "flight-1",
                        "availableDays": ["2026-08-28"],
                        "flights": [
                            {
                                "id": "flight-1",
                                "dayUTC": "2026-08-28",
                                "title": "Flight 1",
                                "startTimeUTC": "2026-08-28T10:00:00Z",
                                "quality": {"level": "amber", "warnings": ["source incomplete"]},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"MENAPIA_PRODUCT_ROOT": str(root)}, clear=False):
                partial = mobile_catalog.uas_flights()

        self.assertEqual(missing["status"]["state"], "error")
        self.assertEqual(partial["status"]["state"], "partial")

    def test_uas_freshness_prefers_optional_builder_heartbeat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            status_path = root / "internal/menapia-products/status.json"
            status_path.parent.mkdir(parents=True)
            old_catalog_time = (
                datetime.now(timezone.utc) - timedelta(hours=3)
            ).isoformat().replace("+00:00", "Z")
            completed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            (root / "catalog.json").write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "generatedAt": old_catalog_time,
                        "lastRunState": "success",
                        "latestFlightID": "flight-1",
                        "availableDays": ["2026-08-28"],
                        "flights": [
                            {
                                "id": "flight-1",
                                "dayUTC": "2026-08-28",
                                "title": "Flight 1",
                                "startTimeUTC": "2026-08-28T10:00:00Z",
                                "quality": {"level": "green", "warnings": []},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            status_path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "state": "success",
                        "completedAt": completed_at,
                        "catalogGeneratedAt": old_catalog_time,
                    }
                ),
                encoding="utf-8",
            )
            env = {
                "MENAPIA_PRODUCT_ROOT": str(root),
                "MENAPIA_PRODUCT_STATUS_PATH": str(status_path),
            }
            with patch.dict(os.environ, env, clear=False):
                fresh = mobile_catalog.uas_flights()
                status_path.write_text(
                    json.dumps(
                        {
                            "schemaVersion": 1,
                            "state": "failed",
                            "completedAt": completed_at,
                            "error": "/private/source/path must not be exposed",
                        }
                    ),
                    encoding="utf-8",
                )
                failed = mobile_catalog.uas_flights()

            self.assertEqual(fresh["generatedAt"], old_catalog_time)
            self.assertEqual(fresh["lastRunAt"], completed_at)
            self.assertEqual(fresh["lastRunState"], "success")
            self.assertEqual(fresh["status"]["state"], "fresh")
            self.assertEqual(failed["lastRunState"], "failed")
            self.assertEqual(failed["status"]["state"], "error")
            self.assertNotIn("/private/source/path", json.dumps(failed))
            self.assertEqual([item["id"] for item in failed["flights"]], ["flight-1"])

    def test_uas_malformed_heartbeat_falls_back_to_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            status_path = root / "status.json"
            status_path.write_text("not-json", encoding="utf-8")
            generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            (root / "catalog.json").write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "generatedAt": generated_at,
                        "lastRunState": "success",
                        "availableDays": [],
                        "flights": [],
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {
                    "MENAPIA_PRODUCT_ROOT": str(root),
                    "MENAPIA_PRODUCT_STATUS_PATH": str(status_path),
                },
                clear=False,
            ):
                response = mobile_catalog.uas_flights()

        self.assertEqual(response["lastRunAt"], generated_at)
        self.assertEqual(response["lastRunState"], "success")
        self.assertEqual(response["status"]["state"], "empty")

    def test_uas_science_quicklooks_use_dedicated_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "uas__summary__20260828.png").write_bytes(b"dated")
            (root / "uas__summary__latest.png").write_bytes(b"latest")
            with patch.dict(os.environ, {"UAS_QUICKLOOK_DIR": str(root)}, clear=False):
                response = mobile_catalog.quicklooks("science", "uas")
                resolved = mobile_catalog.resolve_quicklook_path("science", "uas", "latest")

        self.assertEqual([entry["token"] for entry in response["entries"]], ["latest", "20260828"])
        self.assertTrue(response["entries"][0]["imageURL"].startswith("/media/quicklook/science/uas/latest?v="))
        self.assertEqual(resolved.name, "uas__summary__latest.png")

    def test_shared_pdu_contract_has_only_assigned_outlets(self) -> None:
        self.assertEqual(
            [(title, outlet) for _, title, _, outlet in mobile_catalog.PDU_INSTRUMENTS],
            [("UAS", 4), ("CL61", 5), ("Cloud Radar", 6), ("HATPRO", 8)],
        )

    def test_quicklooks_reports_assigned_instrument_power_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdu = root / "pdu.zarr"
            import numpy as np
            import xarray as xr

            xr.Dataset(
                {"PDUOutlet5State": (("time",), np.array([0.0]))},
                coords={"time": [datetime.now(timezone.utc).replace(tzinfo=None)]},
            ).to_zarr(pdu, mode="w")
            with patch.dict(os.environ, {"AURORA_QUICKLOOK_ROOT": str(root), "PDU_ZARR_PATH": str(pdu)}):
                response = mobile_catalog.quicklooks("science", "ceilometer")

        self.assertEqual(response["powerStatus"]["state"], "Off")
        self.assertIn("PDU sample", response["powerStatus"]["detail"])

    def test_quicklooks_find_latest_and_dated_summary_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            quicklook_dir = root / "power"
            quicklook_dir.mkdir()
            (quicklook_dir / "power__summary__latest.png").write_bytes(b"latest")
            (quicklook_dir / "power__summary__20260705.png").write_bytes(b"dated")

            with patch.dict(os.environ, {"AURORA_QUICKLOOK_ROOT": str(root)}):
                response = mobile_catalog.quicklooks("science", "power")

        self.assertEqual(response["latest"]["token"], "latest")
        self.assertEqual([entry["token"] for entry in response["entries"]], ["latest", "20260705"])

    def test_science_radar_quicklooks_exclude_housekeeping_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            quicklook_dir = root / "cloud_radar"
            quicklook_dir.mkdir()
            science = quicklook_dir / "latest.png"
            housekeeping = quicklook_dir / "cloud_radar__hk_radar__latest.png"
            science.write_bytes(b"science")
            housekeeping.write_bytes(b"housekeeping")

            with patch.dict(os.environ, {"AURORA_QUICKLOOK_ROOT": str(root)}):
                response = mobile_catalog.quicklooks("science", "cloud-radar")
                resolved = mobile_catalog.resolve_quicklook_path("science", "cloud-radar", "latest")

        self.assertEqual(response["latest"]["token"], "latest")
        self.assertEqual(resolved, science)

    def test_science_latest_uses_newer_dated_product_when_alias_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            quicklook_dir = root / "ceilometer"
            quicklook_dir.mkdir()
            stale_latest = quicklook_dir / "latest.png"
            fresh_daily = quicklook_dir / "ceilometer_20260716.png"
            stale_latest.write_bytes(b"stale")
            fresh_daily.write_bytes(b"fresh")
            os.utime(stale_latest, (1, 1))
            os.utime(fresh_daily, (2, 2))

            with patch.dict(os.environ, {"AURORA_QUICKLOOK_ROOT": str(root)}):
                response = mobile_catalog.quicklooks("science", "ceilometer")
                resolved = mobile_catalog.resolve_quicklook_path("science", "ceilometer", "latest")

        self.assertEqual(resolved, fresh_daily)
        self.assertEqual(response["latest"]["title"], "Latest available (2026-07-16)")

    def test_operations_derives_stream_levels_from_snapshot(self) -> None:
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
            alerts.write_text(json.dumps({"active": {"a": {"title": "Storage high", "level": "red"}}}), encoding="utf-8")

            with patch.dict(
                os.environ,
                {
                    "OPS_MONITOR_LATEST_SNAPSHOT": str(snapshot),
                    "OPS_MONITOR_LATEST_HEALTH": str(health),
                    "OPS_MONITOR_ALERT_STATE": str(alerts),
                },
            ):
                response = mobile_catalog.operations()

        ceilometer = next(stream for stream in response["streamStates"] if stream["id"] == "ceilometer")
        self.assertEqual(response["overallLevel"], "red")
        self.assertEqual(ceilometer["level"], "red")
        self.assertEqual(response["alerts"][0]["title"], "Storage high")

    def test_operations_uses_current_soc_thresholds(self) -> None:
        self.assertEqual(mobile_catalog._trend_level("battery-soc", 40), "red")
        self.assertEqual(mobile_catalog._trend_level("battery-soc", 45), "amber")
        self.assertEqual(mobile_catalog._trend_level("battery-soc", 50), "green")

    def test_operations_trend_cards_use_current_aps_metric_names(self) -> None:
        snapshot = {
            "host_aps_data_used_pct": 74.0,
            "aurora_data_used_pct": 77.0,
            "aps_battery_soc_pct": 100.0,
            "aps_battery_voltage_v": 55.6,
        }
        with patch.object(mobile_catalog, "_intentionally_paused_streams", return_value=set()), patch.object(
            mobile_catalog, "_operations_trend_values", return_value={}
        ):
            cards = {card["id"]: card for card in mobile_catalog._trend_cards(snapshot)}

        self.assertEqual(cards["storage"]["value"], 77.0)
        self.assertEqual(cards["battery-soc"]["value"], 100.0)
        self.assertEqual(cards["battery-voltage"]["value"], 55.6)

    def test_operations_trend_history_excludes_intentionally_paused_streams(self) -> None:
        import numpy as np
        import pandas as pd
        import xarray as xr

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ops_monitor.zarr"
            times = pd.date_range(
                datetime.now(timezone.utc) - timedelta(hours=1), periods=3, freq="30min"
            ).tz_localize(None)
            xr.Dataset(
                {
                    "aurora_data_used_pct": (("time",), np.array([75.0, 76.0, 77.0])),
                    "aps_battery_soc_pct": (("time",), np.array([98.0, 99.0, 100.0])),
                    "aps_battery_voltage_v": (("time",), np.array([55.0, 55.3, 55.6])),
                    "cl61_source_age_min": (("time",), np.array([2.0, 3.0, 4.0])),
                    "radar_source_age_min": (("time",), np.array([4.0, 6.0, 8.0])),
                    "hatpro_source_age_min": (("time",), np.array([700.0, 800.0, 900.0])),
                    "cl61_gws_lag_min": (("time",), np.array([1.0, 2.0, 3.0])),
                    "radar_gws_lag_min": (("time",), np.array([2.0, 4.0, 6.0])),
                    "hatpro_gws_lag_min": (("time",), np.array([70.0, 80.0, 90.0])),
                },
                coords={"time": times},
            ).to_zarr(path, mode="w", consolidated=True)
            mobile_catalog._OPERATIONS_TREND_CACHE.clear()
            with patch.dict(os.environ, {"OPS_MONITOR_ZARR_PATH": str(path)}):
                values = mobile_catalog._operations_trend_values({"hatpro"})

        self.assertEqual(values["storage"], 77.0)
        self.assertEqual(values["battery-soc"], 100.0)
        self.assertEqual(values["battery-voltage"], 55.6)
        self.assertEqual(values["source-lag"], 8.0)
        self.assertEqual(values["gws-lag"], 6.0)

    def test_operations_trend_does_not_reuse_an_old_finite_source_lag(self) -> None:
        import numpy as np
        import pandas as pd
        import xarray as xr

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ops_monitor.zarr"
            times = pd.date_range(
                datetime.now(timezone.utc) - timedelta(minutes=10),
                periods=2,
                freq="10min",
            ).tz_localize(None)
            xr.Dataset(
                {
                    "cl61_source_age_min": (
                        ("time",),
                        np.array([12_760.2, np.nan]),
                    ),
                    "cl61_gws_lag_min": (("time",), np.array([4.0, 5.6])),
                },
                coords={"time": times},
            ).to_zarr(path, mode="w", consolidated=True)
            mobile_catalog._OPERATIONS_TREND_CACHE.clear()
            with patch.dict(os.environ, {"OPS_MONITOR_ZARR_PATH": str(path)}):
                values = mobile_catalog._operations_trend_values(set())

        self.assertNotIn("source-lag", values)
        self.assertEqual(values["gws-lag"], 5.6)

    def test_operations_excludes_recovered_alert_history(self) -> None:
        alerts = mobile_catalog._active_alerts(
            {
                "alerts": {
                    "active": {"active": True, "title": "Current condition"},
                    "recovered": {"active": False, "title": "Old condition"},
                }
            }
        )

        self.assertEqual([alert["title"] for alert in alerts], ["Current condition"])

    def test_auroracam_latest_listing_and_media_resolver(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "radar-cam" / "2026-07-05"
            source.mkdir(parents=True)
            image = source / "radar-cam_2026-07-05_12-30.jpg"
            image.write_bytes(b"jpeg")

            with patch.dict(os.environ, {"AURORACAM_RAW_ROOT": str(root)}):
                response = mobile_catalog.auroracam()
                resolved = mobile_catalog.resolve_auroracam_image_path("radar-cam", "2026-07-05", image.name)

        self.assertEqual(response["selectedDay"], "2026-07-05")
        self.assertEqual(response["frames"][0]["previewURL"], "/media/auroracam/preview/radar-cam/2026-07-05/radar-cam_2026-07-05_12-30.jpg")
        self.assertEqual(resolved, image)

    def test_power_returns_a_small_unavailable_payload_without_creating_products(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.zarr"
            with patch.dict(os.environ, {"POWER_DISPLAY_SUMMARY_ZARR_PATH": str(missing)}):
                response = mobile_catalog.power()

        self.assertEqual(response["panels"], [])
        self.assertEqual(response["group"], "all")
        self.assertIn("warning", response)

    def test_forecast_panels_start_at_their_first_forecast_time(self) -> None:
        import numpy as np
        import pandas as pd
        import xarray as xr

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "power_display_summary.zarr"
            times = pd.date_range("2026-07-19T07:00:00", periods=31, freq="1h")
            solar = np.full(len(times), np.nan)
            solar[24:] = [150.0, 350.0, 500.0, 300.0, 100.0, 0.0, 0.0]
            xr.Dataset(
                {
                    "ForecastSolarWatts": (("time",), solar),
                    "OperatingCurrentLoadP50Watts": (("time",), np.full(len(times), 250.0)),
                },
                coords={"time": times},
            ).to_zarr(path, mode="w")
            with patch.dict(os.environ, {"POWER_DISPLAY_SUMMARY_ZARR_PATH": str(path)}), patch.object(
                mobile_catalog, "datetime", wraps=datetime
            ) as mocked_datetime:
                mocked_datetime.now.return_value = datetime(2026, 7, 20, 7, tzinfo=timezone.utc)
                response = mobile_catalog.power(window="24h", group="forecast_96h")

        panel = next(panel for panel in response["panels"] if panel["id"] == "ecmwf_solar_forecast")
        self.assertEqual(panel["info"]["title"], "ECMWF solar and load forecast")
        self.assertTrue(panel["info"]["implementation"])
        self.assertEqual(panel["forecastContext"]["horizonHours"], 96)
        self.assertEqual(panel["forecastContext"]["validTime"], "2026-07-20T13:00:00Z")
        for trace in panel["traces"]:
            self.assertTrue(all(point["time"] >= "2026-07-20T07:00:00" for point in trace["points"]))

    def test_forecast_group_merges_latest_uas_tier3_operating_scenario(self) -> None:
        import numpy as np
        import pandas as pd
        import xarray as xr

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            display_path = root / "power_forecast_display.zarr"
            scenario_path = root / "power_operating_scenarios.zarr"
            times = pd.date_range("2026-07-20T07:00:00", periods=4, freq="3h")
            xr.Dataset(
                {
                    "BatterySOCForecastP50": (("time",), [80.0, 78.0, 76.0, 74.0]),
                    "OperatingSuggested1SOCP50": (("time",), [99.0, 98.0, 97.0, 96.0]),
                },
                coords={"time": times},
                attrs={"forecast_initial_soc_time": times[0].isoformat()},
            ).to_zarr(display_path, mode="w", consolidated=True)
            scenario_ids = np.asarray(["suggested_all_uas_tier3"], dtype=str)
            xr.Dataset(
                {
                    "ScenarioSOCP50": (("scenario", "time"), [[80.0, 72.0, 64.0, 56.0]]),
                    "scenario_label": (("scenario",), ["All instruments + UAS tier 3 (provisional)"]),
                },
                coords={"scenario": scenario_ids, "time": times},
                attrs={
                    "planning_status": "ready",
                    "initial_soc_time": times[0].isoformat(),
                },
            ).to_zarr(scenario_path, mode="w", consolidated=True)
            with patch.dict(
                os.environ,
                {
                    "POWER_FORECAST_DISPLAY_ZARR_PATH": str(display_path),
                    "POWER_OPERATING_SCENARIOS_ZARR_PATH": str(scenario_path),
                },
            ), patch.object(mobile_catalog, "datetime", wraps=datetime) as mocked_datetime:
                mocked_datetime.now.return_value = datetime(2026, 7, 20, 7, tzinfo=timezone.utc)
                response = mobile_catalog.power(window="96h", group="forecast")

        panel = next(panel for panel in response["panels"] if panel["id"] == "operating_plan_scenarios")
        traces = {trace["id"]: trace for trace in panel["traces"]}
        self.assertNotIn("OperatingSuggested1SOCP50", traces)
        self.assertEqual(
            [point["value"] for point in traces["OperatingSuggested8SOCP50"]["points"]],
            [80.0, 72.0, 64.0, 56.0],
        )

    def test_unsafe_cl61_fallback_is_explained_in_mobile_power_payload(self) -> None:
        import numpy as np
        import pandas as pd
        import xarray as xr

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            display_path = root / "power_forecast_display.zarr"
            scenario_path = root / "power_operating_scenarios.zarr"
            anchor = pd.Timestamp("2026-07-20T07:00:00")
            times = pd.date_range(anchor, periods=5, freq="1h")
            xr.Dataset(
                {"BatterySOCForecastP50": (("time",), np.linspace(80.0, 76.0, len(times)))},
                coords={"time": times},
                attrs={"forecast_initial_soc_time": anchor.isoformat()},
            ).to_zarr(display_path, mode="w", consolidated=True)
            xr.Dataset(
                {
                    "ScenarioModeCode": (("scenario", "time"), np.full((1, len(times)), 6)),
                    "ScenarioSOCP50": (("scenario", "time"), np.linspace(80.0, 20.0, len(times))[None, :]),
                    "scenario_label": (("scenario",), ["Optimized CL61 Schedule"]),
                },
                coords={"scenario": ["optimized_cl61"], "time": times},
                attrs={
                    "planning_status": "ready",
                    "initial_soc_time": anchor.isoformat(),
                    "optimized_safe": "false",
                    "optimized_collection_hours": "0",
                    "optimized_status": "no_safe_schedule",
                    "optimized_reason": (
                        "Radar and HATPRO remain fixed on. The zero trace is an unsafe fallback, "
                        "not a recommendation to switch CL61 off."
                    ),
                    "optimized_base_mode_label": "DC + Radar + HATPRO",
                },
            ).to_zarr(scenario_path, mode="w", consolidated=True)
            with patch.dict(
                os.environ,
                {
                    "POWER_FORECAST_DISPLAY_ZARR_PATH": str(display_path),
                    "POWER_OPERATING_SCENARIOS_ZARR_PATH": str(scenario_path),
                },
            ), patch.object(mobile_catalog, "datetime", wraps=datetime) as mocked_datetime:
                mocked_datetime.now.return_value = datetime(2026, 7, 20, 7, tzinfo=timezone.utc)
                response = mobile_catalog.power(window="96h", group="forecast_96h")

        panel = next(panel for panel in response["panels"] if panel["id"] == "operating_plan_schedule")
        self.assertEqual(panel["title"], "No Feasible CL61 Schedule")
        traces = {trace["id"]: trace for trace in panel["traces"]}
        self.assertEqual(
            traces["OperatingCL61OptimizedCL61On"]["label"],
            "Unsafe fallback (CL61 off)",
        )
        self.assertIn("not a recommendation", panel["explanation"])
        self.assertEqual(panel["info"]["title"], "No Feasible CL61 Schedule")
        self.assertTrue(traces["OperatingCL61OptimizedCL61On"]["step"])
        self.assertEqual(
            traces["OperatingCL61OptimizedCL61On"]["stepAlignment"],
            "intervalEnd",
        )

    def test_priority_schedule_exposes_additive_sum_and_three_instrument_traces(self) -> None:
        import numpy as np
        import pandas as pd
        import xarray as xr

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            display_path = root / "power_forecast_display.zarr"
            scenario_path = root / "power_operating_scenarios.zarr"
            anchor = pd.Timestamp("2026-07-20T07:00:00")
            times = pd.date_range(anchor, periods=5, freq="1h")
            xr.Dataset(
                {"BatterySOCForecastP50": (("time",), np.linspace(80.0, 76.0, len(times)))},
                coords={"time": times},
                attrs={"forecast_initial_soc_time": anchor.isoformat()},
            ).to_zarr(display_path, mode="w", consolidated=True)
            xr.Dataset(
                {
                    "ScenarioModeCode": (("scenario", "time"), np.asarray([[0, 1, 3, 7, 0]])),
                    "ScenarioSOCP50": (("scenario", "time"), np.linspace(80.0, 76.0, len(times))[None, :]),
                    "scenario_label": (("scenario",), ["Priority Instrument Schedule"]),
                },
                coords={"scenario": ["optimized_cl61"], "time": times},
                attrs={
                    "planning_status": "ready",
                    "initial_soc_time": anchor.isoformat(),
                    "optimized_safe": "true",
                    "optimized_collection_hours": "3",
                    "optimized_status": "safe_schedule",
                    "optimized_priority_order": '["CL61", "Radar", "HATPRO"]',
                    "optimized_reason": "A safe advisory priority schedule keeps P10 SOC above 40%.",
                },
            ).to_zarr(scenario_path, mode="w", consolidated=True)
            with patch.dict(
                os.environ,
                {
                    "POWER_FORECAST_DISPLAY_ZARR_PATH": str(display_path),
                    "POWER_OPERATING_SCENARIOS_ZARR_PATH": str(scenario_path),
                },
            ), patch.object(mobile_catalog, "datetime", wraps=datetime) as mocked_datetime:
                mocked_datetime.now.return_value = datetime(2026, 7, 20, 7, tzinfo=timezone.utc)
                response = mobile_catalog.power(window="96h", group="forecast_96h")

        panel = next(panel for panel in response["panels"] if panel["id"] == "operating_plan_schedule")
        self.assertEqual(panel["title"], "Recommended Additive Instrument Schedule")
        traces = {trace["id"]: trace for trace in panel["traces"]}
        self.assertEqual(
            [point["value"] for point in traces["OperatingCL61OptimizedActiveCount"]["points"]],
            [0.0, 1.0, 2.0, 3.0, 0.0],
        )
        self.assertEqual(
            [point["value"] for point in traces["OperatingCL61OptimizedCL61On"]["points"]],
            [0.0, 1.0, 1.0, 1.0, 0.0],
        )
        self.assertEqual(
            [point["value"] for point in traces["OperatingCL61OptimizedRadarOn"]["points"]],
            [0.0, 0.0, 1.0, 1.0, 0.0],
        )
        self.assertEqual(
            [point["value"] for point in traces["OperatingCL61OptimizedHATPROOn"]["points"]],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        )
        self.assertEqual(
            panel["leftAxisLabel"],
            "Additive activity (sum 0-3; each instrument 0/1)",
        )

    def test_96_hour_system_and_scenario_cards_share_anchor_values_and_endpoint(self) -> None:
        import numpy as np
        import pandas as pd
        import xarray as xr

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            display_path = root / "power_forecast_display.zarr"
            scenario_path = root / "power_operating_scenarios.zarr"
            anchor = pd.Timestamp("2026-07-20T07:00:00")
            times = pd.date_range(anchor, periods=121, freq="1h")
            fallback = np.linspace(90.0, 50.0, len(times))
            xr.Dataset(
                {
                    "BatterySOCForecastP10": (("time",), fallback - 5.0),
                    "BatterySOCForecastP50": (("time",), fallback),
                    "BatterySOCForecastP90": (("time",), fallback + 5.0),
                    "BatterySOCBelow40Probability": (("time",), np.zeros(len(times))),
                },
                coords={"time": times},
                attrs={
                    "forecast_initial_soc_time": anchor.isoformat(),
                    "forecast_generated_at_utc": "2026-07-20T07:05:00+00:00",
                },
            ).to_zarr(display_path, mode="w", consolidated=True)
            current = np.linspace(90.0, 30.0, len(times))
            xr.Dataset(
                {
                    "ScenarioSOCP10": (("scenario", "time"), [current - 4.0]),
                    "ScenarioSOCP50": (("scenario", "time"), [current]),
                    "ScenarioSOCP90": (("scenario", "time"), [current + 4.0]),
                    "ScenarioBelow40Probability": (
                        ("scenario", "time"),
                        [(current < 40.0).astype(float)],
                    ),
                    "scenario_label": (("scenario",), ["Current: DC + CL61"]),
                },
                coords={"scenario": ["current_mode"], "time": times},
                attrs={
                    "planning_status": "ready",
                    "initial_soc_time": anchor.isoformat(),
                    "generated_at_utc": "2026-07-20T07:10:00+00:00",
                    "operating_decision_horizon_hours": "96",
                    "planning_forecast_initial_soc_time": "2026-07-19T20:00:00",
                },
            ).to_zarr(scenario_path, mode="w", consolidated=True)
            with patch.dict(
                os.environ,
                {
                    "POWER_FORECAST_DISPLAY_ZARR_PATH": str(display_path),
                    "POWER_OPERATING_SCENARIOS_ZARR_PATH": str(scenario_path),
                },
            ), patch.object(mobile_catalog, "datetime", wraps=datetime) as mocked_datetime:
                mocked_datetime.now.return_value = datetime(2026, 7, 20, 7, tzinfo=timezone.utc)
                response = mobile_catalog.power(window="96h", group="forecast")

        near_term = next(panel for panel in response["panels"] if panel["id"] == "soc_24h_forecast")
        system = next(panel for panel in response["panels"] if panel["id"] == "soc_ecmwf_forecast")
        scenarios = next(panel for panel in response["panels"] if panel["id"] == "operating_plan_scenarios")
        near_term_central = next(
            trace for trace in near_term["traces"] if trace["id"] == "SystemAsIsDecisionSOCP50"
        )
        system_central = next(
            trace for trace in system["traces"] if trace["id"] == "SystemAsIsDecisionSOCP50"
        )
        system_p10 = next(
            trace for trace in system["traces"] if trace["id"] == "SystemAsIsDecisionSOCP10"
        )
        system_p90 = next(
            trace for trace in system["traces"] if trace["id"] == "SystemAsIsDecisionSOCP90"
        )
        scenario_current = next(
            trace for trace in scenarios["traces"] if trace["id"] == "OperatingCurrentSOCP50"
        )
        probability = next(
            trace
            for trace in system["traces"]
            if trace["id"] == "SystemAsIsDecisionBelow40Probability"
        )
        self.assertEqual(system_central["points"], scenario_current["points"])
        self.assertNotEqual(system_p10["color"], system_p90["color"])
        self.assertNotEqual(
            system_p10["dash"] or "solid",
            system_p90["dash"] or "solid",
        )
        self.assertEqual(probability["unit"], "%")
        self.assertEqual(
            near_term_central["points"],
            [
                point
                for point in system_central["points"]
                if point["time"] <= "2026-07-21T07:00:00Z"
            ],
        )
        self.assertEqual(near_term["forecastContext"]["anchorTime"], anchor.isoformat())
        self.assertEqual(near_term["forecastContext"]["horizonHours"], 24)
        self.assertEqual(near_term["forecastContext"]["validTime"], "2026-07-21T07:00:00Z")
        self.assertEqual(system["forecastContext"]["anchorTime"], anchor.isoformat())
        self.assertEqual(scenarios["forecastContext"]["anchorTime"], anchor.isoformat())
        self.assertEqual(system["forecastContext"]["horizonHours"], 96)
        self.assertEqual(system["forecastContext"]["validTime"], "2026-07-24T07:00:00Z")
        self.assertEqual(
            system["forecastContext"]["validTime"],
            scenarios["forecastContext"]["validTime"],
        )

    def test_overview_matches_browser_mobile_card_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "latest.json"
            health = root / "latest_health.json"
            alerts = root / "state.json"
            snapshot.write_text(
                json.dumps(
                    {
                        "time_utc": "2026-07-05T07:30:00Z",
                        "aps_battery_soc_pct": 56,
                        "aps_battery_voltage_v": 52.56,
                        "aps_battery_power_w": -57,
                        "aps_battery_capacity_kwh": 26,
                        "aps_battery_depletion_hours": 255,
                        "power_latest_time_utc": "2026-07-05T07:29:00Z",
                    }
                ),
                encoding="utf-8",
            )
            health.write_text(json.dumps({"overall_level": "amber"}), encoding="utf-8")
            alerts.write_text(json.dumps({"active": {}}), encoding="utf-8")
            with patch.dict(
                os.environ,
                {
                    "OPS_MONITOR_LATEST_SNAPSHOT": str(snapshot),
                    "OPS_MONITOR_LATEST_HEALTH": str(health),
                    "OPS_MONITOR_ALERT_STATE": str(alerts),
                    "AURORACAM_RAW_ROOT": str(root / "camera"),
                },
            ), patch.object(
                mobile_catalog,
                "_environmental_signal_cards",
                return_value=[
                    {"id": "shortwave-down"},
                    {"id": "wind-speed"},
                    {"id": "air-temperature"},
                    {"id": "kt15"},
                ],
            ), patch.object(
                mobile_catalog,
                "datetime",
                wraps=datetime,
            ) as mocked_datetime:
                mocked_datetime.now.return_value = datetime(
                    2026, 7, 5, 7, 30, tzinfo=timezone.utc
                )
                response = mobile_catalog.overview()

        self.assertEqual(
            [card["id"] for card in response["cards"]],
            [
                "operations", "battery-soc", "battery-voltage", "battery-depletion", "power", "auroracam",
                "shortwave-down", "wind-speed", "air-temperature", "kt15",
            ],
        )
        depletion = response["cards"][3]
        self.assertEqual(depletion["value"], "10d 15h")
        self.assertIn("14.6 kWh remaining", depletion["detail"])

    def test_overview_propagates_stale_power_into_health_and_alerts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "latest.json"
            health = root / "latest_health.json"
            archive = root / "archive.json"
            alerts = root / "state.json"
            snapshot_payload: dict[str, int | float | str] = {
                "time_utc": "2026-08-17T20:36:55Z",
                "aps_battery_soc_pct": 100.0,
                "aps_battery_soc_age_min": 133.0,
                "aps_battery_voltage_v": 55.9,
                "aps_battery_voltage_age_min": 133.0,
                "aps_battery_power_w": 16.0,
                "power_latest_time_utc": "2026-08-17T18:29:31Z",
            }
            for spec in mobile_catalog.OPERATIONS_STREAMS:
                snapshot_payload[str(spec["source"])] = 1
                for service in spec["services"]:
                    snapshot_payload[str(service)] = 1
            snapshot.write_text(json.dumps(snapshot_payload), encoding="utf-8")
            health.write_text('{"overall_level":"green"}', encoding="utf-8")
            archive.write_text("{}", encoding="utf-8")
            alerts.write_text('{"active":{}}', encoding="utf-8")
            with patch.dict(
                os.environ,
                {
                    "OPS_MONITOR_LATEST_SNAPSHOT": str(snapshot),
                    "OPS_MONITOR_LATEST_HEALTH": str(health),
                    "ARCHIVE_HEALTH_PATH": str(archive),
                    "OPS_MONITOR_ALERT_STATE": str(alerts),
                    "AURORACAM_RAW_ROOT": str(root / "camera"),
                },
            ), patch.object(
                mobile_catalog,
                "_environmental_signal_cards",
                return_value=[],
            ), patch.object(
                mobile_catalog,
                "datetime",
                wraps=datetime,
            ) as mocked_datetime:
                mocked_datetime.now.return_value = datetime(
                    2026, 8, 17, 20, 42, 31, tzinfo=timezone.utc
                )
                response = mobile_catalog.overview()

        cards = {card["id"]: card for card in response["cards"]}
        self.assertEqual(cards["operations"]["level"], "red")
        self.assertEqual(cards["operations"]["detail"], "APS power telemetry is stale")
        self.assertEqual(cards["battery-soc"]["level"], "red")
        self.assertEqual(
            cards["battery-soc"]["updatedAt"],
            "2026-08-17T18:29:31Z",
        )
        self.assertEqual(cards["battery-voltage"]["level"], "red")
        self.assertEqual(cards["battery-depletion"]["level"], "red")
        self.assertEqual(cards["battery-depletion"]["value"], "Unavailable")
        self.assertIn(
            "depletion cannot be estimated from stale data",
            cards["battery-depletion"]["detail"],
        )
        self.assertEqual(cards["power"]["level"], "red")
        self.assertEqual(len(response["activeAlerts"]), 1)
        self.assertEqual(response["activeAlerts"][0]["id"], "power:freshness")
        self.assertEqual(
            response["activeAlerts"][0]["title"],
            "APS power telemetry is stale",
        )
        self.assertIn("18:29 UTC", response["activeAlerts"][0]["detail"])

    def test_environmental_signal_cards_derive_wind_and_preserve_source_times(self) -> None:
        with patch.object(
            mobile_catalog,
            "_latest_zarr_sample",
            side_effect=[
                {"time": "2026-07-24T06:06:53Z", "t2_t": 10.48297},
                {
                    "time": "2026-07-24T06:00:00Z",
                    "sr30_swd_Irr_Avg": 21.92821,
                    "kt15_tem_Avg": 10.96667,
                    "metek_x_out_Avg": -3.745334,
                    "metek_y_out_Avg": 0.8551666,
                },
            ],
        ):
            cards = mobile_catalog._environmental_signal_cards()

        by_id = {card["id"]: card for card in cards}
        self.assertEqual(list(by_id), ["shortwave-down", "wind-speed", "air-temperature", "kt15"])
        self.assertEqual(by_id["air-temperature"]["value"], "10.5 C")
        self.assertEqual(by_id["shortwave-down"]["value"], "21.9 W/m2")
        self.assertEqual(by_id["wind-speed"]["value"], "3.8 m/s")
        self.assertEqual(by_id["kt15"]["value"], "11.0 C")
        self.assertEqual(by_id["kt15"]["updatedAt"], "2026-07-24T06:00:00Z")

    def test_overview_prefers_cached_measured_power_time_without_opening_display(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "latest.json"
            health = root / "latest_health.json"
            alerts = root / "state.json"
            snapshot.write_text(
                json.dumps({"aps_battery_power_time_utc": "2026-07-21T20:40:00Z"}),
                encoding="utf-8",
            )
            health.write_text(json.dumps({}), encoding="utf-8")
            alerts.write_text(json.dumps({}), encoding="utf-8")
            with patch.dict(
                os.environ,
                {
                    "OPS_MONITOR_LATEST_SNAPSHOT": str(snapshot),
                    "OPS_MONITOR_LATEST_HEALTH": str(health),
                    "OPS_MONITOR_ALERT_STATE": str(alerts),
                    "AURORACAM_RAW_ROOT": str(root / "camera"),
                },
            ), patch.object(
                mobile_catalog, "_latest_power_time", side_effect=AssertionError("display Zarr opened")
            ):
                response = mobile_catalog.overview()

        power = next(card for card in response["cards"] if card["id"] == "power")
        self.assertEqual(power["updatedAt"], "2026-07-21T20:40:00Z")
        self.assertEqual(power["value"], "20:40 UTC")

    def test_overview_includes_meteorology_and_radiation_collection_states(self) -> None:
        rows = mobile_catalog._instrument_power_states(
            {
                "vaisalamet_source_age_min": 5,
                "vaisalamet_source_recent_state": 1,
                "asfs_logger_source_age_min": 185,
                "asfs_logger_source_recent_state": 0,
            }
        )

        meteorology = next(row for row in rows if row["id"] == "vaisalamet")
        radiation = next(row for row in rows if row["id"] == "asfs-logger")
        self.assertEqual([row["id"] for row in rows[:2]], ["vaisalamet", "asfs-logger"])
        self.assertEqual((meteorology["state"], meteorology["level"]), ("Collecting", "green"))
        self.assertEqual((radiation["state"], radiation["level"]), ("No recent data", "red"))

    def test_collection_states_fall_back_to_environmental_sample_times(self) -> None:
        rows = mobile_catalog._instrument_power_states(
            {},
            {
                "vaisalamet": mobile_catalog.utc_now_iso(),
                "asfs-logger": mobile_catalog.utc_now_iso(),
            },
        )

        meteorology = next(row for row in rows if row["id"] == "vaisalamet")
        radiation = next(row for row in rows if row["id"] == "asfs-logger")
        self.assertEqual((meteorology["state"], meteorology["level"]), ("Collecting", "green"))
        self.assertEqual((radiation["state"], radiation["level"]), ("Collecting", "green"))
        self.assertIn("Source sample", meteorology["detail"])

    def test_powered_instruments_with_fresh_data_are_collecting(self) -> None:
        import numpy as np
        import xarray as xr

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            product_times = {
                "CEILOMETER_ZARR_PATH": now - timedelta(minutes=4),
                "CLOUD_RADAR_ZARR_PATH": now - timedelta(minutes=11),
                "HATPRO_ZARR_PATH": now - timedelta(minutes=181),
            }
            environment = {}
            for environment_name, sample_time in product_times.items():
                path = root / f"{environment_name.lower()}.zarr"
                xr.Dataset(
                    {"sample": (("time",), np.array([1.0]))},
                    coords={"time": [sample_time]},
                ).to_zarr(path, mode="w")
                environment[environment_name] = str(path)

            with patch.dict(os.environ, environment), patch.object(
                mobile_catalog,
                "_pdu_power_snapshot",
                return_value=({4: True, 5: True, 6: True, 8: True}, "PDU sample 2m old"),
            ), patch.object(
                mobile_catalog,
                "_powered_instrument_labels",
                return_value={"uas": "On (Tier 3)", "ceilometer": "On with Heater/Blower"},
            ):
                rows = mobile_catalog._instrument_power_states({})

        by_id = {row["id"]: row for row in rows}
        self.assertEqual(by_id["ceilometer"]["state"], "Collecting")
        self.assertEqual(by_id["ceilometer"]["detail"], "Source sample 4m old")
        self.assertEqual(by_id["cloud-radar"]["state"], "Collecting")
        self.assertEqual(by_id["cloud-radar"]["detail"], "Source sample 11m old")
        self.assertEqual(by_id["hatpro"]["state"], "On")
        self.assertEqual(by_id["hatpro"]["detail"], "PDU sample 2m old")
        self.assertEqual(by_id["uas"]["state"], "On (Tier 3)")

    def test_collection_freshness_uses_instrument_cadence(self) -> None:
        import numpy as np
        import xarray as xr

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_time = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=120)
            environment = {}
            for environment_name in ("CLOUD_RADAR_ZARR_PATH", "HATPRO_ZARR_PATH"):
                path = root / f"{environment_name.lower()}.zarr"
                xr.Dataset(
                    {"sample": (("time",), np.array([1.0]))},
                    coords={"time": [sample_time]},
                ).to_zarr(path, mode="w")
                environment[environment_name] = str(path)

            with patch.dict(os.environ, environment):
                radar = mobile_catalog._powered_collection_detail("cloud-radar")
                hatpro = mobile_catalog._powered_collection_detail("hatpro")

        self.assertIsNone(radar)
        self.assertEqual(hatpro, "Source sample 2h 0m old")

    def test_malformed_collection_time_fails_closed(self) -> None:
        import numpy as np
        import zarr

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ceilometer.zarr"
            group = zarr.open_group(str(path), mode="w")
            times = group.create_dataset("time", shape=(1,), dtype="f8")
            times[:] = np.array([np.nan])
            times.attrs["units"] = "seconds since 1970-01-01"
            with patch.dict(os.environ, {"CEILOMETER_ZARR_PATH": str(path)}):
                detail = mobile_catalog._powered_collection_detail("ceilometer")

        self.assertIsNone(detail)

    def test_recent_data_never_overrides_powered_off(self) -> None:
        row = mobile_catalog._pdu_instrument_status(
            "ceilometer",
            {5: False},
            "PDU sample 2m old",
            {"ceilometer": "On with Heater/Blower"},
            collection_detail="Source sample 1m old",
        )

        self.assertEqual(row["state"], "Off")
        self.assertEqual(row["detail"], "PDU sample 2m old")

    def test_powered_off_instruments_do_not_open_collection_products(self) -> None:
        with patch.object(
            mobile_catalog,
            "_pdu_power_snapshot",
            return_value=({4: False, 5: False, 6: False, 8: False}, "PDU sample 2m old"),
        ), patch.object(
            mobile_catalog,
            "_powered_instrument_labels",
            return_value={},
        ), patch.object(
            mobile_catalog,
            "_powered_collection_detail",
            side_effect=AssertionError("collection product opened"),
        ):
            rows = mobile_catalog._instrument_power_states({})

        self.assertEqual(
            [row["state"] for row in rows if row["id"] in mobile_catalog.PDU_INSTRUMENT_BY_ID],
            ["Off", "Off", "Off", "Off"],
        )

    def test_cl61_overview_names_the_learned_heater_blower_phase(self) -> None:
        fake_group = SimpleNamespace(
            attrs={
                "current_mode": "dc_cl61",
                "last_observation_time_utc": mobile_catalog.utc_now_iso(),
                "mode_load_profiles": json.dumps(
                    {"dc_cl61": {"current_phase": "fan_high"}}
                ),
            }
        )
        fake_zarr = SimpleNamespace(open_group=lambda *_args, **_kwargs: fake_group)
        with patch.dict(sys.modules, {"zarr": fake_zarr}):
            labels = mobile_catalog._automatic_power_labels()

        self.assertEqual(labels, {"ceilometer": "On with Heater/Blower"})
        row = mobile_catalog._pdu_instrument_status(
            "ceilometer",
            {5: True},
            "PDU sample 2m old",
            labels,
        )
        self.assertEqual(row["state"], "On with Heater/Blower")
        self.assertEqual(row["level"], "green")

    def test_cl61_heater_blower_label_never_overrides_powered_off(self) -> None:
        row = mobile_catalog._pdu_instrument_status(
            "ceilometer",
            {5: False},
            "PDU sample 2m old",
            {"ceilometer": "On with Heater/Blower"},
        )

        self.assertEqual(row["state"], "Off")

    def test_uas_overview_includes_the_latest_fresh_effective_tier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "menapia_mqtt.log"
            now = datetime.now(timezone.utc)
            log.write_text(
                "\n".join(
                    [
                        f"{(now - timedelta(minutes=2)):%Y-%m-%d %H:%M:%S}: 4 4",
                        f"{now:%Y-%m-%d %H:%M:%S}: Tier change 4 3",
                    ]
                ),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"UAS_MQTT_LOG_PATH": str(log)}):
                labels = mobile_catalog._powered_instrument_labels({4: True})

        self.assertEqual(labels.get("uas"), "On (Dock 1 Tier 4; Dock 2 Tier 3)")
        row = mobile_catalog._pdu_instrument_status(
            "uas",
            {4: True},
            "PDU sample 2m old",
            labels,
        )
        self.assertEqual(row["state"], "On (Dock 1 Tier 4; Dock 2 Tier 3)")
        self.assertEqual(row["level"], "green")

    def test_uas_tier_never_overrides_powered_off(self) -> None:
        row = mobile_catalog._pdu_instrument_status(
            "uas",
            {4: False},
            "PDU sample 2m old",
            {"uas": "On (Tier 3)"},
        )

        self.assertEqual(row["state"], "Off")

    def test_uas_overview_ignores_a_stale_effective_tier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "menapia_mqtt.log"
            stale = datetime.now(timezone.utc) - timedelta(
                minutes=mobile_catalog.UAS_TIER_FRESHNESS_MINUTES + 1
            )
            log.write_text(
                f"{stale:%Y-%m-%d %H:%M:%S}: 3 3\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"UAS_MQTT_LOG_PATH": str(log)}):
                label = mobile_catalog._fresh_uas_tier_label()

        self.assertIsNone(label)

    def test_wxcam_discovers_videos_and_thumbnails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            videos = root / "videos" / "fish_hdr"
            thumbs = root / "thumbs" / "fish_hdr" / "20260705"
            videos.mkdir(parents=True)
            thumbs.mkdir(parents=True)
            (videos / "20260705.mp4").write_bytes(b"video")
            thumbnail = thumbs / "HDR_20260705_123000.jpg"
            thumbnail.write_bytes(b"thumb")

            with patch.dict(
                os.environ,
                {
                    "WXCAM_DAILY_VIDEO_DIR": str(root / "videos"),
                    "WXCAM_HOURLY_THUMB_DIR": str(root / "thumbs"),
                    "WXCAM_CATALOG_PATH": str(root / "missing.sqlite"),
                },
            ):
                response = mobile_catalog.wxcam("fish_hdr", "2026-07-05")

        self.assertTrue(response["video"]["exists"])
        self.assertEqual(response["availableDays"], ["2026-07-05"])
        self.assertEqual(response["thumbnails"][0]["hourUTC"], 12)
        self.assertEqual(
            response["thumbnails"][0]["imageURL"],
            "/media/wxcam/thumb/fish_hdr/20260705/HDR_20260705_123000.jpg",
        )

    def test_wxcam_thumbnails_deduplicate_candidates_and_use_filename_hours(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            thumbs = root / "fish_hdr" / "20260818"
            thumbs.mkdir(parents=True)
            for hour in (0, 2):
                for minute in range(0, 31, 5):
                    (thumbs / f"HDR_20260818_{hour:02d}{minute:02d}39.jpg").write_bytes(b"thumb")
            for minute in (0, 5, 10, 15):
                (thumbs / f"HDR_20260818_14{minute:02d}39.jpg").write_bytes(b"thumb")
            (thumbs / "sample.jpg").write_bytes(b"malformed")
            (thumbs / "HDR_20260817_143000.jpg").write_bytes(b"wrong day")

            with patch.dict(os.environ, {"WXCAM_HOURLY_THUMB_DIR": str(root)}):
                records = mobile_catalog.wxcam_thumbnail_records("fish_hdr", "2026-08-18")

        self.assertEqual([record["hourUTC"] for record in records], [0, 2, 14])
        self.assertEqual(
            [record["id"] for record in records],
            [
                "HDR_20260818_003039",
                "HDR_20260818_023039",
                "HDR_20260818_141539",
            ],
        )

    def test_wxcam_media_resolvers_reject_malformed_day_tokens(self) -> None:
        self.assertIsNone(mobile_catalog.resolve_wxcam_video_path("fish_hdr", ".."))
        self.assertIsNone(mobile_catalog.resolve_wxcam_thumbnail_path("fish_hdr", "..", "sample.jpg"))
        self.assertIsNone(mobile_catalog.resolve_wxcam_thumbnail_path("fish_hdr", "20260705", "../sample.jpg"))


if __name__ == "__main__":
    unittest.main()
