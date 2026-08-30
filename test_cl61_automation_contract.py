from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
import xarray as xr

from cl61_automation_contract import (
    AUTHORITY_DIAGNOSTIC,
    AUTHORITY_OPERATIONAL,
    build_intent,
    validate_intent,
)
from generate_cl61_automation_intent import build_diagnostic_intent, publish_diagnostic_intent


class CL61AutomationContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 8, 30, 12, tzinfo=timezone.utc)

    def _intent(self, *, authority: str = AUTHORITY_DIAGNOSTIC, key: bytes | None = None) -> dict:
        return build_intent(
            environment="development",
            authority=authority,
            proposed_action="start",
            desired_outlet_state=True,
            reason_codes=["test"],
            forecast={
                "forecast_system_version": "power-v12-hybrid-candidate",
                "source_cycle_set_id": "cycle",
                "scenario_publication_signature": "scenario-signature",
            },
            safety={
                "control_eligible": authority == AUTHORITY_OPERATIONAL,
                "planning_ready": True,
                "schedule_safe": True,
            },
            input_ages_minutes={"power": 1.0},
            generated_at=self.now,
            ttl_seconds=600,
            signing_key=key,
        )

    def test_diagnostic_intent_is_valid_for_observation_but_not_execution(self) -> None:
        intent = self._intent()

        valid, errors = validate_intent(intent, now=self.now)
        self.assertTrue(valid)
        self.assertEqual(errors, ())
        executable, execution_errors = validate_intent(
            intent,
            now=self.now,
            require_operational=True,
        )
        self.assertFalse(executable)
        self.assertIn("diagnostic_authority", execution_errors)
        self.assertIn("operational_signature_required", execution_errors)

    def test_signed_operational_intent_rejects_tampering(self) -> None:
        key = b"test-signing-key"
        intent = self._intent(authority=AUTHORITY_OPERATIONAL, key=key)

        valid, errors = validate_intent(intent, now=self.now, signing_key=key, require_operational=True)
        self.assertTrue(valid)
        self.assertEqual(errors, ())
        intent["desired_outlet_state"] = False
        valid, errors = validate_intent(intent, now=self.now, signing_key=key, require_operational=True)
        self.assertFalse(valid)
        self.assertIn("content_digest_mismatch", errors)

    def test_validation_rejects_wrong_outlet_and_expired_intent(self) -> None:
        intent = self._intent()
        intent["target"]["pdu_outlet"] = 6
        valid, errors = validate_intent(intent, now=self.now)
        self.assertFalse(valid)
        self.assertIn("invalid_target", errors)

        expired = self._intent()
        valid, errors = validate_intent(expired, now=self.now + timedelta(minutes=11))
        self.assertFalse(valid)
        self.assertIn("expired", errors)

    def test_scenario_publication_is_observe_only_and_append_only(self) -> None:
        times = pd.date_range("2026-08-30T12:00:00", periods=2, freq="1h")
        scenarios = xr.Dataset(
            {
                "ScenarioModeCode": (("scenario", "time"), np.asarray([[0, 1]], dtype=np.int16)),
            },
            coords={"scenario": ["optimized_cl61"], "time": times},
            attrs={
                "planning_status": "ready",
                "optimized_safe": "true",
                "optimized_minimum_p10_soc": "48",
                "minimum_operational_soc_pct": "40",
                "minimum_controlled_run_hours": "12",
                "optimized_schedule_policy": "cl61_primary_v1",
                "publication_signature": "scenario-signature",
            },
        )
        intent = build_diagnostic_intent(scenarios, generated_at=self.now)
        self.assertEqual(intent["authority"], AUTHORITY_DIAGNOSTIC)
        self.assertEqual(intent["proposed_action"], "start")
        self.assertFalse(intent["safety"]["control_eligible"])

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            status = publish_diagnostic_intent(
                scenarios,
                intent_path=root / "intent.json",
                status_path=root / "status.json",
                history_path=root / "history.jsonl",
                generated_at=self.now,
            )
            self.assertEqual(status["mode"], "observe_only")
            self.assertFalse(status["capability"])
            self.assertEqual(len((root / "history.jsonl").read_text(encoding="utf-8").splitlines()), 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
