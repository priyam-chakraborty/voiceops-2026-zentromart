"""Offline tests for Step 3 ticket-confirmation fallback.

Place beside run.py and run:
    py test_step3_handover_fallback.py
No external API calls are made; generate_answer is replaced by a local stub.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
try:
    import run
except Exception as exc:  # pragma: no cover - environment diagnostic
    raise SystemExit(
        "Could not import run.py. Ensure this test is beside run.py and the project dependencies are installed. "
        f"Import error: {type(exc).__name__}: {exc}"
    ) from exc


class Step3HandoverFallbackTests(unittest.TestCase):
    def test_created_ticket_is_confirmed_when_generation_fails(self) -> None:
        ticket = {"created": True, "ticket_id": "TKT-TEST-123"}
        with patch.object(run, "generate_answer", side_effect=RuntimeError("simulated quota error")):
            answer = run._generate_handover_answer(
                client=None,
                question="My television arrived damaged; please escalate.",
                context="Synthetic context",
                ticket=ticket,
                extra_handover_instruction="",
                high_value_claim=True,
            )
        self.assertIn("ticket was created successfully", answer.lower())
        self.assertIn("TKT-TEST-123", answer)
        self.assertIn("[DWM-3]", answer)
        self.assertIn("[ESC-1]", answer)
        self.assertIn("cannot approve the claim", answer.lower())
        self.assertIn("promise a particular outcome", answer.lower())

    def test_failed_ticket_is_never_claimed_as_created_when_generation_fails(self) -> None:
        ticket = {"created": False}
        with patch.object(run, "generate_answer", side_effect=RuntimeError("simulated quota error")):
            answer = run._generate_handover_answer(
                client=None,
                question="Please escalate my issue.",
                context="Synthetic context",
                ticket=ticket,
            )
        self.assertIn("couldn’t confirm", answer.lower())
        self.assertIn("handover is complete", answer.lower())
        self.assertNotIn("ticket was created successfully", answer.lower())

    def test_successful_generation_still_appends_real_ticket_reference(self) -> None:
        ticket = {"created": True, "ticket_id": "TKT-REAL-456"}
        with patch.object(run, "generate_answer", return_value="I have documented the issue for review."):
            answer = run._generate_handover_answer(
                client=None,
                question="Please escalate this.",
                context="Synthetic context",
                ticket=ticket,
            )
        self.assertTrue(answer.endswith("A human-support ticket was created successfully. Reference: TKT-REAL-456."))

    def test_failed_ticket_normal_generation_does_not_claim_success(self) -> None:
        ticket = {"created": False}
        with patch.object(run, "generate_answer", return_value="I have reviewed the request."):
            answer = run._generate_handover_answer(
                client=None,
                question="Please escalate this.",
                context="Synthetic context",
                ticket=ticket,
            )
        self.assertIn("could not confirm ticket creation", answer.lower())
        self.assertNotIn("ticket was created successfully", answer.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
