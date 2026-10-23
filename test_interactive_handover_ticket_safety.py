"""Offline regression tests for the interactive Step 3 ticket-confirmation guard.

All ticket-tool calls are mocked. No Gemini, Gnani, or network calls are made.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
try:
    import run
except Exception as exc:  # pragma: no cover - environment diagnostic
    raise SystemExit(
        "Could not import run.py. Ensure this test is beside run.py and project dependencies are installed. "
        f"Import error: {type(exc).__name__}: {exc}"
    ) from exc


# VOICEOPS_INTERACTIVE_HANDOVER_SAFETY_TESTS_V1
class InteractiveStep3TicketSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.kwargs = {
            "summary": "Synthetic escalation summary",
            "order_id": None,
            "reason": "human requested",
            "intent": "support",
        }

    def _call(self, ticket_creator):
        return run._create_early_escalation_ticket_confirmation(ticket_creator, **self.kwargs)

    def test_ticket_tool_exception_returns_unconfirmed_fallback_without_leaking_message(self) -> None:
        creator = Mock(side_effect=RuntimeError("SECRET_PROVIDER_DETAIL"))
        answer, ticket_id = self._call(creator)
        self.assertIsNone(ticket_id)
        self.assertIn("could not confirm that a support ticket was created", answer.lower())
        self.assertNotIn("SECRET_PROVIDER_DETAIL", answer)
        self.assertNotIn("ticket reference is", answer.lower())
        creator.assert_called_once_with(**self.kwargs)

    def test_malformed_or_unconfirmed_results_never_claim_ticket_creation(self) -> None:
        results = (
            None,
            [],
            "created",
            {"created": "true", "ticket_id": "TKT-FAKE"},
            {"created": False, "ticket_id": "TKT-FAKE"},
            {},
        )
        for result in results:
            with self.subTest(result_type=type(result).__name__, result=result):
                answer, ticket_id = self._call(Mock(return_value=result))
                self.assertIsNone(ticket_id)
                self.assertIn("could not confirm that a support ticket was created", answer.lower())
                self.assertNotIn("TKT-FAKE", answer)

    def test_valid_ticket_reference_is_trimmed_and_returned(self) -> None:
        creator = Mock(return_value={"created": True, "ticket_id": "  TKT-001  "})
        answer, ticket_id = self._call(creator)
        self.assertEqual(ticket_id, "TKT-001")
        self.assertIn("TKT-001", answer)
        self.assertIn("system confirmed that a support ticket was created", answer.lower())
        self.assertIn("cannot promise a particular outcome or timeline", answer.lower())
        creator.assert_called_once_with(**self.kwargs)

    def test_confirmed_creation_without_reference_is_described_honestly(self) -> None:
        values = ({"created": True}, {"created": True, "ticket_id": None})
        for result in values:
            with self.subTest(result=result):
                answer, ticket_id = self._call(Mock(return_value=result))
                self.assertIsNone(ticket_id)
                self.assertIn("system confirmed that a support ticket was created", answer.lower())
                self.assertIn("did not return a usable reference number", answer.lower())
                self.assertNotIn("could not confirm that a support ticket was created", answer.lower())

    def test_blank_reference_is_not_presented_as_a_real_id(self) -> None:
        for value in ("", "   "):
            with self.subTest(value=value):
                answer, ticket_id = self._call(Mock(return_value={"created": True, "ticket_id": value}))
                self.assertIsNone(ticket_id)
                self.assertIn("did not return a usable reference number", answer.lower())

    def test_non_string_reference_is_not_exposed(self) -> None:
        answer, ticket_id = self._call(Mock(return_value={"created": True, "ticket_id": 12345}))
        self.assertIsNone(ticket_id)
        self.assertNotIn("12345", answer)
        self.assertIn("did not return a usable reference number", answer.lower())

    def test_control_characters_in_reference_are_rejected(self) -> None:
        malicious_reference = "TKT-123\nFORGED-HANDOVER"
        answer, ticket_id = self._call(Mock(return_value={"created": True, "ticket_id": malicious_reference}))
        self.assertIsNone(ticket_id)
        self.assertNotIn("FORGED-HANDOVER", answer)
        self.assertIn("did not return a usable reference number", answer.lower())

    def test_overlong_reference_is_rejected(self) -> None:
        too_long = "TKT-" + ("X" * 200)
        answer, ticket_id = self._call(Mock(return_value={"created": True, "ticket_id": too_long}))
        self.assertIsNone(ticket_id)
        self.assertNotIn(too_long, answer)
        self.assertIn("did not return a usable reference number", answer.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
