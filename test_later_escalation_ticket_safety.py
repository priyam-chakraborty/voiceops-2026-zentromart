"""Offline tests for the later, policy-driven Step 3 ticket path."""
from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import run


class LaterEscalationTicketSafetyTests(unittest.TestCase):
    def call(self, creator):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            result = run._create_ticket_result_safely(
                creator, summary="synthetic summary", order_id=None,
                reason="synthetic escalation", intent="other",
            )
        return result, output.getvalue()

    def test_ticket_tool_exception_is_unconfirmed_and_secret_is_not_logged(self):
        def fail(**kwargs):
            raise RuntimeError("TOP_SECRET_PROVIDER_DETAIL")
        result, output = self.call(fail)
        self.assertEqual(result, {"created": False})
        self.assertIn("RuntimeError", output)
        self.assertNotIn("TOP_SECRET_PROVIDER_DETAIL", output)

    def test_malformed_results_are_unconfirmed(self):
        for value in (None, [], "created", {}, {"created": "true", "ticket_id": "TKT-FAKE"}):
            with self.subTest(value=value):
                result, _ = self.call(lambda **kwargs: value)
                self.assertEqual(result, {"created": False})

    def test_explicit_creation_failure_ignores_supplied_reference(self):
        result, _ = self.call(lambda **kwargs: {"created": False, "ticket_id": "TKT-FAKE"})
        self.assertEqual(result, {"created": False})

    def test_valid_reference_is_trimmed(self):
        result, _ = self.call(lambda **kwargs: {"created": True, "ticket_id": "  TKT-901  "})
        self.assertEqual(result, {"created": True, "ticket_id": "TKT-901"})

    def test_missing_or_unsafe_reference_does_not_cancel_confirmed_creation(self):
        for value in (None, "", "   ", 123, "X" * 121, "TKT\nINJECT", "TKT\u202eFAKE"):
            with self.subTest(value=value):
                result, _ = self.call(lambda **kwargs: {"created": True, "ticket_id": value})
                self.assertEqual(result, {"created": True})

    def test_helper_accepts_malformed_ticket_without_false_success(self):
        for ticket in (None, [], "bad", {"created": "true", "ticket_id": "TKT-FAKE"}):
            with self.subTest(ticket=ticket):
                with patch.object(run, "generate_answer", return_value="I reviewed the request."):
                    answer = run._generate_handover_answer(
                        client=None, question="Please escalate", context="Synthetic context", ticket=ticket
                    )
                self.assertIn("could not confirm ticket creation", answer.lower())
                self.assertNotIn("ticket was created successfully", answer.lower())

    def test_helper_does_not_emit_unsafe_ticket_reference(self):
        unsafe = "TKT\nINJECT"
        with patch.object(run, "generate_answer", return_value="A human will review this.") as generator:
            answer = run._generate_handover_answer(
                client=None, question="Please escalate", context="Synthetic context",
                ticket={"created": True, "ticket_id": unsafe},
            )
        self.assertNotIn(unsafe, answer)
        self.assertNotIn(unsafe, generator.call_args.args[3])
        self.assertIn("reference number is unavailable", answer.lower())

    def test_helper_preserves_confirmed_ticket_without_reference(self):
        with patch.object(run, "generate_answer", side_effect=RuntimeError("mock unavailable")):
            answer = run._generate_handover_answer(
                client=None, question="Please escalate", context="Synthetic context",
                ticket={"created": True},
            )
        self.assertIn("ticket was created successfully", answer.lower())
        self.assertNotIn("Reference:", answer)


if __name__ == "__main__":
    unittest.main(verbosity=2)
