"""Integration tests for the actual interactive Step 3 early-escalation path.

These tests drive step3() with mocked ticket/data/provider boundaries. They do not
call Gemini, Gnani, or the network, and they never alter the synthetic dataset.
"""
from __future__ import annotations

import importlib
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import run

# Make sure the project's synthetic dataset package wins module resolution.
_DATASET_PATH = Path(run.DATASET).resolve()
if str(_DATASET_PATH) not in sys.path:
    sys.path.insert(0, str(_DATASET_PATH))
_tools_module = importlib.import_module("tools.tools")


# VOICEOPS_STEP3_EARLY_ESCALATION_INTEGRATION_V1
class Step3InteractiveFlowIntegrationTests(unittest.TestCase):
    def _run_step3(self, ticket_result=None, *, ticket_side_effect=None):
        output = io.StringIO()
        creator = Mock(return_value=ticket_result)
        if ticket_side_effect is not None:
            creator.side_effect = ticket_side_effect
        lookup = Mock(return_value={"verified": False, "found": False})
        # Exit immediately after the first response. step3()'s early path should
        # handle the request before retrieval or answer generation.
        user_inputs = iter(("Please connect me to a human support agent.", "exit"))
        with patch.object(run, "require_dataset"), \
             patch.object(run, "get_client", return_value=object()), \
             patch.object(run, "load_or_build_index", return_value=([], [])), \
             patch.object(run, "_escalation_reason", return_value="human requested"), \
             patch.object(run, "_intent_for", return_value="support"), \
             patch.object(run, "retrieve") as retrieve, \
             patch.object(run, "generate_answer") as generate_answer, \
             patch.object(_tools_module, "create_ticket", creator), \
             patch.object(_tools_module, "lookup_order", lookup), \
             patch("builtins.input", side_effect=lambda _prompt="": next(user_inputs)), \
             redirect_stdout(output):
            run.step3()
        return output.getvalue(), creator, lookup, retrieve, generate_answer

    def test_actual_step3_confirms_successful_human_escalation(self) -> None:
        output, creator, lookup, retrieve, generate_answer = self._run_step3(
            {"created": True, "ticket_id": "  TKT-INTEGRATION-101  "}
        )
        self.assertIn("system confirmed that a support ticket was created", output.lower())
        self.assertIn("TKT-INTEGRATION-101", output)
        self.assertNotIn("could not confirm that a support ticket was created", output.lower())
        creator.assert_called_once()
        lookup.assert_not_called()
        retrieve.assert_not_called()
        generate_answer.assert_not_called()

    def test_actual_step3_sanitizes_ticket_tool_exception(self) -> None:
        output, creator, lookup, retrieve, generate_answer = self._run_step3(
            ticket_side_effect=RuntimeError("INTERNAL_SECRET_PROVIDER_MESSAGE")
        )
        self.assertIn("could not confirm that a support ticket was created", output.lower())
        self.assertNotIn("INTERNAL_SECRET_PROVIDER_MESSAGE", output)
        self.assertNotIn("ticket reference is", output.lower())
        creator.assert_called_once()
        lookup.assert_not_called()
        retrieve.assert_not_called()
        generate_answer.assert_not_called()

    def test_actual_step3_handles_malformed_ticket_result(self) -> None:
        output, creator, lookup, retrieve, generate_answer = self._run_step3(None)
        self.assertIn("could not confirm that a support ticket was created", output.lower())
        self.assertNotIn("ticket reference is", output.lower())
        creator.assert_called_once()
        lookup.assert_not_called()
        retrieve.assert_not_called()
        generate_answer.assert_not_called()

    def test_actual_step3_is_honest_when_ticket_has_no_usable_reference(self) -> None:
        output, creator, lookup, retrieve, generate_answer = self._run_step3(
            {"created": True, "ticket_id": "  "}
        )
        self.assertIn("system confirmed that a support ticket was created", output.lower())
        self.assertIn("did not return a usable reference number", output.lower())
        self.assertNotIn("ticket reference is", output.lower())
        creator.assert_called_once()
        lookup.assert_not_called()
        retrieve.assert_not_called()
        generate_answer.assert_not_called()

    def test_early_escalation_is_handled_before_retrieval_and_model_generation(self) -> None:
        output, creator, lookup, retrieve, generate_answer = self._run_step3(
            {"created": True, "ticket_id": "TKT-EARLY-102"}
        )
        self.assertIn("TKT-EARLY-102", output)
        retrieve.assert_not_called()
        generate_answer.assert_not_called()
        lookup.assert_not_called()
        creator.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
