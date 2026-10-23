"""Offline unit/integration tests for Step 3 order lookup safety."""
from __future__ import annotations

import importlib
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, call, patch

import run

# Resolve the project's dataset-backed tools just as step3() does.
_DATASET_PATH = Path(run.DATASET).resolve()
if str(_DATASET_PATH) not in sys.path:
    sys.path.insert(0, str(_DATASET_PATH))
_tools_module = importlib.import_module("tools.tools")


# VOICEOPS_ORDER_LOOKUP_SAFETY_V1
class OrderLookupSafetyTests(unittest.TestCase):
    def safe_call(self, tool, verification_value="1234"):
        output = io.StringIO()
        with redirect_stdout(output):
            result = run._safe_order_lookup_result(tool, "12345678", verification_value)
        return result, output.getvalue()

    def test_verification_lookup_exception_is_safe_and_sanitized(self):
        def fail(*args):
            raise RuntimeError("PRIVATE_PROVIDER_EXCEPTION_DETAIL")
        result, output = self.safe_call(fail)
        self.assertIs(result.get("_lookup_error"), True)
        self.assertIs(result.get("verified"), False)
        self.assertNotIn("PRIVATE_PROVIDER_EXCEPTION_DETAIL", output)
        self.assertIn("RuntimeError", output)

    def test_malformed_verification_results_fail_closed(self):
        invalid = (
            None,
            [],
            "verified",
            {},
            {"verified": "true", "found": True, "status": "SECRET"},
            {"verified": True, "found": "true", "status": "SECRET"},
            {"verified": True, "found": False, "status": "SECRET"},
        )
        for value in invalid:
            with self.subTest(value=value):
                result, _ = self.safe_call(lambda *args, value=value: value)
                self.assertIs(result.get("_lookup_error"), True)
                self.assertNotIn("status", result)

    def test_failed_verification_strips_order_details(self):
        raw = {"verified": False, "found": False, "status": "PRIVATE_STATUS", "customer_name": "PRIVATE_NAME"}
        result, _ = self.safe_call(lambda *args: raw)
        self.assertEqual(result, {"verified": False, "found": False})
        self.assertNotIn("status", result)
        self.assertNotIn("customer_name", result)

    def test_verified_result_is_preserved_for_trusted_downstream_filtering(self):
        raw = {"verified": True, "found": True, "order_id": "12345678", "status": "Delivered"}
        result, _ = self.safe_call(lambda *args: raw)
        self.assertEqual(result, raw)
        self.assertNotIn("_lookup_error", result)

    def test_unverified_probe_discards_the_raw_tool_result(self):
        tool = Mock(return_value={"verified": False, "found": True, "status": "PRIVATE_STATUS"})
        with redirect_stdout(io.StringIO()):
            result = run._safe_order_lookup_result(tool, "12345678")
        tool.assert_called_once_with("12345678")
        self.assertEqual(result, {"_lookup_error": False})
        self.assertNotIn("status", result)

    def test_unverified_probe_exception_is_converted_to_safe_error(self):
        tool = Mock(side_effect=RuntimeError("PRIVATE_PROVIDER_EXCEPTION_DETAIL"))
        output = io.StringIO()
        with redirect_stdout(output):
            result = run._safe_order_lookup_result(tool, "12345678")
        self.assertIs(result.get("_lookup_error"), True)
        self.assertNotIn("PRIVATE_PROVIDER_EXCEPTION_DETAIL", output.getvalue())

    def _run_step3(self, inputs, lookup_side_effect, order_ids, verification_values):
        output = io.StringIO()
        lookup = Mock(side_effect=lookup_side_effect)
        retrieve = Mock(return_value=[])
        generate_answer = Mock(return_value="The order status is delivered.")
        with patch.object(run, "require_dataset"), \
             patch.object(run, "get_client", return_value=object()), \
             patch.object(run, "load_or_build_index", return_value=([], [])), \
             patch.object(run, "_escalation_reason", return_value=None), \
             patch.object(run, "_extract_order_id", side_effect=order_ids), \
             patch.object(run, "_extract_verification_value", side_effect=verification_values), \
             patch.object(run, "retrieve", retrieve), \
             patch.object(run, "generate_answer", generate_answer), \
             patch.object(_tools_module, "lookup_order", lookup), \
             patch("builtins.input", side_effect=iter(inputs)), \
             redirect_stdout(output):
            run.step3()
        return output.getvalue(), lookup, retrieve, generate_answer

    def test_step3_verification_tool_exception_never_reaches_retrieval(self):
        output, lookup, retrieve, generate = self._run_step3(
            ("Please check this order", "my phone suffix is 1234", "exit"),
            [
                {"verified": False, "found": False},
                RuntimeError("PRIVATE_PROVIDER_EXCEPTION_DETAIL"),
            ],
            ["12345678", None],
            [None, "1234"],
        )
        self.assertIn("lookup service is temporarily unavailable", output.lower())
        self.assertNotIn("PRIVATE_PROVIDER_EXCEPTION_DETAIL", output)
        self.assertNotIn("order status is delivered", output.lower())
        self.assertEqual(lookup.call_args_list, [call("12345678"), call("12345678", "1234")])
        retrieve.assert_not_called()
        generate.assert_not_called()

    def test_step3_malformed_verification_result_never_reaches_retrieval(self):
        output, lookup, retrieve, generate = self._run_step3(
            ("Please check this order", "my phone suffix is 1234", "exit"),
            [
                {"verified": False, "found": False},
                {"verified": "true", "found": True, "status": "PRIVATE_ORDER_STATUS"},
            ],
            ["12345678", None],
            [None, "1234"],
        )
        self.assertIn("lookup service is temporarily unavailable", output.lower())
        self.assertNotIn("PRIVATE_ORDER_STATUS", output)
        self.assertEqual(lookup.call_count, 2)
        retrieve.assert_not_called()
        generate.assert_not_called()

    def test_switching_order_requires_fresh_verification(self):
        output, lookup, retrieve, generate = self._run_step3(
            (
                "Please check first order",
                "my phone suffix is 1234",
                "Please check the next order",
                "exit",
            ),
            [
                {"verified": False, "found": False},
                {
                    "verified": True,
                    "found": True,
                    "order_id": "11111111",
                    "status": "Delivered",
                    "customer_name": "PRIVATE_CUSTOMER_NAME",
                },
                {"verified": False, "found": False},
            ],
            ["11111111", None, "22222222"],
            [None, "1234", None],
        )
        self.assertEqual(
            lookup.call_args_list,
            [call("11111111"), call("11111111", "1234"), call("22222222")],
        )
        self.assertEqual(retrieve.call_count, 1)
        self.assertEqual(generate.call_count, 1)
        self.assertNotIn("PRIVATE_CUSTOMER_NAME", output)
        # The new order is asked for fresh verification before another retrieval/generation.
        self.assertIn("before sharing order-specific information", output.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
