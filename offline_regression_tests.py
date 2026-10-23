"""Offline regression tests for VoiceOps' deterministic support tools.

Run from the VoiceOps project root:
    python offline_regression_tests.py

These tests use only the supplied synthetic dataset. They do not call Gemini,
Gnani, or any network service, and they never print customer data or API keys.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "Capstone_Project_Dataset_FTTT_2026"
if not DATASET.is_dir():
    raise SystemExit(
        "Dataset folder not found. Put this file beside run.py and the "
        "Capstone_Project_Dataset_FTTT_2026 folder."
    )
sys.path.insert(0, str(DATASET))

try:
    from tools import tools as support_tools
except Exception as exc:  # pragma: no cover - startup diagnostics
    raise SystemExit(f"Could not load the synthetic support tools: {type(exc).__name__}: {exc}") from exc


class SupportToolRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not support_tools._ORDERS or not support_tools._CUSTOMERS:
            raise RuntimeError("The synthetic orders/customers tables loaded empty.")
        cls.sample_orders = list(support_tools._ORDERS.items())[:25]

    def test_lookup_without_proof_discloses_no_order_details(self) -> None:
        order_id, _order = self.sample_orders[0]
        result = support_tools.lookup_order(order_id)

        self.assertIs(result.get("verification_required"), True)
        self.assertNotIn("customer_id", result)
        self.assertNotIn("phone", result)
        self.assertNotIn("delivery_pincode", result)
        self.assertNotIn("items", result)
        self.assertNotIn("total_inr", result)

    def test_valid_phone_suffix_verifies_sample_orders(self) -> None:
        for order_id, order in self.sample_orders:
            with self.subTest(case="phone_suffix"):
                customer = support_tools._CUSTOMERS[order["customer_id"]]
                proof = str(customer["phone"])[-4:]
                result = support_tools.lookup_order(order_id, proof)
                self.assertIs(result.get("verified"), True)
                self.assertIs(result.get("found"), True)

    def test_valid_delivery_pincode_verifies_sample_orders(self) -> None:
        for order_id, order in self.sample_orders:
            with self.subTest(case="delivery_pincode"):
                result = support_tools.lookup_order(order_id, str(order["delivery_pincode"]))
                self.assertIs(result.get("verified"), True)
                self.assertIs(result.get("found"), True)

    def test_wrong_phone_suffix_does_not_disclose_order_details(self) -> None:
        order_id, order = self.sample_orders[0]
        customer = support_tools._CUSTOMERS[order["customer_id"]]
        real_suffix = str(customer["phone"])[-4:]
        bad_suffix = f"{(int(real_suffix) + 1) % 10000:04d}"
        if bad_suffix == real_suffix:
            bad_suffix = f"{(int(real_suffix) + 2) % 10000:04d}"

        result = support_tools.lookup_order(order_id, bad_suffix)
        self.assertIs(result.get("verified"), False)
        self.assertNotIn("customer_id", result)
        self.assertNotIn("phone", result)
        self.assertNotIn("delivery_pincode", result)
        self.assertNotIn("items", result)
        self.assertNotIn("total_inr", result)

    def test_ticket_rejects_short_summary_and_records_success_safely(self) -> None:
        original_ticket_path = support_tools._TICKETS
        with tempfile.TemporaryDirectory(prefix="voiceops_ticket_test_") as temp_dir:
            temp_ticket_path = Path(temp_dir) / "tickets_created.csv"
            support_tools._TICKETS = str(temp_ticket_path)
            try:
                rejected = support_tools.create_ticket(summary="too short")
                self.assertIs(rejected.get("created"), False)
                self.assertFalse(temp_ticket_path.exists())

                accepted = support_tools.create_ticket(
                    summary=(
                        "Issue: damaged television; checks made: caller's request recorded; "
                        "escalation reason: high-value item requires human review."
                    ),
                    order_id="TEST-ORDER",
                    reason="High-value damaged item",
                    intent="damaged_wrong_missing",
                )
                self.assertIs(accepted.get("created"), True)
                self.assertTrue(str(accepted.get("ticket_id", "")).startswith("TKT"))
                self.assertTrue(temp_ticket_path.exists())
                self.assertIn(accepted["ticket_id"], temp_ticket_path.read_text(encoding="utf-8"))
            finally:
                support_tools._TICKETS = original_ticket_path

    def test_dataset_counts_match_expected_synthetic_fixture(self) -> None:
        item_count = sum(len(items) for items in support_tools._ITEMS.values())
        self.assertEqual(len(support_tools._ORDERS), 1000)
        self.assertEqual(len(support_tools._CUSTOMERS), 300)
        self.assertEqual(item_count, 1783)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SupportToolRegressionTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
