"""Regression tests for order-ID and verification-token parsing; no API calls."""
from __future__ import annotations
import unittest
import run


class IdentifierParsingTests(unittest.TestCase):
    def test_plain_order_id_is_detected(self):
        self.assertEqual(run._extract_order_id("Where is my order 69520867?"), "69520867")

    def test_spoken_grouped_order_id_is_detected(self):
        self.assertEqual(run._extract_order_id("My order number is 69 52 08 67"), "69520867")
        self.assertEqual(run._extract_order_id("My order number is 69-52-08-67"), "69520867")

    def test_iso_date_is_not_mistaken_for_order_id(self):
        self.assertIsNone(run._extract_order_id("Can you help with the order placed on 2026-10-12?"))
        self.assertIsNone(run._extract_order_id("Can you help with the order placed on 12-10-2026?"))

    def test_date_does_not_hide_real_order_id_or_verification_value(self):
        question = "My order 69520867 from 2026-10-12 has a problem; pincode 160017."
        self.assertEqual(run._extract_order_id(question), "69520867")
        self.assertEqual(run._extract_verification_value(question, "69520867"), "160017")

    def test_product_sku_digits_are_not_treated_as_verification_value(self):
        question = "My order 69520867 about SKU ELE-0012 needs warranty help."
        self.assertEqual(run._extract_order_id(question), "69520867")
        self.assertIsNone(run._extract_verification_value(question, "69520867"))

    def test_real_phone_suffix_and_pincode_still_parse(self):
        self.assertEqual(run._extract_verification_value("My order 69520867, last four digits 1234", "69520867"), "1234")
        self.assertEqual(run._extract_verification_value("My order 69520867, delivery pincode 160017", "69520867"), "160017")

    def test_sku_with_multiple_numbers_does_not_yield_fake_proof(self):
        question = "Order 69520867 contains SKU ELE-0012 and model ABC-2026."
        self.assertIsNone(run._extract_verification_value(question, "69520867"))


    # VOICEOPS_OFFLINE_FOLLOWUP_DATE_TESTS_V1
    def test_slash_and_dot_dates_are_not_mistaken_for_order_ids(self):
        dates = ("2026/10/12", "12/10/2026", "2026.10.12", "12.10.2026", "2026 / 10 / 12")
        for date_text in dates:
            with self.subTest(date=date_text):
                self.assertIsNone(run._extract_order_id(f"Can you help with the order placed on {date_text}?"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
