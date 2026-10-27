"""Offline regression tests for multilingual uncertainty, clarification and prompts."""
from __future__ import annotations

import unittest

import evaluate
import run


class ResponseQualityGuardrailTests(unittest.TestCase):
    def test_hindi_uncertainty_counts_for_unanswerable_kocchi_store_question(self):
        answer = (
            "उपलब्ध दस्तावेज़ों के आधार पर मैं यह पुष्टि नहीं कर सकता कि "
            "कोच्चि में ZentroMart की कोई भौतिक दुकान है या नहीं।"
        )
        row = {"answerable": "False"}
        self.assertTrue(evaluate.answerability_pass(answer, row))

    def test_hindi_human_handover_counts_for_required_escalation_group(self):
        answer = (
            "मैं यह पुष्टि नहीं कर सकता। यदि आप चाहें, तो मैं आपको "
            "हमारे मानव सहायता प्रतिनिधि से जोड़ सकता हूँ।"
        )
        field = (
            "don't have / do not have / not sure / no information / can't find / "
            "not able to find / couldn't find / जानकारी नहीं / pata nahi / "
            "jaankari nahi / connect / सहयोगी"
        )
        met, total, missed = evaluate.groups_met(answer, field)
        self.assertEqual((met, total, missed), (1, 1, []))

    def test_hindi_return_clarification_is_recognized(self):
        answer = "कृपया प्रोडक्ट की कैटेगरी या जानकारी साझा करें!"
        row = {"expected_decision": "clarify", "acceptable_decisions": "clarify"}
        self.assertTrue(evaluate.decision_pass(answer, row))

    def test_rag_prompt_requires_latest_refund_date_and_clarify_first(self):
        prompt = run.RAG_SYSTEM_PROMPT.casefold()
        self.assertIn("estimated latest calendar date", prompt)
        self.assertIn("next business day", prompt)
        self.assertIn("ask one short clarifying question first", prompt)
        self.assertIn("do not list every category-specific return window", prompt)

    def test_step3_prompt_has_same_refund_and_clarification_guardrails(self):
        prompt = run.STEP3_SYSTEM_PROMPT.casefold()
        self.assertIn("estimated latest calendar date", prompt)
        self.assertIn("next business day", prompt)
        self.assertIn("ask one short clarifying question first", prompt)


if __name__ == "__main__":
    unittest.main()
