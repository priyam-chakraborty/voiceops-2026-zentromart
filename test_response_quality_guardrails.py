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
        self.assertIn("business days run monday through saturday", prompt)
        self.assertIn("start counting on the next calendar date after the initiation date as day 1", prompt)
        self.assertIn("skip sundays and dates explicitly listed as holidays", prompt)
        self.assertIn("do not automatically skip saturdays", prompt)
        self.assertIn("ask one short clarifying question first", prompt)
        self.assertIn("do not list every category-specific return window", prompt)

        # VOICEOPS_REFUND_CALENDAR_FORCE_TEST_V1
        dated_question = "I initiated a credit card refund on 17 October. By which date should it arrive?"
        self.assertTrue(run._is_date_specific_refund_question(dated_question))
        self.assertFalse(run._is_date_specific_refund_question("What is your refund policy?"))

        calendar_chunk = {"section_id": "KB-00.3", "status": "current", "text": "calendar"}
        refund_policy = {"section_id": "REF-2", "status": "current", "retrieval_score": 0.8}
        selected = run._append_refund_calendar_evidence(
            [refund_policy], [calendar_chunk], [(0.075, 0)], dated_question, False
        )
        self.assertEqual([item["section_id"] for item in selected], ["REF-2", "KB-00.3"])

        already_selected = run._append_refund_calendar_evidence(
            [refund_policy, calendar_chunk], [calendar_chunk], [(0.075, 0)], dated_question, False
        )
        self.assertEqual([item["section_id"] for item in already_selected].count("KB-00.3"), 1)

        historical = run._append_refund_calendar_evidence(
            [refund_policy], [calendar_chunk], [(0.075, 0)], dated_question, True
        )
        self.assertEqual([item["section_id"] for item in historical], ["REF-2"])

    def test_step3_prompt_has_same_refund_and_clarification_guardrails(self):
        prompt = run.STEP3_SYSTEM_PROMPT.casefold()
        self.assertIn("estimated latest calendar date", prompt)
        self.assertIn("business days run monday through saturday", prompt)
        self.assertIn("start counting on the next calendar date after the initiation date as day 1", prompt)
        self.assertIn("skip sundays and dates explicitly listed as holidays", prompt)
        self.assertIn("do not automatically skip saturdays", prompt)
        self.assertIn("ask one short clarifying question first", prompt)


if __name__ == "__main__":
    unittest.main()
