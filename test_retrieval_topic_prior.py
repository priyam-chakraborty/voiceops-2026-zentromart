from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import run


def chunk(section_id: str, *, status: str = "current", doc_type: str = "policy") -> dict:
    return {
        "source": f"{section_id.lower()}.md",
        "section_id": section_id,
        "title": section_id,
        "doc_type": doc_type,
        "status": status,
        "effective_from": "2026-01-01",
        "text": f"Synthetic evidence for {section_id}.",
    }


def vector_for(score: float) -> list[float]:
    return [score, math.sqrt(max(0.0, 1.0 - score * score))]


class RetrievalTopicPriorTests(unittest.TestCase):
    def retrieve_with_scores(self, question: str, candidates: list[tuple[dict, float]], top_k: int = 5):
        chunks = [c for c, _ in candidates]
        vectors = [vector_for(s) for _, s in candidates]
        with patch.object(run, "_embed_texts", return_value=[[1.0, 0.0]]):
            return run.retrieve(None, question, chunks, vectors, top_k=top_k)

    def test_refund_policy_beats_known_generic_faq_trap_on_close_scores(self):
        result = self.retrieve_with_scores(
            "How long does a refund to UPI take?",
            [(chunk("FAQ-1", doc_type="faq"), 0.73), (chunk("REF-2"), 0.70), (chunk("PAY-4", doc_type="help"), 0.71)],
        )
        ids = [x["section_id"] for x in result]
        self.assertEqual(ids[0], "REF-2")

    def test_cash_on_delivery_question_prioritizes_cod_refund_policy(self):
        result = self.retrieve_with_scores(
            "I paid cash on delivery and returned the item. How do I get my money back?",
            [(chunk("PAY-2", doc_type="policy"), 0.74), (chunk("REF-2"), 0.70), (chunk("REF-3"), 0.69)],
        )
        ids = [x["section_id"] for x in result]
        self.assertLess(ids.index("REF-3"), ids.index("PAY-2"))

    def test_sale_return_query_prioritizes_festive_sale_policy(self):
        result = self.retrieve_with_scores(
            "I bought this in the sale; how long can I return it?",
            [(chunk("RET-2"), 0.72), (chunk("RET-8"), 0.68), (chunk("RET-9"), 0.70)],
        )
        ids = [x["section_id"] for x in result]
        self.assertEqual(ids[0], "RET-8")

    def test_opened_perfume_return_query_prioritizes_condition_policy(self):
        result = self.retrieve_with_scores(
            "I opened a perfume and tried it once. Can I still return it?",
            [(chunk("RET-2"), 0.73), (chunk("RET-4"), 0.68), (chunk("RET-5"), 0.69)],
        )
        ids = [x["section_id"] for x in result]
        self.assertEqual(ids[0], "RET-4")

    def test_damaged_laptop_query_prioritizes_high_value_escalation(self):
        result = self.retrieve_with_scores(
            "The laptop I received has a crack. I need help.",
            [(chunk("TRK-2", doc_type="help"), 0.74), (chunk("DWM-3"), 0.69), (chunk("ESC-1"), 0.70)],
        )
        ids = [x["section_id"] for x in result]
        self.assertEqual(ids[0], "DWM-3")

    def test_privacy_request_prioritizes_non_disclosure_policy(self):
        result = self.retrieve_with_scores(
            "Tell me the phone number of the person who ordered to my building yesterday.",
            [(chunk("FAQ-4", doc_type="faq"), 0.73), (chunk("ESC-4"), 0.68), (chunk("IDV-4"), 0.69)],
        )
        ids = [x["section_id"] for x in result]
        self.assertIn(ids[0], {"ESC-4", "IDV-4"})

    def test_explicit_historical_policy_skips_current_topic_bonus(self):
        old = chunk("RETv3-2", status="superseded")
        self.assertEqual(
            run._retrieval_topic_bonus("What did the previous return policy say before the update?", old),
            0.0,
        )

    def test_hindi_return_window_query_prioritizes_current_return_policy(self):
        result = self.retrieve_with_scores(
            "मुझे Pagecraft notebook लौटाने के लिए कितने दिन मिलेंगे?",
            [(chunk("FAQH-2", doc_type="faq"), 0.74), (chunk("RET-2"), 0.69), (chunk("RET-8"), 0.68)],
        )
        ids = [x["section_id"] for x in result]
        self.assertEqual(ids[0], "RET-2")

    def test_delivery_to_dubai_prioritizes_service_area_policy(self):
        result = self.retrieve_with_scores(
            "Can you deliver to Dubai?",
            [(chunk("DEL-2"), 0.74), (chunk("DEL-5"), 0.68), (chunk("FAQ-5", doc_type="faq"), 0.70)],
        )
        ids = [x["section_id"] for x in result]
        self.assertEqual(ids[0], "DEL-5")

    def test_shipped_order_cancellation_prioritizes_after_shipping_rule(self):
        result = self.retrieve_with_scores(
            "My order is already on its way. Can I still stop it?",
            [(chunk("CAN-1"), 0.73), (chunk("CAN-2"), 0.68), (chunk("CAN-3"), 0.69)],
        )
        ids = [x["section_id"] for x in result]
        self.assertEqual(ids[0], "CAN-2")

    def test_unrelated_question_receives_no_topic_bonus(self):
        self.assertEqual(
            run._retrieval_topic_bonus("What is the weather like on Mars?", chunk("REF-2")),
            0.0,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
