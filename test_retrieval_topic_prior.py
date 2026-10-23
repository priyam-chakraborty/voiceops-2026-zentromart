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
# VOICEOPS_RETRIEVAL_CONTEXT_ISOLATION_TESTS_V1
import unittest as _voiceops_unittest
from unittest.mock import patch as _voiceops_patch
import run as _voiceops_retrieval_run


class RetrievalContextIsolationTests(_voiceops_unittest.TestCase):
    @staticmethod
    def _fixtures():
        current_question = "How long does a wallet refund take now?"
        contextual_query = (
            "Recent customer context: What was the old return policy? | "
            "Current question: " + current_question
        )
        chunks = [
            {
                "section_id": "REF-CURRENT-1",
                "source": "pol_refunds.md",
                "doc_type": "policy",
                "status": "current",
                "effective_from": "2026-01-01",
                "text": "Current wallet refund timing policy.",
            },
            {
                "section_id": "REF-OLD-1",
                "source": "pol_old_refunds.md",
                "doc_type": "policy",
                "status": "superseded",
                "effective_from": "2024-01-01",
                "text": "Superseded historical refund timing policy.",
            },
        ]
        vectors = [[1.0, 0.0], [0.99, 0.01]]
        return current_question, contextual_query, chunks, vectors

    def test_history_mention_does_not_enable_superseded_policy_for_current_question(self):
        current_question, contextual_query, chunks, vectors = self._fixtures()
        with (
            _voiceops_patch.object(_voiceops_retrieval_run, "_embed_texts", return_value=[[1.0, 0.0]]) as embed,
            _voiceops_patch.object(
                _voiceops_retrieval_run,
                "_asks_for_historical_policy",
                wraps=_voiceops_retrieval_run._asks_for_historical_policy,
            ) as historical_check,
            _voiceops_patch.object(_voiceops_retrieval_run, "_retrieval_topic_bonus", return_value=0.0),
            _voiceops_patch.object(_voiceops_retrieval_run, "_product_catalog_chunks_for_question", return_value=[]),
        ):
            selected = _voiceops_retrieval_run.retrieve(
                None, contextual_query, chunks, vectors, topic_question=current_question
            )

        historical_check.assert_called_once_with(current_question)
        self.assertEqual(embed.call_args.args[1], [contextual_query])
        self.assertIn("REF-CURRENT-1", [chunk["section_id"] for chunk in selected])
        self.assertNotIn("REF-OLD-1", [chunk["section_id"] for chunk in selected])

    def test_topic_bonuses_use_current_turn_not_enriched_history(self):
        current_question, contextual_query, chunks, vectors = self._fixtures()
        with (
            _voiceops_patch.object(_voiceops_retrieval_run, "_embed_texts", return_value=[[1.0, 0.0]]),
            _voiceops_patch.object(_voiceops_retrieval_run, "_asks_for_historical_policy", return_value=False),
            _voiceops_patch.object(_voiceops_retrieval_run, "_retrieval_topic_bonus", return_value=0.0) as bonus,
            _voiceops_patch.object(_voiceops_retrieval_run, "_product_catalog_chunks_for_question", return_value=[]),
        ):
            _voiceops_retrieval_run.retrieve(
                None, contextual_query, chunks, vectors, topic_question=current_question
            )

        self.assertTrue(bonus.call_args_list)
        self.assertTrue(all(call.args[0] == current_question for call in bonus.call_args_list))

    def test_legacy_call_uses_its_question_as_topic_question(self):
        current_question, _, chunks, vectors = self._fixtures()
        with (
            _voiceops_patch.object(_voiceops_retrieval_run, "_embed_texts", return_value=[[1.0, 0.0]]),
            _voiceops_patch.object(_voiceops_retrieval_run, "_asks_for_historical_policy", return_value=False) as historical_check,
            _voiceops_patch.object(_voiceops_retrieval_run, "_retrieval_topic_bonus", return_value=0.0) as bonus,
            _voiceops_patch.object(_voiceops_retrieval_run, "_product_catalog_chunks_for_question", return_value=[]),
        ):
            _voiceops_retrieval_run.retrieve(None, current_question, chunks, vectors)

        historical_check.assert_called_once_with(current_question)
        self.assertTrue(bonus.call_args_list)
        self.assertTrue(all(call.args[0] == current_question for call in bonus.call_args_list))
