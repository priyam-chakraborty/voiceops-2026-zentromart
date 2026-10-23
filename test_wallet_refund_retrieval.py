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
        "text": f"Synthetic evidence fixture for {section_id}.",
    }


def vector_for(cosine_score: float) -> list[float]:
    return [cosine_score, math.sqrt(max(0.0, 1.0 - cosine_score * cosine_score))]


class WalletRefundRetrievalTests(unittest.TestCase):
    """Offline synthetic-score regression test for practice question PQ0003."""

    def test_wallet_refund_query_ranks_ref2_above_generic_faq(self):
        question = "How fast are refunds to the ZentroMart Wallet?"
        candidates = [
            # Deliberately give the generic FAQ a higher raw similarity score.
            (chunk("FAQ-1", doc_type="faq"), 0.79),
            (chunk("KB-00.1", doc_type="agent"), 0.73),
            (chunk("REF-3"), 0.72),
            (chunk("REF-2"), 0.70),
            (chunk("RET-2"), 0.64),
        ]
        chunks = [item for item, _ in candidates]
        vectors = [vector_for(score) for _, score in candidates]

        with patch.object(run, "_embed_texts", return_value=[[1.0, 0.0]]):
            result = run.retrieve(None, question, chunks, vectors, top_k=5)

        ranked_ids = [item["section_id"] for item in result]
        self.assertEqual(ranked_ids[0], "REF-2")
        self.assertIn("KB-00.1", ranked_ids[:3])
        self.assertGreater(ranked_ids.index("REF-2"), -1)

    def test_wallet_refund_policy_bonus_is_greater_than_overview_bonus(self):
        question = "How fast are refunds to the ZentroMart Wallet?"
        ref2_bonus = run._retrieval_topic_bonus(question, chunk("REF-2"))
        overview_bonus = run._retrieval_topic_bonus(question, chunk("KB-00.1", doc_type="agent"))
        faq_bonus = run._retrieval_topic_bonus(question, chunk("FAQ-1", doc_type="faq"))

        self.assertGreater(ref2_bonus, overview_bonus)
        self.assertGreater(overview_bonus, faq_bonus)


if __name__ == "__main__":
    unittest.main(verbosity=2)
