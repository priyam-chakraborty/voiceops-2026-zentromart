from __future__ import annotations

import unittest
from unittest.mock import patch

import run


def make_chunk(section_id: str, *, status: str = "current", doc_type: str = "policy") -> dict:
    return {
        "source": f"{section_id.lower()}.md",
        "section_id": section_id,
        "title": section_id,
        "doc_type": doc_type,
        "status": status,
        "effective_from": "2026-01-01",
        "text": f"Synthetic evidence text for {section_id}.",
    }


class RetrievalStatusPriorityTests(unittest.TestCase):
    def setUp(self) -> None:
        # Similarity vector scores: superseded .98, current policy .60, agent guide .61.
        # The raw vector similarity would put the superseded document first without filtering.
        self.chunks = [
            make_chunk("RET-3", status="superseded", doc_type="policy"),
            make_chunk("RET-2", status="current", doc_type="policy"),
            make_chunk("AGENT-RET", status="current", doc_type="agent_guide"),
        ]
        self.vectors = [
            [0.98, (1 - 0.98**2) ** 0.5, 0.0],
            [0.60, 0.80, 0.0],
            [0.61, (1 - 0.61**2) ** 0.5, 0.0],
        ]
        self.embed_patch = patch.object(run, "_embed_texts", return_value=[[1.0, 0.0, 0.0]])
        self.embed_patch.start()
        self.addCleanup(self.embed_patch.stop)

    def test_current_policy_query_excludes_superseded_section_even_if_similarity_is_higher(self):
        results = run.retrieve(None, "What is the current return window?", self.chunks, self.vectors, top_k=5)
        ids = [item["section_id"] for item in results]
        self.assertIn("RET-2", ids)
        self.assertNotIn("RET-3", ids)

    def test_explicit_historical_query_can_retrieve_superseded_policy(self):
        results = run.retrieve(None, "What did the previous return policy say before the update?", self.chunks, self.vectors, top_k=5)
        ids = [item["section_id"] for item in results]
        self.assertIn("RET-3", ids)

    def test_current_policy_precedence_wins_close_score_against_agent_guide(self):
        results = run.retrieve(None, "What is the current return policy?", self.chunks, self.vectors, top_k=5)
        ids = [item["section_id"] for item in results]
        self.assertLess(ids.index("RET-2"), ids.index("AGENT-RET"))

    def test_only_superseded_candidates_return_no_current_policy_evidence_for_normal_query(self):
        only_superseded = [self.chunks[0]]
        only_vector = [self.vectors[0]]
        results = run.retrieve(None, "What is the current return window?", only_superseded, only_vector, top_k=5)
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
