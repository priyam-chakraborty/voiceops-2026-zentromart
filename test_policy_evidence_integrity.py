"""Offline integrity checks for policy evidence ingested by run.py.

These tests validate document chunking and metadata only. They do not call Gemini
or measure semantic embedding retrieval/LLM answer quality.
Run from the project root: py test_policy_evidence_integrity.py
"""
import unittest

try:
    from run import read_knowledge_chunks
except Exception as exc:  # show actionable failure when project deps/data are unavailable
    raise SystemExit(
        "Could not import read_knowledge_chunks from run.py. Run this file beside run.py, "
        f"and install requirements first. Import error: {type(exc).__name__}: {exc}"
    )


class PolicyEvidenceIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chunks = read_knowledge_chunks()
        cls.by_id = {str(c.get("section_id", "")): c for c in cls.chunks}

    def require_section(self, section_id):
        self.assertIn(section_id, self.by_id, f"Required policy section missing: {section_id}")
        return self.by_id[section_id]

    def test_all_84_sections_parse_and_have_unique_ids(self):
        ids = [str(c.get("section_id", "")) for c in self.chunks]
        self.assertEqual(len(self.chunks), 84, "Expected 84 knowledge-base sections from the supplied dataset")
        self.assertTrue(all(ids), "Every chunk must have a section ID")
        self.assertEqual(len(ids), len(set(ids)), "Section IDs should be unique")

    def test_current_return_policy_is_not_confused_with_superseded_policy(self):
        current = self.require_section("RET-2")
        old = self.require_section("RETv3-2")
        self.assertEqual(current["source"], "pol_returns_v4.md")
        self.assertEqual(current["doc_type"], "policy")
        self.assertEqual(current["status"], "current")
        self.assertEqual(current["effective_from"], "2026-08-01")
        self.assertIn("7 days", current["text"].lower())
        self.assertEqual(old["source"], "pol_returns_v3_superseded.md")
        self.assertEqual(old["status"], "superseded")

    def test_refund_timeline_evidence_is_present_and_policy_authoritative(self):
        refund = self.require_section("REF-2")
        self.assertEqual(refund["source"], "pol_refunds.md")
        self.assertEqual(refund["doc_type"], "policy")
        self.assertEqual(refund["status"], "current")
        text = refund["text"].lower()
        for phrase in ("upi", "1 to 3 business days", "5 to 7 business days", "within 2 hours"):
            self.assertIn(phrase, text)

    def test_warranty_policy_distinguishes_assured_and_marketplace_seller_items(self):
        assured = self.require_section("WAR-2")
        seller = self.require_section("MKS-3")
        self.assertEqual(assured["doc_type"], "policy")
        self.assertEqual(assured["status"], "current")
        self.assertIn("12-month", assured["text"].lower())
        self.assertEqual(seller["status"], "current")
        self.assertIn("never applies to marketplace seller items", seller["text"].lower())

    def test_escalation_threshold_and_create_ticket_instruction_are_present(self):
        escalation = self.require_section("ESC-1")
        action = self.require_section("ESC-2")
        self.assertEqual(escalation["doc_type"], "agent_guide")
        self.assertEqual(escalation["status"], "current")
        self.assertIn("₹10,000", escalation["text"])
        self.assertIn("create_ticket", action["text"])
        self.assertIn("before handing over", action["text"].lower())

    def test_knowledge_base_precedence_rule_is_loaded(self):
        precedence = self.require_section("KB-00.1")
        text = precedence["text"].lower()
        self.assertIn("policy documents", text)
        self.assertIn("status `current`", text)
        self.assertIn("superseded", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
