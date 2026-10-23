#!/usr/bin/env python3
"""Offline rubric audit of all 52 supplied reference answers.

Important: this tests the evaluator rubric against dataset reference answers,
NOT generated agent answers and NOT the official capstone score.
"""
from __future__ import annotations
import unittest
import evaluate


class AllReferenceAnswersRubricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = evaluate.read_questions()
        cls.by_id = {r.get("id", ""): r for r in cls.rows}

    def test_dataset_has_all_52_practice_questions(self):
        self.assertEqual(len(self.rows), 52)
        self.assertEqual(len(self.by_id), 52, "Practice question IDs must be unique")

    def test_all_52_reference_answers_pass_the_documented_heuristic(self):
        failures = []
        for row in self.rows:
            answer = (row.get("reference_answer") or "").strip()
            scores = evaluate.score_answer(answer, row)
            if not answer or scores["heuristic_pass"] is not True:
                failures.append((row.get("id"), scores))
        self.assertEqual(failures, [], f"Reference-answer rubric failures: {failures}")

    def test_hindi_clarification_reference_is_recognized(self):
        row = self.by_id["PQ0018"]
        result = evaluate.score_answer(row["reference_answer"], row)
        self.assertTrue(result["decision_heuristic_pass"], result)
        self.assertTrue(result["heuristic_pass"], result)

    def test_unanswerable_cases_require_uncertainty_in_reference_answers(self):
        rows = [r for r in self.rows if str(r.get("answerable", "")).casefold() == "false"]
        self.assertEqual(len(rows), 5)
        failed = []
        for row in rows:
            result = evaluate.score_answer(row["reference_answer"], row)
            if result["answerability_pass"] is not True:
                failed.append(row.get("id"))
        self.assertEqual(failed, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
