"""Offline regression tests for VoiceOps evaluator score denominators.

These tests call evaluate.main() with a tiny synthetic CSV and mocked agent
functions. They make no Gemini/Gnani network calls and overwrite only files in
an automatically removed temporary directory.
"""
from __future__ import annotations

import contextlib
import csv
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import evaluate


FIELDNAMES = [
    "id",
    "question",
    "asr_text",
    "expected_decision",
    "acceptable_decisions",
    "answerable",
    "gold_evidence",
    "must_include",
    "must_not_include",
    "hard_negatives",
]

SYNTHETIC_ROWS = [
    {
        "id": "SYN-PASS",
        "question": "PASS case",
        "asr_text": "PASS case",
        "expected_decision": "answer",
        "acceptable_decisions": "answer",
        "answerable": "true",
        "gold_evidence": "RET-5",
        "must_include": "refunds",
        "must_not_include": "",
        "hard_negatives": "",
    },
    {
        "id": "SYN-ERROR",
        "question": "ERROR case",
        "asr_text": "ERROR case",
        "expected_decision": "answer",
        "acceptable_decisions": "answer",
        "answerable": "true",
        "gold_evidence": "RET-5",
        "must_include": "refunds",
        "must_not_include": "",
        "hard_negatives": "",
    },
    {
        "id": "SYN-PROVIDER-ERROR",
        "question": "PROVIDER ERROR case",
        "asr_text": "PROVIDER ERROR case",
        "expected_decision": "answer",
        "acceptable_decisions": "answer",
        "answerable": "true",
        "gold_evidence": "RET-5",
        "must_include": "refunds",
        "must_not_include": "",
        "hard_negatives": "",
    },
    {
        "id": "SYN-FAIL",
        "question": "FAIL case",
        "asr_text": "FAIL case",
        "expected_decision": "answer",
        "acceptable_decisions": "answer",
        "answerable": "true",
        "gold_evidence": "RET-5",
        "must_include": "warranty",
        "must_not_include": "",
        "hard_negatives": "",
    },
]

SAMPLE_CHUNK = {
    "source": "pol_returns_v4.md",
    "section_id": "RET-5",
    "doc_type": "policy",
    "status": "current",
    "effective_from": "2026-01-01",
    "text": "Synthetic test excerpt for return policy retrieval.",
}


class EvaluatorDenominatorTests(unittest.TestCase):
    def _run_synthetic_evaluation(self, mode: str) -> tuple[str, list[dict[str, str]]]:
        with tempfile.TemporaryDirectory(prefix="voiceops_eval_test_") as temp:
            root = Path(temp)
            questions_file = root / "questions.csv"
            with questions_file.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
                writer.writeheader()
                writer.writerows(SYNTHETIC_ROWS)

            def fake_generate_answer(_client, question, _system_prompt, *_args):
                if question == "ERROR case":
                    raise RuntimeError("synthetic forced failure; no network call")
                if question == "PROVIDER ERROR case":
                    raise RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded; synthetic no-network provider failure")
                if question == "PASS case":
                    return "Refunds are processed within five days."
                return "Your delivery will arrive tomorrow."

            args = ["evaluate.py", "--mode", mode, "--limit", "0"]
            with (
                patch.object(evaluate, "ROOT", root),
                patch.object(evaluate, "QUESTIONS_FILE", questions_file),
                patch.object(evaluate.agent, "require_dataset"),
                patch.object(evaluate.agent, "get_client", return_value=object()),
                patch.object(evaluate.agent, "generate_answer", side_effect=fake_generate_answer),
                patch("sys.argv", args),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                if mode == "step2":
                    with (
                        patch.object(
                            evaluate.agent,
                            "load_or_build_index",
                            return_value=([SAMPLE_CHUNK], [[0.0]]),
                        ),
                        patch.object(evaluate.agent, "retrieve", return_value=[SAMPLE_CHUNK]),
                    ):
                        return_code = evaluate.main()
                else:
                    return_code = evaluate.main()

            self.assertEqual(return_code, 0)
            summary = (root / "evaluation_summary.txt").read_text(encoding="utf-8")
            with (root / "evaluation_results.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                results = list(csv.DictReader(handle))
            return summary, results

    def _assert_denominator_behavior(self, mode: str) -> None:
        summary, rows = self._run_synthetic_evaluation(mode)
        prefix = f"{mode.upper()}"
        self.assertIn(f"{prefix} provider/API errors excluded from score: 1", summary)
        self.assertIn(f"{prefix} evaluation errors excluded from score: 1", summary)
        self.assertIn(f"{prefix} heuristic passes: 1/2 (50.0%)", summary)
        self.assertEqual(len(rows), 4)
        error_row = next(row for row in rows if row["id"] == "SYN-ERROR")
        self.assertEqual(error_row[f"{mode}_status"], "evaluation_error")
        self.assertEqual(error_row[f"{mode}_heuristic_pass"], "")
        provider_error_row = next(row for row in rows if row["id"] == "SYN-PROVIDER-ERROR")
        self.assertEqual(provider_error_row[f"{mode}_status"], "provider_error")
        self.assertEqual(provider_error_row[f"{mode}_heuristic_pass"], "")
        pass_row = next(row for row in rows if row["id"] == "SYN-PASS")
        fail_row = next(row for row in rows if row["id"] == "SYN-FAIL")
        self.assertEqual(pass_row[f"{mode}_heuristic_pass"].casefold(), "true")
        self.assertEqual(fail_row[f"{mode}_heuristic_pass"].casefold(), "false")

    def test_step1_provider_error_is_excluded_from_score_denominator(self):
        self._assert_denominator_behavior("step1")

    def test_step2_provider_error_is_excluded_from_score_denominator(self):
        self._assert_denominator_behavior("step2")


if __name__ == "__main__":
    unittest.main(verbosity=2)
