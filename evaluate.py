"""Beginner-friendly VoiceOps practice-question evaluator.

Run from the VoiceOps project root. This uses the existing run.py functions and
writes evaluation_results.csv plus evaluation_summary.txt. Scores are transparent
heuristics, not official organizer scores or an LLM-as-judge result.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

import run as agent

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "Capstone_Project_Dataset_FTTT_2026"
QUESTIONS_FILE = DATASET / "practice" / "questions.csv"

UNCERTAINTY_MARKERS = (
    "cannot confirm", "can't confirm", "cannot verify", "don't know", "do not know",
    "not sure", "not available", "no information", "not covered", "i don't have",
    "i do not have", "can't find", "cannot find", "unable to confirm", "unable to find",
    "not able to find", "connect you", "human support", "colleague", "जानकारी नहीं",
    "पता नहीं", "jaankari nahi", "pata nahi",
)
ESCALATION_MARKERS = (
    "human", "colleague", "support team", "support ticket", "ticket", "handover",
    "hand over", "priority resolution", "escalat", "सहयोगी",
)
CLARIFY_MARKERS = (
    "please share", "please provide", "could you provide", "which", "what is your",
    "what's your", "can you tell me", "could you tell me", "order id", "order number",
    "last four", "pincode", "pin code",
)
REFUSAL_MARKERS = (
    "cannot approve", "can't approve", "cannot do that", "can't do that", "i cannot",
    "i can't", "i am unable", "i'm unable", "not permitted", "not allowed", "cannot help with that",
)


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").casefold()).strip()


def split_groups(field: str) -> list[list[str]]:
    """Interpret semicolon groups as separate requirements and slash items as variants."""
    groups: list[list[str]] = []
    for group in (field or "").split(";"):
        options = [norm(x).strip(" .,:;\t\n") for x in group.split("/")]
        options = [x for x in options if x]
        if options:
            groups.append(options)
    return groups


def groups_met(answer: str, field: str) -> tuple[int, int, list[str]]:
    text = norm(answer)
    groups = split_groups(field)
    missed: list[str] = []
    met = 0
    for group in groups:
        if any(option in text for option in group):
            met += 1
        else:
            missed.append(" / ".join(group))
    return met, len(groups), missed


def answerability_pass(answer: str, row: dict[str, str]) -> bool | None:
    answerable = norm(row.get("answerable", ""))
    if answerable not in {"true", "false"}:
        return None
    if answerable == "true":
        return True
    text = norm(answer)
    return any(marker in text for marker in UNCERTAINTY_MARKERS)


def decision_pass(answer: str, row: dict[str, str]) -> bool | None:
    expected = norm(row.get("expected_decision", ""))
    acceptable = [norm(x) for x in (row.get("acceptable_decisions") or expected).split(";") if norm(x)]
    if not acceptable:
        return None
    text = norm(answer)
    checks: dict[str, bool] = {
        "answer": bool(text),
        "escalate": any(marker in text for marker in ESCALATION_MARKERS),
        "clarify": ("?" in (answer or "")) or any(marker in text for marker in CLARIFY_MARKERS),
        "refuse": any(marker in text for marker in REFUSAL_MARKERS),
    }
    recognized = [checks[x] for x in acceptable if x in checks]
    return any(recognized) if recognized else None


def score_answer(answer: str, row: dict[str, str]) -> dict[str, Any]:
    required_hit, required_total, missed_required = groups_met(answer, row.get("must_include", ""))
    banned_groups = split_groups(row.get("must_not_include", "")) + split_groups(row.get("hard_negatives", ""))
    text = norm(answer)
    hit_banned = [" / ".join(group) for group in banned_groups if any(option in text for option in group)]
    banned_pass = not hit_banned
    answerability_ok = answerability_pass(answer, row)
    decision_ok = decision_pass(answer, row)
    required_pass = required_hit == required_total
    core_checks = [required_pass, banned_pass]
    if answerability_ok is not None:
        core_checks.append(answerability_ok)
    if decision_ok is not None:
        core_checks.append(decision_ok)
    return {
        "required_groups_met": required_hit,
        "required_groups_total": required_total,
        "required_groups_pass": required_pass,
        "missed_required_groups": " | ".join(missed_required),
        "banned_terms_pass": banned_pass,
        "triggered_banned_groups": " | ".join(hit_banned),
        "answerability_pass": answerability_ok,
        "decision_heuristic_pass": decision_ok,
        "heuristic_pass": all(core_checks),
    }


def transient_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(token in message for token in ("503", "unavailable", "429", "resource_exhausted", "rate limit", "temporarily"))


def with_retry(label: str, fn: Callable[[], Any]) -> Any:
    for attempt in range(3):
        try:
            return fn()
        except Exception as exc:
            if attempt >= 2 or not transient_error(exc):
                raise
            delay = 3 * (attempt + 1)
            print(f"  {label}: temporary service/rate error; retrying in {delay}s (attempt {attempt + 2}/3)")
            time.sleep(delay)
    raise RuntimeError("Unexpected retry loop exit")


def read_questions() -> list[dict[str, str]]:
    if not QUESTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {QUESTIONS_FILE}. Make sure the dataset folder is inside the project folder."
        )
    with QUESTIONS_FILE.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def make_rag_context(sources: list[dict[str, Any]]) -> str:
    parts = []
    for i, chunk in enumerate(sources, start=1):
        parts.append(
            f"[SOURCE {i}] file={chunk['source']}; section=[{chunk['section_id']}]; "
            f"doc_type={chunk['doc_type']}; status={chunk['status']}; "
            f"effective_from={chunk['effective_from']}\n{chunk['text']}"
        )
    return "Retrieved company-document excerpts (may not fully answer the question):\n\n" + "\n\n".join(parts)


def gold_kb_ids(raw: str, known_ids: set[str]) -> list[str]:
    ids = re.findall(r"\b[A-Z][A-Z0-9]*-\d+(?:\.\d+)?\b", raw or "", flags=re.I)
    return sorted({x for x in ids if x in known_ids})


def bool_cell(value: Any) -> str:
    if value is None:
        return "N/A"
    return "PASS" if value else "FAIL"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare prompt-only vs RAG on VoiceOps practice questions.")
    parser.add_argument("--limit", type=int, default=5,
                        help="Number of questions to evaluate (default: 5). Use 0 for all 52.")
    parser.add_argument("--start", type=int, default=0, help="Skip this many questions before starting.")
    parser.add_argument("--mode", choices=["both", "step1", "step2"], default="both",
                        help="Which pipeline(s) to run. Default: both.")
    parser.add_argument("--use-asr-text", action="store_true",
                        help="Use the dataset's ASR-like transcription instead of the clean question text.")
    args = parser.parse_args()

    agent.require_dataset()
    rows = read_questions()
    selected = rows[max(0, args.start):]
    if args.limit > 0:
        selected = selected[:args.limit]
    if not selected:
        print("No questions selected. Check --start and --limit.")
        return 2

    client = agent.get_client()
    chunks: list[dict[str, Any]] = []
    vectors: list[list[float]] = []
    known_ids: set[str] = set()
    if args.mode in {"both", "step2"}:
        chunks, vectors = with_retry("Build/load RAG index", lambda: agent.load_or_build_index(client))
        known_ids = {str(c.get("section_id", "")) for c in chunks}

    result_rows: list[dict[str, Any]] = []
    for number, row in enumerate(selected, start=1):
        qid = row.get("id", f"row-{number}")
        question = (row.get("asr_text") if args.use_asr_text else row.get("question")) or row.get("question", "")
        item: dict[str, Any] = {
            "id": qid,
            "language": row.get("language", ""),
            "question_used": question,
            "expected_decision": row.get("expected_decision", ""),
            "acceptable_decisions": row.get("acceptable_decisions", ""),
            "answerable": row.get("answerable", ""),
            "gold_evidence": row.get("gold_evidence", ""),
        }
        row_gold_ids = gold_kb_ids(row.get("gold_evidence", ""), known_ids) if known_ids else []
        item["gold_kb_section_ids"] = ";".join(row_gold_ids)

        if args.mode in {"both", "step1"}:
            try:
                answer1 = with_retry(f"{qid} Step 1", lambda: agent.generate_answer(client, question, agent.SYSTEM_PROMPT))
                item["step1_answer"] = answer1
                scores1 = score_answer(answer1, row)
                for key, value in scores1.items():
                    item[f"step1_{key}"] = value
            except Exception as exc:
                item["step1_answer"] = ""
                item["step1_error"] = str(exc)
                item["step1_heuristic_pass"] = False

        if args.mode in {"both", "step2"}:
            try:
                sources = with_retry(f"{qid} retrieval", lambda: agent.retrieve(client, question, chunks, vectors, top_k=5))
                rag_context = make_rag_context(sources)
                answer2 = with_retry(f"{qid} Step 2", lambda: agent.generate_answer(client, question, agent.RAG_SYSTEM_PROMPT, rag_context))
                retrieved_ids = [str(s.get("section_id", "")) for s in sources]
                item["step2_answer"] = answer2
                item["retrieved_sections"] = ";".join(retrieved_ids)
                item["evidence_hit"] = any(sid in retrieved_ids for sid in row_gold_ids) if row_gold_ids else "N/A"
                scores2 = score_answer(answer2, row)
                for key, value in scores2.items():
                    item[f"step2_{key}"] = value
                item["step2_source_files"] = ";".join(sorted({str(s.get("source", "")) for s in sources}))
            except Exception as exc:
                item["step2_answer"] = ""
                item["step2_error"] = str(exc)
                item["step2_heuristic_pass"] = False
        result_rows.append(item)
        print(f"[{number}/{len(selected)}] {qid} evaluated.")

    output = ROOT / "evaluation_results.csv"
    headers: list[str] = []
    for item in result_rows:
        for key in item:
            if key not in headers:
                headers.append(key)
    with output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result_rows)

    summary_lines = [
        "VoiceOps practice-question evaluation summary",
        "==============================================",
        f"Questions evaluated: {len(selected)} of {len(rows)} total",
        f"Mode: {args.mode}",
        f"Question text: {'ASR-like text' if args.use_asr_text else 'clean question text'}",
        "",
        "IMPORTANT: These are transparent string/behavior heuristics, not official capstone scores.",
        "Review evaluation_results.csv and manually inspect incorrect or borderline cases.",
    ]
    for mode in ("step1", "step2"):
        pass_key = f"{mode}_heuristic_pass"
        scored = [r for r in result_rows if r.get(pass_key) is not None]
        if scored:
            passed = sum(1 for r in scored if r.get(pass_key) is True)
            summary_lines.append(f"{mode.upper()} heuristic passes: {passed}/{len(scored)} ({100*passed/len(scored):.1f}%)")
            req_rows = [r for r in result_rows if r.get(f"{mode}_required_groups_total", 0) not in (0, None, "")]
            if req_rows:
                total_groups = sum(int(r.get(f"{mode}_required_groups_total", 0) or 0) for r in req_rows)
                hit_groups = sum(int(r.get(f"{mode}_required_groups_met", 0) or 0) for r in req_rows)
                summary_lines.append(f"{mode.upper()} must-include groups: {hit_groups}/{total_groups} ({100*hit_groups/total_groups:.1f}%)")
    step2_with_evidence = [r for r in result_rows if r.get("evidence_hit") in (True, False)]
    if step2_with_evidence:
        hits = sum(1 for r in step2_with_evidence if r["evidence_hit"] is True)
        summary_lines.append(f"STEP2 gold KB evidence hit: {hits}/{len(step2_with_evidence)} ({100*hits/len(step2_with_evidence):.1f}%)")
    summary_lines += ["", f"Detailed CSV: {output.name}"]
    summary_path = ROOT / "evaluation_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n".join(summary_lines))
    print(f"\nSaved: {output}")
    print(f"Saved: {summary_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nEvaluation stopped by user. Any completed file rows may not have been saved yet.")
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nERROR: {exc}")
        raise SystemExit(1)
