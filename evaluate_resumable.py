"""Checkpointed wrapper around the project's existing evaluate.py.

Run from the VoiceOps project root:
    python evaluate_resumable.py --limit 52 --mode both

The wrapper evaluates one practice question per child process and checkpoints each
completed question to evaluation_results_resumable.csv. If a provider/API error
occurs, it stops after saving that question so repeated failures do not cascade
through the remaining practice set. Re-run the same command to resume; successful
stages are skipped. Use --fresh to start a new run.

All core answering/retrieval behavior still comes from evaluate.py and run.py.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "Capstone_Project_Dataset_FTTT_2026"
QUESTIONS_FILE = DATASET / "practice" / "questions.csv"
EVALUATOR = ROOT / "evaluate.py"
OUTPUT_FILE = ROOT / "evaluation_results_resumable.csv"
STATE_FILE = ROOT / "evaluation_resumable_state.json"
LEGACY_OUTPUT = ROOT / "evaluation_results.csv"


def _read_questions() -> list[dict[str, str]]:
    if not QUESTIONS_FILE.is_file():
        raise FileNotFoundError(f"Practice questions file not found: {QUESTIONS_FILE}")
    with QUESTIONS_FILE.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _stage_complete(row: dict[str, Any], stage: str) -> bool:
    answer = str(row.get(f"{stage}_answer", "") or "").strip()
    error = str(row.get(f"{stage}_error", "") or "").strip()
    return bool(answer) and not error


def _next_stage(row: dict[str, Any], requested_mode: str) -> str | None:
    step1_done = _stage_complete(row, "step1")
    step2_done = _stage_complete(row, "step2")
    if requested_mode == "step1":
        return None if step1_done else "step1"
    if requested_mode == "step2":
        return None if step2_done else "step2"
    if step1_done and step2_done:
        return None
    if not step1_done and not step2_done:
        return "both"
    return "step1" if not step1_done else "step2"


def _contains_error(row: dict[str, Any], attempted_stage: str) -> bool:
    if str(row.get("runner_error", "") or "").strip():
        return True
    stages = ("step1", "step2") if attempted_stage == "both" else (attempted_stage,)
    return any(str(row.get(f"{stage}_error", "") or "").strip() for stage in stages)


def _write_results(rows_by_id: dict[str, dict[str, Any]], order_ids: dict[str, int]) -> None:
    ordered_rows = sorted(
        rows_by_id.values(),
        key=lambda row: (order_ids.get(str(row.get("id", "")), 10**9), str(row.get("id", ""))),
    )
    headers: list[str] = []
    for row in ordered_rows:
        for key in row:
            if key not in headers:
                headers.append(key)
    temp_path = OUTPUT_FILE.with_suffix(OUTPUT_FILE.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered_rows)
    os.replace(temp_path, OUTPUT_FILE)


def _write_state(mode: str, use_asr_text: bool, dataset_hash: str) -> None:
    state = {
        "schema_version": 1,
        "mode": mode,
        "use_asr_text": use_asr_text,
        "dataset_sha256": dataset_hash,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    temp_path = STATE_FILE.with_suffix(STATE_FILE.suffix + ".tmp")
    temp_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    os.replace(temp_path, STATE_FILE)


def _write_summary(
    selected: list[dict[str, str]],
    rows_by_id: dict[str, dict[str, Any]],
    mode: str,
    dataset_total: int,
) -> None:
    selected_ids = [str(row.get("id", "")) for row in selected]
    done = sum(_next_stage(rows_by_id.get(qid, {}), mode) is None for qid in selected_ids)
    pending = len(selected_ids) - done
    selected_id_set = set(selected_ids)
    error_rows = sum(
        bool(str(row.get("runner_error", "") or "").strip()) or
        any(str(row.get(f"{stage}_error", "") or "").strip() for stage in ("step1", "step2"))
        for qid, row in rows_by_id.items() if qid in selected_id_set
    )
    summary = [
        "VoiceOps checkpointed evaluation",
        "=================================",
        f"Requested mode: {mode}",
        f"Questions selected this run: {len(selected)} of {dataset_total}",
        f"Fully complete for requested mode: {done}/{len(selected)}",
        f"Pending or failed: {pending}/{len(selected)}",
        f"Rows containing stage errors: {error_rows}",
        "",
        "Scores in the CSV remain transparent heuristic checks, not official course scores.",
        f"Detailed results: {OUTPUT_FILE.name}",
        f"Resume state: {STATE_FILE.name}",
    ]
    summary_path = ROOT / "evaluation_resumable_summary.txt"
    temp_path = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temp_path.write_text("\n".join(summary) + "\n", encoding="utf-8")
    os.replace(temp_path, summary_path)


def _base_row(question: dict[str, str], index: int, use_asr_text: bool) -> dict[str, str]:
    text = (question.get("asr_text") if use_asr_text else question.get("question")) or question.get("question", "")
    return {
        "id": question.get("id", f"row-{index + 1}"),
        "language": question.get("language", ""),
        "question_used": text,
        "expected_decision": question.get("expected_decision", ""),
        "acceptable_decisions": question.get("acceptable_decisions", ""),
        "answerable": question.get("answerable", ""),
        "gold_evidence": question.get("gold_evidence", ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Checkpointed, resumable wrapper for evaluate.py.")
    parser.add_argument("--limit", type=int, default=52, help="Questions to process (default: 52; use 0 for all).")
    parser.add_argument("--start", type=int, default=0, help="Zero-based row offset into practice/questions.csv.")
    parser.add_argument("--mode", choices=("both", "step1", "step2"), default="both")
    parser.add_argument("--use-asr-text", action="store_true", help="Use the dataset's ASR-like transcription field.")
    parser.add_argument("--fresh", action="store_true", help="Discard this wrapper's previous checkpoints and start over.")
    args = parser.parse_args()

    if not EVALUATOR.is_file():
        raise FileNotFoundError(f"Existing evaluator not found: {EVALUATOR}")
    all_questions = _read_questions()
    start = max(0, args.start)
    selected = all_questions[start:]
    if args.limit > 0:
        selected = selected[:args.limit]
    if not selected:
        print("No questions selected. Check --start and --limit.")
        return 2

    dataset_hash = hashlib.sha256(QUESTIONS_FILE.read_bytes()).hexdigest()
    if args.fresh:
        for path in (OUTPUT_FILE, STATE_FILE):
            if path.exists():
                path.unlink()

    state: dict[str, Any] | None = None
    if STATE_FILE.exists():
        try:
            state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise RuntimeError(f"Could not read resume state {STATE_FILE}: {exc}. Use --fresh to restart.") from exc

    if OUTPUT_FILE.exists() and state is None and not args.fresh:
        raise RuntimeError(
            f"{OUTPUT_FILE.name} exists without its resume-state file. Back it up or use --fresh to start a new run."
        )
    if state is not None:
        expected = (args.mode, bool(args.use_asr_text), dataset_hash)
        actual = (state.get("mode"), bool(state.get("use_asr_text")), state.get("dataset_sha256"))
        if expected != actual:
            raise RuntimeError(
                "Resume configuration differs from the previous run (mode, ASR option, or dataset changed). "
                "Use the same options, or choose --fresh for a new run."
            )

    rows_by_id: dict[str, dict[str, Any]] = {
        str(row.get("id", "")): row for row in _read_csv(OUTPUT_FILE) if str(row.get("id", "")).strip()
    }
    order_ids = {str(row.get("id", f"row-{index + 1}")): index for index, row in enumerate(all_questions)}

    # On the first wrapper run, reuse compatible results from the original evaluator
    # instead of spending API quota on questions already completed in evaluation_results.csv.
    if not args.fresh and not state and not rows_by_id and LEGACY_OUTPUT.is_file():
        legacy_rows = _read_csv(LEGACY_OUTPUT)
        questions_by_id = {str(q.get("id", "")): q for q in all_questions}
        compatible = bool(legacy_rows)
        for legacy_row in legacy_rows:
            qid = str(legacy_row.get("id", ""))
            question = questions_by_id.get(qid)
            if question is None:
                compatible = False
                break
            expected_text = (question.get("asr_text") if args.use_asr_text else question.get("question")) or question.get("question", "")
            if legacy_row.get("question_used", "") != expected_text:
                compatible = False
                break
        if compatible:
            rows_by_id = {str(row["id"]): row for row in legacy_rows if str(row.get("id", "")).strip()}
            _write_results(rows_by_id, order_ids)
            _write_state(args.mode, bool(args.use_asr_text), dataset_hash)
            print(f"Imported {len(rows_by_id)} compatible existing evaluation rows; successful stages will be reused.")
        elif legacy_rows:
            print("NOTE: Existing evaluation_results.csv did not match this question-text mode; it will not be imported.")

    # evaluate.py uses a fixed output filename. Preserve the user's prior file and restore it at exit.
    original_legacy_bytes = LEGACY_OUTPUT.read_bytes() if LEGACY_OUTPUT.exists() else None
    temp_backup_dir = tempfile.TemporaryDirectory(prefix="voiceops_eval_backup_")
    backup_file = Path(temp_backup_dir.name) / "evaluation_results.csv"
    if original_legacy_bytes is not None:
        backup_file.write_bytes(original_legacy_bytes)

    provider_error_seen = False
    try:
        for local_index, question in enumerate(selected):
            global_index = start + local_index
            qid = str(question.get("id", f"row-{global_index + 1}"))
            existing = rows_by_id.get(qid, {})
            stage = _next_stage(existing, args.mode)
            if stage is None:
                print(f"[{local_index + 1}/{len(selected)}] {qid}: already complete, skipping.")
                continue

            print(f"\n[{local_index + 1}/{len(selected)}] {qid}: running {stage}...")
            command = [
                sys.executable, str(EVALUATOR),
                "--limit", "1",
                "--start", str(global_index),
                "--mode", stage,
            ]
            if args.use_asr_text:
                command.append("--use-asr-text")

            completed = subprocess.run(command, cwd=str(ROOT), check=False)
            fresh_rows = _read_csv(LEGACY_OUTPUT)
            fresh = next((row for row in fresh_rows if str(row.get("id", "")) == qid), None)

            merged = dict(existing) if existing else _base_row(question, global_index, args.use_asr_text)
            if fresh is not None:
                merged.update(fresh)
                # A successful retry omits the old error column, so clear stale error state
                # only for the stage(s) just attempted. Preserve results from other stages.
                attempted_stages = ("step1", "step2") if stage == "both" else (stage,)
                for attempted in attempted_stages:
                    if str(fresh.get(f"{attempted}_answer", "") or "").strip():
                        merged.pop(f"{attempted}_error", None)
            elif completed.returncode != 0:
                merged["runner_error"] = f"evaluate.py exited with code {completed.returncode} before writing this question's row."
            else:
                merged["runner_error"] = "evaluate.py did not write an output row for the selected question."
            rows_by_id[qid] = merged
            _write_results(rows_by_id, order_ids)
            _write_state(args.mode, bool(args.use_asr_text), dataset_hash)
            _write_summary(selected, rows_by_id, args.mode, len(all_questions))

            attempted_error = _contains_error(merged, stage)
            if attempted_error:
                print(
                    f"STOP: {qid} returned an error during {stage}. The checkpoint is saved. "
                    "Fix the API/provider issue, then rerun the same command to resume without repeating successful stages."
                )
                provider_error_seen = True
                break

        _write_results(rows_by_id, order_ids)
        _write_state(args.mode, bool(args.use_asr_text), dataset_hash)
        _write_summary(selected, rows_by_id, args.mode, len(all_questions))
    finally:
        if original_legacy_bytes is not None:
            LEGACY_OUTPUT.write_bytes(backup_file.read_bytes())
        elif LEGACY_OUTPUT.exists():
            LEGACY_OUTPUT.unlink()
        temp_backup_dir.cleanup()

    finished = sum(
        _next_stage(rows_by_id.get(str(row.get("id", "")), {}), args.mode) is None
        for row in selected
    )
    print(f"\nCheckpointed results: {OUTPUT_FILE}")
    print(f"Summary: {ROOT / 'evaluation_resumable_summary.txt'}")
    print(f"Completed for requested mode: {finished}/{len(selected)} selected questions")
    if provider_error_seen:
        return 2
    return 0 if finished == len(selected) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted. Completed checkpoints remain in evaluation_results_resumable.csv; rerun the same command to resume.")
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nERROR: {type(exc).__name__}: {exc}")
        raise SystemExit(1)
