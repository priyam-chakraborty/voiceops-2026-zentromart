"""Run all deterministic, no-provider VoiceOps quality checks.

Run from the VoiceOps project root:
    py run_offline_quality_checks.py

This runner does not call Gemini, Gnani, or any other network service.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TEST_MODULES = [
    "offline_regression_tests",
    "test_step3_handover_fallback",
    "test_interactive_handover_ticket_safety",
    "test_later_escalation_ticket_safety",
    "test_order_lookup_safety",
    "test_step3_interactive_flow_integration",
    "test_evaluator_score_denominator",
    "test_product_catalog_lookup",
    "test_policy_evidence_integrity",
    "test_retrieval_status_priority",
    "test_retrieval_topic_prior",
    "test_unsupported_midnight_delivery",
    "test_wallet_refund_retrieval",
    "test_reference_answers_52",
    "test_index_cache_hardening",
    "test_identifier_parsing",
    "test_voice_batch_offline",
]


def run(label: str, command: list[str]) -> tuple[bool, str]:
    print(f"\n{'=' * 72}\n{label}\n$ {' '.join(command)}\n{'=' * 72}", flush=True)
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output = completed.stdout.rstrip()
    if output:
        print(output)
    passed = completed.returncode == 0
    print(f"RESULT: {'PASS' if passed else 'FAIL'} (exit {completed.returncode})", flush=True)
    return passed, output


def main() -> int:
    gates: list[tuple[str, bool]] = []

    passed, unit_output = run(
        "Offline unit/regression suites",
        [sys.executable, "-m", "unittest", "-v", *TEST_MODULES],
    )
    gates.append(("offline unit/regression suites", passed))
    test_match = re.search(r"Ran\s+(\d+)\s+tests?", unit_output)
    test_count = int(test_match.group(1)) if test_match else None

    validator = ROOT / "Capstone_Project_Dataset_FTTT_2026" / "validate_dataset.py"
    if validator.is_file():
        passed, _ = run("Synthetic dataset integrity", [sys.executable, str(validator)])
        gates.append(("dataset integrity", passed))
    else:
        print("\nRESULT: FAIL (dataset validator missing)")
        gates.append(("dataset integrity", False))

    audit = ROOT / "audit_practice_evidence_refs.py"
    if audit.is_file():
        passed, _ = run("Practice evidence-reference integrity", [sys.executable, str(audit)])
        gates.append(("practice evidence references", passed))
    else:
        print("\nRESULT: FAIL (evidence-reference audit missing)")
        gates.append(("practice evidence references", False))

    compile_targets = [
        "run.py", "voice_batch.py", "evaluate.py", "evaluate_resumable.py",
        "audit_practice_evidence_refs.py", "offline_regression_tests.py",
        "run_offline_quality_checks.py", "patch_step3_handover_fallback.py",
        "patch_index_cache_hardening.py", "patch_identifier_parsing.py",
        "patch_voice_batch_handover_fallback.py",
        *[f"{name}.py" for name in TEST_MODULES if name != "offline_regression_tests"],
    ]
    compile_targets = [name for name in compile_targets if (ROOT / name).is_file()]
    passed, _ = run("Python syntax/bytecode compilation", [sys.executable, "-m", "py_compile", *compile_targets])
    gates.append(("Python compilation", passed))

    print("\n" + "=" * 72)
    print("OFFLINE QUALITY SUMMARY")
    print("=" * 72)
    if test_count is not None:
        print(f"Unit tests counted: {test_count}")
    else:
        print("Unit tests counted: unavailable (inspect unit-test output above)")
    for label, passed in gates:
        print(f"{'PASS' if passed else 'FAIL'}  {label}")
    failures = [label for label, passed in gates if not passed]
    print(f"Overall: {'PASS' if not failures else 'FAIL'} ({len(gates) - len(failures)}/{len(gates)} gates passed)")
    if failures:
        print("Failed gates: " + ", ".join(failures))
        return 1
    print("Scope: no live Gemini/Gnani generation or provider quota was exercised by this runner.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
