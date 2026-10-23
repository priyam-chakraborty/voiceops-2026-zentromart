# VoiceOps Offline Quality Testing

**Current local hardening snapshot: 23 October 2026**

The current project working tree passed **121 unit/regression tests** and all **4 quality gates** in `run_offline_quality_checks.py`:

1. Offline unit and regression tests
2. Synthetic dataset integrity
3. Practice-evidence reference integrity
4. Python syntax/bytecode compilation

## Primary command

Run from the project root in PowerShell:

```powershell
py run_offline_quality_checks.py
```

For this snapshot, the expected summary is:

```text
Unit tests counted: 121
PASS  offline unit/regression suites
PASS  dataset integrity
PASS  practice evidence references
PASS  Python compilation
Overall: PASS (4/4 gates passed)
```

## Registered unit/regression modules

The aggregate runner executes these modules:

- `offline_regression_tests`
- `test_step3_handover_fallback`
- `test_interactive_handover_ticket_safety`
- `test_later_escalation_ticket_safety`
- `test_order_lookup_safety`
- `test_step3_interactive_flow_integration`
- `test_evaluator_score_denominator`
- `test_product_catalog_lookup`
- `test_policy_evidence_integrity`
- `test_retrieval_status_priority`
- `test_retrieval_topic_prior`
- `test_unsupported_midnight_delivery`
- `test_wallet_refund_retrieval`
- `test_reference_answers_52`
- `test_index_cache_hardening`
- `test_identifier_parsing`
- `test_voice_batch_offline`

The tests cover tool verification and privacy, honest ticket confirmations, malformed and exceptional tool responses, order-switch verification, current versus superseded policy evidence, context-isolated topic ranking, product catalog lookup, embedding-cache/vector validation, conservative identifier extraction, safe error handling, and voice-batch file/state/output behavior.

## Evidence-reference audit

Run separately when needed:

```powershell
py audit_practice_evidence_refs.py
```

For the current dataset, the audit checked 52 practice records (47 marked answerable), 84 known knowledge-base section IDs and 70 product SKUs. It verifies that expected evidence references resolve. It does **not** measure live embedding recall or generated-answer accuracy.

## Scope and limitations

- The offline runner does **not** call live Gemini generation/embedding APIs or Gnani speech-to-text/text-to-speech APIs, and it does not test current provider availability or quota.
- The reference-answer heuristic evaluates supplied reference text; 52 correct reference records do not mean the live agent generated 52 correct answers.
- Synthetic retrieval-ranking tests exercise selected known scenarios; they do not establish recall for every query.
- The dataset contains synthetic records for a fictional store. The mock verification tool is not a production authentication system.
- The 121 tests and 4/4 gates are local regression evidence, not official capstone scoring or proof of production readiness.

Before submission, run the aggregate command on the exact final working tree and synchronize intended code/tests with the public GitHub repository. A local ZIP and a public repository link should refer to the same source snapshot.
