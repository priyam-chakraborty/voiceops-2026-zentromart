# VoiceOps Quality Testing

## Offline regression tests

Run the following tests from the project root:

- `py offline_regression_tests.py`
- `py test_step3_handover_fallback.py`
- `py test_evaluator_score_denominator.py`
- `py test_product_catalog_lookup.py`
- `py test_policy_evidence_integrity.py`
- `py test_retrieval_status_priority.py`
- `py test_retrieval_topic_prior.py`
- `py test_wallet_refund_retrieval.py`
- `py test_unsupported_midnight_delivery.py`
- `py test_reference_answers_52.py`

Run `py audit_practice_evidence_refs.py` to check the practice-question evidence references.

## Interpreting the results

These offline tests validate specific behaviours, such as caller verification, truthful ticket handovers, current-policy precedence, retrieval topic bonuses, and evaluator scoring.

The 52-question reference-answer audit checks the supplied reference answers against the documented local heuristic. It does not establish that the live agent generated correct answers for all 52 practice questions.

Live Step 1 and Step 2 evaluation results must be considered separately. Provider/API errors are excluded from answer-quality score denominators and must not be counted as incorrect answers.

The live evaluation remains incomplete until sufficient practice questions produce successful responses. The reported offline test results are not official capstone scores.
