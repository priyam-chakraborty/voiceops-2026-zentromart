# Offline hardening changelog

These additions target bugs and reliability risks that can be reproduced without live Gemini or Gnani calls.

## Fixes and safeguards

1. **Embedding-index cache:** cache identity now includes all text-bearing knowledge-base metadata, model, dimensionality, and a schema version. Cache contents are checked for matching chunks and finite, numeric, nonzero vectors. Invalid or stale caches rebuild instead of being trusted. Cache writes use a temporary file and atomic replace. If cache persistence fails, the in-memory index remains usable. A blank embedding-model environment value is normalized consistently.
2. **Query-vs-document vectors:** the query vector must match the validated document vector dimension before NumPy similarity math, with clear errors for malformed indexes or provider results.
3. **Order and verification parsing:** common ISO and day-first dates no longer masquerade as eight-digit order IDs; digits in product SKU strings such as `ELE-0012` no longer masquerade as customer verification proofs.
4. **Voice ticket truthfulness:** ticket IDs are scoped to the current response; missing ticket references do not crash or get invented. Ticket-tool errors do not produce false success messages. When a ticket has already been created, a Gemini generation failure no longer hides that fact; the system returns a deterministic, truthful confirmation without promising the outcome or timeline.
5. **Voice availability fallback:** retrieval failures do not lead to an ungrounded policy answer, and generation failures return a safe service-unavailable message instead of propagating an exception out of the conversation agent.
6. **Privacy checks:** automated tests verify API-key redaction, approved evidence fields, and exclusion of phone, email, address and pincode fields from the order context supplied to answer generation.

## Offline verification result

Latest workbench run: **79 unit/regression tests passed across 13 suites**, the dataset validator passed, all practice question evidence IDs and SKU references resolved, and modified Python files compiled successfully. The complete offline runner reported **4/4 quality gates passed**.

This is offline verification on a reconstructed workbench assembled from the project source archives and earlier patch files. The patch scripts were separately applied to a clean source copy and compiled successfully. The user's Windows checkout and GitHub repository were not changed by that workbench run.

## What offline testing cannot establish

- Whether Gemini-generated answers are factually correct across all 52 practice questions.
- End-to-end STT/TTS behavior with real Gnani credentials, real microphones or noisy speech.
- Provider quotas, network availability, or live service latency.
- Actual retrieval recall of live embedding vectors across the entire knowledge base.

Those checks need a later provider-enabled run. Until that is possible, provider failures must be reported as missing/unknown scores rather than silently treated as model failures or passes.
