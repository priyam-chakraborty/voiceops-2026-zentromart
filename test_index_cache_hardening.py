"""Offline tests for embedding-cache correctness. These tests never call Gemini."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run


class IndexCacheHardeningTests(unittest.TestCase):
    def _chunk(self) -> dict:
        return {
            "source": "policy.md",
            "section_id": "RET-2",
            "title": "Return windows",
            "doc_type": "policy",
            "status": "current",
            "effective_from": "2026-10-01",
            "text": "Return window: 7 days.",
        }

    def test_fingerprint_changes_when_any_embedding_metadata_changes(self) -> None:
        base = self._chunk()
        original = run._index_cache_fingerprint([base], "gemini-embedding-001")
        for field, value in (
            ("title", "Changed title"),
            ("doc_type", "faq"),
            ("status", "superseded"),
            ("effective_from", "2025-01-01"),
            ("source", "different.md"),
            ("section_id", "RET-3"),
            ("text", "Changed text."),
        ):
            mutated = dict(base)
            mutated[field] = value
            with self.subTest(field=field):
                self.assertNotEqual(original, run._index_cache_fingerprint([mutated], "gemini-embedding-001"))

    def test_fingerprint_changes_when_model_or_output_dimensions_change(self) -> None:
        chunk = self._chunk()
        base = run._index_cache_fingerprint([chunk], "gemini-embedding-001", 768)
        self.assertNotEqual(base, run._index_cache_fingerprint([chunk], "another-model", 768))
        self.assertNotEqual(base, run._index_cache_fingerprint([chunk], "gemini-embedding-001", 512))

    def test_vector_validator_accepts_well_formed_finite_nonzero_vectors(self) -> None:
        self.assertTrue(run._valid_cached_vectors([[0.1, 0.2, 0.3], [1.0, 0.0, -1.0]], 2, 3))

    def test_vector_validator_rejects_wrong_count_dimension_nan_inf_and_zero_rows(self) -> None:
        bad_cases = [
            ([[0.1, 0.2, 0.3]], 2, 3),
            ([[0.1, 0.2]], 1, 3),
            ([[0.1, float("nan"), 0.3]], 1, 3),
            ([[0.1, float("inf"), 0.3]], 1, 3),
            ([[0.0, 0.0, 0.0]], 1, 3),
            ([[True, 0.2, 0.3]], 1, 3),
            (["not-a-vector"], 1, 3),
        ]
        for vectors, count, dimensions in bad_cases:
            with self.subTest(vectors=repr(vectors)):
                self.assertFalse(run._valid_cached_vectors(vectors, count, dimensions))

    def test_corrupt_matching_cache_is_rebuilt_without_provider_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "Capstone_Project_Dataset_FTTT_2026"
            kb = dataset / "kb"
            kb.mkdir(parents=True)
            (kb / "policy.md").write_text(
                "---\ntitle: Return policy\ndoc_type: policy\nstatus: current\neffective_from: 2026-10-01\n---\n## [RET-2] Return windows\nReturn window: 7 days.\n",
                encoding="utf-8",
            )
            old_root, old_dataset = run.ROOT, run.DATASET
            try:
                run.ROOT, run.DATASET = root, dataset
                chunks = run.read_knowledge_chunks()
                cache_dir = root / ".cache"
                cache_dir.mkdir()
                cache_file = cache_dir / "rag_index.json"
                model = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001").strip() or "gemini-embedding-001"
                fingerprint = run._index_cache_fingerprint(chunks, model, 768)
                cache_file.write_text(
                    json.dumps({"fingerprint": fingerprint, "chunks": chunks, "vectors": [[0.1, 0.2]]}),
                    encoding="utf-8",
                )
                fake_vectors = [[0.01] * 768 for _ in chunks]
                with patch.object(run, "_embed_texts", return_value=fake_vectors) as fake_embed:
                    loaded_chunks, loaded_vectors = run.load_or_build_index(client=None)
                fake_embed.assert_called_once()
                self.assertEqual(loaded_chunks, chunks)
                self.assertTrue(run._valid_cached_vectors(loaded_vectors, len(chunks), 768))
                saved = json.loads(cache_file.read_text(encoding="utf-8"))
                self.assertTrue(run._valid_cached_vectors(saved["vectors"], len(chunks), 768))
            finally:
                run.ROOT, run.DATASET = old_root, old_dataset

    def test_valid_cache_is_used_without_reembedding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "Capstone_Project_Dataset_FTTT_2026"
            kb = dataset / "kb"
            kb.mkdir(parents=True)
            (kb / "policy.md").write_text(
                "---\ntitle: Refunds\ndoc_type: policy\nstatus: current\neffective_from: 2026-10-01\n---\n## [REF-2] Wallet refunds\nRefunds go to wallet.\n",
                encoding="utf-8",
            )
            old_root, old_dataset = run.ROOT, run.DATASET
            try:
                run.ROOT, run.DATASET = root, dataset
                chunks = run.read_knowledge_chunks()
                cache_dir = root / ".cache"
                cache_dir.mkdir()
                vectors = [[0.01] * 768 for _ in chunks]
                model = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001").strip() or "gemini-embedding-001"
                fingerprint = run._index_cache_fingerprint(chunks, model, 768)
                (cache_dir / "rag_index.json").write_text(
                    json.dumps({"fingerprint": fingerprint, "chunks": chunks, "vectors": vectors}),
                    encoding="utf-8",
                )
                with patch.object(run, "_embed_texts", side_effect=AssertionError("must use valid cache")):
                    loaded_chunks, loaded_vectors = run.load_or_build_index(client=None)
                self.assertEqual(loaded_chunks, chunks)
                self.assertEqual(loaded_vectors, vectors)
            finally:
                run.ROOT, run.DATASET = old_root, old_dataset

    def test_valid_legacy_cache_is_kept_when_all_cached_chunks_still_match(self) -> None:
        import hashlib

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "Capstone_Project_Dataset_FTTT_2026"
            kb = dataset / "kb"
            kb.mkdir(parents=True)
            (kb / "policy.md").write_text(
                "---\ntitle: Refunds\ndoc_type: policy\nstatus: current\neffective_from: 2026-10-01\n---\n## [REF-2] Wallet refunds\nRefunds go to wallet.\n",
                encoding="utf-8",
            )
            old_root, old_dataset = run.ROOT, run.DATASET
            try:
                run.ROOT, run.DATASET = root, dataset
                chunks = run.read_knowledge_chunks()
                cache_dir = root / ".cache"
                cache_dir.mkdir()
                model = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
                legacy_source = "\n".join(
                    c["source"] + "|" + c["section_id"] + "|" + c["text"] + "|" + c["status"]
                    for c in chunks
                ) + model
                legacy_fingerprint = hashlib.sha256(legacy_source.encode("utf-8")).hexdigest()
                vectors = [[0.02] * 768 for _ in chunks]
                (cache_dir / "rag_index.json").write_text(
                    json.dumps({"fingerprint": legacy_fingerprint, "chunks": chunks, "vectors": vectors}),
                    encoding="utf-8",
                )
                with patch.object(run, "_embed_texts", side_effect=AssertionError("valid legacy cache should not trigger API")):
                    loaded_chunks, loaded_vectors = run.load_or_build_index(client=None)
                self.assertEqual(loaded_chunks, chunks)
                self.assertEqual(loaded_vectors, vectors)
            finally:
                run.ROOT, run.DATASET = old_root, old_dataset

    def test_old_cache_with_metadata_drift_is_rebuilt_even_if_legacy_hash_matches(self) -> None:
        import hashlib

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "Capstone_Project_Dataset_FTTT_2026"
            kb = dataset / "kb"
            kb.mkdir(parents=True)
            (kb / "policy.md").write_text(
                "---\ntitle: New title\ndoc_type: policy\nstatus: current\neffective_from: 2026-10-01\n---\n## [REF-2] Wallet refunds\nRefunds go to wallet.\n",
                encoding="utf-8",
            )
            old_root, old_dataset = run.ROOT, run.DATASET
            try:
                run.ROOT, run.DATASET = root, dataset
                chunks = run.read_knowledge_chunks()
                cached_chunks = [dict(chunk, title="Old title") for chunk in chunks]
                cache_dir = root / ".cache"
                cache_dir.mkdir()
                model = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
                # The legacy hash ignores title, so this emulates an old cache that looks current by that hash.
                legacy_source = "\n".join(
                    c["source"] + "|" + c["section_id"] + "|" + c["text"] + "|" + c["status"]
                    for c in cached_chunks
                ) + model
                legacy_fingerprint = hashlib.sha256(legacy_source.encode("utf-8")).hexdigest()
                vectors = [[0.02] * 768 for _ in chunks]
                (cache_dir / "rag_index.json").write_text(
                    json.dumps({"fingerprint": legacy_fingerprint, "chunks": cached_chunks, "vectors": vectors}),
                    encoding="utf-8",
                )
                fresh_vectors = [[0.03] * 768 for _ in chunks]
                with patch.object(run, "_embed_texts", return_value=fresh_vectors) as fake_embed:
                    loaded_chunks, loaded_vectors = run.load_or_build_index(client=None)
                fake_embed.assert_called_once()
                self.assertEqual(loaded_chunks, chunks)
                self.assertEqual(loaded_vectors, fresh_vectors)
            finally:
                run.ROOT, run.DATASET = old_root, old_dataset

    def test_blank_embedding_model_uses_one_consistent_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "Capstone_Project_Dataset_FTTT_2026"
            kb = dataset / "kb"
            kb.mkdir(parents=True)
            (kb / "policy.md").write_text(
                "---\ntitle: Return policy\ndoc_type: policy\nstatus: current\neffective_from: 2026-10-01\n---\n## [RET-2] Return windows\nReturn window: 7 days.\n",
                encoding="utf-8",
            )
            old_root, old_dataset = run.ROOT, run.DATASET
            try:
                run.ROOT, run.DATASET = root, dataset
                fake_vectors = [[0.01] * 768]
                with patch.dict(os.environ, {"GEMINI_EMBEDDING_MODEL": ""}):
                    with patch.object(run, "_embed_texts", return_value=fake_vectors) as fake_embed:
                        chunks, vectors = run.load_or_build_index(client=None)
                    self.assertEqual(fake_embed.call_args.args[2], "gemini-embedding-001")
                    cache = json.loads((root / ".cache" / "rag_index.json").read_text(encoding="utf-8"))
                    self.assertEqual(
                        cache["fingerprint"],
                        run._index_cache_fingerprint(chunks, "gemini-embedding-001", 768),
                    )
                    with patch.object(run, "_embed_texts", side_effect=AssertionError("consistent cache should be reused")):
                        reloaded_chunks, reloaded_vectors = run.load_or_build_index(client=None)
                self.assertEqual(reloaded_chunks, chunks)
                self.assertEqual(reloaded_vectors, vectors)
            finally:
                run.ROOT, run.DATASET = old_root, old_dataset

    def test_invalid_fresh_document_embeddings_fail_before_cache_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "Capstone_Project_Dataset_FTTT_2026"
            kb = dataset / "kb"
            kb.mkdir(parents=True)
            (kb / "policy.md").write_text(
                "---\ntitle: Return policy\ndoc_type: policy\nstatus: current\neffective_from: 2026-10-01\n---\n## [RET-2] Return windows\nReturn window: 7 days.\n",
                encoding="utf-8",
            )
            old_root, old_dataset = run.ROOT, run.DATASET
            try:
                run.ROOT, run.DATASET = root, dataset
                with patch.object(run, "_embed_texts", return_value=[[0.1, 0.2]]) as fake_embed:
                    with self.assertRaisesRegex(RuntimeError, "invalid document vector"):
                        run.load_or_build_index(client=None)
                fake_embed.assert_called_once()
                self.assertFalse((root / ".cache" / "rag_index.json").exists())
            finally:
                run.ROOT, run.DATASET = old_root, old_dataset

    def test_invalid_query_embedding_fails_with_clear_error_before_numpy_math(self) -> None:
        chunks = [self._chunk()]
        vectors = [[0.01] * 768]
        with patch.object(run, "_embed_texts", return_value=[[0.1, 0.2]]):
            with self.assertRaisesRegex(RuntimeError, "invalid query vector"):
                run.retrieve(client=None, question="return policy", chunks=chunks, vectors=vectors)


    # VOICEOPS_OFFLINE_FOLLOWUP_CACHE_TESTS_V1
    def test_blank_embedding_model_uses_default_for_query_embedding(self) -> None:
        chunks = [self._chunk()]
        vectors = [[0.02] * 768]
        with patch.dict(os.environ, {"GEMINI_EMBEDDING_MODEL": ""}):
            with patch.object(run, "_embed_texts", return_value=[[0.01] * 768]) as fake_embed:
                run.retrieve(client=None, question="return window", chunks=chunks, vectors=vectors, top_k=1)
        fake_embed.assert_called_once()
        self.assertEqual(fake_embed.call_args.args[2], "gemini-embedding-001")

    def test_invalid_document_vectors_fail_before_query_embedding(self) -> None:
        chunks = [self._chunk()]
        vectors = [[0.0] * 768]
        with patch.object(run, "_embed_texts", side_effect=AssertionError("query embedding must not be called")) as fake_embed:
            with self.assertRaisesRegex(RuntimeError, "invalid document vector"):
                run.retrieve(client=None, question="return window", chunks=chunks, vectors=vectors)
        fake_embed.assert_not_called()

    def test_vector_validator_rejects_float_overflow_and_huge_integers(self) -> None:
        invalid_rows = [
            [10**1000, 0, 1],
            [1e300, 0, 1],
        ]
        for vector in invalid_rows:
            with self.subTest(first_value=type(vector[0]).__name__):
                self.assertFalse(run._valid_cached_vectors([vector], 1, 3))


if __name__ == "__main__":
    unittest.main(verbosity=2)
