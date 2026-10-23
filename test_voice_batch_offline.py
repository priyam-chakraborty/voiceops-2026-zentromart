"""Offline-only regression tests for VoiceOps batch helpers and privacy boundaries."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run
import voice_batch


class VoiceBatchOfflineTests(unittest.TestCase):
    def test_safe_error_redacts_both_configured_api_keys(self) -> None:
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-gemini-secret", "GNANI_API_KEY": "test-gnani-secret"}):
            message = voice_batch._safe_error(RuntimeError("keys test-gemini-secret and test-gnani-secret failed"))
        self.assertNotIn("test-gemini-secret", message)
        self.assertNotIn("test-gnani-secret", message)
        self.assertEqual(message.count("[REDACTED]"), 2)

    def test_speech_cleanup_removes_markdown_and_source_citations(self) -> None:
        spoken = voice_batch._speech_friendly("## Result\n**Hello** [RET-2] and [ELE-0012]. Keep `this`.")
        self.assertNotIn("RET-2", spoken)
        self.assertNotIn("ELE-0012", spoken)
        self.assertNotIn("**", spoken)
        self.assertNotIn("`", spoken)
        self.assertIn("Hello", spoken)
        self.assertIn("Keep this", spoken)

    def test_json_evidence_projection_uses_only_approved_fields(self) -> None:
        chunk = {
            "section_id": "RET-2", "source": "policy.md", "title": "Returns",
            "doc_type": "policy", "status": "current", "effective_from": "2026-10-01",
            "retrieval_score": 0.91, "text": "7-day window", "phone": "9999999999",
            "email": "synthetic@example.test", "api_key": "should-never-serialize",
        }
        safe = voice_batch._chunk_for_json(chunk)
        self.assertEqual(set(safe), {"section_id", "source", "title", "doc_type", "status", "effective_from", "retrieval_score", "text"})
        self.assertNotIn("phone", safe)
        self.assertNotIn("email", safe)
        self.assertNotIn("api_key", safe)

    def test_audio_discovery_filters_extensions_and_separates_root_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "standalone.wav").write_bytes(b"audio")
            (root / "ignore.txt").write_text("not audio", encoding="utf-8")
            (root / "session_a").mkdir()
            (root / "session_b").mkdir()
            (root / "session_a" / "turn2.WAV").write_bytes(b"audio")
            (root / "session_a" / "turn1.wav").write_bytes(b"audio")
            (root / "session_b" / "clip.mp3").write_bytes(b"audio")
            files, sessions = voice_batch._build_sessions(root)
            relative = [p.relative_to(root).as_posix() for p in files]
            self.assertEqual(relative, ["session_a/turn1.wav", "session_a/turn2.WAV", "session_b/clip.mp3", "standalone.wav"])
            self.assertEqual(len(sessions["folder:session_a"]), 2)
            self.assertEqual(len(sessions["folder:session_b"]), 1)
            self.assertEqual(len(sessions["single:standalone.wav"]), 1)
            self.assertNotIn("ignore.txt", " ".join(relative))

    def test_order_request_without_verification_does_not_retrieve_or_disclose(self) -> None:
        # A synthetic order ID from the supplied test fixture; no network/provider is used.
        order_id = "69520867"
        agent = voice_batch.VoiceStep3Agent(run, client=None, chunks=[], vectors=[])
        with patch.object(agent, "lookup_order", return_value={"verification_required": True, "order_id": order_id}):
            with patch.object(run, "retrieve", side_effect=AssertionError("RAG must not run before verification")):
                reply = agent.respond(f"Where is my order {order_id}?")
        lower = reply.lower()
        self.assertIn("last four digits", lower)
        self.assertIn("pincode", lower)
        self.assertNotIn("shipped", lower)
        self.assertNotIn("160017", reply)
        self.assertEqual(agent.last_evidence, [])

    def test_ticket_creation_without_reference_does_not_raise_or_invent_id(self) -> None:
        agent = voice_batch.VoiceStep3Agent(run, client=None, chunks=[], vectors=[])
        with patch.object(agent, "create_ticket", return_value={"created": True}):
            reply = agent.respond("Please connect me to a human support agent.")
        self.assertIn("ticket", reply.lower())
        self.assertIn("reference number is unavailable", reply.lower())
        self.assertIsNone(agent.last_ticket_id)
        self.assertNotIn("None", reply)

    def test_normal_ticket_creation_survives_gemini_generation_failure(self) -> None:
        agent = voice_batch.VoiceStep3Agent(run, client=None, chunks=[], vectors=[])
        # Simulate a verified high-value damaged item to reach the normal escalation branch.
        agent.verified_order = {
            "order_id": "69520867", "items": [{"sku": "ELE-0012", "title": "Laptop", "qty": "1", "unit_price_inr": "12001"}],
            "contacts_last_7_days": 0,
        }
        with patch.object(agent, "create_ticket", return_value={"created": True, "ticket_id": "TKT-OFFLINE-9"}):
            with patch.object(run, "retrieve", return_value=[]):
                with patch.object(run, "generate_answer", side_effect=RuntimeError("simulated provider quota")):
                    reply = agent.respond("My damaged laptop is broken.")
        self.assertIn("ticket was created successfully", reply.lower())
        self.assertIn("TKT-OFFLINE-9", reply)
        self.assertIn("Priority Resolution", reply)
        self.assertIn("cannot promise", reply.lower())
        self.assertEqual(agent.last_ticket_id, "TKT-OFFLINE-9")

    def test_ticket_id_is_cleared_before_each_new_turn(self) -> None:
        agent = voice_batch.VoiceStep3Agent(run, client=None, chunks=[], vectors=[])
        with patch.object(agent, "create_ticket", return_value={"created": True, "ticket_id": "TKT-VOICE-1"}):
            first = agent.respond("Please connect me to a human support agent.")
        self.assertIn("TKT-VOICE-1", first)
        self.assertEqual(agent.last_ticket_id, "TKT-VOICE-1")
        with patch.object(run, "retrieve", return_value=[]):
            with patch.object(run, "generate_answer", return_value="The current return policy is described in the returns policy."):
                agent.respond("What is the return policy?")
        self.assertIsNone(agent.last_ticket_id)

    def test_retrieval_failure_returns_a_grounded_fallback_without_model_call(self) -> None:
        agent = voice_batch.VoiceStep3Agent(run, client=None, chunks=[], vectors=[])
        with patch.object(run, "retrieve", side_effect=RuntimeError("simulated embedding outage")):
            with patch.object(run, "generate_answer", side_effect=AssertionError("do not generate without retrieval")):
                reply = agent.respond("What is the return policy?")
        lower = reply.lower()
        self.assertIn("evidence-retrieval service is temporarily unavailable", lower)
        self.assertIn("cannot confidently answer", lower)
        self.assertEqual(agent.last_evidence, [])

    def test_general_generation_failure_returns_safe_non_ticket_fallback(self) -> None:
        agent = voice_batch.VoiceStep3Agent(run, client=None, chunks=[], vectors=[])
        with patch.object(run, "retrieve", return_value=[]):
            with patch.object(run, "generate_answer", side_effect=RuntimeError("simulated provider outage")):
                reply = agent.respond("What is the return policy?")
        self.assertIn("detailed-response service is temporarily unavailable", reply.lower())
        self.assertNotIn("ticket was created successfully", reply.lower())

    def test_ticket_tool_exception_does_not_claim_early_handover_success(self) -> None:
        agent = voice_batch.VoiceStep3Agent(run, client=None, chunks=[], vectors=[])
        with patch.object(agent, "create_ticket", side_effect=OSError("simulated local write failure")):
            reply = agent.respond("Please connect me to a human support agent.")
        self.assertIn("couldn’t confirm", reply.lower())
        self.assertNotIn("ticket reference", reply.lower())
        self.assertIsNone(agent.last_ticket_id)

    def test_normal_ticket_tool_exception_and_model_failure_never_claim_handover(self) -> None:
        agent = voice_batch.VoiceStep3Agent(run, client=None, chunks=[], vectors=[])
        agent.verified_order = {
            "order_id": "69520867", "items": [{"sku": "ELE-0012", "title": "Laptop", "qty": "1", "unit_price_inr": "12001"}],
            "contacts_last_7_days": 0,
        }
        with patch.object(agent, "create_ticket", side_effect=OSError("simulated local write failure")):
            with patch.object(run, "retrieve", return_value=[]):
                with patch.object(run, "generate_answer", side_effect=RuntimeError("simulated provider outage")):
                    reply = agent.respond("My damaged laptop is broken.")
        self.assertIn("couldn’t confirm", reply.lower())
        self.assertIn("say the handover is complete", reply.lower())
        self.assertNotIn("ticket was created successfully", reply.lower())
        self.assertIsNone(agent.last_ticket_id)

    def test_safe_order_context_excludes_direct_identifiers_even_from_nested_items(self) -> None:
        order = {
            "order_id": "69520867", "status": "shipped", "delivery_pincode": "160017",
            "phone": "9999991234", "email": "customer@example.test", "address": "private address",
            "refund_rrn": "123456789012", "items": [{"sku": "ELE-0012", "title": "Kestrel K7", "qty": "1", "phone": "9999991234", "email": "customer@example.test"}],
        }
        safe = run._safe_order_context(order)
        self.assertNotIn("delivery_pincode", safe)
        self.assertNotIn("phone", safe)
        self.assertNotIn("email", safe)
        self.assertNotIn("address", safe)
        self.assertEqual(set(safe["items"][0]), {"sku", "title", "qty"})
        self.assertNotIn("9999991234", repr(safe))
        self.assertNotIn("customer@example.test", repr(safe))
        self.assertNotIn("160017", repr(safe))




class VoiceBatchOutputPathSafetyTests(unittest.TestCase):
    def test_same_stem_different_extensions_get_distinct_output_bases(self) -> None:
        import tempfile
        from pathlib import Path
        import voice_batch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            paths = [input_dir / "case.wav", input_dir / "case.mp3"]
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"synthetic")

            bases = voice_batch._build_output_bases(input_dir, output_dir, paths)
            keys = [str(bases[path]).casefold() for path in paths]
            self.assertEqual(len(set(keys)), 2)
            self.assertIn("case.wav", bases[paths[0]].name.casefold())
            self.assertIn("case.mp3", bases[paths[1]].name.casefold())

    def test_output_names_are_case_insensitively_unique(self) -> None:
        import tempfile
        from pathlib import Path
        import voice_batch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            paths = [input_dir / "Call.wav", input_dir / "call.mp3"]
            bases = voice_batch._build_output_bases(input_dir, output_dir, paths)
            keys = [bases[path].relative_to(output_dir).as_posix().casefold() for path in paths]
            self.assertEqual(len(keys), len(set(keys)))

    def test_collision_suffix_cannot_overwrite_an_existing_unique_stem(self) -> None:
        import tempfile
        from pathlib import Path
        import voice_batch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            paths = [
                input_dir / "call.wav",
                input_dir / "call.mp3",
                input_dir / "call.wav.mp3",
            ]
            bases = voice_batch._build_output_bases(input_dir, output_dir, paths)
            keys = [bases[path].relative_to(output_dir).as_posix().casefold() for path in paths]
            self.assertEqual(len(keys), len(set(keys)))
            self.assertNotEqual(bases[paths[0]], bases[paths[2]])

    def test_same_basename_in_separate_subdirectories_keeps_separate_outputs(self) -> None:
        import tempfile
        from pathlib import Path
        import voice_batch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            paths = [input_dir / "caller_a" / "clip.wav", input_dir / "caller_b" / "clip.wav"]
            bases = voice_batch._build_output_bases(input_dir, output_dir, paths)
            self.assertNotEqual(bases[paths[0]], bases[paths[1]])
            self.assertEqual(bases[paths[0]].relative_to(output_dir).parts[0], "caller_a")
            self.assertEqual(bases[paths[1]].relative_to(output_dir).parts[0], "caller_b")

    def test_run_batch_rejects_same_input_and_output_directory_before_credentials(self) -> None:
        import contextlib
        import io
        import tempfile
        from pathlib import Path
        from argparse import Namespace
        import voice_batch

        with tempfile.TemporaryDirectory() as tmp:
            same_dir = Path(tmp) / "same"
            same_dir.mkdir()
            args = Namespace(
                input_dir=str(same_dir), output_dir=str(same_dir),
                stt_language="en-IN", tts_language="en-IN", voice="Kaveri",
                tts_model="timbre-v2.5", speed=1.0,
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result = voice_batch.run_batch(args)
            self.assertEqual(result, 2)
            self.assertIn("input and output", output.getvalue().lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
