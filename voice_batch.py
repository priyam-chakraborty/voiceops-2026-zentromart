"""VoiceOps Step 4 batch runner using Gnani STT/TTS and the local Step 3 RAG agent.

Input convention:
- Audio files directly inside --input-dir are treated as independent conversations.
- Audio files inside the same immediate subfolder share conversation state, in filename order.
- Each audio clip should be no longer than 60 seconds for the STT REST endpoint.

Outputs per clip: transcript.txt, reply.txt, reply.wav (when TTS succeeds), and result.json.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

SUPPORTED_AUDIO = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}


def _safe_error(exc: Exception) -> str:
    message = str(exc)
    for key_name in ("GEMINI_API_KEY", "GNANI_API_KEY"):
        secret = os.getenv(key_name, "")
        if secret:
            message = message.replace(secret, "[REDACTED]")
    return f"{type(exc).__name__}: {message}"


def _speech_friendly(text: str) -> str:
    """Remove common Markdown and retrieval citations before speaking the reply."""
    value = text
    value = re.sub(r"\[([A-Z][A-Z0-9]*-\d+(?:\s*,\s*[A-Z][A-Z0-9]*-\d+)*)\]", "", value)
    value = value.replace("**", "").replace("__", "").replace("`", "")
    value = re.sub(r"(?m)^\s*[-*#]+\s*", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _chunk_for_json(chunk: dict[str, Any]) -> dict[str, Any]:
    keys = ("section_id", "source", "title", "doc_type", "status", "effective_from", "retrieval_score", "text")
    return {key: chunk.get(key) for key in keys if key in chunk}


class VoiceStep3Agent:
    """Stateful Step 3 agent wrapper so audio transcripts can use the same tools and RAG."""

    def __init__(self, core: Any, client: Any, chunks: list[dict], vectors: list[list[float]]):
        self.core = core
        self.client = client
        self.chunks = chunks
        self.vectors = vectors
        self.history: list[tuple[str, str]] = []
        self.current_order_id: str | None = None
        self.verified_order: dict | None = None
        self.verification_attempts = 0
        self.verification_locked = False
        self.first_reply = True
        self.last_evidence: list[dict] = []
        self.last_ticket_id: str | None = None

        if str(core.DATASET) not in sys.path:
            sys.path.insert(0, str(core.DATASET))
        from tools.tools import lookup_order, create_ticket
        self.lookup_order = lookup_order
        self.create_ticket = create_ticket

    def _finish(self, question: str, answer: str, sources: list[dict] | None = None) -> str:
        answer = answer.strip()
        greeting_starts = (
            "i’m zentromart", "i'm zentromart", "i am zentromart",
            "hello, i'm zentromart", "hello, i’m zentromart",
            "hello! i am zentromart", "hello! i'm zentromart",
            "hi, i'm zentromart", "hi, i’m zentromart",
        )
        if self.first_reply and not answer.lower().startswith(greeting_starts):
            answer = "I’m ZentroMart’s AI support assistant. " + answer
        self.first_reply = False
        self.last_evidence = list(sources or [])
        self.history.extend([("Customer", question), ("Agent", answer)])
        self.history = self.history[-20:]
        return answer

    # VOICEOPS_VOICE_BATCH_HANDOVER_FALLBACK_V1
    def respond(self, question: str) -> str:
        question = question.strip()
        self.last_evidence = []
        # A batch result reports the ticket for this clip, never a previous clip.
        self.last_ticket_id = None
        if not question:
            return self._finish(question, "Sorry, I didn't catch that. Could you say it again?")

        candidate_order_id = self.core._extract_order_id(question)
        if candidate_order_id and candidate_order_id != self.current_order_id:
            self.current_order_id = candidate_order_id
            self.verified_order = None
            self.verification_attempts = 0
            self.verification_locked = False

        verification_value = self.core._extract_verification_value(question, self.current_order_id)

        # Handle explicit human/safety/legal handovers using the provided mock ticket tool.
        early_reason = self.core._escalation_reason(question, self.verified_order)
        if early_reason and any(term in early_reason.lower() for term in ("human", "legal", "reported order, payment", "safety risk")):
            summary = (
                f"Issue: {question[:350]}; intent: {self.core._intent_for(question)}; "
                f"order ID provided: {self.current_order_id or 'none'}; "
                f"checks: reviewed customer message and applicable support workflow; "
                f"escalation reason: {early_reason}"
            )
            try:
                result = self.create_ticket(summary=summary, order_id=self.current_order_id, reason=early_reason,
                                            intent=self.core._intent_for(question))
            except Exception as exc:
                print(f"WARNING: Support-ticket creation failed ({type(exc).__name__}); no handover was confirmed.")
                result = {"created": False}
            if result.get("created") is True:
                self.last_ticket_id = str(result.get("ticket_id") or "").strip() or None
                reference_text = (
                    f"Your ticket reference is {self.last_ticket_id}. "
                    if self.last_ticket_id else "A reference number is unavailable. "
                )
                answer = (
                    "I’m ZentroMart’s AI support assistant. I’ve created a support ticket for a human colleague. "
                    + reference_text
                    + "The team will review the issue; I cannot promise a particular outcome or timeline."
                )
            else:
                answer = ("I’m ZentroMart’s AI support assistant. I couldn’t confirm that a support ticket was created, "
                          "so I cannot claim a handover is complete. Please contact ZentroMart support directly.")
            return self._finish(question, answer)

        # Do not disclose order details until the dataset tool verifies the customer.
        if self.current_order_id and self.verified_order is None and not self.verification_locked:
            if verification_value:
                result = self.lookup_order(self.current_order_id, verification_value)
                if result.get("verified") is True and result.get("found") is True:
                    self.verified_order = result
                    self.verification_attempts = 0
                else:
                    self.verification_attempts += 1
                    if self.verification_attempts >= 2:
                        self.verification_locked = True
                        answer = ("I couldn’t verify the order after two attempts, so I can’t share its details. "
                                  "Please contact support from the phone number registered to the account. "
                                  "I can still answer general policy questions.")
                    else:
                        answer = ("Those details did not match our verification check. Please try once more with either "
                                  "the last four digits of the registered phone number or the six-digit delivery pincode. "
                                  "I won’t reveal any order details until verification succeeds.")
                    return self._finish(question, answer)
            elif candidate_order_id or self.core._needs_specific_order_lookup(question):
                self.lookup_order(self.current_order_id)
                answer = ("I’m ZentroMart’s AI support assistant. I can help with that order. Before sharing order-specific "
                          "information, please provide either the last four digits of the registered phone number or "
                          "the six-digit delivery pincode.")
                return self._finish(question, answer)
        elif self.verification_locked and self.core._needs_specific_order_lookup(question):
            answer = ("I can’t share details for that order because verification failed twice. Please contact support "
                      "from the phone number registered to the account. I can still help with general ZentroMart policies.")
            return self._finish(question, answer)

        if not self.current_order_id and self.core._needs_specific_order_lookup(question):
            answer = ("I’m ZentroMart’s AI support assistant. I can help with your order. Please share the eight-digit Order ID first. "
                      "Before I discuss order-specific details, I’ll ask you to verify either the last four digits of the registered "
                      "phone number or the six-digit delivery pincode.")
            return self._finish(question, answer)

        # Ground the answer in retrieved current documents and the limited verified order context.
        prior_customer_turns = [msg for role, msg in self.history if role == "Customer"][-3:]
        retrieval_query = question
        if prior_customer_turns:
            retrieval_query = "Recent customer context: " + " | ".join(prior_customer_turns) + "\nCurrent question: " + question
        try:
            sources = self.core.retrieve(self.client, retrieval_query, self.chunks, self.vectors)
        except Exception as exc:
            print(f"WARNING: Evidence retrieval failed ({type(exc).__name__}); returning a safe fallback.")
            return self._finish(
                question,
                "I’m ZentroMart’s AI support assistant. The evidence-retrieval service is temporarily unavailable, "
                "so I cannot confidently answer from current company policy right now. Please try again shortly "
                "or contact ZentroMart support directly.",
            )
        reason = self.core._escalation_reason(question, self.verified_order)
        required_sections: list[str] = []
        if reason:
            required_sections.extend(["ESC-1", "ESC-2", "ESC-3", "ESC-4"])
        if reason and "more than ₹10,000" in reason:
            required_sections.append("DWM-3")
        if required_sections:
            sources = self.core._ensure_policy_sections(sources, self.chunks, required_sections)

        context_parts = []
        for idx, chunk in enumerate(sources, start=1):
            context_parts.append(
                f"[SOURCE {idx}] file={chunk['source']}; section=[{chunk['section_id']}]; "
                f"doc_type={chunk['doc_type']}; status={chunk['status']}; "
                f"effective_from={chunk['effective_from']}\n{chunk['text']}"
            )
        if self.history:
            context_parts.append("Recent conversation history (for reference only):\n" +
                                 "\n".join(f"{role}: {msg}" for role, msg in self.history[-12:]))
        if self.verified_order:
            context_parts.append(
                "Verified synthetic order-tool data (caller verification succeeded; do not reveal restricted identity fields):\n" +
                json.dumps(self.core._safe_order_context(self.verified_order), ensure_ascii=False)
            )
        context_parts.append(
            "Voice interaction note: keep the reply concise and suitable to hear aloud. "
            "Reply in the customer's language when it is clear. Do not read section citations aloud as part of the answer."
        )
        context = "Retrieved company-document excerpts (may not fully answer the question):\n\n" + "\n\n".join(context_parts)

        if reason:
            summary = (
                f"Issue: {question[:350]}; intent: {self.core._intent_for(question)}; "
                f"order ID: {self.current_order_id or 'not provided'}; "
                f"checks: retrieved current company-document sections" + (", verified order data" if self.verified_order else "; no order details disclosed") + "; "
                f"escalation reason: {reason}"
            )
            try:
                ticket = self.create_ticket(summary=summary, order_id=self.current_order_id, reason=reason,
                                            intent=self.core._intent_for(question))
            except Exception as exc:
                print(f"WARNING: Support-ticket creation failed ({type(exc).__name__}); no handover was confirmed.")
                ticket = {"created": False}
            ticket_created = ticket.get("created") is True
            self.last_ticket_id = str(ticket.get("ticket_id") or "").strip() or None if ticket_created else None
            extra = "\n\nConfirm the ticket honestly and do not promise an outcome or timeline."
            high_value_claim = "more than ₹10,000" in reason
            if high_value_claim:
                extra += (" This is a high-value damaged claim. Explicitly state that Priority Resolution handles claims "
                          "above ₹10,000 and that the AI cannot approve it, citing [DWM-3] and [ESC-1].")
            answer = self.core._generate_handover_answer(
                self.client, question, context, ticket,
                extra_handover_instruction=extra,
                high_value_claim=high_value_claim,
            )
        else:
            try:
                answer = self.core.generate_answer(self.client, question, self.core.STEP3_SYSTEM_PROMPT, context)
            except Exception as exc:
                print(f"WARNING: Detailed voice response generation failed ({type(exc).__name__}); using a safe fallback.")
                answer = (
                    "I’m ZentroMart’s AI support assistant. The detailed-response service is temporarily unavailable, "
                    "so I could not complete a grounded answer just now. Please try again shortly or contact "
                    "ZentroMart support directly."
                )

        return self._finish(question, answer, sources)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def _build_sessions(input_dir: Path) -> tuple[list[Path], dict[str, list[Path]]]:
    files = sorted((p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_AUDIO),
                   key=lambda p: str(p.relative_to(input_dir)).lower())
    sessions: dict[str, list[Path]] = {}
    for path in files:
        relative = path.relative_to(input_dir)
        if len(relative.parts) == 1:
            # Each top-level file is independent to avoid leaking one caller's context to another.
            session_key = f"single:{relative.as_posix()}"
        else:
            # Files in the same immediate subfolder share conversation state in filename order.
            session_key = f"folder:{relative.parts[0]}"
        sessions.setdefault(session_key, []).append(path)
    return files, sessions


# VOICEOPS_BATCH_OUTPUT_COLLISION_HARDENING_V1
def _build_output_bases(input_dir: Path, output_dir: Path, files: list[Path]) -> dict[Path, Path]:
    """Map each input audio file to a unique output base path.

    Audio files such as ``call.wav`` and ``call.mp3`` would otherwise both
    produce ``call.transcript.txt`` and overwrite each other's results. The
    mapping preserves legacy names for non-colliding files, disambiguates only
    collisions, and checks collisions case-insensitively for Windows portability.
    """
    import hashlib

    relative_bases = {path: path.relative_to(input_dir).with_suffix("") for path in files}

    def key(relative_path: Path) -> str:
        return relative_path.as_posix().casefold()

    counts: dict[str, int] = {}
    for relative_base in relative_bases.values():
        base_key = key(relative_base)
        counts[base_key] = counts.get(base_key, 0) + 1
    colliding = {path for path, relative_base in relative_bases.items() if counts[key(relative_base)] > 1}

    # Reserve every name that will keep its legacy path before assigning renamed paths.
    assigned_keys = {
        key(relative_base)
        for path, relative_base in relative_bases.items()
        if path not in colliding
    }
    resolved: dict[Path, Path] = {}
    ordered_files = sorted(
        files,
        key=lambda path: (
            path.relative_to(input_dir).as_posix().casefold(),
            path.relative_to(input_dir).as_posix(),
        ),
    )
    for path in ordered_files:
        relative_base = relative_bases[path]
        if path not in colliding:
            resolved[path] = output_dir / relative_base
            continue

        extension = path.suffix.lower().lstrip(".") or "audio"
        candidate = relative_base.with_name(f"{relative_base.name}.{extension}")
        candidate_key = key(candidate)
        if candidate_key in assigned_keys:
            digest = hashlib.sha256(path.relative_to(input_dir).as_posix().encode("utf-8")).hexdigest()[:10]
            candidate = relative_base.with_name(f"{relative_base.name}.{extension}.{digest}")
            candidate_key = key(candidate)
            counter = 1
            while candidate_key in assigned_keys:
                candidate = relative_base.with_name(f"{relative_base.name}.{extension}.{digest}.{counter}")
                candidate_key = key(candidate)
                counter += 1
        assigned_keys.add(candidate_key)
        resolved[path] = output_dir / candidate

    return resolved


def run_batch(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"ERROR: input folder does not exist: {input_dir}")
        return 2
    # An output directory nested under the input directory is allowed and excluded
    # from discovery below. Reject equal paths and output directories that contain
    # the input tree, which would cause discovery to filter out every input file.
    if input_dir == output_dir or output_dir in input_dir.parents:
        print("ERROR: input and output folders must differ; output cannot contain the input folder.")
        return 2

    if not os.getenv("GNANI_API_KEY", "").strip():
        print("ERROR: GNANI_API_KEY is missing from this project's .env file.")
        return 2
    if not os.getenv("GEMINI_API_KEY", "").strip():
        print("ERROR: GEMINI_API_KEY is missing from this project's .env file.")
        return 2

    files, sessions = _build_sessions(input_dir)
    # Exclude generated output if it happens to sit under the input tree.
    files = [p for p in files if output_dir not in p.parents and p != output_dir]
    if not files:
        print(f"No supported audio files found in {input_dir}.")
        print("Supported extensions: " + ", ".join(sorted(SUPPORTED_AUDIO)))
        return 2
    # Rebuild session map with the final file list.
    allowed = set(files)
    sessions = {key: [p for p in group if p in allowed] for key, group in sessions.items()}
    sessions = {key: group for key, group in sessions.items() if group}
    output_bases = _build_output_bases(input_dir, output_dir, files)

    import run as core
    try:
        from gnani.stt import GnaniSTTClient
        from gnani.tts import AudioConfig, GnaniTTSClient
    except ImportError as exc:
        print(f"ERROR: Cannot import Gnani SDK: {_safe_error(exc)}")
        print("Install it with: python -m pip install gnani-vachana")
        return 2

    try:
        core.require_dataset()
        client = core.get_client()
        chunks, vectors = core.load_or_build_index(client)
        stt = GnaniSTTClient()
        tts = GnaniTTSClient()
    except Exception as exc:
        print(f"ERROR: Could not initialise agent or Gnani clients: {_safe_error(exc)}")
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    agents: dict[str, VoiceStep3Agent] = {}
    session_key_for_file: dict[Path, str] = {}
    for key, group in sessions.items():
        agents[key] = VoiceStep3Agent(core, client, chunks, vectors)
        for path in group:
            session_key_for_file[path] = key

    success_count = 0
    failure_count = 0
    print(f"Found {len(files)} audio file(s). Output folder: {output_dir}")
    print(f"STT language: {args.stt_language}; TTS language: {args.tts_language}; voice: {args.voice}; model: {args.tts_model}")

    for index, audio_path in enumerate(files, start=1):
        relative = audio_path.relative_to(input_dir)
        out_base = output_bases[audio_path]
        out_base.parent.mkdir(parents=True, exist_ok=True)
        result: dict[str, Any] = {
            "input_audio": str(relative),
            "stt_language": args.stt_language,
            "tts_language": args.tts_language,
            "transcript": "",
            "reply_text": "",
            "reply_audio": None,
            "retrieved_evidence": [],
            "ticket_id": None,
            "status": "failed",
        }
        print(f"[{index}/{len(files)}] Processing {relative}...")
        try:
            stt_result = stt.transcribe(str(audio_path), language_code=args.stt_language)
            transcript = stt_result.get("transcript", "") if isinstance(stt_result, dict) else ""
            if not transcript or not transcript.strip():
                raise RuntimeError("Gnani STT returned an empty transcript.")
            result["transcript"] = transcript.strip()
            _write_text(out_base.with_name(out_base.name + ".transcript.txt"), transcript.strip())

            agent = agents[session_key_for_file[audio_path]]
            answer = agent.respond(transcript.strip())
            result["reply_text"] = answer
            result["retrieved_evidence"] = [_chunk_for_json(chunk) for chunk in agent.last_evidence]
            result["ticket_id"] = agent.last_ticket_id
            _write_text(out_base.with_name(out_base.name + ".reply.txt"),
                        "CUSTOMER TRANSCRIPT\n" + transcript.strip() + "\n\nAGENT REPLY\n" + answer +
                        "\n\nRETRIEVED EVIDENCE\n" + (
                            "\n".join(f"[{c.get('section_id')}] {c.get('source')} | status={c.get('status')} | score={c.get('retrieval_score')}" for c in agent.last_evidence)
                            if agent.last_evidence else "No document chunks were retrieved for this turn."
                        ))

            spoken = _speech_friendly(answer)
            audio_bytes = tts.synthesize(
                spoken,
                voice=args.voice,
                model=args.tts_model,
                language=args.tts_language,
                speed=args.speed,
                audio_config=AudioConfig(container="wav", encoding="linear_pcm", sample_rate=48000),
            )
            if not isinstance(audio_bytes, (bytes, bytearray)) or not audio_bytes:
                raise RuntimeError("Gnani TTS returned empty audio data.")
            reply_audio_path = out_base.with_name(out_base.name + ".reply.wav")
            reply_audio_path.write_bytes(audio_bytes)
            result["reply_audio"] = str(reply_audio_path.relative_to(output_dir))
            result["status"] = "success"
            success_count += 1
        except Exception as exc:
            result["error"] = _safe_error(exc)
            failure_count += 1
            print(f"  FAILED: {_safe_error(exc)}")
        finally:
            json_path = out_base.with_name(out_base.name + ".result.json")
            json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "total_audio_files": len(files),
        "success_count": success_count,
        "failure_count": failure_count,
        "stt_language": args.stt_language,
        "tts_language": args.tts_language,
        "voice": args.voice,
        "tts_model": args.tts_model,
        "results": [
            str(output_bases[p].with_name(output_bases[p].name + ".result.json").relative_to(output_dir))
            for p in files
        ],
    }
    (output_dir / "batch_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nVoiceOps batch run complete.")
    print(f"Successful files: {success_count}/{len(files)}")
    print(f"Failed files: {failure_count}/{len(files)}")
    print(f"Outputs saved to: {output_dir}")
    print("Each audio file has a .transcript.txt, .reply.txt, .result.json, and (if TTS succeeded) .reply.wav.")
    return 0 if failure_count == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transcribe a folder of customer audio with Gnani, run the Step 3 ZentroMart agent, and synthesize spoken replies."
    )
    parser.add_argument("--input-dir", required=True, help="Folder containing audio clips; nested subfolders can represent conversations.")
    parser.add_argument("--output-dir", default="voice_outputs", help="Folder for text, JSON evidence, and WAV replies (default: voice_outputs).")
    parser.add_argument("--stt-language", default="en-IN", help="Gnani STT language code, e.g. en-IN, hi-IN, or en-IN for Hinglish transcription.")
    parser.add_argument("--tts-language", default="en-IN", help="Gnani TTS language code, e.g. en-IN, hi-IN, or hi-en for Hinglish.")
    parser.add_argument("--voice", default="Kaveri", help="Gnani voice (default: Kaveri).")
    parser.add_argument("--tts-model", default="timbre-v2.5", help="Gnani TTS model (default: timbre-v2.5).")
    parser.add_argument("--speed", type=float, default=1.0, help="TTS speed, default 1.0.")
    args = parser.parse_args()
    return run_batch(args)


if __name__ == "__main__":
    raise SystemExit(main())
