"""Small, safe smoke test for Gnani TTS and STT.

Loads GNANI_API_KEY from the project's .env file, generates a short WAV with Gnani TTS,
then transcribes that WAV with Gnani STT. It does not print or expose the API key.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


def main() -> int:
    if not os.getenv("GNANI_API_KEY", "").strip():
        print("ERROR: GNANI_API_KEY is missing in this folder's .env file.")
        print("Copy this script into your VoiceOps project folder and run it there.")
        return 1

    try:
        from gnani.stt import GnaniSTTClient
        from gnani.tts import AudioConfig, GnaniTTSClient
    except ImportError as exc:
        print(f"ERROR: Gnani SDK import failed: {exc}")
        print("In your activated .venv, run: python -m pip install gnani-vachana")
        return 1

    wav_path = ROOT / "gnani_voice_test.wav"
    test_text = (
        "Hello, this is a short voice test for the ZentroMart customer support assistant. "
        "How can I help you today?"
    )

    try:
        print("1/2 Calling Gnani text-to-speech. Please wait...")
        tts = GnaniTTSClient()
        audio_bytes = tts.synthesize(
            test_text,
            voice="Kaveri",
            model="timbre-v2.5",
            language="en-IN",
            speed=1.0,
            audio_config=AudioConfig(
                container="wav",
                encoding="linear_pcm",
                sample_rate=48000,
            ),
        )
        if not isinstance(audio_bytes, (bytes, bytearray)) or not audio_bytes:
            raise RuntimeError("Gnani TTS returned empty audio data.")
        wav_path.write_bytes(audio_bytes)
        print(f"TTS PASS: audio saved to {wav_path.name} ({len(audio_bytes)} bytes).")
    except Exception as exc:
        print("TTS FAILED. The SDK installed, but the service call did not complete.")
        print(f"Error type: {type(exc).__name__}")
        print(f"Details: {exc}")
        print("Do not paste your API key in chat. Share this error text only after checking it contains no secrets.")
        return 1

    try:
        print("2/2 Calling Gnani speech-to-text on the generated test audio...")
        stt = GnaniSTTClient()
        result = stt.transcribe(str(wav_path), language_code="en-IN")
        transcript = result.get("transcript", "") if isinstance(result, dict) else ""
        if not transcript.strip():
            raise RuntimeError(f"STT returned no transcript. Response type: {type(result).__name__}")
        print("STT PASS: transcript received:")
        print(transcript)
        print("\nGNANI VOICE CHECK PASSED: both TTS and STT API calls worked.")
        print(f"Keep {wav_path.name}; it is a generated test file that you may use for a quick playback check.")
        return 0
    except Exception as exc:
        print("STT FAILED. TTS worked, but the transcription service call did not complete.")
        print(f"Error type: {type(exc).__name__}")
        print(f"Details: {exc}")
        print("Do not paste your API key in chat. Share this error text only after checking it contains no secrets.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
