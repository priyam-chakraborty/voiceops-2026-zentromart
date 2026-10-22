# VoiceOps 2026: ZentroMart Support Agent

## Overview

This repository contains a customer-support agent for the VoiceOps 2026 capstone. It supports prompt-only answers, document-grounded RAG, selected order-verification and escalation flows, and a Gnani.ai voice batch runner.

> Before submission: replace this note with your actual team/participant details and public repository URL. Confirm every command below against the files you include.

## Requirements

- Python 3.13 (the development machine used Python 3.13.6)
- Gemini API key for generation and embeddings
- Gnani.ai API key for both STT and TTS
- The supplied capstone dataset extracted beside the project during local testing; do not include a copy of the dataset in the submitted ZIP or public repository

## Setup on Windows

Open Command Prompt in the project directory and run:

```bat
py -3.13 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create a local `.env` file by copying `.env.example`, then add your own keys:

```text
GEMINI_API_KEY=
GEMINI_MODEL=gemini-3.6-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
GNANI_API_KEY=
```

Use the model name that has actually been verified for your Gemini project. Do not commit `.env`, API keys, `.venv`, or dataset files.

## Validate and run

Validate the extracted dataset:

```bat
python run.py check
```

Run the prompt-only chatbot:

```bat
python run.py step1
```

Run RAG:

```bat
python run.py step2
```

Run the conversation/tool agent:

```bat
python run.py step3
```

Run the voice batch pipeline (the required batch-style entry point):

```bat
python voice_batch.py --input-dir input_audio --output-dir voice_outputs
```

Each input audio file should be a short customer turn (keep clips within Gnani's documented 60-second STT limit). The batch output includes `*.transcript.txt`, `*.reply.txt`, `*.result.json` and, when TTS succeeds, `*.reply.wav`, plus a batch summary. `*.reply.txt` includes the answer and retrieved evidence where available.

## Evaluation

The evaluation helper can compare Step 1 and Step 2 on the supplied practice questions:

```bat
python evaluate.py --limit 5
```

The helper uses heuristic checks, not official course scoring. Review errors and manually inspect borderline cases. Full evaluation requires API quota and should only be run when quota is available.

## Security and data handling

- Use only the synthetic dataset supplied for the capstone.
- Never commit `.env`, tokens, API keys, `.venv`, or a duplicate of the supplied dataset.
- Only report ticket creation when the mock tool confirms success.
- Verify a caller before revealing order-specific details.
- Gnani.ai is used for both STT and TTS; do not substitute another speech service.

## Submission information

- **Public repository URL:** [Add URL]
- **Exact batch command:** `python voice_batch.py --input-dir input_audio --output-dir voice_outputs`
- **Report:** `report.pdf` (maximum four pages)
- **Demo:** `demo.mp4` (maximum two minutes)
