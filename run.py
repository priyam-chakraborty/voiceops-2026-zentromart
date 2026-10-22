from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "Capstone_Project_Dataset_FTTT_2026"
load_dotenv(ROOT / ".env")

SYSTEM_PROMPT = """
You are ZentroMart's customer support assistant for a fictional Indian online store.
Start by making clear that you are an AI assistant. Be polite, concise, and practical.
For Step 1, you do NOT have access to company policy documents or order records.
Do not invent specific ZentroMart policy dates, return windows, refund timelines, amounts,
order statuses, or tool actions. When the exact policy is unknown, say you need to check
ZentroMart's official policy instead of guessing. Never claim that you looked up an order,
created a ticket, or contacted a human unless a real tool call confirmed it.
""".strip()

RAG_SYSTEM_PROMPT = """
You are ZentroMart's customer support assistant for a fictional Indian online store.
Use only the supplied retrieved company-document excerpts and explicit conversation context.
Do not invent policy rules, dates, deadlines, amounts, stock status, or order details.
If the excerpts do not support an answer, clearly say that you cannot confirm it from the
available policy documents and offer a human handover when appropriate.
When documents conflict, prefer a current policy, then an agent guide, help article, FAQ,
and finally a marketing page. A superseded document is not current policy authority.
Cite the section ID(s) used, in square brackets, for example [RET-2]. Be concise and kind.
This first RAG stage does not have order lookup or ticket tools, so never claim to have used them.
""".strip()


def require_dataset() -> None:
    if not DATASET.exists():
        raise FileNotFoundError(
            f"Dataset folder is missing: {DATASET}\n"
            "Extract Capstone_Project_Dataset_FTTT_2026.zip into this project folder."
        )


def require_gemini_key() -> str:
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Copy .env.example to .env and add your Gemini API key. "
            "Do not send the key in chat. See README_BEGINNER.md."
        )
    return key


def get_client():
    require_gemini_key()
    from google import genai
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"].strip())


def generate_answer(client, question: str, system_instruction: str, extra_context: str = "") -> str:
    from google.genai import types
    model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()
    prompt = question if not extra_context else f"{extra_context}\n\nCustomer question: {question}"
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.2,
        ),
    )
    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError("The model returned no text. Try again or check your API usage limits.")
    return text.strip()


def step1() -> None:
    client = get_client()
    print("\nZentroMart Step 1: prompt-only chatbot")
    print("This version does not retrieve company documents. Type exit to stop.\n")
    while True:
        question = input("Customer: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("Good work. You have finished a first Step 1 test.")
            return
        if not question:
            continue
        try:
            print("Agent:", generate_answer(client, question, SYSTEM_PROMPT), "\n")
        except Exception as exc:
            print(f"ERROR: {exc}\n")


def parse_frontmatter(text: str) -> dict[str, str]:
    meta: dict[str, str] = {}
    if not text.startswith("---"):
        return meta
    parts = text.split("---", 2)
    if len(parts) < 3:
        return meta
    for line in parts[1].splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip().strip('"\'')
    return meta


def read_knowledge_chunks() -> list[dict]:
    import re
    chunks: list[dict] = []
    kb_dir = DATASET / "kb"
    for path in sorted(kb_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        metadata = parse_frontmatter(raw)
        body = raw.split("---", 2)[2] if raw.startswith("---") and raw.count("---") >= 2 else raw
        # The dataset uses stable IDs in headings, for example: ## [RET-2] Return windows.
        heading_pattern = re.compile(r"(?m)^#{1,4}\s+\[([^\]]+)\]\s*(.*)$")
        matches = list(heading_pattern.finditer(body))
        if not matches:
            section_id = path.stem.upper()
            chunks.append(_chunk(path, metadata, section_id, body.strip()))
            continue
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
            section_text = body[start:end].strip()
            chunks.append(_chunk(path, metadata, match.group(1).strip(), section_text))
    return chunks


def _chunk(path: Path, metadata: dict[str, str], section_id: str, text: str) -> dict:
    return {
        "source": path.name,
        "section_id": section_id,
        "title": metadata.get("title", path.stem),
        "doc_type": metadata.get("doc_type", "unknown"),
        "status": metadata.get("status", "current"),
        "effective_from": metadata.get("effective_from", "unknown"),
        "text": text,
    }


def _embed_texts(client, texts: list[str], model: str, task_type: str) -> list[list[float]]:
    from google.genai import types
    vectors: list[list[float]] = []
    # Small batches keep requests manageable and make API errors easier to diagnose.
    batch_size = 8
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        result = client.models.embed_content(
            model=model,
            contents=batch,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=768,
            ),
        )
        got = getattr(result, "embeddings", None) or []
        if len(got) != len(batch):
            raise RuntimeError(
                f"Embedding API returned {len(got)} vectors for a batch of {len(batch)} texts."
            )
        vectors.extend([[float(x) for x in item.values] for item in got])
        print(f"Embedded {min(start + len(batch), len(texts))}/{len(texts)} document sections...", end="\r")
    print()
    return vectors


def load_or_build_index(client) -> tuple[list[dict], list[list[float]]]:
    import hashlib
    import json
    cache_dir = ROOT / ".cache"
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / "rag_index.json"
    chunks = read_knowledge_chunks()
    fingerprint_source = "\n".join(
        c["source"] + "|" + c["section_id"] + "|" + c["text"] + "|" + c["status"]
        for c in chunks
    )
    fingerprint_source += os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
    fingerprint = hashlib.sha256(fingerprint_source.encode("utf-8")).hexdigest()
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if cached.get("fingerprint") == fingerprint:
                print(f"Loaded {len(cached['chunks'])} sections from local embedding cache.")
                return cached["chunks"], cached["vectors"]
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    print(f"Building RAG index from {len(chunks)} company-document sections...")
    embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001").strip()
    texts = [
        f"Title: {c['title']}\nType: {c['doc_type']}\nStatus: {c['status']}\n"
        f"Effective from: {c['effective_from']}\nSection: [{c['section_id']}]\n{c['text']}"
        for c in chunks
    ]
    vectors = _embed_texts(client, texts, embedding_model, "RETRIEVAL_DOCUMENT")
    cache_file.write_text(
        json.dumps({"fingerprint": fingerprint, "chunks": chunks, "vectors": vectors}),
        encoding="utf-8",
    )
    print(f"Saved local index to {cache_file.relative_to(ROOT)}.")
    return chunks, vectors


def retrieve(client, question: str, chunks: list[dict], vectors: list[list[float]], top_k: int = 5) -> list[dict]:
    import numpy as np
    embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001").strip()
    q_vectors = _embed_texts(client, [question], embedding_model, "RETRIEVAL_QUERY")
    q = np.asarray(q_vectors[0], dtype=np.float32)
    matrix = np.asarray(vectors, dtype=np.float32)
    q_norm = np.linalg.norm(q) + 1e-12
    doc_norms = np.linalg.norm(matrix, axis=1) + 1e-12
    scores = (matrix @ q) / (doc_norms * q_norm)
    adjusted = []
    for idx, score in enumerate(scores.tolist()):
        chunk = chunks[idx]
        # Small reliability penalties help avoid treating superseded policies as current.
        if chunk["status"].lower() == "superseded":
            score -= 0.18
        if chunk["doc_type"].lower() == "policy" and chunk["status"].lower() == "current":
            score += 0.025
        adjusted.append((score, idx))
    selected = [chunks[i] | {"retrieval_score": round(score, 4)} for score, i in
                sorted(adjusted, reverse=True)[:top_k]]
    return selected


def step2() -> None:
    require_dataset()
    client = get_client()
    chunks, vectors = load_or_build_index(client)
    print("\nZentroMart Step 2: RAG chatbot")
    print("Answers are based on retrieved company document sections. Type exit to stop.\n")
    while True:
        question = input("Customer: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("Good work. You have tested document-grounded answers.")
            return
        if not question:
            continue
        try:
            sources = retrieve(client, question, chunks, vectors)
            context_parts = []
            for i, chunk in enumerate(sources, start=1):
                context_parts.append(
                    f"[SOURCE {i}] file={chunk['source']}; section=[{chunk['section_id']}]; "
                    f"doc_type={chunk['doc_type']}; status={chunk['status']}; "
                    f"effective_from={chunk['effective_from']}\n{chunk['text']}"
                )
            context = "Retrieved company-document excerpts (may not fully answer the question):\n\n" + "\n\n".join(context_parts)
            answer = generate_answer(client, question, RAG_SYSTEM_PROMPT, context)
            print("Agent:", answer, "\n")
            print("Retrieved evidence:")
            for chunk in sources:
                print(f"  [{chunk['section_id']}] {chunk['source']} (status={chunk['status']}, score={chunk['retrieval_score']})")
            print()
        except Exception as exc:
            print(f"ERROR: {exc}\n")



STEP3_SYSTEM_PROMPT = """
You are ZentroMart's conversational customer support assistant for a fictional Indian online store.
On the first reply, identify yourself as an AI assistant. Be concise, patient and helpful.
Use the retrieved current company-document excerpts and the verified order-tool data supplied in the context.
Do not invent rules, dates, deadlines, amounts, stock status, order statuses or tool actions.
Prefer current policy, then agent guides, help articles, FAQs and marketing material. Superseded documents
are not current policy authority. Cite section IDs in square brackets, such as [RET-2].
For order-specific information, only use the verified order data supplied in context. Never reveal a full
phone number, email or delivery address, and never echo identity-verification codes.
The tool layer performs order verification and ticket creation. Never claim an order lookup succeeded unless
verified tool data is supplied. Never claim a ticket was created unless the ticket tool explicitly succeeded.
If the documents do not support an answer, say you cannot confirm it and offer a human handover when appropriate.
The dataset reference date is 20 October 2026. Be clear when a date or timeline depends on this reference date.
""".strip()


def _number_groups(text: str) -> list[str]:
    import re
    groups = re.findall(r"(?<![A-Za-z0-9])(?:\d[\s-]*){4,12}(?!\d)", text)
    result = []
    for group in groups:
        digits = "".join(ch for ch in group if ch.isdigit())
        if len(digits) in {4, 6, 8} and digits not in result:
            result.append(digits)
    return result


def _extract_order_id(text: str) -> str | None:
    for digits in _number_groups(text):
        if len(digits) == 8:
            return digits
    return None


def _extract_verification_value(text: str, order_id: str | None) -> str | None:
    for digits in _number_groups(text):
        if digits != order_id and len(digits) in {4, 6}:
            return digits
    return None


def _needs_specific_order_lookup(question: str) -> bool:
    import re
    q = question.lower().replace("’", "'")
    patterns = [
        r"\bmy order\b", r"\bmy refund\b", r"\bmy parcel\b", r"\bmy delivery\b",
        r"\bmy shipment\b", r"\border status\b", r"\btrack my order\b",
        r"\bwhere is (?:my|the) (?:order|parcel|package)\b",
        r"\b(?:order|parcel|package) (?:still )?(?:hasn't|has not|never) arrived\b",
        r"\b(?:haven't|have not|didn't|did not) receive (?:my|the) (?:order|parcel|package)\b",
        r"\bcancel my order\b", r"\bhelp with (?:my )?order\b",
        r"\brefund for my order\b", r"\btracking (?:for )?my order\b",
    ]
    return any(re.search(p, q) for p in patterns)


def _safe_order_context(order: dict) -> dict:
    """Return only order fields needed for support. Excludes phone, email and pincode."""
    allowed = [
        "order_id", "status", "placed_at", "promised_date", "shipped_on", "delivered_on",
        "payment_method", "delivery_type", "delivery_attempts", "subtotal_inr", "total_inr",
        "coupon_code", "coupon_discount_inr", "delivery_fee_inr", "cod_fee_inr",
        "cancellation_refund_initiated_on", "return_refund_initiated_on",
        "refund_initiated_on", "refund_rrn", "return_rrn", "cancellation_rrn",
        "contacts_last_7_days", "customer_first_name", "is_plus", "items",
    ]
    clean = {key: order.get(key) for key in allowed if key in order and order.get(key) not in (None, "")}
    clean_items = []
    for item in order.get("items", []) or []:
        clean_items.append({k: item.get(k) for k in ("sku", "title", "return_category", "qty", "unit_price_inr") if item.get(k) not in (None, "")})
    if "items" in clean:
        clean["items"] = clean_items
    return clean


def _intent_for(question: str) -> str:
    q = question.lower()
    if any(k in q for k in ("refund", "money back", "rrn")):
        return "refund"
    if any(k in q for k in ("cancel", "cancellation")):
        return "cancellation"
    if any(k in q for k in ("damage", "damaged", "broken", "defective", "expired", "wrong item", "missing")):
        return "damaged_wrong_missing"
    if any(k in q for k in ("delivery", "arrived", "tracking", "track", "shipment", "parcel")):
        return "delivery_tracking"
    if any(k in q for k in ("warranty", "repair", "replacement")):
        return "warranty"
    if any(k in q for k in ("compensation", "compensate", "waive", "exception")):
        return "compensation_or_exception"
    return "general_support"


def _escalation_reason(question: str, verified_order: dict | None) -> str | None:
    q = question.lower().replace("’", "'")
    if any(k in q for k in ("human agent", "human support", "speak to a human", "talk to a human", "real person", "live agent", "connect me to a person", "colleague")):
        return "Customer requested a human support agent."
    if any(k in q for k in ("legal action", "consumer court", "police complaint", "sue zentromart", "take legal action")):
        return "Customer mentioned legal action, consumer court or a police complaint."
    if any(k in q for k in ("not my order", "didn't place", "did not place", "unauthorized transaction", "unauthorised transaction", "someone used my account", "account activity i did not make")):
        return "Customer reported order, payment or account activity they did not make."
    if any(k in q for k in ("swelling battery", "battery is swollen", "burning smell", "smoke from", "sparks", "caught fire", "injury", "electrical shock")):
        return "Product-safety risk reported by customer."
    if any(k in q for k in ("compensation", "compensate me", "pay me extra", "waive the fee", "make an exception", "exception to the policy")):
        return "Customer requested compensation or a policy exception that the assistant cannot approve."
    if verified_order:
        try:
            if int(verified_order.get("contacts_last_7_days") or 0) >= 2:
                return "This is the third or later contact about the same issue within seven days."
        except (TypeError, ValueError):
            pass
        if any(k in q for k in ("damaged", "damage", "broken", "defective", "wrong item")):
            for item in verified_order.get("items", []) or []:
                try:
                    value = int(item.get("unit_price_inr") or 0) * int(item.get("qty") or 1)
                    if value > 10000:
                        return "A damaged, defective or wrong item appears to be worth more than ₹10,000."
                except (TypeError, ValueError):
                    continue
    return None


def _ensure_policy_sections(
    sources: list[dict], chunks: list[dict], required_ids: list[str]
) -> list[dict]:
    """Add policy sections required by a detected escalation to the evidence context.

    Similarity search is useful but must not omit mandatory escalation rules when
    a tool has already detected an escalation condition. Added sections are marked
    as required policy evidence rather than being assigned a fabricated score.
    """
    present = {item.get("section_id") for item in sources}
    for section_id in required_ids:
        if section_id in present:
            continue
        match = next(
            (item for item in chunks
             if item.get("section_id") == section_id
             and item.get("status", "current").lower() == "current"),
            None,
        )
        if match is not None:
            required = dict(match)
            required["retrieval_score"] = "required-policy"
            sources.append(required)
            present.add(section_id)
    return sources


def _display_evidence(sources: list[dict]) -> None:
    print("Retrieved evidence:")
    for chunk in sources:
        print(f"  [{chunk['section_id']}] {chunk['source']} (status={chunk['status']}, score={chunk['retrieval_score']})")
    print()


def step3() -> None:
    """Multi-turn RAG support agent with dataset-backed verification and ticket tools."""
    import json
    import sys as _sys
    require_dataset()
    client = get_client()
    chunks, vectors = load_or_build_index(client)
    tools_dir_parent = str(DATASET)
    if tools_dir_parent not in _sys.path:
        _sys.path.insert(0, tools_dir_parent)
    from tools.tools import lookup_order, create_ticket

    print("\nZentroMart Step 3: conversational RAG agent")
    print("Remembers recent turns, verifies order lookups, and can create handover tickets.")
    print("The dataset contains synthetic records only. Type exit to stop.\n")

    history: list[tuple[str, str]] = []
    current_order_id: str | None = None
    verified_order: dict | None = None
    verification_attempts = 0
    verification_locked = False
    last_ticket_id: str | None = None
    first_reply = True

    while True:
        question = input("Customer: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("Conversation ended. Thank you for testing Step 3.")
            return
        if not question:
            continue

        # Extract an order ID when it is supplied, even if the caller says it in groups.
        candidate_order_id = _extract_order_id(question)
        if candidate_order_id and candidate_order_id != current_order_id:
            current_order_id = candidate_order_id
            verified_order = None
            verification_attempts = 0
            verification_locked = False

        verification_value = _extract_verification_value(question, current_order_id)

        # An explicit human/safety/legal/escalation request can be ticketed without revealing order details.
        early_reason = _escalation_reason(question, verified_order)
        if early_reason and ("human" in early_reason.lower() or "legal" in early_reason.lower() or "reported order, payment" in early_reason.lower() or "safety risk" in early_reason.lower()):
            summary = (
                f"Issue: {question[:350]}; intent: {_intent_for(question)}; "
                f"order ID provided: {current_order_id or 'none'}; "
                f"checks: reviewed customer message and applicable support workflow; "
                f"escalation reason: {early_reason}"
            )
            result = create_ticket(summary=summary, order_id=current_order_id, reason=early_reason, intent=_intent_for(question))
            if result.get("created") is True:
                answer = f"I’m ZentroMart’s AI support assistant. I’ve created a support ticket for a human colleague. Your ticket reference is {result['ticket_id']}. The team will review the issue; I can’t promise a particular outcome or timeline."
                last_ticket_id = result["ticket_id"]
            else:
                answer = "I’m ZentroMart’s AI support assistant. I couldn’t confirm that a support ticket was created, so I can’t claim a handover is complete. Please contact ZentroMart support directly for human assistance."
            first_reply = False
            print("Agent:", answer, "\n")
            history.extend([("Customer", question), ("Agent", answer)])
            continue

        # Do not disclose order-specific information until lookup_order verifies the caller.
        if current_order_id and verified_order is None and not verification_locked:
            if verification_value:
                result = lookup_order(current_order_id, verification_value)
                if result.get("verified") is True and result.get("found") is True:
                    verified_order = result
                    verification_attempts = 0
                    # Continue to retrieval and answer this same turn using verified data.
                else:
                    verification_attempts += 1
                    if verification_attempts >= 2:
                        verification_locked = True
                        answer = "I couldn’t verify the order after two attempts, so I can’t share its details. For order-specific help, please contact support from the phone number registered to the account. I can still answer general policy questions."
                    else:
                        answer = "Those details did not match our verification check. Please try once more with either the last four digits of the registered phone number or the six-digit delivery pincode. I won’t reveal any order details until verification succeeds."
                    first_reply = False
                    print("Agent:", answer, "\n")
                    history.extend([("Customer", question), ("Agent", answer)])
                    continue
            elif candidate_order_id or _needs_specific_order_lookup(question):
                result = lookup_order(current_order_id)
                # The no-verification tool call is designed to return no sensitive order data.
                answer = "I’m ZentroMart’s AI support assistant. I can help with that order. Before sharing order-specific information, please provide either the last four digits of the registered phone number or the six-digit delivery pincode."
                first_reply = False
                print("Agent:", answer, "\n")
                history.extend([("Customer", question), ("Agent", answer)])
                continue
        elif verification_locked and _needs_specific_order_lookup(question):
            answer = "I can’t share details for that order because verification failed twice. Please contact support from the phone number registered to the account. I can still help with general ZentroMart policies."
            first_reply = False
            print("Agent:", answer, "\n")
            history.extend([("Customer", question), ("Agent", answer)])
            continue

        if not current_order_id and _needs_specific_order_lookup(question):
            answer = "I’m ZentroMart’s AI support assistant. I can help with your order. Please share the eight-digit Order ID first. Before I discuss order-specific details, I’ll ask you to verify either the last four digits of the registered phone number or the six-digit delivery pincode."
            first_reply = False
            print("Agent:", answer, "\n")
            history.extend([("Customer", question), ("Agent", answer)])
            continue

        try:
            # Include recent dialogue in retrieval queries so short follow-ups can use context.
            prior_customer_turns = [msg for role, msg in history if role == "Customer"][-3:]
            retrieval_query = question
            if prior_customer_turns:
                retrieval_query = "Recent customer context: " + " | ".join(prior_customer_turns) + "\nCurrent question: " + question
            sources = retrieve(client, retrieval_query, chunks, vectors)
            reason = _escalation_reason(question, verified_order)
            required_sections = []
            if reason:
                # Every handover should be grounded in the current escalation guide.
                required_sections.extend(["ESC-1", "ESC-2", "ESC-3", "ESC-4"])
            if reason and "more than ₹10,000" in reason:
                # High-value damaged-item claims have a mandatory Priority Resolution path.
                required_sections.append("DWM-3")
            if required_sections:
                sources = _ensure_policy_sections(sources, chunks, required_sections)
            context_parts = []
            for i, chunk in enumerate(sources, start=1):
                context_parts.append(
                    f"[SOURCE {i}] file={chunk['source']}; section=[{chunk['section_id']}]; "
                    f"doc_type={chunk['doc_type']}; status={chunk['status']}; "
                    f"effective_from={chunk['effective_from']}\n{chunk['text']}"
                )
            if history:
                recent_history = history[-12:]
                context_parts.append("Recent conversation history (for reference only):\n" + "\n".join(f"{role}: {msg}" for role, msg in recent_history))
            if verified_order:
                context_parts.append("Verified synthetic order-tool data (caller verification succeeded; do not reveal restricted identity fields):\n" + json.dumps(_safe_order_context(verified_order), ensure_ascii=False))
            context = "Retrieved company-document excerpts (may not fully answer the question):\n\n" + "\n\n".join(context_parts)

            if reason:
                summary = (
                    f"Issue: {question[:350]}; intent: {_intent_for(question)}; "
                    f"order ID: {current_order_id or 'not provided'}; "
                    f"checks: retrieved current company-document sections" + (", verified order data" if verified_order else "; no order details disclosed") + "; "
                    f"escalation reason: {reason}"
                )
                ticket = create_ticket(summary=summary, order_id=current_order_id, reason=reason, intent=_intent_for(question))
                if ticket.get("created") is True:
                    last_ticket_id = ticket["ticket_id"]
                    extra_handover_instruction = ""
                    if "more than ₹10,000" in reason:
                        extra_handover_instruction = (
                            " The item is a high-value damaged claim. Explicitly state that the "
                            "Priority Resolution team handles claims above ₹10,000 and that the "
                            "AI assistant cannot approve the claim, grounding this in [DWM-3] and [ESC-1]."
                        )
                    answer = generate_answer(client, question, STEP3_SYSTEM_PROMPT,
                        context + f"\n\nTOOL RESULT: create_ticket returned created=True, ticket_id={last_ticket_id}. Confirm the ticket honestly, but do not promise an outcome or timeline." + extra_handover_instruction)
                    answer += f"\n\nA human-support ticket was created successfully. Reference: {last_ticket_id}."
                else:
                    answer = generate_answer(client, question, STEP3_SYSTEM_PROMPT,
                        context + "\n\nTOOL RESULT: create_ticket did not return created=True. Do not claim a ticket was created or a handover completed.")
                    answer += "\n\nI could not confirm ticket creation, so I cannot say the handover is complete."
            else:
                answer = generate_answer(client, question, STEP3_SYSTEM_PROMPT, context)

            normalized_start = answer.strip().lower()
            greeting_starts = (
                "i’m zentromart", "i'm zentromart", "i am zentromart",
                "hello, i'm zentromart", "hello, i’m zentromart",
                "hello! i am zentromart", "hello! i'm zentromart",
                "hi, i'm zentromart", "hi, i’m zentromart",
            )
            if first_reply and not normalized_start.startswith(greeting_starts):
                answer = "I’m ZentroMart’s AI support assistant. " + answer
            first_reply = False
            print("Agent:", answer, "\n")
            _display_evidence(sources)
            history.extend([("Customer", question), ("Agent", answer)])
            # Keep the in-memory context bounded for predictable prompts.
            history = history[-20:]
        except Exception as exc:
            print(f"ERROR: {exc}\n")

def check() -> None:
    require_dataset()
    validator = DATASET / "validate_dataset.py"
    result = subprocess.run([sys.executable, str(validator)], cwd=str(DATASET), text=True)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="VoiceOps beginner starter pack")
    parser.add_argument("command", choices=["check", "step1", "step2", "step3"], help="What to run")
    args = parser.parse_args()
    try:
        if args.command == "check":
            check()
        elif args.command == "step1":
            step1()
        elif args.command == "step2":
            step2()
        elif args.command == "step3":
            step3()
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as exc:
        print(f"\nERROR: {exc}")
        raise SystemExit(1)

if __name__ == "__main__":
    main()
