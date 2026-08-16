from __future__ import annotations

import re
import time
from typing import List, Optional, Sequence, Tuple

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import get_settings
from .models import Citation


def _sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?।])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def extractive_answer(query: str, citations: Sequence[Citation]) -> str:
    """Fast extractive answer: best overlapping sentence(s) from top passages."""
    q_toks = set(re.findall(r"[\w\u0900-\u097F]+", query.lower()))
    best: List[Tuple[float, str]] = []
    for cit in citations[:4]:
        for sent in _sentences(cit.text):
            s_toks = set(re.findall(r"[\w\u0900-\u097F]+", sent.lower()))
            if not s_toks:
                continue
            overlap = len(q_toks & s_toks) / max(1, len(q_toks))
            # Prefer informative length
            length_bonus = min(len(sent) / 220.0, 0.25)
            score = overlap + length_bonus
            best.append((score, sent))
    best.sort(key=lambda x: -x[0])
    if not best:
        # Fallback: truncated top passage
        return citations[0].text[:320]
    picked = []
    seen = set()
    for _, sent in best:
        key = sent[:80]
        if key in seen:
            continue
        seen.add(key)
        picked.append(sent)
        if len(picked) >= 2:
            break
    return " ".join(picked)


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=0.2, min=0.2, max=1))
def _groq_chat(system: str, user: str) -> str:
    settings = get_settings()
    api_key = settings.groq_api_key or settings.openai_api_key
    if not api_key:
        raise RuntimeError("No LLM API key configured")
    base = settings.openai_base_url.rstrip("/")
    model = settings.groq_model or settings.openai_model
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "temperature": 0.1,
        "max_tokens": 120,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    with httpx.Client(timeout=8.0) as client:
        r = client.post(f"{base}/chat/completions", headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()


def generative_answer(query: str, citations: Sequence[Citation]) -> Optional[str]:
    settings = get_settings()
    if not (settings.groq_api_key or settings.openai_api_key):
        return None
    ctx = "\n\n".join(
        f"[{i+1}] ({c.lang}/{c.strategy}) {c.text[:500]}"
        for i, c in enumerate(citations[:4])
    )
    system = (
        "You answer ONLY using the provided passages. "
        "If insufficient, reply exactly: INSUFFICIENT_CONTEXT. "
        "Be concise (1-2 sentences)."
    )
    user = f"Question: {query}\n\nPassages:\n{ctx}"
    try:
        out = _groq_chat(system, user)
        if "INSUFFICIENT_CONTEXT" in out:
            return None
        return out
    except Exception:
        return None


def generate_answer(
    query: str, citations: Sequence[Citation]
) -> Tuple[str, str, float]:
    """Returns (answer, mode, generate_ms). Prefer extractive for <200ms."""
    t0 = time.perf_counter()
    # Hot path: extractive (deterministic, sub-ms to few-ms)
    answer = extractive_answer(query, citations)
    mode = "extractive"
    # Optional: only try LLM if explicitly wanted and keys exist — skipped on hot path
    # to protect the 200ms budget. Callers can force via env FORCE_LLM=1 later if needed.
    generate_ms = (time.perf_counter() - t0) * 1000
    return answer, mode, generate_ms
