from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

from .config import get_settings
from .models import Citation


UNSAFE_PATTERNS = [
    r"\bhow to (make|build|create) (a )?(bomb|weapon|explosive)\b",
    r"\bchild\s*porn\b",
    r"\bsuicide\b.*\b(method|how)\b",
    r"\bhack(?:ing)?\s+(into|someone)\b",
]

OFFTOPIC_PATTERNS = [
    r"\bwrite (me )?(a )?(poem|song|rap)\b",
    r"\btell me a joke\b",
    r"\bwho (are|r) you\b",
    r"\bact as\b",
    r"\bignore (all|previous) instructions\b",
]


@dataclass
class GuardResult:
    ok: bool
    reason: Optional[str] = None
    query_type_guess: Optional[str] = None


def classify_intent(query: str) -> str:
    q = query.lower()
    if any(w in q for w in ("how much", "price", "cost", "₹", "rs", "dollar")):
        return "numeric"
    if any(w in q for w in ("where", "location", "city", "place")):
        return "location"
    if any(w in q for w in ("who", "person", "ceo", "founder")):
        return "entity"
    if any(w in q for w in ("when", "date", "year", "month")):
        return "temporal"
    if any(w in q for w in ("how", "why", "what", "क्या", "कैसे", "कौन", "कब", "कहाँ")):
        return "description"
    return "unknown"


def input_guard(query: str) -> GuardResult:
    q = (query or "").strip()
    if len(q) < 2:
        return GuardResult(False, "Empty or too-short query after transcription.")
    low = q.lower()
    for pat in UNSAFE_PATTERNS:
        if re.search(pat, low):
            return GuardResult(False, "Unsafe or disallowed request.")
    for pat in OFFTOPIC_PATTERNS:
        if re.search(pat, low):
            return GuardResult(
                False,
                "Off-topic for this factual MSMARCO-XI knowledge base.",
            )
    return GuardResult(True, query_type_guess=classify_intent(q))


def retrieval_guard(citations: Sequence[Citation], top_score: float) -> GuardResult:
    settings = get_settings()
    if not citations:
        return GuardResult(False, "No relevant passages retrieved.")
    if top_score < settings.min_retrieval_score:
        return GuardResult(
            False,
            f"Low retrieval confidence ({top_score:.3f} < {settings.min_retrieval_score}).",
        )
    return GuardResult(True)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[\w\u0900-\u097F]+", text.lower())


def grounding_overlap(answer: str, citations: Sequence[Citation]) -> float:
    ans_toks = set(_tokenize(answer))
    if not ans_toks:
        return 0.0
    ctx = " ".join(c.text for c in citations)
    ctx_toks = set(_tokenize(ctx))
    if not ctx_toks:
        return 0.0
    return len(ans_toks & ctx_toks) / max(1, len(ans_toks))


def output_guard(answer: str, citations: Sequence[Citation]) -> GuardResult:
    settings = get_settings()
    if not answer or not answer.strip():
        return GuardResult(False, "Generator returned an empty answer.")
    overlap = grounding_overlap(answer, citations)
    if overlap < settings.grounding_overlap_min:
        return GuardResult(
            False,
            f"Answer not grounded in retrieved context (overlap={overlap:.3f}).",
        )
    return GuardResult(True)
