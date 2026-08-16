from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class StepName(str, Enum):
    classify_intent = "classify_intent"
    input_guard = "input_guard"
    retrieve = "retrieve"
    generate = "generate"
    verify_grounding = "verify_grounding"
    transcribe = "transcribe"


class StepLog(BaseModel):
    name: StepName
    ok: bool
    latency_ms: float
    detail: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class Citation(BaseModel):
    chunk_id: str
    parent_id: str
    strategy: str
    lang: str
    score: float
    text: str
    query_type: Optional[str] = None


class Timings(BaseModel):
    embed_ms: float = 0.0
    retrieve_ms: float = 0.0
    generate_ms: float = 0.0
    guard_ms: float = 0.0
    total_ms: float = 0.0
    stt_ms: Optional[float] = None


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(6, ge=1, le=20)
    language_hint: Optional[str] = None


class AskResponse(BaseModel):
    query: str
    transcript: Optional[str] = None
    answered: bool
    answer: Optional[str] = None
    refuse_reason: Optional[str] = None
    citations: List[Citation] = Field(default_factory=list)
    timings: Timings
    steps: List[StepLog] = Field(default_factory=list)
    retrieval_debug: Dict[str, Any] = Field(default_factory=dict)
    mode: str = "extractive"


class MetricsSummary(BaseModel):
    n: int
    p50_ms: float
    p70_ms: float
    p100_ms: float
    mean_ms: float
    under_200ms_pct: float
    breakdown: Dict[str, float]
    source: str
