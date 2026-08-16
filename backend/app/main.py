"""HH Goa 2026 Task 2 — Voice-enabled RAG API."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import ROOT, get_settings
from .harness import run_ask
from .metrics import load_metrics
from .models import AskRequest, AskResponse, MetricsSummary, StepLog, StepName
from .retrieve import get_index
from .stt import STTError, transcribe_audio

settings = get_settings()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    idx = get_index()
    if not idx.ready:
        try:
            idx.load()
        except FileNotFoundError:
            pass
    yield


app = FastAPI(
    title="HH Goa RAG",
    description="Voice-enabled RAG on ai4bharat/MSMARCO-XI",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list + ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    idx = get_index()
    return {
        "ok": True,
        "index_ready": idx.ready,
        "chunks": len(idx.chunks) if idx.ready else 0,
        "meta": idx.meta if idx.ready else {},
        "sarvam_configured": bool(settings.sarvam_api_key),
        "llm_configured": bool(settings.groq_api_key or settings.openai_api_key),
    }


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    return run_ask(req.query, top_k=req.top_k)


@app.post("/api/voice-ask", response_model=AskResponse)
async def voice_ask(
    audio: UploadFile = File(...),
    top_k: int = Form(6),
) -> AskResponse:
    data = await audio.read()
    if not data:
        raise HTTPException(400, "Empty audio upload")
    mime = audio.content_type or "audio/webm"
    try:
        transcript, stt_ms = transcribe_audio(
            data, filename=audio.filename or "audio.webm", mime=mime
        )
    except STTError as e:
        raise HTTPException(502, str(e)) from e

    resp = run_ask(transcript, top_k=top_k, transcript=transcript, stt_ms=stt_ms)
    # prepend STT step
    resp.steps = [
        StepLog(
            name=StepName.transcribe,
            ok=True,
            latency_ms=stt_ms,
            detail={"chars": len(transcript)},
        ),
        *resp.steps,
    ]
    return resp


@app.get("/api/metrics", response_model=MetricsSummary)
def metrics() -> MetricsSummary:
    raw = load_metrics()
    if not raw:
        raise HTTPException(
            404,
            "No metrics yet. Run: python -m backend.scripts.bench_latency",
        )
    return MetricsSummary(**raw["summary"])


@app.get("/api/metrics/raw")
def metrics_raw():
    raw = load_metrics()
    if not raw:
        raise HTTPException(404, "No metrics yet")
    return raw


frontend_dist = ROOT / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/assets", StaticFiles(directory=frontend_dist / "assets"), name="assets")

    @app.get("/")
    def spa_index():
        return FileResponse(frontend_dist / "index.html")

    @app.get("/{full_path:path}")
    def spa_fallback(full_path: str):
        candidate = frontend_dist / full_path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(frontend_dist / "index.html")
