# HH Goa 2026 — Task 2: Voice-Enabled RAG

Voice → **Sarvam** speech-to-text → hybrid chunking/retrieval on
[`ai4bharat/MSMARCO-XI`](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI) →
grounded answer with harness + guardrails.

Hashtag: **`#RAGInGoa`**  
Submission form: https://forms.gle/MNvCjcv23Hn2Eeu58  
Deadline: 22 Aug 2026, 11:59 PM (no resubmissions)

## Pipeline

```
Mic / text
  → Sarvam STT (voice path)
  → Input guard + intent classify
  → Hybrid retrieve (dense + BM25 → RRF, parent–child expansion)
  → Extractive grounded generate (hot path; optional Groq)
  → Grounding verify / refuse
```

### Chunking (non-naive)

Offline indexing builds **multiple strategies** per passage:

1. **Passage-level** parent chunks  
2. **Fixed-size overlapping** windows (48 / 16 tokens)  
3. **Semantic sentence packs** (and half-splits for long single sentences)  
4. **Parent–child**: retrieve on children, expand parent for answering  
5. **Metadata-aware** boosts (`query_type`, `is_selected`, bilingual en/hi)

### Latency target (&lt;200ms)

- **Chunking/indexing is offline** (not paid from the online budget).
- **Online budget** = embed + retrieve + generate + guards through final JSON.
- **STT is measured separately** (shown in UI / response `timings.stt_ms`).
- Hot-path answers are **extractive** so P50/P70 stay under 200ms without a slow LLM round-trip.

## Latency (measured)

From `python -m backend.scripts.bench_latency --n 200` on the Hindi MSMARCO-XI subset index (TF-IDF+SVD + BM25 hybrid, extractive generate):

| Metric | Value |
|---|---|
| P50 | ~31 ms |
| P70 | ~36 ms |
| P100 | ~59 ms |
| Under 200ms | **100%** |

STT (Sarvam) is timed separately in `timings.stt_ms` and is **not** part of the online 200ms budget. Chunking/indexing is offline.

See `artifacts/latency_metrics.json`.

### Harness

Structured steps with retries (STT/LLM), Pydantic I/O, and per-step logs:
`transcribe → input_guard → classify_intent → retrieve → generate → verify_grounding`.

### Guardrails

- Off-topic / unsafe / empty transcript → **refuse**
- Low dense retrieval confidence → **refuse**
- Answer not grounded in citations → **refuse**

Refuse is a first-class UI state (try the “Try refuse” button).

## Quick start

```bash
# from hh-goa-rag/
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
copy .env.example .env   # or: cp .env.example .env
# add SARVAM_API_KEY for voice

# Build index (tries Hugging Face MSMARCO-XI hi; falls back to sample corpus)
python -m backend.scripts.build_index --max-queries 400
# offline-only:
# python -m backend.scripts.build_index --sample

python -m backend.scripts.bench_latency --n 120

# API
uvicorn backend.app.main:app --reload --port 8000

# UI (other terminal)
cd frontend
npm install
npm run dev
```

Open http://127.0.0.1:5173

### Production-ish single process

```bash
cd frontend && npm install && npm run build && cd ..
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

FastAPI serves `frontend/dist` when present.

## Environment

| Variable | Purpose |
|---|---|
| `SARVAM_API_KEY` | Required for `/api/voice-ask` |
| `GROQ_API_KEY` | Optional generative path (not used on latency hot path) |
| `MIN_RETRIEVAL_SCORE` | Dense cosine refuse floor |
| `INDEX_DIR` | Index output directory |

## API

- `GET /api/health`
- `POST /api/ask` `{ "query": "..." }`
- `POST /api/voice-ask` multipart `audio`
- `GET /api/metrics` → P50/P70/P100 summary

## Repo layout

```
backend/app/          FastAPI + harness + retrieval
backend/scripts/      build_index, bench_latency
frontend/             Vite React mic UI + metrics
data/index/           Built hybrid index
artifacts/            Latency JSON
docs/SUBMISSION.md    Video + social checklist
```

## Deploy notes

- Put API on Railway/Render/Fly; set env vars; run `build_index` in release phase or bake `data/index` into the image.
- Point `VITE_API_URL` at the public API when hosting UI separately, or ship the built UI from FastAPI.

See [docs/SUBMISSION.md](docs/SUBMISSION.md) for videos, `#RAGInGoa` posts, and the one-shot form checklist.
