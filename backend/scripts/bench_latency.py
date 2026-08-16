from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.harness import run_ask
from backend.app.metrics import save_metrics, summarize_latencies
from backend.app.retrieve import get_index
from backend.scripts.build_index import SAMPLE_ROWS


def default_queries() -> List[str]:
    qs = [r["query"] for r in SAMPLE_ROWS]
    # Hindi variants
    qs += [
        "प्रकाश संश्लेषण क्या है?",
        "ताज महल कहाँ है?",
        "माउंट एवरेस्ट कितना ऊँचा है?",
        "रामायण किसने लिखी?",
    ]
    # Negatives / refuse cases
    qs += [
        "tell me a joke",
        "write me a poem about Goa",
        "asdf qwer zxcv",
    ]
    return qs


def load_queries(n: int) -> List[str]:
    eval_path = ROOT / "data" / "index" / "eval_queries.json"
    qs: List[str] = []
    if eval_path.exists():
        qs = json.loads(eval_path.read_text(encoding="utf-8"))
    qs = list(qs) + default_queries()
    out: List[str] = []
    i = 0
    while len(out) < n:
        out.append(qs[i % len(qs)])
        i += 1
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark RAG online latency")
    parser.add_argument("--n", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args()

    idx = get_index()
    if not idx.ready:
        idx.load()

    queries = load_queries(args.n + args.warmup)
    print(f"Warmup ({args.warmup}) …")
    for q in queries[: args.warmup]:
        run_ask(q, top_k=args.top_k)

    totals: List[float] = []
    embeds: List[float] = []
    retrieves: List[float] = []
    generates: List[float] = []
    rows = []

    print(f"Benchmarking {args.n} queries …")
    for q in queries[args.warmup :]:
        t0 = time.perf_counter()
        resp = run_ask(q, top_k=args.top_k)
        wall = (time.perf_counter() - t0) * 1000
        totals.append(resp.timings.total_ms)
        embeds.append(resp.timings.embed_ms)
        retrieves.append(resp.timings.retrieve_ms)
        generates.append(resp.timings.generate_ms)
        rows.append(
            {
                "query": q,
                "answered": resp.answered,
                "total_ms": resp.timings.total_ms,
                "wall_ms": wall,
                "embed_ms": resp.timings.embed_ms,
                "retrieve_ms": resp.timings.retrieve_ms,
                "generate_ms": resp.timings.generate_ms,
                "refuse_reason": resp.refuse_reason,
            }
        )

    summary = summarize_latencies(
        totals, embeds, retrieves, generates, source=f"bench_n={args.n}"
    )
    payload = {
        "summary": summary.model_dump(),
        "runs": rows,
        "notes": {
            "definition": (
                "Online latency = embed + retrieve + generate + guards "
                "(chunking/indexing is offline). STT measured separately."
            ),
            "target": "total_ms P50/P70 under 200ms",
        },
    }
    path = save_metrics(payload)
    print(json.dumps(payload["summary"], indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
