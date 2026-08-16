from __future__ import annotations

import time
from typing import Optional

from .generate import generate_answer
from .guardrails import input_guard, output_guard, retrieval_guard
from .models import AskResponse, StepLog, StepName, Timings
from .retrieve import get_index


def run_ask(
    query: str,
    top_k: int = 6,
    transcript: Optional[str] = None,
    stt_ms: Optional[float] = None,
) -> AskResponse:
    """Structured RAG harness: classify → guard → retrieve → generate → verify."""
    t_all = time.perf_counter()
    steps: list[StepLog] = []
    q = (transcript or query).strip()

    t0 = time.perf_counter()
    g_in = input_guard(q)
    guard_ms = (time.perf_counter() - t0) * 1000
    steps.append(
        StepLog(
            name=StepName.input_guard,
            ok=g_in.ok,
            latency_ms=guard_ms,
            detail={"query_type_guess": g_in.query_type_guess},
            error=None if g_in.ok else g_in.reason,
        )
    )
    steps.append(
        StepLog(
            name=StepName.classify_intent,
            ok=True,
            latency_ms=0.0,
            detail={"query_type": g_in.query_type_guess},
        )
    )
    if not g_in.ok:
        total = (time.perf_counter() - t_all) * 1000
        return AskResponse(
            query=q,
            transcript=transcript,
            answered=False,
            refuse_reason=g_in.reason,
            timings=Timings(guard_ms=guard_ms, total_ms=total, stt_ms=stt_ms),
            steps=steps,
        )

    index = get_index()
    if not index.ready:
        total = (time.perf_counter() - t_all) * 1000
        steps.append(
            StepLog(
                name=StepName.retrieve,
                ok=False,
                latency_ms=0.0,
                error="Index not built. Run build_index.",
            )
        )
        return AskResponse(
            query=q,
            transcript=transcript,
            answered=False,
            refuse_reason="Knowledge index unavailable.",
            timings=Timings(guard_ms=guard_ms, total_ms=total, stt_ms=stt_ms),
            steps=steps,
        )

    results, debug, r_timings = index.retrieve(
        q, top_k=top_k, query_type=g_in.query_type_guess
    )
    citations = [r.citation for r in results]
    # Use dense cosine for confidence; RRF ranks are tiny absolute values.
    top_score = results[0].dense_score if results else 0.0
    steps.append(
        StepLog(
            name=StepName.retrieve,
            ok=bool(results),
            latency_ms=r_timings["embed_ms"] + r_timings["retrieve_ms"],
            detail={
                "top_dense_score": top_score,
                "top_fused_score": results[0].fused_score if results else 0.0,
                "n": len(results),
                "strategies": debug.get("strategies_hit"),
            },
        )
    )

    g_ret = retrieval_guard(citations, top_score)
    if not g_ret.ok:
        total = (time.perf_counter() - t_all) * 1000
        return AskResponse(
            query=q,
            transcript=transcript,
            answered=False,
            refuse_reason=g_ret.reason,
            citations=citations,
            timings=Timings(
                embed_ms=r_timings["embed_ms"],
                retrieve_ms=r_timings["retrieve_ms"],
                guard_ms=guard_ms,
                total_ms=total,
                stt_ms=stt_ms,
            ),
            steps=steps,
            retrieval_debug=debug,
        )

    answer, mode, gen_ms = generate_answer(q, citations)
    steps.append(
        StepLog(
            name=StepName.generate,
            ok=bool(answer),
            latency_ms=gen_ms,
            detail={"mode": mode},
        )
    )

    t_g = time.perf_counter()
    g_out = output_guard(answer, citations)
    out_guard_ms = (time.perf_counter() - t_g) * 1000
    guard_ms += out_guard_ms
    steps.append(
        StepLog(
            name=StepName.verify_grounding,
            ok=g_out.ok,
            latency_ms=out_guard_ms,
            error=None if g_out.ok else g_out.reason,
        )
    )
    total = (time.perf_counter() - t_all) * 1000
    if not g_out.ok:
        return AskResponse(
            query=q,
            transcript=transcript,
            answered=False,
            refuse_reason=g_out.reason,
            citations=citations,
            timings=Timings(
                embed_ms=r_timings["embed_ms"],
                retrieve_ms=r_timings["retrieve_ms"],
                generate_ms=gen_ms,
                guard_ms=guard_ms,
                total_ms=total,
                stt_ms=stt_ms,
            ),
            steps=steps,
            retrieval_debug=debug,
            mode=mode,
        )

    return AskResponse(
        query=q,
        transcript=transcript,
        answered=True,
        answer=answer,
        citations=citations,
        timings=Timings(
            embed_ms=r_timings["embed_ms"],
            retrieve_ms=r_timings["retrieve_ms"],
            generate_ms=gen_ms,
            guard_ms=guard_ms,
            total_ms=total,
            stt_ms=stt_ms,
        ),
        steps=steps,
        retrieval_debug=debug,
        mode=mode,
    )
