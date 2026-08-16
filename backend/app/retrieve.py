from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from .config import get_settings
from .embeddings import get_embedder
from .models import Citation


@dataclass
class Retrieved:
    citation: Citation
    dense_score: float
    sparse_score: float
    fused_score: float


class HybridIndex:
    def __init__(self, index_dir: Optional[Path] = None):
        settings = get_settings()
        self.index_dir = Path(index_dir or settings.index_path)
        self.chunks: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.parents: Dict[str, Dict[str, Any]] = {}
        self.bm25: Optional[BM25Okapi] = None
        self.tokenized: List[List[str]] = []
        self.meta: Dict[str, Any] = {}
        self._loaded = False

    @property
    def ready(self) -> bool:
        return self._loaded and len(self.chunks) > 0

    def load(self) -> None:
        chunks_path = self.index_dir / "chunks.jsonl"
        emb_path = self.index_dir / "embeddings.npy"
        parents_path = self.index_dir / "parents.json"
        meta_path = self.index_dir / "meta.json"
        if not chunks_path.exists() or not emb_path.exists():
            raise FileNotFoundError(
                f"Index missing in {self.index_dir}. Run: python -m backend.scripts.build_index"
            )
        chunks: List[Dict[str, Any]] = []
        with chunks_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
        self.chunks = chunks
        self.embeddings = np.load(emb_path).astype(np.float32)
        if parents_path.exists():
            self.parents = json.loads(parents_path.read_text(encoding="utf-8"))
        if meta_path.exists():
            self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.tokenized = [c["text"].lower().split() for c in self.chunks]
        self.bm25 = BM25Okapi(self.tokenized)
        self._loaded = True

    def _dense(self, qvec: np.ndarray, top_n: int) -> List[Tuple[int, float]]:
        assert self.embeddings is not None
        scores = self.embeddings @ qvec.reshape(-1)
        if top_n >= len(scores):
            idx = np.argsort(-scores)
        else:
            idx = np.argpartition(-scores, top_n)[:top_n]
            idx = idx[np.argsort(-scores[idx])]
        return [(int(i), float(scores[i])) for i in idx]

    def _sparse(self, query: str, top_n: int) -> List[Tuple[int, float]]:
        assert self.bm25 is not None
        toks = query.lower().split()
        scores = np.asarray(self.bm25.get_scores(toks), dtype=np.float32)
        if scores.max() > 0:
            scores = scores / (scores.max() + 1e-9)
        if top_n >= len(scores):
            idx = np.argsort(-scores)
        else:
            idx = np.argpartition(-scores, top_n)[:top_n]
            idx = idx[np.argsort(-scores[idx])]
        return [(int(i), float(scores[i])) for i in idx]

    @staticmethod
    def _rrf(
        dense: List[Tuple[int, float]],
        sparse: List[Tuple[int, float]],
        k: int = 60,
    ) -> Dict[int, float]:
        fused: Dict[int, float] = {}
        for rank, (i, _) in enumerate(dense):
            fused[i] = fused.get(i, 0.0) + 1.0 / (k + rank + 1)
        for rank, (i, _) in enumerate(sparse):
            fused[i] = fused.get(i, 0.0) + 1.0 / (k + rank + 1)
        return fused

    def retrieve(
        self,
        query: str,
        top_k: int = 6,
        query_type: Optional[str] = None,
    ) -> Tuple[List[Retrieved], Dict[str, Any], Dict[str, float]]:
        if not self.ready:
            self.load()
        t0 = time.perf_counter()
        embedder = get_embedder()
        qvec = embedder.encode([query])[0]
        embed_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        pool = max(40, top_k * 8)
        dense = self._dense(qvec, pool)
        sparse = self._sparse(query, pool)
        fused = self._rrf(dense, sparse)
        dense_map = dict(dense)
        sparse_map = dict(sparse)

        # Metadata-aware boost by query_type + stronger BM25 signal
        scored: List[Tuple[int, float]] = []
        for i, base in fused.items():
            boost = 0.15 * float(sparse_map.get(i, 0.0))
            if query_type:
                qt = (self.chunks[i].get("query_type") or "").lower()
                if qt and qt == query_type.lower():
                    boost += 0.05
                if self.chunks[i].get("is_selected") == 1:
                    boost += 0.02
            scored.append((i, base + boost))
        scored.sort(key=lambda x: -x[1])

        # Deduplicate by parent, prefer child hits then expand parent text
        seen_parents = set()
        results: List[Retrieved] = []
        strategy_counts: Dict[str, int] = {}
        for i, fused_score in scored:
            ch = self.chunks[i]
            parent_id = ch["parent_id"]
            if parent_id in seen_parents:
                continue
            seen_parents.add(parent_id)
            parent = self.parents.get(parent_id, {})
            text = parent.get("text") or ch["text"]
            strat = ch.get("strategy", "unknown")
            strategy_counts[strat] = strategy_counts.get(strat, 0) + 1
            cit = Citation(
                chunk_id=ch["chunk_id"],
                parent_id=parent_id,
                strategy=strat,
                lang=ch.get("lang", ""),
                score=float(dense_map.get(i, fused_score)),
                text=text[:1200],
                query_type=ch.get("query_type"),
            )
            results.append(
                Retrieved(
                    citation=cit,
                    dense_score=float(dense_map.get(i, 0.0)),
                    sparse_score=float(sparse_map.get(i, 0.0)),
                    fused_score=float(fused_score),
                )
            )
            if len(results) >= top_k:
                break

        retrieve_ms = (time.perf_counter() - t1) * 1000
        debug = {
            "strategies_hit": strategy_counts,
            "dense_top": [
                {"i": i, "score": s, "strategy": self.chunks[i]["strategy"]}
                for i, s in dense[:5]
            ],
            "sparse_top": [
                {"i": i, "score": s, "strategy": self.chunks[i]["strategy"]}
                for i, s in sparse[:5]
            ],
            "query_type_filter": query_type,
            "corpus_chunks": len(self.chunks),
            "embedding_backend": embedder.backend,
        }
        timings = {"embed_ms": embed_ms, "retrieve_ms": retrieve_ms}
        return results, debug, timings


_index: Optional[HybridIndex] = None


def get_index() -> HybridIndex:
    global _index
    if _index is None:
        _index = HybridIndex()
        try:
            _index.load()
        except FileNotFoundError:
            pass
    return _index
