from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


SENTENCE_SPLIT = re.compile(r"(?<=[.!?।|])\s+")
WORD_SPLIT = re.compile(r"\s+")


@dataclass
class Chunk:
    chunk_id: str
    parent_id: str
    text: str
    strategy: str
    lang: str
    query_id: int
    query_type: str
    is_selected: int
    source: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _cid(*parts: Any) -> str:
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _tokens(text: str) -> List[str]:
    return [t for t in WORD_SPLIT.split(text.strip()) if t]


def fixed_overlap_windows(
    text: str, size: int = 48, overlap: int = 16
) -> List[str]:
    toks = _tokens(text)
    if not toks:
        return []
    if len(toks) <= size:
        # Still emit a window variant for strategy diversity on short passages
        return [text.strip()] if len(toks) >= 8 else []
    step = max(1, size - overlap)
    windows: List[str] = []
    for i in range(0, len(toks), step):
        piece = " ".join(toks[i : i + size]).strip()
        if piece:
            windows.append(piece)
        if i + size >= len(toks):
            break
    return windows


def semantic_sentence_packs(text: str, target_tokens: int = 40) -> List[str]:
    sentences = [s.strip() for s in SENTENCE_SPLIT.split(text) if s.strip()]
    if len(sentences) <= 1:
        # Fall back: half-split long single sentence for semantic child
        toks = _tokens(text)
        if len(toks) < 12:
            return []
        mid = len(toks) // 2
        return [" ".join(toks[:mid]), " ".join(toks[mid:])]
    packs: List[str] = []
    buf: List[str] = []
    count = 0
    for s in sentences:
        n = len(_tokens(s))
        if buf and count + n > target_tokens:
            packs.append(" ".join(buf))
            buf = [s]
            count = n
        else:
            buf.append(s)
            count += n
    if buf:
        packs.append(" ".join(buf))
    return packs


def chunk_passage(
    *,
    parent_id: str,
    text: str,
    lang: str,
    query_id: int,
    query_type: str,
    is_selected: int,
    source: str,
) -> List[Chunk]:
    text = (text or "").strip()
    if not text:
        return []

    chunks: List[Chunk] = []
    meta_base = {
        "query_id": query_id,
        "query_type": query_type,
        "is_selected": is_selected,
        "source": source,
        "lang": lang,
    }

    chunks.append(
        Chunk(
            chunk_id=_cid(parent_id, "passage", lang),
            parent_id=parent_id,
            text=text,
            strategy="passage",
            lang=lang,
            query_id=query_id,
            query_type=query_type,
            is_selected=is_selected,
            source=source,
            metadata={**meta_base, "role": "parent"},
        )
    )

    for i, win in enumerate(fixed_overlap_windows(text)):
        chunks.append(
            Chunk(
                chunk_id=_cid(parent_id, "fixed", lang, i, win[:40]),
                parent_id=parent_id,
                text=win,
                strategy="fixed_overlap",
                lang=lang,
                query_id=query_id,
                query_type=query_type,
                is_selected=is_selected,
                source=source,
                metadata={**meta_base, "role": "child", "window": i},
            )
        )

    for i, pack in enumerate(semantic_sentence_packs(text)):
        chunks.append(
            Chunk(
                chunk_id=_cid(parent_id, "semantic", lang, i, pack[:40]),
                parent_id=parent_id,
                text=pack,
                strategy="semantic_pack",
                lang=lang,
                query_id=query_id,
                query_type=query_type,
                is_selected=is_selected,
                source=source,
                metadata={**meta_base, "role": "child", "pack": i},
            )
        )

    return chunks


def build_chunks_from_row(row: Dict[str, Any], passage_idx: int) -> List[Chunk]:
    query_id = int(row.get("query_id") or 0)
    query_type = str(row.get("query_type") or "unknown")
    passages = row.get("passages") or {}
    is_selected_list = passages.get("is_selected") or []
    eng = passages.get("English_passages") or []
    hi = passages.get("Translated_passages") or []

    is_sel = 0
    if passage_idx < len(is_selected_list):
        is_sel = int(is_selected_list[passage_idx] or 0)

    parent_id = f"q{query_id}_p{passage_idx}"
    out: List[Chunk] = []

    if passage_idx < len(eng) and eng[passage_idx]:
        out.extend(
            chunk_passage(
                parent_id=parent_id,
                text=str(eng[passage_idx]),
                lang="en",
                query_id=query_id,
                query_type=query_type,
                is_selected=is_sel,
                source="english",
            )
        )
    if passage_idx < len(hi) and hi[passage_idx]:
        out.extend(
            chunk_passage(
                parent_id=parent_id,
                text=str(hi[passage_idx]),
                lang="hi",
                query_id=query_id,
                query_type=query_type,
                is_selected=is_sel,
                source="translated",
            )
        )
    return out
