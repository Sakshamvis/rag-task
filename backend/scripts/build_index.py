from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Allow running as module from repo root
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.chunking import Chunk, build_chunks_from_row
from backend.app.config import get_settings
from backend.app.embeddings import EmbeddingModel


SAMPLE_ROWS: List[Dict[str, Any]] = [
    {
        "query_id": 1,
        "query_type": "description",
        "query": "What is photosynthesis?",
        "Answer": "Photosynthesis is how plants convert light into chemical energy.",
        "passages": {
            "is_selected": [1, 0],
            "English_passages": [
                "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods from carbon dioxide and water. It generally involves the green pigment chlorophyll and generates oxygen as a byproduct.",
                "Respiration is a different metabolic process that releases energy from glucose.",
            ],
            "Translated_passages": [
                "प्रकाश संश्लेषण वह प्रक्रिया है जिसमें हरे पौधे सूर्य के प्रकाश की सहायता से कार्बन डाइऑक्साइड और जल से भोजन बनाते हैं। इसमें क्लोरोफिल महत्वपूर्ण भूमिका निभाता है और ऑक्सीजन उत्पन्न होती है।",
                "श्वसन एक अलग प्रक्रिया है जो ग्लूकोज से ऊर्जा मुक्त करती है।",
            ],
        },
    },
    {
        "query_id": 2,
        "query_type": "numeric",
        "query": "How tall is Mount Everest?",
        "Answer": "Mount Everest is about 8,849 meters tall.",
        "passages": {
            "is_selected": [1],
            "English_passages": [
                "Mount Everest is Earth's highest mountain above sea level, located in the Himalayas. Its elevation is commonly cited as 8,848.86 meters (29,031.7 feet)."
            ],
            "Translated_passages": [
                "माउंट एवरेस्ट पृथ्वी की सबसे ऊँची चोटी है। इसकी ऊँचाई लगभग 8,848.86 मीटर मानी जाती है।"
            ],
        },
    },
    {
        "query_id": 3,
        "query_type": "entity",
        "query": "Who invented the telephone?",
        "Answer": "Alexander Graham Bell is credited with inventing the telephone.",
        "passages": {
            "is_selected": [1],
            "English_passages": [
                "Alexander Graham Bell is widely credited with inventing the first practical telephone and was awarded the first US patent for the telephone in 1876."
            ],
            "Translated_passages": [
                "अलेक्जेंडर ग्राहम बेल को व्यावहारिक टेलीफोन का आविष्कारक माना जाता है। उन्हें 1876 में टेलीफोन का पहला अमेरिकी पेटेंट मिला।"
            ],
        },
    },
    {
        "query_id": 4,
        "query_type": "location",
        "query": "Where is the Taj Mahal located?",
        "Answer": "The Taj Mahal is in Agra, India.",
        "passages": {
            "is_selected": [1],
            "English_passages": [
                "The Taj Mahal is an ivory-white marble mausoleum on the right bank of the river Yamuna in Agra, Uttar Pradesh, India. It was commissioned in 1631 by Mughal emperor Shah Jahan."
            ],
            "Translated_passages": [
                "ताज महल आगरा, उत्तर प्रदेश, भारत में यमुना नदी के किनारे स्थित एक संगमरमर का मकबरा है। इसे शाहजहाँ ने बनवाया था।"
            ],
        },
    },
    {
        "query_id": 5,
        "query_type": "temporal",
        "query": "When did World War II end?",
        "Answer": "World War II ended in 1945.",
        "passages": {
            "is_selected": [1],
            "English_passages": [
                "World War II ended in 1945. Nazi Germany surrendered in May 1945 and Japan surrendered in September 1945 after the atomic bombings of Hiroshima and Nagasaki."
            ],
            "Translated_passages": [
                "द्वितीय विश्व युद्ध 1945 में समाप्त हुआ। जर्मनी ने मई 1945 में और जापान ने सितंबर 1945 में आत्मसमर्पण किया।"
            ],
        },
    },
    {
        "query_id": 6,
        "query_type": "description",
        "query": "What causes tides?",
        "Answer": "Tides are mainly caused by the Moon's gravitational pull.",
        "passages": {
            "is_selected": [1],
            "English_passages": [
                "Ocean tides are primarily caused by the gravitational interaction between Earth and the Moon, with the Sun also contributing. The Moon's gravity pulls water toward it creating tidal bulges."
            ],
            "Translated_passages": [
                "समुद्री ज्वार-भाटा मुख्य रूप से पृथ्वी और चंद्रमा के गुरुत्वाकर्षण के कारण होते हैं। सूर्य भी इसमें योगदान देता है।"
            ],
        },
    },
    {
        "query_id": 7,
        "query_type": "description",
        "query": "What is machine learning?",
        "Answer": "Machine learning is a field of AI where systems learn patterns from data.",
        "passages": {
            "is_selected": [1],
            "English_passages": [
                "Machine learning is a subset of artificial intelligence that enables computers to learn patterns from data and make predictions or decisions without being explicitly programmed for every case."
            ],
            "Translated_passages": [
                "मशीन लर्निंग कृत्रिम बुद्धिमत्ता की एक शाखा है जिसमें कंप्यूटर डेटा से पैटर्न सीखकर भविष्यवाणियाँ या निर्णय लेते हैं।"
            ],
        },
    },
    {
        "query_id": 8,
        "query_type": "entity",
        "query": "Who wrote the Ramayana?",
        "Answer": "The Ramayana is traditionally attributed to the sage Valmiki.",
        "passages": {
            "is_selected": [1],
            "English_passages": [
                "The Ramayana is an ancient Indian epic traditionally attributed to the sage Valmiki. It narrates the life of Rama, an incarnation of Vishnu."
            ],
            "Translated_passages": [
                "रामायण एक प्राचीन भारतीय महाकाव्य है जिसे परंपरागत रूप से महर्षि वाल्मीकि द्वारा रचित माना जाता है।"
            ],
        },
    },
]


def load_msmarco(max_queries: int, use_sample: bool) -> List[Dict[str, Any]]:
    if use_sample:
        print("Using built-in sample corpus (offline).")
        return SAMPLE_ROWS
    try:
        from huggingface_hub import hf_hub_download
        import duckdb

        print("Downloading/loading train/hintrain.parquet via DuckDB …")
        path = hf_hub_download(
            repo_id="ai4bharat/MSMARCO-XI",
            repo_type="dataset",
            filename="train/hintrain.parquet",
        )
        con = duckdb.connect()
        df = con.execute(
            """
            SELECT query_id, query_type, query, Answer, passages, Eng_Query, Eng_Answer
            FROM read_parquet(?)
            LIMIT ?
            """,
            [path, int(max_queries)],
        ).fetchdf()
        rows: List[Dict[str, Any]] = []
        for _, rec in df.iterrows():
            passages = rec["passages"]
            if isinstance(passages, str):
                passages = json.loads(passages)
            rows.append(
                {
                    "query_id": int(rec["query_id"]),
                    "query_type": str(rec["query_type"] or "unknown"),
                    "query": str(rec["query"] or ""),
                    "Answer": str(rec["Answer"] or ""),
                    "Eng_Query": str(rec.get("Eng_Query") or ""),
                    "Eng_Answer": str(rec.get("Eng_Answer") or ""),
                    "passages": passages
                    if isinstance(passages, dict)
                    else {
                        "English_passages": [],
                        "Translated_passages": [],
                        "is_selected": [],
                    },
                }
            )
        print(f"Loaded {len(rows)} Hindi query rows from Hugging Face.")
        return rows
    except Exception as e:
        print(f"HF load failed ({e}); falling back to sample corpus.")
        return SAMPLE_ROWS


def expand_rows(rows: List[Dict[str, Any]], prefer_selected: bool = True) -> List[Chunk]:
    chunks: List[Chunk] = []
    for row in rows:
        passages = row.get("passages") or {}
        n = max(
            len(passages.get("English_passages") or []),
            len(passages.get("Translated_passages") or []),
            len(passages.get("is_selected") or []),
        )
        indices = list(range(n))
        if prefer_selected:
            selected = [
                i
                for i, flag in enumerate(passages.get("is_selected") or [])
                if int(flag or 0) == 1
            ]
            if selected:
                indices = selected
            # keep a couple non-selected distractors when available
            others = [i for i in range(n) if i not in indices][:1]
            indices = list(dict.fromkeys(indices + others))
        for i in indices:
            chunks.extend(build_chunks_from_row(row, i))
    return chunks


def write_index(chunks: List[Chunk], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    embedder = EmbeddingModel()
    texts = [c.text for c in chunks]
    print(f"Fitting/encoding {len(texts)} chunks …")
    embedder.fit_corpus(texts)
    print(f"Embedding with backend={embedder.backend} …")
    embs = embedder.encode(texts, batch_size=64)
    embedder.save(out_dir / "embedder.pkl")

    parents: Dict[str, Any] = {}
    for c in chunks:
        if c.strategy == "passage":
            parents[c.parent_id] = {
                "text": c.text,
                "lang": c.lang,
                "query_id": c.query_id,
                "query_type": c.query_type,
            }

    with (out_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    np.save(out_dir / "embeddings.npy", embs)
    (out_dir / "parents.json").write_text(
        json.dumps(parents, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    strategies: Dict[str, int] = {}
    for c in chunks:
        strategies[c.strategy] = strategies.get(c.strategy, 0) + 1
    meta = {
        "n_chunks": len(chunks),
        "n_parents": len(parents),
        "strategies": strategies,
        "embedding_backend": embedder.backend,
        "embedding_dim": int(embs.shape[1]) if len(embs) else 0,
        "chunking": [
            "passage (parent)",
            "fixed_overlap windows",
            "semantic sentence packs",
            "parent-child expansion at retrieve",
            "metadata-aware query_type boost",
        ],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"Wrote index -> {out_dir}")


def save_eval_queries(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    queries = []
    for r in rows:
        q = (r.get("Eng_Query") or r.get("query") or "").strip()
        hq = (r.get("query") or "").strip()
        if q:
            queries.append(q)
        if hq and hq != q:
            queries.append(hq)
    path = out_dir / "eval_queries.json"
    path.write_text(json.dumps(queries[:500], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(queries[:500])} eval queries -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build hybrid MSMARCO-XI index")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Force offline sample corpus",
    )
    parser.add_argument(
        "--hf",
        action="store_true",
        help="Prefer Hugging Face download",
    )
    args = parser.parse_args()
    settings = get_settings()
    max_q = args.max_queries or settings.max_corpus_queries
    rows = load_msmarco(max_q, use_sample=bool(args.sample))
    chunks = expand_rows(rows)
    if not chunks:
        raise SystemExit("No chunks produced")
    out = Path(settings.index_dir)
    write_index(chunks, out)
    save_eval_queries(rows, out)


if __name__ == "__main__":
    main()
