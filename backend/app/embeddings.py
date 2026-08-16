from __future__ import annotations

import threading
from typing import List, Optional

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .config import Settings, get_settings


class EmbeddingModel:
    """Multilingual-capable embeddings with TF-IDF+SVD fallback (no torch required)."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._model = None
        self._tfidf: Optional[TfidfVectorizer] = None
        self._svd: Optional[TruncatedSVD] = None
        self._lock = threading.Lock()
        self.backend = "tfidf"
        self.dim = 256
        self._fitted = False

    def _try_st(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.settings.embedding_model)
            self.backend = "sentence-transformers"
            self.dim = int(self._model.get_sentence_embedding_dimension())
        except Exception:
            self._model = None

    def fit_corpus(self, texts: List[str]) -> None:
        """Fit TF-IDF+SVD on the corpus (used at index build time)."""
        with self._lock:
            self._try_st()
            if self._model is not None:
                self._fitted = True
                return
            self._tfidf = TfidfVectorizer(
                max_features=20000,
                ngram_range=(1, 2),
                min_df=1,
                lowercase=True,
            )
            matrix = self._tfidf.fit_transform(texts)
            n_comp = min(self.dim, max(2, matrix.shape[1] - 1), matrix.shape[0] - 1)
            self._svd = TruncatedSVD(n_components=n_comp, random_state=42)
            self._svd.fit(matrix)
            self.dim = n_comp
            self.backend = "tfidf-svd"
            self._fitted = True

    def _tfidf_encode(self, texts: List[str]) -> np.ndarray:
        if self._tfidf is None or self._svd is None:
            # Unfitted emergency hash
            return self._hash_embed(texts)
        matrix = self._tfidf.transform(texts)
        emb = self._svd.transform(matrix).astype(np.float32)
        return normalize(emb)

    def _hash_embed(self, texts: List[str]) -> np.ndarray:
        mats = []
        for t in texts:
            v = np.zeros(self.dim, dtype=np.float32)
            tokens = t.lower().split()
            if not tokens:
                mats.append(v)
                continue
            for tok in tokens:
                h = hash(tok) % self.dim
                sign = 1.0 if (hash(tok + "#") % 2 == 0) else -1.0
                v[h] += sign
            n = np.linalg.norm(v) + 1e-9
            mats.append(v / n)
        return np.vstack(mats)

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self._model is not None:
            emb = self._model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            return np.asarray(emb, dtype=np.float32)
        if self._tfidf is not None:
            return self._tfidf_encode(texts)
        # Lazy: try ST once, else hash until fit_corpus is called
        self._try_st()
        if self._model is not None:
            return self.encode(texts, batch_size=batch_size)
        return self._hash_embed(texts)

    def save(self, path) -> None:
        import pickle
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "backend": self.backend,
            "dim": self.dim,
            "tfidf": self._tfidf,
            "svd": self._svd,
        }
        path.write_bytes(pickle.dumps(payload))

    def load(self, path) -> bool:
        import pickle
        from pathlib import Path

        path = Path(path)
        if not path.exists():
            return False
        payload = pickle.loads(path.read_bytes())
        self.backend = payload.get("backend", "tfidf-svd")
        self.dim = int(payload.get("dim", 256))
        self._tfidf = payload.get("tfidf")
        self._svd = payload.get("svd")
        self._fitted = self._tfidf is not None
        return True


_embedder: Optional[EmbeddingModel] = None


def get_embedder() -> EmbeddingModel:
    global _embedder
    if _embedder is None:
        _embedder = EmbeddingModel()
        # Load fitted vectorizer if present
        from .config import get_settings

        _embedder.load(get_settings().index_path / "embedder.pkl")
    return _embedder
