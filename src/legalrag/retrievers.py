"""Retrievers: BM25 (keyword), dense (embeddings), hybrid (RRF), and reranking.

All share one interface:
    retriever.index(chunks)
    retriever.search(query, k, doc_filter=None) -> list[(Chunk, score)]
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from .types import Chunk

Hit = tuple[Chunk, float]

STOPWORDS = {"a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "in", "into", "is", "it", "its", "of", "on", "or", "that", "the", "their", "there", "these", "this", "to", "was", "were", "which", "with", "any", "such", "other", "than", "can"}
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class Retriever:
    def index(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self._by_doc: dict[str, np.ndarray] = {}
        groups: dict[str, list[int]] = defaultdict(list)
        for i, c in enumerate(chunks):
            groups[c.doc_id].append(i)
        self._by_doc = {d: np.array(ix) for d, ix in groups.items()}

    def _candidates(self, doc_filter: set[str] | None) -> np.ndarray | None:
        if not doc_filter:
            return None
        parts = [self._by_doc[d] for d in doc_filter if d in self._by_doc]
        return np.concatenate(parts) if parts else np.array([], dtype=int)

    @staticmethod
    def _top(scores: np.ndarray, ids: np.ndarray, k: int) -> list[tuple[int, float]]:
        k = min(k, len(ids))
        if k == 0:
            return []
        part = np.argpartition(-scores, k - 1)[:k]
        order = part[np.argsort(-scores[part])]
        return [(int(ids[i]), float(scores[i])) for i in order]

    def search(self, query: str, k: int = 10, doc_filter: set[str] | None = None) -> list[Hit]:
        raise NotImplementedError


class BM25Retriever(Retriever):
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b

    def index(self, chunks: list[Chunk]) -> None:
        from rank_bm25 import BM25Okapi

        super().index(chunks)
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks], k1=self.k1, b=self.b)

    def search(self, query, k=10, doc_filter=None):
        tokens = tokenize(query)
        cand = self._candidates(doc_filter)
        if cand is None:
            ids = np.arange(len(self.chunks))
            scores = self.bm25.get_scores(tokens)
        else:
            ids = cand
            scores = np.asarray(self.bm25.get_batch_scores(tokens, cand.tolist()))
        return [(self.chunks[i], s) for i, s in self._top(scores, ids, k)]


class DenseRetriever(Retriever):
    """Embedding search with a local sentence-transformers model.

    Many retrieval models expect instruction prefixes, e.g. BGE wants
    "Represent this sentence for searching relevant passages: " before
    queries and E5 wants "query: " / "passage: ". Getting these wrong
    quietly costs several points of recall, so they are config options.
    """

    def __init__(
        self,
        model: str = "BAAI/bge-small-en-v1.5",
        query_prefix: str = "",
        doc_prefix: str = "",
        batch_size: int = 64,
        cache_dir: str = ".cache/embeddings",
        device: str | None = None,
    ):
        self.model_name = model
        self.query_prefix, self.doc_prefix = query_prefix, doc_prefix
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir)
        self.device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def _cache_path(self, chunks: list[Chunk]) -> Path:
        h = hashlib.sha256(f"{self.model_name}|{self.doc_prefix}".encode())
        for c in chunks:
            h.update(c.chunk_id.encode())
            h.update(c.text.encode())
        return self.cache_dir / f"{h.hexdigest()[:16]}.npy"

    def index(self, chunks):
        super().index(chunks)
        path = self._cache_path(chunks)
        if path.exists():
            self.emb = np.load(path)
            return
        texts = [self.doc_prefix + c.text for c in chunks]
        self.emb = self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=True
        ).astype(np.float32)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.emb)

    def search(self, query, k=10, doc_filter=None):
        q = self.model.encode([self.query_prefix + query], normalize_embeddings=True)[0]
        cand = self._candidates(doc_filter)
        ids = np.arange(len(self.chunks)) if cand is None else cand
        scores = self.emb[ids] @ q
        return [(self.chunks[i], s) for i, s in self._top(scores, ids, k)]


class HybridRetriever(Retriever):
    """Reciprocal Rank Fusion: score = sum over retrievers of 1 / (rrf_k + rank).

    RRF only uses ranks, so it needs no tuning to put BM25 scores and cosine
    similarities on the same scale.
    """

    def __init__(self, retrievers: list[Retriever], rrf_k: int = 60, candidates: int = 100):
        self.retrievers, self.rrf_k, self.candidates = retrievers, rrf_k, candidates

    def index(self, chunks):
        super().index(chunks)
        for r in self.retrievers:
            r.index(chunks)

    def search(self, query, k=10, doc_filter=None):
        fused: dict[str, float] = defaultdict(float)
        by_id: dict[str, Chunk] = {}
        for r in self.retrievers:
            for rank, (chunk, _) in enumerate(r.search(query, self.candidates, doc_filter)):
                fused[chunk.chunk_id] += 1 / (self.rrf_k + rank + 1)
                by_id[chunk.chunk_id] = chunk
        best = sorted(fused.items(), key=lambda x: -x[1])[:k]
        return [(by_id[cid], score) for cid, score in best]


class RerankRetriever(Retriever):
    """Retrieve `candidates` with a first-stage retriever, re-score with a cross-encoder."""

    def __init__(self, base: Retriever, model: str = "BAAI/bge-reranker-base", candidates: int = 50,
                 device: str | None = None):
        self.base, self.model_name, self.candidates, self.device = base, model, candidates, device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def index(self, chunks):
        super().index(chunks)
        self.base.index(chunks)

    def search(self, query, k=10, doc_filter=None):
        hits = self.base.search(query, self.candidates, doc_filter)
        if not hits:
            return []
        scores = self.model.predict([(query, c.text) for c, _ in hits])
        order = np.argsort(-np.asarray(scores))[:k]
        return [(hits[i][0], float(scores[i])) for i in order]


def build_retriever(cfg: dict) -> Retriever:
    """Build a retriever from a config dict like {"type": "bm25", ...}."""
    cfg = dict(cfg)
    kind = cfg.pop("type")
    if kind == "bm25":
        return BM25Retriever(**cfg)
    if kind == "dense":
        return DenseRetriever(**cfg)
    if kind == "hybrid":
        parts = [build_retriever(c) for c in cfg.pop("retrievers")]
        return HybridRetriever(parts, **cfg)
    if kind == "rerank":
        base = build_retriever(cfg.pop("base"))
        return RerankRetriever(base, **cfg)
    raise ValueError(f"Unknown retriever type: {kind!r}")
