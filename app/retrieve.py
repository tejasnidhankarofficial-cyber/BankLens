"""Dense, BM25 and hybrid (RRF) retrieval with optional bank/year metadata filter."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.chunking import Chunk
from app.config import Config
from app.embeddings import Embedder
from app.index import Index

BANK_ALIASES = {
    "JPM": ["jpm", "jpmorgan", "jp morgan", "chase"],
    "BAC": ["bofa", "bank of america", "bac"],
    "C": ["citi", "citigroup", "citibank"],
    "WFC": ["wells fargo", "wells", "wfc"],
}
_YEAR = re.compile(r"\b(20\d\d)\b")
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass
class Filter:
    tickers: list[str]
    years: list[int]

    @property
    def empty(self) -> bool:
        return not self.tickers and not self.years

    def matches(self, c: Chunk) -> bool:
        return (not self.tickers or c.ticker in self.tickers) and (
            not self.years or c.fiscal_year in self.years
        )

    def where(self) -> dict | None:
        conds = []
        if self.tickers:
            conds.append({"ticker": {"$in": self.tickers}})
        if self.years:
            conds.append({"fiscal_year": {"$in": self.years}})
        if not conds:
            return None
        return conds[0] if len(conds) == 1 else {"$and": conds}


def detect_filter(query: str) -> Filter:
    q = query.lower()
    tickers = [
        t for t, aliases in BANK_ALIASES.items() if any(re.search(rf"\b{re.escape(a)}\b", q) for a in aliases)
    ]
    years = sorted({int(y) for y in _YEAR.findall(query)})
    return Filter(tickers, years)


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank: int


class Retriever:
    def __init__(self, cfg: Config, index: Index, embedder: Embedder):
        self.cfg = cfg
        self.index = index
        self.embedder = embedder
        self._bm25_cache: dict = {}

    # -- filter ---------------------------------------------------------
    def resolve_filter(self, query: str) -> Filter:
        if not self.cfg.retrieval.metadata_filter:
            return Filter([], [])
        f = detect_filter(query)
        if f.empty:
            return f
        if not any(f.matches(c) for c in self.index.chunks):
            # e.g. the year is not in the corpus: keep the bank filter only.
            # (A FY2025 10-K also contains FY2024 comparatives.)
            f = Filter(f.tickers, [])
            if not any(f.matches(c) for c in self.index.chunks):
                f = Filter([], [])
        return f

    # -- retrievers -----------------------------------------------------
    def dense(self, query: str, f: Filter, k: int) -> list[Hit]:
        qv = self.embedder.embed([query])[0].tolist()
        n = min(k, self.index.collection.count())
        if n == 0:
            return []
        res = self.index.collection.query(query_embeddings=[qv], n_results=n, where=f.where())
        ids, dists = res["ids"][0], res["distances"][0]
        return [Hit(self.index.by_id[i], 1.0 - d, r) for r, (i, d) in enumerate(zip(ids, dists), start=1)]

    def _bm25(self, f: Filter):
        key = (tuple(f.tickers), tuple(f.years))
        if key not in self._bm25_cache:
            from rank_bm25 import BM25Okapi

            subset = [c for c in self.index.chunks if f.matches(c)]
            corpus = [tokenize(c.embed_text) or ["_"] for c in subset]
            self._bm25_cache[key] = (BM25Okapi(corpus) if corpus else None, subset)
        return self._bm25_cache[key]

    def bm25(self, query: str, f: Filter, k: int) -> list[Hit]:
        bm, subset = self._bm25(f)
        if bm is None:
            return []
        scores = bm.get_scores(tokenize(query))
        order = sorted(range(len(subset)), key=lambda i: -scores[i])[:k]
        return [Hit(subset[i], float(scores[i]), r) for r, i in enumerate(order, start=1)]

    def hybrid(self, query: str, f: Filter, k: int) -> list[Hit]:
        rrf_k = self.cfg.retrieval.rrf_k
        ck = self.cfg.retrieval.candidate_k
        fused: dict[str, float] = {}
        for hits in (self.dense(query, f, ck), self.bm25(query, f, ck)):
            for h in hits:
                fused[h.chunk.chunk_id] = fused.get(h.chunk.chunk_id, 0.0) + 1.0 / (rrf_k + h.rank)
        ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        return [Hit(self.index.by_id[cid], s, r) for r, (cid, s) in enumerate(ranked, start=1)]

    def retrieve(self, query: str, k: int) -> tuple[list[Hit], Filter]:
        """Return up to ``k`` candidate hits plus the filter that was applied."""
        f = self.resolve_filter(query)
        mode = self.cfg.retrieval.mode
        if mode == "dense":
            return self.dense(query, f, k), f
        if mode == "bm25":
            return self.bm25(query, f, k), f
        return self.hybrid(query, f, k), f
