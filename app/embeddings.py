"""Embedding providers (openai | local | hash), all cached in SQLite."""

from __future__ import annotations

import hashlib
import re

import numpy as np

from app.cache import Cache, CostLedger, cache_key
from app.chunking import count_tokens
from app.config import Config

HASH_DIM = 256
_TOK = re.compile(r"[a-z0-9]+")


def hash_embed(text: str) -> list[float]:
    """Fake embedding: signed feature hashing of tokens. Deterministic, offline, and
    lexically meaningful enough for tests to retrieve sensibly."""
    v = np.zeros(HASH_DIM, dtype=np.float32)
    for tok in _TOK.findall(text.lower()):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        v[h % HASH_DIM] += 1.0 if (h >> 64) & 1 else -1.0
    n = np.linalg.norm(v)
    return (v / n if n else v).tolist()


class Embedder:
    BATCH = 96

    def __init__(self, cfg: Config, cache: Cache, ledger: CostLedger):
        self.provider = cfg.embedding.provider
        self.model = cfg.embedding.model
        self.price = cfg.llm.embed_price
        self.cache = cache
        self.ledger = ledger
        self._client = None
        self._local = None

    def _key(self, text: str) -> str:
        return cache_key(f"emb:{self.provider}:{self.model}", text)

    def embed(self, texts: list[str]) -> np.ndarray:
        keys = [self._key(t) for t in texts]
        found = self.cache.get_many(keys)
        todo = {}  # key -> text (dedupe)
        for k, t in zip(keys, texts):
            if k not in found:
                todo[k] = t
        if todo:
            items = list(todo.items())
            for i in range(0, len(items), self.BATCH):
                batch = items[i : i + self.BATCH]
                vecs = self._embed_uncached([t for _, t in batch])
                new = {k: v for (k, _), v in zip(batch, vecs)}
                self.cache.set_many(new)
                found.update(new)
        return np.array([found[k] for k in keys], dtype=np.float32)

    def _embed_uncached(self, texts: list[str]) -> list[list[float]]:
        if self.provider == "hash":
            return [hash_embed(t) for t in texts]
        if self.provider == "local":
            if self._local is None:
                from sentence_transformers import SentenceTransformer

                self._local = SentenceTransformer(self.model)
            return self._local.encode(texts, normalize_embeddings=True).tolist()
        if self.provider == "openai":
            if self._client is None:
                from openai import OpenAI

                self._client = OpenAI()
            # Truncate very long inputs to stay under the 8191-token limit.
            texts = [t[:24000] for t in texts]
            resp = self._client.embeddings.create(model=self.model, input=texts)
            tokens = resp.usage.total_tokens if resp.usage else sum(count_tokens(t) for t in texts)
            self.ledger.record("embedding", self.model, tokens, 0, tokens * self.price / 1e6)
            return [d.embedding for d in resp.data]
        raise ValueError(self.provider)
