"""Pydantic config loaded from YAML. One YAML per experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent


class ChunkingConfig(BaseModel):
    strategy: Literal["fixed", "section", "section_table"] = "fixed"
    chunk_size: int = 512
    overlap: int = 64
    contextual_header: bool = False


class EmbeddingConfig(BaseModel):
    provider: Literal["openai", "local", "hash"] = "hash"
    model: str = "text-embedding-3-small"


class RetrievalConfig(BaseModel):
    mode: Literal["dense", "bm25", "hybrid"] = "dense"
    top_k: int = 5
    candidate_k: int = 30
    rrf_k: int = 60
    metadata_filter: bool = False


class RerankConfig(BaseModel):
    enabled: bool = False
    model: str = "BAAI/bge-reranker-base"


class LLMConfig(BaseModel):
    provider: Literal["openai", "fake"] = "fake"
    model: str = "gpt-4.1-mini"
    temperature: float = 0.0
    max_tokens: int = 600
    # USD per 1M tokens. Check current OpenAI pricing before running evals.
    price_in: float = 0.40
    price_out: float = 1.60
    embed_price: float = 0.02


class Config(BaseModel):
    name: str = "default"
    parser: Literal["pypdf", "pymupdf"] = "pypdf"
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    # Paths (overridable so tests can use temp dirs)
    manifest_path: str = "data/manifest.yaml"
    raw_dir: str = "data/raw"
    index_dir: str = "indexes"
    cache_path: str = "cache/cache.sqlite"
    log_dir: str = "logs"

    @property
    def index_id(self) -> str:
        """Hash of parser + chunking + embedding only, so configs that differ
        only in retrieval/rerank/llm share one index."""
        payload = {
            "parser": self.parser,
            "chunking": self.chunking.model_dump(),
            "embedding": self.embedding.model_dump(),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]

    def path(self, attr: str) -> Path:
        p = Path(getattr(self, attr))
        return p if p.is_absolute() else ROOT / p


def load_config(path: str | Path) -> Config:
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = ROOT / p
    with open(p) as f:
        data = yaml.safe_load(f) or {}
    data.setdefault("name", p.stem)
    return Config(**data)


class DocumentInfo(BaseModel):
    doc_id: str
    file: str
    bank: str
    ticker: str
    fiscal_year: int
    form: str = "10-K"
    source_url: str = ""


def load_manifest(cfg: Config) -> list[DocumentInfo]:
    with open(cfg.path("manifest_path")) as f:
        data = yaml.safe_load(f) or {}
    return [DocumentInfo(**d) for d in data.get("documents", [])]
