"""Build / load the per-index_id store: Chroma collection + chunks.jsonl (for BM25)."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path

from app.cache import Cache, CostLedger
from app.chunking import Chunk, chunk_document
from app.config import Config, DocumentInfo, load_manifest
from app.embeddings import Embedder
from app.parsing import parse_pdf

COLLECTION = "chunks"


def index_path(cfg: Config) -> Path:
    return cfg.path("index_dir") / cfg.index_id


def _chroma(path: Path):
    import chromadb
    from chromadb.config import Settings

    return chromadb.PersistentClient(path=str(path / "chroma"), settings=Settings(anonymized_telemetry=False))


def _parse_and_chunk(args: tuple) -> tuple[str, int, list[Chunk]]:
    """Worker (top-level so it can be pickled): parse one PDF and chunk it."""
    pdf, doc, parser, chunking = args
    pages = parse_pdf(pdf, parser)
    return doc.doc_id, len(pages), chunk_document(pages, doc, chunking)


class Index:
    def __init__(self, chunks: list[Chunk], collection):
        self.chunks = chunks
        self.by_id = {c.chunk_id: c for c in chunks}
        self.collection = collection


def build_index(
    cfg: Config,
    cache: Cache,
    ledger: CostLedger,
    docs: list[DocumentInfo] | None = None,
    force: bool = False,
    log=print,
) -> Index:
    path = index_path(cfg)
    if path.exists() and force:
        from chromadb.api.client import SharedSystemClient

        SharedSystemClient.clear_system_cache()  # drop cached handles before deleting files
        shutil.rmtree(path)
    if (path / "chunks.jsonl").exists():
        log(f"index {cfg.index_id} already exists, loading")
        return load_index(cfg)
    path.mkdir(parents=True, exist_ok=True)

    docs = docs if docs is not None else load_manifest(cfg)
    embedder = Embedder(cfg, cache, ledger)
    chunks: list[Chunk] = []
    jobs = []
    for doc in docs:
        pdf = cfg.path("raw_dir") / doc.file
        if not pdf.exists():
            log(f"  ! missing {pdf}, skipping")
            continue
        jobs.append((str(pdf), doc, cfg.parser, cfg.chunking))
    if len(jobs) > 1:  # parsing is CPU-bound and slow on 400-page PDFs: one process per document
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=min(len(jobs), os.cpu_count() or 1)) as ex:
            results = list(ex.map(_parse_and_chunk, jobs))
    else:
        results = [_parse_and_chunk(j) for j in jobs]
    for doc_id, n_pages, cs in results:
        log(f"  {doc_id}: {n_pages} pages -> {len(cs)} chunks")
        chunks.extend(cs)
    if not chunks:
        raise RuntimeError("no chunks produced; are the PDFs in data/raw/?")

    log(f"embedding {len(chunks)} chunks with {cfg.embedding.provider}:{cfg.embedding.model}")
    vecs = embedder.embed([c.embed_text for c in chunks])

    client = _chroma(path)
    col = client.get_or_create_collection(
        COLLECTION, metadata={"hnsw:space": "cosine"}, embedding_function=None
    )
    B = 1000
    for i in range(0, len(chunks), B):
        part = chunks[i : i + B]
        col.add(
            ids=[c.chunk_id for c in part],
            embeddings=vecs[i : i + B].tolist(),
            documents=[c.text for c in part],
            metadatas=[c.metadata() for c in part],
        )
    with open(path / "chunks.jsonl", "w") as f:
        f.writelines(json.dumps(asdict(c)) + "\n" for c in chunks)
    (path / "meta.json").write_text(
        json.dumps(
            {
                "index_id": cfg.index_id,
                "config": cfg.model_dump(),
                "n_chunks": len(chunks),
            },
            indent=2,
        )
    )
    log(f"index {cfg.index_id} built: {len(chunks)} chunks. session cost ${ledger.session_usd:.4f}")
    return Index(chunks, col)


def load_index(cfg: Config) -> Index:
    path = index_path(cfg)
    f = path / "chunks.jsonl"
    if not f.exists():
        raise FileNotFoundError(f"No index at {path}. Run: python -m scripts.ingest --config <config.yaml>")
    with open(f) as fh:
        chunks = [Chunk(**json.loads(line)) for line in fh if line.strip()]
    col = _chroma(path).get_collection(COLLECTION, embedding_function=None)
    return Index(chunks, col)
