"""FastAPI app. Config chosen by env var BANKLENS_CONFIG."""

from __future__ import annotations

import os
import re
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response

from app.config import load_config, load_manifest
from app.pipeline import Pipeline
from app.schemas import (
    AskRequest,
    AskResponse,
    DocumentOut,
    VerifyRequest,
    VerifyResponse,
)

load_dotenv()
STATE: dict = {}


def get_pipeline() -> Pipeline:
    if "pipeline" not in STATE:
        cfg = load_config(os.getenv("BANKLENS_CONFIG", "configs/05_rerank.yaml"))
        STATE["cfg"] = cfg
        STATE["pipeline"] = Pipeline(cfg)
    return STATE["pipeline"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        get_pipeline()
    except FileNotFoundError as e:  # app still starts; /health reports it
        STATE["error"] = str(e)
    yield


app = FastAPI(title="BankLens", lifespan=lifespan)


@app.get("/health")
def health():
    cfg = STATE.get("cfg") or load_config(os.getenv("BANKLENS_CONFIG", "configs/05_rerank.yaml"))
    p = STATE.get("pipeline")
    return {
        "status": "ok" if p else "no_index",
        "config": cfg.name,
        "error": STATE.get("error"),
        "n_chunks": len(p.index.chunks) if p else 0,
        "session_cost_usd": p.ledger.session_usd if p else 0.0,
    }


@app.get("/documents", response_model=list[DocumentOut])
def documents():
    cfg = STATE.get("cfg") or load_config(os.getenv("BANKLENS_CONFIG", "configs/05_rerank.yaml"))
    return [DocumentOut(**d.model_dump()) for d in load_manifest(cfg)]


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        return get_pipeline().ask(req.question)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e)) from e


@app.post("/verify", response_model=VerifyResponse)
def verify(req: VerifyRequest):
    try:
        return get_pipeline().verify(req.claim)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e)) from e


def _find(page, needle: str) -> list:
    for size in (80, 40):
        found = page.search_for(needle[:size])
        if found:
            return found
    return []


def highlight_rects(page, snippet: str, max_rects: int = 80) -> list:
    """Rectangles on ``page`` for the cited passage.

    Text chunks: search each line. Table chunks are markdown ("| a | b |"), which never appears verbatim in the
    PDF, so search each cell. Generic cells ("2025", "Standardized") also occur in surrounding prose, so the
    table's region is anchored on its distinctive numbers and matches outside that band are dropped."""
    is_table = snippet.lstrip().startswith("Table")
    if is_table:
        cells = [c.strip() for ln in snippet.splitlines()[1:] for c in ln.strip().strip("|").split("|")]
        needles = [c for c in cells if len(c) >= 4 and not set(c) <= set("-: ")]
    else:
        needles = [re.sub(r"\s+", " ", ln).strip() for ln in snippet.splitlines()]
        needles = [n for n in needles if len(n) >= 12]

    band = None
    if is_table:  # anchor: numbers that occur exactly once on the page
        anchors = []
        for n in dict.fromkeys(needles):
            if len(n) >= 5 and re.fullmatch(r"[\d,.$%()\s-]+", n) and re.search(r"\d", n):
                found = _find(page, n)
                if len(found) == 1:
                    anchors.append(found[0])
        if anchors:
            band = (min(r.y0 for r in anchors) - 45, max(r.y1 for r in anchors) + 8)

    rects, seen = [], set()
    for n in dict.fromkeys(needles):
        for r in _find(page, n):
            if band and not (band[0] <= r.y0 and r.y1 <= band[1]):
                continue
            key = tuple(round(v) for v in r)
            if key not in seen:
                seen.add(key)
                rects.append(r)
        if len(rects) >= max_rects:
            break
    return rects


@app.get("/page_image/{doc_id}/{page_index}")
def page_image(
    doc_id: str, page_index: int, highlight: str = Query(default=""), chunk_id: str = Query(default="")
):
    import fitz

    get_pipeline()
    cfg = STATE["cfg"]
    docs = {d.doc_id: d for d in load_manifest(cfg)}
    if doc_id not in docs:
        raise HTTPException(404, "unknown doc_id")
    pdf = cfg.path("raw_dir") / docs[doc_id].file
    if not pdf.exists():
        raise HTTPException(404, "PDF not found on server")
    doc = fitz.open(pdf)
    if not 1 <= page_index <= len(doc):
        raise HTTPException(404, "page out of range")
    page = doc.load_page(page_index - 1)
    if chunk_id:  # preferred: the server looks the cited passage up, so the URL stays short
        chunk = STATE["pipeline"].index.by_id.get(chunk_id)
        if chunk is not None and chunk.doc_id == doc_id:
            highlight = chunk.text
    if highlight:
        for r in highlight_rects(page, highlight):
            page.add_highlight_annot(r)
    png = page.get_pixmap(matrix=fitz.Matrix(1.6, 1.6)).tobytes("png")
    doc.close()
    return Response(png, media_type="image/png")
