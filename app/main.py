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


def _search_rects(page, snippet: str):
    text = re.sub(r"\s+", " ", snippet).strip()
    for n in (80, 50, 30):
        rects = page.search_for(text[:n])
        if rects:
            return rects
    return []


@app.get("/page_image/{doc_id}/{page_index}")
def page_image(doc_id: str, page_index: int, highlight: str = Query(default="")):
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
    if highlight:
        for r in _search_rects(page, highlight):
            page.add_highlight_annot(r)
    png = page.get_pixmap(matrix=fitz.Matrix(1.6, 1.6)).tobytes("png")
    doc.close()
    return Response(png, media_type="image/png")
