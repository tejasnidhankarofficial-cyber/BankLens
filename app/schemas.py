from __future__ import annotations

from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str


class VerifyRequest(BaseModel):
    claim: str


class Citation(BaseModel):
    n: int
    chunk_id: str
    doc_id: str
    bank: str
    fiscal_year: int
    page_label: str
    page_index: int
    section: str
    snippet: str


class RetrievedItem(BaseModel):
    chunk_id: str
    doc_id: str
    page_index: int
    score: float


class AskResponse(BaseModel):
    answer: str | None
    abstain: bool
    citations: list[Citation]
    latency_ms: float
    cost_usd: float
    invalid_citations: int = 0
    retrieved: list[RetrievedItem] = []
    prompt_hash: str = ""


class VerifyResponse(BaseModel):
    verdict: str
    explanation: str
    evidence: list[Citation]
    latency_ms: float
    cost_usd: float
    invalid_citations: int = 0
    retrieved: list[RetrievedItem] = []
    prompt_hash: str = ""


class DocumentOut(BaseModel):
    doc_id: str
    bank: str
    ticker: str
    fiscal_year: int
    form: str
    source_url: str = ""
