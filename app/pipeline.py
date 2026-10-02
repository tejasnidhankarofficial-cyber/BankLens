"""Pipeline(config).ask(q) / .verify(claim) with per-query logging."""

from __future__ import annotations

import json
import time
from pathlib import Path

from app.cache import Cache, CostLedger
from app.config import Config
from app.embeddings import Embedder
from app.index import Index, load_index
from app.llm import LLM
from app.prompts import build_ask_prompt, build_verify_prompt
from app.retrieve import Hit, Retriever
from app.schemas import AskResponse, Citation, RetrievedItem, VerifyResponse

VERDICTS = {"SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"}


class Pipeline:
    def __init__(self, cfg: Config, index: Index | None = None):
        self.cfg = cfg
        self.cache = Cache(cfg.path("cache_path"))
        self.ledger = CostLedger(cfg.path("log_dir"))
        self.index = index or load_index(cfg)
        self.embedder = Embedder(cfg, self.cache, self.ledger)
        self.retriever = Retriever(cfg, self.index, self.embedder)
        self.llm = LLM(cfg, self.cache, self.ledger)
        self._reranker = None
        self.log_path = Path(cfg.path("log_dir")) / "queries.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def retrieve(self, query: str) -> list[Hit]:
        r = self.cfg.retrieval
        if self.cfg.rerank.enabled:
            cands, _ = self.retriever.retrieve(query, r.candidate_k)
            if self._reranker is None:
                from app.rerank import Reranker

                self._reranker = Reranker(self.cfg.rerank.model)
            return self._reranker.rerank(query, cands, r.top_k)
        hits, _ = self.retriever.retrieve(query, r.top_k)
        return hits

    @staticmethod
    def _valid(ids, n: int) -> tuple[list[int], int]:
        good, bad = [], 0
        for i in ids or []:
            if isinstance(i, int) and not isinstance(i, bool) and 1 <= i <= n:
                if i not in good:
                    good.append(i)
            else:
                bad += 1
        return good, bad

    def _citations(self, ids: list[int], hits: list[Hit]) -> list[Citation]:
        out = []
        for n in ids:
            c = hits[n - 1].chunk
            out.append(
                Citation(
                    n=n,
                    chunk_id=c.chunk_id,
                    doc_id=c.doc_id,
                    bank=c.bank,
                    fiscal_year=c.fiscal_year,
                    page_label=c.page_label,
                    page_index=c.page_index,
                    section=c.section,
                    snippet=c.text,
                )
            )
        return out

    @staticmethod
    def _retrieved(hits: list[Hit]) -> list[RetrievedItem]:
        return [
            RetrievedItem(
                chunk_id=h.chunk.chunk_id,
                doc_id=h.chunk.doc_id,
                page_index=h.chunk.page_index,
                score=h.score,
            )
            for h in hits
        ]

    def _log(
        self, mode: str, text: str, hits: list[Hit], prompt_hash: str, out: dict
    ) -> None:
        row = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "config": self.cfg.name,
            "mode": mode,
            "input": text,
            "retrieved": [
                {"chunk_id": h.chunk.chunk_id, "score": h.score} for h in hits
            ],
            "prompt_hash": prompt_hash,
            "output": out,
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(row) + "\n")

    # ------------------------------------------------------------------
    def ask(self, question: str) -> AskResponse:
        t0, cost0 = time.perf_counter(), self.ledger.session_usd
        hits = self.retrieve(question)
        system, user = build_ask_prompt(question, [h.chunk for h in hits])
        res = self.llm.complete_json(system, user, kind="ask")
        p = res.parsed or {}
        abstain = bool(p.get("abstain")) or p.get("answer") in (None, "")
        ids, bad = self._valid(p.get("citations"), len(hits))
        answer = None if abstain else str(p.get("answer"))
        resp = AskResponse(
            answer=answer,
            abstain=abstain,
            citations=[] if abstain else self._citations(ids, hits),
            latency_ms=(time.perf_counter() - t0) * 1000,
            cost_usd=self.ledger.session_usd - cost0,
            invalid_citations=bad,
            retrieved=self._retrieved(hits),
            prompt_hash=res.prompt_hash,
        )
        self._log(
            "ask",
            question,
            hits,
            res.prompt_hash,
            {"raw": res.text, "invalid_citations": bad},
        )
        return resp

    def verify(self, claim: str) -> VerifyResponse:
        t0, cost0 = time.perf_counter(), self.ledger.session_usd
        hits = self.retrieve(claim)
        system, user = build_verify_prompt(claim, [h.chunk for h in hits])
        res = self.llm.complete_json(system, user, kind="verify")
        p = res.parsed or {}
        verdict = str(p.get("verdict", "")).upper()
        if verdict not in VERDICTS:
            verdict = "NOT_ENOUGH_INFO"
        ids, bad = self._valid(p.get("evidence"), len(hits))
        resp = VerifyResponse(
            verdict=verdict,
            explanation=str(p.get("explanation", "")),
            evidence=self._citations(ids, hits),
            latency_ms=(time.perf_counter() - t0) * 1000,
            cost_usd=self.ledger.session_usd - cost0,
            invalid_citations=bad,
            retrieved=self._retrieved(hits),
            prompt_hash=res.prompt_hash,
        )
        self._log(
            "verify",
            claim,
            hits,
            res.prompt_hash,
            {"raw": res.text, "invalid_citations": bad},
        )
        return resp
