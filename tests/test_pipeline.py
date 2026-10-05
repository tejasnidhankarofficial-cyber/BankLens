import json

from fastapi.testclient import TestClient

from app.cache import Cache, CostLedger
from app.embeddings import Embedder
from app.index import build_index
from app.pipeline import Pipeline
from app.retrieve import Filter, detect_filter
from tests.conftest import make_cfg


def test_detect_filter():
    f = detect_filter("What was Wells Fargo's net income in 2025 vs JPM?")
    assert set(f.tickers) == {"WFC", "JPM"} and f.years == [2025]
    assert detect_filter("what is the capital ratio").empty
    assert detect_filter("BofA and Citi 2024").tickers == ["BAC", "C"]


def test_cache_makes_reembedding_free(cfg):
    cache, ledger = Cache(cfg.path("cache_path")), CostLedger(cfg.path("log_dir"))
    e = Embedder(cfg, cache, ledger)
    a = e.embed(["hello world", "net income"])
    h0 = cache.hits
    b = e.embed(["hello world", "net income"])
    assert (a == b).all() and cache.hits == h0 + 2


def test_index_shared_across_retrieval_configs(fake_corpus):
    a = make_cfg(fake_corpus, retrieval={"mode": "dense"})
    b = make_cfg(
        fake_corpus,
        retrieval={"mode": "hybrid", "metadata_filter": True},
        rerank={"enabled": True},
    )
    c = make_cfg(fake_corpus, chunking={"strategy": "fixed"})
    assert a.index_id == b.index_id != c.index_id


def _pipe(fake_corpus, **over):
    cfg = make_cfg(fake_corpus, **over)
    cache, ledger = Cache(cfg.path("cache_path")), CostLedger(cfg.path("log_dir"))
    idx = build_index(cfg, cache, ledger, force=True, log=lambda *_: None)
    return cfg, Pipeline(cfg, idx)


def test_all_retrieval_modes_and_filter(fake_corpus):
    for mode in ("dense", "bm25", "hybrid"):
        cfg, p = _pipe(
            fake_corpus,
            retrieval={
                "mode": mode,
                "top_k": 4,
                "candidate_k": 20,
                "metadata_filter": True,
            },
        )
        hits, f = p.retriever.retrieve("What was Wells Fargo net income in 2025?", 4)
        assert hits and all(h.chunk.ticker == "WFC" and h.chunk.fiscal_year == 2025 for h in hits)
        assert f == Filter(["WFC"], [2025])


def test_year_filter_falls_back_to_bank(fake_corpus):
    cfg, p = _pipe(fake_corpus)
    hits, f = p.retriever.retrieve("What will JPMorgan's dividend be in 2030?", 4)
    assert f.tickers == ["JPM"] and f.years == []
    assert hits and all(h.chunk.ticker == "JPM" for h in hits)


def test_bm25_finds_table_numbers(fake_corpus):
    cfg, p = _pipe(fake_corpus, retrieval={"mode": "bm25", "top_k": 3, "metadata_filter": False})
    hits, _ = p.retriever.retrieve("net income 58,000", 3)
    assert hits[0].chunk.kind == "table" and hits[0].chunk.ticker == "JPM"


def test_ask_verify_and_logging(fake_corpus):
    cfg, p = _pipe(fake_corpus)
    r = p.ask("What was the CET1 ratio of JPMorgan in 2025?")
    assert not r.abstain and r.answer and r.citations
    c = r.citations[0]
    assert c.bank == "JPMorgan Chase" and c.fiscal_year == 2025 and c.page_index >= 1
    assert r.cost_usd == 0.0 and r.latency_ms > 0

    r2 = p.ask("What will JPMorgan's dividend be in 2030?")
    assert r2.abstain and r2.answer is None and r2.citations == []

    v = p.verify("JPMorgan net income was 58,000 in 2025")
    assert v.verdict == "SUPPORTED" and v.evidence
    v2 = p.verify("Wells Fargo is about to be acquired")
    assert v2.verdict == "NOT_ENOUGH_INFO"

    rows = [json.loads(line) for line in open(p.log_path)][-4:]
    assert [r["mode"] for r in rows] == ["ask", "ask", "verify", "verify"]
    assert rows[0]["retrieved"] and rows[0]["prompt_hash"]


def test_invalid_citations_dropped(fake_corpus):
    cfg, p = _pipe(fake_corpus)
    ids, bad = p._valid([1, 9, 0, "2", 1, True], 4)
    assert ids == [1] and bad == 4


def test_api(fake_corpus, monkeypatch):
    import app.main as m

    cfg, p = _pipe(fake_corpus)
    m.STATE.clear()
    m.STATE.update(cfg=cfg, pipeline=p)
    client = TestClient(m.app)
    assert client.get("/health").json()["status"] == "ok"
    assert len(client.get("/documents").json()) == 4
    j = client.post("/ask", json={"question": "CET1 ratio for Wells Fargo in 2025"}).json()
    assert j["citations"][0]["bank"] == "Wells Fargo"
    assert (
        client.post("/verify", json={"claim": "Wells Fargo net income was 19,000"}).json()["verdict"]
        == "SUPPORTED"
    )
    img = client.get("/page_image/jpm_fy2025/2", params={"highlight": "CET1 capital ratio was 15.7%"})
    assert img.status_code == 200 and img.content[:4] == b"\x89PNG"
    by_id = client.get("/page_image/jpm_fy2025/3", params={"chunk_id": "jpm_fy2025:p3:c1"})
    assert by_id.status_code == 200 and by_id.content[:4] == b"\x89PNG"
    assert client.get("/page_image/jpm_fy2025/99").status_code == 404
    assert client.get("/page_image/nope/1").status_code == 404
    m.STATE.clear()


def test_highlight_rects_text_and_table(fake_corpus):
    import pymupdf

    from app.main import highlight_rects

    doc = pymupdf.open(fake_corpus / "raw" / "jpm_fy2025.pdf")
    text_rects = highlight_rects(
        doc[1], "The Common Equity Tier 1 (CET1) capital ratio was 15.7% at December 31, 2025."
    )
    assert text_rects
    table = "Table: Consolidated Results of Operations (in millions)\n| Metric | 2025 | 2024 |\n| --- | --- | --- |\n| Net income | 58,000 | 54,000 |"
    table_rects = highlight_rects(doc[2], table)
    assert len(table_rects) >= 3  # markdown never matches verbatim; cells must
    assert highlight_rects(doc[2], "text that is nowhere on this page at all") == []
