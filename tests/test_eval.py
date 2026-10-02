import argparse
import json

from app.cache import Cache, CostLedger
from app.index import build_index
from eval.judge import Judge
from eval.run_eval import run
from tests.conftest import make_cfg

QUESTIONS = [
    {"id": "t1", "type": "numeric", "question": "What was Wells Fargo's net income in 2025?",
     "expected": {"value": 19000.0, "unit": "USD_millions"}, "sources": [{"doc_id": "wfc_fy2025", "pages": [3]}]},
    {"id": "t2", "type": "factual", "question": "What was the CET1 ratio of JPMorgan in 2025?",
     "expected": {"answer": "15.7%", "keywords": ["15.7"]}, "sources": [{"doc_id": "jpm_fy2025", "pages": [2]}]},
    {"id": "t3", "type": "unanswerable", "question": "What will JPMorgan's dividend be in 2030?", "sources": []},
]  # fmt: skip
CLAIMS = [
    {"id": "c1", "claim": "JPMorgan net income was 58,000 in 2025", "label": "SUPPORTED", "perturbation": "none",
     "sources": [{"doc_id": "jpm_fy2025", "pages": [3]}]},
    {"id": "c2", "claim": "Citigroup is about to be acquired", "label": "NOT_ENOUGH_INFO", "perturbation": "unverifiable"},
]  # fmt: skip


class StubLLM:
    def complete_json(self, system, user, kind="x"):
        from app.llm import LLMResult

        txt = '{"correct": true, "claims": [{"claim": "a", "supported": true}, {"claim": "b", "supported": false}]}'
        return LLMResult(txt, json.loads(txt), 0, 0, 0.0, False, "h")


def test_judge_with_stub():
    j = Judge(StubLLM())
    assert j.faithfulness("ans", ["x"]) == 0.5 and j.correct("q", "ref", "ans") is True


def test_run_eval_end_to_end(fake_corpus, tmp_path):
    cfg = make_cfg(fake_corpus)
    build_index(
        cfg, Cache(cfg.path("cache_path")), CostLedger(cfg.path("log_dir")), force=True, log=lambda *_: None
    )
    qf, cf = tmp_path / "q.json", tmp_path / "c.json"
    qf.write_text(json.dumps(QUESTIONS))
    cf.write_text(json.dumps(CLAIMS))
    args = argparse.Namespace(
        judge=False, limit=0, retrieval_only=False, questions=str(qf), claims=str(cf), max_cost=1.0
    )
    res = run(cfg, args)
    qm = res["question_metrics"]
    assert qm["numeric"]["hit@5"] == 1.0 and qm["unanswerable"]["answer_correct"] == 1.0
    assert qm["_abstention"]["abstain_recall"] == 1.0
    assert res["verify_metrics"]["n"] == 2 and res["git_commit"] and res["config"]["name"] == "test"
    assert res["cost_usd"] == 0.0

    args.retrieval_only = True
    r2 = run(cfg, args)
    assert "verify_metrics" not in r2 and r2["question_metrics"]["factual"]["hit@5"] == 1.0
