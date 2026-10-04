"""python -m eval.run_eval --config configs/x.yaml [--judge] [--limit N] [--retrieval-only]"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from app.config import ROOT, Config, load_config
from app.pipeline import Pipeline
from eval import metrics as M
from eval.judge import Judge

EVAL_DIR = ROOT / "eval"


def git_commit() -> str:
    try:
        h = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
        return h + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def load_json(path: Path) -> list[dict]:
    if not path.exists():
        print(f"  (skipping: {path} not found)")
        return []
    return json.loads(path.read_text())


def eval_question(pipe: Pipeline, q: dict, judge: Judge | None, retrieval_only: bool) -> dict:
    unans = q["type"] == "unanswerable"
    row: dict = {"id": q["id"], "type": q["type"], "question": q["question"]}
    if retrieval_only:
        hits = pipe.retrieve(q["question"])
        retrieved = [(h.chunk.doc_id, h.chunk.page_index) for h in hits]
        row.update(abstain=False, retrieved=[list(r) for r in retrieved])
        if not unans:
            row.update(M.retrieval_metrics(retrieved, q.get("sources", [])))
        return row

    resp = pipe.ask(q["question"])
    retrieved = [(r.doc_id, r.page_index) for r in resp.retrieved]
    cited = [(c.doc_id, c.page_index) for c in resp.citations]
    row.update(
        answer=resp.answer, abstain=resp.abstain, retrieved=[list(r) for r in retrieved],
        cited=[c.chunk_id for c in resp.citations], invalid_citations=resp.invalid_citations,
        latency_ms=resp.latency_ms, cost_usd=resp.cost_usd,
    )  # fmt: skip
    if unans:
        row["answer_correct"] = float(resp.abstain)  # correct behaviour = abstain
        return row
    row.update(M.retrieval_metrics(retrieved, q.get("sources", [])))
    row["citation_precision"] = M.citation_precision(cited, q.get("sources", []))
    exp = q.get("expected", {})
    correct = None
    if resp.abstain:
        correct = False
    elif q["type"] == "numeric" and "value" in exp:
        correct = M.numeric_match(resp.answer, exp["value"], exp.get("unit", "USD"))
    elif exp.get("keywords"):
        correct = M.keywords_match(resp.answer, exp["keywords"])
    elif judge and exp.get("answer"):
        correct = judge.correct(q["question"], exp["answer"], resp.answer or "")
    row["answer_correct"] = None if correct is None else float(correct)
    if judge and not resp.abstain and resp.citations:
        row["faithfulness"] = judge.faithfulness(resp.answer or "", [c.snippet for c in resp.citations])
    return row


def eval_claim(pipe: Pipeline, c: dict, retrieval_only: bool) -> dict:
    row: dict = {
        "id": c["id"],
        "claim": c["claim"],
        "gold": c["label"],
        "perturbation": c.get("perturbation", "none"),
    }
    if retrieval_only:
        hits = pipe.retrieve(c["claim"])
        retrieved = [(h.chunk.doc_id, h.chunk.page_index) for h in hits]
        row.update(M.retrieval_metrics(retrieved, c.get("sources", [])))
        return row
    resp = pipe.verify(c["claim"])
    retrieved = [(r.doc_id, r.page_index) for r in resp.retrieved]
    row.update(
        pred=resp.verdict, explanation=resp.explanation, evidence=[e.chunk_id for e in resp.evidence],
        invalid_citations=resp.invalid_citations, latency_ms=resp.latency_ms, cost_usd=resp.cost_usd,
        retrieved=[list(r) for r in retrieved],
    )  # fmt: skip
    row.update(M.retrieval_metrics(retrieved, c.get("sources", [])))
    return row


def run(cfg: Config, args) -> dict:
    pipe = Pipeline(cfg)
    judge = Judge(pipe.llm) if args.judge else None
    questions = load_json(Path(args.questions))[: args.limit or None]
    claims = load_json(Path(args.claims))[: args.limit or None]
    cost0, t0 = pipe.ledger.session_usd, time.time()

    def guard():
        if pipe.ledger.session_usd - cost0 > args.max_cost:
            sys.exit(f"ABORT: run exceeded --max-cost ${args.max_cost:.2f}. Partial results not saved.")

    q_rows = []
    for i, q in enumerate(questions, 1):
        q_rows.append(eval_question(pipe, q, judge, args.retrieval_only))
        guard()
        print(f"  q {i}/{len(questions)} {q['id']}", end="\r")
    c_rows = []
    for i, c in enumerate(claims, 1):
        c_rows.append(eval_claim(pipe, c, args.retrieval_only))
        guard()
        print(f"  c {i}/{len(claims)} {c['id']}", end="\r")

    # A fully cached re-run (e.g. after a metric fix) reports ~0 ms and $0. Keep the original measurements.
    spent = pipe.ledger.session_usd - cost0
    prior_path = Path(args.out_dir) / f"{cfg.name}{'_retrieval' if args.retrieval_only else ''}.json"
    carried = None
    if spent == 0 and prior_path.exists():
        carried = json.loads(prior_path.read_text())
        old_rows = {r["id"]: r for r in carried.get("questions", []) + carried.get("claims", [])}
        for r in q_rows + c_rows:
            o = old_rows.get(r["id"], {})
            for k in ("latency_ms", "cost_usd"):
                if k in o:
                    r[k] = o[k]

    result: dict = {
        "name": cfg.name, "config": cfg.model_dump(), "git_commit": git_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "judge": bool(judge),
        "retrieval_only": args.retrieval_only, "n_questions": len(q_rows), "n_claims": len(c_rows),
        "cost_usd": carried["cost_usd"] if carried else spent, "wall_seconds": time.time() - t0,
        "rescored_from_cache": bool(carried),
        "questions": q_rows, "claims": c_rows,
    }  # fmt: skip
    result["question_metrics"] = M.aggregate_by_type(q_rows) if q_rows else {}
    result["question_metrics"]["_overall"] = M.aggregate(q_rows) if q_rows else {}
    if c_rows:
        result["claim_retrieval"] = M.aggregate(c_rows)
        if not args.retrieval_only:
            gold, pred = [r["gold"] for r in c_rows], [r["pred"] for r in c_rows]
            result["verify_metrics"] = M.verify_metrics(gold, pred)
            result["verify_metrics"]["by_perturbation"] = M.verify_by_perturbation(c_rows)
    return result


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--judge", action="store_true", help="LLM-judge faithfulness + text-answer correctness")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--retrieval-only", action="store_true", help="skip the LLM; retrieval metrics only")
    ap.add_argument("--questions", default=str(EVAL_DIR / "questions.json"))
    ap.add_argument("--claims", default=str(EVAL_DIR / "claims.json"))
    ap.add_argument("--out-dir", default=str(EVAL_DIR / "results"))
    ap.add_argument("--max-cost", type=float, default=1.0, help="abort if this run's spend exceeds USD")
    args = ap.parse_args()

    cfg = load_config(args.config)
    result = run(cfg, args)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    suffix = "_retrieval" if args.retrieval_only else ""
    path = out / f"{cfg.name}{suffix}.json"
    path.write_text(json.dumps(result, indent=2))

    ov = result["question_metrics"].get("_overall", {})
    print(f"\n{cfg.name}: n_q={result['n_questions']} n_claims={result['n_claims']}")
    for k in ("hit@5", "mrr", "answer_correct", "faithfulness", "citation_precision"):
        if ov.get(k) is not None:
            print(f"  {k}: {ov[k]:.3f}")
    if "verify_metrics" in result:
        print(
            f"  verify accuracy: {result['verify_metrics']['accuracy']:.3f}  macro-F1: {result['verify_metrics']['macro_f1']:.3f}"
        )
    from app.cache import CostLedger

    print(
        f"this run: ${result['cost_usd']:.4f} | all-time spend: ${CostLedger(cfg.path('log_dir')).total_all_time():.4f}"
    )
    print(f"saved {path}")


if __name__ == "__main__":
    main()
