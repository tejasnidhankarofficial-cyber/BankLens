import json

from eval import compare


def test_compare_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(compare, "RES", tmp_path)
    for name, hit in (("01_a", 0.4), ("02_b", 0.8)):
        res = {
            "name": name, "cost_usd": 0.1, "retrieval_only": False,
            "question_metrics": {
                "numeric": {"hit@5": hit, "answer_correct": hit},
                "_overall": {"hit@5": hit, "mrr": hit / 2, "answer_correct": hit, "latency_ms": 900.0},
                "_abstention": {"abstain_precision": 1.0, "abstain_recall": 0.5, "false_abstention_rate": 0.0},
            },
            "verify_metrics": {"accuracy": 0.7, "macro_f1": 0.6, "by_perturbation": {"number": {"n": 3, "accuracy": 0.7}}},
        }  # fmt: skip
        (tmp_path / f"{name}.json").write_text(json.dumps(res))
    compare.main()
    md = (tmp_path / "results.md").read_text()
    assert "01_a" in md and "02_b" in md and "number" in md
    assert (tmp_path / "ablation.png").exists() and (tmp_path / "by_type.png").exists()
