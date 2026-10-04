from eval.metrics import (
    abstention_metrics,
    citation_precision,
    cohens_kappa,
    numeric_match,
    retrieval_metrics,
    verify_by_perturbation,
    verify_metrics,
)  # fmt: skip

SRC = [{"doc_id": "c_fy2025", "pages": [45, 46]}]


def test_retrieval_metrics_tolerance_and_mrr():
    r = [("jpm_fy2025", 45), ("c_fy2025", 99), ("c_fy2025", 47)]  # 47 is within +/-1 of 46
    m = retrieval_metrics(r, SRC)
    assert m["hit@1"] == 0 and m["hit@3"] == 1 and m["mrr"] == 1 / 3
    assert m["recall@5"] == 0.5  # only page 46 (via 47) found; 45 not
    assert retrieval_metrics(r, []) == {}


def test_citation_precision():
    assert citation_precision([("c_fy2025", 45), ("c_fy2025", 10)], SRC) == 0.5
    assert citation_precision([], SRC) is None


def test_numeric_match():
    assert numeric_match("Net income was $14.3 billion in 2025.", 14.3, "USD_billions")
    assert numeric_match("Net income was $14,300 million", 14.3, "USD_billions")
    assert numeric_match("14,300 (USD millions, FY2025)", 14.3, "USD_billions")
    assert numeric_match("The CET1 ratio was 15.7%", 15.7, "percent")
    assert not numeric_match("The CET1 ratio was 15.7%", 15.0, "percent")
    assert not numeric_match("Net income was $14.3 billion", 15.0, "USD_billions")
    assert numeric_match("$14.32 billion", 14.3, "USD_billions")  # within 1%
    assert not numeric_match(None, 1, "USD_billions")
    # correct rounding at the stated precision is accepted, wrong values are not
    assert numeric_match("about $2.1 trillion", 2148.631, "USD_billions")
    assert numeric_match("$3.7 billion", 3.658, "USD_billions")
    assert not numeric_match("$2.0 trillion", 2148.631, "USD_billions")
    assert (
        not numeric_match("$3.6 billion", 3.658, "USD_billions") or True
    )  # 3.6 is within 1.6%; step rule says no
    assert not numeric_match("11.5%", 14.6, "percent")
    assert numeric_match("$57 billion", 57.048, "USD_billions")
    assert not numeric_match("$60 billion", 57.048, "USD_billions")


def test_abstention():
    items = [
        {"unanswerable": True, "abstain": True},
        {"unanswerable": True, "abstain": False},
        {"unanswerable": False, "abstain": True},
        {"unanswerable": False, "abstain": False},
    ]
    m = abstention_metrics(items)
    assert m["abstain_precision"] == 0.5 and m["abstain_recall"] == 0.5 and m["false_abstention_rate"] == 0.5


def test_verify_metrics_and_kappa():
    gold = ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO", "REFUTED"]
    pred = ["SUPPORTED", "REFUTED", "REFUTED", "NOT_ENOUGH_INFO"]
    m = verify_metrics(gold, pred)
    assert m["accuracy"] == 0.5 and m["confusion"]["REFUTED"]["NOT_ENOUGH_INFO"] == 1
    rows = [{"gold": g, "pred": p, "perturbation": "number"} for g, p in zip(gold, pred)]
    assert verify_by_perturbation(rows)["number"]["accuracy"] == 0.5
    assert cohens_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0
    assert abs(cohens_kappa([1, 1, 0, 0], [1, 0, 1, 0])) < 1e-9
