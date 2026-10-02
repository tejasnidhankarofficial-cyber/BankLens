"""Evaluation metrics. Pure functions; no I/O so they are easy to unit-test.

A retrieval/citation match = same doc_id and page_index within +/-1.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from statistics import mean

LABELS = ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]
PAGE_TOL = 1

# ------------------------------------------------------------------ retrieval


def expected_pages(sources: list[dict]) -> list[tuple[str, int]]:
    return [(s["doc_id"], p) for s in sources for p in s["pages"]]


def page_match(doc_id: str, page: int, exp: list[tuple[str, int]]) -> bool:
    return any(doc_id == d and abs(page - p) <= PAGE_TOL for d, p in exp)


def retrieval_metrics(retrieved: list[tuple[str, int]], sources: list[dict], ks=(1, 3, 5)) -> dict:
    """retrieved: ranked [(doc_id, page_index)] (duplicates allowed)."""
    exp = expected_pages(sources)
    if not exp:
        return {}
    out: dict[str, float] = {}
    for k in ks:
        top = retrieved[:k]
        out[f"hit@{k}"] = float(any(page_match(d, p, exp) for d, p in top))
    top = retrieved[: max(ks)]
    found = sum(any(d == ed and abs(p - ep) <= PAGE_TOL for d, p in top) for ed, ep in exp)
    out[f"recall@{max(ks)}"] = found / len(exp)
    rr = 0.0
    for rank, (d, p) in enumerate(retrieved, start=1):
        if page_match(d, p, exp):
            rr = 1.0 / rank
            break
    out["mrr"] = rr
    return out


def citation_precision(cited: list[tuple[str, int]], sources: list[dict]) -> float | None:
    exp = expected_pages(sources)
    if not cited or not exp:
        return None
    return sum(page_match(d, p, exp) for d, p in cited) / len(cited)


# ------------------------------------------------------------------ numeric answers

_SCALE = {
    "trillion": 1e12, "billion": 1e9, "bn": 1e9, "b": 1e9,
    "million": 1e6, "mm": 1e6, "m": 1e6, "thousand": 1e3, "k": 1e3,
}  # fmt: skip
_NUM = re.compile(
    r"(?<![\w.])\(?-?\$?\s*(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*"
    r"(trillion|billion|million|thousand|bn|mm|%|percent|b|m|k)?(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_UNIT_TARGET = {"USD_billions": 1e9, "USD_millions": 1e6, "USD_thousands": 1e3, "USD": 1.0}


def parse_numbers(text: str) -> list[tuple[float, str]]:
    """All numbers in text as (value, suffix) where suffix is a lowercase scale word, '%' or ''."""
    out = []
    for m in _NUM.finditer(text):
        v = float(m.group(1).replace(",", ""))
        suf = (m.group(2) or "").lower()
        out.append((v, "%" if suf == "percent" else suf))
    return out


def numeric_match(answer: str | None, value: float, unit: str, tol: float = 0.01) -> bool:
    """Does any number in ``answer`` equal ``value`` (in ``unit``) within ``tol``?
    USD amounts are normalized to dollars; a bare number inherits the only scale word
    mentioned in the answer (e.g. '58,000 ... in millions')."""
    if not answer:
        return False
    nums = parse_numbers(answer)
    if unit in ("percent", "pct", "%"):
        return any(suf == "%" and abs(v - value) <= tol * max(abs(value), 1e-9) for v, suf in nums)
    target = value * _UNIT_TARGET[unit]
    words = {w for w in re.findall(r"trillion|billion|million|thousand", answer.lower())}
    ambient = _SCALE[next(iter(words))] if len(words) == 1 else None
    for v, suf in nums:
        if suf == "%":
            continue
        mult = _SCALE.get(suf) or ambient
        cands = [v * mult] if mult else [v, v * 1e3, v * 1e6, v * 1e9]
        if any(abs(c - target) <= tol * abs(target) for c in cands):
            return True
    return False


def keywords_match(answer: str | None, keywords: list[str]) -> bool:
    if not answer or not keywords:
        return False
    a = answer.lower()
    return all(k.lower() in a for k in keywords)


# ------------------------------------------------------------------ abstention


def abstention_metrics(items: list[dict]) -> dict:
    """items: [{'unanswerable': bool, 'abstain': bool}]"""
    if not items:
        return {}
    tp = sum(i["unanswerable"] and i["abstain"] for i in items)
    fp = sum((not i["unanswerable"]) and i["abstain"] for i in items)
    n_un = sum(i["unanswerable"] for i in items)
    n_ans = len(items) - n_un
    return {
        "abstain_precision": tp / (tp + fp) if tp + fp else None,
        "abstain_recall": tp / n_un if n_un else None,
        "false_abstention_rate": fp / n_ans if n_ans else None,
    }


# ------------------------------------------------------------------ verification


def verify_metrics(gold: list[str], pred: list[str]) -> dict:
    conf = {g: {p: 0 for p in LABELS} for g in LABELS}
    for g, p in zip(gold, pred):
        conf[g][p] += 1
    f1s = {}
    for lab in LABELS:
        tp = conf[lab][lab]
        fp = sum(conf[g][lab] for g in LABELS if g != lab)
        fn = sum(conf[lab][p] for p in LABELS if p != lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s[lab] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    n = len(gold)
    return {
        "n": n,
        "accuracy": sum(g == p for g, p in zip(gold, pred)) / n if n else None,
        "macro_f1": mean(f1s.values()) if n else None,
        "f1_per_class": f1s,
        "confusion": conf,
    }


def verify_by_perturbation(rows: list[dict]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r.get("perturbation", "none")].append(r)
    return {
        k: {"n": len(v), "accuracy": sum(r["gold"] == r["pred"] for r in v) / len(v)}
        for k, v in sorted(groups.items())
    }


# ------------------------------------------------------------------ agreement


def cohens_kappa(a: list, b: list) -> float | None:
    if not a or len(a) != len(b):
        return None
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


# ------------------------------------------------------------------ aggregation


def mean_defined(rows: list[dict], key: str):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return mean(vals) if vals else None


AGG_KEYS = [
    "hit@1", "hit@3", "hit@5", "recall@5", "mrr", "answer_correct",
    "faithfulness", "citation_precision", "invalid_citations", "latency_ms", "cost_usd",
]  # fmt: skip


def aggregate(rows: list[dict]) -> dict:
    out = {"n": len(rows)}
    for k in AGG_KEYS:
        out[k] = mean_defined(rows, k)
    return out


def aggregate_by_type(rows: list[dict]) -> dict:
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[r["type"]].append(r)
    res = {t: aggregate(v) for t, v in sorted(by.items())}
    # Abstention is a property of the whole set, but report it per answerable/unanswerable split.
    res["_abstention"] = abstention_metrics(
        [{"unanswerable": r["type"] == "unanswerable", "abstain": r["abstain"]} for r in rows]
    )
    return res
