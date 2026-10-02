"""Find source pages fast while labeling. Free: uses BM25/substring over chunks.jsonl, no API calls.

python -m eval.label_helper --config configs/label.yaml "CET1 capital ratio" --ticker JPM --year 2025
python -m eval.label_helper --config configs/label.yaml --exact "Standardized" -n 5
"""

from __future__ import annotations

import argparse

from rank_bm25 import BM25Okapi

from app.config import load_config
from app.index import index_path, load_index
from app.retrieve import tokenize


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phrase", nargs="?", default="")
    ap.add_argument("--config", default="configs/label.yaml")
    ap.add_argument("--exact", help="case-insensitive substring match instead of BM25")
    ap.add_argument("--ticker")
    ap.add_argument("--year", type=int)
    ap.add_argument("-n", type=int, default=8)
    a = ap.parse_args()

    cfg = load_config(a.config)
    chunks = load_index(cfg).chunks
    chunks = [
        c
        for c in chunks
        if (not a.ticker or c.ticker == a.ticker.upper()) and (not a.year or c.fiscal_year == a.year)
    ]
    if a.exact:
        hits = [c for c in chunks if a.exact.lower() in c.text.lower()][: a.n]
    else:
        bm = BM25Okapi([tokenize(c.embed_text) or ["_"] for c in chunks])
        scores = bm.get_scores(tokenize(a.phrase))
        hits = [chunks[i] for i in sorted(range(len(chunks)), key=lambda i: -scores[i])[: a.n]]
    print(f"index: {index_path(cfg)}  ({len(chunks)} chunks searched)\n")
    for c in hits:
        snip = " ".join(c.text.split())[:240]
        print(
            f"{c.doc_id}  page_index={c.page_index}  printed={c.page_label}  [{c.kind}]  {c.section}\n    {snip}\n"
        )


if __name__ == "__main__":
    main()
