"""Judge validation: hand-label ~25 answers, then compute Cohen's kappa vs the LLM judge.

  python -m eval.judge_agreement --export eval/results/05_rerank.json   # writes eval/human_labels.json template
  (fill in "human_correct" / "human_faithful" with true/false by hand)
  python -m eval.judge_agreement --result eval/results/05_rerank.json   # prints agreement
Faithfulness judge scores are fractions; they are binarized at 1.0 (fully supported) for agreement.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from eval.metrics import cohens_kappa

LABELS = Path(__file__).parent / "human_labels.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export")
    ap.add_argument("--result")
    ap.add_argument("--labels", default=str(LABELS))
    ap.add_argument("-n", type=int, default=25)
    a = ap.parse_args()

    if a.export:
        res = json.loads(Path(a.export).read_text())
        pool = [q for q in res["questions"] if q.get("answer") and q["type"] != "unanswerable"]
        random.Random(0).shuffle(pool)
        rows = [
            {"id": q["id"], "question": q["question"], "answer": q["answer"], "cited": q.get("cited"),
             "judge_correct": q.get("answer_correct"), "judge_faithfulness": q.get("faithfulness"),
             "human_correct": None, "human_faithful": None}
            for q in pool[: a.n]
        ]  # fmt: skip
        Path(a.labels).write_text(json.dumps(rows, indent=2))
        print(f"wrote {len(rows)} rows to {a.labels}. Label them by hand, then rerun with --result.")
        return

    rows = json.loads(Path(a.labels).read_text())
    for name, h_key, j_key, conv in [
        ("correctness", "human_correct", "judge_correct", lambda x: bool(x)),
        ("faithfulness", "human_faithful", "judge_faithfulness", lambda x: x == 1.0),
    ]:
        pairs = [
            (r[h_key], conv(r[j_key])) for r in rows if r.get(h_key) is not None and r.get(j_key) is not None
        ]
        if not pairs:
            print(f"{name}: no labeled rows yet")
            continue
        h, j = zip(*pairs)
        agree = sum(x == y for x, y in pairs) / len(pairs)
        print(f"{name}: n={len(pairs)} agreement={agree:.2f} kappa={cohens_kappa(list(h), list(j)):.2f}")


if __name__ == "__main__":
    main()
