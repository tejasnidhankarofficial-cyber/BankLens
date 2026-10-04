"""Interactive hand-labeling for the judge-validation check.

  python -m eval.label_cli            # label the rows in eval/human_labels.json (resumes where you stopped)

For each row you see the question, the reference answer, the model's answer and the cited passages.
You answer two yes/no questions. The judge's own verdict is hidden so it cannot bias you.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

from app.config import ROOT, load_config
from app.index import index_path

LABELS = ROOT / "eval" / "human_labels.json"


def ask_yn(prompt: str) -> bool | None:
    """True/False, or None to skip. 'q' raises KeyboardInterrupt so the caller can save and stop."""
    while True:
        a = input(prompt + " [y/n/s=skip/q=quit]: ").strip().lower()
        if a in ("y", "yes"):
            return True
        if a in ("n", "no"):
            return False
        if a in ("s", "skip", ""):
            return None
        if a in ("q", "quit"):
            raise KeyboardInterrupt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(LABELS))
    ap.add_argument(
        "--config", default="configs/05_rerank.yaml", help="config whose index holds the cited chunks"
    )
    a = ap.parse_args()

    rows = json.loads(Path(a.labels).read_text())
    cfg = load_config(a.config)
    with open(index_path(cfg) / "chunks.jsonl") as f:
        chunks = {c["chunk_id"]: c for c in map(json.loads, f)}
    questions = {q["id"]: q for q in json.loads((ROOT / "eval" / "questions.json").read_text())}

    todo = [r for r in rows if r.get("human_correct") is None or r.get("human_faithful") is None]
    print(
        f"{len(rows) - len(todo)} of {len(rows)} already labeled. {len(todo)} to go. Progress is saved after every row.\n"
    )
    try:
        for n, r in enumerate(todo, 1):
            ref = questions.get(r["id"], {}).get("expected", {}).get("answer")
            print("=" * 100)
            print(f"[{n}/{len(todo)}] {r['id']}")
            print(f"QUESTION:         {r['question']}")
            print(f"REFERENCE ANSWER: {ref}")
            print(f"MODEL ANSWER:     {r['answer']}\n")
            print("CITED PASSAGES (what the model said it relied on):")
            for cid in r.get("cited") or []:
                c = chunks.get(cid)
                if c:
                    body = " ".join(c["text"].split())[:900]
                    print(f"  - {c['bank']} FY{c['fiscal_year']}, PDF page {c['page_index']}:")
                    print(textwrap.indent(textwrap.fill(body, 96), "      "))
            print()
            r["human_correct"] = ask_yn("1) CORRECT: does the model answer match the reference answer?")
            r["human_faithful"] = ask_yn(
                "2) FAITHFUL: is EVERY statement in the model answer supported by the cited passages above?"
            )
            Path(a.labels).write_text(json.dumps(rows, indent=2))
    except (KeyboardInterrupt, EOFError):
        Path(a.labels).write_text(json.dumps(rows, indent=2))
        print("\nSaved. Run again to continue.")
        return
    print("\nAll rows labeled. Now run: python -m eval.judge_agreement")


if __name__ == "__main__":
    main()
