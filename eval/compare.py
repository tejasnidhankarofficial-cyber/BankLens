"""python -m eval.compare -> eval/results/results.md + charts (png)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES = Path(__file__).parent / "results"
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442"]


def f(x, pct=False, nd=3):
    if x is None:
        return "–"
    return f"{x * 100:.1f}%" if pct else f"{x:.{nd}f}"


def load() -> list[dict]:
    rows = []
    for p in sorted(RES.glob("*.json")):
        d = json.loads(p.read_text())
        if "question_metrics" in d and not d.get("retrieval_only"):
            rows.append(d)
    return rows


def main_table(results: list[dict]) -> str:
    head = ("| Config | Hit@5 | Recall@5 | MRR | Answer correct | Faithfulness | Cite prec. | Abstain P | Abstain R "
            "| False abstain | Verify acc | Verify macro-F1 | Latency ms | $ / run |\n" + "|---" * 14 + "|\n")  # fmt: skip
    lines = []
    for r in results:
        q = r["question_metrics"]
        ov, ab = q.get("_overall", {}), q.get("_abstention", {})
        v = r.get("verify_metrics", {})
        lines.append(
            f"| {r['name']} | {f(ov.get('hit@5'))} | {f(ov.get('recall@5'))} | {f(ov.get('mrr'))} | "
            f"{f(ov.get('answer_correct'))} | {f(ov.get('faithfulness'))} | {f(ov.get('citation_precision'))} | "
            f"{f(ab.get('abstain_precision'))} | {f(ab.get('abstain_recall'))} | {f(ab.get('false_abstention_rate'))} | "
            f"{f(v.get('accuracy'))} | {f(v.get('macro_f1'))} | {f(ov.get('latency_ms'), nd=0)} | {r['cost_usd']:.3f} |"
        )
    return head + "\n".join(lines)


def by_type_table(results: list[dict], metric: str) -> str:
    types = sorted({t for r in results for t in r["question_metrics"] if not t.startswith("_")})
    out = f"| Config | {' | '.join(types)} |\n" + "|---" * (len(types) + 1) + "|\n"
    for r in results:
        cells = [f(r["question_metrics"].get(t, {}).get(metric)) for t in types]
        out += f"| {r['name']} | {' | '.join(cells)} |\n"
    return out


def verify_by_pert(results: list[dict]) -> str:
    perts = sorted({p for r in results for p in r.get("verify_metrics", {}).get("by_perturbation", {})})
    if not perts:
        return ""
    out = f"| Config | {' | '.join(perts)} |\n" + "|---" * (len(perts) + 1) + "|\n"
    for r in results:
        bp = r.get("verify_metrics", {}).get("by_perturbation", {})
        out += f"| {r['name']} | {' | '.join(f(bp.get(p, {}).get('accuracy')) for p in perts)} |\n"
    return out


def chart(results: list[dict], metrics: list[tuple[str, str]], fname: str, title: str) -> None:
    names = [r["name"] for r in results]
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(names)), 4))
    w = 0.8 / len(metrics)
    for i, (key, label) in enumerate(metrics):
        vals = [(r["question_metrics"].get("_overall", {}).get(key) or 0) for r in results]
        pos = [x + i * w for x in range(len(names))]
        ax.bar(pos, vals, w, label=label, color=OKABE_ITO[i % len(OKABE_ITO)])
        for x, v in zip(pos, vals):
            ax.text(x, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks([x + w * (len(metrics) - 1) / 2 for x in range(len(names))], names, rotation=20, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(RES / fname, dpi=160)
    plt.close(fig)


def chart_by_type(results: list[dict], metric: str, fname: str, title: str) -> None:
    types = sorted({t for r in results for t in r["question_metrics"] if not t.startswith("_")})
    fig, ax = plt.subplots(figsize=(max(6, 1.8 * len(types)), 4))
    w = 0.8 / max(len(results), 1)
    for i, r in enumerate(results):
        vals = [(r["question_metrics"].get(t, {}).get(metric) or 0) for t in types]
        ax.bar(
            [x + i * w for x in range(len(types))],
            vals,
            w,
            label=r["name"],
            color=OKABE_ITO[i % len(OKABE_ITO)],
        )
    ax.set_xticks([x + w * (len(results) - 1) / 2 for x in range(len(types))], types)
    ax.set_ylim(0, 1.1)
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(RES / fname, dpi=160)
    plt.close(fig)


def main() -> None:
    results = load()
    if not results:
        raise SystemExit(f"No result files in {RES}. Run eval.run_eval first.")
    md = ["# BankLens results\n", "## Overall\n", main_table(results), "\n## Answer correctness by question type\n",
          by_type_table(results, "answer_correct"), "\n## Hit@5 by question type\n", by_type_table(results, "hit@5"),
          "\n## Claim verification accuracy by perturbation\n", verify_by_pert(results),
          "\n_Small eval set: indicative, not statistically conclusive._\n"]  # fmt: skip
    (RES / "results.md").write_text("\n".join(md))
    chart(results, [("hit@5", "Hit@5"), ("mrr", "MRR"), ("answer_correct", "Answer correct")], "ablation.png",
          "Retrieval and answer quality by config")  # fmt: skip
    chart_by_type(results, "answer_correct", "by_type.png", "Answer correctness by question type")
    print("\n".join(md))
    print(f"\nwrote {RES / 'results.md'} and charts")


if __name__ == "__main__":
    main()
