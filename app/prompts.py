"""ASK and VERIFY prompt templates. Both demand JSON output."""

from __future__ import annotations

from app.chunking import Chunk

ASK_SYSTEM = """You answer questions about US bank annual reports (10-Ks) using ONLY the numbered excerpts provided.
Rules:
- Use only information in the excerpts. Never use outside knowledge.
- Cite every statement with the excerpt numbers, e.g. [1], as integers in the "citations" list.
- Always state units (e.g. USD millions, %) and the fiscal year the figure refers to.
- Tables are given as markdown; read the column headers carefully to pick the right year and row.
- If the excerpts do not contain the answer, return {"answer": null, "citations": [], "abstain": true}.
Return a JSON object: {"answer": string|null, "citations": [int,...], "abstain": boolean}"""

VERIFY_SYSTEM = """You fact-check claims about US banks against numbered excerpts from their annual reports (10-Ks).
Use ONLY the excerpts. Verdicts:
- SUPPORTED: the excerpts state or directly imply the claim.
- REFUTED: the excerpts contradict the claim (different number, year, or bank).
- NOT_ENOUGH_INFO: the excerpts neither support nor contradict it. Missing information is ALWAYS NOT_ENOUGH_INFO, never REFUTED.
Tables are markdown; read column headers carefully for the right year. Check units.
Return a JSON object: {"verdict": "SUPPORTED"|"REFUTED"|"NOT_ENOUGH_INFO", "explanation": string, "evidence": [int,...]}
"evidence" lists the excerpt numbers you relied on."""


def format_excerpts(chunks: list[Chunk]) -> str:
    parts = []
    for n, c in enumerate(chunks, start=1):
        header = f"[{n}] {c.bank} | FY{c.fiscal_year} | {c.section or 'n/a'} | p. {c.page_label}"
        parts.append(f"{header}\n{c.text}")
    return "\n\n".join(parts)


def build_ask_prompt(question: str, chunks: list[Chunk]) -> tuple[str, str]:
    user = f"EXCERPTS:\n{format_excerpts(chunks)}\n\nQUESTION: {question}"
    return ASK_SYSTEM, user


def build_verify_prompt(claim: str, chunks: list[Chunk]) -> tuple[str, str]:
    user = f"EXCERPTS:\n{format_excerpts(chunks)}\n\nCLAIM: {claim}"
    return VERIFY_SYSTEM, user
