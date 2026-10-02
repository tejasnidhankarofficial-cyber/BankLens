"""LLM-as-judge for answer correctness and faithfulness. Calls are cached like any other LLM call."""

from __future__ import annotations

FAITH_SYSTEM = """You audit an answer for faithfulness to source excerpts.
Split the ANSWER into atomic factual claims. For each claim decide whether the EXCERPTS support it
(the claim must follow from the excerpt text, including numbers, units and years; outside knowledge does not count).
Return JSON: {"claims": [{"claim": string, "supported": boolean}]}"""

CORRECT_SYSTEM = """You grade a model answer against a reference answer for a question about a bank annual report.
The model answer is correct if it conveys the same facts as the reference (numbers equal after unit conversion,
same year and entity). Extra correct detail is fine. Return JSON: {"correct": boolean, "reason": string}"""


class Judge:
    def __init__(self, llm):
        self.llm = llm

    def faithfulness(self, answer: str, excerpts: list[str]) -> float | None:
        if not answer or not excerpts:
            return None
        user = (
            "EXCERPTS:\n"
            + "\n\n".join(f"[{i}] {e}" for i, e in enumerate(excerpts, 1))
            + f"\n\nANSWER: {answer}"
        )
        p = self.llm.complete_json(FAITH_SYSTEM, user, kind="judge").parsed or {}
        claims = [c for c in p.get("claims", []) if isinstance(c, dict) and "supported" in c]
        if not claims:
            return None
        return sum(bool(c["supported"]) for c in claims) / len(claims)

    def correct(self, question: str, reference: str, answer: str) -> bool | None:
        if not answer:
            return False
        user = f"QUESTION: {question}\nREFERENCE ANSWER: {reference}\nMODEL ANSWER: {answer}"
        p = self.llm.complete_json(CORRECT_SYSTEM, user, kind="judge").parsed or {}
        return bool(p["correct"]) if "correct" in p else None
