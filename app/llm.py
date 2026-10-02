"""LLM client: openai | fake. JSON mode, retries, SQLite caching, cost tracking."""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass

from app.cache import Cache, CostLedger, cache_key
from app.chunking import count_tokens
from app.config import Config


@dataclass
class LLMResult:
    text: str
    parsed: dict | None
    tokens_in: int
    tokens_out: int
    usd: float
    cached: bool
    prompt_hash: str


def _parse_json(text: str) -> dict | None:
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
                return obj if isinstance(obj, dict) else None
            except json.JSONDecodeError:
                return None
        return None


def _split_excerpts(user: str) -> tuple[list[str], str]:
    body = user.split("EXCERPTS:\n", 1)[-1]
    body, _, tail = body.rpartition("\n\n")
    parts = re.split(r"(?m)^\[(\d+)\] ", body)
    excerpts = [parts[i + 1] for i in range(1, len(parts) - 1, 2)]
    return excerpts, tail


def fake_complete(system: str, user: str) -> str:
    """Deterministic stand-in so tests never hit the API."""
    excerpts, tail = _split_excerpts(user)
    if tail.startswith("CLAIM:"):
        claim = tail[6:].strip()
        nums = re.findall(r"\d[\d,\.]*", claim)
        if not excerpts:
            return json.dumps(
                {
                    "verdict": "NOT_ENOUGH_INFO",
                    "explanation": "no evidence",
                    "evidence": [],
                }
            )
        for n, ex in enumerate(excerpts, start=1):
            if nums and all(x.rstrip(".,") in ex for x in nums):
                return json.dumps(
                    {
                        "verdict": "SUPPORTED",
                        "explanation": "numbers found",
                        "evidence": [n],
                    }
                )
        return json.dumps(
            {"verdict": "NOT_ENOUGH_INFO", "explanation": "not found", "evidence": []}
        )
    question = tail.replace("QUESTION:", "").strip()
    years = [int(y) for y in re.findall(r"\b(20\d\d)\b", question)]
    if (
        not excerpts
        or any(y >= 2026 for y in years)
        or re.search(r"\bwill\b", question.lower())
    ):
        return json.dumps({"answer": None, "citations": [], "abstain": True})
    first = excerpts[0].split("\n", 1)[-1].strip().splitlines()[0][:200]
    return json.dumps({"answer": first, "citations": [1], "abstain": False})


class LLM:
    def __init__(self, cfg: Config, cache: Cache, ledger: CostLedger):
        self.cfg = cfg.llm
        self.cache = cache
        self.ledger = ledger
        self._client = None

    def complete_json(self, system: str, user: str, kind: str = "llm") -> LLMResult:
        c = self.cfg
        prompt_hash = hashlib.sha256((system + "\x00" + user).encode()).hexdigest()[:16]
        key = cache_key(
            f"llm:{c.provider}:{c.model}:{c.temperature}:{c.max_tokens}",
            system + "\x00" + user,
        )
        hit = self.cache.get(key)
        if hit is not None:  # cached calls cost $0
            return LLMResult(
                hit["text"],
                _parse_json(hit["text"]),
                hit["in"],
                hit["out"],
                0.0,
                True,
                prompt_hash,
            )

        if c.provider == "fake":
            text = fake_complete(system, user)
            tin, tout = count_tokens(system + user), count_tokens(text)
        else:
            text, tin, tout = self._openai(system, user)
        usd = (
            (tin * c.price_in + tout * c.price_out) / 1e6
            if c.provider == "openai"
            else 0.0
        )
        if usd:
            self.ledger.record(kind, c.model, tin, tout, usd)
        self.cache.set(key, {"text": text, "in": tin, "out": tout})
        return LLMResult(text, _parse_json(text), tin, tout, usd, False, prompt_hash)

    def _openai(self, system: str, user: str) -> tuple[str, int, int]:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI()
        last: Exception | None = None
        for attempt in range(4):
            try:
                resp = self._client.chat.completions.create(
                    model=self.cfg.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=self.cfg.temperature,
                    max_completion_tokens=self.cfg.max_tokens,
                    response_format={"type": "json_object"},
                )
                u = resp.usage
                return (
                    resp.choices[0].message.content or "",
                    u.prompt_tokens,
                    u.completion_tokens,
                )
            except Exception as e:  # rate limits, timeouts, 5xx
                last = e
                time.sleep(2**attempt)
        raise RuntimeError(f"LLM call failed after retries: {last}")
