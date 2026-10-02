# BankLens — working notes

Full spec lives in `banklens-spec.md` (keep it in the repo root if you want it auto-read). Key conventions:

- Python 3.11, type hints, Pydantic v2, `ruff` (config in pyproject.toml). Venv: `.venv/`.
- Never hit the OpenAI API in tests: use `embedding.provider: hash` and `llm.provider: fake`.
- All embeddings and LLM calls go through the SQLite cache and cost ledger.
- Git-ignored: data/raw/, indexes/, cache/, logs/, .env. Commit after each working step.
- Eval results store full config, git hash, timestamp and cost.

## Decision log
- 2026-10-02: Domain = Big Four US bank 10-Ks (FY2024, FY2025); added claim-verification mode.
- 2026-10-02: Scope cut to 3 days: 40 questions + 30 claims; contextual-header experiment is a stretch goal.
- 2026-10-02: `page_index` is the 1-based PDF page number (matches PDF viewers and eval labels).
- 2026-10-02: Seeded eval with 10 unanswerable questions and 8 rumor-style NOT_ENOUGH_INFO claims; the rest need hand labeling against the real PDFs.
