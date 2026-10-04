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
- 2026-10-03: Real 10-K tables are borderless, so PyMuPDF `find_tables()` alone missed them (JPM capital table came out one number per line). The pymupdf parser now rebuilds rows from word positions, detects runs of numeric rows as tables and aligns cells to columns. Ruled tables still use `find_tables()`.
- 2026-10-03: Sections come from PDF bookmarks when present (JPM, Citi), else the heading regex (BAC has none).
- 2026-10-03: The Wells Fargo files first downloaded were the short 10-K wrapper (no financials). Need the Annual Report / Exhibit 13.
- 2026-10-03: Fixed on real data: decorative 1-2 row ruled boxes are ignored (they swallowed column headers), running page headers/footers are excluded from content (kept for page labels), BAC footers ("Bank of America 98") parse. Known limits: BAC two-column glossary pages interleave text; BAC sections are regex-only and sticky; Citi tables can show a trailing "$" in the wrong cell.
- 2026-10-03: Ingest parses one PDF per process; label index (hash embeddings, free) built over 6 good PDFs + 2 WFC wrapper files.
- 2026-10-04: Judge validation on 25 hand-labeled answers: correctness agreement 92% (kappa 0.70); faithfulness judge too lenient (all 25 'faithful' vs 20/25 human, kappa 0.00). Faithfulness column is an upper bound. Did not re-tune the judge on these labels.
