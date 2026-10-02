from app.chunking import chunk_document, count_tokens
from app.config import ChunkingConfig, DocumentInfo
from app.parsing import Block, Page, parse_pdf

DOC = DocumentInfo(
    doc_id="x_fy2025", file="x.pdf", bank="X Bank", ticker="X", fiscal_year=2025
)


def test_chunks_stay_on_page_and_ids(fake_corpus):
    pages = parse_pdf(str(fake_corpus / "raw" / "jpm_fy2025.pdf"), "pymupdf")
    for strat in ("fixed", "section", "section_table"):
        chunks = chunk_document(
            pages, DOC, ChunkingConfig(strategy=strat, chunk_size=60, overlap=10)
        )
        assert chunks
        for c in chunks:
            assert c.chunk_id.startswith(f"x_fy2025:p{c.page_index}:c")
        assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_section_table_isolates_tables(fake_corpus):
    pages = parse_pdf(str(fake_corpus / "raw" / "jpm_fy2025.pdf"), "pymupdf")
    chunks = chunk_document(
        pages, DOC, ChunkingConfig(strategy="section_table", chunk_size=300)
    )
    tables = [c for c in chunks if c.kind == "table"]
    assert len(tables) == 1
    assert tables[0].text.startswith("Table")
    assert "in millions" in tables[0].text and "Net income" in tables[0].text


def test_table_split_repeats_header():
    rows = "\n".join(f"| Row {i} | {i * 1000} | {i * 900} |" for i in range(80))
    md = "| Metric | 2025 | 2024 |\n| --- | --- | --- |\n" + rows
    page = Page(
        1, "1", "S", [Block("table", md, section="S", title="Big", units="in millions")]
    )
    chunks = chunk_document(
        [page], DOC, ChunkingConfig(strategy="section_table", chunk_size=120)
    )
    assert len(chunks) > 1
    for c in chunks:
        assert "| Metric | 2025 | 2024 |" in c.text
        assert count_tokens(c.text) <= 160


def test_contextual_header():
    page = Page(
        1,
        "87",
        "Capital Risk Management",
        [Block("text", "CET1 was 15.7%.", section="Capital Risk Management")],
    )
    c = chunk_document(
        [page], DOC, ChunkingConfig(strategy="section", contextual_header=True)
    )[0]
    assert c.embed_text.startswith(
        "[X Bank | FY2025 | Capital Risk Management | p. 87]\n"
    )
    assert c.text == "CET1 was 15.7%."


def test_fixed_windows_overlap():
    page = Page(1, "1", "", [Block("text", " ".join(f"w{i}" for i in range(500)))])
    chunks = chunk_document(
        [page], DOC, ChunkingConfig(strategy="fixed", chunk_size=100, overlap=20)
    )
    assert len(chunks) >= 5
