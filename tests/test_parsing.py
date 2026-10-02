from app.parsing import UNITS_RE, detect_page_label, heading_of, parse_pdf


def test_label_detection():
    assert detect_page_label(["Foo", "JPMorgan Form 10-K 45"], 99) == "45"
    assert detect_page_label(["header", "body", "46"], 99) == "46"
    assert detect_page_label(["no number here"], 7) == "7"


def test_units_regex():
    assert UNITS_RE.search("(in millions, except ratios)")
    assert UNITS_RE.search("($ in billions)")
    assert not UNITS_RE.search("(in percent)")


def test_headings():
    assert heading_of("Item 7. Management's Discussion") == "Item 7. Management's Discussion"
    assert heading_of("capital risk management") == "Capital Risk Management"
    assert heading_of("The company has capital risk management practices and more words") is None


def test_pymupdf_tables_labels_sections(fake_corpus):
    pages = parse_pdf(str(fake_corpus / "raw" / "jpm_fy2025.pdf"), "pymupdf")
    assert [p.page_index for p in pages] == [1, 2, 3]
    assert pages[1].page_label == "45" and pages[2].page_label == "46"
    # TOC page (7 Item headings) must not set a section
    assert pages[0].section == ""
    assert pages[1].section == "Capital Risk Management"
    tables = [b for b in pages[2].blocks if b.kind == "table"]
    assert len(tables) == 1
    t = tables[0]
    assert "Net income" in t.text and "58,000" in t.text
    assert t.units == "in millions"
    assert "Consolidated Results of Operations" in t.title or t.title == "(in millions)" or t.title
    assert t.section == "Consolidated Results of Operations"


def test_pypdf_baseline_has_no_tables(fake_corpus):
    pages = parse_pdf(str(fake_corpus / "raw" / "jpm_fy2025.pdf"), "pypdf")
    assert all(b.kind == "text" for p in pages for b in p.blocks)
    assert "58,000" in pages[2].text
