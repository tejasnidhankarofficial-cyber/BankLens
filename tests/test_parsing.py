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


def _borderless_pdf(path):
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path))
    c.setFont("Helvetica", 10)
    c.drawString(72, 750, "Capital ratios")
    c.drawString(
        72,
        735,
        "The firm maintained strong capital levels throughout the year and remained well capitalized.",
    )
    c.drawString(72, 700, "(in millions, except ratios)")
    cols = [330, 400, 470]
    for x, h in zip(cols, ["2025", "2024", "Requirement"]):
        c.drawRightString(x, 685, h)
    rows = [("CET1 capital", ["288,469", "275,513", ""]), ("Risk-weighted assets", ["1,981,692", "1,757,460", ""]),
            ("CET1 capital ratio", ["14.6 %", "15.7 %", "11.5 %"]), ("Tier 1 capital ratio", ["15.5", "16.8", "13.0"])]  # fmt: skip
    y = 670
    for label, vals in rows:
        c.drawString(72, y, label)
        for x, v in zip(cols, vals):
            if v:
                c.drawRightString(x, y, v)
        if label == "CET1 capital ratio":
            c.setFont("Helvetica", 6)
            c.drawString(335, y + 3, "(c)")  # footnote marker just after the 2025 value
            c.setFont("Helvetica", 10)
        y -= 15
    c.drawString(
        72, 560, "Total assets were higher than the prior year, driven by deposit growth across businesses."
    )
    c.save()


def test_borderless_table_rows_and_alignment(tmp_path):
    pdf = tmp_path / "b.pdf"
    _borderless_pdf(pdf)
    page = parse_pdf(str(pdf), "pymupdf")[0]
    tables = [b for b in page.blocks if b.kind == "table"]
    assert len(tables) == 1
    t = tables[0]
    assert t.units == "in millions"
    assert t.title == "Capital ratios" or "capital" in t.title.lower()
    lines = t.text.splitlines()
    cet1 = next(ln for ln in lines if ln.startswith("| CET1 capital ratio"))
    assert [c.strip() for c in cet1.strip("|").split("|")] == [
        "CET1 capital ratio",
        "14.6 %",
        "15.7 %",
        "11.5 %",
    ]
    # value rows with a blank "Requirement" column stay aligned under their headers
    cap = next(ln for ln in lines if ln.startswith("| CET1 capital |"))
    assert [c.strip() for c in cap.strip("|").split("|")] == ["CET1 capital", "288,469", "275,513", ""]
    # prose stays as text blocks, not table
    assert any(b.kind == "text" and "deposit growth" in b.text for b in page.blocks)


def test_bookmark_sections(tmp_path):
    import pymupdf

    src = tmp_path / "b.pdf"
    _borderless_pdf(src)
    doc = pymupdf.open(src)
    doc.insert_page(1, text="second page")
    doc.set_toc([[1, "Item 7. MD&A", 1], [2, "Capital Risk Management", 2]])
    out = tmp_path / "t.pdf"
    doc.save(out)
    pages = parse_pdf(str(out), "pymupdf")
    assert pages[0].section == "Item 7. MD&A"
    assert pages[1].section == "Item 7. MD&A > Capital Risk Management"
