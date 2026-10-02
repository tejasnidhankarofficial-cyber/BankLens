"""PDF parsing: pypdf (plain-text baseline) and pymupdf (text blocks + tables as markdown).

Conventions:
  * ``page_index`` is the 1-based PDF page number (what a PDF viewer shows).
    The viewer / page-image endpoint uses it; eval labels use it too.
  * ``page_label`` is the printed page number if we can detect it, else str(page_index).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

UNITS_RE = re.compile(r"\((?:dollars |\$ )?in (millions|billions|thousands)[^)]*\)", re.IGNORECASE)
ITEM_RE = re.compile(r"^Item\s+\d+[A-C]?\.?", re.IGNORECASE)

# Bank MD&A / financial-statement headings that update the current section.
MDA_HEADINGS = [
    "Management's Discussion and Analysis",
    "Executive Overview",
    "Financial Highlights",
    "Selected Financial Data",
    "Consolidated Results of Operations",
    "Business Segment Results",
    "Net Interest Income",
    "Noninterest Income",
    "Noninterest Expense",
    "Income Taxes",
    "Capital Risk Management",
    "Capital Management",
    "Credit Risk Management",
    "Credit Risk",
    "Liquidity Risk Management",
    "Liquidity Risk",
    "Market Risk Management",
    "Market Risk",
    "Operational Risk Management",
    "Allowance for Credit Losses",
    "Risk Factors",
    "Consolidated Statements of Income",
    "Consolidated Balance Sheets",
    "Consolidated Statements of Cash Flows",
    "Notes to Consolidated Financial Statements",
]
_MDA_LOOKUP = {h.lower().replace("’", "'"): h for h in MDA_HEADINGS}

_LABEL_PATTERNS = [
    re.compile(r"Form\s+10-K\s*[|·\-–]?\s*(\d{1,3})\s*$", re.IGNORECASE),
    re.compile(r"^(\d{1,3})\s*[|·\-–]?\s*.*Form\s+10-K", re.IGNORECASE),
    re.compile(r"Annual Report\s*[|·\-–]?\s*(\d{1,3})\s*$", re.IGNORECASE),
    re.compile(r"^(\d{1,3})\s*[|·\-–]?\s*.*Annual Report", re.IGNORECASE),
    re.compile(r"^(\d{1,3})$"),
    re.compile(r"^[-–]\s*(\d{1,3})\s*[-–]$"),
]


@dataclass
class Block:
    kind: str  # "text" | "table"
    text: str
    section: str = ""
    title: str = ""  # tables only
    units: str = ""  # tables only
    y: float = 0.0


@dataclass
class Page:
    page_index: int  # 1-based PDF page number
    page_label: str
    section: str  # section active at the end of the page
    blocks: list[Block] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks)


def detect_page_label(lines: list[str], page_index: int) -> str:
    """Look in the first/last 3 non-empty lines for a standalone number or a
    '... Form 10-K 45' style footer. Fall back to the PDF index."""
    clean = [ln.strip() for ln in lines if ln.strip()]
    candidates = clean[-3:][::-1] + clean[:3]
    for ln in candidates:
        for pat in _LABEL_PATTERNS:
            m = pat.search(ln)
            if m:
                return m.group(1)
    return str(page_index)


def heading_of(line: str) -> str | None:
    s = " ".join(line.split())
    if not s or len(s) > 120:
        return None
    if ITEM_RE.match(s):
        return s[:80]
    key = s.lower().replace("’", "'").rstrip(".:")
    return _MDA_LOOKUP.get(key)


def count_item_headings(lines: list[str]) -> int:
    return sum(1 for ln in lines if ITEM_RE.match(ln.strip()))


def update_section(lines: list[str], current: str, toc: bool) -> str:
    """Return the section after reading ``lines``. TOC pages never update it."""
    if toc:
        return current
    for ln in lines:
        h = heading_of(ln)
        if h:
            current = h
    return current


def find_units(*texts: str) -> str:
    for t in texts:
        m = UNITS_RE.search(t or "")
        if m:
            return f"in {m.group(1).lower()}"
    return ""


# ---------------------------------------------------------------- pypdf


def parse_pypdf(path: str) -> list[Page]:
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages: list[Page] = []
    section = ""
    for i, pg in enumerate(reader.pages, start=1):
        text = pg.extract_text() or ""
        lines = text.splitlines()
        toc = count_item_headings(lines) > 5
        section = update_section(lines, section, toc)
        blocks = [Block("text", text.strip(), section=section)] if text.strip() else []
        pages.append(Page(i, detect_page_label(lines, i), section, blocks))
    return pages


# ---------------------------------------------------------------- pymupdf


def _bbox_intersects(a, b) -> bool:
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])


def parse_pymupdf(path: str) -> list[Page]:
    import fitz  # PyMuPDF

    doc = fitz.open(path)
    pages: list[Page] = []
    section = ""
    for i, pg in enumerate(doc, start=1):
        try:
            tables = list(pg.find_tables().tables)
        except Exception:  # find_tables can fail on odd pages; degrade to text only
            tables = []
        table_boxes = [tuple(t.bbox) for t in tables]

        raw = pg.get_text("blocks")  # (x0,y0,x1,y1,text,block_no,block_type)
        text_blocks = [
            (b[1], b[0], b[4].strip(), b)
            for b in raw
            if b[6] == 0 and b[4].strip() and not any(_bbox_intersects(b[:4], tb) for tb in table_boxes)
        ]
        # Items to be ordered by vertical position: ("text", y, text) / ("table", y, table)
        items: list[tuple[float, str, object]] = [(y, "text", txt) for y, _x, txt, _ in text_blocks]
        for t in tables:
            items.append((t.bbox[1], "table", t))
        items.sort(key=lambda it: it[0])

        all_lines = [ln for _, k, v in items if k == "text" for ln in str(v).splitlines()]
        toc = count_item_headings(all_lines) > 5

        blocks: list[Block] = []
        prev_texts: list[str] = []
        for y, kind, val in items:
            if kind == "text":
                txt = str(val)
                section = update_section(txt.splitlines(), section, toc)
                blocks.append(Block("text", txt, section=section, y=y))
                prev_texts.append(txt)
            else:
                try:
                    md = val.to_markdown().strip()  # type: ignore[attr-defined]
                except Exception:
                    md = ""
                if not md:
                    continue
                title = ""
                if prev_texts:
                    last = prev_texts[-1].strip().splitlines()
                    title = last[-1].strip() if last else ""
                    if len(title) > 150 or UNITS_RE.fullmatch(title.strip()):
                        title = last[0].strip()[:150] if last else ""
                units = find_units(*(prev_texts[-2:][::-1]), md[:400])
                blocks.append(Block("table", md, section=section, title=title, units=units, y=y))

        lines = [ln for ln in pg.get_text("text").splitlines()]
        pages.append(Page(i, detect_page_label(lines, i), section, blocks))
    doc.close()
    return pages


def parse_pdf(path: str, parser: str) -> list[Page]:
    if parser == "pypdf":
        return parse_pypdf(path)
    if parser == "pymupdf":
        return parse_pymupdf(path)
    raise ValueError(f"unknown parser {parser}")
