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


# Footers like "Bank of America 98" / "99 Bank of America": only trusted in the last 3 lines.
_TAIL_PATTERNS = [
    re.compile(r"^[A-Za-z][A-Za-z .,&'’/-]{2,40}\s+(\d{1,3})$"),
    re.compile(r"^(\d{1,3})\s+[A-Za-z][A-Za-z .,&'’/-]{2,40}$"),
]
HEADER_BAND = 40.0  # pt from top / bottom treated as running header / footer (not content)
FOOTER_BAND = 45.0


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
    tail = clean[-3:][::-1]
    for ln in tail + clean[:3]:
        for pat in _LABEL_PATTERNS:
            m = pat.search(ln)
            if m:
                return m.group(1)
    for ln in tail:
        for pat in _TAIL_PATTERNS:
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
# Real 10-K tables are mostly borderless: PyMuPDF's ruled-line detector misses them and plain text
# extraction emits one cell per line. So we (1) keep find_tables() for ruled tables, and (2) rebuild
# visual rows from word positions, then treat runs of numeric rows as tables.

CELL_GAP = 9.0  # horizontal gap (pt) that separates table cells
ROW_TOL = 3.5  # vertical tolerance (pt) for words on the same row
MIN_TABLE_ROWS = 3
_FOOTNOTE = re.compile(r"^\(?[a-z0-9]{1,2}\)(\([a-z0-9]{1,2}\))*$")
_NUMLIKE = re.compile(r"^[-–—]?\d[\d,]*(\.\d+)?$")


@dataclass
class _Row:
    y0: float
    y1: float
    cells: list[str]
    geo: list[tuple[float, float]] = field(default_factory=list)  # (x0, x1) per cell

    @property
    def text(self) -> str:
        return " ".join(self.cells)


def _bbox_intersects(a, b) -> bool:
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])


def _in_boxes(w, boxes) -> bool:
    cx, cy = (w[0] + w[2]) / 2, (w[1] + w[3]) / 2
    return any(bx0 <= cx <= bx1 and by0 <= cy <= by1 for bx0, by0, bx1, by1 in boxes)


def _is_num_cell(c: str) -> bool:
    c = re.sub(r"\([a-z0-9]{1,2}\)", "", c)
    c = c.replace("$", "").replace("%", "").replace("(", "").replace(")", "").strip()
    return bool(_NUMLIKE.match(c)) or c in {"—", "–", "-", "NM", "n/a", "NA"}


def _clean_cells(cells: list[tuple[str, float, float]]) -> list[tuple[str, float, float]]:
    out: list[tuple[str, float, float]] = []
    for c, x0, x1 in cells:
        c = c.strip()
        if not c or c == "$":
            continue
        if c == "%" and out:
            t, a, _ = out[-1]
            out[-1] = (t + "%", a, x1)
            continue
        if _FOOTNOTE.match(c):  # stray footnote marker in its own cell
            continue
        out.append((c, x0, x1))
    return out


def rows_from_words(words: list[tuple]) -> list[_Row]:
    """Cluster words (x0, y0, x1, y1, text, ...) into visual rows, then split rows into cells."""
    ws = sorted(words, key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    groups: list[list[tuple]] = []
    cur_y = 0.0
    for w in ws:
        yc = (w[1] + w[3]) / 2
        if groups and abs(yc - cur_y) <= ROW_TOL:
            groups[-1].append(w)
            cur_y = sum((x[1] + x[3]) / 2 for x in groups[-1]) / len(groups[-1])
        else:
            groups.append([w])
            cur_y = yc
    rows = []
    for g in groups:
        g.sort(key=lambda w: w[0])
        cells: list[tuple[str, float, float]] = []
        cur: list[tuple] = []
        for w in g:
            if cur and _FOOTNOTE.match(w[4]) and (_is_num_cell(cur[-1][4]) or cur[-1][4] == "%"):
                continue  # footnote marker right after a number, e.g. "14.6 % (c)"
            if cur and w[0] - cur[-1][2] > CELL_GAP:
                cells.append((" ".join(x[4] for x in cur), cur[0][0], cur[-1][2]))
                cur = []
            cur.append(w)
        cells.append((" ".join(x[4] for x in cur), cur[0][0], cur[-1][2]))
        cells = _clean_cells(cells)
        if cells:
            rows.append(
                _Row(
                    min(w[1] for w in g),
                    max(w[3] for w in g),
                    [c[0] for c in cells],
                    [(c[1], c[2]) for c in cells],
                )
            )
    rows.sort(key=lambda r: r.y0)
    return rows


def _is_table_row(r: _Row) -> bool:
    return len(r.cells) >= 2 and sum(_is_num_cell(c) for c in r.cells) >= 1 and not _is_num_cell(r.cells[0])


def _is_header_row(r: _Row) -> bool:
    return len(r.cells) >= 2 or bool(UNITS_RE.search(r.text))


def _is_light_row(r: _Row) -> bool:
    return len(r.cells) == 1 and len(r.text) < 100 and not r.text.rstrip().endswith(".")


def find_table_runs(rows: list[_Row]) -> list[tuple[int, int]]:
    """Return [(start, end)) row-index ranges that look like borderless tables."""
    flags = [_is_table_row(r) for r in rows]
    runs, i, n = [], 0, len(rows)
    while i < n:
        if not flags[i]:
            i += 1
            continue
        j, last = i + 1, i
        while j < n:
            if flags[j]:
                last = j
                j += 1
            elif _is_light_row(rows[j]) and j + 1 < n and flags[j + 1]:
                j += 1  # section label row inside a table, e.g. "Risk-based capital metrics:"
            else:
                break
        if sum(flags[i : last + 1]) >= MIN_TABLE_ROWS:
            start = i
            while start > 0 and i - start < 4 and _is_header_row(rows[start - 1]) and not flags[start - 1]:
                start -= 1
            if runs and start < runs[-1][1]:
                start = runs[-1][1]
            runs.append((start, last + 1))
        i = last + 1
    return runs


def column_anchors(rows: list[_Row], tol: float = 14.0) -> list[tuple[float, float]]:
    """x-intervals of the table's value columns, from the right edges of its numeric cells."""
    nums = sorted(
        (g for r in rows for c, g in zip(r.cells[1:], r.geo[1:]) if _is_num_cell(c)), key=lambda g: g[1]
    )
    clusters: list[list[tuple[float, float]]] = []
    for g in nums:
        if clusters and g[1] - clusters[-1][-1][1] <= tol:
            clusters[-1].append(g)
        else:
            clusters.append([g])
    return [(min(x[0] for x in c), max(x[1] for x in c)) for c in clusters]


def _assign_columns(row: _Row, anchors: list[tuple[float, float]]) -> list[str]:
    out = [""] * (len(anchors) + 1)  # column 0 = row label
    for k, (c, (x0, x1)) in enumerate(zip(row.cells, row.geo)):
        best, best_ov = -1, 0.0
        for a, (ax0, ax1) in enumerate(anchors):
            ov = min(x1, ax1) - max(x0, ax0)
            if ov > best_ov:
                best, best_ov = a, ov
        if best < 0:
            if k == 0 and (not anchors or x1 <= anchors[0][0] + 2):
                col = 0
            elif anchors:
                cx = (x0 + x1) / 2
                col = 1 + min(
                    range(len(anchors)), key=lambda a: abs(cx - (anchors[a][0] + anchors[a][1]) / 2)
                )
            else:
                col = min(k, len(out) - 1)
        else:
            col = best + 1
        out[col] = (out[col] + " " + c).strip() if out[col] else c
    return out


def rows_to_markdown(rows: list[_Row]) -> str:
    anchors = column_anchors(rows)
    grid = [_assign_columns(r, anchors) for r in rows] if anchors else [r.cells for r in rows]
    ncols = max(len(g) for g in grid)
    lines = []
    for k, cells in enumerate(grid):
        cells = cells + [""] * (ncols - len(cells))
        lines.append("| " + " | ".join(cells) + " |")
        if k == 0:
            lines.append("|" + " --- |" * ncols)
    return "\n".join(lines)


def paragraphs_from_rows(rows: list[_Row]) -> list[tuple[float, str]]:
    """Group consecutive rows into paragraphs, splitting on larger vertical gaps."""
    out: list[tuple[float, str]] = []
    cur: list[str] = []
    y_start = 0.0
    prev: _Row | None = None
    for r in rows:
        if prev is not None and (r.y0 - prev.y1) > 0.45 * max(prev.y1 - prev.y0, 1.0):
            out.append((y_start, "\n".join(cur)))
            cur = []
        if not cur:
            y_start = r.y0
        cur.append(r.text)
        prev = r
    if cur:
        out.append((y_start, "\n".join(cur)))
    return out


def paragraphs_from_words(words: list[tuple]) -> list[tuple[float, str]]:
    """Reading-order paragraphs from PyMuPDF words, following its block/line numbers (keeps columns apart)."""
    blocks: dict[int, dict[int, list[tuple]]] = {}
    for w in words:  # words arrive in block order
        blocks.setdefault(w[5], {}).setdefault(w[6], []).append(w)
    out: list[tuple[float, str]] = []
    for lines_by_no in blocks.values():
        lines = []
        for ws in lines_by_no.values():
            ws.sort(key=lambda w: w[0])
            lines.append(_Row(min(w[1] for w in ws), max(w[3] for w in ws), [" ".join(w[4] for w in ws)]))
        lines.sort(key=lambda r: r.y0)
        out.extend(paragraphs_from_rows(lines))
    return out


def _title_like(line: str) -> bool:
    line = line.strip()
    return (
        3 <= len(line) <= 90
        and line[0].isupper()
        and not line.endswith((".", ",", ";"))
        and not UNITS_RE.search(line)
        and sum(ch.isdigit() for ch in line) <= len(line) // 3
    )


def find_title(prev_texts: list[str]) -> str:
    """Nearest heading-like line above a table: prefer single-line blocks, else scan lines upward."""
    for t in reversed(prev_texts[-4:]):
        lines = [ln for ln in t.splitlines() if ln.strip()]
        if len(lines) == 1 and _title_like(lines[0]):
            return lines[0].strip()
    flat = [ln for t in prev_texts[-3:] for ln in t.splitlines() if ln.strip()]
    for ln in reversed(flat[-12:]):
        if _title_like(ln):
            return ln.strip()
    return ""


def toc_sections(doc) -> list[tuple[int, str]]:
    """(pdf_page, 'Parent > Child') from PDF bookmarks, in document order. Empty if none."""
    try:
        toc = doc.get_toc()
    except Exception:
        return []
    stack: list[tuple[int, str]] = []
    out: list[tuple[int, str]] = []
    for lvl, title, pg in toc:
        title = " ".join(str(title).split())
        while stack and stack[-1][0] >= lvl:
            stack.pop()
        stack.append((lvl, title))
        if pg and pg > 0:
            out.append((pg, " > ".join(t for _, t in stack[-2:])[:120]))
    return out


def parse_pymupdf(path: str) -> list[Page]:
    import pymupdf

    doc = pymupdf.open(path)
    toc = toc_sections(doc)
    toc_i, toc_cur = 0, ""
    pages: list[Page] = []
    section = ""
    for i, pg in enumerate(doc, start=1):
        try:
            # Decorative header boxes come back as 1-2 row "tables" and would swallow real column headers.
            ruled = [t for t in pg.find_tables().tables if t.row_count >= 3 and t.col_count >= 2]
        except Exception:  # find_tables can fail on odd pages; degrade to text only
            ruled = []
        boxes = [tuple(t.bbox) for t in ruled]
        words = [
            w for w in pg.get_text("words")
            if not any(_bbox_intersects(w[:4], tb) for tb in boxes)
        ]  # fmt: skip
        all_rows = rows_from_words(words)
        H = pg.rect.height
        rows = [r for r in all_rows if r.y1 >= HEADER_BAND and r.y0 <= H - FOOTER_BAND]
        runs = find_table_runs(rows)

        # Page-level section source: PDF bookmarks if available, else heading regex.
        while toc and toc_i < len(toc) and toc[toc_i][0] <= i:
            toc_cur = toc[toc_i][1]
            toc_i += 1
        all_lines = [r.text for r in rows]
        is_toc_page = count_item_headings(all_lines) > 5
        if toc:
            section = toc_cur

        # Table regions found by row reconstruction (borderless tables)
        run_boxes = [
            (
                min(g[0] for r in rows[x:y] for g in r.geo),
                rows[x].y0 - 1,
                max(g[1] for r in rows[x:y] for g in r.geo),
                rows[y - 1].y1 + 1,
            )
            for x, y in runs
        ]

        text_words = [
            w for w in words if w[3] >= HEADER_BAND and w[1] <= H - FOOTER_BAND and not _in_boxes(w, run_boxes)
        ]  # fmt: skip
        # Text keeps the PDF's block order (so two-column pages are not interleaved); tables are
        # slotted in by vertical position.
        seq: list[tuple[float, str, object]] = [(y, "text", t) for y, t in paragraphs_from_words(text_words)]
        tables: list[tuple[float, str, object]] = [(rows[x].y0, "btable", rows[x:y]) for x, y in runs]
        tables += [(t.bbox[1], "table", t) for t in ruled]
        for tb in sorted(tables, key=lambda it: it[0]):
            pos = next((k for k, it in enumerate(seq) if it[1] == "text" and it[0] > tb[0]), len(seq))
            seq.insert(pos, tb)
        items = seq

        blocks: list[Block] = []
        prev_texts: list[str] = []
        for y, kind, val in items:
            if kind == "text":
                txt = str(val)
                if not toc:
                    section = update_section(txt.splitlines(), section, is_toc_page)
                blocks.append(Block("text", txt, section=section, y=y))
                prev_texts.append(txt)
                continue
            if kind == "table":
                try:
                    md = val.to_markdown().strip()  # type: ignore[attr-defined]
                except Exception:
                    md = ""
                head = ""
            else:
                trs: list[_Row] = val  # type: ignore[assignment]
                md = rows_to_markdown(trs)
                head = " ".join(r.text for r in trs[:4])
            if not md:
                continue
            title = find_title(prev_texts)
            units = find_units(*(prev_texts[-2:][::-1]), head, md[:400])
            blocks.append(Block("table", md, section=section, title=title or section, units=units, y=y))

        lines = [r.text for r in all_rows]  # header/footer rows included: the printed page number lives there
        pages.append(Page(i, detect_page_label(lines, i), section, blocks))
    doc.close()
    return pages


def parse_pdf(path: str, parser: str) -> list[Page]:
    if parser == "pypdf":
        return parse_pypdf(path)
    if parser == "pymupdf":
        return parse_pymupdf(path)
    raise ValueError(f"unknown parser {parser}")
