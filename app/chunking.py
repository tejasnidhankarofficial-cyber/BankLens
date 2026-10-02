"""Chunking strategies. Chunks never cross page boundaries so citations stay exact
(trade-off: facts split across pages are retrieved as two separate chunks)."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from app.config import ChunkingConfig, DocumentInfo
from app.parsing import Block, Page

_ENC = None


def _encoder():
    global _ENC
    if _ENC is None:
        try:
            import tiktoken

            _ENC = tiktoken.get_encoding("cl100k_base")
        except Exception:  # offline fallback: ~1 token per word
            _ENC = False
    return _ENC


def count_tokens(text: str) -> int:
    enc = _encoder()
    return len(enc.encode(text)) if enc else len(text.split())


def _split_tokens(text: str, size: int, overlap: int) -> list[str]:
    """Sliding token window."""
    enc = _encoder()
    toks = enc.encode(text) if enc else text.split()
    if len(toks) <= size:
        return [text.strip()] if text.strip() else []
    step = max(1, size - overlap)
    out = []
    for start in range(0, len(toks), step):
        window = toks[start : start + size]
        out.append((enc.decode(window) if enc else " ".join(window)).strip())
        if start + size >= len(toks):
            break
    return [o for o in out if o]


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    bank: str
    ticker: str
    fiscal_year: int
    page_index: int
    page_label: str
    section: str
    kind: str  # text | table
    text: str  # raw text shown as citation snippet
    embed_text: str  # text actually embedded / BM25-indexed (may carry a header)

    def metadata(self) -> dict:
        d = asdict(self)
        d.pop("text")
        d.pop("embed_text")
        return d


def context_header(doc: DocumentInfo, section: str, page_label: str) -> str:
    return (
        f"[{doc.bank} | FY{doc.fiscal_year} | {section or 'n/a'} | p. {page_label}]\n"
    )


def _paragraphs(text: str) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return paras


def _split_table(md: str, prefix: str, size: int) -> list[str]:
    """Split a markdown table by rows, repeating the header row (+ separator)."""
    lines = [ln for ln in md.splitlines() if ln.strip()]
    if count_tokens(prefix + md) <= size or len(lines) <= 3:
        return [prefix + md]
    header = lines[:2]
    rows = lines[2:]
    base = count_tokens(prefix + "\n".join(header))
    parts, cur, cur_tok = [], [], base
    for row in rows:
        t = count_tokens(row) + 1
        if cur and cur_tok + t > size:
            parts.append(prefix + "\n".join(header + cur))
            cur, cur_tok = [], base
        cur.append(row)
        cur_tok += t
    if cur:
        parts.append(prefix + "\n".join(header + cur))
    return parts


def _table_prefix(b: Block) -> str:
    head = f"Table: {b.title}" if b.title else "Table"
    if b.units:
        head += f" ({b.units})"
    return head + "\n"


def _pack(paras: list[tuple[str, str]], size: int) -> list[tuple[str, str]]:
    """Pack (text, section) paragraphs into <=size-token chunks. Section = first para's."""
    out: list[tuple[str, str]] = []
    cur: list[str] = []
    cur_tok = 0
    cur_sec = ""
    for text, sec in paras:
        t = count_tokens(text)
        if t > size:  # oversized paragraph: flush, then window-split it
            if cur:
                out.append(("\n\n".join(cur), cur_sec))
                cur, cur_tok = [], 0
            for piece in _split_tokens(text, size, 0):
                out.append((piece, sec))
            continue
        if cur and cur_tok + t > size:
            out.append(("\n\n".join(cur), cur_sec))
            cur, cur_tok = [], 0
        if not cur:
            cur_sec = sec
        cur.append(text)
        cur_tok += t
    if cur:
        out.append(("\n\n".join(cur), cur_sec))
    return out


def chunk_page(page: Page, doc: DocumentInfo, cfg: ChunkingConfig) -> list[Chunk]:
    raw: list[tuple[str, str, str]] = []  # (text, section, kind)

    if cfg.strategy == "fixed":
        for piece in _split_tokens(page.text, cfg.chunk_size, cfg.overlap):
            raw.append((piece, page.section, "text"))
    else:
        pending: list[tuple[str, str]] = []

        def flush():
            for text, sec in _pack(pending, cfg.chunk_size):
                raw.append((text, sec, "text"))
            pending.clear()

        for b in page.blocks:
            if b.kind == "table" and cfg.strategy == "section_table":
                flush()
                for part in _split_table(b.text, _table_prefix(b), cfg.chunk_size):
                    raw.append((part, b.section, "table"))
            else:
                for p in _paragraphs(b.text) or (
                    [b.text.strip()] if b.text.strip() else []
                ):
                    pending.append((p, b.section))
        flush()

    chunks = []
    for i, (text, section, kind) in enumerate(raw):
        header = (
            context_header(doc, section, page.page_label)
            if cfg.contextual_header
            else ""
        )
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}:p{page.page_index}:c{i}",
                doc_id=doc.doc_id,
                bank=doc.bank,
                ticker=doc.ticker,
                fiscal_year=doc.fiscal_year,
                page_index=page.page_index,
                page_label=page.page_label,
                section=section,
                kind=kind,
                text=text,
                embed_text=header + text,
            )
        )
    return chunks


def chunk_document(
    pages: list[Page], doc: DocumentInfo, cfg: ChunkingConfig
) -> list[Chunk]:
    out: list[Chunk] = []
    for p in pages:
        out.extend(chunk_page(p, doc, cfg))
    return out
