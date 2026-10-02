from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from app.config import Config

STY = getSampleStyleSheet()
BANKS = {
    "jpm": (
        "JPMorgan Chase",
        "JPM",
        {"2025": ("15.7%", "58,000", "54,000"), "2024": ("15.7%", "54,000", "49,000")},
    ),
    "wfc": (
        "Wells Fargo",
        "WFC",
        {"2025": ("11.1%", "19,000", "19,100"), "2024": ("11.0%", "19,100", "18,000")},
    ),
}


def make_pdf(
    path: Path, bank: str, year: str, cet1: str, ni: str, ni_prev: str
) -> None:
    def footer(canvas, doc):
        n = doc.page
        label = {1: None, 2: f"{bank} Form 10-K 45", 3: "46"}.get(n)
        if label:
            canvas.setFont("Helvetica", 8)
            canvas.drawString(72, 30, label)

    prev = str(int(year) - 1)
    story = [Paragraph(f"{bank} Annual Report {year}", STY["Title"])]
    for i in range(1, 8):
        story.append(
            Paragraph(f"Item {i}. Placeholder table of contents entry", STY["Normal"])
        )
    story.append(PageBreak())
    story += [
        Paragraph("Item 7. Management's Discussion and Analysis", STY["Heading2"]),
        Paragraph("Capital Risk Management", STY["Heading3"]),
        Paragraph(
            f"The Common Equity Tier 1 (CET1) capital ratio was {cet1} at December 31, {year}.",
            STY["Normal"],
        ),
        Spacer(1, 12),
        Paragraph(
            "Liquidity risk is managed centrally by corporate treasury.", STY["Normal"]
        ),
        PageBreak(),
        Paragraph("Consolidated Results of Operations", STY["Heading3"]),
        Paragraph("(in millions)", STY["Normal"]),
        Spacer(1, 6),
    ]
    t = Table(
        [
            ["Metric", year, prev],
            ["Net income", ni, ni_prev],
            ["Total revenue", "180,000", "170,000"],
        ]
    )
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.8, colors.black)]))
    story.append(t)
    SimpleDocTemplate(str(path), pagesize=letter).build(
        story, onFirstPage=footer, onLaterPages=footer
    )


@pytest.fixture(scope="session")
def fake_corpus(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("corpus")
    raw = root / "raw"
    raw.mkdir()
    docs = []
    for tk, (bank, ticker, years) in BANKS.items():
        for y, (cet1, ni, prev) in years.items():
            make_pdf(raw / f"{tk}_fy{y}.pdf", bank, y, cet1, ni, prev)
            docs.append(
                {
                    "doc_id": f"{tk}_fy{y}",
                    "file": f"{tk}_fy{y}.pdf",
                    "bank": bank,
                    "ticker": ticker,
                    "fiscal_year": int(y),
                    "form": "10-K",
                }
            )
    (root / "manifest.yaml").write_text(yaml.safe_dump({"documents": docs}))
    return root


def make_cfg(root: Path, **over) -> Config:
    base = dict(
        name="test",
        parser="pymupdf",
        chunking={"strategy": "section_table", "chunk_size": 200, "overlap": 20},
        embedding={"provider": "hash"},
        retrieval={
            "mode": "hybrid",
            "top_k": 4,
            "candidate_k": 20,
            "metadata_filter": True,
        },
        rerank={"enabled": False},
        llm={"provider": "fake"},
        manifest_path=str(root / "manifest.yaml"),
        raw_dir=str(root / "raw"),
        index_dir=str(root / "indexes"),
        cache_path=str(root / "cache.sqlite"),
        log_dir=str(root / "logs"),
    )
    base.update(over)
    return Config(**base)


@pytest.fixture
def cfg(fake_corpus) -> Config:
    return make_cfg(fake_corpus)
