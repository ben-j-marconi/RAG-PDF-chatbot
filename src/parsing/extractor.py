"""
parsing/extractor.py — Page-level text and table extraction.

Uses pdfplumber to extract:
  - Raw text per page (preserving layout where possible)
  - Tables as structured row/column data

pdfplumber was chosen over PyPDF2/pypdf because it has the best table
extraction for text-native PDFs out of the box. The table extraction
uses spatial analysis of the PDF's drawing primitives, not heuristics.

Design rationale:
  Standard RAG loaders call extract_text() and call it done. We extract
  tables SEPARATELY from text because if you concatenate table rows into
  a flat string, the semantic structure collapses and vector search can't
  find numeric facts reliably. Tables get their own chunk type.
"""

from dataclasses import dataclass, field
from typing import Optional

import pdfplumber


@dataclass
class RawTableBlock:
    """A table extracted from a PDF page, as a list of rows (list of cells)."""
    rows: list[list[Optional[str]]]
    bbox: Optional[tuple] = None     # (x0, y0, x1, y1) for provenance


@dataclass
class RawPage:
    """Raw extraction output for a single PDF page."""
    page_num: int                    # 1-indexed
    raw_text: str                    # Full text including non-table content
    tables: list[RawTableBlock] = field(default_factory=list)
    width: float = 0.0
    height: float = 0.0


def extract_pages(filepath: str) -> list[RawPage]:
    """
    Extract all pages from a PDF file.
    Returns a list of RawPage objects, one per page.
    """
    pages: list[RawPage] = []

    with pdfplumber.open(filepath) as pdf:
        for i, page in enumerate(pdf.pages):
            raw_page = _extract_single_page(page, page_num=i + 1)
            pages.append(raw_page)

    return pages


def _extract_single_page(page: pdfplumber.page.Page, page_num: int) -> RawPage:
    """Extract text and tables from a single pdfplumber Page object."""

    # Extract tables first so we can filter their regions from the text
    tables: list[RawTableBlock] = []
    table_bboxes: list[tuple] = []

    for table in page.find_tables():
        rows = table.extract()
        if rows:
            # Normalize: replace None cells with empty string
            cleaned_rows = [
                [cell if cell is not None else "" for cell in row]
                for row in rows
            ]
            bbox = table.bbox
            tables.append(RawTableBlock(rows=cleaned_rows, bbox=bbox))
            table_bboxes.append(bbox)

    # Extract text, filtering out table regions to avoid double-counting
    if table_bboxes:
        # Crop out table areas and extract remaining text
        text_page = page
        for bbox in table_bboxes:
            try:
                # Mask the table region so extract_text ignores it
                text_page = text_page.outside_bbox(bbox)
            except Exception:
                pass
        raw_text = text_page.extract_text(x_tolerance=3, y_tolerance=3) or ""
    else:
        raw_text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""

    return RawPage(
        page_num=page_num,
        raw_text=raw_text,
        tables=tables,
        width=float(page.width),
        height=float(page.height),
    )
