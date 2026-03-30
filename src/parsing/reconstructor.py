"""
parsing/reconstructor.py — Document structure reconstruction.

Converts cleaned RawPage objects into StructuredKnowledgeObjects —
the canonical intermediate representation that feeds the chunker.

Structure detected:
  - Document title (from page 1 content)
  - Section headings (ALL CAPS lines or Title Case short lines)
  - Paragraphs (multi-sentence blocks)
  - Tables (already extracted by extractor.py, formatted as Markdown)
  - Page boundaries (preserved as metadata, not discarded)

Design rationale:
  This is where PDF != flat text becomes concrete. A line like "Income
  Approach Snapshot" is a section heading, not a sentence. If we treat it
  as a paragraph, the chunk that follows loses its semantic label.
  Preserving headings as metadata lets us answer "what does the appraisal
  say about income?" with the correct chunk type, not a random match.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from src.parsing.extractor import RawPage
from src.parsing.cleaner import clean_text, table_rows_to_markdown


@dataclass
class TableBlock:
    """A single table within a page, formatted as Markdown."""
    markdown: str
    row_count: int
    col_count: int
    bbox: Optional[tuple] = None


@dataclass
class PageContent:
    """Structured content extracted from a single page."""
    page_num: int
    heading: Optional[str]           # Dominant section heading for this page
    paragraphs: list[str]            # Cleaned paragraph blocks
    tables: list[TableBlock]
    raw_text: str                    # Cleaned full text (for BM25 indexing)


@dataclass
class StructuredDocument:
    """The canonical structured representation of a full document."""
    document_id: str
    filename: str
    source_uri: str
    pdf_type: str                    # "native" | "scanned"
    pages: list[PageContent] = field(default_factory=list)
    title: Optional[str] = None


def reconstruct_document(
    document_id: str,
    filename: str,
    source_uri: str,
    pdf_type: str,
    raw_pages: list[RawPage],
) -> StructuredDocument:
    """
    Convert raw extraction output into a StructuredDocument.
    This is the output of the Document Restructuring pipeline stage.
    """
    doc = StructuredDocument(
        document_id=document_id,
        filename=filename,
        source_uri=source_uri,
        pdf_type=pdf_type,
    )

    for raw_page in raw_pages:
        page_content = _reconstruct_page(raw_page)
        doc.pages.append(page_content)

    # Infer document title from first page heading or first non-empty paragraph
    if doc.pages:
        first = doc.pages[0]
        if first.heading:
            doc.title = first.heading
        elif first.paragraphs:
            doc.title = first.paragraphs[0][:80]

    return doc


def _reconstruct_page(raw_page: RawPage) -> PageContent:
    """Reconstruct structure for a single page."""
    cleaned_text = clean_text(raw_page.raw_text)

    # Split into lines, detect headings, group into paragraphs
    heading, paragraphs = _extract_headings_and_paragraphs(cleaned_text)

    # Convert raw table blocks to formatted TableBlock objects
    table_blocks: list[TableBlock] = []
    for tbl in raw_page.tables:
        md = table_rows_to_markdown(tbl.rows)
        if md:
            row_count = len(tbl.rows)
            col_count = max(len(r) for r in tbl.rows) if tbl.rows else 0
            table_blocks.append(TableBlock(
                markdown=md,
                row_count=row_count,
                col_count=col_count,
                bbox=tbl.bbox,
            ))

    return PageContent(
        page_num=raw_page.page_num,
        heading=heading,
        paragraphs=paragraphs,
        tables=table_blocks,
        raw_text=cleaned_text,
    )


# ── Heading detection ─────────────────────────────────────────────────────────

# A heading candidate: short line, starts with capital, no sentence-ending punctuation
_HEADING_RE = re.compile(
    r"^([A-Z][A-Za-z0-9 &:/\-,']{2,80})$"
)
# All-caps short line
_ALL_CAPS_RE = re.compile(r"^[A-Z][A-Z0-9 &:/\-,']{2,60}$")


def _is_heading_candidate(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) < 3:
        return False
    if stripped.endswith((".", "?", "!")):
        return False
    if len(stripped.split()) > 10:
        return False
    return bool(_HEADING_RE.match(stripped)) or bool(_ALL_CAPS_RE.match(stripped))


def _extract_headings_and_paragraphs(text: str) -> tuple[Optional[str], list[str]]:
    """
    Parse cleaned page text into a dominant heading and a list of paragraphs.
    The first heading found becomes the page heading.
    Subsequent content is grouped into paragraphs separated by blank lines.
    """
    if not text:
        return None, []

    lines = text.split("\n")
    heading: Optional[str] = None
    current_block: list[str] = []
    paragraphs: list[str] = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            # Blank line → flush current block as paragraph
            if current_block:
                para = " ".join(current_block).strip()
                if para:
                    paragraphs.append(para)
                current_block = []
            continue

        if _is_heading_candidate(stripped):
            # Flush any pending block
            if current_block:
                para = " ".join(current_block).strip()
                if para:
                    paragraphs.append(para)
                current_block = []
            # First heading becomes the page heading
            if heading is None:
                heading = stripped
            else:
                # Subsequent headings become labeled paragraphs
                paragraphs.append(f"[Section: {stripped}]")
        else:
            current_block.append(stripped)

    # Flush remaining content
    if current_block:
        para = " ".join(current_block).strip()
        if para:
            paragraphs.append(para)

    return heading, paragraphs
