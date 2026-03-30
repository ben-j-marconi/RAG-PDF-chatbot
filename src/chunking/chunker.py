"""
chunking/chunker.py — Section-aware and table-preserving chunking.

Converts a StructuredDocument into a flat list of Chunk objects ready
for embedding and indexing.

Chunking strategy:
  1. Tables → one chunk each (never split; table semantics require intact rows)
  2. Text sections → chunked by paragraph boundaries, with a token-size guard
  3. If a paragraph block exceeds MAX_TOKENS, it splits at sentence boundaries
     with a small overlap to preserve context

This avoids the two canonical RAG failure modes:
  a) Fixed-size chunking that splits mid-table (destroys numeric retrieval)
  b) Whole-page chunking that dilutes specificity (retrieves too much noise)

Design rationale:
  Chunk type is preserved as metadata. When a query asks about cap rates,
  the retriever can down-score pure-text chunks from narrative sections
  and up-score table chunks from the appraisal and comps documents.
"""

import uuid
from dataclasses import dataclass, field
from typing import Optional

from src.parsing.reconstructor import StructuredDocument, PageContent, TableBlock

MAX_TOKENS = 400       # Approximate token limit per text chunk (1 token ≈ 4 chars)
OVERLAP_SENTENCES = 2  # Sentences of overlap between split chunks


@dataclass
class Chunk:
    """
    The atomic retrieval unit. Every chunk carries full provenance so that
    citations can always be traced back to a specific page and document.
    """
    chunk_id: str
    document_id: str
    source_filename: str
    page_num: int                    # 1-indexed; for multi-page chunks, the start page
    chunk_type: str                  # "text" | "table"
    section_heading: Optional[str]   # Nearest section heading above this chunk
    text: str                        # The actual content to embed and retrieve

    # Populated post-retrieval
    rerank_score: Optional[float] = None


def chunk_document(doc: StructuredDocument) -> list[Chunk]:
    """
    Convert a StructuredDocument into a list of Chunks.
    Returns chunks in page order.
    """
    chunks: list[Chunk] = []
    current_heading: Optional[str] = None

    for page in doc.pages:
        # Update running heading from this page
        if page.heading:
            current_heading = page.heading

        # ── Table chunks ──────────────────────────────────────────────────────
        for table in page.tables:
            chunk = Chunk(
                chunk_id=str(uuid.uuid4()),
                document_id=doc.document_id,
                source_filename=doc.filename,
                page_num=page.page_num,
                chunk_type="table",
                section_heading=current_heading,
                text=_table_to_chunk_text(table, current_heading),
            )
            chunks.append(chunk)

        # ── Text chunks ───────────────────────────────────────────────────────
        for paragraph in page.paragraphs:
            if not paragraph.strip():
                continue

            # Check if this paragraph fits in one chunk
            if _token_estimate(paragraph) <= MAX_TOKENS:
                chunk = Chunk(
                    chunk_id=str(uuid.uuid4()),
                    document_id=doc.document_id,
                    source_filename=doc.filename,
                    page_num=page.page_num,
                    chunk_type="text",
                    section_heading=current_heading,
                    text=paragraph.strip(),
                )
                chunks.append(chunk)
            else:
                # Split long paragraph at sentence boundaries
                sub_chunks = _split_long_text(
                    paragraph,
                    doc.document_id,
                    doc.filename,
                    page.page_num,
                    current_heading,
                )
                chunks.extend(sub_chunks)

    return chunks


# ── Helpers ───────────────────────────────────────────────────────────────────

def _table_to_chunk_text(table: TableBlock, heading: Optional[str]) -> str:
    """Wrap a Markdown table with its section context for better retrieval."""
    prefix = f"[Table from section: {heading}]\n" if heading else "[Table]\n"
    return prefix + table.markdown


def _token_estimate(text: str) -> int:
    """Rough token count: 1 token ≈ 4 characters."""
    return len(text) // 4


def _split_long_text(
    text: str,
    document_id: str,
    filename: str,
    page_num: int,
    heading: Optional[str],
) -> list[Chunk]:
    """Split a long paragraph into overlapping sentence-boundary chunks."""
    import re
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[Chunk] = []
    current_sentences: list[str] = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = _token_estimate(sent)
        if current_tokens + sent_tokens > MAX_TOKENS and current_sentences:
            # Emit current chunk
            chunks.append(Chunk(
                chunk_id=str(uuid.uuid4()),
                document_id=document_id,
                source_filename=filename,
                page_num=page_num,
                chunk_type="text",
                section_heading=heading,
                text=" ".join(current_sentences),
            ))
            # Keep overlap
            current_sentences = current_sentences[-OVERLAP_SENTENCES:]
            current_tokens = sum(_token_estimate(s) for s in current_sentences)

        current_sentences.append(sent)
        current_tokens += sent_tokens

    # Flush remaining
    if current_sentences:
        chunks.append(Chunk(
            chunk_id=str(uuid.uuid4()),
            document_id=document_id,
            source_filename=filename,
            page_num=page_num,
            chunk_type="text",
            section_heading=heading,
            text=" ".join(current_sentences),
        ))

    return chunks
