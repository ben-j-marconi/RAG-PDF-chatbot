"""
parsing/detector.py — PDF type detection.

Distinguishes native text PDFs from scanned/image-based PDFs.
This is the first decision gate in the document restructuring pipeline.

Design rationale:
  Standard RAG pipelines assume all PDFs are text-native. Our system
  explicitly detects the document type before choosing a parsing strategy.
  Scanned PDFs get routed to the OCR adapter. This is why we can handle
  a mixed corpus without silent quality degradation.
"""

import pdfplumber

# Minimum average characters per page to classify as "native text"
NATIVE_TEXT_CHAR_THRESHOLD = 100


def detect_pdf_type(filepath: str) -> str:
    """
    Returns "native" if the PDF has extractable text, "scanned" otherwise.
    Uses pdfplumber to sample character density across all pages.
    """
    total_chars = 0
    total_pages = 0

    try:
        with pdfplumber.open(filepath) as pdf:
            total_pages = len(pdf.pages)
            for page in pdf.pages:
                text = page.extract_text() or ""
                total_chars += len(text.strip())
    except Exception:
        return "unknown"

    if total_pages == 0:
        return "unknown"

    avg_chars_per_page = total_chars / total_pages

    if avg_chars_per_page >= NATIVE_TEXT_CHAR_THRESHOLD:
        return "native"
    else:
        return "scanned"
