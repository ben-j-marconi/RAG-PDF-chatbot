"""
parsing/ocr_adapter.py — OCR interface (stub for MVP).

This module defines the interface for OCR-based text extraction.
The implementation is intentionally left as a stub for the MVP because
all 5 demo PDFs are native text. The interface is designed so a full
implementation (e.g., using pytesseract or Google Document AI) can be
dropped in without touching any other module.

Design rationale:
  We didn't implement full OCR in the MVP because the pilot corpus is
  text-native. But the architecture makes it a single-module swap.
  In production on Vertex AI, this would call Document AI Layout Parser,
  which handles scanned PDFs, handwriting, and complex table layouts natively.
"""

import logging

logger = logging.getLogger(__name__)


class OCRNotImplementedError(Exception):
    """Raised when OCR is requested but not configured."""
    pass


def extract_text_via_ocr(filepath: str, page_num: int) -> str:
    """
    Extract text from a scanned PDF page via OCR.

    MVP stub: logs a warning and returns an empty string so the pipeline
    degrades gracefully rather than crashing on scanned documents.

    Production implementation would:
      1. Render the page to an image (pdf2image / poppler)
      2. Pass to pytesseract or Google Document AI
      3. Return cleaned text with layout hints
    """
    logger.warning(
        "ocr_requested_but_not_implemented",
        extra={
            "filepath": filepath,
            "page_num": page_num,
            "action": "returning_empty_string",
            "note": "Swap in pytesseract or Document AI here for production",
        },
    )
    return ""


class DocumentAIAdapter:
    """
    Future production adapter for Google Document AI Layout Parser.

    Usage (production):
      adapter = DocumentAIAdapter(project_id, location, processor_id)
      text = adapter.process_document(filepath)

    This class exists to make the production path visible during the demo
    without requiring a GCP connection for the local MVP.
    """

    def __init__(self, project_id: str, location: str, processor_id: str):
        self.project_id = project_id
        self.location = location
        self.processor_id = processor_id

    def process_document(self, filepath: str) -> str:
        raise NotImplementedError(
            "DocumentAIAdapter is a production stub. "
            "Set LLM_PROVIDER=vertex and configure GOOGLE_CLOUD_PROJECT to enable."
        )
