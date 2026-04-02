"""
ingestion/pipeline.py — End-to-end ingestion orchestrator.

Coordinates the full Document Restructuring → Knowledge Indexing pipeline:
  1. Scan PDF directory for new/changed/deleted documents
  2. For each new/changed document:
     a. Detect PDF type (native vs scanned)
     b. Extract pages (text + tables)
     c. Clean and reconstruct structure
     d. Chunk into Chunk objects
     e. Embed chunks
     f. Upsert to vector store
     g. Add to BM25 index
     h. Update document registry
  3. Handle deletions: remove from vector store, BM25, and registry

This module is called by both `ingest.py` (CLI) and the Streamlit sidebar button.
"""

import logging
import os
from typing import Callable, Optional

from config import Config
from src.ingestion.registry import DocumentRegistry
from src.ingestion.scanner import scan_pdf_directory
from src.parsing.detector import detect_pdf_type
from src.parsing.extractor import extract_pages
from src.parsing.reconstructor import reconstruct_document
from src.chunking.chunker import chunk_document
from src.indexing.embedder import embed_texts
from src.indexing.vector_store import get_vector_store
from src.indexing.lexical_index import LexicalIndex
from src.observability.logger import (
    get_logger,
    check_extraction_quality,
)


class IngestionPipeline:
    """
    Orchestrates the full ingest → parse → chunk → index pipeline.
    Designed to be called from both CLI and Streamlit UI.
    """

    def __init__(self, config: Config, progress_callback: Optional[Callable] = None):
        """
        Args:
            config: Application configuration
            progress_callback: Optional callable(message: str) for streaming
                               progress updates to a UI (e.g., Streamlit st.write)
        """
        self.config = config
        self.progress = progress_callback or (lambda msg: None)
        self.logger = get_logger("ingestion.pipeline", config)

        self.registry = DocumentRegistry(config.registry_path, gcs_bucket=config.gcs_bucket)
        self.vector_store = get_vector_store(config)
        self.lexical_index = LexicalIndex(config.bm25_index_path, gcs_bucket=config.gcs_bucket)

    def run(self, force_reindex: bool = False) -> dict:
        """
        Run the full ingestion pipeline.

        Args:
            force_reindex: If True, re-ingest all documents regardless of hash.

        Returns:
            Summary dict with counts of processed, skipped, failed documents.
        """
        source = (
            f"gs://{self.config.gcs_bucket}/{self.config.gcs_pdf_prefix}"
            if self.config.gcs_bucket
            else self.config.pdf_dir
        )
        self.logger.info("ingestion_start", extra={
            "event": "ingestion_start",
            "source": source,
            "force_reindex": force_reindex,
        })

        if self.config.gcs_bucket:
            self.progress(f"Source: gs://{self.config.gcs_bucket}/{self.config.gcs_pdf_prefix}")

        scan = scan_pdf_directory(
            self.config.pdf_dir,
            self.registry,
            gcs_bucket=self.config.gcs_bucket,
            gcs_pdf_prefix=self.config.gcs_pdf_prefix,
            gcp_project=self.config.gcp_project,
        )

        # Handle deletions
        for doc_id in scan.deleted:
            self._remove_document(doc_id)

        # Files to process: new + changed, plus all if force_reindex
        if force_reindex:
            to_process = scan.new + scan.changed + scan.unchanged
            skipped = 0
        else:
            to_process = scan.new + scan.changed
            skipped = len(scan.unchanged)

        self.progress(f"Found {len(to_process)} document(s) to process, "
                      f"{len(scan.unchanged) if not force_reindex else 0} unchanged, "
                      f"{len(scan.deleted)} deleted.")

        results = {"processed": 0, "skipped": skipped, "failed": 0}

        for i, filepath in enumerate(to_process):
            filename = os.path.basename(filepath)
            self.progress(f"[{i+1}/{len(to_process)}] Processing: {filename}")
            try:
                self._ingest_document(filepath)
                results["processed"] += 1
            except Exception as e:
                self.logger.error("ingestion_failed", extra={
                    "event": "ingestion_failed",
                    "filepath": filepath,
                    "error": str(e),
                })
                self.progress(f"  ✗ Failed: {filename} — {e}")
                results["failed"] += 1

        self.progress(
            f"Ingestion complete: {results['processed']} processed, "
            f"{results['skipped']} skipped, {results['failed']} failed."
        )
        self.logger.info("ingestion_complete", extra={
            "event": "ingestion_complete", **results
        })

        # Clean up any GCS temp download directory
        scan.cleanup()

        return results

    # ── Per-document pipeline ─────────────────────────────────────────────────

    def _ingest_document(self, filepath: str) -> None:
        """Full pipeline for a single PDF file."""
        filename = os.path.basename(filepath)

        # 1. Register document (creates/updates record, computes hash)
        record = self.registry.register(filepath)
        doc_id = record["document_id"]

        # 2. Always clean old chunks before re-indexing.
        # We unconditionally remove to prevent duplicate chunk accumulation
        # when force_reindex=True reprocesses a hash-unchanged document.
        self.vector_store.delete_by_document_id(doc_id)
        self.lexical_index.remove_document(doc_id)

        # 3. Detect PDF type
        pdf_type = detect_pdf_type(filepath)
        self.logger.info("pdf_type_detected", extra={
            "event": "pdf_type_detected",
            "doc_filename": filename,
            "pdf_type": pdf_type,
        })
        self.progress(f"  → Type: {pdf_type}")

        if pdf_type == "scanned":
            self.progress("  ⚠ Scanned PDF detected — OCR path (stub). Partial extraction only.")

        # 4. Extract pages
        raw_pages = extract_pages(filepath)
        self.progress(f"  → Extracted {len(raw_pages)} pages")

        # 5. Reconstruct structure
        doc = reconstruct_document(
            document_id=doc_id,
            filename=filename,
            source_uri=os.path.abspath(filepath),
            pdf_type=pdf_type,
            raw_pages=raw_pages,
        )

        # 6. Check extraction quality
        check_extraction_quality(doc, self.logger)

        # 7. Chunk
        chunks = chunk_document(doc)
        self.progress(f"  → Created {len(chunks)} chunks "
                      f"({sum(1 for c in chunks if c.chunk_type=='table')} tables, "
                      f"{sum(1 for c in chunks if c.chunk_type=='text')} text)")

        if not chunks:
            self.logger.warning("no_chunks_produced", extra={
                "event": "no_chunks_produced", "doc_filename": filename
            })
            return

        # 8. Embed
        self.progress("  → Embedding chunks...")
        texts = [c.text for c in chunks]
        embeddings = embed_texts(texts, model_name=self.config.embedding_model)

        # 9. Index: vector store
        self.vector_store.upsert_chunks(chunks, embeddings)
        self.logger.info("vector_index_updated", extra={
            "event": "vector_index_updated",
            "doc_filename": filename,
            "chunk_count": len(chunks),
        })

        # 10. Index: BM25
        self.lexical_index.add_chunks(chunks)

        # 11. Mark indexed in registry
        self.registry.mark_indexed(doc_id, chunk_count=len(chunks))
        self.progress(f"  ✓ Indexed {filename} ({len(chunks)} chunks)")

    def _remove_document(self, document_id: str) -> None:
        """Remove a deleted document from all indexes."""
        record = self.registry.get(document_id)
        filename = record["filename"] if record else document_id
        self.progress(f"  → Removing deleted document: {filename}")
        self.vector_store.delete_by_document_id(document_id)
        self.lexical_index.remove_document(document_id)
        self.registry.remove(document_id)
        self.logger.info("document_removed", extra={
            "event": "document_removed", "document_id": document_id
        })
