"""
ingestion/scanner.py — PDF directory scanner.

Walks the source PDF directory, compares against the registry,
and returns a work list: new, changed, and deleted documents.

Design rationale:
  A production system would poll Cloud Storage or receive Pub/Sub events.
  For the MVP we keep it local-first and file-hash-based — the logic is
  identical whether the source is a local folder or a GCS bucket.
"""

import os
from dataclasses import dataclass
from typing import Optional

from src.ingestion.registry import DocumentRegistry


@dataclass
class ScanResult:
    new: list[str]           # file paths to ingest for the first time
    changed: list[str]       # file paths that have been updated
    unchanged: list[str]     # file paths with no changes (skip)
    deleted: list[str]       # document_ids whose source file is gone


def scan_pdf_directory(pdf_dir: str, registry: DocumentRegistry) -> ScanResult:
    """
    Scan `pdf_dir` for PDF files and compare against the registry.
    Returns a ScanResult describing what needs to happen.
    """
    if not os.path.isdir(pdf_dir):
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    # All PDF files currently on disk
    disk_files = {
        fname: os.path.join(pdf_dir, fname)
        for fname in os.listdir(pdf_dir)
        if fname.lower().endswith(".pdf")
    }

    new_files: list[str] = []
    changed_files: list[str] = []
    unchanged_files: list[str] = []

    for fname, fpath in disk_files.items():
        record = registry.get_by_filename(fname)
        if record is None:
            new_files.append(fpath)
        elif _hash_changed(fpath, record["file_hash"]):
            changed_files.append(fpath)
        else:
            unchanged_files.append(fpath)

    # Documents in registry whose source file no longer exists
    registered_filenames = {r["filename"] for r in registry.all_records()}
    deleted_ids = [
        r["document_id"]
        for r in registry.all_records()
        if r["filename"] not in disk_files
    ]

    return ScanResult(
        new=new_files,
        changed=changed_files,
        unchanged=unchanged_files,
        deleted=deleted_ids,
    )


def _hash_changed(filepath: str, known_hash: str) -> bool:
    import hashlib
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest() != known_hash
