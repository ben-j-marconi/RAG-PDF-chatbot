"""
ingestion/registry.py — Canonical document registry.

Maintains a JSON-backed registry of every PDF that has been ingested.
Each entry tracks document identity, provenance, version, and ingestion state.

Design rationale:
  This is the "source of truth" for what's been indexed. The file hash enables
  delta ingestion — only changed or new documents get reprocessed. This is how
  you avoid re-embedding a 10,000-page corpus every time one document changes.
"""

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Optional


class DocumentRegistry:
    """Persistent registry of ingested documents, backed by a JSON file with optional GCS sync."""

    _GCS_OBJECT = "registry/document_registry.json"

    def __init__(self, registry_path: str, gcs_bucket: str = ""):
        self.registry_path = registry_path
        self.gcs_bucket = gcs_bucket
        os.makedirs(os.path.dirname(registry_path), exist_ok=True)
        self._records: dict[str, dict] = {}
        self._load()

    # ── Public API ────────────────────────────────────────────────────────────

    def register(
        self,
        filepath: str,
        *,
        acl_metadata: Optional[dict] = None,
    ) -> dict:
        """
        Register or update a document. Returns the registry record.
        If the file hash matches the existing record, marks it as unchanged.
        """
        filename = os.path.basename(filepath)
        file_hash = _sha256(filepath)

        existing = self._find_by_filename(filename)

        if existing and existing["file_hash"] == file_hash:
            existing["status"] = "unchanged"
            self._save()
            return existing

        record = {
            "document_id": existing["document_id"] if existing else str(uuid.uuid4()),
            "filename": filename,
            "source_uri": os.path.abspath(filepath),
            "file_hash": file_hash,
            "version": (existing["version"] + 1) if existing else 1,
            "ingestion_time": datetime.now(timezone.utc).isoformat(),
            "acl_metadata": acl_metadata or {},
            "chunk_count": 0,
            "status": "pending",   # pending → indexed
        }

        self._records[record["document_id"]] = record
        self._save()
        return record

    def mark_indexed(self, document_id: str, chunk_count: int) -> None:
        """Called after successful indexing to finalize the record."""
        if document_id in self._records:
            self._records[document_id]["status"] = "indexed"
            self._records[document_id]["chunk_count"] = chunk_count
            self._save()

    def mark_failed(self, document_id: str, error: str) -> None:
        if document_id in self._records:
            self._records[document_id]["status"] = "failed"
            self._records[document_id]["error"] = error
            self._save()

    def remove(self, document_id: str) -> None:
        """Remove a document from the registry (used when source PDF is deleted)."""
        self._records.pop(document_id, None)
        self._save()

    def all_records(self) -> list[dict]:
        return list(self._records.values())

    def get(self, document_id: str) -> Optional[dict]:
        return self._records.get(document_id)

    def get_by_filename(self, filename: str) -> Optional[dict]:
        return self._find_by_filename(filename)

    def summary(self) -> dict:
        records = self.all_records()
        return {
            "total": len(records),
            "indexed": sum(1 for r in records if r["status"] == "indexed"),
            "pending": sum(1 for r in records if r["status"] == "pending"),
            "failed": sum(1 for r in records if r["status"] == "failed"),
            "unchanged": sum(1 for r in records if r["status"] == "unchanged"),
            "total_chunks": sum(r.get("chunk_count", 0) for r in records),
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _find_by_filename(self, filename: str) -> Optional[dict]:
        for rec in self._records.values():
            if rec["filename"] == filename:
                return rec
        return None

    def _load(self) -> None:
        if not os.path.exists(self.registry_path) and self.gcs_bucket:
            _gcs_download(self.gcs_bucket, self._GCS_OBJECT, self.registry_path)
        if os.path.exists(self.registry_path):
            with open(self.registry_path, "r") as f:
                data = json.load(f)
                self._records = {r["document_id"]: r for r in data}

    def _save(self) -> None:
        with open(self.registry_path, "w") as f:
            json.dump(list(self._records.values()), f, indent=2)
        if self.gcs_bucket:
            _gcs_upload(self.gcs_bucket, self._GCS_OBJECT, self.registry_path)


def _gcs_upload(bucket_name: str, object_path: str, local_path: str) -> None:
    try:
        from google.cloud import storage
        storage.Client().bucket(bucket_name).blob(object_path).upload_from_filename(local_path)
    except Exception:
        pass


def _gcs_download(bucket_name: str, object_path: str, local_path: str) -> None:
    try:
        from google.cloud import storage
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        storage.Client().bucket(bucket_name).blob(object_path).download_to_filename(local_path)
    except Exception:
        pass


def _sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
