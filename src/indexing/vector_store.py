"""
indexing/vector_store.py — ChromaDB vector store wrapper.

Manages a persistent local ChromaDB collection for chunk embeddings.
Supports upsert (idempotent re-indexing), query, and deletion by document_id.

Design rationale:
  ChromaDB was chosen for the MVP because it requires zero infrastructure —
  it persists to a local directory. The interface (upsert, query, delete_by_doc)
  is the same contract you'd implement against Vertex AI Vector Search in
  production. Swapping backends is a single class change.
"""

import os
from typing import Optional

import chromadb
from chromadb.config import Settings

from src.chunking.chunker import Chunk

COLLECTION_NAME = "property_valuation_chunks"


class VectorStore:
    """ChromaDB-backed vector store for property valuation chunks."""

    def __init__(self, chroma_dir: str):
        os.makedirs(chroma_dir, exist_ok=True)
        self._client = chromadb.PersistentClient(path=chroma_dir)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Write operations ──────────────────────────────────────────────────────

    def upsert_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """
        Upsert chunks with their embeddings.
        Uses chunk_id as the stable key — safe to call multiple times.
        """
        if not chunks:
            return

        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[_chunk_to_metadata(c) for c in chunks],
        )

    def delete_by_document_id(self, document_id: str) -> None:
        """Remove all chunks belonging to a specific document."""
        results = self._collection.get(
            where={"document_id": document_id},
            include=[],
        )
        if results["ids"]:
            self._collection.delete(ids=results["ids"])

    # ── Query operations ──────────────────────────────────────────────────────

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 20,
        where: Optional[dict] = None,
    ) -> list[Chunk]:
        """
        Semantic similarity search. Returns Chunk objects ranked by cosine distance.
        `where` supports ChromaDB metadata filters, e.g. {"chunk_type": "table"}
        """
        kwargs = dict(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count() or 1),
            include=["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        chunks: list[Chunk] = []
        if not results["ids"] or not results["ids"][0]:
            return chunks

        for chunk_id, text, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunk = _metadata_to_chunk(chunk_id, text, meta)
            # Convert cosine distance (0=identical, 2=opposite) to similarity score
            chunk.rerank_score = round(1.0 - dist / 2.0, 4)
            chunks.append(chunk)

        return chunks

    def count(self) -> int:
        return self._collection.count()

    def list_documents(self) -> list[dict]:
        """Return unique document metadata (filename, document_id) indexed."""
        results = self._collection.get(include=["metadatas"])
        seen = {}
        for meta in results.get("metadatas", []):
            doc_id = meta.get("document_id", "")
            if doc_id not in seen:
                seen[doc_id] = {
                    "document_id": doc_id,
                    "source_filename": meta.get("source_filename", ""),
                }
        return list(seen.values())


# ── Serialization helpers ─────────────────────────────────────────────────────

def _chunk_to_metadata(chunk: Chunk) -> dict:
    """Convert a Chunk to a ChromaDB-compatible metadata dict (str/int/float/bool only)."""
    return {
        "document_id": chunk.document_id,
        "source_filename": chunk.source_filename,
        "page_num": chunk.page_num,
        "chunk_type": chunk.chunk_type,
        "section_heading": chunk.section_heading or "",
    }


def _metadata_to_chunk(chunk_id: str, text: str, meta: dict) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=meta.get("document_id", ""),
        source_filename=meta.get("source_filename", ""),
        page_num=meta.get("page_num", 0),
        chunk_type=meta.get("chunk_type", "text"),
        section_heading=meta.get("section_heading") or None,
        text=text,
    )
