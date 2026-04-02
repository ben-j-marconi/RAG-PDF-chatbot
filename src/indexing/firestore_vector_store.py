"""
indexing/firestore_vector_store.py — Firestore vector store backend.

Implements the same interface as VectorStore (ChromaDB) using Google Cloud
Firestore with its native vector similarity search capability.

Design rationale:
  Firestore was selected as the managed GCP vector store for this deployment.
  It provides automatic replication, IAM-controlled access, and no infrastructure
  to maintain. The same chunking, embedding, and retrieval pipeline operates
  identically — only the storage backend changes.

  For billion-scale ANN search, this would be replaced with Vertex AI Vector
  Search. That swap is isolated to this module — nothing else in the pipeline
  changes.

Collection layout:
  property_valuation_chunks/{chunk_id}
    ├── embedding       : Vector (384-dim, all-MiniLM-L6-v2)
    ├── text            : str
    ├── document_id     : str
    ├── source_filename : str
    ├── page_num        : int
    ├── chunk_type      : str  ("text" | "table")
    └── section_heading : str
"""

import warnings
from typing import Optional

from src.chunking.chunker import Chunk

COLLECTION_NAME = "property_valuation_chunks"


class FirestoreVectorStore:
    """
    Firestore-backed vector store. Matches the VectorStore (ChromaDB) interface
    exactly so it can be swapped in without changing any other module.
    """

    def __init__(self, project: str):
        warnings.filterwarnings("ignore")
        from google.cloud import firestore
        self._db = firestore.Client(project=project)
        self._col = self._db.collection(COLLECTION_NAME)

    # ── Write operations ──────────────────────────────────────────────────────

    def upsert_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """
        Upsert chunks with their embeddings into Firestore.
        Uses chunk_id as the stable document key — idempotent.
        Batches writes for efficiency (max 500 per batch).
        """
        if not chunks:
            return

        from google.cloud.firestore_v1.vector import Vector

        BATCH_SIZE = 400
        for batch_start in range(0, len(chunks), BATCH_SIZE):
            batch = self._db.batch()
            batch_chunks = chunks[batch_start:batch_start + BATCH_SIZE]
            batch_embeddings = embeddings[batch_start:batch_start + BATCH_SIZE]

            for chunk, embedding in zip(batch_chunks, batch_embeddings):
                doc_ref = self._col.document(chunk.chunk_id)
                batch.set(doc_ref, {
                    "embedding": Vector(embedding),
                    "text": chunk.text,
                    "document_id": chunk.document_id,
                    "source_filename": chunk.source_filename,
                    "page_num": chunk.page_num,
                    "chunk_type": chunk.chunk_type,
                    "section_heading": chunk.section_heading or "",
                })
            batch.commit()

    def delete_by_document_id(self, document_id: str) -> None:
        """Remove all chunks belonging to a specific document."""
        docs = self._col.where("document_id", "==", document_id).stream()
        batch = self._db.batch()
        count = 0
        for doc in docs:
            batch.delete(doc.reference)
            count += 1
            if count % 400 == 0:
                batch.commit()
                batch = self._db.batch()
        if count % 400 != 0:
            batch.commit()

    # ── Query operations ──────────────────────────────────────────────────────

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 20,
        where: Optional[dict] = None,
    ) -> list[Chunk]:
        """
        Vector similarity search using Firestore's find_nearest().
        Returns Chunk objects ranked by cosine similarity (highest first).
        Applies metadata filters as post-query filtering for compatibility.
        """
        from google.cloud.firestore_v1.vector import Vector
        from google.cloud.firestore_v1.base_vector_query import DistanceMeasure

        # Fetch more candidates than needed so post-filters have enough to work with
        fetch_k = top_k * 4 if where else top_k

        vector_query = self._col.find_nearest(
            vector_field="embedding",
            query_vector=Vector(query_embedding),
            distance_measure=DistanceMeasure.COSINE,
            limit=min(fetch_k, 200),
            distance_result_field="_distance",
        )

        chunks: list[Chunk] = []
        for doc in vector_query.stream():
            data = doc.to_dict()

            # Apply metadata filter (post-query)
            if where:
                match = all(data.get(k) == v for k, v in where.items())
                if not match:
                    continue

            chunk = Chunk(
                chunk_id=doc.id,
                document_id=data.get("document_id", ""),
                source_filename=data.get("source_filename", ""),
                page_num=data.get("page_num", 0),
                chunk_type=data.get("chunk_type", "text"),
                section_heading=data.get("section_heading") or None,
                text=data.get("text", ""),
            )
            # Convert cosine distance (0=identical) to similarity score (1=identical)
            distance = data.get("_distance", 0.0)
            chunk.rerank_score = round(1.0 - distance, 4)
            chunks.append(chunk)

            if len(chunks) >= top_k:
                break

        return chunks

    def count(self) -> int:
        """Return total number of chunks in Firestore."""
        from google.cloud.firestore_v1.aggregation import AggregationQuery
        agg = self._col.count()
        results = agg.get()
        return results[0][0].value

    def list_documents(self) -> list[dict]:
        """Return unique document metadata indexed in the store."""
        docs = self._col.select(["document_id", "source_filename"]).stream()
        seen = {}
        for doc in docs:
            data = doc.to_dict()
            doc_id = data.get("document_id", "")
            if doc_id not in seen:
                seen[doc_id] = {
                    "document_id": doc_id,
                    "source_filename": data.get("source_filename", ""),
                }
        return list(seen.values())
