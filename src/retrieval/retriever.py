"""
retrieval/retriever.py — Hybrid retrieval with Reciprocal Rank Fusion.

Implements the Retrieval & Grounding layer:
  1. Vector search (semantic similarity via ChromaDB)
  2. BM25 lexical search (exact term matching)
  3. Optional metadata filtering (by chunk_type or filename)
  4. Merge via Reciprocal Rank Fusion (RRF)
  5. Deduplicate by chunk_id
  6. Return top-N candidates for reranking

Reciprocal Rank Fusion (RRF):
  score(d) = Σ 1 / (k + rank(d, list_i))
  where k=60 (standard constant), rank is 1-indexed position in each list.
  RRF is parameter-light, doesn't require tuning score scales across systems,
  and consistently outperforms linear combination in IR benchmarks.

Design rationale:
  The merge strategy matters. Linear score combination requires you to
  normalize across fundamentally different score distributions (cosine
  similarity vs BM25 TF-IDF scores). RRF sidesteps this by working in
  rank space, which is why it's the default fusion strategy in production
  hybrid search systems like Elasticsearch 8.x and Vertex AI RAG Engine.
"""

import logging
from typing import Optional

from config import Config
from src.chunking.chunker import Chunk
from src.indexing.embedder import embed_query
from src.indexing.vector_store import get_vector_store
from src.indexing.lexical_index import LexicalIndex
from src.observability.logger import get_logger

RRF_K = 60  # Standard RRF constant; higher = less aggressive rank discounting


class HybridRetriever:
    """
    Hybrid retriever combining semantic vector search and BM25 lexical search,
    merged via Reciprocal Rank Fusion.
    """

    def __init__(self, config: Config):
        self.config = config
        self.vector_store = get_vector_store(config)
        self.lexical_index = LexicalIndex(config.bm25_index_path, gcs_bucket=config.gcs_bucket)
        self.logger = get_logger("retrieval.retriever", config)

    def retrieve(
        self,
        query: str,
        top_k: int = 30,
        metadata_filter: Optional[dict] = None,
    ) -> list[Chunk]:
        """
        Execute hybrid retrieval for a query.

        Args:
            query: Natural language question from the user
            top_k: Number of candidates to return before reranking
            metadata_filter: ChromaDB-compatible filter dict, e.g.
                             {"chunk_type": "table"} or {"source_filename": "01_...pdf"}

        Returns:
            Deduplicated list of Chunk objects, ranked by RRF score (desc).
        """
        self.logger.info("retrieval_start", extra={
            "event": "retrieval_start",
            "query_preview": query[:80],
            "metadata_filter": str(metadata_filter),
        })

        # ── Vector retrieval ──────────────────────────────────────────────────
        query_embedding = embed_query(query, model_name=self.config.embedding_model)
        vector_results = self.vector_store.query(
            query_embedding,
            top_k=self.config.vector_top_k,
            where=metadata_filter,
        )
        self.logger.info("vector_retrieval_complete", extra={
            "event": "vector_retrieval_complete",
            "count": len(vector_results),
        })

        # ── BM25 lexical retrieval ────────────────────────────────────────────
        bm25_results = self.lexical_index.query(query, top_k=self.config.bm25_top_k)
        self.logger.info("bm25_retrieval_complete", extra={
            "event": "bm25_retrieval_complete",
            "count": len(bm25_results),
        })

        # ── Reciprocal Rank Fusion ────────────────────────────────────────────
        merged = _rrf_merge(vector_results, bm25_results, k=RRF_K)

        # ── Deduplicate ───────────────────────────────────────────────────────
        seen_ids: set[str] = set()
        candidates: list[Chunk] = []
        for chunk in merged:
            if chunk.chunk_id not in seen_ids:
                seen_ids.add(chunk.chunk_id)
                candidates.append(chunk)
            if len(candidates) >= top_k:
                break

        self.logger.info("retrieval_candidates_ready", extra={
            "event": "retrieval_candidates_ready",
            "candidate_count": len(candidates),
            "sources": list({c.source_filename for c in candidates}),
        })

        return candidates


def _rrf_merge(
    list_a: list[Chunk],
    list_b: list[Chunk],
    k: int = RRF_K,
) -> list[Chunk]:
    """
    Merge two ranked lists of chunks using Reciprocal Rank Fusion.
    Returns chunks sorted by combined RRF score (highest first).
    """
    scores: dict[str, float] = {}
    chunks_by_id: dict[str, Chunk] = {}

    for rank, chunk in enumerate(list_a, start=1):
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + 1.0 / (k + rank)
        chunks_by_id[chunk.chunk_id] = chunk

    for rank, chunk in enumerate(list_b, start=1):
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + 1.0 / (k + rank)
        chunks_by_id[chunk.chunk_id] = chunk

    sorted_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)

    merged: list[Chunk] = []
    for cid in sorted_ids:
        chunk = chunks_by_id[cid]
        chunk.rerank_score = round(scores[cid], 6)
        merged.append(chunk)

    return merged
