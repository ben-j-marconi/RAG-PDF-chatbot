"""
retrieval/reranker.py — Cross-encoder reranking.

Takes the merged candidate set from hybrid retrieval and reranks it using
a cross-encoder model that jointly encodes the query and each candidate chunk.

Why cross-encoder vs bi-encoder (embedding similarity)?
  - Bi-encoder: encode query and document independently → fast, approximate
  - Cross-encoder: encode [query, document] together → slower, but reads
    both inputs simultaneously so it captures query-document interaction
    that embedding dot product cannot express

For a corpus of ~50 chunks this is nearly instantaneous.
The model (ms-marco-MiniLM-L-6-v2) was trained on MS MARCO passage ranking
which is the standard benchmark for short document retrieval.

Design rationale:
  This is the quality gate before the LLM sees any context. Without reranking,
  a query about "DSCR" might surface a chunk that mentions "debt" and "service"
  separately but never the combined concept. The cross-encoder catches that.
  In production on Vertex AI, this maps to the Ranking API (Vertex AI Reranker).
"""

import logging
from typing import Optional

from config import Config
from src.chunking.chunker import Chunk
from src.observability.logger import get_logger

_reranker_cache: dict[str, object] = {}


def get_reranker(model_name: str):
    """Return a cached CrossEncoder instance."""
    if model_name not in _reranker_cache:
        from sentence_transformers import CrossEncoder
        _reranker_cache[model_name] = CrossEncoder(model_name)
    return _reranker_cache[model_name]


class Reranker:
    """
    Cross-encoder reranker. Scores query-chunk pairs and returns top-k chunks.
    """

    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger("retrieval.reranker", config)

    def rerank(self, query: str, candidates: list[Chunk]) -> list[Chunk]:
        """
        Rerank candidate chunks against the query using a cross-encoder.

        Args:
            query: The user's natural language question
            candidates: Output from HybridRetriever.retrieve()

        Returns:
            Top-k chunks sorted by cross-encoder relevance score (desc).
        """
        if not candidates:
            return []

        top_k = self.config.rerank_top_k
        model = get_reranker(self.config.reranker_model)

        # Build (query, chunk_text) pairs for the cross-encoder
        pairs = [(query, c.text) for c in candidates]
        scores = model.predict(pairs, show_progress_bar=False)

        # Attach scores and sort
        scored_chunks = sorted(
            zip(scores, candidates),
            key=lambda x: float(x[0]),
            reverse=True,
        )

        top_chunks: list[Chunk] = []
        for score, chunk in scored_chunks[:top_k]:
            chunk.rerank_score = round(float(score), 4)
            top_chunks.append(chunk)

        self.logger.info("rerank_complete", extra={
            "event": "rerank_complete",
            "candidates_in": len(candidates),
            "top_k_out": len(top_chunks),
            "top_scores": [c.rerank_score for c in top_chunks],
            "top_sources": [
                f"{c.source_filename}:p{c.page_num}" for c in top_chunks
            ],
        })

        return top_chunks
