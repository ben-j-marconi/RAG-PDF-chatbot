"""
indexing/embedder.py — Local embedding generation.

Wraps sentence-transformers to produce dense vector representations of chunks.
The embedding model is loaded once and cached for the session.

Model choice: all-MiniLM-L6-v2
  - 384 dimensions (fast, low memory)
  - Strong performance on short text / factual retrieval
  - No API key required — runs entirely locally
  - Swappable: change EMBEDDING_MODEL in .env to upgrade to a larger model
    or to text-embedding-004 via the Vertex AI Embeddings API in production

Design rationale:
  The embedding model is injected via config so we can A/B test it. In
  production on Vertex AI we'd use text-embedding-004 which is trained
  on multilingual and domain-specific corpora and integrates with
  Vertex AI Vector Search for billion-scale retrieval.
"""

from typing import Optional
import numpy as np

_model_cache: dict[str, object] = {}


def get_embedder(model_name: str = "all-MiniLM-L6-v2"):
    """Return a cached SentenceTransformer instance."""
    if model_name not in _model_cache:
        from sentence_transformers import SentenceTransformer
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def embed_texts(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> list[list[float]]:
    """
    Embed a list of text strings. Returns a list of float vectors.
    Batched automatically by sentence-transformers.
    """
    if not texts:
        return []
    model = get_embedder(model_name)
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings.tolist()


def embed_query(query: str, model_name: str = "all-MiniLM-L6-v2") -> list[float]:
    """Embed a single query string."""
    return embed_texts([query], model_name=model_name)[0]
