"""
indexing/lexical_index.py — BM25 lexical search index.

Provides keyword-based retrieval as the complement to vector search.
BM25 (Best Match 25) is the industry-standard probabilistic retrieval model
used by Elasticsearch and Solr under the hood.

Why BM25 in addition to vector search?
  - Exact term matching: "DSCR 1.31x" → vector search may miss this because
    the embedding space treats "1.31" as a number, not a keyword
  - Rare terms: property addresses, specific dollar amounts, unit numbers
  - Acronyms: NOI, LTV, HVAC — these tokenize poorly for sentence transformers

The index is serialized to disk so it survives app restarts without rebuilding.

Design rationale:
  Hybrid retrieval is a well-established pattern. The key insight is that
  vector search and BM25 are complementary failure modes. BM25 fails on
  paraphrase ("net operating income" vs "NOI"); vectors fail on rare
  exact terms. Combining them via RRF covers both cases.
"""

import os
import pickle
from dataclasses import dataclass
from typing import Optional

from rank_bm25 import BM25Okapi

from src.chunking.chunker import Chunk


@dataclass
class BM25Index:
    """Persistent BM25 index over all indexed chunks."""
    _bm25: Optional[BM25Okapi] = None
    _chunk_ids: list[str] = None
    _chunks_by_id: dict[str, Chunk] = None

    def __post_init__(self):
        if self._chunk_ids is None:
            self._chunk_ids = []
        if self._chunks_by_id is None:
            self._chunks_by_id = {}


class LexicalIndex:
    """BM25-backed lexical search index, persisted as a pickle file with optional GCS sync."""

    _GCS_OBJECT = "registry/bm25_index.pkl"

    def __init__(self, index_path: str, gcs_bucket: str = ""):
        self.index_path = index_path
        self.gcs_bucket = gcs_bucket
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        self._state = BM25Index()
        self._load()

    # ── Write operations ──────────────────────────────────────────────────────

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Add chunks to the index. Rebuilds BM25 over all stored chunks."""
        for chunk in chunks:
            self._state._chunks_by_id[chunk.chunk_id] = chunk

        self._rebuild()
        self._save()

    def remove_document(self, document_id: str) -> None:
        """Remove all chunks for a document and rebuild the index."""
        self._state._chunks_by_id = {
            cid: c
            for cid, c in self._state._chunks_by_id.items()
            if c.document_id != document_id
        }
        self._rebuild()
        self._save()

    # ── Query operations ──────────────────────────────────────────────────────

    def query(self, query_text: str, top_k: int = 20) -> list[Chunk]:
        """
        BM25 retrieval. Returns chunks ranked by BM25 score descending.
        Low-scoring chunks (score=0) are excluded.
        """
        if self._state._bm25 is None or not self._state._chunk_ids:
            return []

        tokens = _tokenize(query_text)
        scores = self._state._bm25.get_scores(tokens)

        # Pair scores with chunk_ids and sort
        scored = sorted(
            zip(scores, self._state._chunk_ids),
            key=lambda x: x[0],
            reverse=True,
        )

        results: list[Chunk] = []
        for score, chunk_id in scored[:top_k]:
            if score <= 0:
                break
            chunk = self._state._chunks_by_id.get(chunk_id)
            if chunk:
                c = Chunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    source_filename=chunk.source_filename,
                    page_num=chunk.page_num,
                    chunk_type=chunk.chunk_type,
                    section_heading=chunk.section_heading,
                    text=chunk.text,
                    rerank_score=round(float(score), 4),
                )
                results.append(c)

        return results

    def count(self) -> int:
        return len(self._state._chunks_by_id)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _rebuild(self) -> None:
        """Rebuild the BM25 model over all current chunks."""
        all_chunks = list(self._state._chunks_by_id.values())
        if not all_chunks:
            self._state._bm25 = None
            self._state._chunk_ids = []
            return

        self._state._chunk_ids = [c.chunk_id for c in all_chunks]
        tokenized = [_tokenize(c.text) for c in all_chunks]
        self._state._bm25 = BM25Okapi(tokenized)

    def _save(self) -> None:
        with open(self.index_path, "wb") as f:
            pickle.dump(self._state, f)
        if self.gcs_bucket:
            _gcs_upload(self.gcs_bucket, self._GCS_OBJECT, self.index_path)

    def _load(self) -> None:
        if not os.path.exists(self.index_path) and self.gcs_bucket:
            _gcs_download(self.gcs_bucket, self._GCS_OBJECT, self.index_path)
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, "rb") as f:
                    self._state = pickle.load(f)
            except Exception:
                self._state = BM25Index()


def _gcs_upload(bucket_name: str, object_path: str, local_path: str) -> None:
    try:
        from google.cloud import storage
        storage.Client().bucket(bucket_name).blob(object_path).upload_from_filename(local_path)
    except Exception:
        pass  # GCS sync is best-effort; local file is always the source of truth


def _gcs_download(bucket_name: str, object_path: str, local_path: str) -> None:
    try:
        from google.cloud import storage
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        storage.Client().bucket(bucket_name).blob(object_path).download_to_filename(local_path)
    except Exception:
        pass  # If GCS download fails, _load() will start with an empty index


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenization for BM25."""
    import re
    # Keep alphanumeric, dollar amounts, percentages intact
    tokens = re.findall(r"\$[\d,]+|\d+\.?\d*%?|\w+", text.lower())
    return tokens
