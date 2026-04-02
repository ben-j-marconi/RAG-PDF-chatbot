"""
config.py — Central configuration loader.

Reads from environment variables (populated via .env).
All modules import `get_config()` rather than reading os.environ directly.
This makes the Vertex AI swap trivial: change LLM_PROVIDER in .env and restart.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # LLM provider
    llm_provider: str = "openai"          # "openai" | "vertex"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    gemini_api_key: str = ""
    gcp_project: str = ""
    gcp_location: str = "us-central1"
    vertex_model: str = "models/gemini-2.5-flash"

    # Vector store backend
    vector_store_backend: str = "chroma"   # "chroma" | "firestore"

    # Google Cloud Storage (optional — falls back to local if not set)
    gcs_bucket: str = ""                  # e.g. "rec-valuation-docs"
    gcs_pdf_prefix: str = "pdfs/"        # prefix within the bucket

    # Paths (local)
    pdf_dir: str = "data/pdfs"
    registry_path: str = "data/registry/document_registry.json"
    chroma_dir: str = "data/chroma_db"
    bm25_index_path: str = "data/registry/bm25_index.pkl"

    # Retrieval
    vector_top_k: int = 20
    bm25_top_k: int = 20
    rerank_top_k: int = 5

    # Models
    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Logging
    log_level: str = "INFO"
    log_file: str = "data/rag_chatbot.log"


def get_config() -> Config:
    return Config(
        llm_provider=os.getenv("LLM_PROVIDER", "openai"),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        gcp_project=os.getenv("GOOGLE_CLOUD_PROJECT", ""),
        gcp_location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
        vertex_model=os.getenv("VERTEX_MODEL", "models/gemini-2.5-flash"),
        vector_store_backend=os.getenv("VECTOR_STORE_BACKEND", "chroma"),
        gcs_bucket=os.getenv("GCS_BUCKET", ""),
        gcs_pdf_prefix=os.getenv("GCS_PDF_PREFIX", "pdfs/"),
        pdf_dir=os.getenv("PDF_DIR", "data/pdfs"),
        registry_path=os.getenv("REGISTRY_PATH", "data/registry/document_registry.json"),
        chroma_dir=os.getenv("CHROMA_DIR", "data/chroma_db"),
        bm25_index_path=os.getenv("BM25_INDEX_PATH", "data/registry/bm25_index.pkl"),
        vector_top_k=int(os.getenv("VECTOR_TOP_K", "20")),
        bm25_top_k=int(os.getenv("BM25_TOP_K", "20")),
        rerank_top_k=int(os.getenv("RERANK_TOP_K", "5")),
        embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        reranker_model=os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        log_file=os.getenv("LOG_FILE", "data/rag_chatbot.log"),
    )
