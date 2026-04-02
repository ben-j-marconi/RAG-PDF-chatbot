"""
llm/vertex_provider.py — Google Gemini provider using the google-genai SDK.

Uses Google's unified GenAI SDK (google-genai), which supports both:
  - Gemini API (API key auth) — for development
  - Vertex AI backend (GCP project auth) — for production

━━━ HOW TO ACTIVATE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Option A — Gemini API key (development):
  GEMINI_API_KEY=your-api-key

Option B — Vertex AI backend (production):
  GOOGLE_CLOUD_PROJECT=your-gcp-project-id
  GOOGLE_CLOUD_LOCATION=us-central1
  Then: gcloud auth application-default login

Set LLM_PROVIDER=vertex in .env and restart. Zero code changes required.

━━━ ARCHITECTURE NOTE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The pluggable provider pattern means the entire generation layer is
decoupled from the rest of the system. The chat engine, retrieval layer,
and UI call generate_answer(query, context, config) and never import
any LLM SDK directly. Switching between providers is a single environment
variable change — no application code changes.

In a full GCP production deployment:
  - Swap the embedding model to text-embedding-004 (Vertex AI Embeddings API)
  - Replace ChromaDB with Vertex AI Vector Search (managed, billion-scale)
  - Replace the cross-encoder with Vertex AI Ranking API
  - Add IAM-enforced ACL metadata filtering to the retrieval layer
  - Use Cloud Logging for structured observability

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import warnings
from typing import Optional

from src.chunking.chunker import Chunk
from src.llm.base import LLMProvider, AnswerResult, SYSTEM_PROMPT


class VertexProvider(LLMProvider):
    """
    Google Gemini provider using the google-genai SDK.

    Supports both Gemini API (API key) and Vertex AI (GCP project) backends.
    Lazy-initializes on first call so the app starts without credentials
    and fails gracefully at query time.
    """

    def __init__(self, config):
        self.config = config
        self._client = None

    def _ensure_initialized(self):
        if self._client is not None:
            return

        try:
            from google import genai
        except ImportError:
            raise RuntimeError(
                "google-genai is not installed.\n"
                "Run: pip install google-genai"
            )

        warnings.filterwarnings("ignore", category=FutureWarning)

        if self.config.gemini_api_key:
            # Gemini API path (API key auth)
            self._client = genai.Client(api_key=self.config.gemini_api_key)
        elif self.config.gcp_project:
            # Vertex AI path (GCP project + application default credentials)
            self._client = genai.Client(
                vertexai=True,
                project=self.config.gcp_project,
                location=self.config.gcp_location,
            )
        else:
            raise RuntimeError(
                "No credentials configured for Gemini provider.\n"
                "Set GEMINI_API_KEY or GOOGLE_CLOUD_PROJECT in .env"
            )

    def generate_answer(
        self,
        query: str,
        context_chunks: list[Chunk],
        chat_history: Optional[list[dict]] = None,
    ) -> AnswerResult:
        self._ensure_initialized()

        if not context_chunks:
            return AnswerResult(
                answer="The available documents do not contain sufficient information to answer this question.",
                citations=[],
                confidence="insufficient_evidence",
            )

        context_str = self._build_context_string(context_chunks)
        citations = self._build_citations(context_chunks)

        full_prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Evidence from Real Estate Co documents:\n\n"
            f"{context_str}\n\n"
            f"---\n\n"
            f"Question: {query}"
        )

        try:
            from google.genai import types
            response = self._client.models.generate_content(
                model=self.config.vertex_model,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=1024,
                ),
            )
            raw_answer = response.text or ""
        except Exception as e:
            raise RuntimeError(f"Gemini generation failed: {e}")

        from src.llm.openai_provider import _parse_answer
        return _parse_answer(raw_answer, context_chunks, citations)
