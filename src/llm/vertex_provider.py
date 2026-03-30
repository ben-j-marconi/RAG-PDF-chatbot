"""
llm/vertex_provider.py — Vertex AI Gemini provider (production mode).

Implements the LLMProvider interface using the Vertex AI SDK and Gemini models.
This is the production path for Google Cloud deployment.

━━━ HOW TO ACTIVATE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Install the SDK:
   pip install google-cloud-aiplatform

2. Authenticate locally:
   gcloud auth application-default login

3. Set environment variables in .env:
   LLM_PROVIDER=vertex
   GOOGLE_CLOUD_PROJECT=your-gcp-project-id
   GOOGLE_CLOUD_LOCATION=us-central1
   VERTEX_MODEL=gemini-2.0-flash-001

4. Restart the app. Zero code changes required.

━━━ ARCHITECTURE NOTE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The pluggable provider pattern means the entire generation layer is
decoupled from the rest of the system. The chat engine, retrieval layer,
and UI call generate_answer(query, context, config) and never import
OpenAI or Vertex AI directly. Switching from gpt-4o-mini to Gemini on
Vertex AI is a single environment variable change — no code changes.

In a full GCP production deployment:
  - Swap the embedding model to text-embedding-004 (Vertex AI Embeddings API)
  - Replace ChromaDB with Vertex AI Vector Search (managed, billion-scale)
  - Replace the cross-encoder with Vertex AI Ranking API (Vertex AI Reranker)
  - Add IAM-enforced ACL metadata filtering to the retrieval layer
  - Use Cloud Logging for structured observability

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from typing import Optional

from src.chunking.chunker import Chunk
from src.llm.base import LLMProvider, AnswerResult, SYSTEM_PROMPT


class VertexProvider(LLMProvider):
    """
    Vertex AI Gemini provider.

    Lazy-initializes the Vertex AI SDK on first call so the app can
    start without GCP credentials and fail gracefully at query time.
    """

    def __init__(self, config):
        self.config = config
        self._model = None

    def _ensure_initialized(self):
        if self._model is not None:
            return

        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel
        except ImportError:
            raise RuntimeError(
                "google-cloud-aiplatform is not installed.\n"
                "Run: pip install google-cloud-aiplatform\n"
                "Then: gcloud auth application-default login"
            )

        if not self.config.gcp_project:
            raise RuntimeError(
                "GOOGLE_CLOUD_PROJECT is not set in .env.\n"
                "Add: GOOGLE_CLOUD_PROJECT=your-gcp-project-id"
            )

        try:
            vertexai.init(
                project=self.config.gcp_project,
                location=self.config.gcp_location,
            )
            self._model = GenerativeModel(self.config.vertex_model)
        except Exception as e:
            raise RuntimeError(
                f"Vertex AI initialization failed: {e}\n"
                f"Ensure you have run: gcloud auth application-default login\n"
                f"Project: {self.config.gcp_project} · Location: {self.config.gcp_location}"
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

        # Gemini takes a single string prompt (no separate system role in basic API)
        full_prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Evidence from Real Estate Co documents:\n\n"
            f"{context_str}\n\n"
            f"---\n\n"
            f"Question: {query}"
        )

        try:
            from vertexai.generative_models import GenerationConfig
            response = self._model.generate_content(
                full_prompt,
                generation_config=GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=1024,
                ),
            )
            raw_answer = response.text or ""
        except Exception as e:
            raise RuntimeError(f"Vertex AI generation failed: {e}")

        from src.llm.openai_provider import _parse_answer
        return _parse_answer(raw_answer, context_chunks, citations)
