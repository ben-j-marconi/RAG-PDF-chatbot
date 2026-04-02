"""
llm/base.py — Abstract LLM provider interface.

Defines the contract that all LLM providers must implement.
The generate_answer() method is the single seam where the generation
backend can be swapped — from OpenAI to Vertex AI Gemini — by changing
one environment variable.

Design rationale:
  This is the architecture that makes the prototype production-ready.
  The chatbot logic in chat/engine.py never imports OpenAI or Vertex AI
  directly — it only calls generate_answer(). Adding a new LLM backend
  is implementing one class and registering it in get_provider().
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from src.chunking.chunker import Chunk


@dataclass
class AnswerResult:
    """
    The structured output of the LLM generation step.
    Carries the answer, citations, confidence, and suggested follow-ups.
    """
    answer: str
    citations: list[dict]            # [{"filename": ..., "page": ..., "chunk_type": ...}]
    confidence: str                  # "high" | "medium" | "low" | "insufficient_evidence"
    follow_up_questions: list[str] = field(default_factory=list)
    raw_context_used: int = 0        # Number of chunks included in context


class LLMProvider(ABC):
    """Abstract base class for all LLM providers."""

    @abstractmethod
    def generate_answer(
        self,
        query: str,
        context_chunks: list[Chunk],
        chat_history: Optional[list[dict]] = None,
    ) -> AnswerResult:
        """
        Generate a grounded answer using the provided context chunks.

        Args:
            query: The user's question
            context_chunks: Reranked chunks from the retrieval layer
            chat_history: Optional list of {"role": ..., "content": ...} dicts

        Returns:
            AnswerResult with answer, citations, confidence, and follow-ups
        """
        ...

    def _build_context_string(self, chunks: list[Chunk]) -> str:
        """Format chunks into a numbered context block for the prompt."""
        parts = []
        for i, chunk in enumerate(chunks, start=1):
            heading = f" [{chunk.section_heading}]" if chunk.section_heading else ""
            parts.append(
                f"[Evidence {i}] {chunk.source_filename}, Page {chunk.page_num}"
                f"{heading} ({chunk.chunk_type}):\n{chunk.text}"
            )
        return "\n\n---\n\n".join(parts)

    def _build_citations(self, chunks: list[Chunk]) -> list[dict]:
        """Build citation objects from the chunks used as context."""
        return [
            {
                "evidence_num": i,
                "filename": chunk.source_filename,
                "page": chunk.page_num,
                "chunk_type": chunk.chunk_type,
                "section": chunk.section_heading or "—",
                "rerank_score": chunk.rerank_score,
                "preview": chunk.text[:120] + ("..." if len(chunk.text) > 120 else ""),
                "full_text": chunk.text,
            }
            for i, chunk in enumerate(chunks, start=1)
        ]


# ── Provider factory ──────────────────────────────────────────────────────────

def get_provider(config) -> LLMProvider:
    """
    Return the appropriate LLM provider based on config.llm_provider.
    Supports "openai" (default) and "vertex" (production Gemini).
    """
    provider_name = config.llm_provider.lower()

    if provider_name == "openai":
        from src.llm.openai_provider import OpenAIProvider
        return OpenAIProvider(config)
    elif provider_name == "vertex":
        from src.llm.vertex_provider import VertexProvider
        return VertexProvider(config)
    else:
        raise ValueError(
            f"Unknown LLM provider: '{provider_name}'. "
            f"Set LLM_PROVIDER=openai or LLM_PROVIDER=vertex in .env"
        )


# ── Shared prompt template ────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a Property Valuation Assistant for Real Estate Co.
You help property analysts and underwriters find accurate, evidence-backed answers
from the company's internal valuation documents.

STRICT GROUNDING RULES — follow these without exception:
1. Answer ONLY using the numbered Evidence sections provided. Never use external knowledge.
2. Cite every factual claim with [Evidence N] where N is the evidence number.
3. For every financial figure (dollar amounts, cap rates, NOI, LTV, DSCR, percentages),
   state the source document name and page number in addition to the evidence citation.
4. If the evidence does not contain enough information to answer, respond:
   "The available documents do not contain sufficient information to answer this question."
   Then briefly state what evidence IS available on related topics.
5. Never invent, interpolate, or extrapolate property facts not stated in the evidence.
6. If two documents contain conflicting figures, present both and note the discrepancy.
7. Be concise and precise — these answers inform investment and underwriting decisions.

Format your response exactly as follows:

**Answer:** [your grounded answer with inline [Evidence N] citations for every claim]

**Confidence:** [High | Medium | Low | Insufficient Evidence]
- High: evidence directly answers the question with specific figures
- Medium: evidence is relevant but incomplete or indirect
- Low: evidence only partially addresses the question
- Insufficient Evidence: the evidence does not address the question

**Suggested Follow-ups:**
- [one specific follow-up question]
- [one specific follow-up question]
"""
