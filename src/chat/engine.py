"""
chat/engine.py — Chat engine: orchestrates retrieval → rerank → generate.

This is the "brains" of the chatbot — it wires together all pipeline stages
into a single ask() call that the Streamlit UI invokes per user message.

Data flow:
  user query
    → HybridRetriever.retrieve() [vector + BM25 + RRF merge]
    → Reranker.rerank() [cross-encoder top-k selection]
    → LLMProvider.generate_answer() [grounded answer with citations]
    → AnswerResult [returned to UI]

Design rationale:
  The engine is intentionally thin. It doesn't contain business logic —
  it's a coordinator. This makes it easy to test each stage independently
  and to swap any stage (retriever, reranker, LLM) without affecting others.
"""

import logging
from typing import Optional

from config import Config
from src.retrieval.retriever import HybridRetriever
from src.retrieval.reranker import Reranker
from src.llm.base import get_provider, AnswerResult
from src.observability.logger import (
    get_logger,
    check_retrieval_relevance,
    check_groundedness,
)


class ChatEngine:
    """
    Coordinates the retrieval and generation pipeline for a single query.
    Stateless — chat history is passed in by the caller (Streamlit session state).
    """

    def __init__(self, config: Config):
        self.config = config
        self.retriever = HybridRetriever(config)
        self.reranker = Reranker(config)
        self.llm = get_provider(config)
        self.logger = get_logger("chat.engine", config)

    def ask(
        self,
        query: str,
        chat_history: Optional[list[dict]] = None,
        metadata_filter: Optional[dict] = None,
    ) -> AnswerResult:
        """
        Full RAG pipeline for a user query.

        Args:
            query: The user's natural language question
            chat_history: Prior conversation turns for multi-turn context
            metadata_filter: Optional ChromaDB-compatible filter to scope retrieval
                             e.g. {"chunk_type": "table"} to prefer tables
                             e.g. {"source_filename": "01_Rivergate_..."} to scope doc

        Returns:
            AnswerResult with answer, citations, confidence, and follow-ups
        """
        self.logger.info("chat_query_received", extra={
            "event": "chat_query_received",
            "query_preview": query[:100],
        })

        # 1. Hybrid retrieval
        candidates = self.retriever.retrieve(
            query,
            top_k=30,
            metadata_filter=metadata_filter,
        )

        if not candidates:
            self.logger.warning("no_candidates_retrieved", extra={
                "event": "no_candidates_retrieved",
                "query": query,
            })
            return AnswerResult(
                answer=(
                    "No relevant documents found. Make sure PDFs have been ingested "
                    "by running `python ingest.py` or clicking Re-ingest in the sidebar."
                ),
                citations=[],
                confidence="insufficient_evidence",
                follow_up_questions=["Have the documents been indexed yet?"],
            )

        # 2. Rerank
        top_chunks = self.reranker.rerank(query, candidates)

        # 3. Observability: log retrieval quality
        check_retrieval_relevance(query, top_chunks, self.logger)

        # 4. Generate grounded answer
        result = self.llm.generate_answer(
            query=query,
            context_chunks=top_chunks,
            chat_history=chat_history,
        )

        # 5. Observability: groundedness check
        check_groundedness(result.answer, top_chunks, self.logger)

        self.logger.info("answer_generated", extra={
            "event": "answer_generated",
            "confidence": result.confidence,
            "citations_count": len(result.citations),
            "context_chunks_used": result.raw_context_used,
        })

        return result

    def ask_with_chunks(
        self,
        query: str,
        chat_history: Optional[list[dict]] = None,
        metadata_filter: Optional[dict] = None,
    ) -> tuple[AnswerResult, list]:
        """
        Same as ask() but also returns the reranked chunks for UI debug display.
        The UI stores these in session state so the debug panel can show
        exactly what the model saw — crucial for full pipeline transparency.
        """
        self.logger.info("chat_query_received", extra={
            "event": "chat_query_received",
            "query_preview": query[:100],
        })

        candidates = self.retriever.retrieve(query, top_k=30, metadata_filter=metadata_filter)

        if not candidates:
            self.logger.warning("no_candidates_retrieved", extra={
                "event": "no_candidates_retrieved", "query": query,
            })
            empty_result = AnswerResult(
                answer=(
                    "No relevant documents found. Make sure PDFs have been ingested "
                    "by running `python3 ingest.py` or clicking Re-ingest in the sidebar."
                ),
                citations=[],
                confidence="insufficient_evidence",
                follow_up_questions=["Have the documents been indexed yet?"],
            )
            return empty_result, []

        top_chunks = self.reranker.rerank(query, candidates)
        check_retrieval_relevance(query, top_chunks, self.logger)

        result = self.llm.generate_answer(
            query=query,
            context_chunks=top_chunks,
            chat_history=chat_history,
        )

        check_groundedness(result.answer, top_chunks, self.logger)

        self.logger.info("answer_generated", extra={
            "event": "answer_generated",
            "confidence": result.confidence,
            "citations_count": len(result.citations),
            "context_chunks_used": result.raw_context_used,
        })

        return result, top_chunks
