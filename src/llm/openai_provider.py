"""
llm/openai_provider.py — OpenAI LLM provider (dev/default mode).

Implements the LLMProvider interface using the OpenAI Python SDK.
Default model: gpt-4o-mini (fast, cheap, sufficient for demo quality).

To use: set OPENAI_API_KEY in .env and LLM_PROVIDER=openai.
"""

import re
from typing import Optional

from openai import OpenAI

from src.chunking.chunker import Chunk
from src.llm.base import LLMProvider, AnswerResult, SYSTEM_PROMPT


class OpenAIProvider(LLMProvider):
    """OpenAI-backed LLM provider for development and demo use."""

    def __init__(self, config):
        self.config = config
        self.client = OpenAI(api_key=config.openai_api_key)
        self.model = config.openai_model

    def generate_answer(
        self,
        query: str,
        context_chunks: list[Chunk],
        chat_history: Optional[list[dict]] = None,
    ) -> AnswerResult:
        if not context_chunks:
            return AnswerResult(
                answer="I couldn't find relevant evidence in the indexed documents to answer this question.",
                citations=[],
                confidence="insufficient_evidence",
                follow_up_questions=["Can you rephrase your question?",
                                     "Which document are you looking for information from?"],
            )

        context_str = self._build_context_string(context_chunks)

        user_message = f"""Evidence from Real Estate Co documents:

{context_str}

---

Question: {query}"""

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Include recent chat history for multi-turn context
        if chat_history:
            messages.extend(chat_history[-4:])  # Last 2 turns

        messages.append({"role": "user", "content": user_message})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1,   # Low temperature for factual, grounded answers
            max_tokens=1024,
        )

        raw_answer = response.choices[0].message.content or ""
        return _parse_answer(raw_answer, context_chunks, self._build_citations(context_chunks))


def _parse_answer(raw: str, chunks: list[Chunk], citations: list[dict]) -> AnswerResult:
    """Parse the structured LLM response into an AnswerResult."""
    answer = raw
    confidence = "medium"
    follow_ups: list[str] = []

    # Extract confidence
    conf_match = re.search(r"\*\*Confidence:\*\*\s*(.+)", raw, re.IGNORECASE)
    if conf_match:
        conf_text = conf_match.group(1).strip().lower()
        if "high" in conf_text:
            confidence = "high"
        elif "low" in conf_text:
            confidence = "low"
        elif "insufficient" in conf_text:
            confidence = "insufficient_evidence"

    # Extract follow-up questions
    fu_match = re.search(
        r"\*\*Suggested Follow-ups:\*\*(.+?)(?:\n\n|\Z)", raw, re.DOTALL | re.IGNORECASE
    )
    if fu_match:
        fu_block = fu_match.group(1)
        follow_ups = [
            line.strip().lstrip("-•").strip()
            for line in fu_block.split("\n")
            if line.strip().lstrip("-•").strip()
        ]

    # Clean up the answer (remove the metadata sections for the main display)
    answer_match = re.search(r"\*\*Answer:\*\*(.+?)(?=\*\*Confidence:|$)", raw, re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()

    return AnswerResult(
        answer=answer,
        citations=citations,
        confidence=confidence,
        follow_up_questions=follow_ups[:2],
        raw_context_used=len(chunks),
    )
