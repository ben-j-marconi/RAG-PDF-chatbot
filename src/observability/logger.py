"""
observability/logger.py — Structured logging and evaluation hooks.

Design rationale:
  Observability is what separates a prototype from a trustworthy system.
  Every pipeline stage emits a structured event so we can audit ingestion
  quality, retrieval relevance, and groundedness — all visible in the logs
  and surfaced in the UI debug panel.
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any


def get_logger(name: str, config=None) -> logging.Logger:
    """Return a logger that emits JSON-formatted structured events."""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # Already configured

    level = logging.INFO
    if config:
        level = getattr(logging, config.log_level.upper(), logging.INFO)

    logger.setLevel(level)

    formatter = _JsonFormatter()

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler (if configured)
    if config and config.log_file:
        os.makedirs(os.path.dirname(config.log_file), exist_ok=True)
        fh = logging.FileHandler(config.log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.propagate = False
    return logger


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Attach any extra structured fields the caller passed
        # These are all standard LogRecord attributes — skip them
        _SKIP = frozenset({
            "name", "msg", "args", "levelname", "levelno", "pathname",
            "filename", "module", "exc_info", "exc_text", "stack_info",
            "lineno", "funcName", "created", "msecs", "relativeCreated",
            "thread", "threadName", "processName", "process", "message",
            "taskName",
        })
        for key, val in record.__dict__.items():
            if key.startswith("_") or key in _SKIP:
                continue
            try:
                # Only include JSON-serializable values
                import json as _json
                _json.dumps(val)
                payload[key] = val
            except (TypeError, ValueError):
                payload[key] = str(val)
        return json.dumps(payload)


# ── Evaluation hooks ──────────────────────────────────────────────────────────

def check_extraction_quality(doc_obj: Any, logger: logging.Logger) -> dict:
    """
    Basic sanity checks after parsing a document.
    Returns a quality summary dict and logs it.
    Note: "We check extraction quality before indexing so bad parse
    results don't silently pollute the vector store."
    """
    pages = getattr(doc_obj, "pages", [])
    total_chars = sum(len(p.raw_text) for p in pages if hasattr(p, "raw_text"))
    table_count = sum(len(p.tables) for p in pages if hasattr(p, "tables"))
    empty_pages = sum(1 for p in pages if hasattr(p, "raw_text") and not p.raw_text.strip())

    quality = {
        "event": "extraction_quality_check",
        "total_pages": len(pages),
        "total_chars": total_chars,
        "table_count": table_count,
        "empty_pages": empty_pages,
        "passed": total_chars > 100,
    }
    log_level = logging.INFO if quality["passed"] else logging.WARNING
    logger.log(log_level, "extraction_quality_check", extra=quality)
    return quality


def check_retrieval_relevance(query: str, chunks: list, logger: logging.Logger) -> dict:
    """
    Log retrieval results for audit. In production this would compare
    against a golden eval set. For MVP we log the top chunk scores.
    Note: "We emit retrieval events so we can spot when the wrong
    document type is surfacing for a given query class."
    """
    result = {
        "event": "retrieval_relevance_check",
        "query_preview": query[:80],
        "chunks_returned": len(chunks),
        "top_sources": [
            {
                "filename": getattr(c, "source_filename", "?"),
                "page": getattr(c, "page_num", "?"),
                "chunk_type": getattr(c, "chunk_type", "?"),
                "score": getattr(c, "rerank_score", None),
            }
            for c in chunks[:3]
        ],
    }
    logger.info("retrieval_relevance_check", extra=result)
    return result


def check_groundedness(answer: str, chunks: list, logger: logging.Logger) -> dict:
    """
    Lightweight groundedness check: look for numeric values from retrieved
    chunks in the answer. A grounded answer should echo source numbers.
    Note: "This isn't a full faithfulness model — it's a heuristic
    that catches hallucinations of numbers not present in the evidence."
    """
    import re
    source_numbers = set()
    for c in chunks:
        text = getattr(c, "text", "")
        source_numbers.update(re.findall(r"\$[\d,]+|\d+\.\d+%|\d+x", text))

    answer_numbers = set(re.findall(r"\$[\d,]+|\d+\.\d+%|\d+x", answer))
    unsupported = answer_numbers - source_numbers

    result = {
        "event": "groundedness_check",
        "source_numbers_found": len(source_numbers),
        "answer_numbers_found": len(answer_numbers),
        "potentially_unsupported_numbers": list(unsupported),
        "passed": len(unsupported) == 0,
    }
    log_level = logging.INFO if result["passed"] else logging.WARNING
    logger.log(log_level, "groundedness_check", extra=result)
    return result
