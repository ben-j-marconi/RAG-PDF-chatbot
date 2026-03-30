"""
parsing/cleaner.py — Text normalization and artifact removal.

Cleans raw extracted text before structure reconstruction.
Targets artifacts common in real estate PDFs:
  - Repeated headers/footers with company/document names
  - Page number lines
  - Broken line wraps from PDF rendering
  - Excessive whitespace
  - OCR-style noise (double spaces, stray characters)

Design rationale:
  This is the step that makes or breaks retrieval quality. The header
  "Real Estate Co - Internal Valuation Materials Rivergate Appraisal"
  appears on nearly every page. Without stripping it, it appears in
  almost every chunk and becomes a high-IDF noise token that misleads
  both BM25 and cosine similarity. Cleaning before indexing is the
  difference between "finding everything" and "finding the right thing."
"""

import re


# ── Header/footer patterns to remove line-by-line ────────────────────────────
# Ordered from most specific to most general.
# Each pattern is anchored to match the whole stripped line.

_STRIP_PATTERNS = [
    # Real Estate Co document headers
    r"^Real Estate Co\s*[-–]\s*Internal Valuation Materials",
    r"^Confidential\s*[-–]\s*Internal dataset",
    r"^Confidential\s*&\s*Proprietary\.?$",

    # Page number lines: "Page 1", "Page 2 of 5", standalone digits
    r"^Page\s+\d+(\s+of\s+\d+)?$",
    r"^\d+\s*$",

    # Google / careers footer artifacts
    r"careers\.google\.com",
    r"^Google\s*$",
    r"^Google Cloud\s*$",

    # Generic repeated document title artifacts (from pdfplumber header extraction)
    # Matches lines that are ONLY the document title repeated
    r"^Rivergate Apartments\s*[-–]\s*PCA Summary",
    r"^Rivergate Appraisal\s*$",
    r"^Rivergate Rent Roll\s*$",
    r"^Market Comps Memo\s*$",
    r"^Condition Assessment\s*$",
    r"^Underwriting Narrative\s*$",

    # Lines that are just "Confidential" or similar
    r"^Confidential\s*$",
]

_STRIP_RE = re.compile(
    "|".join(f"(?:{p})" for p in _STRIP_PATTERNS),
    re.IGNORECASE,
)

# Broken line wrap: newline followed by a lowercase letter that isn't a
# list item. We rejoin these into continuous sentences.
_BROKEN_WRAP_RE = re.compile(r"(?<![.!?:,])\n(?=[a-z])")

# Three or more consecutive newlines → two newlines (paragraph break)
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")

# Multiple spaces → single space
_MULTI_SPACE_RE = re.compile(r" {2,}")


def clean_text(raw_text: str) -> str:
    """
    Full cleaning pipeline for raw page text.
    Returns cleaned text preserving paragraph structure.
    """
    if not raw_text:
        return ""

    # 1. Strip header/footer lines
    lines = raw_text.split("\n")
    lines = [ln for ln in lines if not _is_noise_line(ln)]
    text = "\n".join(lines)

    # 2. Rejoin broken line wraps (mid-sentence line breaks from PDF rendering)
    text = _BROKEN_WRAP_RE.sub(" ", text)

    # 3. Normalize paragraph spacing
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)

    # 4. Normalize horizontal whitespace
    text = _MULTI_SPACE_RE.sub(" ", text)

    # 5. Strip trailing whitespace per line, then overall
    text = "\n".join(ln.rstrip() for ln in text.split("\n"))
    text = text.strip()

    return text


def _is_noise_line(line: str) -> bool:
    """Return True if this line should be discarded as a header/footer artifact."""
    stripped = line.strip()
    if not stripped:
        return False  # Blank lines are handled by multi-newline collapse, not stripped
    return bool(_STRIP_RE.search(stripped))


def table_rows_to_markdown(rows: list[list[str]]) -> str:
    """
    Convert a 2D list of table cells to a Markdown table string.
    First row is treated as the header row.

    Design rationale:
      Tables are stored as Markdown because:
      1. LLMs understand Markdown tables natively — no special handling needed
      2. Markdown survives chunking and remains human-readable in the debug panel
      3. BM25 can match numeric values (cap rates, dollar amounts) within the text
      4. The table structure isn't lost the way it is with flat text extraction
    """
    if not rows:
        return ""

    # Remove fully empty rows
    rows = [r for r in rows if any(c.strip() for c in r)]
    if not rows:
        return ""

    # Normalize column count across all rows
    max_cols = max(len(r) for r in rows)
    rows = [r + [""] * (max_cols - len(r)) for r in rows]

    def _cell(s: str) -> str:
        return s.strip().replace("|", "\\|").replace("\n", " ")

    header_row = "| " + " | ".join(_cell(c) for c in rows[0]) + " |"
    separator  = "| " + " | ".join("---" for _ in rows[0]) + " |"
    body_rows  = "\n".join(
        "| " + " | ".join(_cell(c) for c in row) + " |"
        for row in rows[1:]
    )

    parts = [header_row, separator]
    if body_rows:
        parts.append(body_rows)
    return "\n".join(parts)
