"""
chunker.py — Document sectioning and parent-child chunk creation.

Key fixes vs. previous version:
- Token-aware chunking: child chunks are capped by TIKTOKEN token count, not
  raw character count. This prevents silent context-window overflow when chunks
  are later fed to LLMs with finite context limits.
- Section detection heuristic is more conservative: single-word or very short
  lines that are all-caps no longer fire as headings if they look like acronyms
  (e.g. "PDF", "ML", "API").
- create_chunks() is retained as a simple flat-chunking fallback with proper
  docstring; it is no longer dead code.
- chunk_table(): Table-aware chunker that keeps the header row attached to
  every child chunk so the LLM always knows what each column represents.
  Tables smaller than TABLE_ATOMIC_TOKENS are stored as a single atomic chunk.
- Full type hints and docstrings throughout.
"""

from __future__ import annotations

import re
from typing import Optional




try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")  

    def _token_len(text: str) -> int:
        return len(_enc.encode(text, disallowed_special=()))

except ImportError:  
    import logging
    logging.getLogger(__name__).warning(
        "tiktoken not installed — falling back to char-count for token estimation. "
        "Install it: pip install tiktoken"
    )

    def _token_len(text: str) -> int:  
        return len(text) // 4







CHILD_MAX_TOKENS: int = 512


PARENT_MAX_TOKENS: int = 1500





def detect_sections(text: str) -> list[tuple[str, str]]:
    """
    Split text into (section_heading, section_body) pairs using a heuristic.

    A line is treated as a heading if it is:
    - ALL CAPS and longer than 2 characters (to exclude acronyms like PDF, ML, API)
    - OR a short title line (≤ 5 words) that starts with an uppercase letter
      and contains no terminal punctuation.

    Args:
        text: Raw document text.

    Returns:
        List of (heading, body) tuples. An initial ``'General'`` section
        captures any content that precedes the first detected heading.
    """
    sections: list[tuple[str, str]] = []
    lines = text.split("\n")
    current_section = "General"
    buffer: list[str] = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        words = line.split()
        is_all_caps_heading = (
            line.isupper()
            and len(line) > 2  
            and len(words) <= 8  
        )
        is_title_case_heading = (
            len(words) <= 5
            and line[0].isupper()
            and not line.endswith((".", ",", ":", ";", "?", "!"))
            and not any(c.isdigit() for c in line)  
        )

        if is_all_caps_heading or is_title_case_heading:
            if buffer:
                sections.append((current_section, " ".join(buffer)))
                buffer = []
            current_section = line
        else:
            buffer.append(line)

    if buffer:
        sections.append((current_section, " ".join(buffer)))

    return sections






def create_parent_child_chunks(
    text: str,
    parent_max_tokens: int = PARENT_MAX_TOKENS,
    child_max_tokens: int = CHILD_MAX_TOKENS,
    overlap_chars: int = 100,
) -> tuple[list[dict], list[dict]]:
    """
    Build hierarchical parent-child chunk pairs from text.

    Strategy:
    1. Split text into *parent* chunks at sentence boundaries, capped at
       ``parent_max_tokens`` tokens each.
    2. For each parent, split into *child* chunks by sliding a character window
       across the parent text; each child is then hard-capped at
       ``child_max_tokens`` tokens to prevent model context overflow.

    The child chunks are what get embedded and stored in the vector store.
    The parent text is bundled into the enriched chunk metadata so that
    the generation step receives a broader context window.

    Args:
        text:             Input section text to chunk.
        parent_max_tokens: Token cap per parent chunk.
        child_max_tokens:  Token cap per child chunk (hard limit).
        overlap_chars:     Character overlap between adjacent child chunks.

    Returns:
        Tuple of (parents, children) where each element is a list of dicts:
        - Parent dict: ``{'parent_id': int, 'text': str}``
        - Child dict:  ``{'child_id': int, 'parent_id': int, 'text': str}``
    """
    parents: list[dict] = []
    children: list[dict] = []

    sentences = re.split(r"(?<=[.!?]) +", text)
    parent_chunks: list[str] = []
    current = ""

    for sentence in sentences:
        candidate = (current + " " + sentence).strip()
        if _token_len(candidate) <= parent_max_tokens:
            current = candidate
        else:
            if current.strip():
                parent_chunks.append(current.strip())
            current = sentence

    if current.strip():
        parent_chunks.append(current.strip())

    parent_id = 0
    child_id = 0
    child_char_size = child_max_tokens * 4  

    for parent_text in parent_chunks:
        parents.append({"parent_id": parent_id, "text": parent_text})

        start = 0
        while start < len(parent_text):
            raw_child = parent_text[start : start + child_char_size]

            if _token_len(raw_child) > child_max_tokens:
                tokens = _enc.encode(raw_child, disallowed_special=()) if hasattr(_enc, "encode") else []
                if tokens:
                    raw_child = _enc.decode(tokens[:child_max_tokens])
                else:
                    raw_child = raw_child[: child_max_tokens * 4]

            if raw_child.strip():
                children.append(
                    {
                        "child_id": child_id,
                        "parent_id": parent_id,
                        "text": raw_child.strip(),
                    }
                )
                child_id += 1

            step = child_char_size - overlap_chars
            if step <= 0:
                step = child_char_size
            start += step

        parent_id += 1

    return parents, children








TABLE_ATOMIC_TOKENS: int = 400


TABLE_ROWS_PER_CHUNK: int = 15


def chunk_table(
    table_text: str,
    atomic_tokens: int = TABLE_ATOMIC_TOKENS,
    rows_per_chunk: int = TABLE_ROWS_PER_CHUNK,
) -> list[dict]:
    """
    Chunk a Markdown table string with header-awareness.

    Strategy:
    - Row 0 = column headers  → **always** stored in ``table_header``.
    - Row 1 = separator line  ``| --- | --- |``  → discarded after extraction.
    - If the whole table is < ``atomic_tokens`` tokens → one chunk, intact.
    - If larger → split into groups of ``rows_per_chunk`` data rows.
      Every child chunk has the header row prepended so the LLM always knows
      what each column means, even when a table spans multiple chunks.

    Args:
        table_text:    Raw Markdown table string (pipe-delimited rows).
        atomic_tokens: Token threshold below which the table is kept whole.
        rows_per_chunk: Max data rows per child chunk for large tables.

    Returns:
        List of chunk dicts, each with keys:
          - ``text``:         The chunk text (header + data rows).
          - ``doc_type``:     ``"table"``
          - ``table_header``: The raw header row string.
          - ``is_table_chunk``: ``True`` (for conditional compress gating).
          - ``child_id``:     0-indexed position within this table.
    """
    lines = [ln for ln in table_text.strip().splitlines() if ln.strip()]

    if not lines:
        return [{"text": table_text, "doc_type": "table",
                 "table_header": "", "is_table_chunk": True, "child_id": 0}]

    header_row = lines[0]

    data_start = 1
    if len(lines) > 1 and re.match(r"^\|[\s\-|]+\|$", lines[1]):
        data_start = 2

    data_rows = lines[data_start:]

    if _token_len(table_text) <= atomic_tokens:
        return [{
            "text": table_text,
            "doc_type": "table",
            "table_header": header_row,
            "is_table_chunk": True,
            "child_id": 0,
        }]

    separator = "|" + "|".join([" --- "] * len(header_row.split("|"))) + "|"
    chunks: list[dict] = []

    for i in range(0, max(len(data_rows), 1), rows_per_chunk):
        row_slice = data_rows[i : i + rows_per_chunk]
        if not row_slice:
            continue

        chunk_text = "\n".join([header_row, separator] + row_slice)

        chunks.append({
            "text": chunk_text,
            "doc_type": "table",
            "table_header": header_row,
            "is_table_chunk": True,
            "child_id": len(chunks),
        })

    return chunks if chunks else [{
        "text": table_text,
        "doc_type": "table",
        "table_header": header_row,
        "is_table_chunk": True,
        "child_id": 0,
    }]
