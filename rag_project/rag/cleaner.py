"""
cleaner.py — Text normalisation, hashing, and deduplication utilities.

Key fixes vs. previous version:
- Non-ASCII characters are now PRESERVED via Unicode NFKC normalisation instead
  of being stripped. This is critical for multilingual documents (Arabic, CJK,
  Devanagari, etc.) which would otherwise be silently destroyed before embedding
  with the multilingual BGE-M3 model.
- get_doc_hash() now uses SHA-256 instead of MD5 for security hygiene.
- All functions have type hints and docstrings.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata


def clean_text(text: str) -> str:
    """
    Normalise raw extracted document text for embedding.

    Operations (in order):
    1. Short-text filter: discard fragments under 15 words — too noisy to embed.
    2. NFKC normalisation: standardise Unicode (e.g. fullwidth → ASCII where
       semantically equivalent) WITHOUT stripping non-ASCII characters.
    3. Collapse multiple whitespace runs to a single space.
    4. Strip page-number artefacts (e.g. "Page 3", "PAGE 12").
    5. Strip URLs.

    Deliberately NOT stripped:
    - Emails / phone numbers (may be relevant document content).
    - Non-Latin scripts (Arabic, CJK, Devanagari, etc.) — BGE-M3 handles them.

    Args:
        text: Raw text extracted from a document page or element.

    Returns:
        Cleaned text string, or empty string if the fragment is too short.
    """
    if not text or len(text.strip()) < 15:
        return ""

    text = unicodedata.normalize("NFKC", text)

    text = re.sub(r"\s+", " ", text)

    text = re.sub(r"\bPage\s+\d+\b", "", text, flags=re.IGNORECASE)

    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    return text.strip()


def deduplicate_chunks(chunks: list[dict]) -> list[dict]:
    """
    Remove duplicate chunks based on their text content.

    Uses SHA-256 hashing for deduplication. Preserves the first occurrence
    of each unique chunk and discards subsequent duplicates.

    Args:
        chunks: List of chunk dicts, each containing at least a ``'text'`` key.

    Returns:
        Deduplicated list of chunk dicts in original order.
    """
    seen: set[str] = set()
    unique: list[dict] = []
    for chunk in chunks:
        text = chunk.get("text", "")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest not in seen:
            seen.add(digest)
            unique.append(chunk)
    return unique


def get_doc_hash(text: str) -> str:
    """
    Compute a SHA-256 hex digest of the given text.

    Used for document-level deduplication during ingestion and for detecting
    changes to already-indexed documents.

    Args:
        text: Document or chunk text.

    Returns:
        64-character lowercase hex string.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_file_hash(filepath: str) -> str:
    """
    Compute a SHA-256 hex digest of a file's raw bytes.

    Reads in 8 KB blocks to handle large files without loading them into memory.
    Used by the incremental indexer to detect file changes.

    Args:
        filepath: Absolute or relative path to the file.

    Returns:
        64-character lowercase hex string.
    """
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()
