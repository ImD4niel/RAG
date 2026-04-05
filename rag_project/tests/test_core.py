"""
Tests for helper modules cleaner.py and chunker.py.

Designed for pytest execution, targeting edge cases and basic logic.
"""

from rag.cleaner import clean_text, deduplicate_chunks
from rag.chunker import detect_sections, create_parent_child_chunks

def test_clean_text_short():
    """Text under 15 words should be dropped."""
    assert clean_text("Short text.") == ""

def test_clean_text_nfkc_multilingual():
    """Non-ASCII should be preserved, double spaces collapsed."""
    assert clean_text("Hello  世界 and 15 words or more padding padding padding padding padding padding padding padding padding") == "Hello 世界 and 15 words or more padding padding padding padding padding padding padding padding padding"

def test_deduplicate_chunks_same_text():
    chunks = [{"text": "same", "id": 1}, {"text": "same", "id": 2}]
    res = deduplicate_chunks(chunks)
    assert len(res) == 1
    assert res[0]["id"] == 1

def test_detect_sections_caps():
    text = "INTRODUCTION\nThis is intro.\nDATA\nThis is data.\nPDF"
    sections = detect_sections(text)
    assert len(sections) == 2
    assert sections[0][0] == "INTRODUCTION"
    assert sections[1][0] == "DATA"
    assert "PDF" in sections[1][1]

def test_parent_child_chunking_token_limit():
    text = "word " * 1000
    parents, children = create_parent_child_chunks(text, child_max_tokens=20)
    assert len(parents) == 1
    assert len(children) > 1
    for c in children:
        assert len(c["text"].split()) <= 40 
