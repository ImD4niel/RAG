"""
proposer.py — Proposition-Level Indexing for DRAGON RAG.

Splits paragraph-level chunk texts into atomic factual claims ("propositions").
Embedding propositions rather than paragraphs drastically improves retrieval
precision, as semantic density is maximised and contradictory facts within
a single paragraph are disentangled.
"""

from __future__ import annotations

import asyncio
import logging
import json
from typing import Optional

from .llm import safe_llm_call
from .config import get_settings

logger = logging.getLogger(__name__)

async def extract_propositions(chunk: dict) -> list[str]:
    """
    Given a paragraph or text chunk dictionary, extract a list of self-contained,
    atomic factual propositions.

    Example:
        Input: {"text": "Google was founded in 1998 by Larry Page and Sergey Brin..."}
        Output: [
            "Google was founded in 1998.",
            "Google was founded by Larry Page and Sergey Brin.",
            "Larry Page and Sergey Brin were Ph.D. students at Stanford University."
        ]

    Args:
        chunk: Chunk dictionary containing 'text' and other metadata.

    Returns:
        List of proposition string sentences. Returns [chunk["text"]] upon failure.
    """
    chunk_text = chunk.get("text", "")
    if not chunk_text or len(chunk_text.strip()) < 20:
        return [chunk_text] if chunk_text else []

    system_prompt = (
        "You are a factual proposition extractor. "
        "Decompose the given text into self-contained, atomic factual propositions. "
        "Each proposition must make sense on its own (resolve pronouns, KEEP specific entity names). "
        "Output ONLY a JSON array of strings. No conversational text."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Text to decompose:\n{chunk_text}"}
    ]

    try:
        response = await safe_llm_call(messages, temperature=0.0)
        if response is None:
            return [chunk_text]

        content = response.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        propositions = json.loads(content.strip())
        if isinstance(propositions, list) and all(isinstance(p, str) for p in propositions):
            return [p.strip() for p in propositions if p.strip()]

    except Exception as e:
        logger.warning(f"Proposition extraction failed (falling back to original chunk): {e}")

    return [chunk_text]
