"""
self_rag.py — Self-RAG post-generation faithfulness verification.

Implements the "Generate → Critique → Repair" loop described in:
  Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection
  (Asai et al., 2023).

Key design decisions:
- Verification runs AFTER the full answer is buffered (not during streaming).
- Maximum 2 repair attempts to bound latency and cost.
- Violated sentences are explicitly surfaced to the repair prompt so the LLM
  knows exactly what to remove or soften.
- Falls back to the original answer + caveat on max-retries rather than
  blocking indefinitely or returning empty content.
"""

from __future__ import annotations

import logging
from typing import Optional

from .llm import safe_llm_call

logger = logging.getLogger(__name__)


MAX_SELF_RAG_RETRIES: int = 2




def _get_context_char_limit() -> int:
    try:
        from .config import get_settings
        return min(int(get_settings().max_context_tokens * 0.4 * 4), 20_000)
    except Exception:
        return 8_000  


_CONTEXT_CHAR_LIMIT: int = _get_context_char_limit()


async def verify_and_repair(
    answer: str,
    question: str,
    context_chunks: list[dict],
    max_retries: int = MAX_SELF_RAG_RETRIES,
) -> tuple[str, bool]:
    """
    Verify that an LLM-generated answer is grounded in the retrieved context.

    If violations are found, the answer is regenerated with an explicit constraint
    to remove or soften unsupported claims. Runs at most ``max_retries`` repair
    cycles before returning the best available answer.

    Args:
        answer:         The full generated answer string to verify.
        question:       The original user question (used for context in repair).
        context_chunks: The retrieved source chunks used to generate the answer.
        max_retries:    Maximum repair iterations before giving up.

    Returns:
        Tuple of (final_answer: str, was_faithful: bool).
        ``was_faithful`` is True if the answer passed without any violation found.
    """
    if not answer.strip() or not context_chunks:
        return answer, True  

    context_text = "\n\n---\n\n".join(c.get("text", "") for c in context_chunks)

    for attempt in range(max_retries + 1):
        verdict = await _critique(answer, question, context_text)
        if verdict is None:
            logger.warning(
                "[SAFETY_SKIP] Self-RAG critique call failed — "
                "returning answer with unverified caveat."
            )
            caveat = (
                "\n\n---\n*(Note: Faithfulness verification could not be completed "
                "for this response. Please cross-check critical details against source "
                "documents.)*"
            )
            return answer + caveat, False

        is_faithful: bool = verdict.get("is_faithful", True)
        violated: list[str] = verdict.get("violated_claims", [])

        if is_faithful or not violated:
            return answer, True  

        logger.warning(
            f"Self-RAG [attempt {attempt + 1}/{max_retries + 1}]: "
            f"{len(violated)} unsupported claim(s) detected."
        )

        if attempt < max_retries:
            answer = await _repair(answer, question, context_text, violated)
            if not answer:
                break

    caveat = (
        "\n\n---\n*(Note: Parts of this response could not be fully verified "
        "against the source documents. Please cross-check critical details.)*"
    )
    return answer + caveat, False


async def _critique(answer: str, question: str, context_text: str) -> Optional[dict]:
    """
    Ask the LLM to identify any claims in ``answer`` not supported by ``context_text``.

    Returns:
        Dict with keys "is_faithful" (bool) and "violated_claims" (list[str]),
        or None if the LLM call fails.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict faithfulness critic. Your ONLY job is to identify claims "
                "in an AI-generated answer that are NOT explicitly supported by the provided "
                "source context.\n\n"
                "A claim is UNSUPPORTED if it:\n"
                "1. Contains specific facts (numbers, names, dates, percentages) not in context.\n"
                "2. Uses qualifier language ('typically', 'usually', 'generally') implying "
                "broader knowledge beyond the context.\n"
                "3. Draws conclusions or implications ('therefore', 'this means') not stated "
                "in the context.\n\n"
                "Return ONLY valid JSON — no other text:\n"
                '{"is_faithful": true/false, '
                '"violated_claims": ["exact sentence from answer that is unsupported"]}'
            ),
        },
        {
            "role": "user",
            "content": (
                f"Source Context:\n{context_text[:_CONTEXT_CHAR_LIMIT]}\n\n"
                f"Question: {question}\n\n"
                f"Answer to verify:\n{answer}"
            ),
        },
    ]

    response = await safe_llm_call(messages, temperature=0)
    if response is None:
        return None

    try:
        import json as _json
        content = response.choices[0].message.content.strip()
        content = content.replace("```json", "").replace("```", "").strip()
        result = _json.loads(content)
        return {
            "is_faithful": bool(result.get("is_faithful", True)),
            "violated_claims": list(result.get("violated_claims", [])),
        }
    except Exception as e:
        logger.warning(f"Self-RAG _critique: failed to parse JSON response: {e}")
        return None


async def _repair(
    answer: str,
    question: str,
    context_text: str,
    violated_claims: list[str],
) -> str:
    """
    Regenerate an answer with explicit instruction to remove violated claims.

    Returns the repaired answer, or the original on failure.
    """
    violations_str = "\n".join(f"- {v}" for v in violated_claims)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise editor. Rewrite the provided answer to strictly remove "
                "or soften any claims that are not supported by the source context. "
                "Do NOT add new information. Preserve all supported facts exactly. "
                "If a claim cannot be verified from the context, replace it with: "
                "'[This detail is not confirmed in the source documents.]'"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Source Context:\n{context_text[:_CONTEXT_CHAR_LIMIT]}\n\n"
                f"Question: {question}\n\n"
                f"Original Answer:\n{answer}\n\n"
                f"Unsupported claims to remove or soften:\n{violations_str}\n\n"
                "Rewrite the answer now, removing only the unsupported claims:"
            ),
        },
    ]

    response = await safe_llm_call(messages, temperature=0)
    if response is None:
        logger.warning("Self-RAG _repair: LLM call failed — returning original answer.")
        return answer

    return response.choices[0].message.content.strip()


async def enforce_citations(
    answer: str,
    chunks: list[dict],
    similarity_threshold: float = 0.75
) -> tuple[str, list[str]]:
    """
    Verify factual claims sentence-by-sentence via cosine similarity over embeddings.
    Embeds each sentence of the generated answer and checks if it closely matches
    any chunk's embedding. If not, meaning it's unsupported or fabricated,
    prepends [Unverified] to the sentence.
    """
    if not answer.strip() or not chunks:
        return answer, []

    import re
    import numpy as np
    from .embeddings import get_embedding

    sentences = re.split(r'(?<=[.!?])\s+', answer)
    unverified_claims = []
    chunk_embs = []
    for c in chunks:
        emb = c.get("embedding")
        if emb is not None:
            chunk_embs.append(np.array(emb, dtype=np.float32))
    if not chunk_embs:
        return answer, []

    modified_sentences = []
    ignore_lower = {
        "additionally,", "in summary", "in summary,", "moreover,", 
        "furthermore,", "in conclusion", "in conclusion,", "to summarize",
        "to summarize,", "therefore,"
    }
    for sent in sentences:
        s = sent.strip()
        if not s:
            modified_sentences.append(sent)
            continue
        if len(s) < 15 or s.lower() in ignore_lower:
            modified_sentences.append(sent)
            continue
        try:
            sent_emb_tuple = await get_embedding(s)
            sent_emb = np.array(sent_emb_tuple[0], dtype=np.float32)
            sent_norm = sent_emb / (np.linalg.norm(sent_emb) + 1e-10)
            max_sim = -1.0
            for c_emb in chunk_embs:
                c_norm = c_emb / (np.linalg.norm(c_emb) + 1e-10)
                sim = np.dot(sent_norm, c_norm)
                if sim > max_sim:
                    max_sim = sim
            if max_sim < similarity_threshold:
                unverified_claims.append(s)
                if sent.endswith(" "):
                    modified_sentences.append(f"[Unverified] {s} ")
                else:
                    modified_sentences.append(f"[Unverified] {s}")
            else:
                modified_sentences.append(sent)
        except Exception as e:
            logger.warning(f"Error enforcing citation for sentence: {e}")
            modified_sentences.append(sent)
    modified_answer = " ".join(modified_sentences) if len(modified_sentences) > 1 else modified_sentences[0]
    return modified_answer, unverified_claims
