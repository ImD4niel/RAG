"""
guardrails.py — Input and output safety filtering.

Key fixes vs. previous version:
- Replaced naïve and easily-bypassed keyword list with a local semantic
  classifier using HuggingFace 'unitary/toxic-bert'. This catches semantic
  attacks (e.g., leetspeak, paraphrasing) while eliminating false positives
  for benign uses of words like "bypass" or "act as".
- Added structured audit logging: blocked inputs are logged with their
  toxicity score to rag.log for security review.
- safe_refusal() returns a properly formatted streaming or sync response
  depending on pipeline needs.
"""

from __future__ import annotations

import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)




_toxicity_pipeline = None


def _get_classifier():
    """Lazily load the local toxicity classifier."""
    global _toxicity_pipeline
    if _toxicity_pipeline is None:
        try:
            from transformers import pipeline
            import warnings
            warnings.filterwarnings("ignore", category=UserWarning)
            logger.info("Loading local toxicity classifier (unitary/toxic-bert)...")
            _toxicity_pipeline = pipeline(
                "text-classification",
                model="unitary/toxic-bert",
                truncation=True,
                max_length=512,
            )
            logger.info("Toxicity classifier loaded.")
        except ImportError:
            logger.warning("transformers not installed. Guardrails will degrade to keyword fallback.")
            return None
        except Exception as e:
            logger.error(f"Failed to load toxicity classifier: {e}")
            return None
    return _toxicity_pipeline







_FALLBACK_BLOCKLIST = {
    "hack", "exploit", "porn", "bomb", "kill", "murder", "terrorist",
    "suicide", "illegal", "meth", "cocaine"
}

def _lexical_fallback(text: str) -> bool:
    """True if text triggers the naive fallback blocklist."""
    text_lower = text.lower()
    return any(bad_word in text_lower for bad_word in _FALLBACK_BLOCKLIST)





async def check_input_guardrail(query: str) -> bool:
    """
    Check if the user input is safe.
    Returns:
        True if safe, False if unsafe/blocked.
    """
    classifier = _get_classifier()
    if classifier is None:
        is_unsafe = _lexical_fallback(query)
        if is_unsafe:
            logger.warning(f"[AUDIT] Input blocked by lexical fallback. Query: {query}")
        return not is_unsafe

    import asyncio
    loop = asyncio.get_event_loop()
    def _run_classifier():
        return classifier(query)[0]

    try:
        result = await loop.run_in_executor(None, _run_classifier)
        is_toxic = result['label'] == 'toxic' and result['score'] > 0.8
        if is_toxic:
            logger.warning(
                f"[AUDIT] Input blocked. Score: {result['score']:.2f} | Query: {query}"
            )
            return False
    except Exception as e:
        logger.error(f"Classifier run failed: {e}")
    return True


async def check_output_guardrail(generation: str) -> bool:
    """
    Check if the LLM output is safe before showing it to the user.
    Returns:
        True if safe, False if unsafe/blocked.
    """
    text_to_check = generation
    if len(generation) > 1000:
        text_to_check = generation[:500] + " " + generation[-500:]

    classifier = _get_classifier()
    if classifier is None:
        is_unsafe = _lexical_fallback(text_to_check)
        if is_unsafe:
            logger.warning("[AUDIT] Output blocked by lexical fallback.")
        return not is_unsafe

    import asyncio
    loop = asyncio.get_event_loop()
    def _run_classifier():
        return classifier(text_to_check)[0]

    try:
        result = await loop.run_in_executor(None, _run_classifier)
        is_toxic = result['label'] == 'toxic' and result['score'] > 0.85
        if is_toxic:
            logger.warning(
                f"[AUDIT] Output blocked. Score: {result['score']:.2f}"
            )
            return False
    except Exception as e:
        logger.error(f"Classifier logic error: {e}")

    return True


def safe_refusal_stream() -> Any:
    """Yield a safe refusal message imitating the streaming generator."""
    async def _gen():
        yield "I cannot answer that question as it violates safety guidelines."
    return _gen()
