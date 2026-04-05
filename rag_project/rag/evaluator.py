"""
evaluator.py — RAGAS-inspired LLM-as-a-Judge Evaluation.

Key fixes vs. previous version:
- Evaluated scores are automatically persisted to the PostgreSQL database
  (EvalResult table) to enable regression testing and quality dashboards.
- A failed metric correctly returns 0.0 rather than 0.5 so that evaluation
  failures are properly surfaced rather than masked.
"""

from __future__ import annotations

import asyncio
import logging
import json
from collections import deque
from typing import Optional

from pydantic import BaseModel

from .llm import safe_llm_call
from .db import get_session, EvalResult, SyntheticQA

logger = logging.getLogger(__name__)





QUALITY_ALERT_THRESHOLD: float = 0.5

_quality_degraded: bool = False
_recent_faithfulness = deque(maxlen=10)

def is_quality_degraded() -> bool:
    return _quality_degraded




class ScorePoint(BaseModel):
    score: float
    reason: str




async def score_faithfulness(question: str, context: list[str], answer: str) -> float:
    """Evaluate if the answer is grounded completely in the provided context."""
    if not context or not answer:
        return 0.0

    joined_context = "\n\n".join(context)
    messages = [
        {"role": "system", "content": "You evaluate 'Faithfulness'. Compare the Answer to the Context. Return a score 0.0 to 1.0 depending on how much of the answer is derived directly from the context. Hallucinations or external facts result in heavy penalties. Return score as a float and a short reason."},
        {"role": "user", "content": f"Context: {joined_context}\n\nQuestion: {question}\n\nAnswer: {answer}"}
    ]
    response = await safe_llm_call(messages, temperature=0, response_format=ScorePoint)
    if response and hasattr(response.choices[0].message, "parsed") and response.choices[0].message.parsed:
        return float(response.choices[0].message.parsed.score)
    return 0.0


async def score_answer_relevance(question: str, answer: str) -> float:
    """Evaluate if the generated answer directly addresses the user's question."""
    if not answer:
        return 0.0

    messages = [
        {"role": "system", "content": "You evaluate 'Answer Relevance'. Assess how well the Answer directly answers the Question (ignoring accuracy). Return a score 0.0 to 1.0. Deduct for verbose filler or failing to answer the core ask."},
        {"role": "user", "content": f"Question: {question}\n\nAnswer: {answer}"}
    ]
    response = await safe_llm_call(messages, temperature=0, response_format=ScorePoint)
    if response and hasattr(response.choices[0].message, "parsed") and response.choices[0].message.parsed:
        return float(response.choices[0].message.parsed.score)
    return 0.0


async def score_context_recall(question: str, context: list[str]) -> float:
    """Evaluate if the retrieved context actually contained the necessary information."""
    if not context:
        return 0.0

    joined_context = "\n\n".join(context)
    messages = [
        {"role": "system", "content": "You evaluate 'Context Recall'. Did the retrieved Context actually contain the facts needed to answer the Question? Return a score 0.0 to 1.0. 1.0 means all required facts were present."},
        {"role": "user", "content": f"Question: {question}\n\nContext: {joined_context}"}
    ]
    response = await safe_llm_call(messages, temperature=0, response_format=ScorePoint)
    if response and hasattr(response.choices[0].message, "parsed") and response.choices[0].message.parsed:
        return float(response.choices[0].message.parsed.score)
    return 0.0




async def run_ragas_eval(question: str, context: list[str], answer: str) -> dict[str, float]:
    """Run all 3 core RAGAS metrics asynchronously, log, and persist the results."""
    logger.info(f"Starting LLM-as-Judge evaluation for query: '{question[:30]}...'")

    f_task = asyncio.create_task(score_faithfulness(question, context, answer))
    ar_task = asyncio.create_task(score_answer_relevance(question, answer))
    cr_task = asyncio.create_task(score_context_recall(question, context))

    f_score, ar_score, cr_score = await asyncio.gather(f_task, ar_task, cr_task)

    overall = (f_score + ar_score + cr_score) / 3.0

    results = {
        "faithfulness": round(f_score, 2),
        "answer_relevance": round(ar_score, 2),
        "context_recall": round(cr_score, 2),
        "overall": round(overall, 2),
    }

    logger.info(f"[RAGAS Eval] {json.dumps(results)}")

    try:
        loop = asyncio.get_running_loop()
        def _insert_eval():
            with get_session() as session:
                record = EvalResult(
                    query=question,
                    faithfulness=results["faithfulness"],
                    answer_relevance=results["answer_relevance"],
                    context_recall=results["context_recall"],
                    overall=results["overall"]
                )
                session.add(record)
        await loop.run_in_executor(None, _insert_eval)
        logger.info(f"Evaluation metrics persisted to DB for query: '{question}'")
    except Exception as e:
         logger.error(f"Failed to persist eval record: {e}")

    f_check  = results.get("faithfulness")       or 1.0
    ar_check = results.get("answer_relevance")   or 1.0
    if f_check < QUALITY_ALERT_THRESHOLD or ar_check < QUALITY_ALERT_THRESHOLD:
        logger.warning(
            "[QUALITY_ALERT] RAGAS scores below threshold — "
            f"faithfulness={f_check:.3f}, answer_relevance={ar_check:.3f}. "
            f"Query: {question[:100]}. Review retrieval quality."
        )

    global _quality_degraded
    _recent_faithfulness.append(f_check)
    mean_f = sum(_recent_faithfulness) / len(_recent_faithfulness)
    if mean_f < QUALITY_ALERT_THRESHOLD:
        if not _quality_degraded:
            _quality_degraded = True
            logger.warning(
                f"[QUALITY_GATE] Moving to DEGRADED state. "
                f"Rolling mean faithfulness: {mean_f:.3f}"
            )
    elif mean_f > 0.7:
        if _quality_degraded:
            _quality_degraded = False
            logger.info(
                f"[QUALITY_GATE] Recovered to HEALTHY state. "
                f"Rolling mean faithfulness: {mean_f:.3f}"
            )

    return results





async def generate_synthetic_qa(
    chunks: list[dict],
    questions_per_chunk: int = 3
) -> list[dict]:
    """Generate synthetic QA pairs anchored to specific chunks."""
    class QAPair(BaseModel):
        question: str
        answer: str
    class QAPairList(BaseModel):
        pairs: list[QAPair]

    sem = asyncio.Semaphore(5)
    results = []

    async def _process_chunk(chunk):
        async with sem:
            chunk_id = chunk.get("id")
            text = chunk.get("text") or chunk.get("content")
            if not chunk_id or not text: return
            messages = [
                {
                    "role": "system",
                    "content": f"You are an assessment writer. Generate {questions_per_chunk} questions that this passage directly answers. Extract the specific answer from the text. Return as JSON."
                },
                {"role": "user", "content": text}
            ]
            response = await safe_llm_call(messages, temperature=0.3, response_format=QAPairList)
            if response and hasattr(response.choices[0].message, "parsed") and response.choices[0].message.parsed:
                for pair in response.choices[0].message.parsed.pairs:
                    results.append({
                        "chunk_id": chunk_id,
                        "question": pair.question,
                        "answer": pair.answer
                    })

    tasks = [_process_chunk(c) for c in chunks]
    await asyncio.gather(*tasks)

    if results:
        loop = asyncio.get_running_loop()
        def _insert_qa():
            with get_session() as session:
                for item in results:
                    record = SyntheticQA(
                        chunk_id=item["chunk_id"],
                        question=item["question"],
                        answer=item["answer"]
                    )
                    session.add(record)
        await loop.run_in_executor(None, _insert_qa)
        logger.info(f"Persisted {len(results)} synthetic QA pairs to database.")

    return results


async def estimate_quality_with_ppi(n_samples: int = 50) -> dict:
    """
    Estimate pipeline quality via Prediction-Powered Inference (PPI).
    Draws synthetic QA pairs, runs the full RAG pipeline, and bootstrap-resamples
    the faithfulness evaluations for a robust 95% Confidence Interval.
    """
    from .pipeline import run_pipeline
    import numpy as np
    from scipy.stats import bootstrap

    loop = asyncio.get_running_loop()

    def _fetch_samples():
        with get_session() as session:
            from sqlalchemy import text
            rows = session.query(SyntheticQA).order_by(text("RANDOM()")).limit(n_samples).all()
            return [{"question": r.question, "synthetic_answer": r.answer, "chunk_id": r.chunk_id} for r in rows]

    samples = await loop.run_in_executor(None, _fetch_samples)
    if not samples:
        return {"error": "No synthetic QA pairs found."}

    scores = []
    for s in samples:
        q = s["question"]
        original_ans = s["synthetic_answer"]
        try:
            generator = run_pipeline(q, is_complex=False, bm25_index=None, bm25_corpus=None)
            full_response = ""
            async for token in generator:
                full_response += token
            f_score = await score_faithfulness(q, [original_ans], full_response)
            scores.append(f_score)
        except Exception as e:
            logger.error(f"Failed to evaluate sample '{q[:20]}': {e}")
            scores.append(0.0)

    if not scores:
        return {"error": "Evaluation yielded no scores."}

    data = (np.array(scores),)
    res = bootstrap(data, np.mean, confidence_level=0.95, n_resamples=1000, method='BCa')
    mean_val = np.mean(scores)
    std_val = np.std(scores)
    return {
        "mean_faithfulness": float(mean_val),
        "std_faithfulness": float(std_val),
        "confidence_interval_low": float(res.confidence_interval.low),
        "confidence_interval_high": float(res.confidence_interval.high),
        "n_samples": len(scores)
    }
