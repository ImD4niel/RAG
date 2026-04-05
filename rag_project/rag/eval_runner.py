"""
eval_runner.py — CLI Runner for ARES-style synthetic evaluation generation and scaling.

Usage:
  python -m rag.eval_runner
"""

import sys
import asyncio
import logging

from .db import get_session, Document
from .evaluator import generate_synthetic_qa, estimate_quality_with_ppi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    print("="*60)
    print(" DRAGON RAG — Synthetic QA Generation & Benchmarking")
    print("="*60)
    def _fetch_all_chunks():
        with get_session() as session:
            docs = session.query(Document).filter(Document.level == 1, Document.media_type != "image").limit(10).all()
            return [{"id": d.id, "text": d.content} for d in docs]
    loop = asyncio.get_running_loop()
    all_chunks = await loop.run_in_executor(None, _fetch_all_chunks)
    if not all_chunks:
        print("No chunks found in database. Run ingestion first.")
        return
    print(f"Generating synthetic QA pairs for {len(all_chunks)} chunks...")
    results = await generate_synthetic_qa(all_chunks, questions_per_chunk=3)
    print(f"Generated and pushed {len(results)} distinct QA pairs to DB.")
    print("Estimating generative quality using PPI (Prediction-Powered Inference)...")
    quality = await estimate_quality_with_ppi(n_samples=5)
    if "error" in quality:
        print(f"Error during estimation: {quality['error']}")
        return
    print("\n--- ARES METRICS ---\n")
    print(f"Estimated faithfulness:  {quality['mean_faithfulness']:.3f} (± {quality.get('std_faithfulness', 0):.3f} std deviation)")
    print(f"95% Confidence Interval: ({quality['confidence_interval_low']:.3f} – {quality['confidence_interval_high']:.3f})")
    print(f"Samples Tested:          {quality['n_samples']}")
    print("\n" + "="*60)

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Interrupted by user.")
