import sys
import os
import json
import time
import asyncio
import logging

# Make sure we can import from `rag`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.pipeline import run_pipeline_stream, lazy_load_index
from rag.db import get_session, EvalResult, Document
from rag.config import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("benchmark")

def count_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text, disallowed_special=()))
    except ImportError:
        return len(text.split())

async def run_query(query: str, ground_truth: str):
    logger.info(f"--- Running Query: {query} ---")
    start_time = time.perf_counter()
    
    # Run the pipeline
    final_answer = ""
    async for token in run_pipeline_stream(question=query):
        final_answer += token
        
    latency = time.perf_counter() - start_time
    
    # Let background tasks (like RAGAS eval) finish
    await asyncio.sleep(8) 
    
    # Retrieve RAGAS evaluation from DB
    ragas_scores = {"faithfulness": 0.0, "answer_relevance": 0.0, "context_recall": 0.0, "overall": 0.0}
    try:
        with get_session() as session:
            record = session.query(EvalResult).filter(EvalResult.query == query).order_by(EvalResult.id.desc()).first()
            if record:
                ragas_scores = {
                    "faithfulness": record.faithfulness,
                    "answer_relevance": record.answer_relevance,
                    "context_recall": record.context_recall,
                    "overall": record.overall
                }
    except Exception as e:
        logger.warning(f"Could not fetch eval results from DB: {e}")

    # Estimate Costs
    # $0.150 / 1M Input Tokens, $0.600 / 1M Output Tokens (Approximate for GPT-4o-mini)
    out_tokens = count_tokens(final_answer)
    estimated_cost = (out_tokens / 1_000_000) * 0.60
    
    return {
        "query": query,
        "latency_sec": round(latency, 2),
        "out_tokens": out_tokens,
        "estimated_cost_usd": round(estimated_cost, 6),
        "scores": ragas_scores,
        "generated_answer": final_answer
    }

async def main():
    print("="*60)
    print(" DRAGON RAG — Comprehensive Benchmarking Suite")
    print("="*60)

    # 1. Check Data Volume
    try:
        with get_session() as session:
            docs_count = session.query(Document).count()
            print(f"Data Volume: {docs_count} total chunks in DB.")
    except Exception:
        pass

    # 2. Warm up DB/Index
    await lazy_load_index()

    # 3. Load Gold Set
    gold_set_path = os.path.join(os.path.dirname(__file__), "gold_set.json")
    with open(gold_set_path, "r", encoding="utf-8") as f:
        gold_queries = json.load(f)

    print(f"Loaded {len(gold_queries)} queries from Gold Set. Beginning evaluation...\n")
    
    results = []
    
    for item in gold_queries:
        res = await run_query(item["query"], item["ground_truth"])
        results.append(res)
    
    # Aggregate Metrics
    avg_latency = sum(r["latency_sec"] for r in results) / len(results)
    avg_faithfulness = sum(r["scores"]["faithfulness"] for r in results) / len(results)
    avg_relevance = sum(r["scores"]["answer_relevance"] for r in results) / len(results)
    avg_recall = sum(r["scores"]["context_recall"] for r in results) / len(results)
    total_cost = sum(r["estimated_cost_usd"] for r in results)
    
    report = [
        "# Benchmark Execution Report",
        "",
        f"**Total Queries Tested:** {len(results)}",
        f"**Total Document Volume:** {docs_count if 'docs_count' in locals() else 'Unknown'} chunks",
        f"**Average Latency:** {avg_latency:.2f} seconds",
        f"**Total Generation Cost (Approximate):** ${total_cost:.6f}",
        "",
        "## RAGAS Quality Metrics (Averages)",
        f"- **Faithfulness (Safety):** {avg_faithfulness:.2f}/1.0",
        f"- **Answer Relevance:** {avg_relevance:.2f}/1.0",
        f"- **Context Recall (Hit Rate):** {avg_recall:.2f}/1.0",
        "",
        "## Query Details"
    ]
    
    for r in results:
        report.append(f"### Q: {r['query']}")
        report.append(f"- Latency: {r['latency_sec']}s")
        report.append(f"- Cost: ${r['estimated_cost_usd']}")
        report.append(f"- Scores: Faithfulness={r['scores']['faithfulness']}, Relevance={r['scores']['answer_relevance']}, Context Recall={r['scores']['context_recall']}")
        report.append(f"- Answer preview: {r['generated_answer'][-200:]}...")
        report.append("")

    report_path = os.path.join(os.path.dirname(__file__), "benchmark_results.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
        
    print(f"\nBenchmark complete. Report written to {report_path}")

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
