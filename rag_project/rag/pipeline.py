
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from contextlib import contextmanager
from typing import AsyncGenerator

import numpy as np
from rank_bm25 import BM25Okapi

from .agent import run_agent, is_complex_query
from .config import get_settings, SYSTEM_PROMPT
from .embeddings import get_embedding
from .evaluator import run_ragas_eval
from .llm import (
    compress_context,
    detect_contradictions,
    generate_multi_perspective_hyde,
    grade_retrieval_tristate,
    plan_queries,
    route_query,
    stream_llm_call,
    safe_llm_call,
)
from .retrieval import colbert_rerank, cross_encoder_rerank, hybrid_search, mmr_select
from .self_rag import verify_and_repair, enforce_citations
from .vector_store import build_or_load_index, search_postgres

logger = logging.getLogger(__name__)



bm25_index: BM25Okapi | None = None
bm25_corpus: list[dict] = []
conversation_history: list[dict] = []
memory_summary: str = ""
answer_cache: dict[str, str] = {}
CORPUS_NOUNS: set[str] | None = None

def _get_corpus_nouns() -> set[str]:
    global CORPUS_NOUNS
    if CORPUS_NOUNS is not None:
        return CORPUS_NOUNS
    try:
        from .db import get_session, Document
        import re
        CORPUS_NOUNS = set()
        with get_session() as session:
            docs = session.query(Document.source).distinct().all()
            for (source,) in docs:
                if not source:
                    continue
                basename = source.split("/")[-1].split("\\")[-1]
                words = re.split(r'[^A-Za-z]+', basename.lower())
                for w in words:
                    if len(w) > 4:
                        CORPUS_NOUNS.add(w)
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Failed to load CORPUS_NOUNS: {e}")
        CORPUS_NOUNS = set()
    return CORPUS_NOUNS

def _is_compound_query(query: str) -> bool:
    words = query.lower().split()
    if len(words) <= 8:
        return False
    corpus_nouns = _get_corpus_nouns()
    if not corpus_nouns:
        return True
    query_words = {
        w.strip(".,?!") for w in words if len(w) > 4
    }
    return bool(query_words & corpus_nouns)

def _select_alpha(query: str, is_complex: bool) -> float:
    q = query.lower()
    has_specific_ref = bool(re.search(
        r'\b\d+\.\d+|\bcl\w+\s+\d+|v\d+\.\d+|clause\s+\d+|section\s+\d+', q
    ))
    has_code_or_error = bool(re.search(r'[A-Z]{2,}[-_]\d+|error\s+\w+|error\s+code', query))
    has_exact_phrase  = bool(re.search(r'"[^"]+"', query))  
    is_short_specific = len(set(q.split())) < 5 and len(q.split()) >= 2

    if has_specific_ref or has_code_or_error or has_exact_phrase:
        return 0.2
    elif is_complex and not is_short_specific:
        return 0.75
    else:
        return 0.55


def _should_use_hyde(query: str) -> bool:
    specific_patterns = [
        r'\b\d+\.\d+',           
        r'\b[A-Z]+-\d+',         
        r'clause\s+\d+',          
        r'section\s+\d+',
        r'article\s+\d+',
        r'error\s+\w+',           
        r'error\s+code',
        r'\'[A-Z_]{3,}\'',        
        r'"[^"]+"',               
    ]
    return not any(re.search(p, query, re.IGNORECASE) for p in specific_patterns)


def _sanitise_query(question: str) -> str:
    _ROLE_PREFIX = re.compile(
        r"(?i)(system\s*:|assistant\s*:|<\|im_start\||<\|system\||ignore (all |previous |above )?(instructions?|rules?|system prompt))",
        re.MULTILINE,
    )
    sanitised = _ROLE_PREFIX.sub("[FILTERED]", question).strip()
    if sanitised != question:
        logger.warning(
            f"Role-injection pattern removed from query. "
            f"Original length: {len(question)}, Sanitised length: {len(sanitised)}"
        )
    return sanitised


def _get_cache_key(question: str) -> str:
    history_ctx = json.dumps([h.get("content") for h in conversation_history[-2:]])
    return hashlib.sha256(f"{question}::{history_ctx}".encode()).hexdigest()


def reorder_for_attention(chunks: list) -> list:
    if len(chunks) <= 2:
        return chunks
    sorted_chunks = sorted(
        chunks, key=lambda c: c.get("cross_encoder_score", 0.0), reverse=True
    )
    reordered = [sorted_chunks[0]]
    reordered.extend(sorted_chunks[2:])
    reordered.append(sorted_chunks[1])
    return reordered


@contextmanager
def _timer(stage: str):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    logger.info(f"[TIMING] {stage}: {elapsed:.3f}s")


def _build_source_citation(chunks: list[dict]) -> str:
    seen = set()
    lines = []
    for c in chunks:
        src = c.get("source", "Unknown")
        src_name = src.replace("\\", "/").split("/")[-1] if src else "Unknown"
        page = c.get("page")
        entry = f"• {src_name}" + (f", p.{page}" if page else "")
        if entry not in seen:
            seen.add(entry)
            lines.append(entry)
    return "\n".join(lines)


async def _summarise_memory(history_to_summarise: list[dict], existing_summary: str) -> str:
    if not history_to_summarise:
        return existing_summary

    turns_text = "\n".join(
        f"{m['role'].capitalize()}: {m.get('content', '')[:300]}"
        for m in history_to_summarise
    )
    prior = f"Previous summary:\n{existing_summary}\n\n" if existing_summary else ""

    messages = [
        {
            "role": "system",
            "content": (
                "You are a conversation historian. Produce a single dense paragraph "
                "summarising the key topics, facts, and decisions from the conversation "
                "below. Preserve all specific figures, names, and technical terms. "
                "This summary will be used to give a research assistant context for a "
                "continuing conversation."
            ),
        },
        {
            "role": "user",
            "content": f"{prior}New conversation turns to incorporate:\n{turns_text}",
        },
    ]
    resp = await safe_llm_call(messages, temperature=0)
    if resp is None:
        logger.warning("[Memory] Summarisation LLM call failed — retaining previous summary.")
        return existing_summary
    new_summary = resp.choices[0].message.content.strip()
    logger.info(f"[Memory] Updated memory_summary ({len(new_summary)} chars).")
    return new_summary


def _build_gen_prompt(
    question: str,
    compressed_context: str,
    sources_text: str,
    crag_grade: str,
    conflict_summary: str = "",
) -> str:
    if conflict_summary:
        conflict_preamble = (
            "CRITICAL — CONFLICTING INFORMATION DETECTED:\n"
            f"{conflict_summary}\n"
            "You MUST explicitly acknowledge this conflict in your response. "
            "Cite both sources by name. State which is more authoritative (based on date "
            "or document type). Do NOT silently pick one side.\n\n"
        )
    else:
        conflict_preamble = ""

    if crag_grade == "AMBIGUOUS":
        faithfulness_rule = (
            "The retrieved context is related to this question but may not contain the "
            "complete or exact answer. ONLY state facts that are explicitly and literally "
            "present in the Context. If the specific answer cannot be confirmed from the "
            "Context, say clearly: 'This specific information is not available in the "
            "provided documents.' Do NOT approximate, infer, or fill gaps. "
            "Do NOT use your training knowledge."
        )
    else:
        faithfulness_rule = (
            "Answer using ONLY the facts explicitly stated in the Context below. "
            "Do NOT use your training knowledge, parametric memory, or general knowledge "
            "to supplement, extend, or fill any gaps — even if you are highly confident "
            "you know the answer. If the specific answer is not explicitly present in "
            "the Context, say: 'This information is not available in the provided "
            "documents.' Do not infer or assume."
        )

    return (
        f"{conflict_preamble}"
        f"Context:\n{compressed_context}\n\n"
        f"Sources referenced:\n{sources_text}\n\n"
        f"Question: {question}\n\n"
        f"Instructions:\n"
        f"1. Answer based ONLY on the provided Context.\n"
        f"2. {faithfulness_rule}\n"
        f"3. EVERY factual claim must reference the source document name.\n"
    )




async def lazy_load_index() -> None:
    global bm25_index, bm25_corpus
    if bm25_index is None:
        logger.info("Lazy loading index...")
        bm25_index, bm25_corpus = await build_or_load_index()


def drop_index() -> None:
    global bm25_index, bm25_corpus, conversation_history, memory_summary
    bm25_index = None
    bm25_corpus = []
    conversation_history.clear()
    memory_summary = ""
    answer_cache.clear()




async def run_pipeline_stream(
    question: str,
    session_id: str | None = None,
) -> AsyncGenerator[str, None]:
    global conversation_history, memory_summary

    yield "▤ Initialising...\n"
    await lazy_load_index()

    question = _sanitise_query(question)

    cache_key = _get_cache_key(question)
    if cache_key in answer_cache:
        yield "⚡ (From cache)\n\n"
        yield answer_cache[cache_key]
        return

    yield "▤ Routing...\n"
    intent = await route_query(question)
    if intent == "GREETING" and not _is_compound_query(question):
        greeting_messages = (
            [{"role": "system", "content": SYSTEM_PROMPT}]
            + conversation_history[-4:]
            + [{"role": "user", "content": question}]
        )
        response_text = ""
        async for chunk in stream_llm_call(greeting_messages, temperature=0.7):
            response_text += chunk
        if not response_text.strip():
            response_text = "Hello! How can I help you today?"
        answer_cache[cache_key] = response_text
        conversation_history.append({"role": "user", "content": question})
        conversation_history.append({"role": "assistant", "content": response_text})
        yield response_text
        return

    yield "▤ Analysing query...\n"
    complex_query = await is_complex_query(question)

    rewritten_query = question
    if conversation_history:
        recent = [
            {"role": h["role"], "content": h.get("content", "")[:200]}
            for h in conversation_history[-4:]
        ]
        rewrite_prompt = (
            f"Conversation history:\n{json.dumps(recent)}\n\n"
            f"Current question: {question}\n\n"
            "If the question contains pronouns or references (e.g. 'that', 'it', 'they', "
            "'how does that compare') that refer to earlier conversation context, rewrite "
            "it to be fully self-contained so a search engine can understand it without "
            "context. If it is already self-contained, return it unchanged."
        )
        resp = await safe_llm_call(
            [{"role": "user", "content": rewrite_prompt}], temperature=0
        )
        if resp and resp.choices:
            rewritten_query = resp.choices[0].message.content.strip()

    settings = get_settings()
    _max_ctx = settings.max_context_tokens
    CAG_LIMIT = int(_max_ctx * 0.75)

    total_words = sum(len(c.get("text", "").split()) for c in bm25_corpus)
    estimated_tokens = int(total_words * 1.3)

    if bm25_corpus and estimated_tokens < CAG_LIMIT:
        yield f"⚡ CAG Mode: Corpus is {estimated_tokens} tokens. Bypassing retrieval.\n"
        final_chunks = list(bm25_corpus)
        grade = "EXACT"
        use_agent = False
    else:
        search_queries = [rewritten_query]
        if complex_query:
            yield "▤ Planning sub-queries...\n"
            sub_queries = await plan_queries(rewritten_query)

            if _should_use_hyde(rewritten_query):
                hyde_queries = await generate_multi_perspective_hyde(rewritten_query)
            else:
                hyde_queries = []
                logger.info("HyDE disabled: specific reference detected in query")

            search_queries = list(set(sub_queries + hyde_queries + [rewritten_query]))
            yield f"▤ Using {len(search_queries)} search angles...\n"

        with _timer("Hybrid Retrieval"):
            from .evaluator import is_quality_degraded
            if is_quality_degraded():
                retrieval_top_k = min(settings.retrieval_top_k * 2, 25)
                force_complex = True
                logger.info(
                    "[QUALITY_GATE] Quality degraded — expanding "
                    "retrieval top_k to %d", retrieval_top_k
                )
            else:
                retrieval_top_k = settings.retrieval_top_k
                force_complex = False

            all_chunks: list[dict] = []
            sem = asyncio.Semaphore(5)

            async def fetch_for_q(q: str) -> list[dict]:
                async with sem:
                    q_emb, _, q_sparse = await get_embedding(q)
                    sem_res = search_postgres(np.array(q_emb, dtype=np.float32), top_k=retrieval_top_k)
                    scores = [c.get("distance", 1.0) for c in sem_res]
                    alpha = _select_alpha(q, complex_query or force_complex)
                    hyb, _ = hybrid_search(
                        sem_res, scores, bm25_index, bm25_corpus, q,
                        query_sparse_vec=q_sparse,
                        alpha=alpha, top_k=retrieval_top_k
                    )
                    return hyb

            results = await asyncio.gather(*(fetch_for_q(q) for q in search_queries))
            for res in results:
                all_chunks.extend(res)

            unique_chunks = {c.get("id"): c for c in all_chunks if c.get("id")}
            candidate_chunks = list(unique_chunks.values())

        if not candidate_chunks:
            yield "No relevant information found in the documents.\n"
            return

        with _timer("Reranking"):
            yield "▤ Reranking (ColBERT)...\n"
            colbert_top = await colbert_rerank(rewritten_query, candidate_chunks, top_k=10)
            yield "▤ Reranking (CrossEncoder)...\n"
            cross_top = await cross_encoder_rerank(rewritten_query, colbert_top, top_k=7)

            chunk_embeddings = [
                c.get("embedding") for c in cross_top if c.get("embedding") is not None
            ]
            chunks_with_emb = [c for c in cross_top if c.get("embedding") is not None]
            if chunk_embeddings and len(chunks_with_emb) > 2:
                q_emb_mmr, _, _ = await get_embedding(rewritten_query)
                final_chunks = mmr_select(
                    np.array(q_emb_mmr, dtype=np.float32),
                    chunks_with_emb,
                    [np.array(e, dtype=np.float32) if not hasattr(e, 'shape') else e
                     for e in chunk_embeddings],
                    top_k=5,
                    lambda_param=0.7,
                )
            else:
                final_chunks = cross_top[:5]
            final_chunks = reorder_for_attention(final_chunks)

        yield "▤ Grading context relevance...\n"
        grade = await grade_retrieval_tristate(rewritten_query, final_chunks)
        logger.info(f"[CRAG] Grade: {grade} for query: '{rewritten_query[:60]}'")

        if grade == "AMBIGUOUS":
            yield "▤ Context ambiguous — reformulating query and retrying retrieval...\n"
            reformulate_prompt = (
                f"The query '{rewritten_query}' matched related but incomplete documents. "
                f"Rephrase it as a more specific, targeted search query that would find the "
                f"exact answer. Return ONLY the reformulated query, nothing else."
            )
            reform_resp = await safe_llm_call(
                [{"role": "user", "content": reformulate_prompt}], temperature=0
            )
            if reform_resp and reform_resp.choices:
                reformulated = reform_resp.choices[0].message.content.strip()
                q_emb2, _, q_sparse2 = await get_embedding(reformulated)
                retry_sem = search_postgres(np.array(q_emb2, dtype=np.float32), top_k=10)
                retry_alpha = _select_alpha(reformulated, complex_query)
                retry_hyb, _ = hybrid_search(
                    retry_sem, [c.get("distance", 1.0) for c in retry_sem],
                    bm25_index, bm25_corpus, reformulated, query_sparse_vec=q_sparse2, alpha=retry_alpha, top_k=8
                )
                retry_colbert = await colbert_rerank(reformulated, retry_hyb, top_k=5)
                retry_grade = await grade_retrieval_tristate(reformulated, retry_colbert)
                if retry_grade in ("EXACT", "AMBIGUOUS"):
                    final_chunks = await cross_encoder_rerank(reformulated, retry_colbert, top_k=5)
                    grade = retry_grade
                    logger.info(f"[CRAG] Retry succeeded. New grade: {grade}")

        if grade == "IRRELEVANT":
            yield "⚠️ Documents irrelevant — escalating to agent search...\n"
            use_agent = True
        else:
            use_agent = False

    final_answer: str = ""  

    if "grade" in dir() and grade == "IRRELEVANT" and not use_agent:
        logger.warning(
            "Post-retry grade is IRRELEVANT — escalating to agent search."
        )
        use_agent = True

    if use_agent:
        yield "▤ Agent searching (ReAct / Plan-Execute)...\n\n"
        final_answer = await run_agent(
            question,
            conversation_history=conversation_history[-4:],
            memory_summary=memory_summary,
            is_complex=complex_query,
            bm25_index=bm25_index,
            bm25_corpus=bm25_corpus,
        )

        agent_context_chunks = []
        if "final_chunks" in dir() and final_chunks:
            agent_context_chunks = final_chunks
        if agent_context_chunks:
            with _timer("Agent Self-RAG Verification"):
                final_answer, was_faithful = await verify_and_repair(
                    final_answer, question, agent_context_chunks
                )
                if not was_faithful:
                    logger.warning(
                        "Agent answer repaired by Self-RAG — unfaithful claims removed."
                    )

        yield final_answer

    else:
        has_conflict = False
        conflict_summary = ""
        distinct_sources = {
            c.get("source", "").replace("\\", "/").split("/")[-1]
            for c in final_chunks
        }
        if complex_query and len(final_chunks) >= 2 and len(distinct_sources) >= 2:
            conflict_check = await detect_contradictions(rewritten_query, final_chunks)
            has_conflict = conflict_check.get("has_contradiction", False)
            conflict_summary = conflict_check.get("summary", "")
            if has_conflict:
                logger.info(f"[CONTRADICTION] Detected: {conflict_summary}")

        raw_texts = [c.get("text", "") for c in final_chunks]
        has_table_chunks = any(c.get("doc_type") == "table" for c in final_chunks)
        total_chars = sum(len(t) for t in raw_texts)

        with _timer("Context Compression"):
            if not has_table_chunks and total_chars > 6000:
                compressed = await compress_context(rewritten_query, raw_texts)
            else:
                compressed = "\n\n---\n\n".join(raw_texts)  

        sources_text = _build_source_citation(final_chunks)

        gen_prompt = _build_gen_prompt(question, compressed, sources_text, grade, conflict_summary)
        messages = (
            [{"role": "system", "content": SYSTEM_PROMPT}]
            + [{"role": m["role"], "content": m["content"]} for m in conversation_history[-4:]]
            + [{"role": "user", "content": gen_prompt}]
        )

        yield "▤ Generating answer...\n\n"
        final_answer = ""
        _max_ctx = get_settings().max_context_tokens
        with _timer("Generation"):
            async for chunk in stream_llm_call(messages, temperature=0.3, max_context_tokens=_max_ctx):
                final_answer += chunk

        with _timer("Self-RAG Verification"):
            verified_answer, was_faithful = await verify_and_repair(
                final_answer, rewritten_query, final_chunks
            )
            if not was_faithful:
                logger.warning("[Self-RAG] Answer repaired — unfaithful claims removed.")
            final_answer = verified_answer
        if hasattr(settings, "enforce_citations") and settings.enforce_citations:
            with _timer("Citation Enforcement"):
                final_answer, unverified = await enforce_citations(
                    final_answer, final_chunks
                )
                if unverified:
                    logger.warning(
                        "[CITATION_ALERT] %d unverified claims: %s",
                        len(unverified), unverified
                    )

        if sources_text and sources_text.strip("\u2022 \n"):
            cited_sources = [
                line.strip("\u2022 ").split(",")[0].strip()
                for line in sources_text.splitlines()
                if line.strip()
            ]
            answer_lower = final_answer.lower()
            any_cited = any(s.lower() in answer_lower for s in cited_sources if s)
            if not any_cited:
                logger.warning(
                    "No source document name detected in generated answer. "
                    f"Expected at least one of: {cited_sources[:3]}"
                )

        yield final_answer

    conversation_history.append({"role": "user", "content": question})
    conversation_history.append({"role": "assistant", "content": final_answer})
    answer_cache[cache_key] = final_answer

    MAX_HISTORY = 8
    if len(conversation_history) > MAX_HISTORY:
        evicted = conversation_history[:-MAX_HISTORY]
        conversation_history = conversation_history[-MAX_HISTORY:]
        async def _update_memory():
            global memory_summary
            memory_summary = await _summarise_memory(evicted, memory_summary)
        asyncio.create_task(_update_memory())

    eval_chunks = []
    if "final_chunks" in dir() and final_chunks:
        eval_chunks = final_chunks

    if final_answer and eval_chunks:
        async def do_eval():
            try:
                await run_ragas_eval(
                    question,
                    [c.get("text", "") for c in eval_chunks],
                    final_answer,
                )
            except Exception as e:
                logger.error(f"RAGAS eval crashed: {e}")

        asyncio.create_task(do_eval())
    elif final_answer and not eval_chunks:
        logger.info("RAGAS skipped — no retrieved chunks available for agent path.")

