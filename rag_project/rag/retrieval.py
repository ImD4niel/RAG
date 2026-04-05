"""
retrieval.py — Hybrid search, MMR, and multi-stage reranking.

Key fixes vs. previous version:
- colbert_rerank() model initialisation is now dispatched via
  asyncio.run_in_executor to prevent event loop blocking.
- cross_encoder_rerank() added as a second-stage, high-precision reranker
  after the broad ColBERT pass, utilising the previously dead model in config.py.
- mmr_select() vectors are explicitly L2-normalised to ensure dot product
  correctly equates to cosine similarity.
- Full type hints and docstrings.
"""

from __future__ import annotations

import asyncio
import logging
import math
from typing import Any, Optional

import numpy as np
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)




_colbert_tokenizer = None
_colbert_model = None


def _load_colbert():
    """Lazily load the Jina ColBERT model (synchronous block)."""
    global _colbert_tokenizer, _colbert_model
    if _colbert_model is None:
        from transformers import AutoModel, AutoTokenizer
        logger.info("Loading Jina ColBERT-v1 reranker...")
        model_id = "jinaai/jina-colbert-v1-en"
        _colbert_tokenizer = AutoTokenizer.from_pretrained(model_id)
        _colbert_model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        _colbert_model.eval()
        logger.info("Jina ColBERT loaded.")
    return _colbert_tokenizer, _colbert_model





def _mmr_score(
    item_idx: int,
    candidate_indices: list[int],
    selected_indices: list[int],
    similarity_to_query: np.ndarray,
    similarity_matrix: np.ndarray,
    lambda_param: float = 0.5,
) -> float:
    """Calculate the MMR score for a candidate item."""
    sim_to_q = similarity_to_query[item_idx]
    if not selected_indices:
        return float(sim_to_q)

    max_sim_to_selected = max(
        similarity_matrix[item_idx][s_idx] for s_idx in selected_indices
    )
    return float(lambda_param * sim_to_q - (1 - lambda_param) * max_sim_to_selected)


def mmr_select(
    query_emb: np.ndarray,
    chunks: list[dict],
    embeddings: list[np.ndarray],
    top_k: int = 5,
    lambda_param: float = 0.5,
) -> list[dict]:
    """
    Select diverse chunks using Maximal Marginal Relevance.

    Balances relevance to the query with diversity among the selected chunks.

    Args:
        query_emb:    L2-normalised query vector.
        chunks:       List of candidate chunk dicts.
        embeddings:   List of candidate chunk vectors corresponding to chunks.
        top_k:        Number of chunks to return.
        lambda_param: Diversity factor (1.0 = pure relevance, 0.0 = pure diversity).

    Returns:
        List of selected chunk dicts.
    """
    if not chunks:
        return []
    if len(chunks) <= top_k:
        return chunks

    q = query_emb / (np.linalg.norm(query_emb) + 1e-10)
    embs = np.array(embeddings)
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10
    embs = embs / norms

    sim_to_query = np.dot(embs, q)
    sim_matrix = np.dot(embs, embs.T)

    selected: list[int] = []
    candidates = list(range(len(chunks)))

    for _ in range(min(top_k, len(chunks))):
        best_score = -float("inf")
        best_idx = -1

        for c_idx in candidates:
            score = _mmr_score(
                c_idx, candidates, selected, sim_to_query, sim_matrix, lambda_param
            )
            if score > best_score:
                best_score = score
                best_idx = c_idx

        if best_idx != -1:
            selected.append(best_idx)
            candidates.remove(best_idx)

    return [chunks[i] for i in selected]





async def colbert_rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """
    Rerank chunks using Jina ColBERT-v1 via asynchronous executor.

    ColBERT performs late-interaction (MaxSim) between query tokens and chunk
    tokens, providing strong out-of-domain generalisation compared to dense models.

    Args:
        query:  The search query.
        chunks: Candidate chunks to rerank.
        top_k:  Number of chunks to return.

    Returns:
        Sorted list of top_k chunk dicts. Falls back to original chunk list on error.
    """
    if not chunks or len(chunks) <= top_k:
        return chunks

    import torch
    from .embeddings import get_embedding

    loop = asyncio.get_running_loop()
    try:
        _, q_colbert_vecs, _ = await get_embedding(query)

        bge_chunks = []
        jina_chunks = []
        for c in chunks:
            if c.get("colbert_vecs") is not None and q_colbert_vecs is not None:
                bge_chunks.append(c)
            else:
                jina_chunks.append(c)

        scored_results: list[tuple[dict, float]] = []

        if bge_chunks and q_colbert_vecs is not None:
            def _run_native_colbert() -> list[tuple[dict, float]]:
                results = []
                q_norm = torch.tensor(q_colbert_vecs, dtype=torch.float32)
                q_len = max(q_norm.shape[0], 1)
                with torch.no_grad():
                    for c in bge_chunks:
                        d_norm_i = torch.tensor(c["colbert_vecs"], dtype=torch.float32)
                        sim_matrix = torch.matmul(q_norm, d_norm_i.T)
                        max_sim, _ = torch.max(sim_matrix, dim=1)
                        score = torch.sum(max_sim).item() / q_len
                        c["colbert_score"] = float(score)
                        results.append((c, float(score)))
                return results

            bge_scores = await loop.run_in_executor(None, _run_native_colbert)
            scored_results.extend(bge_scores)

        if jina_chunks:
            tokenizer, model = await loop.run_in_executor(None, _load_colbert)

            def _run_jina_colbert() -> list[tuple[dict, float]]:
                texts = [c.get("text", "") for c in jina_chunks]
                results = []

                with torch.no_grad():
                    q_enc = tokenizer(query, return_tensors="pt", truncation=True, max_length=128)
                    q_raw = model(**q_enc).last_hidden_state[0]
                    q_norms = q_raw.norm(dim=-1, keepdim=True).clamp(min=1e-10)
                    q_norm = q_raw / q_norms
                    q_len = q_norm.shape[0]

                    BATCH_SIZE = 4
                    for batch_start in range(0, len(texts), BATCH_SIZE):
                        batch_texts = texts[batch_start : batch_start + BATCH_SIZE]
                        d_enc = tokenizer(
                            batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=256
                        )
                        d_raw = model(**d_enc).last_hidden_state

                        for i in range(d_raw.shape[0]):
                            d_i = d_raw[i]
                            d_norms = d_i.norm(dim=-1, keepdim=True).clamp(min=1e-10)
                            d_norm_i = d_i / d_norms

                            sim_matrix = torch.matmul(q_norm, d_norm_i.T)
                            max_sim, _ = torch.max(sim_matrix, dim=1)
                            score = torch.sum(max_sim).item() / max(q_len, 1)
                            chunk_obj = jina_chunks[batch_start + i]
                            chunk_obj["colbert_score"] = float(score)
                            results.append((chunk_obj, float(score)))

                return results

            jina_scores = await loop.run_in_executor(None, _run_jina_colbert)
            scored_results.extend(jina_scores)

        scored_results.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in scored_results[:top_k]]

    except Exception as e:
        logger.error(f"ColBERT reranking failed: {e}")
        return chunks[:top_k]





async def cross_encoder_rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """
    Rerank chunks using a CrossEncoder (e.g. BAAI/bge-reranker).

    Unlike bi-encoders (which embed query/doc separately) or late-interaction
    models like ColBERT, cross-encoders concatenate query and doc and process
    them through the Transformer together. Slow but highly precise.

    Should be used as the final pass on a small candidate set (e.g. top 10).

    Args:
        query:  The search query.
        chunks: Small candidate chunk list to re-order.
        top_k:  Final number of chunks to return.

    Returns:
        Sorted list of top_k chunk dicts.
    """
    if not chunks:
        return []
    if len(chunks) <= 1:
        return chunks

    from .config import get_reranker_model
    model = get_reranker_model()

    if model is None:
        return chunks[:top_k]

    def _run_cross_encoder() -> list[dict]:
        pairs = [[query, c.get("text", "")] for c in chunks]
        scores = model.predict(pairs)

        if isinstance(scores, (float, int)):
            scores = [scores]

        scored = list(zip(chunks, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        for c, s in scored:
            c["cross_encoder_score"] = float(s)

        return [c for c, _ in scored[:top_k]]

    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(None, _run_cross_encoder)
    except Exception as e:
        logger.error(f"CrossEncoder reranking failed: {e}")
        return chunks[:top_k]





def hybrid_search(
    semantic_chunks: list[dict],
    semantic_scores: list[float],
    bm25: Optional[BM25Okapi],
    bm25_corpus: list[dict],
    query: str,
    query_sparse_vec: Optional[dict[str, float]] = None,
    alpha: float = 0.5,
    top_k: int = 20,
    include_propositions: bool = False,
) -> tuple[list[dict], list[float]]:
    """
    Combine semantic vector scores and BM25 lexical scores using Reciprocal Rank Fusion (RRF).

    Changes vs. older versions: no longer a naive overlap score. Requires the BM25Okapi
    index and corpus built during ingestion to be passed in.

    Args:
        semantic_chunks: The chunks retrieved via vector search.
        semantic_scores: Cosine distances corresponding to the chunks.
        bm25:            Fitted BM25 model, or None if unavailable.
        bm25_corpus:     Original text corpus BM25 was fitted on.
        query:           The search query.
        query_sparse_vec: Optional sparse lexical weights from BGE-M3 model.
        alpha:           Weight given to semantic search (1.0 = semantic only).
        top_k:           Number of initial hybrid results to return.

    Returns:
        Tuple of (combined_chunks, combined_scores).
    """
    import math

    if not include_propositions:
        filtered_sem_chunks = []
        filtered_sem_scores = []
        for c, s in zip(semantic_chunks, semantic_scores):
            if c.get("level") != 0:
                filtered_sem_chunks.append(c)
                filtered_sem_scores.append(s)
        semantic_chunks = filtered_sem_chunks
        semantic_scores = filtered_sem_scores

    if not semantic_chunks:
        return [], []

    if bm25 is None or not bm25_corpus:
        return semantic_chunks[:top_k], semantic_scores[:top_k]

    k = 60
    chunk_scores: dict[int, float] = {}
    chunk_map: dict[int, dict] = {}

    for rank, (chunk, _score) in enumerate(zip(semantic_chunks, semantic_scores)):
        chunk_id = chunk.get("id")
        if chunk_id is None:
            continue
        rrf = 1.0 / (k + rank + 1)
        chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0.0) + (alpha * rrf)
        chunk_map[chunk_id] = chunk

    lexical_pairs = []
    if query_sparse_vec is not None:
        for chunk in bm25_corpus:
            if not include_propositions and chunk.get("level") == 0:
                continue
            doc_sparse = chunk.get("sparse_vector")
            if not doc_sparse:
                continue
            score = 0.0
            for tok, w_q in query_sparse_vec.items():
                if tok in doc_sparse:
                    score += w_q * doc_sparse[tok]
            if score > 0:
                lexical_pairs.append((chunk.get("id"), score, chunk))
    else:
        tokenized_query = query.lower().split()
        bm25_doc_scores = bm25.get_scores(tokenized_query)
        for doc_idx, score in enumerate(bm25_doc_scores):
            if score > 0:
                chunk = bm25_corpus[doc_idx]
                if not include_propositions and chunk.get("level") == 0:
                    continue
                lexical_pairs.append((chunk.get("id"), score, chunk))
    lexical_pairs.sort(key=lambda x: x[1], reverse=True)
    lexical_pairs = lexical_pairs[:top_k * 2]

    for rank, (chunk_id, _score, chunk) in enumerate(lexical_pairs):
        if chunk_id not in chunk_map:
            chunk_map[chunk_id] = chunk
            chunk_scores[chunk_id] = 0.0
        rrf = 1.0 / (k + rank + 1)
        chunk_scores[chunk_id] += (1.0 - alpha) * rrf

    sorted_ids = sorted(chunk_scores.keys(), key=lambda x: chunk_scores[x], reverse=True)
    top_ids = sorted_ids[:top_k]

    final_chunks = [chunk_map[i] for i in top_ids]
    final_scores = [chunk_scores[i] for i in top_ids]

    if not include_propositions:
        return final_chunks, final_scores

    from .db import get_session, Document
    expanded_chunks = []
    expanded_scores = []
    seen_parent_ids = set()

    with get_session() as session:
        for chunk, score in zip(final_chunks, final_scores):
            if chunk.get("level") == 0:
                parent_id = chunk.get("parent_id")
                if parent_id and parent_id not in seen_parent_ids:
                    seen_parent_ids.add(parent_id)
                    parent_doc = session.query(Document).filter_by(id=parent_id).first()
                    if parent_doc:
                        exp_chunk = chunk.copy()
                        exp_chunk["text"] = parent_doc.content
                        exp_chunk["level"] = parent_doc.level
                        expanded_chunks.append(exp_chunk)
                        expanded_scores.append(score)
                    else:
                        expanded_chunks.append(chunk)
                        expanded_scores.append(score)
            else:
                chunk_id = chunk.get("id")
                if chunk_id not in seen_parent_ids:
                    expanded_chunks.append(chunk)
                    expanded_scores.append(score)
                    if chunk.get("level") == 1:
                        seen_parent_ids.add(chunk_id)

    return expanded_chunks, expanded_scores
