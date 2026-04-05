"""
raptor.py — RAPTOR Hierarchical Indexing for DRAGON RAG.

After the base chunk index is built, RAPTOR clusters semantically similar
chunks using UMAP dimensionality reduction + Gaussian Mixture Model (GMM)
clustering, then generates an LLM summary for each cluster. These summaries
are embedded and added to the index as level-2 nodes.

At retrieval time, the vector search spans all levels automatically:
- Level 1: raw chunked text (fine-grained, specific answers)
- Level 2: cluster summaries (broader, conceptual answers)

Result: multi-granularity retrieval — broad questions hit summaries,
specific questions hit chunks, and both are scored together.

Token cost: ~1 LLM call per cluster (one-time at ingestion, not per query).
"""

import asyncio
import logging
import numpy as np

logger = logging.getLogger(__name__)


RAPTOR_MIN_CLUSTER_SIZE = 3   
RAPTOR_MAX_CLUSTERS = 20      
RAPTOR_UMAP_DIMS = 10         


async def build_raptor_tree(chunks: list[dict], level: int = 2) -> list[dict]:
    """
    Build level-2 RAPTOR summary nodes from base-level chunks.

    Args:
        chunks: List of chunk dicts, each with 'text' and 'embedding'.
                These should be level=1 (raw) chunks only.

    Returns:
        List of new summary chunk dicts (level=2) ready for insertion.
        Each dict contains: text, embedding, source, page, level=2.
    """
    text_chunks = [c for c in chunks if c.get("level", 1) == 1 and c.get("text")]
    if len(text_chunks) < RAPTOR_MIN_CLUSTER_SIZE:
        logger.info(f"RAPTOR: Not enough chunks ({len(text_chunks)}) to cluster. Skipping.")
        return []

    logger.info(f"RAPTOR: Building tree from {len(text_chunks)} level-1 chunks...")

    embeddings = np.array([c["embedding"] for c in text_chunks], dtype=np.float32)

    reduced = _umap_reduce(embeddings, n_components=min(RAPTOR_UMAP_DIMS, embeddings.shape[1]))

    labels, n_clusters = _gmm_cluster(reduced, max_clusters=min(RAPTOR_MAX_CLUSTERS, len(text_chunks) // RAPTOR_MIN_CLUSTER_SIZE))
    logger.info(f"RAPTOR: Identified {n_clusters} clusters.")

    summary_nodes = []
    tasks = []
    cluster_groups = []
    for cluster_id in range(n_clusters):
        members = [text_chunks[i] for i, lbl in enumerate(labels) if lbl == cluster_id]
        if len(members) < 2:
            continue
        cluster_groups.append((cluster_id, members))
        tasks.append(_summarise_cluster(cluster_id, members))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    for (cluster_id, members), result in zip(cluster_groups, results):
        if isinstance(result, Exception):
            logger.warning(f"RAPTOR cluster {cluster_id} summarisation failed: {result}")
            continue
        summary_text, summary_embedding = result
        if summary_text and summary_embedding is not None:
            summary_nodes.append({
                "text": summary_text,
                "source": f"[RAPTOR-L2] cluster_{cluster_id}",
                "page": None,
                "media_type": "text",
                "media_path": None,
                "embedding": summary_embedding,
                "level": level,
                "section": f"cluster_{cluster_id}",
                "chunk_id": -(cluster_id + 10000),
            })

    logger.info(f"RAPTOR: Generated {len(summary_nodes)} level-{level} summary nodes.")

    from .config import get_settings
    settings = get_settings()

    if (
        level < settings.raptor_max_level and 
        len(summary_nodes) >= RAPTOR_MIN_CLUSTER_SIZE
    ):
        l_next_nodes = await build_raptor_tree(
            summary_nodes, level=level + 1
        )
        return summary_nodes + l_next_nodes

    return summary_nodes


async def _summarise_cluster(cluster_id: int, members: list[dict]) -> tuple:
    """Generate an LLM summary for a cluster of chunks and embed it."""
    from .llm import safe_llm_call
    from .embeddings import get_embedding

    combined = "\n\n".join([m["text"][:600] for m in members[:8]])  
    messages = [
        {
            "role": "system",
            "content": (
                "You are a document summarisation expert. "
                "Write a comprehensive, dense paragraph summarising the key facts, "
                "concepts, and information in these document passages. "
                "Be specific — preserve names, numbers, and definitions. "
                "The summary will be used for information retrieval, not for human reading."
            )
        },
        {
            "role": "user",
            "content": f"Summarise these passages:\n\n{combined}"
        }
    ]
    response = await safe_llm_call(messages, temperature=0)
    if response is None:
        return None, None

    summary_text = response.choices[0].message.content.strip()
    prefixed = f"[Cluster Summary — {len(members)} chunks]\n\n{summary_text}"
    embedding = await get_embedding(prefixed)
    return prefixed, embedding


def _umap_reduce(embeddings: np.ndarray, n_components: int = 10) -> np.ndarray:
    """Reduce embedding dimensionality with UMAP for clustering."""
    try:
        import umap
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=min(15, len(embeddings) - 1),
            min_dist=0.0,
            metric="cosine",
            random_state=42,
        )
        return reducer.fit_transform(embeddings)
    except Exception as e:
        logger.warning(f"RAPTOR: UMAP failed ({e}), using raw embeddings for clustering.")
        return embeddings


def _gmm_cluster(embeddings: np.ndarray, max_clusters: int = 10) -> tuple[np.ndarray, int]:
    """Cluster embeddings with Gaussian Mixture Model; select n_clusters via BIC."""
    try:
        from sklearn.mixture import GaussianMixture
        from sklearn.exceptions import ConvergenceWarning
        import warnings

        best_n = 2
        best_bic = np.inf

        for n in range(2, min(max_clusters + 1, len(embeddings))):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    gm = GaussianMixture(n_components=n, random_state=42, max_iter=200)
                    gm.fit(embeddings)
                    bic = gm.bic(embeddings)
                    if bic < best_bic:
                        best_bic = bic
                        best_n = n
            except Exception:
                break

        gm_final = GaussianMixture(n_components=best_n, random_state=42, max_iter=200)
        labels = gm_final.fit_predict(embeddings)
        return labels, best_n

    except Exception as e:
        logger.warning(f"RAPTOR: GMM clustering failed ({e}), assigning all to cluster 0.")
        return np.zeros(len(embeddings), dtype=int), 1
