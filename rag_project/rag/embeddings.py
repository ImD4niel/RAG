"""
embeddings.py — Multi-provider embedding with async safety and caching.
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from typing import Union

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)





class _LRUCache:
    """
    Thread-compatible LRU cache for embedding vectors.
    Limited to ``maxsize`` entries; oldest entry is evicted when full.
    """

    def __init__(self, maxsize: int = 1000) -> None:
        self._store: OrderedDict[str, tuple[list[float], Optional[list[list[float]]], Optional[dict[str, float]]]] = OrderedDict()
        self._maxsize = maxsize if maxsize > 0 else float("inf")

    def get(self, key: str) -> tuple[list[float], Optional[list[list[float]]], Optional[dict[str, float]]] | None:
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def set(self, key: str, value: tuple[list[float], Optional[list[list[float]]], Optional[dict[str, float]]]) -> None:
        self._store[key] = value
        self._store.move_to_end(key)
        while len(self._store) > self._maxsize:
            self._store.popitem(last=False)

    def __contains__(self, key: str) -> bool:
        return key in self._store

    def __setitem__(self, key: str, value: tuple[list[float], Optional[list[list[float]]], Optional[dict[str, float]]]) -> None:
        self.set(key, value)

    def __getitem__(self, key: str) -> tuple[list[float], Optional[list[list[float]]], Optional[dict[str, float]]]:
        result = self.get(key)
        if result is None:
            raise KeyError(key)
        return result


def _make_cache() -> _LRUCache:
    from .config import get_settings
    return _LRUCache(maxsize=get_settings().embed_cache_max)



_cache: _LRUCache | None = None


def get_embedding_cache() -> _LRUCache:
    global _cache
    if _cache is None:
        _cache = _make_cache()
    return _cache







_jina_tokenizer = None
_jina_model = None


def _load_jina():
    """Load jinaai/jina-embeddings-v3 on first call (lazy singleton)."""
    global _jina_tokenizer, _jina_model
    if _jina_model is None:
        from transformers import AutoModel, AutoTokenizer
        from .config import get_settings
        model_id = get_settings().embed_model or "jinaai/jina-embeddings-v3"
        logger.info(f"Loading Jina model: {model_id}")
        _jina_tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        _jina_model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        _jina_model.eval()
        logger.info("Jina model loaded.")
    return _jina_tokenizer, _jina_model


async def late_chunk_embed(
    full_text: str, chunk_texts: list[str]
) -> list[tuple[list[float], Optional[list[list[float]]], Optional[dict[str, float]]]]:
    """
    Late Chunking with Jina embeddings-v3.

    Instead of embedding each chunk independently, this function:
    1. Tokenizes the entire document at once (preserving full context).
    2. Runs the model once to get token-level hidden states.
    3. For each chunk, locates its character-span in the tokenized sequence.
    4. Mean-pools the token embeddings over that span.

    Each chunk embedding therefore 'sees' the full document context, not just
    its local text — prevents context amnesia in standard per-chunk embedding.

    Falls back to standard per-chunk embedding if Jina is unavailable or if
    any tokenization step fails.

    Args:
        full_text:   The complete source document text.
        chunk_texts: Individual chunk strings to embed.

    Returns:
        List of tuples (dense_embedding, colbert_vecs, sparse_vector) for each chunk.
    """
    import torch

    try:
        tokenizer, model = _load_jina()
        encoded = tokenizer(
            full_text,
            return_tensors="pt",
            max_length=8192,
            truncation=True,
            return_offsets_mapping=True,
        )
        offset_mapping = encoded.pop("offset_mapping")[0].tolist()

        def _run_model():
            with torch.no_grad():
                return model(**encoded).last_hidden_state[0]

        loop = asyncio.get_running_loop()
        token_embeddings = await loop.run_in_executor(None, _run_model)

        chunk_embeddings: list[tuple[list[float], Optional[list[list[float]]], Optional[dict[str, float]]]] = []
        search_start = 0

        for chunk_text in chunk_texts:
            chunk_start = full_text.find(chunk_text[:50], search_start)
            if chunk_start == -1:
                emb = token_embeddings.mean(dim=0).numpy()
            else:
                chunk_end = chunk_start + len(chunk_text)
                search_start = chunk_start
                span_indices = [
                    i
                    for i, (s, e) in enumerate(offset_mapping)
                    if s >= chunk_start and e <= chunk_end and s < e
                ]
                emb = (
                    token_embeddings[span_indices].mean(dim=0).numpy()
                    if span_indices
                    else token_embeddings.mean(dim=0).numpy()
                )

            emb = emb / (np.linalg.norm(emb) + 1e-10)
            chunk_embeddings.append((emb.astype(np.float32).tolist(), None, None))

        return chunk_embeddings

    except Exception as e:
        logger.warning(f"late_chunk_embed failed, falling back to per-chunk: {e}")
        return [await get_embedding(chunk) for chunk in chunk_texts]





async def get_embedding(input_data: Union[str, "Image.Image"]) -> tuple[list[float], Optional[list[list[float]]], Optional[dict[str, float]]]:
    """
    Embed a single text string or PIL Image using the configured provider.

    Results are automatically cached in the bounded LRU cache using the
    text content (or ``'<image>'`` for images) as the cache key.

    For the HF provider, the synchronous ``encode()`` call is dispatched to a
    thread pool executor so it never blocks the asyncio event loop.

    Args:
        input_data: Text string or PIL Image to embed.

    Returns:
        A tuple of (dense_embedding, colbert_vecs, sparse_vector).
        `dense_embedding` is a list of floats (L2-normalised).
        `colbert_vecs` is an optional list of lists of floats.
        `sparse_vector` is an optional dict mapping tokens (str) to weight (float).

    Raises:
        ValueError: If EMBED_PROVIDER is ``'jina'`` and called directly
                    (should call ``late_chunk_embed()`` instead).
        RuntimeError: If the embedding API call fails.
    """
    from .config import get_settings, get_embedding_model

    cache = get_embedding_cache()
    settings = get_settings()

    cache_key = input_data if isinstance(input_data, str) else "<image>"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    provider = settings.embed_provider

    if provider == "hf":
        model = get_embedding_model()
        loop = asyncio.get_running_loop()

        if isinstance(input_data, Image.Image):
            emb = await loop.run_in_executor(None, model.encode, input_data)
            emb = emb / (np.linalg.norm(emb) + 1e-10)
            result = (emb.astype(np.float32).tolist(), None, None)
        else:
            if "bge-m3" in settings.embed_model.lower():
                def _bge_m3_encode():
                    try:
                        out = model.encode(
                            [input_data], 
                            return_dense=True, 
                            return_colbert_vecs=True,
                            return_lexical_weights=True
                        )
                        dense = out["dense_vecs"][0]
                        c_vecs = out["colbert_vecs"][0]
                        s_vecs = out["lexical_weights"][0]
                        c_vecs_list = c_vecs.astype(np.float32).tolist()
                    except (TypeError, ValueError):
                        dense = model.encode([input_data])[0]
                        c_vecs_list = None
                        s_vecs = None
                    dense = [float(x) for x in dense.tolist()]
                    return (dense, c_vecs_list, s_vecs)
                result = await loop.run_in_executor(None, _bge_m3_encode)
            else:
                emb = await loop.run_in_executor(
                    None, lambda: model.encode([input_data])[0]
                )
                emb = emb / (np.linalg.norm(emb) + 1e-10)
                result = (emb.astype(np.float32).tolist(), None, None)

    elif provider == "jina":
        logger.warning(
            "get_embedding() called with EMBED_PROVIDER=jina. "
            "Use late_chunk_embed() for the full late-chunking benefit. "
            "Falling back to single-chunk Jina embed."
        )
        results = await late_chunk_embed(
            input_data if isinstance(input_data, str) else "",
            [input_data if isinstance(input_data, str) else ""],
        )
        zero_emb = [0.0] * settings.embedding_dim
        result = results[0] if results else (zero_emb, None, None)

    elif provider == "openai":
        from .config import get_llm_client
        client = get_llm_client()
        text = input_data if isinstance(input_data, str) else str(input_data)
        resp = await client.embeddings.create(model=settings.embed_model, input=text)
        emb = np.array(resp.data[0].embedding, dtype=np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-10)
        result = (emb.tolist(), None, None)

    elif provider == "ollama":
        import httpx
        text = input_data if isinstance(input_data, str) else str(input_data)
        async with httpx.AsyncClient() as http_client:
            r = await http_client.post(
                f"{settings.ollama_endpoint.rstrip('/v1')}/api/embeddings",
                json={"model": settings.embed_model, "prompt": text},
                timeout=60.0,
            )
        emb = np.array(r.json()["embedding"], dtype=np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-10)
        result = (emb.tolist(), None, None)

    else:
        raise ValueError(f"Unsupported EMBED_PROVIDER: {provider!r}")

    cache.set(cache_key, result)
    return result
