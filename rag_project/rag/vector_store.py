"""
vector_store.py — Document ingestion and PostgreSQL + pgvector management.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

import numpy as np


from rank_bm25 import BM25Okapi
from sqlalchemy import text

from .chunker import create_parent_child_chunks, chunk_table
from .cleaner import get_file_hash, get_doc_hash, deduplicate_chunks
from .config import get_settings, get_embed_signature
from .db import get_session, IndexedFile, Document
from .embeddings import get_embedding, late_chunk_embed
from .loaders import (
    load_txt, load_pdf, load_docx, load_pptx,
    load_tabular_data, load_unstructured,
)
from .raptor import build_raptor_tree

logger = logging.getLogger(__name__)





def _get_changed_files(folder_path: str) -> list[str]:
    """
    Compare files on disk with hashes in PG `indexed_files` table.
    Returns absolute paths of files that are new or whose hash has changed.
    """
    changed = []
    embed_sig = get_embed_signature()

    with get_session() as session:
        for filename in os.listdir(folder_path):
            if filename.startswith(".") or not os.path.isfile(os.path.join(folder_path, filename)):
                continue

            filepath = os.path.abspath(os.path.join(folder_path, filename))
            current_hash = get_file_hash(filepath)

            record = session.query(IndexedFile).filter_by(filename=filename).first()

            if not record:
                changed.append(filepath)
            elif record.file_hash != current_hash:
                logger.info(f"File changed: {filename}")
                changed.append(filepath)
                settings = get_settings()
                old_sig_compat = not settings.embed_model_revision and (
                    record.embed_model == embed_sig.rsplit("_", 1)[0]
                    or record.embed_model == embed_sig  
                )
                if old_sig_compat:
                    logger.info(
                        f"Signature format migration for {filename}: "
                        f"'{record.embed_model}' → '{embed_sig}' (no revision pinned, "
                        f"treating as compatible — updating registry without purge)."
                    )
                    record.embed_model = embed_sig
                    session.commit()
                else:
                    logger.warning(
                        f"Embedding model changed ({record.embed_model} → {embed_sig}). "
                        f"Purging {filename} embeddings and re-indexing."
                    )
                    deleted = (
                        session.query(Document)
                        .filter(Document.source.contains(filename))
                        .delete(synchronize_session=False)
                    )
                    session.commit()
                    logger.info(f"Purged {deleted} stale Document rows for {filename}.")
                    changed.append(filepath)

    return changed


def _update_file_registry(filepaths: list[str]) -> None:
    """Record successfully indexed files and their hashes in the DB."""
    embed_sig = get_embed_signature()
    with get_session() as session:
        for filepath in filepaths:
            filename = os.path.basename(filepath)
            file_hash = get_file_hash(filepath)

            record = session.query(IndexedFile).filter_by(filename=filename).first()
            if record:
                record.file_hash = file_hash
                record.embed_model = embed_sig
            else:
                record = IndexedFile(
                    filename=filename,
                    file_hash=file_hash,
                    embed_model=embed_sig,
                )
                session.add(record)
        session.commit()





async def _embed_chunks_with_semaphore(chunks: list[dict], concurrency: int) -> list[dict]:
    """
    Assign embeddings to a list of chunks, bounded by a semaphore to prevent 429s.
    Supports Jina Late Chunking when configured.
    """
    settings = get_settings()

    if settings.embed_provider == "jina":
        logger.info("Using Jina Late Chunking for document embeddings...")
        source_groups: dict[str, list[dict]] = {}
        for c in chunks:
            source = c.get("source", "unknown")
            source_groups.setdefault(source, []).append(c)

        for source, group_chunks in source_groups.items():
            if not group_chunks:
                continue
            full_text = "\n\n".join(c.get("text", "") for c in group_chunks if c.get("media_type", "text") == "text")
            texts_to_embed = [c.get("text", "") for c in group_chunks if c.get("media_type", "text") == "text"]

            if texts_to_embed:
                chunk_embeds = await late_chunk_embed(full_text, texts_to_embed)
                emb_idx = 0
                for c in group_chunks:
                    if c.get("media_type", "text") == "text" and emb_idx < len(chunk_embeds):
                        c["embedding"], c["colbert_vecs"], c["sparse_vector"] = chunk_embeds[emb_idx]
                        emb_idx += 1
                    elif c.get("media_type") == "image":
                        c["embedding"] = np.zeros(settings.embedding_dim, dtype=np.float32)
                        c["sparse_vector"] = None

            for c in group_chunks:
                if "embedding" not in c or c["embedding"] is None:
                    c["embedding"] = np.zeros(settings.embedding_dim, dtype=np.float32)
                    c["sparse_vector"] = None

        return chunks

    sem = asyncio.Semaphore(concurrency)

    async def _embed_one(chunk: dict):
        async with sem:
            if chunk.get("needs_vlm_description") and chunk.get("media_path"):
                try:
                    from .llm import describe_image
                    with open(chunk["media_path"], "rb") as img_file:
                        img_bytes = img_file.read()
                    description = await describe_image(img_bytes)
                    if description:
                        chunk["text"] = description
                        chunk.pop("needs_vlm_description", None)
                        if settings.vlm_endpoint and chunk.get("media_type") == "image" and description:
                            emb, colbert, sparse = await get_embedding(description)
                            chunk["embedding"] = emb
                            chunk["colbert_vecs"] = colbert
                            chunk["sparse_vector"] = sparse
                            chunk["embedding_valid"] = True
                            return chunk
                except Exception as e:
                    logger.warning(
                        f"VLM description failed for {chunk.get('media_path')}: {e}. "
                        f"Falling back to zero vector."
                    )
                chunk["embedding"] = np.zeros(settings.embedding_dim, dtype=np.float32)
                chunk["sparse_vector"] = None
                chunk["embedding_valid"] = False
                return chunk

            if chunk.get("media_type") == "image":
                chunk["embedding"] = np.zeros(settings.embedding_dim, dtype=np.float32)
                chunk["sparse_vector"] = None
                chunk["embedding_valid"] = False
                return chunk

            text = chunk.get("text", "")
            try:
                emb, colbert, sparse = await get_embedding(text)
                chunk["embedding"] = emb
                chunk["colbert_vecs"] = colbert
                chunk["sparse_vector"] = sparse
                chunk["embedding_valid"] = True
            except Exception as e:
                logger.error(
                    f"Failed to embed chunk (id={chunk.get('chunk_id')}): {e}. "
                    "Storing zero-vector; chunk excluded from retrieval."
                )
                chunk["embedding"] = np.zeros(settings.embedding_dim, dtype=np.float32)
                chunk["sparse_vector"] = None
                chunk["embedding_valid"] = False
            return chunk

    tasks = [_embed_one(c) for c in chunks]
    return await asyncio.gather(*tasks)





def _insert_into_postgres_sync(chunks: list[dict]) -> None:
    """Synchronous function to insert chunks into the Postgres documents table."""
    from pgvector.sqlalchemy import Vector

    with get_session() as session:
        existing_hashes = {
            r[0] for r in session.query(Document.doc_hash).all()
        }

        new_docs = []
        for c in chunks:
            h = get_doc_hash(c.get("text", ""))
            if h in existing_hashes:
                continue

            emb = c.get("embedding")
            if emb is not None:
                 if isinstance(emb, np.ndarray):
                     emb = emb.tolist()

            doc = Document(
                doc_hash=h,
                doc_type=c.get("doc_type", "chunk"),
                source=c.get("source", "unknown"),
                page=c.get("page"),
                file_hash=c.get("file_hash"),
                parent_id=c.get("parent_id"),
                level=c.get("level", 1),
                content=c.get("text", ""),
                embedding=emb,
                colbert_vecs=c.get("colbert_vecs"),
                sparse_vector=c.get("sparse_vector"),
                media_path=c.get("media_path"),
                media_type=c.get("media_type", "text"),
                embedding_valid=c.get("embedding_valid", True),
            )
            new_docs.append(doc)
            existing_hashes.add(h)

        if new_docs:
            session.add_all(new_docs)
            try:
                session.commit()
                logger.info(f"Inserted {len(new_docs)} new rows into database.")
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to insert documents: {e}")
                raise


async def insert_into_postgres(chunks: list[dict]) -> None:
    """Async wrapper to prevent DB inserts from blocking the event loop."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _insert_into_postgres_sync, chunks)





def _build_bm25_from_db() -> tuple[Optional[BM25Okapi], list[dict]]:
    """
    Fetch documents from DB and build BM25 index.
    """
    with get_session() as session:
        docs = (
            session.query(Document)
            .filter(Document.media_type != "image")
            .all()
        )

        if not docs:
            return None, []

        corpus_dicts = []
        texts = []
        for d in docs:
            texts.append(d.content)
            corpus_dicts.append({
                "id": d.id,
                "text": d.content,
                "doc_type": d.doc_type,
                "source": d.source,
                "page": d.page,
                "level": d.level,  
                "sparse_vector": d.sparse_vector,
            })

    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    return bm25, corpus_dicts





async def build_or_load_index(
    folder_path: str | None = None, force_reindex: bool = False
) -> tuple[Optional[BM25Okapi], list[dict]]:
    """
    Ingest folder contents incrementally, embed them, insert into postgres,
    build RAPTOR nodes if enabled, and return the BM25 index.
    """
    if folder_path is None:
        base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        folder_path = os.path.join(base, "data")

    if force_reindex:
        logger.info("force_reindex=True. Attempting full reparse of data folder.")
        changed_files = [
            os.path.abspath(os.path.join(folder_path, f))
            for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f))
        ]
        with get_session() as session:
            session.query(Document).delete()
            session.query(IndexedFile).delete()
            session.commit()
            logger.info("Cleared DB for force reindex.")
    else:
        changed_files = _get_changed_files(folder_path)

    if not changed_files:
        logger.info("No files modified. Index is up to date.")
        return await asyncio.get_running_loop().run_in_executor(None, _build_bm25_from_db)

    raw_docs = []
    from .loaders import load_txt, load_pdf, load_docx, load_pptx, load_tabular_data, load_unstructured
    for filepath in changed_files:
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".txt":
            raw_docs.extend(load_txt(filepath))
        elif ext == ".pdf":
            raw_docs.extend(load_pdf(filepath))
        elif ext == ".docx":
            raw_docs.extend(load_docx(filepath))
        elif ext == ".pptx":
            raw_docs.extend(load_pptx(filepath))
        elif ext in (".csv", ".xls", ".xlsx"):
             raw_docs.extend(load_tabular_data(filepath))
        else:
             raw_docs.extend(load_unstructured(filepath))

    if not raw_docs:
        logger.warning("Changed files resulted in 0 extracted chunks.")
        _update_file_registry(changed_files)
        return await asyncio.get_running_loop().run_in_executor(None, _build_bm25_from_db)

    chunks = []
    for doc in raw_docs:
        if doc.get("media_type") == "image":
            chunks.append(doc)
            continue

        text = doc.get("text", "")
        if not text:
            continue

        file_hash = get_file_hash(doc["source"])

        if doc.get("doc_type") in ("pdf_table", "table"):
            table_chunks = chunk_table(text)
            for tc in table_chunks:
                meta = doc.copy()
                meta["text"] = tc["text"]
                meta["doc_type"] = "table"           
                meta["table_header"] = tc.get("table_header", "")
                meta["is_table_chunk"] = True
                meta["file_hash"] = file_hash
                meta["level"] = 1
                chunks.append(meta)
        else:
            _, child_chunks = create_parent_child_chunks(text)
            for child in child_chunks:
                child_meta = doc.copy()
                child_meta["text"] = child["text"]
                child_meta["parent_id"] = child["parent_id"]
                child_meta["file_hash"] = file_hash
                child_meta["level"] = 1
                chunks.append(child_meta)

    chunks = deduplicate_chunks(chunks)

    settings = get_settings()
    concurrency = settings.embed_concurrency
    logger.info(f"Embedding {len(chunks)} chunks (concurrency={concurrency})...")
    chunks = await _embed_chunks_with_semaphore(chunks, concurrency)

    await insert_into_postgres(chunks)

    with get_session() as session:
        l2_count = session.query(Document).filter(Document.level == 2).count()
        if l2_count > 0:
            logger.info(f"Deleting existing RAPTOR nodes for rebuild...")
            session.query(Document).filter(Document.level >= 2).delete()
            session.commit()

        l1_docs = session.query(Document).filter(Document.level == 1).all()
        l1_chunks = []
        for d in l1_docs:
             if d.embedding is not None and d.media_type != "image":
                 emb = np.array(d.embedding, dtype=np.float32)
                 l1_chunks.append({"id": d.id, "text": d.content, "embedding": emb})

    if l1_chunks and len(l1_chunks) > 10: 
        logger.info(f"Building RAPTOR tree from {len(l1_chunks)} L1 nodes...")
        raptor_nodes = await build_raptor_tree(l1_chunks)

        if raptor_nodes:
            logger.info(f"Embedding {len(raptor_nodes)} RAPTOR summaries...")
            raptor_nodes = await _embed_chunks_with_semaphore(raptor_nodes, concurrency)

            for rn in raptor_nodes:
                rn["doc_type"] = "raptor_summary"
            await insert_into_postgres(raptor_nodes)

    _update_file_registry(changed_files)

    logger.info("Ingestion pipeline complete.")

    return await asyncio.get_running_loop().run_in_executor(None, _build_bm25_from_db)





def search_postgres(query_embedding: np.ndarray, top_k: int = 20) -> list[dict]:
    """Retrieve top_k chunks by cosine distance using pgvector ORM."""
    query_emb_list = [float(x) for x in query_embedding.tolist()]

    chunks = []
    with get_session() as session:
        results = (
            session.query(
                Document,
                Document.embedding.cosine_distance(query_emb_list).label("distance")
            )
            .filter(Document.embedding.isnot(None))
            .filter(Document.embedding_valid == True)
            .order_by(text("distance")) 
            .limit(top_k)
            .all()
        )

        for doc, dist in results:
            chunks.append({
                "id": doc.id,
                "text": doc.content,
                "doc_type": doc.doc_type,
                "source": doc.source,
                "page": doc.page,
                "level": doc.level,
                "distance": float(dist) if dist is not None else 1.0,
                "embedding": np.array(doc.embedding, dtype=np.float32) if doc.embedding is not None else None,
            })

    return chunks
