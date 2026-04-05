"""
enrich.py — Offline Enrichment Runner.

Executes offline tasks such as proposition-level indexing.
"""

import asyncio
import logging
from typing import Optional

from sqlalchemy import select
import numpy as np

from .db import get_session, Document
from .embeddings import get_embedding
from .proposer import extract_propositions

logger = logging.getLogger(__name__)

async def run_proposition_enrichment(document_id: Optional[str] = None) -> dict:
    """
    Offline process to decompose standard L1 chunks into atomic L0 propositions.
    If document_id is provided, filter by that file_hash.
    Only processes chunks that do not already have L0 children.
    """
    logger.info(f"Starting proposition enrichment (document_id={document_id})")
    stats = {"chunks_processed": 0, "propositions_created": 0, "errors": 0}

    with get_session() as session:
        l0_subquery = select(Document.parent_id).where(Document.level == 0).subquery()
        query = session.query(Document).filter(
            Document.level == 1,
            Document.media_type != "image",
            Document.id.notin_(l0_subquery)
        )
        if document_id:
            query = query.filter(Document.file_hash == document_id)
        chunks_to_process = query.all()
        chunks_data = [
            {
                "id": c.id, 
                "text": c.content, 
                "source": c.source, 
                "file_hash": c.file_hash,
                "doc_type": c.doc_type,
            } for c in chunks_to_process
        ]

    if not chunks_data:
        logger.info("No eligible chunks found for enrichment.")
        return stats

    logger.info(f"Found {len(chunks_data)} chunks to enrich.")

    for chunk in chunks_data:
        try:
            propositions = await extract_propositions(chunk)
            new_docs = []
            for prop_text in propositions:
                emb, colbert, _ = await get_embedding(prop_text)
                doc = Document(
                    doc_hash=f"{chunk['id']}_{hash(prop_text)}",  
                    doc_type="proposition",
                    source=chunk["source"],
                    file_hash=chunk["file_hash"],
                    parent_id=chunk["id"],
                    level=0,
                    content=prop_text,
                    embedding=emb,
                    colbert_vecs=colbert,
                    media_type="text",
                    embedding_valid=True
                )
                new_docs.append(doc)
            with get_session() as session:
                session.add_all(new_docs)
                session.commit()
            stats["chunks_processed"] += 1
            stats["propositions_created"] += len(new_docs)
            logger.debug(f"Enriched chunk {chunk['id']} -> {len(new_docs)} propositions")
        except Exception as e:
            logger.error(f"Failed enriching chunk {chunk['id']}: {e}")
            stats["errors"] += 1

    logger.info(f"Enrichment complete. Stats: {stats}")
    return stats
