"""
db.py — Database schema and session management.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from sqlalchemy import (
    Boolean,
    Column,
    Integer,
    String,
    Float,
    DateTime,
    ForeignKey,
    create_engine,
    text,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Mapped
from sqlalchemy.pool import QueuePool
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB

from .config import get_settings

logger = logging.getLogger(__name__)

Base = declarative_base()




class IndexedFile(Base):
    __tablename__ = "indexed_files"
    id: Mapped[int] = Column(Integer, primary_key=True, autoincrement=True)
    filename: Mapped[str] = Column(String, unique=True, nullable=False, index=True)
    file_hash: Mapped[str] = Column(String, nullable=False)
    embed_model: Mapped[str] = Column(String, nullable=False)
    processed_at: Mapped[datetime] = Column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )

class Document(Base):
    __tablename__ = "documents"
    id: Mapped[int] = Column(Integer, primary_key=True, autoincrement=True)
    doc_hash: Mapped[str] = Column(String, unique=True, index=True, nullable=False)
    doc_type: Mapped[str] = Column(String, nullable=False) 
    source: Mapped[str] = Column(String, nullable=False)
    page: Mapped[int] = Column(Integer, nullable=True)
    file_hash: Mapped[str] = Column(String, nullable=True) 
    parent_id: Mapped[int] = Column(Integer, nullable=True) 
    level: Mapped[int] = Column(Integer, default=1) 
    content: Mapped[str] = Column(String, nullable=False)
    embedding = Column(Vector(1024), nullable=True)
    colbert_vecs: Mapped[list] = Column(JSONB, nullable=True)
    sparse_vector: Mapped[dict] = Column(JSONB, nullable=True)
    media_path: Mapped[str] = Column(String, nullable=True)
    media_type: Mapped[str] = Column(String, default="text")
    embedding_valid: Mapped[bool] = Column(
        Boolean, nullable=False, default=True, server_default="true"
    )
    created_at: Mapped[datetime] = Column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )

class EvalResult(Base):
    """Stores LLM-as-a-judge evaluation results for longitudinal quality tracking."""
    __tablename__ = "eval_results"
    id: Mapped[int] = Column(Integer, primary_key=True, autoincrement=True)
    query: Mapped[str] = Column(String, nullable=False)
    faithfulness: Mapped[float] = Column(Float, nullable=False)
    answer_relevance: Mapped[float] = Column(Float, nullable=False)
    context_recall: Mapped[float] = Column(Float, nullable=False)
    overall: Mapped[float] = Column(Float, nullable=False)
    evaluated_at: Mapped[datetime] = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), index=True
    )

class SyntheticQA(Base):
    """Stores LLM-generated Synthetic QA pairs for validation bootstrapping."""
    __tablename__ = "synthetic_qa"
    id: Mapped[int] = Column(Integer, primary_key=True, autoincrement=True)
    chunk_id: Mapped[int] = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    question: Mapped[str] = Column(String, nullable=False)
    answer: Mapped[str] = Column(String, nullable=False)
    created_at: Mapped[datetime] = Column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )




_engine = None
_SessionLocal = None

def _get_engine():
    global _engine, _SessionLocal
    if _engine is None:
        settings = get_settings()
        _engine = create_engine(
            settings.database_url,
            poolclass=QueuePool,
            pool_size=settings.embed_concurrency + 5,
            max_overflow=10,
            pool_pre_ping=True,
        )
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
    return _engine, _SessionLocal

@contextmanager
def get_session():
    """Provide a transactional scope around a series of operations."""
    _, SessionLocal = _get_engine()
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()




def init_db():
    """
    Initialise DB schema as a runtime safety net.

    NOTE: Alembic is now the canonical schema authority.
    Use `alembic upgrade head` from the project root for all schema changes.
    This function exists only as a fallback for first-time local setup
    when Alembic has not yet been run. It will NOT handle schema migrations
    (column additions, index changes, type changes).

    Production workflow:
        alembic upgrade head        # first run, or after pulling new migrations
        alembic revision --autogenerate -m "your change"   # to generate a new one
    """
    engine, _ = _get_engine()
    # Create vector extension if not exists (idempotent)
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()

    Base.metadata.create_all(bind=engine)

    try:
        with engine.connect() as conn:
            conn.execute(text(
                """
                CREATE INDEX IF NOT EXISTS documents_embedding_hnsw
                ON documents USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 128)
                """
            ))
            conn.commit()
    except Exception as e:
        logger.warning(f"Could not create HNSW index: {e}")

