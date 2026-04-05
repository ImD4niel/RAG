"""
Alembic env.py — wired to DRAGON RAG ORM models.

Connection URL is read from rag.config.Settings (honours the project .env file)
so that `alembic upgrade head` uses exactly the same database as the pipeline.

Usage:
    alembic upgrade head                                    # apply all pending migrations
    alembic revision --autogenerate -m "describe change"   # generate a new migration
    alembic downgrade -1                                    # roll back one step
"""

from __future__ import annotations

import sys
import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context

# ---------------------------------------------------------------------------
# Make the rag package importable from the project root
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import project ORM Base and all models so autogenerate can see them
from rag.db import Base  # noqa: E402 — must be after sys.path insert
import rag.db  # noqa: F401 — registers all ORM models on Base.metadata

# ---------------------------------------------------------------------------
# Alembic Config
# ---------------------------------------------------------------------------
config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Point autogenerate at the project's metadata
target_metadata = Base.metadata


def _get_url() -> str:
    """Read DB URL from project Settings (respects .env). Never hardcoded."""
    from rag.config import get_settings
    return get_settings().database_url


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (generates SQL without connecting)."""
    url = _get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode (connects to DB and applies changes)."""
    configuration = config.get_section(config.config_ini_section, {})
    # Override INI url with Settings-derived url — no duplicate connection string
    configuration["sqlalchemy.url"] = _get_url()

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,            # detect column type changes
            compare_server_default=True,  # detect default value changes
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
