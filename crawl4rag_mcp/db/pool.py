from __future__ import annotations

"""Connection-pool utilities for Postgres/pgvector."""

import os
import logging
from typing import Optional

import asyncpg  # type: ignore

logger = logging.getLogger(__name__)

# Cache the pool globally so repeated calls reuse the same object.
_pool: Optional[asyncpg.Pool] = None


async def get_db_pool() -> asyncpg.Pool:
    """Create (or return existing) global asyncpg pool.

    Connection params are read from environment variables compatible with the
    existing .env layout used by the original project.
    """

    global _pool  # noqa: PLW0603  # pylint: disable=global-statement

    if _pool is not None:
        return _pool

    host = os.getenv("POSTGRES_HOST", "localhost")
    port = int(os.getenv("POSTGRES_PORT", "5432"))
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "postgres")
    database = os.getenv("POSTGRES_DB", "crawl4rag")

    logger.info("Creating asyncpg pool -> %s:%s/%s", host, port, database)
    _pool = await asyncpg.create_pool(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
        min_size=2,
        max_size=20,
        statement_cache_size=0,  # avoid stale prepared statements across versions
    )
    return _pool


async def close_db_pool() -> None:
    """Close and reset the global connection pool (if initialised)."""

    global _pool  # noqa: PLW0603
    if _pool is not None:
        logger.info("Closing database pool")
        await _pool.close()
        _pool = None