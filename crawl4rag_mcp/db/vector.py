from __future__ import annotations

"""Vector store helper utilities built on PostgreSQL + pgvector.

This module houses the DDL statements (for manual execution / migrations) and
runtime helper functions to insert and search documents/code examples using
cosine similarity against pgvector embeddings.
"""

import json
import logging
import math
import os
from typing import Any, Iterable, List, Sequence, Tuple

import httpx  # type: ignore
import asyncpg  # type: ignore

from .pool import get_db_pool

logger = logging.getLogger(__name__)

###############################################################################
# Constants & schema
###############################################################################

VECTOR_DIMENSIONS = 1024  # bge-m3 default dimensionality

# -- DDL ---------------------------------------------------------------------
# The following SQL should be applied to your database (migrations tool,
# psql, etc.) once.  It is provided here for convenience/documentation.
SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS "vector";

CREATE TABLE IF NOT EXISTS sources (
    source_id TEXT PRIMARY KEY,
    summary TEXT,
    total_word_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS crawled_pages (
    id BIGSERIAL PRIMARY KEY,
    url TEXT NOT NULL,
    chunk_number INTEGER NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB,
    source_id TEXT REFERENCES sources(source_id) ON DELETE CASCADE,
    embedding VECTOR(%(dims)s),
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Approx-nearest-neighbor index (IVFFLAT) – requires pgvector ≥ 0.5
CREATE INDEX IF NOT EXISTS crawled_pages_embedding_idx ON crawled_pages
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE TABLE IF NOT EXISTS code_examples (
    id BIGSERIAL PRIMARY KEY,
    url TEXT NOT NULL,
    chunk_number INTEGER NOT NULL,
    content TEXT NOT NULL,
    summary TEXT NOT NULL,
    metadata JSONB,
    source_id TEXT REFERENCES sources(source_id) ON DELETE CASCADE,
    embedding VECTOR(%(dims)s),
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS code_examples_embedding_idx ON code_examples
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
""" % {"dims": VECTOR_DIMENSIONS}

###############################################################################
# Embedding helpers – local HTTP server (bge-m3) + OpenAI fallback
###############################################################################

class EmbeddingError(Exception):
    """Raised when embedding service fails."""


def _chunk_list(seq: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


async def _embed_texts(texts: Sequence[str]) -> List[List[float]]:
    """Call local embedding server (or OpenAI) to embed given texts.

    The function checks `EMBEDDING_URL` env var.  If not set, raises
    `EmbeddingError`.
    """

    base_url = os.getenv("EMBEDDING_URL")
    model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-bge-m3")
    if not base_url:
        raise EmbeddingError("EMBEDDING_URL is not configured")

    # The local server exposes a POST /embeddings compatible with OpenAI spec.
    payload = {"input": list(texts), "model": model_name}
    headers = {"Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(base_url.rstrip("/") + "/embeddings", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return [d["embedding"] for d in data["data"]]
        except (httpx.HTTPError, KeyError) as exc:
            raise EmbeddingError(f"Embedding request failed: {exc}") from exc


###############################################################################
# Core DB operations
###############################################################################

async def add_documents_to_db(
    pool: asyncpg.Pool,
    urls: Sequence[str],
    chunk_numbers: Sequence[int],
    contents: Sequence[str],
    metadatas: Sequence[dict[str, Any]],
    url_to_full_document: dict[str, str] | None = None,
    *,
    batch_size: int = 20,
) -> None:
    """Insert documentation chunks into `crawled_pages` with embeddings."""

    if not (len(urls) == len(chunk_numbers) == len(contents) == len(metadatas)):
        raise ValueError("Input sequence length mismatch for document insertion")

    # Generate embeddings in batches to avoid OOM.
    embeddings: List[List[float]] = []
    for batch_contents in _chunk_list(contents, batch_size):
        batch_emb = await _embed_texts(batch_contents)
        embeddings.extend(batch_emb)

    insert_sql = """
        INSERT INTO crawled_pages
            (url, chunk_number, content, metadata, source_id, embedding)
        VALUES ($1, $2, $3, $4, $5, $6)
        ON CONFLICT (url, chunk_number) DO NOTHING;
    """

    async with pool.acquire() as conn:
        await conn.executemany(
            insert_sql,
            [
                (
                    urls[i],
                    chunk_numbers[i],
                    contents[i],
                    json.dumps(metadatas[i]),
                    metadatas[i].get("source"),
                    embeddings[i],
                )
                for i in range(len(urls))
            ],
        )


async def add_code_examples_to_db(
    pool: asyncpg.Pool,
    urls: Sequence[str],
    chunk_numbers: Sequence[int],
    examples: Sequence[str],
    summaries: Sequence[str],
    metadatas: Sequence[dict[str, Any]],
    *,
    batch_size: int = 20,
) -> None:
    """Insert code examples into `code_examples` table with embeddings."""

    if not (len(urls) == len(chunk_numbers) == len(examples) == len(summaries) == len(metadatas)):
        raise ValueError("Input sequence length mismatch for code example insertion")

    embeddings: List[List[float]] = []
    for batch_content in _chunk_list(examples, batch_size):
        batch_emb = await _embed_texts(batch_content)
        embeddings.extend(batch_emb)

    insert_sql = """
        INSERT INTO code_examples
            (url, chunk_number, content, summary, metadata, source_id, embedding)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        ON CONFLICT (url, chunk_number) DO NOTHING;
    """

    async with pool.acquire() as conn:
        await conn.executemany(
            insert_sql,
            [
                (
                    urls[i],
                    chunk_numbers[i],
                    examples[i],
                    summaries[i],
                    json.dumps(metadatas[i]),
                    metadatas[i].get("source"),
                    embeddings[i],
                )
                for i in range(len(urls))
            ],
        )


###############################################################################
# Vector search helpers
###############################################################################

async def _search_by_embedding(
    conn: asyncpg.Connection,
    table: str,
    query_embedding: List[float],
    match_count: int,
    *,
    source_filter: str | None = None,
) -> List[asyncpg.Record]:
    """Internal helper for cosine-similarity search using pgvector."""

    filter_clause = """AND source_id = $3""" if source_filter else ""
    sql = f"""
        SELECT *, 1 - (embedding <=> $1::vector) AS similarity
        FROM {table}
        WHERE embedding IS NOT NULL {filter_clause}
        ORDER BY embedding <=> $1::vector ASC
        LIMIT $2
    """

    args: List[Any] = [query_embedding, match_count]
    if source_filter:
        args.append(source_filter)

    return await conn.fetch(sql, *args)


async def search_documents(
    pool: asyncpg.Pool,
    query: str,
    *,
    match_count: int = 5,
    source_filter: str | None = None,
) -> List[dict[str, Any]]:
    """Vector search `crawled_pages` by semantic similarity."""

    query_embedding = (await _embed_texts([query]))[0]

    async with pool.acquire() as conn:
        rows = await _search_by_embedding(
            conn, "crawled_pages", query_embedding, match_count, source_filter=source_filter
        )

    return [dict(r) for r in rows]


async def search_code_examples(
    pool: asyncpg.Pool,
    query: str,
    *,
    match_count: int = 5,
    source_id: str | None = None,
) -> List[dict[str, Any]]:
    """Vector search `code_examples` table."""

    query_embedding = (await _embed_texts([query]))[0]
    async with pool.acquire() as conn:
        rows = await _search_by_embedding(
            conn, "code_examples", query_embedding, match_count, source_filter=source_id
        )
    return [dict(r) for r in rows]


###############################################################################
# Source helpers
###############################################################################

async def update_source_info(
    pool: asyncpg.Pool,
    source_id: str,
    summary: str,
    total_word_count: int,
) -> None:
    """Upsert record into `sources` table."""

    sql = """
        INSERT INTO sources (source_id, summary, total_word_count)
        VALUES ($1, $2, $3)
        ON CONFLICT (source_id) DO UPDATE
        SET summary = EXCLUDED.summary,
            total_word_count = EXCLUDED.total_word_count,
            updated_at = now();
    """
    async with pool.acquire() as conn:
        await conn.execute(sql, source_id, summary, total_word_count)