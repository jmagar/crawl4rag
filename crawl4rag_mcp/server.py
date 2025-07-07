from __future__ import annotations

"""FastMCP server instance for Crawl4RAG.

This centralises the FastMCP object so that other modules (tools, routes, etc.)
can import it without circular dependencies.
"""

import os
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastmcp import FastMCP  # type: ignore

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[None]:
    # Heavy resources will be initialised lazily by each module when first
    # requested (e.g. DB pool, crawler).  Nothing to do here yet.
    try:
        yield
    finally:
        # At shutdown we could close the db pool or crawler if they were
        # initialised.  These helpers are safe to call even if not started.
        from .db.pool import close_db_pool  # local import to avoid upfront cost

        await close_db_pool()


# ---------------------------------------------------------------------------
# FastMCP instance
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="crawl4rag-mcp",
    description="Self-hostable RAG stack powered by Crawl4AI and pgvector.",
    lifespan=lifespan,
)