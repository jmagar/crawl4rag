from __future__ import annotations

"""Entry point for the Crawl4RAG MCP server.

This minimal bootstrap file sets up a FastMCP instance using the preferred
streamable-HTTP transport (alias "http").  More complex lifecycle management
and tool registration will be added in subsequent iterations.
"""

import os
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastmcp import FastMCP  # type: ignore

###############################################################################
# Lifespan – initialise heavy resources here (browser pools, DB pools, etc.).
# For now we keep it empty; concrete init/teardown will be added later.
###############################################################################


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[None]:
    """Placeholder async lifespan context manager.

    FastMCP will call this once on startup and once on shutdown.  Populate with
    real initialisation (e.g. AsyncWebCrawler, asyncpg pool) in later stages.
    """

    # --- PROVISIONING PLACEHOLDER -----------------------------------------
    # Insert initialisation logic here in future commits, e.g.:
    #   crawler = AsyncWebCrawler(...)
    #   await crawler.__aenter__()
    #   server.state["crawler"] = crawler
    # ----------------------------------------------------------------------
    try:
        yield
    finally:
        # Perform graceful teardown of resources when they are added.
        pass


###############################################################################
# FastMCP server definition.
###############################################################################

mcp = FastMCP(
    name="crawl4rag-mcp",
    description="Modern, self-hostable RAG stack powered by Crawl4AI and pgvector.",
    lifespan=lifespan,
)


###############################################################################
# Bootstrap – run with streamable-http transport.
###############################################################################

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8051"))

    # FastMCP 2.x uses "http" as the canonical name for streamable-HTTP.
    # The alias "streamable-http" is also accepted but kept here for clarity.
    mcp.run(transport="http", host=host, port=port)