from __future__ import annotations

"""FastMCP tool registrations for Crawl4RAG."""

import json
import logging
from typing import Optional

from fastmcp.server.context import Context  # type: ignore

from ..server import mcp
from .crawl import crawl_single_page, smart_crawl_url
from .rag import rag_query, code_example_query
from ..db.pool import get_db_pool

logger = logging.getLogger(__name__)

###############################################################################
# Crawl tools
###############################################################################


@mcp.tool()
async def crawl_single_page_tool(ctx: Context, url: str) -> str:  # noqa: D401
    """Crawl a single page and store its content in the database."""

    result = await crawl_single_page(url)
    return json.dumps(result, indent=2)


@mcp.tool()
async def smart_crawl_url_tool(
    ctx: Context,
    url: str,
    max_depth: int = 3,
    max_concurrent: int = 10,
    chunk_size: int = 5000,
) -> str:
    """Intelligently crawl a URL (sitemap, txt or recursive webpage)."""

    result = await smart_crawl_url(
        url,
        max_depth=max_depth,
        max_concurrent=max_concurrent,
        chunk_size=chunk_size,
    )
    return json.dumps(result, indent=2)


###############################################################################
# RAG tools
###############################################################################


@mcp.tool()
async def perform_rag_query_tool(
    ctx: Context,
    query: str,
    source: Optional[str] = None,
    match_count: int = 5,
) -> str:
    """Perform a RAG (vector) query over documentation chunks."""

    result = await rag_query(query=query, match_count=match_count, source_filter=source)
    return json.dumps(result, indent=2)


@mcp.tool()
async def search_code_examples_tool(
    ctx: Context,
    query: str,
    source_id: Optional[str] = None,
    match_count: int = 5,
) -> str:
    """Search for relevant code examples."""

    result = await code_example_query(query=query, match_count=match_count, source_filter=source_id)
    return json.dumps(result, indent=2)


###############################################################################
# Source listing tool (simple SQL)
###############################################################################


@mcp.tool()
async def get_available_sources_tool(ctx: Context) -> str:  # noqa: D401
    """Return list of all sources with summaries and stats."""

    pool = await get_db_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT * FROM sources ORDER BY source_id")

    sources = [dict(r) for r in rows]
    return json.dumps({"success": True, "sources": sources, "count": len(sources)}, indent=2)