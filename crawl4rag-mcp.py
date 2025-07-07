from __future__ import annotations

"""Entry point for the Crawl4RAG MCP server.

This minimal bootstrap file sets up a FastMCP instance using the preferred
streamable-HTTP transport (alias "http").  More complex lifecycle management
and tool registration will be added in subsequent iterations.
"""

import os
from crawl4rag_mcp.server import mcp

###############################################################################
# Bootstrap – run with streamable-http transport.
###############################################################################

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8051"))

    # FastMCP 2.x uses "http" as the canonical name for streamable-HTTP.
    # The alias "streamable-http" is also accepted but kept here for clarity.
    mcp.run(transport="http", host=host, port=port)