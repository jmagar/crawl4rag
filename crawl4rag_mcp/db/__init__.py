"""Database helpers for Crawl4RAG MCP.

This sub-package centralises all persistence logic (PostgreSQL + pgvector).
Other modules should *only* import the public helpers exposed here to avoid
coupling to the underlying database driver.
"""

from __future__ import annotations

from .pool import get_db_pool, close_db_pool  # noqa: F401
from .vector import (
    add_documents_to_db,  # noqa: F401
    add_code_examples_to_db,  # noqa: F401
    search_documents,  # noqa: F401
    search_code_examples,  # noqa: F401
    update_source_info,  # noqa: F401
)

__all__ = [
    "get_db_pool",
    "close_db_pool",
    "add_documents_to_db",
    "add_code_examples_to_db",
    "search_documents",
    "search_code_examples",
    "update_source_info",
]