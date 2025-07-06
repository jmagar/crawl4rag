from __future__ import annotations

"""Retrieval-augmented generation helpers (vector search + reranking).

This module provides *low-level* async functions that wrap pgvector similarity
search and optional CrossEncoder reranking.  FastMCP tools will call these to
serve user queries.
"""

from typing import Any, List, Dict, Optional
import logging
import os

from sentence_transformers import CrossEncoder  # type: ignore

from ..db import search_documents, search_code_examples
from ..db.vector import EmbeddingError
from ..db.pool import get_db_pool

logger = logging.getLogger(__name__)

###############################################################################
# Reranker initialisation (lazy singleton)
###############################################################################

_RERANK_MODEL: Optional[CrossEncoder] = None


def _get_reranker(force_reload: bool = False) -> Optional[CrossEncoder]:  # noqa: D401
    """Return a singleton CrossEncoder if USE_RERANKING=true and model loads."""

    global _RERANK_MODEL  # noqa: PLW0603
    if _RERANK_MODEL is not None and not force_reload:
        return _RERANK_MODEL

    if os.getenv("USE_RERANKING", "false").lower() != "true":
        return None

    model_name = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    try:
        logger.info("Loading rerank model: %s", model_name)
        _RERANK_MODEL = CrossEncoder(model_name)
        return _RERANK_MODEL
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load rerank model – disabling rerank (%s)", exc)
        _RERANK_MODEL = None
        return None


###############################################################################
# Core rerank helper
###############################################################################


def _rerank_results(
    model: CrossEncoder,
    query: str,
    results: List[Dict[str, Any]],
    *,
    content_key: str = "content",
) -> List[Dict[str, Any]]:
    if not results:
        return results

    texts = [r.get(content_key, "") for r in results]
    pairs = [[query, t] for t in texts]
    try:
        scores = model.predict(pairs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Rerank failed: %s", exc)
        return results

    for i, s in enumerate(scores):
        results[i]["rerank_score"] = float(s)

    return sorted(results, key=lambda x: x.get("rerank_score", 0), reverse=True)


###############################################################################
# Public async helpers
###############################################################################


async def rag_query(
    query: str,
    *,
    match_count: int = 5,
    source_filter: str | None = None,
) -> Dict[str, Any]:
    """Execute vector search (and optional rerank) over crawled_pages."""

    if not query or not isinstance(query, str):
        return {"success": False, "error": "Query must be non-empty string"}

    try:
        pool = await get_db_pool()
        results = await search_documents(
            pool=pool,
            query=query,
            match_count=match_count,
            source_filter=source_filter,
        )
    except EmbeddingError as e:
        return {"success": False, "error": f"Embedding error: {e}", "type": "embedding_error"}
    except Exception as exc:  # noqa: BLE001
        logger.exception("RAG vector search failed")
        return {"success": False, "error": str(exc)}

    rerank_applied = False
    model = _get_reranker()
    if model and results:
        results = _rerank_results(model, query, results, content_key="content")
        rerank_applied = True

    return {
        "success": True,
        "query": query,
        "source_filter": source_filter or "",
        "results": results,
        "count": len(results),
        "rerank_applied": rerank_applied,
    }


async def code_example_query(
    query: str,
    *,
    match_count: int = 5,
    source_filter: str | None = None,
) -> Dict[str, Any]:
    """Vector search + optional rerank over code_examples."""

    if not query or not isinstance(query, str):
        return {"success": False, "error": "Query must be non-empty string"}

    try:
        pool = await get_db_pool()
        results = await search_code_examples(
            pool=pool,
            query=query,
            match_count=match_count,
            source_id=source_filter,
        )
    except EmbeddingError as e:
        return {"success": False, "error": f"Embedding error: {e}", "type": "embedding_error"}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Code example search failed")
        return {"success": False, "error": str(exc)}

    rerank_applied = False
    model = _get_reranker()
    if model and results:
        results = _rerank_results(model, query, results, content_key="content")
        rerank_applied = True

    return {
        "success": True,
        "query": query,
        "source_filter": source_filter or "",
        "results": results,
        "count": len(results),
        "rerank_applied": rerank_applied,
    }