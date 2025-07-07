from __future__ import annotations

"""Knowledge graph support tools.

These tools are optional and only available when USE_KNOWLEDGE_GRAPH=true and
relevant Python modules are importable.  The implementation mirrors the
original project but trimmed for clarity.
"""

import json
import logging
import os
from typing import Any, Dict, Optional, cast

from fastmcp.server.context import Context  # type: ignore

from ..server import mcp

logger = logging.getLogger(__name__)

###############################################################################
# Lazy import helpers (avoid mandatory deps)
###############################################################################

_KG_MODULES_IMPORTED = False
KnowledgeGraphValidator = None  # type: ignore
DirectNeo4jExtractor = None  # type: ignore
AIScriptAnalyzer = None  # type: ignore
HallucinationReporter = None  # type: ignore


def _import_kg_modules() -> bool:
    global _KG_MODULES_IMPORTED, KnowledgeGraphValidator, DirectNeo4jExtractor, AIScriptAnalyzer, HallucinationReporter  # noqa: E501, PLW0603

    if _KG_MODULES_IMPORTED:  # already attempted
        return KnowledgeGraphValidator is not None

    try:
        from knowledge_graphs.knowledge_graph_validator import KnowledgeGraphValidator  # type: ignore  # noqa: E501
        from knowledge_graphs.parse_repo_into_neo4j import DirectNeo4jExtractor  # type: ignore
        from knowledge_graphs.ai_script_analyzer import AIScriptAnalyzer  # type: ignore
        from knowledge_graphs.hallucination_reporter import HallucinationReporter  # type: ignore
    except ImportError as exc:
        logger.warning("Knowledge graph modules not available: %s", exc)

    _KG_MODULES_IMPORTED = True
    return KnowledgeGraphValidator is not None


###############################################################################
# Environment helpers
###############################################################################

def _neo4j_creds() -> Dict[str, str | None]:
    return {
        "uri": os.getenv("NEO4J_URI"),
        "user": os.getenv("NEO4J_USER"),
        "password": os.getenv("NEO4J_PASSWORD"),
    }


###############################################################################
# MCP tools
###############################################################################


@mcp.tool()
async def parse_github_repository_tool(ctx: Context, repo_url: str) -> str:  # noqa: D401
    """Clone and parse a GitHub repo into Neo4j knowledge graph."""

    if os.getenv("USE_KNOWLEDGE_GRAPH", "false").lower() != "true":
        return json.dumps({"success": False, "error": "Knowledge graph disabled"})

    if not _import_kg_modules():
        return json.dumps({"success": False, "error": "KG modules not installed"})

    creds = _neo4j_creds()
    if not all(creds.values()):
        return json.dumps({"success": False, "error": "Neo4j credentials missing"})

    # At this point mypy/pylance cannot know the class is imported; assert for type checker.
    assert DirectNeo4jExtractor is not None  # for typing
    extractor = DirectNeo4jExtractor(  # type: ignore[arg-type,call-arg]
        cast(str, creds["uri"]), cast(str, creds["user"]), cast(str, creds["password"])
    )
    await extractor.initialize()
    try:
        await extractor.analyze_repository(repo_url)
    finally:
        await extractor.close()

    return json.dumps({"success": True, "repo_url": repo_url})


@mcp.tool()
async def check_ai_script_hallucinations_tool(ctx: Context, script_path: str) -> str:  # noqa: D401
    """Validate Python script against KG for hallucinations."""

    if os.getenv("USE_KNOWLEDGE_GRAPH", "false").lower() != "true":
        return json.dumps({"success": False, "error": "Knowledge graph disabled"})

    if not _import_kg_modules():
        return json.dumps({"success": False, "error": "KG modules not installed"})

    creds = _neo4j_creds()
    if not all(creds.values()):
        return json.dumps({"success": False, "error": "Neo4j credentials missing"})

    # At this point mypy/pylance cannot know the class is imported; assert for type checker.
    assert KnowledgeGraphValidator is not None  # type: ignore[assert-type]
    validator = KnowledgeGraphValidator(  # type: ignore[arg-type,call-arg]
        cast(str, creds["uri"]), cast(str, creds["user"]), cast(str, creds["password"])
    )
    await validator.initialize()
    analyzer = AIScriptAnalyzer()
    try:
        analysis = analyzer.analyze_script(script_path)
        validation = await validator.validate_script(analysis)
        reporter = HallucinationReporter()
        report = reporter.generate_comprehensive_report(validation)
    finally:
        await validator.close()

    return json.dumps({"success": True, **report}, indent=2)


@mcp.tool()
async def query_knowledge_graph_tool(ctx: Context, command: str) -> str:  # noqa: D401
    """Run repo/class/method queries against the KG via helper commands."""

    if os.getenv("USE_KNOWLEDGE_GRAPH", "false").lower() != "true":
        return json.dumps({"success": False, "error": "Knowledge graph disabled"})

    if not _import_kg_modules():
        return json.dumps({"success": False, "error": "KG modules not installed"})

    creds = _neo4j_creds()
    if not all(creds.values()):
        return json.dumps({"success": False, "error": "Neo4j credentials missing"})

    # At this point mypy/pylance cannot know the class is imported; assert for type checker.
    assert DirectNeo4jExtractor is not None
    extractor = DirectNeo4jExtractor(  # type: ignore[arg-type,call-arg]
        cast(str, creds["uri"]), cast(str, creds["user"]), cast(str, creds["password"])
    )
    await extractor.initialize()
    try:
        async with extractor.driver.session() as session:  # type: ignore[attr-defined]
            # very naive implementation: treat command as raw Cypher
            records = []
            try:
                result = await session.run(command)
                async for rec in result:
                    records.append(dict(rec))
            except Exception as exc:  # noqa: BLE001
                return json.dumps({"success": False, "error": str(exc)})
    finally:
        await extractor.close()

    return json.dumps({"success": True, "records": records, "count": len(records)}, indent=2)