from __future__ import annotations

"""Crawling utilities leveraging Crawl4AI v0.6.

This module provides high-level helper functions that will later be wrapped in
FastMCP tools.  They replicate the behaviour of the original crawl_single_page
and smart_crawl_url while using the modern Crawl4AI API (AsyncWebCrawler,
CrawlerRunConfig, adeep_crawl, etc.).
"""

from collections.abc import Sequence
import asyncio
import logging
import os
import re
from typing import Any, List, Dict, Optional
from urllib.parse import urlparse, urldefrag
from xml.etree import ElementTree

import requests  # type: ignore
from crawl4ai import (  # type: ignore
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    CacheMode,
    MemoryAdaptiveDispatcher,  # still useful for arun_many concurrency tuning
)

from ..db import (
    add_code_examples_to_db,
    add_documents_to_db,
    update_source_info,
)
from ..db.vector import EmbeddingError  # noqa: F401
from ..db.pool import get_db_pool

logger = logging.getLogger(__name__)

###############################################################################
# Helper utilities
###############################################################################


def is_sitemap(url: str) -> bool:
    return url.endswith("sitemap.xml") or "sitemap" in urlparse(url).path


def is_txt(url: str) -> bool:
    return url.endswith(".txt")


# Simple markdown chunker (kept from legacy implementation; can be swapped for
# Crawl4AI's built-in chunker in future).

def smart_chunk_markdown(text: str, chunk_size: int = 5000) -> List[str]:
    chunks: List[str] = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = start + chunk_size
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        chunk = text[start:end]
        code_block = chunk.rfind("```")
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block
        elif "\n\n" in chunk:
            last_break = chunk.rfind("\n\n")
            if last_break > chunk_size * 0.3:
                end = start + last_break
        elif ". " in chunk:
            last_period = chunk.rfind(". ")
            if last_period > chunk_size * 0.3:
                end = start + last_period + 1

        chunks.append(text[start:end].strip())
        start = end
    return chunks


def extract_section_info(chunk: str) -> Dict[str, Any]:
    headers = re.findall(r"^(#+)\s+(.+)$", chunk, re.MULTILINE)
    header_str = "; ".join([f"{h[0]} {h[1]}" for h in headers]) if headers else ""
    return {
        "headers": header_str,
        "char_count": len(chunk),
        "word_count": len(chunk.split()),
    }


async def _get_crawler() -> AsyncWebCrawler:
    """Instantiate a crawler with global BrowserConfig according to .env."""

    browser_cfg = BrowserConfig(headless=True, verbose=False)
    crawler = AsyncWebCrawler(config=browser_cfg)
    await crawler.__aenter__()
    return crawler


async def _close_crawler(crawler: AsyncWebCrawler) -> None:
    try:
        await crawler.__aexit__(None, None, None)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Error closing crawler: %s", exc)


###############################################################################
# Public API – these will be wrapped by FastMCP tools later
###############################################################################


async def crawl_single_page(url: str) -> Dict[str, Any]:
    """Crawl a single page and persist its chunks + optional code blocks.

    Returns a dict with metadata similar to the legacy implementation (success,
    counts, etc.) so that higher-level FastMCP tools can JSON-dump it.
    """

    if not url or not isinstance(url, str):
        return {"success": False, "error": "Valid URL is required"}

    crawler = await _get_crawler()
    db_pool = await get_db_pool()

    timing: Dict[str, Any] = {}
    try:
        run_cfg = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
        result = await crawler.arun(url=url, config=run_cfg)
        if not result.success or not result.markdown:
            return {
                "success": False,
                "error": result.error_message or "No content found",
            }

        # Split markdown into chunks
        chunks = smart_chunk_markdown(result.markdown, chunk_size=5000)
        parsed_url = urlparse(url)
        source_id = parsed_url.netloc or parsed_url.path

        # Prepare bulk-insert data
        urls: List[str] = []
        chunk_numbers: List[int] = []
        contents: List[str] = []
        metadatas: List[dict[str, Any]] = []
        total_word_count = 0
        for i, chunk in enumerate(chunks):
            urls.append(url)
            chunk_numbers.append(i)
            contents.append(chunk)
            meta = extract_section_info(chunk)
            meta.update({
                "chunk_index": i,
                "url": url,
                "source": source_id,
            })
            metadatas.append(meta)
            total_word_count += meta["word_count"]

        # Upsert source info
        await update_source_info(db_pool, source_id, result.markdown[:5000], total_word_count)
        await add_documents_to_db(db_pool, urls, chunk_numbers, contents, metadatas, {url: result.markdown})

        return {
            "success": True,
            "url": url,
            "chunks_stored": len(chunks),
            "source_id": source_id,
            "total_word_count": total_word_count,
        }
    except EmbeddingError as e:
        return {"success": False, "error": f"Embedding error: {e}"}
    finally:
        await _close_crawler(crawler)


async def smart_crawl_url(
    url: str,
    *,
    max_depth: int = 3,
    max_concurrent: int = 10,
    chunk_size: int = 5000,
) -> Dict[str, Any]:
    """Intelligently crawl URL (single page, sitemap, txt, or recursive)."""

    if not url or not isinstance(url, str):
        return {"success": False, "error": "Valid URL is required"}

    crawler = await _get_crawler()
    db_pool = await get_db_pool()

    try:
        crawl_type: str
        crawl_results: List[Any]
        run_cfg = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)

        if is_txt(url):
            # Treat as plaintext file containing markdown / urls
            txt_result = await crawler.arun(url=url, config=run_cfg)
            if not txt_result.success or not txt_result.markdown:
                return {"success": False, "error": txt_result.error_message or "No content"}
            crawl_results = [{"url": url, "markdown": txt_result.markdown}]
            crawl_type = "text_file"
        elif is_sitemap(url):
            sitemap_urls = _parse_sitemap_urls(url)
            if not sitemap_urls:
                return {"success": False, "error": "No URLs found in sitemap"}
            crawl_results_any = await crawler.arun_many(urls=sitemap_urls, config=run_cfg)
            crawl_results = [
                {"url": r.url, "markdown": r.markdown}
                for r in crawl_results_any
                if getattr(r, "success", False) and getattr(r, "markdown", None)
            ]
            crawl_type = "sitemap"
        else:
            # Use adeep_crawl for internal recursive crawling
            async_gen = await crawler.adeep_crawl(
                start_url=url,
                strategy="bfs",
                max_depth=max_depth,
                max_pages=1000,
                config=run_cfg,
            )
            crawl_results = []
            async for r in async_gen:
                if getattr(r, "success", False) and getattr(r, "markdown", None):
                    crawl_results.append({"url": r.url, "markdown": r.markdown})
            crawl_type = "webpage"

        if not crawl_results:
            return {"success": False, "error": "No content crawled"}

        # Process and store chunks collectively
        url_to_full_doc: Dict[str, str] = {}
        urls: List[str] = []
        chunk_nums: List[int] = []
        contents: List[str] = []
        metadatas: List[dict[str, Any]] = []
        source_word_counts: Dict[str, int] = {}
        total_chunks = 0

        for doc in crawl_results:
            md = doc["markdown"]
            page_chunks = smart_chunk_markdown(md, chunk_size=chunk_size)
            parsed = urlparse(doc["url"])
            source_id = parsed.netloc or parsed.path
            url_to_full_doc[doc["url"]] = md
            if source_id not in source_word_counts:
                source_word_counts[source_id] = 0
            for i, chunk in enumerate(page_chunks):
                urls.append(doc["url"])
                chunk_nums.append(i)
                contents.append(chunk)
                meta = extract_section_info(chunk)
                meta.update({
                    "chunk_index": i,
                    "url": doc["url"],
                    "source": source_id,
                    "crawl_type": crawl_type,
                })
                metadatas.append(meta)
                source_word_counts[source_id] += meta["word_count"]
                total_chunks += 1

        # Upsert sources first
        for source_id, word_count in source_word_counts.items():
            summary_snip = "".join([contents[i] for i, m in enumerate(metadatas) if m["source"] == source_id][:1])
            await update_source_info(db_pool, source_id, summary_snip[:5000], word_count)

        await add_documents_to_db(
            db_pool,
            urls,
            chunk_nums,
            contents,
            metadatas,
            url_to_full_doc,
            batch_size=20,
        )

        return {
            "success": True,
            "crawl_type": crawl_type,
            "pages_crawled": len(crawl_results),
            "chunks_stored": total_chunks,
        }

    except EmbeddingError as e:
        return {"success": False, "error": f"Embedding error: {e}"}
    finally:
        await _close_crawler(crawler)


###############################################################################
# Internal sitemap parser
###############################################################################

def _parse_sitemap_urls(sitemap_url: str) -> List[str]:
    try:
        resp = requests.get(sitemap_url, timeout=10.0)
        resp.raise_for_status()
        tree = ElementTree.fromstring(resp.content)
        return [loc.text for loc in tree.findall('.//{*}loc') if loc.text]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Error parsing sitemap %s – %s", sitemap_url, exc)
        return []