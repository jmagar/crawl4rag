"""
MCP server for web crawling with Crawl4AI.

This server provides tools to crawl websites using Crawl4AI, automatically detecting
the appropriate crawl method based on URL type (sitemap, txt file, or regular webpage).
Also includes AI hallucination detection and repository parsing tools using Neo4j knowledge graphs.
"""
from mcp.server.fastmcp import FastMCP, Context
from sentence_transformers import CrossEncoder
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, cast
from urllib.parse import urlparse, urldefrag, urljoin
from xml.etree import ElementTree
from dotenv import load_dotenv
import asyncpg
from pathlib import Path
import requests
import asyncio
import json
import os
import re
import concurrent.futures
import sys
import logging
import time

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, MemoryAdaptiveDispatcher

# Add knowledge_graphs folder to path for importing knowledge graph modules
knowledge_graphs_path = Path(__file__).resolve().parent.parent / 'knowledge_graphs'
sys.path.append(str(knowledge_graphs_path))

from utils import (
    get_db_pool, 
    close_db_pool,
    add_documents_to_db, 
    search_documents,
    extract_code_blocks,
    generate_code_example_summary,
    add_code_examples_to_db,
    update_source_info,
    extract_source_summary,
    search_code_examples as search_code_examples_util,
    EmbeddingError
)

# Import knowledge graph modules only when needed
KnowledgeGraphValidator = None
DirectNeo4jExtractor = None
AIScriptAnalyzer = None
HallucinationReporter = None

def _import_knowledge_graph_modules():
    """Import knowledge graph modules only when needed."""
    global KnowledgeGraphValidator, DirectNeo4jExtractor, AIScriptAnalyzer, HallucinationReporter
    try:
        from knowledge_graphs.knowledge_graph_validator import KnowledgeGraphValidator
        from knowledge_graphs.parse_repo_into_neo4j import DirectNeo4jExtractor
        from knowledge_graphs.ai_script_analyzer import AIScriptAnalyzer
        from knowledge_graphs.hallucination_reporter import HallucinationReporter
        return True
    except ImportError as e:
        logger.warning(f"Knowledge graph modules not available: {e}")
        return False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load environment variables from the project root .env file
project_root = Path(__file__).resolve().parent.parent
dotenv_path = project_root / '.env'

# Force override of existing environment variables
load_dotenv(dotenv_path, override=True)

# Configuration validation
def get_neo4j_credentials() -> Dict[str, Optional[str]]:
    """Get Neo4j credentials from environment variables or files."""
    credentials = {}
    
    # Get URI
    credentials['uri'] = os.getenv("NEO4J_URI")
    if not credentials['uri']:
        logger.warning("NEO4J_URI not found in environment variables")
    
    # Get user
    credentials['user'] = os.getenv("NEO4J_USER")
    if not credentials['user']:
        logger.warning("NEO4J_USER not found in environment variables")
    
    # Get password - try file first, then environment
    password_file = os.getenv("NEO4J_PASSWORD_FILE")
    if password_file and os.path.exists(password_file):
        try:
            with open(password_file, 'r') as f:
                credentials['password'] = f.read().strip()
        except Exception as e:
            logger.error(f"Failed to read Neo4j password file {password_file}: {e}")
    else:
        credentials['password'] = os.getenv("NEO4J_PASSWORD")
    
    if not credentials['password']:
        logger.warning("NEO4J_PASSWORD not found in environment variables or file")
    
    return credentials

def validate_neo4j_connection() -> bool:
    """Check if Neo4j environment variables are configured."""
    creds = get_neo4j_credentials()
    return all(creds.values())

def format_neo4j_error(error: Exception) -> str:
    """Format Neo4j connection errors for user-friendly messages."""
    error_str = str(error).lower()
    if "authentication" in error_str or "unauthorized" in error_str:
        return "Neo4j authentication failed. Check NEO4J_USER and NEO4J_PASSWORD."
    elif "connection" in error_str or "refused" in error_str or "timeout" in error_str:
        return "Cannot connect to Neo4j. Check NEO4J_URI and ensure Neo4j is running."
    elif "database" in error_str:
        return "Neo4j database error. Check if the database exists and is accessible."
    else:
        return f"Neo4j error: {str(error)}"

def validate_script_path(script_path: str) -> Dict[str, Any]:
    """Validate script path and return error info if invalid."""
    if not script_path or not isinstance(script_path, str):
        return {"valid": False, "error": "Script path is required"}
    
    if not os.path.exists(script_path):
        return {"valid": False, "error": f"Script not found: {script_path}"}
    
    if not script_path.endswith('.py'):
        return {"valid": False, "error": "Only Python (.py) files are supported"}
    
    try:
        # Check if file is readable
        with open(script_path, 'r', encoding='utf-8') as f:
            f.read(1)  # Read first character to test
        return {"valid": True}
    except Exception as e:
        return {"valid": False, "error": f"Cannot read script file: {str(e)}"}

def validate_github_url(repo_url: str) -> Dict[str, Any]:
    """Validate GitHub repository URL."""
    if not repo_url or not isinstance(repo_url, str):
        return {"valid": False, "error": "Repository URL is required"}
    
    repo_url = repo_url.strip()
    
    # Basic GitHub URL validation
    if not ("github.com" in repo_url.lower() or repo_url.endswith(".git")):
        return {"valid": False, "error": "Please provide a valid GitHub repository URL"}
    
    # Check URL format
    if not (repo_url.startswith("https://") or repo_url.startswith("git@")):
        return {"valid": False, "error": "Repository URL must start with https:// or git@"}
    
    return {"valid": True, "repo_name": repo_url.split('/')[-1].replace('.git', '')}

# Create a dataclass for our application context
@dataclass
class Crawl4AIContext:
    """Context for the Crawl4AI MCP server."""
    crawler: AsyncWebCrawler
    db_pool: asyncpg.Pool
    reranking_model: Optional[CrossEncoder] = None
    knowledge_validator: Optional[Any] = None  # KnowledgeGraphValidator when available
    repo_extractor: Optional[Any] = None       # DirectNeo4jExtractor when available

@asynccontextmanager
async def crawl4ai_lifespan(server: FastMCP) -> AsyncIterator[Crawl4AIContext]:
    """
    Manages the Crawl4AI client lifecycle with proper error handling and resource management.
    
    Args:
        server: The FastMCP server instance
        
    Yields:
        Crawl4AIContext: The context containing the Crawl4AI crawler and database pool
    """
    logger.info("Starting Crawl4AI MCP server initialization...")
    
    # Create browser configuration
    browser_config = BrowserConfig(
        headless=True,
        verbose=False
    )
    
    # Initialize components
    crawler = None
    db_pool = None
    reranking_model = None
    knowledge_validator = None
    repo_extractor = None
    
    try:
        # Initialize the crawler
        logger.info("Initializing web crawler...")
        crawler = AsyncWebCrawler(config=browser_config)
        await crawler.__aenter__()
        logger.info("✓ Web crawler initialized")
        
        # Initialize DB Pool
        logger.info("Initializing database connection pool...")
        db_pool = await get_db_pool()
        logger.info("✓ Database connection pool initialized")
        
        # Initialize cross-encoder model for reranking if enabled
        if os.getenv("USE_RERANKING", "false") == "true":
            try:
                logger.info("Loading reranking model...")
                reranking_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                logger.info("✓ Reranking model loaded")
            except Exception as e:
                logger.error(f"Failed to load reranking model: {e}")
                reranking_model = None
        
        # Initialize Neo4j components if configured and enabled
        knowledge_graph_enabled = os.getenv("USE_KNOWLEDGE_GRAPH", "false") == "true"
        
        if knowledge_graph_enabled:
            logger.info("Knowledge graph functionality enabled")
            
            # Try to import knowledge graph modules
            if _import_knowledge_graph_modules():
                creds = get_neo4j_credentials()
                
                if all(creds.values()):
                    # Narrow Optional[str] -> str after runtime validation
                    uri: str = cast(str, creds['uri'])
                    user: str = cast(str, creds['user'])
                    password: str = cast(str, creds['password'])

                    try:
                        logger.info("Initializing knowledge graph components...")

                        if KnowledgeGraphValidator:
                            knowledge_validator = KnowledgeGraphValidator(uri, user, password)
                            await knowledge_validator.initialize()
                            logger.info("✓ Knowledge graph validator initialized")

                        if DirectNeo4jExtractor:
                            repo_extractor = DirectNeo4jExtractor(uri, user, password)
                            await repo_extractor.initialize()
                            logger.info("✓ Repository extractor initialized")

                    except Exception as e:
                        logger.error("Failed to initialize Neo4j components: %s", format_neo4j_error(e))
                        knowledge_validator = None
                        repo_extractor = None
                else:
                    logger.warning("Neo4j credentials not fully configured - knowledge graph tools will be unavailable")
            else:
                logger.warning("Knowledge graph modules not available - knowledge graph tools will be unavailable")
        else:
            logger.info("Knowledge graph functionality disabled - set USE_KNOWLEDGE_GRAPH=true to enable")
        
        logger.info("Server initialization completed successfully")
        
        yield Crawl4AIContext(
            crawler=crawler,
            db_pool=db_pool,
            reranking_model=reranking_model,
            knowledge_validator=knowledge_validator,
            repo_extractor=repo_extractor
        )
        
    except Exception as e:
        logger.error(f"Failed to initialize server components: {e}")
        raise
    finally:
        # Cleanup resources gracefully
        logger.info("Starting cleanup...")
        
        # Close crawler
        if crawler:
            try:
                await crawler.__aexit__(None, None, None)
                logger.info("✓ Crawler closed")
            except Exception as e:
                logger.error(f"Error closing crawler: {e}")
        
        # Close database pool
        if db_pool:
            try:
                await close_db_pool()
                logger.info("✓ Database pool closed")
            except Exception as e:
                logger.error(f"Error closing database pool: {e}")
        
        # Close knowledge graph components
        if knowledge_validator:
            try:
                await knowledge_validator.close()
                logger.info("✓ Knowledge graph validator closed")
            except Exception as e:
                logger.error(f"Error closing knowledge validator: {e}")
                
        if repo_extractor:
            try:
                await repo_extractor.close()
                logger.info("✓ Repository extractor closed")
            except Exception as e:
                logger.error(f"Error closing repository extractor: {e}")
        
        logger.info("Cleanup completed")

# Initialize FastMCP server with validation
try:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8051"))
    
    # Validate port range
    if not (1 <= port <= 65535):
        raise ValueError(f"Invalid port number: {port}. Must be between 1 and 65535.")
    
    logger.info(f"Initializing FastMCP server on {host}:{port}")
    
    mcp = FastMCP(
        "mcp-crawl4ai-rag",
        description="MCP server for RAG and web crawling with Crawl4AI",
        lifespan=crawl4ai_lifespan,
        host=host,
        port=port
    )
    
except Exception as e:
    logger.error(f"Failed to initialize FastMCP server: {e}")
    raise

def rerank_results(model: CrossEncoder, query: str, results: List[Dict[str, Any]], content_key: str = "content") -> List[Dict[str, Any]]:
    """
    Rerank search results using a cross-encoder model with caching.
    
    Args:
        model: The cross-encoder model to use for reranking
        query: The search query
        results: List of search results
        content_key: The key in each result dict that contains the text content
        
    Returns:
        Reranked list of results
    """
    if not model or not results:
        return results
    
    try:
        # Extract content from results
        texts = [result.get(content_key, "") for result in results]
        
        # Create pairs of [query, document] for the cross-encoder
        pairs = [[query, text] for text in texts]
        
        # Get relevance scores from the cross-encoder
        scores = model.predict(pairs)
        
        # Add scores to results and sort by score (descending)
        for i, result in enumerate(results):
            result["rerank_score"] = float(scores[i])
        
        # Sort by rerank score
        reranked = sorted(results, key=lambda x: x.get("rerank_score", 0), reverse=True)
        
        return reranked
    except Exception as e:
        logger.error(f"Error during reranking: {e}")
        return results

def is_sitemap(url: str) -> bool:
    """
    Check if a URL is a sitemap.
    
    Args:
        url: URL to check
        
    Returns:
        True if the URL is a sitemap, False otherwise
    """
    return url.endswith('sitemap.xml') or 'sitemap' in urlparse(url).path

def is_txt(url: str) -> bool:
    """
    Check if a URL is a text file.
    
    Args:
        url: URL to check
        
    Returns:
        True if the URL is a text file, False otherwise
    """
    return url.endswith('.txt')

def parse_sitemap(sitemap_url: str) -> List[str]:
    """
    Parse a sitemap and extract URLs.
    
    Args:
        sitemap_url: URL of the sitemap
        
    Returns:
        List of URLs found in the sitemap
    """
    resp = requests.get(sitemap_url)
    urls: List[str] = []

    if resp.status_code == 200:
        try:
            tree = ElementTree.fromstring(resp.content)
            urls.extend([loc.text for loc in tree.findall('.//{*}loc') if loc.text])
        except Exception as e:
            logger.error(f"Error parsing sitemap XML: {e}")

    return urls

def smart_chunk_markdown(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calculate end position
        end = start + chunk_size

        # If we're at the end of the text, just take what's left
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Try to find a code block boundary first (```)
        chunk = text[start:end]
        code_block = chunk.rfind('```')
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block

        # If no code block, try to break at a paragraph
        elif '\n\n' in chunk:
            # Find the last paragraph break
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_break

        # If no paragraph break, try to break at a sentence
        elif '. ' in chunk:
            # Find the last sentence break
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_period + 1

        # Extract chunk and clean it up
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position for next chunk
        start = end

    return chunks

def extract_section_info(chunk: str) -> Dict[str, Any]:
    """
    Extracts headers and stats from a chunk.
    
    Args:
        chunk: Markdown chunk
        
    Returns:
        Dictionary with headers and stats
    """
    headers = re.findall(r'^(#+)\s+(.+)$', chunk, re.MULTILINE)
    header_str = '; '.join([f'{h[0]} {h[1]}' for h in headers]) if headers else ''

    return {
        "headers": header_str,
        "char_count": len(chunk),
        "word_count": len(chunk.split())
    }



@mcp.tool()
async def crawl_single_page(ctx: Context, url: str) -> str:
    """
    Crawl a single web page and store its content in the database.
    
    This tool is ideal for quickly retrieving content from a specific URL without following links.
    The content is stored in the database for later retrieval and querying.
    
    Args:
        ctx: The MCP server provided context
        url: URL of the web page to crawl
    
    Returns:
        Summary of the crawling operation and storage in the database
    """
    if not url or not isinstance(url, str):
        return json.dumps({
            "success": False,
            "error": "Valid URL is required"
        })
    
    import time
    start_time = time.time()
    timing_data = {}
    
    try:
        # Get the crawler from the context
        crawler = ctx.request_context.lifespan_context.crawler
        db_pool = ctx.request_context.lifespan_context.db_pool
        
        logger.info(f"Starting to crawl single page: {url}")
        
        # Configure the crawl
        run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
        
        # Crawl the page - Track crawling time
        crawl_start = time.time()
        result = await crawler.arun(url=url, config=run_config)
        crawl_end = time.time()
        timing_data["page_crawl_seconds"] = round(crawl_end - crawl_start, 2)
        
        if not result.success:
            error_msg = f"Failed to crawl URL: {result.error_message or 'Unknown error'}"
            logger.error(error_msg)
            return json.dumps({
                "success": False,
                "error": error_msg
            })
        
        if not result.markdown:
            logger.warning(f"No content found for URL: {url}")
            return json.dumps({
                "success": False,
                "error": "No content found on the page"
            })
        
        # Extract source_id
        parsed_url = urlparse(url)
        source_id = parsed_url.netloc or parsed_url.path
        
        # Chunk the content - Track chunking time
        chunk_start = time.time()
        chunks = smart_chunk_markdown(result.markdown)
        chunk_end = time.time()
        timing_data["content_chunking_seconds"] = round(chunk_end - chunk_start, 2)
        logger.info(f"Split content into {len(chunks)} chunks")
        
        # Prepare data for database
        urls = []
        chunk_numbers = []
        contents = []
        metadatas = []
        total_word_count = 0
        
        for i, chunk in enumerate(chunks):
            urls.append(url)
            chunk_numbers.append(i)
            contents.append(chunk)
            
            # Extract metadata
            meta = extract_section_info(chunk)
            meta["chunk_index"] = i
            meta["url"] = url
            meta["source"] = source_id
            current_task = asyncio.current_task()
            meta["crawl_timestamp"] = current_task.get_name() if current_task else "unknown"
            metadatas.append(meta)
            
            # Accumulate word count
            total_word_count += meta.get("word_count", 0)
        
        # Create url_to_full_document mapping
        url_to_full_document = {url: result.markdown}
        
        # Update source information FIRST (before inserting documents)
        source_start = time.time()
        logger.info("Generating source summary...")
        source_summary = await extract_source_summary(source_id, result.markdown[:5000])  # Use first 5000 chars for summary
        await update_source_info(db_pool, source_id, source_summary, total_word_count)
        source_end = time.time()
        timing_data["source_processing_seconds"] = round(source_end - source_start, 2)
        
        # Add documentation chunks to DB (AFTER source exists) - Track embedding + storage time
        db_start = time.time()
        logger.info("Storing document chunks in database...")
        await add_documents_to_db(db_pool, urls, chunk_numbers, contents, metadatas, url_to_full_document)
        db_end = time.time()
        timing_data["embedding_and_storage_seconds"] = round(db_end - db_start, 2)
        
        # Extract and process code examples only if enabled
        code_examples_count = 0
        extract_code_examples = os.getenv("USE_AGENTIC_RAG", "false") == "true"
        
        if extract_code_examples:
            code_start = time.time()
            logger.info("Extracting code examples...")
            code_blocks = extract_code_blocks(result.markdown)
            
            if code_blocks:
                logger.info(f"Found {len(code_blocks)} code blocks, processing summaries...")
                
                # Process code examples with memory-aware batching
                batch_size = int(os.getenv("CODE_PROCESSING_BATCH_SIZE", "5"))
                code_urls = []
                code_chunk_numbers = []
                code_examples = []
                code_summaries = []
                code_metadatas = []
                
                # Process code examples in smaller batches to avoid memory issues
                for i in range(0, len(code_blocks), batch_size):
                    batch_end = min(i + batch_size, len(code_blocks))
                    batch_blocks = code_blocks[i:batch_end]
                    
                    # Generate summaries asynchronously for this batch
                    summary_tasks = []
                    for block in batch_blocks:
                        task = generate_code_example_summary(
                            block['code'], 
                            block['context_before'], 
                            block['context_after']
                        )
                        summary_tasks.append(task)
                    
                    # Wait for all summaries in this batch
                    batch_summaries = await asyncio.gather(*summary_tasks, return_exceptions=True)
                    
                    # Process results for this batch
                    for j, (block, summary) in enumerate(zip(batch_blocks, batch_summaries, strict=True)):
                        global_index = i + j
                        
                        # Handle exceptions in summary generation
                        if isinstance(summary, Exception):
                            logger.warning(f"Failed to generate summary for code block {global_index}: {summary}")
                            summary = "Code example for demonstration purposes."
                        
                        code_urls.append(url)
                        code_chunk_numbers.append(global_index)
                        code_examples.append(block['code'])
                        code_summaries.append(summary)
                        
                        # Create metadata for code example
                        code_meta = {
                            "chunk_index": global_index,
                            "url": url,
                            "source": source_id,
                            "language": block.get('language', ''),
                            "char_count": len(block['code']),
                            "word_count": len(block['code'].split()),
                            "crawl_timestamp": meta.get("crawl_timestamp", "unknown")
                        }
                        code_metadatas.append(code_meta)
                
                # Add code examples to DB
                if code_examples:
                    logger.info(f"Storing {len(code_examples)} code examples in database...")
                    await add_code_examples_to_db(
                        db_pool, 
                        code_urls, 
                        code_chunk_numbers, 
                        code_examples, 
                        code_summaries, 
                        code_metadatas
                    )
                    code_examples_count = len(code_examples)
            
            code_end = time.time()
            timing_data["code_examples_seconds"] = round(code_end - code_start, 2)
        
        # Calculate total time
        end_time = time.time()
        timing_data["total_seconds"] = round(end_time - start_time, 2)
        
        logger.info(f"Successfully processed page: {url} in {timing_data['total_seconds']} seconds")
        
        return json.dumps({
            "success": True,
            "url": url,
            "chunks_stored": len(chunks),
            "code_examples_stored": code_examples_count,
            "content_length": len(result.markdown),
            "total_word_count": total_word_count,
            "source_id": source_id,
            "links_count": {
                "internal": len(result.links.get("internal", [])) if result.links else 0,
                "external": len(result.links.get("external", [])) if result.links else 0
            },
            "performance": timing_data
        }, indent=2)
        
    except EmbeddingError as e:
        error_msg = f"Embedding generation failed: {str(e)}"
        logger.error(error_msg)
        return json.dumps({
            "success": False,
            "error": error_msg,
            "type": "embedding_error"
        })
    except Exception as e:
        error_msg = f"Error crawling page: {str(e)}"
        logger.error(error_msg)
        return json.dumps({
            "success": False,
            "error": error_msg,
            "type": "crawl_error"
        })

@mcp.tool()
async def smart_crawl_url(ctx: Context, url: str, max_depth: int = 3, max_concurrent: int = 10, chunk_size: int = 5000) -> str:
    """
    Intelligently crawl a URL based on its type and store content in Supabase.
    
    This tool automatically detects the URL type and applies the appropriate crawling method:
    - For sitemaps: Extracts and crawls all URLs in parallel
    - For text files (llms.txt): Directly retrieves the content
    - For regular webpages: Recursively crawls internal links up to the specified depth
    
    All crawled content is chunked and stored in Supabase for later retrieval and querying.
    
    Args:
        ctx: The MCP server provided context
        url: URL to crawl (can be a regular webpage, sitemap.xml, or .txt file)
        max_depth: Maximum recursion depth for regular URLs (default: 3)
        max_concurrent: Maximum number of concurrent browser sessions (default: 10)
        chunk_size: Maximum size of each content chunk in characters (default: 1000)
    
    Returns:
        JSON string with crawl summary and storage information
    """
    # Input validation
    if not url or not isinstance(url, str):
        return json.dumps({
            "success": False,
            "error": "Valid URL is required"
        }, indent=2)
    
    start_time = time.time()
    timing_data = {}
    
    try:
        # Get the crawler from the context
        crawler = ctx.request_context.lifespan_context.crawler
        db_pool = ctx.request_context.lifespan_context.db_pool
        
        if not crawler:
            return json.dumps({
                "success": False,
                "url": url,
                "error": "Crawler not available in context"
            }, indent=2)
        
        if not db_pool:
            return json.dumps({
                "success": False,
                "url": url,
                "error": "Database pool not available in context"
            }, indent=2)
        
        logger.info(f"Starting smart crawl for URL: {url}")
        
        # Determine the crawl strategy - Track strategy detection time
        strategy_start = time.time()
        crawl_results = []
        crawl_type = None
        
        if is_txt(url):
            # For text files, use simple crawl
            crawl_results = await crawl_markdown_file(crawler, url)
            crawl_type = "text_file"
        elif is_sitemap(url):
            # For sitemaps, extract URLs and crawl in parallel
            sitemap_urls = parse_sitemap(url)
            if not sitemap_urls:
                return json.dumps({
                    "success": False,
                    "url": url,
                    "error": "No URLs found in sitemap"
                }, indent=2)
            crawl_results = await crawl_batch(crawler, sitemap_urls, max_concurrent=max_concurrent)
            crawl_type = "sitemap"
        else:
            # For regular URLs, use recursive crawl
            crawl_results = await crawl_recursive_internal_links(crawler, [url], max_depth=max_depth, max_concurrent=max_concurrent)
            crawl_type = "webpage"
        
        strategy_end = time.time()
        timing_data["crawl_strategy_and_pages_seconds"] = round(strategy_end - strategy_start, 2)
        timing_data["pages_crawled"] = len(crawl_results)
        
        if not crawl_results:
            return json.dumps({
                "success": False,
                "url": url,
                "error": "No content found"
            }, indent=2)
        
        # Process results and store in Supabase - Track content processing time
        processing_start = time.time()
        urls = []
        chunk_numbers = []
        contents = []
        metadatas = []
        chunk_count = 0
        
        # Track sources and their content
        source_content_map = {}
        source_word_counts = {}
        
        # Process documentation chunks
        for doc in crawl_results:
            source_url = doc['url']
            md = doc['markdown']
            chunks = smart_chunk_markdown(md, chunk_size=chunk_size)
            
            # Extract source_id
            parsed_url = urlparse(source_url)
            source_id = parsed_url.netloc or parsed_url.path
            
            # Store content for source summary generation
            if source_id not in source_content_map:
                source_content_map[source_id] = md[:5000]  # Store first 5000 chars
                source_word_counts[source_id] = 0
            
            for i, chunk in enumerate(chunks):
                urls.append(source_url)
                chunk_numbers.append(i)
                contents.append(chunk)
                
                # Extract metadata
                meta = extract_section_info(chunk)
                meta["chunk_index"] = i
                meta["url"] = source_url
                meta["source"] = source_id
                meta["crawl_type"] = crawl_type
                current_task = asyncio.current_task()
                meta["crawl_timestamp"] = current_task.get_name() if current_task else "unknown"
                metadatas.append(meta)
                
                # Accumulate word count
                source_word_counts[source_id] += meta.get("word_count", 0)
                
                chunk_count += 1
        
        processing_end = time.time()
        timing_data["content_processing_seconds"] = round(processing_end - processing_start, 2)
        timing_data["total_chunks"] = chunk_count
        
        # Create url_to_full_document mapping
        url_to_full_document = {}
        for doc in crawl_results:
            url_to_full_document[doc['url']] = doc['markdown']
        
        # Update source information for each unique source FIRST (before inserting documents) - Track source processing
        source_start = time.time()
        source_summary_args = [
            (sid, content) for sid, content in source_content_map.items()
        ]

        # `extract_source_summary` is async, so run concurrently using asyncio.gather
        source_summaries = await asyncio.gather(
            *[extract_source_summary(sid, content) for sid, content in source_summary_args],
            return_exceptions=True,
        )

        # Replace exceptions with default summaries
        cleaned_summaries: List[str] = []
        for (sid, _), summary in zip(source_summary_args, source_summaries, strict=True):
            if isinstance(summary, Exception):
                logger.warning("Failed to generate summary for %s: %s", sid, summary)
                cleaned_summary = f"Content from {sid}"
            else:
                cleaned_summary = cast(str, summary)
            cleaned_summaries.append(cleaned_summary)

        source_summaries = cleaned_summaries
        
        # Actually create/update sources in database FIRST (before inserting documents)
        logger.info("Creating/updating sources in database...")
        for (source_id, _), summary in zip(source_summary_args, source_summaries, strict=True):
            word_count = source_word_counts.get(source_id, 0)
            await update_source_info(db_pool, source_id, summary, word_count)
        
        source_end = time.time()
        timing_data["source_processing_seconds"] = round(source_end - source_start, 2)
        
        # Add documentation chunks to DB (AFTER sources exist) - Track embedding + storage time
        db_start = time.time()
        batch_size = 20
        await add_documents_to_db(db_pool, urls, chunk_numbers, contents, metadatas, url_to_full_document, batch_size=batch_size)
        db_end = time.time()
        timing_data["embedding_and_storage_seconds"] = round(db_end - db_start, 2)
        
        # Extract and process code examples from all documents only if enabled - Track code examples time
        extract_code_examples_enabled = os.getenv("USE_AGENTIC_RAG", "false") == "true"
        code_examples = []
        if extract_code_examples_enabled:
            code_start = time.time()
            all_code_blocks = []
            code_urls = []
            code_chunk_numbers = []
            code_examples = []
            code_summaries = []
            code_metadatas = []
            
            # Extract code blocks from all documents
            for doc in crawl_results:
                source_url = doc['url']
                md = doc['markdown']
                code_blocks = extract_code_blocks(md)
                
                if code_blocks:
                    # Process code examples asynchronously
                    summary_tasks = []
                    for block in code_blocks:
                        task = generate_code_example_summary(
                            block['code'], 
                            block['context_before'], 
                            block['context_after']
                        )
                        summary_tasks.append(task)
                    
                    # Generate summaries in parallel
                    summaries = await asyncio.gather(*summary_tasks, return_exceptions=True)
                    
                    # Prepare code example data
                    parsed_url = urlparse(source_url)
                    source_id = parsed_url.netloc or parsed_url.path
                    
                    if len(code_blocks) != len(summaries):
                        raise ValueError("Mismatch between code blocks and summaries count")
                    for i, (block, summary) in enumerate(zip(code_blocks, summaries, strict=True)):
                        # Handle exceptions in summary generation
                        if isinstance(summary, Exception):
                            logger.warning(f"Failed to generate summary for code block {i}: {summary}")
                            summary = "Code example for demonstration purposes."
                        code_urls.append(source_url)
                        code_chunk_numbers.append(len(code_examples))  # Use global code example index
                        code_examples.append(block['code'])
                        code_summaries.append(summary)
                        
                        # Create metadata for code example
                        code_meta = {
                            "chunk_index": len(code_examples) - 1,
                            "url": source_url,
                            "source": source_id,
                            "char_count": len(block['code']),
                            "word_count": len(block['code'].split())
                        }
                        code_metadatas.append(code_meta)
            
            # Add all code examples to DB
            if code_examples:
                await add_code_examples_to_db(
                    db_pool, 
                    code_urls, 
                    code_chunk_numbers, 
                    code_examples, 
                    code_summaries, 
                    code_metadatas,
                    batch_size=batch_size
                )
            
            code_end = time.time()
            timing_data["code_examples_seconds"] = round(code_end - code_start, 2)
        
        # Calculate total time
        end_time = time.time()
        timing_data["total_seconds"] = round(end_time - start_time, 2)
        
        # Calculate performance metrics
        if timing_data["total_seconds"] > 0:
            timing_data["pages_per_second"] = round(len(crawl_results) / timing_data["total_seconds"], 2)
            timing_data["chunks_per_second"] = round(chunk_count / timing_data["total_seconds"], 2)
        
        logger.info(f"Smart crawl completed in {timing_data['total_seconds']} seconds - "
                   f"{len(crawl_results)} pages, {chunk_count} chunks")
        
        return json.dumps({
            "success": True,
            "url": url,
            "crawl_type": crawl_type,
            "pages_crawled": len(crawl_results),
            "chunks_stored": chunk_count,
            "code_examples_stored": len(code_examples),
            "sources_updated": len(source_content_map),
            "urls_crawled": [doc['url'] for doc in crawl_results][:5] + (["..."] if len(crawl_results) > 5 else []),
            "performance": timing_data
        }, indent=2)
    except EmbeddingError as e:
        error_msg = f"Embedding generation failed: {str(e)}"
        logger.error(error_msg)
        return json.dumps({
            "success": False,
            "url": url,
            "error": error_msg,
            "type": "embedding_error"
        }, indent=2)
    except Exception as e:
        error_msg = f"Error during smart crawl: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return json.dumps({
            "success": False,
            "url": url,
            "error": error_msg,
            "type": "crawl_error"
        }, indent=2)

@mcp.tool()
async def get_available_sources(ctx: Context) -> str:
    """
    Get all available sources from the sources table.
    
    This tool returns a list of all unique sources (domains) that have been crawled and stored
    in the database, along with their summaries and statistics. This is useful for discovering 
    what content is available for querying.

    Always use this tool before calling the RAG query or code example query tool
    with a specific source filter!
    
    Args:
        ctx: The MCP server provided context
    
    Returns:
        JSON string with the list of available sources and their details
    """
    try:
        # Get the DB pool from the context
        db_pool = ctx.request_context.lifespan_context.db_pool
        
        # Query the sources table directly
        async with db_pool.acquire() as connection:
            result = await connection.fetch("SELECT * FROM sources ORDER BY source_id")

        # Format the sources with their details
        sources = []
        if result:
            for source in result:
                sources.append({
                    "source_id": source.get("source_id"),
                    "summary": source.get("summary"),
                    "total_words": source.get("total_word_count") or 0,
                    "created_at": str(source.get("created_at")),
                    "updated_at": str(source.get("updated_at"))
                })
        
        return json.dumps({
            "success": True,
            "sources": sources,
            "count": len(sources)
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, indent=2)

@mcp.tool()
async def perform_rag_query(ctx: Context, query: str, source: Optional[str] = None, match_count: int = 5) -> str:
    """
    Perform a RAG (Retrieval Augmented Generation) query on the stored content.
    
    This tool searches the vector database for content relevant to the query and returns
    the matching documents. Optionally filter by source domain.
    Get the source by using the get_available_sources tool before calling this search!
    
    Args:
        ctx: The MCP server provided context
        query: The search query
        source: Optional source domain to filter results (e.g., 'example.com')
        match_count: Maximum number of results to return (default: 5)
    
    Returns:
        JSON string with the search results
    """
    # Input validation
    if not query or not isinstance(query, str) or not query.strip():
        return json.dumps({
            "success": False,
            "error": "Query parameter is required and must be a non-empty string"
        })
    
    # Validate match_count
    if not isinstance(match_count, int) or match_count < 1 or match_count > 50:
        return json.dumps({
            "success": False,
            "error": "match_count must be an integer between 1 and 50"
        })
    
    try:
        # Get the DB pool from the context
        db_pool = ctx.request_context.lifespan_context.db_pool
        
        logger.info(f"Starting RAG query: '{query}' with source filter: {source}")
        
        # Check if hybrid search is enabled
        use_hybrid_search = os.getenv("USE_HYBRID_SEARCH", "false") == "true"
        
        # Prepare filter if source is provided and not empty
        if source:
            source_filter = cast(str, source).strip()
            if not source_filter:
                source_filter = None
        else:
            source_filter = None
        
        if use_hybrid_search:
            logger.debug("Using hybrid search mode")
            # Hybrid search: combine vector and keyword search
            
            # 1. Get vector search results (get more to account for filtering)
            try:
                vector_results = await search_documents(
                    pool=db_pool,
                    query=query,
                    match_count=match_count * 2,  # Get double to have room for filtering
                    source_filter=source_filter
                )
            except EmbeddingError as e:
                logger.error(f"Vector search failed: {e}")
                return json.dumps({
                    "success": False,
                    "error": f"Vector search failed: {str(e)}",
                    "type": "embedding_error"
                })
            
            # 2. Get keyword search results using ILIKE
            try:
                async with db_pool.acquire() as connection:
                    keyword_query_str = """
                        SELECT id, url, chunk_number, content, metadata, source_id, 0.5 as similarity
                        FROM crawled_pages 
                        WHERE content ILIKE $1
                    """
                    params: List[Any] = [f'%{query}%']
                    
                    if source_filter:
                        keyword_query_str += " AND source_id = $2"
                        params.append(cast(str, source_filter))
                        keyword_query_str += " LIMIT $3"
                        params.append(match_count * 2)
                    else:
                        keyword_query_str += " LIMIT $2"
                        params.append(match_count * 2)

                    keyword_response = await connection.fetch(keyword_query_str, *params)
                
                keyword_results = [dict(row) for row in keyword_response] if keyword_response else []
            except Exception as e:
                logger.error(f"Keyword search failed: {e}")
                # Continue with vector results only
                keyword_results = []
            
            # 3. Combine results with preference for items appearing in both
            seen_ids = set()
            combined_results = []
            
            # First, add items that appear in both searches (these are the best matches)
            vector_ids = {r.get('id') for r in vector_results if r.get('id')}
            for kr in keyword_results:
                if kr['id'] in vector_ids and kr['id'] not in seen_ids:
                    # Find the vector result to get similarity score
                    for vr in vector_results:
                        if vr.get('id') == kr['id']:
                            # Boost similarity score for items in both results
                            vr['similarity'] = min(1.0, vr.get('similarity', 0) * 1.2)
                            combined_results.append(vr)
                            seen_ids.add(kr['id'])
                            break
            
            # Then add remaining vector results (semantic matches without exact keyword)
            for vr in vector_results:
                if vr.get('id') and vr['id'] not in seen_ids and len(combined_results) < match_count:
                    combined_results.append(vr)
                    seen_ids.add(vr['id'])
            
            # Finally, add pure keyword matches if we still need more results
            for kr in keyword_results:
                if kr['id'] not in seen_ids and len(combined_results) < match_count:
                    combined_results.append(kr)
                    seen_ids.add(kr['id'])
            
            # Use combined results
            results = combined_results[:match_count]
            
        else:
            logger.debug("Using vector search only")
            # Standard vector search only
            try:
                results = await search_documents(
                    pool=db_pool,
                    query=query,
                    match_count=match_count,
                    source_filter=source_filter
                )
            except EmbeddingError as e:
                logger.error(f"Vector search failed: {e}")
                return json.dumps({
                    "success": False,
                    "error": f"Vector search failed: {str(e)}",
                    "type": "embedding_error"
                })
        
        # Apply reranking if enabled
        use_reranking = os.getenv("USE_RERANKING", "false") == "true"
        reranking_applied = False
        
        if use_reranking and ctx.request_context.lifespan_context.reranking_model and results:
            try:
                logger.debug("Applying reranking to results")
                results = rerank_results(ctx.request_context.lifespan_context.reranking_model, query, results, content_key="content")
                reranking_applied = True
            except Exception as e:
                logger.warning(f"Reranking failed, continuing with unranked results: {e}")
        
        # Format the results
        formatted_results = []
        for result in results:
            formatted_result = {
                "url": result.get("url"),
                "content": result.get("content"),
                "metadata": result.get("metadata"),
                "source_id": result.get("source_id"),
                "similarity": result.get("similarity")
            }
            # Include rerank score if available
            if "rerank_score" in result:
                formatted_result["rerank_score"] = result["rerank_score"]
            formatted_results.append(formatted_result)
        
        logger.info(f"RAG query completed. Found {len(formatted_results)} results")
        
        return json.dumps({
            "success": True,
            "query": query,
            "source_filter": source_filter or "",
            "search_mode": "hybrid" if use_hybrid_search else "vector",
            "reranking_applied": reranking_applied,
            "results": formatted_results,
            "count": len(formatted_results)
        }, indent=2)
        
    except Exception as e:
        logger.error(f"Error in RAG query: {e}")
        return json.dumps({
            "success": False,
            "query": query,
            "error": str(e),
            "type": "query_error"
        })

@mcp.tool()
async def search_code_examples(ctx: Context, query: str, source_id: Optional[str] = None, match_count: int = 5) -> str:
    """
    Search for code examples relevant to the query.
    
    This tool searches the vector database for code examples relevant to the query and returns
    the matching examples with their summaries. Optionally filter by source_id.
    Get the source_id by using the get_available_sources tool before calling this search!

    Use the get_available_sources tool first to see what sources are available for filtering.
    
    Args:
        ctx: The MCP server provided context
        query: The search query
        source_id: Optional source ID to filter results (e.g., 'example.com')
        match_count: Maximum number of results to return (default: 5)
    
    Returns:
        JSON string with the search results
    """
    # Check if code example extraction is enabled
    extract_code_examples_enabled = os.getenv("USE_AGENTIC_RAG", "false") == "true"
    if not extract_code_examples_enabled:
        return json.dumps({
            "success": False,
            "error": "Code example extraction is disabled. Set USE_AGENTIC_RAG=true to enable this feature."
        })
    
    # Input validation
    if not query or not isinstance(query, str) or not query.strip():
        return json.dumps({
            "success": False,
            "error": "Query parameter is required and must be a non-empty string"
        })
    
    # Validate match_count
    if not isinstance(match_count, int) or match_count < 1 or match_count > 50:
        return json.dumps({
            "success": False,
            "error": "match_count must be an integer between 1 and 50"
        })
    
    try:
        # Get the DB pool from the context
        db_pool = ctx.request_context.lifespan_context.db_pool
        
        logger.info(f"Starting code examples search: '{query}' with source filter: {source_id}")
        
        # Check if hybrid search is enabled
        use_hybrid_search = os.getenv("USE_HYBRID_SEARCH", "false") == "true"
        
        # Prepare filter if source is provided and not empty
        if source_id:
            source_filter = cast(str, source_id).strip()
            if not source_filter:
                source_filter = None
        else:
            source_filter = None
        
        if use_hybrid_search:
            logger.debug("Using hybrid search mode for code examples")
            # Hybrid search: combine vector and keyword search
            
            # 1. Get vector search results (get more to account for filtering)
            try:
                vector_results = await search_code_examples_util(
                    pool=db_pool,
                    query=query,
                    match_count=match_count * 2,  # Get double to have room for filtering
                    source_id=source_filter
                )
            except EmbeddingError as e:
                logger.error(f"Vector search for code examples failed: {e}")
                return json.dumps({
                    "success": False,
                    "error": f"Vector search failed: {str(e)}",
                    "type": "embedding_error"
                })
            
            # 2. Get keyword search results using ILIKE on both content and summary
            try:
                async with db_pool.acquire() as connection:
                    keyword_query_str = """
                        SELECT id, url, chunk_number, content, summary, metadata, source_id, 0.5 as similarity
                        FROM code_examples 
                        WHERE (content ILIKE $1 OR summary ILIKE $1)
                    """
                    params: List[Any] = [f'%{query}%']
                    
                    if source_filter:
                        keyword_query_str += " AND source_id = $2"
                        params.append(cast(str, source_filter))
                        keyword_query_str += " LIMIT $3"
                        params.append(match_count * 2)
                    else:
                        keyword_query_str += " LIMIT $2"
                        params.append(match_count * 2)
                        
                    keyword_response = await connection.fetch(keyword_query_str, *params)

                keyword_results = [dict(row) for row in keyword_response] if keyword_response else []
            except Exception as e:
                logger.error(f"Keyword search for code examples failed: {e}")
                # Continue with vector results only
                keyword_results = []
            
            # 3. Combine results with preference for items appearing in both
            seen_ids = set()
            combined_results = []
            
            # First, add items that appear in both searches (these are the best matches)
            vector_ids = {r.get('id') for r in vector_results if r.get('id')}
            for kr in keyword_results:
                if kr['id'] in vector_ids and kr['id'] not in seen_ids:
                    # Find the vector result to get similarity score
                    for vr in vector_results:
                        if vr.get('id') == kr['id']:
                            # Boost similarity score for items in both results
                            vr['similarity'] = min(1.0, vr.get('similarity', 0) * 1.2)
                            combined_results.append(vr)
                            seen_ids.add(kr['id'])
                            break
            
            # Then add remaining vector results (semantic matches without exact keyword)
            for vr in vector_results:
                if vr.get('id') and vr['id'] not in seen_ids and len(combined_results) < match_count:
                    combined_results.append(vr)
                    seen_ids.add(vr['id'])
            
            # Finally, add pure keyword matches if we still need more results
            for kr in keyword_results:
                if kr['id'] not in seen_ids and len(combined_results) < match_count:
                    combined_results.append(kr)
                    seen_ids.add(kr['id'])
            
            # Use combined results
            results = combined_results[:match_count]
            
        else:
            logger.debug("Using vector search only for code examples")
            # Standard vector search only
            try:
                results = await search_code_examples_util(
                    pool=db_pool,
                    query=query,
                    match_count=match_count,
                    source_id=source_filter
                )
            except EmbeddingError as e:
                logger.error(f"Vector search for code examples failed: {e}")
                return json.dumps({
                    "success": False,
                    "error": f"Vector search failed: {str(e)}",
                    "type": "embedding_error"
                })
        
        # Apply reranking if enabled
        use_reranking = os.getenv("USE_RERANKING", "false") == "true"
        reranking_applied = False
        
        if use_reranking and ctx.request_context.lifespan_context.reranking_model and results:
            try:
                logger.debug("Applying reranking to code examples")
                results = rerank_results(ctx.request_context.lifespan_context.reranking_model, query, results, content_key="content")
                reranking_applied = True
            except Exception as e:
                logger.warning(f"Reranking failed, continuing with unranked results: {e}")
        
        # Format the results
        formatted_results = []
        for result in results:
            formatted_result = {
                "url": result.get("url"),
                "code": result.get("content"),
                "summary": result.get("summary"),
                "metadata": result.get("metadata"),
                "source_id": result.get("source_id"),
                "similarity": result.get("similarity")
            }
            # Include rerank score if available
            if "rerank_score" in result:
                formatted_result["rerank_score"] = result["rerank_score"]
            formatted_results.append(formatted_result)
        
        logger.info(f"Code examples search completed. Found {len(formatted_results)} results")
        
        return json.dumps({
            "success": True,
            "query": query,
            "source_filter": source_filter or "",
            "search_mode": "hybrid" if use_hybrid_search else "vector",
            "reranking_applied": reranking_applied,
            "results": formatted_results,
            "count": len(formatted_results)
        }, indent=2)
        
    except Exception as e:
        logger.error(f"Error in code examples search: {e}")
        return json.dumps({
            "success": False,
            "query": query,
            "error": str(e),
            "type": "query_error"
        })

@mcp.tool()
async def check_ai_script_hallucinations(ctx: Context, script_path: str) -> str:
    """
    Analyze a Python script for potential AI hallucinations by validating imports, classes, and methods against a knowledge graph.
    
    This tool performs comprehensive static analysis of Python scripts to detect:
    - Invalid imports (non-existent libraries or modules)
    - Non-existent classes being used
    - Invalid method calls on known classes
    - Incorrect function signatures and parameters
    - Misused attributes and properties
    
    The analysis uses AST parsing combined with Neo4j knowledge graph validation
    to provide detailed feedback on potential hallucinations with confidence scores.
    
    Args:
        ctx: The MCP server provided context
        script_path: Path to the Python script to analyze
    
    Returns:
        JSON string with detailed analysis results, confidence scores, and recommendations
    """
    try:
        # Check if knowledge graph functionality is enabled
        knowledge_graph_enabled = os.getenv("USE_KNOWLEDGE_GRAPH", "false") == "true"
        if not knowledge_graph_enabled:
            return json.dumps({
                "success": False,
                "error": "Knowledge graph functionality is disabled. Set USE_KNOWLEDGE_GRAPH=true to enable hallucination detection."
            }, indent=2)
        
        # Check if knowledge graph modules are available
        if not AIScriptAnalyzer or not HallucinationReporter:
            return json.dumps({
                "success": False,
                "error": "Knowledge graph modules not available. Hallucination detection requires knowledge graph dependencies."
            }, indent=2)
        
        # Get knowledge validator from context
        knowledge_validator = ctx.request_context.lifespan_context.knowledge_validator
        if not knowledge_validator:
            return json.dumps({
                "success": False,
                "error": "Knowledge graph validator not available. Check Neo4j configuration."
            }, indent=2)
        
        # Validate script path
        validation = validate_script_path(script_path)
        if not validation["valid"]:
            return json.dumps({
                "success": False,
                "script_path": script_path,
                "error": validation["error"]
            }, indent=2)
        
        # Step 1: Analyze script structure using AST
        analyzer = AIScriptAnalyzer()
        analysis_result = analyzer.analyze_script(script_path)
        
        if analysis_result.errors:
            print(f"Analysis warnings for {script_path}: {analysis_result.errors}")
        
        # Step 2: Validate against knowledge graph
        validation_result = await knowledge_validator.validate_script(analysis_result)
        
        # Step 3: Generate comprehensive report
        reporter = HallucinationReporter()
        report = reporter.generate_comprehensive_report(validation_result)
        
        # Format response with comprehensive information
        return json.dumps({
            "success": True,
            "script_path": script_path,
            "overall_confidence": validation_result.overall_confidence,
            "validation_summary": {
                "total_validations": report["validation_summary"]["total_validations"],
                "valid_count": report["validation_summary"]["valid_count"],
                "invalid_count": report["validation_summary"]["invalid_count"],
                "uncertain_count": report["validation_summary"]["uncertain_count"],
                "not_found_count": report["validation_summary"]["not_found_count"],
                "hallucination_rate": report["validation_summary"]["hallucination_rate"]
            },
            "hallucinations_detected": report["hallucinations_detected"],
            "recommendations": report["recommendations"],
            "analysis_metadata": {
                "total_imports": report["analysis_metadata"]["total_imports"],
                "total_classes": report["analysis_metadata"]["total_classes"],
                "total_methods": report["analysis_metadata"]["total_methods"],
                "total_attributes": report["analysis_metadata"]["total_attributes"],
                "total_functions": report["analysis_metadata"]["total_functions"]
            },
            "libraries_analyzed": report.get("libraries_analyzed", [])
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "script_path": script_path,
            "error": f"Analysis failed: {str(e)}"
        }, indent=2)

@mcp.tool()
async def query_knowledge_graph(ctx: Context, command: str) -> str:
    """
    Query and explore the Neo4j knowledge graph containing repository data.
    
    This tool provides comprehensive access to the knowledge graph for exploring repositories,
    classes, methods, functions, and their relationships. Perfect for understanding what data
    is available for hallucination detection and debugging validation results.
    
    **⚠️ IMPORTANT: Always start with the `repos` command first!**
    Before using any other commands, run `repos` to see what repositories are available
    in your knowledge graph. This will help you understand what data you can explore.
    
    ## Available Commands:
    
    **Repository Commands:**
    - `repos` - **START HERE!** List all repositories in the knowledge graph
    - `explore <repo_name>` - Get detailed overview of a specific repository
    
    **Class Commands:**  
    - `classes` - List all classes across all repositories (limited to 20)
    - `classes <repo_name>` - List classes in a specific repository
    - `class <class_name>` - Get detailed information about a specific class including methods and attributes
    
    **Method Commands:**
    - `method <method_name>` - Search for methods by name across all classes
    - `method <method_name> <class_name>` - Search for a method within a specific class
    
    **Custom Query:**
    - `query <cypher_query>` - Execute a custom Cypher query (results limited to 20 records)
    
    ## Knowledge Graph Schema:
    
    **Node Types:**
    - Repository: `(r:Repository {name: string})`
    - File: `(f:File {path: string, module_name: string})`
    - Class: `(c:Class {name: string, full_name: string})`
    - Method: `(m:Method {name: string, params_list: [string], params_detailed: [string], return_type: string, args: [string]})`
    - Function: `(func:Function {name: string, params_list: [string], params_detailed: [string], return_type: string, args: [string]})`
    - Attribute: `(a:Attribute {name: string, type: string})`
    
    **Relationships:**
    - `(r:Repository)-[:CONTAINS]->(f:File)`
    - `(f:File)-[:DEFINES]->(c:Class)`
    - `(c:Class)-[:HAS_METHOD]->(m:Method)`
    - `(c:Class)-[:HAS_ATTRIBUTE]->(a:Attribute)`
    - `(f:File)-[:DEFINES]->(func:Function)`
    
    ## Example Workflow:
    ```
    1. repos                                    # See what repositories are available
    2. explore pydantic-ai                      # Explore a specific repository
    3. classes pydantic-ai                      # List classes in that repository
    4. class Agent                              # Explore the Agent class
    5. method run_stream                        # Search for run_stream method
    6. method __init__ Agent                    # Find Agent constructor
    7. query "MATCH (c:Class)-[:HAS_METHOD]->(m:Method) WHERE m.name = 'run' RETURN c.name, m.name LIMIT 5"
    ```
    
    Args:
        ctx: The MCP server provided context
        command: Command string to execute (see available commands above)
    
    Returns:
        JSON string with query results, statistics, and metadata
    """
    try:
        # Check if knowledge graph functionality is enabled
        knowledge_graph_enabled = os.getenv("USE_KNOWLEDGE_GRAPH", "false") == "true"
        if not knowledge_graph_enabled:
            return json.dumps({
                "success": False,
                "error": "Knowledge graph functionality is disabled. Set USE_KNOWLEDGE_GRAPH=true in environment."
            }, indent=2)
        
        # Get Neo4j driver from context
        repo_extractor = ctx.request_context.lifespan_context.repo_extractor
        if not repo_extractor or not repo_extractor.driver:
            return json.dumps({
                "success": False,
                "error": "Neo4j connection not available. Check Neo4j configuration in environment variables."
            }, indent=2)
        
        # Parse command
        command = command.strip()
        if not command:
            return json.dumps({
                "success": False,
                "command": "",
                "error": "Command cannot be empty. Available commands: repos, explore <repo>, classes [repo], class <name>, method <name> [class], query <cypher>"
            }, indent=2)
        
        parts = command.split()
        cmd = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
        
        async with repo_extractor.driver.session() as session:
            # Route to appropriate handler
            if cmd == "repos":
                return await _handle_repos_command(session, command)
            elif cmd == "explore":
                if not args:
                    return json.dumps({
                        "success": False,
                        "command": command,
                        "error": "Repository name required. Usage: explore <repo_name>"
                    }, indent=2)
                return await _handle_explore_command(session, command, args[0])
            elif cmd == "classes":
                repo_name = args[0] if args else None
                return await _handle_classes_command(session, command, repo_name)
            elif cmd == "class":
                if not args:
                    return json.dumps({
                        "success": False,
                        "command": command,
                        "error": "Class name required. Usage: class <class_name>"
                    }, indent=2)
                return await _handle_class_command(session, command, args[0])
            elif cmd == "method":
                if not args:
                    return json.dumps({
                        "success": False,
                        "command": command,
                        "error": "Method name required. Usage: method <method_name> [class_name]"
                    }, indent=2)
                method_name = args[0]
                class_name = args[1] if len(args) > 1 else None
                return await _handle_method_command(session, command, method_name, class_name)
            elif cmd == "query":
                if not args:
                    return json.dumps({
                        "success": False,
                        "command": command,
                        "error": "Cypher query required. Usage: query <cypher_query>"
                    }, indent=2)
                cypher_query = " ".join(args)
                return await _handle_query_command(session, command, cypher_query)
            else:
                return json.dumps({
                    "success": False,
                    "command": command,
                    "error": f"Unknown command '{cmd}'. Available commands: repos, explore <repo>, classes [repo], class <name>, method <name> [class], query <cypher>"
                }, indent=2)
                
    except Exception as e:
        return json.dumps({
            "success": False,
            "command": command,
            "error": f"Query execution failed: {str(e)}"
        }, indent=2)


async def _handle_repos_command(session, command: str) -> str:
    """Handle 'repos' command - list all repositories"""
    query = "MATCH (r:Repository) RETURN r.name as name ORDER BY r.name"
    result = await session.run(query)
    
    repos = []
    async for record in result:
        repos.append(record['name'])
    
    return json.dumps({
        "success": True,
        "command": command,
        "data": {
            "repositories": repos
        },
        "metadata": {
            "total_results": len(repos),
            "limited": False
        }
    }, indent=2)


async def _handle_explore_command(session, command: str, repo_name: str) -> str:
    """Handle 'explore <repo>' command - get repository overview"""
    # Check if repository exists
    repo_check_query = "MATCH (r:Repository {name: $repo_name}) RETURN r.name as name"
    result = await session.run(repo_check_query, repo_name=repo_name)
    repo_record = await result.single()
    
    if not repo_record:
        return json.dumps({
            "success": False,
            "command": command,
            "error": f"Repository '{repo_name}' not found in knowledge graph"
        }, indent=2)
    
    # Get file count
    files_query = """
    MATCH (r:Repository {name: $repo_name})-[:CONTAINS]->(f:File)
    RETURN count(f) as file_count
    """
    result = await session.run(files_query, repo_name=repo_name)
    file_count = (await result.single())['file_count']
    
    # Get class count
    classes_query = """
    MATCH (r:Repository {name: $repo_name})-[:CONTAINS]->(f:File)-[:DEFINES]->(c:Class)
    RETURN count(DISTINCT c) as class_count
    """
    result = await session.run(classes_query, repo_name=repo_name)
    class_count = (await result.single())['class_count']
    
    # Get function count
    functions_query = """
    MATCH (r:Repository {name: $repo_name})-[:CONTAINS]->(f:File)-[:DEFINES]->(func:Function)
    RETURN count(DISTINCT func) as function_count
    """
    result = await session.run(functions_query, repo_name=repo_name)
    function_count = (await result.single())['function_count']
    
    # Get method count
    methods_query = """
    MATCH (r:Repository {name: $repo_name})-[:CONTAINS]->(f:File)-[:DEFINES]->(c:Class)-[:HAS_METHOD]->(m:Method)
    RETURN count(DISTINCT m) as method_count
    """
    result = await session.run(methods_query, repo_name=repo_name)
    method_count = (await result.single())['method_count']
    
    return json.dumps({
        "success": True,
        "command": command,
        "data": {
            "repository": repo_name,
            "statistics": {
                "files": file_count,
                "classes": class_count,
                "functions": function_count,
                "methods": method_count
            }
        },
        "metadata": {
            "total_results": 1,
            "limited": False
        }
    }, indent=2)


async def _handle_classes_command(session, command: str, repo_name: Optional[str] = None) -> str:
    """Handle 'classes [repo]' command - list classes"""
    limit = 20
    
    if repo_name:
        query = """
        MATCH (r:Repository {name: $repo_name})-[:CONTAINS]->(f:File)-[:DEFINES]->(c:Class)
        RETURN c.name as name, c.full_name as full_name
        ORDER BY c.name
        LIMIT $limit
        """
        result = await session.run(query, repo_name=repo_name, limit=limit)
    else:
        query = """
        MATCH (c:Class)
        RETURN c.name as name, c.full_name as full_name
        ORDER BY c.name
        LIMIT $limit
        """
        result = await session.run(query, limit=limit)
    
    classes = []
    async for record in result:
        classes.append({
            'name': record['name'],
            'full_name': record['full_name']
        })
    
    return json.dumps({
        "success": True,
        "command": command,
        "data": {
            "classes": classes,
            "repository_filter": repo_name or "All"
        },
        "metadata": {
            "total_results": len(classes),
            "limited": len(classes) >= limit
        }
    }, indent=2)


async def _handle_class_command(session, command: str, class_name: str) -> str:
    """Handle 'class <name>' command - explore specific class"""
    # Find the class
    class_query = """
    MATCH (c:Class)
    WHERE c.name = $class_name OR c.full_name = $class_name
    RETURN c.name as name, c.full_name as full_name
    LIMIT 1
    """
    result = await session.run(class_query, class_name=class_name)
    class_record = await result.single()
    
    if not class_record:
        return json.dumps({
            "success": False,
            "command": command,
            "error": f"Class '{class_name}' not found in knowledge graph"
        }, indent=2)
    
    actual_name = class_record['name']
    full_name = class_record['full_name']
    
    # Get methods
    methods_query = """
    MATCH (c:Class)-[:HAS_METHOD]->(m:Method)
    WHERE c.name = $class_name OR c.full_name = $class_name
    RETURN m.name as name, m.params_list as params_list, m.params_detailed as params_detailed, m.return_type as return_type
    ORDER BY m.name
    """
    result = await session.run(methods_query, class_name=class_name)
    
    methods = []
    async for record in result:
        # Use detailed params if available, fall back to simple params
        params_to_use = record['params_detailed'] or record['params_list'] or []
        methods.append({
            'name': record['name'],
            'parameters': params_to_use,
            'return_type': record['return_type'] or 'Any'
        })
    
    # Get attributes
    attributes_query = """
    MATCH (c:Class)-[:HAS_ATTRIBUTE]->(a:Attribute)
    WHERE c.name = $class_name OR c.full_name = $class_name
    RETURN a.name as name, a.type as type
    ORDER BY a.name
    """
    result = await session.run(attributes_query, class_name=class_name)
    
    attributes = []
    async for record in result:
        attributes.append({
            'name': record['name'],
            'type': record['type'] or 'Any'
        })
    
    return json.dumps({
        "success": True,
        "command": command,
        "data": {
            "class": {
                "name": actual_name,
                "full_name": full_name,
                "methods": methods,
                "attributes": attributes
            }
        },
        "metadata": {
            "total_results": 1,
            "methods_count": len(methods),
            "attributes_count": len(attributes),
            "limited": False
        }
    }, indent=2)


async def _handle_method_command(session, command: str, method_name: str, class_name: Optional[str] = None) -> str:
    """Handle 'method <name> [class]' command - search for methods"""
    limit = 20
    
    if class_name:
        query = """
        MATCH (c:Class)-[:HAS_METHOD]->(m:Method)
        WHERE (c.name = $class_name OR c.full_name = $class_name)
          AND m.name = $method_name
        RETURN c.name as class_name, c.full_name as class_full_name,
               m.name as method_name, m.params_list as params_list, 
               m.params_detailed as params_detailed, m.return_type as return_type, m.args as args
        """
        result = await session.run(query, class_name=class_name, method_name=method_name)
    else:
        query = """
        MATCH (c:Class)-[:HAS_METHOD]->(m:Method)
        WHERE m.name = $method_name
        RETURN c.name as class_name, c.full_name as class_full_name,
               m.name as method_name, m.params_list as params_list, 
               m.params_detailed as params_detailed, m.return_type as return_type, m.args as args
        ORDER BY c.name
        LIMIT 20
        """
        result = await session.run(query, method_name=method_name)
    
    methods = []
    async for record in result:
        # Use detailed params if available, fall back to simple params
        params_to_use = record['params_detailed'] or record['params_list'] or []
        methods.append({
            'class_name': record['class_name'],
            'class_full_name': record['class_full_name'],
            'method_name': record['method_name'],
            'parameters': params_to_use,
            'return_type': record['return_type'] or 'Any',
            'legacy_args': record['args'] or []
        })
    
    if not methods:
        return json.dumps({
            "success": False,
            "command": command,
            "error": f"Method '{method_name}'" + (f" in class '{class_name}'" if class_name else "") + " not found"
        }, indent=2)
    
    return json.dumps({
        "success": True,
        "command": command,
        "data": {
            "methods": methods,
            "class_filter": class_name or "All"
        },
        "metadata": {
            "total_results": len(methods),
            "limited": len(methods) >= 20 and not class_name
        }
    }, indent=2)


async def _handle_query_command(session, command: str, cypher_query: str) -> str:
    """Handle 'query <cypher>' command - execute custom Cypher query"""
    try:
        # Execute the query with a limit to prevent overwhelming responses
        result = await session.run(cypher_query)
        
        records = []
        count = 0
        async for record in result:
            records.append(dict(record))
            count += 1
            if count >= 20:  # Limit results to prevent overwhelming responses
                break
        
        return json.dumps({
            "success": True,
            "command": command,
            "data": {
                "query": cypher_query,
                "results": records
            },
            "metadata": {
                "total_results": len(records),
                "limited": len(records) >= 20
            }
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "command": command,
            "error": f"Cypher query error: {str(e)}",
            "data": {
                "query": cypher_query
            }
        }, indent=2)


@mcp.tool()
async def parse_github_repository(ctx: Context, repo_url: str) -> str:
    """
    Parse a GitHub repository into the Neo4j knowledge graph.
    
    This tool clones a GitHub repository, analyzes its Python files, and stores
    the code structure (classes, methods, functions, imports) in Neo4j for use
    in hallucination detection. The tool:
    
    - Clones the repository to a temporary location
    - Analyzes Python files to extract code structure
    - Stores classes, methods, functions, and imports in Neo4j
    - Provides detailed statistics about the parsing results
    - Automatically handles module name detection for imports
    
    Args:
        ctx: The MCP server provided context
        repo_url: GitHub repository URL (e.g., 'https://github.com/user/repo.git')
    
    Returns:
        JSON string with parsing results, statistics, and repository information
    """
    try:
        # Check if knowledge graph functionality is enabled
        knowledge_graph_enabled = os.getenv("USE_KNOWLEDGE_GRAPH", "false") == "true"
        if not knowledge_graph_enabled:
            return json.dumps({
                "success": False,
                "error": "Knowledge graph functionality is disabled. Set USE_KNOWLEDGE_GRAPH=true in environment."
            }, indent=2)
        
        # Get the repository extractor from context
        repo_extractor = ctx.request_context.lifespan_context.repo_extractor
        
        if not repo_extractor:
            return json.dumps({
                "success": False,
                "error": "Repository extractor not available. Check Neo4j configuration in environment variables."
            }, indent=2)
        
        # Validate repository URL
        validation = validate_github_url(repo_url)
        if not validation["valid"]:
            return json.dumps({
                "success": False,
                "repo_url": repo_url,
                "error": validation["error"]
            }, indent=2)
        
        repo_name = validation["repo_name"]
        
        # Parse the repository (this includes cloning, analysis, and Neo4j storage)
        print(f"Starting repository analysis for: {repo_name}")
        await repo_extractor.analyze_repository(repo_url)
        print(f"Repository analysis completed for: {repo_name}")
        
        # Query Neo4j for statistics about the parsed repository
        async with repo_extractor.driver.session() as session:
            # Get comprehensive repository statistics
            stats_query = """
            MATCH (r:Repository {name: $repo_name})
            OPTIONAL MATCH (r)-[:CONTAINS]->(f:File)
            OPTIONAL MATCH (f)-[:DEFINES]->(c:Class)
            OPTIONAL MATCH (c)-[:HAS_METHOD]->(m:Method)
            OPTIONAL MATCH (f)-[:DEFINES]->(func:Function)
            OPTIONAL MATCH (c)-[:HAS_ATTRIBUTE]->(a:Attribute)
            WITH r, 
                 count(DISTINCT f) as files_count,
                 count(DISTINCT c) as classes_count,
                 count(DISTINCT m) as methods_count,
                 count(DISTINCT func) as functions_count,
                 count(DISTINCT a) as attributes_count
            
            // Get some sample module names
            OPTIONAL MATCH (r)-[:CONTAINS]->(sample_f:File)
            WITH r, files_count, classes_count, methods_count, functions_count, attributes_count,
                 collect(DISTINCT sample_f.module_name)[0..5] as sample_modules
            
            RETURN 
                r.name as repo_name,
                files_count,
                classes_count, 
                methods_count,
                functions_count,
                attributes_count,
                sample_modules
            """
            
            result = await session.run(stats_query, repo_name=repo_name)
            record = await result.single()
            
            if record:
                stats = {
                    "repository": record['repo_name'],
                    "files_processed": record['files_count'],
                    "classes_created": record['classes_count'],
                    "methods_created": record['methods_count'], 
                    "functions_created": record['functions_count'],
                    "attributes_created": record['attributes_count'],
                    "sample_modules": record['sample_modules'] or []
                }
            else:
                return json.dumps({
                    "success": False,
                    "repo_url": repo_url,
                    "error": f"Repository '{repo_name}' not found in database after parsing"
                }, indent=2)
        
        return json.dumps({
            "success": True,
            "repo_url": repo_url,
            "repo_name": repo_name,
            "message": f"Successfully parsed repository '{repo_name}' into knowledge graph",
            "statistics": stats,
            "ready_for_validation": True,
            "next_steps": [
                "Repository is now available for hallucination detection",
                f"Use check_ai_script_hallucinations to validate scripts against {repo_name}",
                "The knowledge graph contains classes, methods, and functions from this repository"
            ]
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "repo_url": repo_url,
            "error": f"Repository parsing failed: {str(e)}"
        }, indent=2)

async def crawl_markdown_file(crawler: AsyncWebCrawler, url: str) -> List[Dict[str, Any]]:
    """
    Crawl a .txt or markdown file.
    
    Args:
        crawler: AsyncWebCrawler instance
        url: URL of the file
        
    Returns:
        List of dictionaries with URL and markdown content
    """
    crawl_config = CrawlerRunConfig()

    result_any: Any = await crawler.arun(url=url, config=crawl_config)
    if cast(Any, result_any).success and cast(Any, result_any).markdown:  # type: ignore[attr-defined]
        return [{'url': url, 'markdown': cast(Any, result_any).markdown}]  # type: ignore[attr-defined]
    else:
        print(f"Failed to crawl {url}: {getattr(result_any, 'error_message', 'unknown error')}")
        return []

async def crawl_batch(crawler: AsyncWebCrawler, urls: List[str], max_concurrent: int = 10) -> List[Dict[str, Any]]:
    """
    Batch crawl multiple URLs in parallel.
    
    Args:
        crawler: AsyncWebCrawler instance
        urls: List of URLs to crawl
        max_concurrent: Maximum number of concurrent browser sessions
        
    Returns:
        List of dictionaries with URL and markdown content
    """
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=max_concurrent
    )

    results_any: List[Any] = await crawler.arun_many(urls=urls, config=crawl_config, dispatcher=dispatcher)  # type: ignore[assignment]
    filtered: List[Dict[str, str]] = []
    for r in results_any:
        if getattr(r, 'success', False) and getattr(r, 'markdown', None):
            filtered.append({'url': cast(Any, r).url, 'markdown': cast(Any, r).markdown})  # type: ignore[attr-defined]
    return filtered

async def crawl_recursive_internal_links(crawler: AsyncWebCrawler, start_urls: List[str], max_depth: int = 3, max_concurrent: int = 10) -> List[Dict[str, Any]]:
    """
    Recursively crawl internal links from start URLs up to a maximum depth.
    
    Args:
        crawler: AsyncWebCrawler instance
        start_urls: List of starting URLs
        max_depth: Maximum recursion depth
        max_concurrent: Maximum number of concurrent browser sessions
        
    Returns:
        List of dictionaries with URL and markdown content
    """
    run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=max_concurrent
    )

    visited = set()

    def normalize_url(url):
        return urldefrag(url)[0]

    current_urls = set([normalize_url(u) for u in start_urls])
    results_all = []

    for depth in range(max_depth):
        urls_to_crawl = [normalize_url(url) for url in current_urls if normalize_url(url) not in visited]
        if not urls_to_crawl:
            break

        results_any: List[Any] = await crawler.arun_many(urls=urls_to_crawl, config=run_config, dispatcher=dispatcher)  # type: ignore[assignment]
        next_level_urls = set()

        for result_any in results_any:
            norm_url = normalize_url(cast(Any, result_any).url)  # type: ignore[attr-defined]
            visited.add(norm_url)

            if getattr(result_any, 'success', False) and getattr(result_any, 'markdown', None):
                results_all.append({'url': cast(Any, result_any).url, 'markdown': cast(Any, result_any).markdown})  # type: ignore[attr-defined]
                # Resolve and validate internal links before enqueueing them for the next depth level
                for link in getattr(result_any, 'links', {}).get("internal", []):  # type: ignore[attr-defined]
                    raw_href: str = link.get("href", "")

                    # Resolve relative URLs against the current page URL to obtain an absolute URL
                    resolved_href = urljoin(cast(Any, result_any).url, raw_href)

                    # Normalise (remove fragments) and validate the URL
                    next_url = normalize_url(resolved_href)

                    # Only follow links that stay on the **same origin** as the start page to avoid external crawls
                    try:
                        if urlparse(next_url).netloc != urlparse(cast(Any, result_any).url).netloc:
                            continue  # Skip external domains
                    except Exception:
                        continue  # Skip malformed URLs

                    # Enqueue the URL for the next crawl depth if we haven't visited it yet
                    if next_url and next_url not in visited:
                        next_level_urls.add(next_url)

        current_urls = next_level_urls

    return results_all

async def main():
    """Main entry point for the MCP server."""
    transport = os.getenv("TRANSPORT", "sse")
    
    if transport == 'sse':
        try:
            print(f"Starting MCP server on {os.getenv('HOST', '0.0.0.0')}:{os.getenv('PORT', '8051')}")
            # Run the MCP server with sse transport
            await mcp.run_sse_async()
        except KeyboardInterrupt:
            print("Server shutdown requested...")
        except Exception as e:
            print(f"Server error: {e}")
            raise
    else:
        try:
            print("Starting MCP server with stdio transport")
            # Run the MCP server with stdio transport  
            await mcp.run_stdio_async()
        except KeyboardInterrupt:
            print("Server shutdown requested...")
        except Exception as e:
            print(f"Server error: {e}")
            raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server stopped by user")
    except Exception as e:
        print(f"Failed to start server: {e}")
        sys.exit(1)