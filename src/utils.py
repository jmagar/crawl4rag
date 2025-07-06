"""
Utility functions for the Crawl4AI MCP server.
Includes database operations, embedding generation, and configuration management.
"""
import os
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple
import json
import asyncpg
from urllib.parse import urlparse
from openai import AsyncOpenAI, OpenAI
import time
import logging
from dataclasses import dataclass
import asyncio
from functools import lru_cache

# Configure logging
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Configuration validation and management
# -----------------------------------------------------------------------------

# OpenAI clients will be initialized after configuration is loaded
embedding_client: OpenAI
async_openai_client: AsyncOpenAI

@dataclass
class DatabaseConfig:
    """Database configuration with validation"""
    user: str
    password: str
    database: str
    host: str
    port: int
    min_connections: int = 2
    max_connections: int = 10
    default_owner_id: str = "mcp_app_user"
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.port <= 0:
            raise ValueError("Port must be positive")
        if self.min_connections <= 0:
            raise ValueError("min_connections must be positive")
        if self.max_connections < self.min_connections:
            raise ValueError("max_connections must be >= min_connections")
        if not all([self.host, self.database, self.user, self.password]):
            raise ValueError("All database fields are required")
    
    @classmethod
    def from_env(cls) -> 'DatabaseConfig':
        """Create config from environment variables with validation"""
        # Support both direct credentials and file-based secrets
        user = cls._get_credential('POSTGRES_USER')
        password = cls._get_credential('POSTGRES_PASSWORD')
        
        if not user or not password:
            raise ValueError("PostgreSQL credentials not found. Set POSTGRES_USER/POSTGRES_PASSWORD or use Docker secrets.")
        
        return cls(
            user=user,
            password=password,
            database=os.getenv("POSTGRES_DB", "crawl4rag"),
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            min_connections=int(os.getenv("DB_MIN_CONNECTIONS", "2")),
            max_connections=int(os.getenv("DB_MAX_CONNECTIONS", "10")),
            default_owner_id=os.getenv("DEFAULT_OWNER_ID", "mcp_app_user")
        )
    
    @staticmethod
    def _get_credential(env_var: str) -> Optional[str]:
        """Get credential from environment variable or file"""
        # First try direct environment variable
        value = os.getenv(env_var)
        if value:
            return value
        
        # Try file-based secret (Docker secrets)
        file_var = f"{env_var}_FILE"
        file_path = os.getenv(file_var)
        if file_path and os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    return f.read().strip()
            except Exception as e:
                logger.error(f"Failed to read credential file {file_path}: {e}")
        
        return None

@dataclass 
class OpenAIConfig:
    """OpenAI configuration with validation"""
    api_key: str
    embedding_url: str
    embedding_model: str
    contextual_model: str
    max_retries: int = 3
    retry_delay: float = 1.0
    # Contextual embedding optimization settings
    contextual_max_tokens: int = 75
    contextual_document_limit: int = 15000
    # Intelligent batching configuration
    openai_tier: int = 2
    use_intelligent_batching: bool = True
    contextual_batch_size_override: Optional[int] = None
    
    @classmethod
    def from_env(cls) -> 'OpenAIConfig':
        """Create config from environment variables with validation"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required")
        
        embedding_url = os.getenv("EMBEDDING_URL", "https://api.openai.com/v1/")
        if not embedding_url.endswith("/"):
            embedding_url += "/"
        
        return cls(
            api_key=api_key,
            embedding_url=embedding_url,
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-bge-m3"),
            contextual_model=os.getenv("CONTEXTUAL_MODEL", "gpt-4o-mini"),
            max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "3")),
            retry_delay=float(os.getenv("OPENAI_RETRY_DELAY", "1.0")),
            contextual_max_tokens=int(os.getenv("CONTEXTUAL_MAX_TOKENS", "75")),
            contextual_document_limit=int(os.getenv("CONTEXTUAL_DOCUMENT_LIMIT", "15000")),
            openai_tier=int(os.getenv("OPENAI_TIER", "2")),
            use_intelligent_batching=os.getenv("USE_INTELLIGENT_BATCHING", "true") == "true",
            contextual_batch_size_override=int(os.getenv("CONTEXTUAL_BATCH_SIZE_OVERRIDE", "0")) or None
        )


# Initialize global configuration
try:
    db_config = DatabaseConfig.from_env()
    openai_config = OpenAIConfig.from_env()
except Exception as e:
    logger.error(f"Configuration error: {e}")
    raise

# Initialize OpenAI clients after configuration is loaded
# Dedicated client for embeddings
embedding_client = OpenAI(api_key=openai_config.api_key, base_url=openai_config.embedding_url)

# Async client for contextual embeddings
async_openai_client = AsyncOpenAI(api_key=openai_config.api_key)

logger.info(f"Configured OpenAI clients - Embedding: {openai_config.embedding_model}, "
            f"Contextual: {openai_config.contextual_model}, "
            f"Tier: {openai_config.openai_tier}, "
            f"Intelligent Batching: {openai_config.use_intelligent_batching}")

# Global connection pool (will be initialized once)
_db_pool: Optional[asyncpg.Pool] = None

async def get_db_pool() -> asyncpg.Pool:
    """Get or create an asyncpg connection pool with proper resource limits."""
    global _db_pool

    # `asyncpg.Pool` exposes the boolean attribute `closed` in type stubs. Fallback to
    # `True` when it is missing to keep static analyzers happy.
    pool_closed = True if _db_pool is None else getattr(_db_pool, "closed", True)

    if _db_pool is None or pool_closed:
        try:
            logger.info(
                "Creating database connection pool (min=%s, max=%s)",
                db_config.min_connections,
                db_config.max_connections,
            )
            _db_pool = await asyncpg.create_pool(
                user=db_config.user,
                password=db_config.password,
                database=db_config.database,
                host=db_config.host,
                port=db_config.port,
                min_size=db_config.min_connections,
                max_size=db_config.max_connections,
                command_timeout=60,
                server_settings={
                    "application_name": "crawl4ai_mcp_server",
                    "jit": "off",  # Disable JIT for better connection performance
                },
            )
            logger.info("Database connection pool created successfully")
        except Exception as e:
            logger.error("Failed to create database pool: %s", e)
            raise

    return _db_pool

async def close_db_pool() -> None:
    """Close the database connection pool if it is still open."""
    global _db_pool
    if _db_pool and not getattr(_db_pool, "closed", True):
        await _db_pool.close()
        logger.info("Database connection pool closed")

class EmbeddingError(Exception):
    """Custom exception for embedding generation errors"""
    pass

def create_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Create embeddings for multiple texts in a single API call with proper error handling.
    
    Args:
        texts: List of texts to create embeddings for
        
    Returns:
        List of embeddings (each embedding is a list of floats)
        
    Raises:
        EmbeddingError: If embedding generation fails completely
    """
    if not texts:
        return []
    
    # Ensure each embedding matches DB vector dimension (1024)
    target_dim = 1024
    def _fix_dim(vec: List[float]) -> List[float]:
        if len(vec) == target_dim:
            return vec
        if len(vec) > target_dim:
            return vec[:target_dim]
        # pad with zeros
        return vec + [0.0] * (target_dim - len(vec))
    
    for retry in range(openai_config.max_retries):
        try:
            logger.debug(f"Creating embeddings for {len(texts)} texts (attempt {retry + 1})")
            response = embedding_client.embeddings.create(
                model=openai_config.embedding_model,
                input=texts
            )

            embeddings_batch = [_fix_dim(d.embedding) for d in response.data]
            embeddings = embeddings_batch
            logger.debug(f"Successfully created {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            if retry < openai_config.max_retries - 1:
                wait_time = openai_config.retry_delay * (2 ** retry)  # Exponential backoff
                logger.warning(f"Error creating batch embeddings (attempt {retry + 1}/{openai_config.max_retries}): {e}")
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"Failed to create batch embeddings after {openai_config.max_retries} attempts: {e}")
                
                # Try creating embeddings one by one as fallback
                logger.info("Attempting to create embeddings individually...")
                embeddings = []
                successful_count = 0
                
                for i, text in enumerate(texts):
                    try:
                        individual_response = embedding_client.embeddings.create(
                            model=openai_config.embedding_model,
                            input=[text]
                        )
                        embeddings.append(_fix_dim(individual_response.data[0].embedding))
                        successful_count += 1
                    except Exception as individual_error:
                        logger.warning(f"Failed to create embedding for text {i}: {individual_error}")
                        # Raise error instead of using zero embeddings
                        raise EmbeddingError(f"Failed to create embedding for text {i}: {individual_error}") from individual_error
                
                logger.info(f"Successfully created {successful_count}/{len(texts)} embeddings individually")
                return embeddings
    
    raise EmbeddingError("Failed to create embeddings after all retry attempts")

def create_embedding(text: str) -> List[float]:
    """
    Create an embedding for a single text using OpenAI's API with proper error handling.
    
    Args:
        text: Text to create an embedding for
        
    Returns:
        List of floats representing the embedding
        
    Raises:
        EmbeddingError: If embedding generation fails
    """
    try:
        embeddings = create_embeddings_batch([text])
        return embeddings[0] if embeddings else []
    except Exception as e:
        logger.error(f"Error creating single embedding: {e}")
        raise EmbeddingError(f"Failed to create embedding: {e}") from e

async def generate_contextual_embedding(full_document: str, chunk: str) -> Tuple[str, bool]:
    """
    Generate contextual information for a chunk within a document to improve retrieval.
    Uses async OpenAI client with optimized settings for better performance.
    
    Args:
        full_document: The complete document text
        chunk: The specific chunk of text to generate context for
        
    Returns:
        Tuple containing:
        - The contextual text that situates the chunk within the document
        - Boolean indicating if contextual embedding was performed
    """
    try:
        # Create the prompt with reduced document size for faster processing
        prompt = f"""<document> 
{full_document[:openai_config.contextual_document_limit]} 
</document>
Here is the chunk we want to situate within the whole document 
<chunk> 
{chunk}
</chunk> 
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else. Keep it under 50 words."""

        # Call the OpenAI API with reduced token limits for faster responses
        response = await async_openai_client.chat.completions.create(
            model=openai_config.contextual_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides concise contextual information in 50 words or less."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=openai_config.contextual_max_tokens  # Reduced from 200 to 75
        )
        
        # Extract the generated context
        context = response.choices[0].message.content.strip() if response.choices[0].message.content else ""
        
        # Combine the context with the original chunk
        contextual_text = f"{context}\n---\n{chunk}"
        
        return contextual_text, True
    
    except Exception as e:
        logger.warning(f"Error generating contextual embedding: {e}. Using original chunk instead.")
        return chunk, False

async def process_chunk_with_context(args: Tuple[str, str, str]) -> Tuple[str, bool]:
    """
    Process a single chunk with contextual embedding asynchronously.
    
    Args:
        args: Tuple containing (url, content, full_document)
        
    Returns:
        Tuple containing:
        - The contextual text that situates the chunk within the document
        - Boolean indicating if contextual embedding was performed
    """
    url, content, full_document = args
    return await generate_contextual_embedding(full_document, content)

def calculate_optimal_batch_size(tier: int, avg_chunk_size: int = 500) -> dict:
    """Calculate optimal batch sizes based on OpenAI tier limits."""
    
    tier_limits = {
        1: {"rpm": 500, "tpm": 30000},
        2: {"rpm": 5000, "tpm": 450000}, 
        3: {"rpm": 5000, "tpm": 800000},
        4: {"rpm": 10000, "tpm": 2000000},
        5: {"rpm": 10000, "tpm": 4000000}
    }
    
    limits = tier_limits.get(tier, tier_limits[1])
    
    # Estimate tokens per contextual request
    # Document context (15k chars ≈ 3750 tokens) + chunk (500 chars ≈ 125 tokens) + response (75 tokens)
    tokens_per_request = 3750 + 125 + 75  # ≈ 3950 tokens
    
    # Calculate max requests based on token limits
    max_requests_by_tokens = limits["tpm"] // tokens_per_request
    max_requests_by_rpm = limits["rpm"]
    
    # Use the more restrictive limit
    max_concurrent_requests = min(max_requests_by_tokens, max_requests_by_rpm)
    
    # For batching: estimate how many chunks can fit in one API call
    # Conservative estimate: 8000 tokens for document + chunks + responses
    tokens_per_chunk_in_batch = 200  # chunk + small context
    max_chunks_per_batch = min(8000 // tokens_per_chunk_in_batch, 10)  # Cap at 10 for reliability
    
    return {
        "tier": tier,
        "max_concurrent_requests": max_concurrent_requests,
        "max_chunks_per_batch": max_chunks_per_batch,
        "estimated_speedup": f"{max_chunks_per_batch}x",
        "rpm_limit": limits["rpm"],
        "tpm_limit": limits["tpm"]
    }

async def generate_contextual_embeddings_batch(
    full_document: str, 
    chunks: List[str],
    tier: int = 2
) -> List[Tuple[str, bool]]:
    """
    Generate contextual embeddings using intelligent batching based on OpenAI tier limits.
    """
    batch_start_time = time.time()
    
    if not openai_config.use_intelligent_batching:
        # Fall back to individual processing
        logger.info(f"Intelligent batching disabled - processing {len(chunks)} chunks individually")
        individual_start = time.time()
        result = await _fallback_individual_context(full_document, chunks)
        individual_end = time.time()
        logger.info(f"Individual processing took {individual_end - individual_start:.2f} seconds "
                   f"({len(chunks)} chunks = {(individual_end - individual_start) / len(chunks):.3f}s per chunk)")
        return result
    
    config = calculate_optimal_batch_size(tier)
    
    # Use override if specified, otherwise use calculated batch size
    batch_size = openai_config.contextual_batch_size_override or config["max_chunks_per_batch"]
    max_concurrent = min(config["max_concurrent_requests"], 20)  # Safety cap
    
    num_batches = (len(chunks) + batch_size - 1) // batch_size  # Ceiling division
    estimated_api_calls = num_batches
    
    logger.info(f"Intelligent batching enabled - processing {len(chunks)} chunks with Tier {tier} limits: "
                f"batch_size={batch_size}, max_concurrent={max_concurrent}, "
                f"estimated_api_calls={estimated_api_calls} (vs {len(chunks)} individual calls)")
    
    # Split chunks into batches
    batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
    
    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_batch(batch_chunks: List[str]) -> List[Tuple[str, bool]]:
        async with semaphore:
            return await _generate_multi_chunk_context(full_document, batch_chunks)
    
    # Process all batches concurrently
    batch_processing_start = time.time()
    tasks = [process_batch(batch) for batch in batches]
    batch_results = await asyncio.gather(*tasks, return_exceptions=True)
    batch_processing_end = time.time()
    
    # Flatten results
    results = []
    successful_batches = 0
    failed_batches = 0
    
    for batch_idx, batch_result in enumerate(batch_results):
        if isinstance(batch_result, Exception):
            logger.error(f"Batch {batch_idx} failed: {batch_result}")
            # Add fallback results for failed batch
            batch_chunks = batches[batch_idx]
            results.extend([(chunk, False) for chunk in batch_chunks])
            failed_batches += 1
        elif isinstance(batch_result, list):
            results.extend(batch_result)
            successful_batches += 1
        else:
            # Handle unexpected result type
            logger.warning(f"Unexpected batch result type: {type(batch_result)}")
            batch_chunks = batches[batch_idx]
            results.extend([(chunk, False) for chunk in batch_chunks])
            failed_batches += 1
    
    total_time = time.time() - batch_start_time
    batch_processing_time = batch_processing_end - batch_processing_start
    
    # Calculate performance metrics
    actual_api_calls = successful_batches + failed_batches * len(batches[0])  # Estimate for failed batches
    api_call_reduction = ((len(chunks) - actual_api_calls) / len(chunks)) * 100 if len(chunks) > 0 else 0
    
    logger.info(f"Contextual embedding batching completed in {total_time:.2f}s - "
               f"Processed {len(chunks)} chunks in {successful_batches} successful batches "
               f"({failed_batches} failed). "
               f"API call reduction: {api_call_reduction:.1f}% "
               f"({actual_api_calls} calls vs {len(chunks)} individual calls)")
    
    return results

async def _generate_multi_chunk_context(
    full_document: str, 
    chunks: List[str]
) -> List[Tuple[str, bool]]:
    """Generate context for multiple chunks in a single API call."""
    
    if len(chunks) == 1:
        # Single chunk - use existing optimized function
        return [await generate_contextual_embedding(full_document, chunks[0])]
    
    # Multi-chunk batching
    prompt = f"""<document>
{full_document[:openai_config.contextual_document_limit]}
</document>

Provide concise context (max 30 words each) for these chunks:

{chr(10).join([f"CHUNK_{i+1}: {chunk[:300]}..." for i, chunk in enumerate(chunks)])}

Respond with exactly {len(chunks)} lines, each starting with "CHUNK_X:" followed by the context."""

    try:
        response = await async_openai_client.chat.completions.create(
            model=openai_config.contextual_model,
            messages=[
                {"role": "system", "content": "Provide concise context for each chunk. Format: 'CHUNK_X: [context]'"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=min(len(chunks) * 50, 500)  # Scale with chunk count
        )
        
        content = response.choices[0].message.content
        if content is None:
            content = ""
        content = content.strip()
        lines = content.split('\n')
        
        # Parse responses
        contexts = []
        for i, line in enumerate(lines):
            if f"CHUNK_{i+1}:" in line:
                context = line.split(f"CHUNK_{i+1}:", 1)[1].strip()
                contexts.append(context)
            else:
                contexts.append(f"Context for chunk {i+1}")  # Fallback
        
        # Ensure we have the right number of contexts
        while len(contexts) < len(chunks):
            contexts.append(f"Context for chunk {len(contexts)+1}")
        
        # Combine contexts with chunks
        results = []
        for i, chunk in enumerate(chunks):
            context = contexts[i] if i < len(contexts) else f"Context for chunk {i+1}"
            contextual_text = f"{context}\n---\n{chunk}"
            results.append((contextual_text, True))
        
        logger.debug(f"Successfully processed batch of {len(chunks)} chunks")
        return results
        
    except Exception as e:
        logger.warning(f"Batch context generation failed: {e}")
        # Fallback to individual processing
        return await _fallback_individual_context(full_document, chunks)

async def _fallback_individual_context(
    full_document: str, 
    chunks: List[str]
) -> List[Tuple[str, bool]]:
    """Fallback to individual processing if batch fails."""
    logger.info(f"Falling back to individual processing for {len(chunks)} chunks")
    
    tasks = [generate_contextual_embedding(full_document, chunk) for chunk in chunks]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    final_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.warning(f"Individual context generation failed: {result}")
            final_results.append((chunks[i], False))  # Use original chunk
        else:
            final_results.append(result)
    
    return final_results

async def add_documents_to_db(
    pool: asyncpg.Pool,
    urls: List[str],
    chunk_numbers: List[int],
    contents: List[str],
    metadatas: List[Dict[str, Any]],
    url_to_full_document: Dict[str, str],
    batch_size: int = 20,
) -> None:
    """Batch-insert crawled document chunks into `crawled_pages` table."""
    if not urls:
        logger.warning("No URLs provided to add_documents_to_db")
        return

    unique_urls: List[str] = list(set(urls))
    use_contextual_embeddings = os.getenv("USE_CONTEXTUAL_EMBEDDINGS", "false") == "true"
    logger.info("Using contextual embeddings: %s", use_contextual_embeddings)

    async with pool.acquire() as connection:
        total_processed = 0
        
        # Start transaction
        async with connection.transaction():
            try:
                if unique_urls:
                    await connection.execute(
                        "DELETE FROM crawled_pages WHERE url = ANY($1)", unique_urls
                    )
                    logger.info("Deleted existing records for %s URLs", len(unique_urls))

                for start in range(0, len(contents), batch_size):
                    end = min(start + batch_size, len(contents))

                    batch_urls = urls[start:end]
                    batch_chunk_numbers = chunk_numbers[start:end]
                    batch_contents = contents[start:end]
                    batch_metadatas = [meta.copy() for meta in metadatas[start:end]]

                    # Contextual embedding (optional)
                    if use_contextual_embeddings:
                        try:
                            # Group chunks by document to enable better batching
                            doc_to_chunks = {}
                            for i, (url, content) in enumerate(zip(batch_urls, batch_contents)):
                                full_doc = url_to_full_document.get(url, "")
                                if full_doc not in doc_to_chunks:
                                    doc_to_chunks[full_doc] = []
                                doc_to_chunks[full_doc].append((i, content))
                            
                            # Process each document's chunks
                            all_results: List[Optional[Tuple[str, bool]]] = [None] * len(batch_contents)
                            for full_doc, chunks_with_indices in doc_to_chunks.items():
                                indices, chunks = zip(*chunks_with_indices)
                                contextual_results = await generate_contextual_embeddings_batch(
                                    full_doc, list(chunks), openai_config.openai_tier
                                )
                                for idx, result in zip(indices, contextual_results):
                                    all_results[idx] = result
                            
                            contextual_contents: List[str] = []
                            for i, result in enumerate(all_results):
                                if result and result[1]:
                                    contextual_contents.append(result[0])
                                    batch_metadatas[i]["contextual_embedding"] = True
                                else:
                                    contextual_contents.append(batch_contents[i])
                        except Exception as e:
                            logger.error(f"Contextual embedding failed for batch, using original content: {e}")
                            contextual_contents = batch_contents
                    else:
                        contextual_contents = batch_contents

                    # Vector embeddings
                    try:
                        batch_embeddings = create_embeddings_batch(contextual_contents)
                    except EmbeddingError as e:
                        logger.error(f"Vector embedding failed for batch, skipping batch: {e}")
                        continue  # Skip this batch

                    # Prepare data for insertion
                    batch_data = []
                    for i, embedding in enumerate(batch_embeddings):
                        parsed_url = urlparse(batch_urls[i])
                        source_id = parsed_url.netloc or parsed_url.path
                        owner_id = db_config.default_owner_id
                        chunk_size = len(contextual_contents[i])
                        metadata_json = json.dumps({"chunk_size": chunk_size, **batch_metadatas[i]})
                        batch_data.append(
                            (
                                batch_urls[i], batch_chunk_numbers[i], batch_contents[i],
                                metadata_json, source_id, owner_id, json.dumps(embedding),
                            )
                        )

                    # Insert data
                    try:
                        await connection.executemany(
                            """
                            INSERT INTO crawled_pages (url, chunk_number, content, metadata, source_id, owner_id, embedding)
                            VALUES ($1, $2, $3, $4, $5, $6, $7)
                            """,
                            batch_data,
                        )
                        total_processed += len(batch_data)
                        logger.debug("Successfully inserted batch of %s records", len(batch_data))
                    except Exception as e:
                        logger.error(f"Database insertion failed for batch, skipping batch: {e}")
                        continue # Skip this batch
                
                logger.info("Successfully inserted %s documents into database", total_processed)

            except Exception as e:
                logger.error("Major error during document insertion transaction: %s", e)
                # The transaction will be rolled back automatically by the `async with` block
                raise

    if total_processed == 0 and len(contents) > 0:
        logger.error("CRITICAL: Failed to insert any documents. All batches failed.")

async def search_documents(
    pool: asyncpg.Pool,
    query: str,
    match_count: int = 10,
    filter_metadata: Optional[Dict[str, Any]] = None,
    source_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Search documents using vector similarity."""
    try:
        query_embedding = create_embedding(query)
    except EmbeddingError as e:
        logger.error("Failed to create query embedding: %s", e)
        raise

    try:
        async with pool.acquire() as connection:
            result = await connection.fetch(
                "SELECT * FROM match_crawled_pages($1, $2, $3, $4)",
                json.dumps(query_embedding),  # store embedding as JSON string
                match_count,
                json.dumps(filter_metadata or {}),
                source_filter,
            )
        return [dict(row) for row in result]
    except Exception as e:
        logger.error("Error searching documents: %s", e)
        return []


def extract_code_blocks(markdown_content: str, min_length: int = 1000) -> List[Dict[str, Any]]:
    """
    Extract code blocks from markdown content along with context.
    
    Args:
        markdown_content: The markdown content to extract code blocks from
        min_length: Minimum length of code blocks to extract (default: 1000 characters)
        
    Returns:
        List of dictionaries containing code blocks and their context
    """
    code_blocks = []
    
    # Skip if content starts with triple backticks (edge case for files wrapped in backticks)
    content = markdown_content.strip()
    start_offset = 0
    if content.startswith('```'):
        # Skip the first triple backticks
        start_offset = 3
        print("Skipping initial triple backticks")
    
    # Find all occurrences of triple backticks
    backtick_positions = []
    pos = start_offset
    while True:
        pos = markdown_content.find('```', pos)
        if pos == -1:
            break
        backtick_positions.append(pos)
        pos += 3
    
    # Process pairs of backticks
    i = 0
    while i < len(backtick_positions) - 1:
        start_pos = backtick_positions[i]
        end_pos = backtick_positions[i + 1]
        
        # Extract the content between backticks
        code_section = markdown_content[start_pos+3:end_pos]
        
        # Check if there's a language specifier on the first line
        lines = code_section.split('\n', 1)
        if len(lines) > 1:
            # Check if first line is a language specifier (no spaces, common language names)
            first_line = lines[0].strip()
            if first_line and not ' ' in first_line and len(first_line) < 20:
                language = first_line
                code_content = lines[1].strip() if len(lines) > 1 else ""
            else:
                language = ""
                code_content = code_section.strip()
        else:
            language = ""
            code_content = code_section.strip()
        
        # Skip if code block is too short
        if len(code_content) < min_length:
            i += 2  # Move to next pair
            continue
        
        # Extract context before (1000 chars)
        context_start = max(0, start_pos - 1000)
        context_before = markdown_content[context_start:start_pos].strip()
        
        # Extract context after (1000 chars)
        context_end = min(len(markdown_content), end_pos + 3 + 1000)
        context_after = markdown_content[end_pos + 3:context_end].strip()
        
        code_blocks.append({
            'code': code_content,
            'language': language,
            'context_before': context_before,
            'context_after': context_after,
            'full_context': f"{context_before}\n\n{code_content}\n\n{context_after}"
        })
        
        # Move to next pair (skip the closing backtick we just processed)
        i += 2
    
    return code_blocks


async def generate_code_example_summary(code: str, context_before: str, context_after: str) -> str:
    """
    Generate a summary for a code example using its surrounding context.
    Now uses async OpenAI client for better performance.
    
    Args:
        code: The code example
        context_before: Context before the code
        context_after: Context after the code
        
    Returns:
        A summary of what the code example demonstrates
    """
    # Create the prompt
    prompt = f"""<context_before>
{context_before[-500:] if len(context_before) > 500 else context_before}
</context_before>

<code_example>
{code[:1500] if len(code) > 1500 else code}
</code_example>

<context_after>
{context_after[:500] if len(context_after) > 500 else context_after}
</context_after>

Based on the code example and its surrounding context, provide a concise summary (2-3 sentences) that describes what this code example demonstrates and its purpose. Focus on the practical application and key concepts illustrated.
"""
    
    try:
        response = await async_openai_client.chat.completions.create(
            model=openai_config.contextual_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides concise code example summaries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=100
        )
        
        return response.choices[0].message.content.strip() if response.choices[0].message.content else "Code example for demonstration purposes."
    
    except Exception as e:
        logger.warning(f"Error generating code example summary: {e}")
        return "Code example for demonstration purposes."


async def add_code_examples_to_db(
    pool: asyncpg.Pool,
    urls: List[str],
    chunk_numbers: List[int],
    code_examples: List[str],
    summaries: List[str],
    metadatas: List[Dict[str, Any]],
    batch_size: int = 20,
) -> None:
    """Batch-insert code examples into `code_examples` table."""
    if not urls:
        logger.warning("No URLs provided to add_code_examples_to_db")
        return

    unique_urls = list(set(urls))

    async with pool.acquire() as connection, connection.transaction():
        try:
            if unique_urls:
                await connection.execute(
                    "DELETE FROM code_examples WHERE url = ANY($1)", unique_urls
                )
                logger.info(
                    "Deleted existing code examples for %s URLs", len(unique_urls)
                )

            total_processed = 0
            total_items = len(urls)

            for start in range(0, total_items, batch_size):
                end = min(start + batch_size, total_items)

                batch_texts = [
                    f"{code_examples[i]}\n\nSummary: {summaries[i]}" for i in range(start, end)
                ]

                try:
                    embeddings = create_embeddings_batch(batch_texts)
                except EmbeddingError as e:
                    logger.error("Failed to create embeddings for code example batch: %s", e)
                    raise

                batch_data = []
                for offset, embedding in enumerate(embeddings):
                    idx = start + offset
                    parsed_url = urlparse(urls[idx])
                    source_id = parsed_url.netloc or parsed_url.path
                    owner_id = db_config.default_owner_id

                    batch_data.append(
                        (
                            urls[idx],
                            chunk_numbers[idx],
                            code_examples[idx],
                            summaries[idx],
                            json.dumps(metadatas[idx]),
                            source_id,
                            owner_id,
                            json.dumps(embedding),
                        )
                    )

                await connection.executemany(
                    """
                    INSERT INTO code_examples (url, chunk_number, content, summary, metadata, source_id, owner_id, embedding)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """,
                    batch_data,
                )
                total_processed += len(batch_data)
                logger.debug("Inserted batch of %s code examples", len(batch_data))

            logger.info(
                "Successfully inserted %s code examples into database", total_processed
            )
        except Exception as e:
            logger.error("Error during code examples insertion: %s", e)
            raise

async def update_source_info(pool: asyncpg.Pool, source_id: str, summary: str, word_count: int) -> None:
    """
    Update or insert source information in the sources table with proper error handling.
    
    Args:
        pool: Database connection pool
        source_id: The source identifier
        summary: Summary of the source content
        word_count: Total word count for the source
    """
    try:
        async with pool.acquire() as connection:
            owner_id = db_config.default_owner_id
            
            await connection.execute("""
                INSERT INTO sources (source_id, summary, total_word_count, owner_id, updated_at)
                VALUES ($1, $2, $3, $4, NOW())
                ON CONFLICT (source_id) DO UPDATE SET
                    summary = EXCLUDED.summary,
                    total_word_count = EXCLUDED.total_word_count,
                    updated_at = NOW();
            """, source_id, summary, word_count, owner_id)
            
            logger.info(f"Updated source info for: {source_id}")
            
    except Exception as e:
        logger.error(f"Error updating source {source_id}: {e}")
        raise

async def extract_source_summary(source_id: str, content: str, max_length: int = 500) -> str:
    """
    Extract a summary for a source from its content using an LLM.
    Now uses async OpenAI client for better performance.
    
    Args:
        source_id: The source ID (domain)
        content: The content to extract a summary from
        max_length: Maximum length of the summary
        
    Returns:
        A summary string
    """
    # Default summary if we can't extract anything meaningful
    default_summary = f"Content from {source_id}"
    
    if not content or len(content.strip()) == 0:
        return default_summary
    
    # Limit content length to avoid token limits
    truncated_content = content[:25000] if len(content) > 25000 else content
    
    # Create the prompt for generating the summary
    prompt = f"""<source_content>
{truncated_content}
</source_content>

The above content is from the documentation for '{source_id}'. Please provide a concise summary (3-5 sentences) that describes what this library/tool/framework is about. The summary should help understand what the library/tool/framework accomplishes and the purpose.
"""
    
    try:
        # Call the OpenAI API to generate the summary
        response = await async_openai_client.chat.completions.create(
            model=openai_config.contextual_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides concise library/tool/framework summaries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=150
        )
        
        # Extract the generated summary
        summary = response.choices[0].message.content.strip() if response.choices[0].message.content else ""
        
        # Ensure the summary is not too long
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."
            
        return summary
    
    except Exception as e:
        logger.warning(f"Error generating summary with LLM for {source_id}: {e}. Using default summary.")
        return default_summary


async def search_code_examples(
    pool: asyncpg.Pool,
    query: str,
    match_count: int = 10,
    filter_metadata: Optional[Dict[str, Any]] = None,
    source_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Search for code examples using vector similarity."""
    enhanced_query = (
        f"Code example for {query}\n\nSummary: Example code showing {query}"
    )

    try:
        query_embedding = create_embedding(enhanced_query)
    except EmbeddingError as e:
        logger.error("Failed to create query embedding for code examples: %s", e)
        raise

    try:
        async with pool.acquire() as connection:
            result = await connection.fetch(
                "SELECT * FROM match_code_examples($1, $2, $3, $4)",
                json.dumps(query_embedding),
                match_count,
                json.dumps(filter_metadata or {}),
                source_id,
            )
        return [dict(row) for row in result]
    except Exception as e:
        logger.error("Error searching code examples: %s", e)
        return []

@lru_cache(maxsize=128)
def validate_url(url: str) -> bool:
    """
    Validate URL format with caching for performance.
    
    Args:
        url: URL to validate
        
    Returns:
        True if URL is valid
    """
    if not url or not isinstance(url, str):
        return False
    
    url = url.strip()
    if not url:
        return False
    
    # Basic URL validation
    return url.startswith(('http://', 'https://')) and len(url) > 10

async def cleanup_old_data(days: int = 30) -> int:
    """
    Clean up old crawled data beyond specified days.
    
    Args:
        days: Number of days to keep data for
        
    Returns:
        Number of records deleted
    """
    if days <= 0:
        raise ValueError("Days must be positive")
    
    pool = await get_db_pool()
    
    async with pool.acquire() as connection, connection.transaction():
        # Delete old crawled pages and count affected rows
        crawled_pages_result = await connection.fetch("""
            DELETE FROM crawled_pages 
            WHERE created_at < NOW() - INTERVAL '%s days'
            RETURNING id
        """, days)
        crawled_pages_count = len(crawled_pages_result)
        
        # Delete old code examples and count affected rows
        code_examples_result = await connection.fetch("""
            DELETE FROM code_examples 
            WHERE created_at < NOW() - INTERVAL '%s days'
            RETURNING id
        """, days)
        code_examples_count = len(code_examples_result)
        
        # Delete orphaned sources and count affected rows
        sources_result = await connection.fetch("""
            DELETE FROM sources 
            WHERE source_id NOT IN (
                SELECT DISTINCT source_id FROM crawled_pages
                UNION
                SELECT DISTINCT source_id FROM code_examples
            )
            RETURNING source_id
        """)
        sources_count = len(sources_result)
        
        deleted_count = crawled_pages_count + code_examples_count + sources_count
        
        logging.info(f"Cleaned up {deleted_count} old records (crawled_pages: {crawled_pages_count}, code_examples: {code_examples_count}, sources: {sources_count})")
        return deleted_count