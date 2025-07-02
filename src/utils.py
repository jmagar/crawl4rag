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
import openai
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
            max_connections=int(os.getenv("DB_MAX_CONNECTIONS", "10"))
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
            retry_delay=float(os.getenv("OPENAI_RETRY_DELAY", "1.0"))
        )

@dataclass
class SecurityConfig:
    """Security configuration for database and API access."""
    
    neo4j_uri: str
    neo4j_password: str
    enable_rls: bool = True
    
    def __post_init__(self):
        """Validate security configuration."""
        if not self.neo4j_uri:
            raise ValueError("Neo4j URI is required")
        if not self.neo4j_password:
            raise ValueError("Neo4j password is required")

# Global configuration instances
try:
    db_config = DatabaseConfig.from_env()
    openai_config = OpenAIConfig.from_env()
except Exception as e:
    logger.error(f"Configuration error: {e}")
    raise

# -----------------------------------------------------------------------------
# OpenAI client configuration
# -----------------------------------------------------------------------------

# Dedicated client for embeddings
embedding_client = OpenAI(api_key=openai_config.api_key, base_url=openai_config.embedding_url)

# Async client for contextual embeddings
async_openai_client = AsyncOpenAI(api_key=openai_config.api_key)

# Set default client for backwards compatibility
openai.api_key = openai_config.api_key

# Global connection pool (will be initialized once)
_db_pool: Optional[asyncpg.Pool] = None

async def get_db_pool() -> asyncpg.Pool:
    """
    Get or create an asyncpg connection pool with proper resource limits.
    """
    global _db_pool
    
    if _db_pool is None or _db_pool.is_closed():
        try:
            logger.info(f"Creating database connection pool (min={db_config.min_connections}, max={db_config.max_connections})")
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
                    'application_name': 'crawl4ai_mcp_server',
                    'jit': 'off'  # Disable JIT for better connection performance
                }
            )
            logger.info("Database connection pool created successfully")
        except Exception as e:
            logger.error(f"Failed to create database pool: {e}")
            raise
    
    return _db_pool

async def close_db_pool():
    """Close the database connection pool"""
    global _db_pool
    if _db_pool and not _db_pool.is_closed():
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
    
    for retry in range(openai_config.max_retries):
        try:
            logger.debug(f"Creating embeddings for {len(texts)} texts (attempt {retry + 1})")
            response = embedding_client.embeddings.create(
                model=openai_config.embedding_model,
                input=texts
            )
            embeddings = [item.embedding for item in response.data]
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
                        embeddings.append(individual_response.data[0].embedding)
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
    Uses async OpenAI client to avoid blocking.
    
    Args:
        full_document: The complete document text
        chunk: The specific chunk of text to generate context for
        
    Returns:
        Tuple containing:
        - The contextual text that situates the chunk within the document
        - Boolean indicating if contextual embedding was performed
    """
    try:
        # Create the prompt for generating contextual information
        prompt = f"""<document> 
{full_document[:25000]} 
</document>
Here is the chunk we want to situate within the whole document 
<chunk> 
{chunk}
</chunk> 
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""

        # Call the OpenAI API to generate contextual information using async client
        response = await async_openai_client.chat.completions.create(
            model=openai_config.contextual_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides concise contextual information."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=200
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

async def add_documents_to_db(
    pool: asyncpg.Pool,
    urls: List[str], 
    chunk_numbers: List[int],
    contents: List[str], 
    metadatas: List[Dict[str, Any]],
    url_to_full_document: Dict[str, str],
    batch_size: int = 20
) -> None:
    """
    Add documents to the database table in batches with proper transaction handling.
    Uses transactions to prevent race conditions and ensure data consistency.
    
    Args:
        pool: asyncpg connection pool
        urls: List of URLs
        chunk_numbers: List of chunk numbers
        contents: List of document contents
        metadatas: List of document metadata
        url_to_full_document: Dictionary mapping URLs to their full document content
        batch_size: Size of each batch for insertion
    """
    if not urls:
        logger.warning("No URLs provided to add_documents_to_db")
        return
    
    # Get unique URLs to delete existing records
    unique_urls = list(set(urls))
    
    # Check if contextual embeddings are enabled
    use_contextual_embeddings = os.getenv("USE_CONTEXTUAL_EMBEDDINGS", "false") == "true"
    logger.info(f"Using contextual embeddings: {use_contextual_embeddings}")
    
    # Use a single transaction for the entire operation to prevent race conditions
    async with pool.acquire() as connection, connection.transaction():
            try:
                # Delete existing records for these URLs
                if unique_urls:
                    await connection.execute("DELETE FROM crawled_pages WHERE url = ANY($1)", unique_urls)
                    logger.info(f"Deleted existing records for {len(unique_urls)} URLs")
                
                # Process in batches to avoid memory issues
                total_processed = 0
                for i in range(0, len(contents), batch_size):
                    batch_end = min(i + batch_size, len(contents))
                    
                    # Get batch slices
                    batch_urls = urls[i:batch_end]
                    batch_chunk_numbers = chunk_numbers[i:batch_end]
                    batch_contents = contents[i:batch_end]
                    batch_metadatas = metadatas[i:batch_end].copy()  # Make a copy to avoid modifying original
                    
                    # Apply contextual embedding to each chunk if enabled
                    if use_contextual_embeddings:
                        logger.debug(f"Processing contextual embeddings for batch {i//batch_size + 1}")
                        
                        # Process contextual embeddings asynchronously
                        contextual_tasks = []
                        for j, content in enumerate(batch_contents):
                            url = batch_urls[j]
                            full_document = url_to_full_document.get(url, "")
                            task = process_chunk_with_context((url, content, full_document))
                            contextual_tasks.append(task)
                        
                        # Wait for all contextual embeddings to complete
                        contextual_results = await asyncio.gather(*contextual_tasks, return_exceptions=True)
                        
                        # Process results
                        contextual_contents = []
                        for j, result in enumerate(contextual_results):
                            if isinstance(result, Exception):
                                logger.warning(f"Error processing contextual embedding for chunk {j}: {result}")
                                contextual_contents.append(batch_contents[j])
                            else:
                                contextual_text, success = result
                                contextual_contents.append(contextual_text)
                                if success:
                                    batch_metadatas[j]["contextual_embedding"] = True
                    else:
                        # If not using contextual embeddings, use original contents
                        contextual_contents = batch_contents
                    
                    # Create embeddings for the entire batch at once
                    try:
                        batch_embeddings = create_embeddings_batch(contextual_contents)
                        logger.debug(f"Created {len(batch_embeddings)} embeddings for batch")
                    except EmbeddingError as e:
                        logger.error(f"Failed to create embeddings for batch: {e}")
                        raise
                    
                    # Prepare batch data for insertion
                    batch_data = []
                    for j in range(len(contextual_contents)):
                        # Extract metadata fields
                        chunk_size = len(contextual_contents[j])
                        
                        # Extract source_id from URL
                        parsed_url = urlparse(batch_urls[j])
                        source_id = parsed_url.netloc or parsed_url.path
                        
                        # Add owner_id for proper authorization
                        owner_id = "mcp_app_user"  # Default owner for MCP operations
                        
                        # Prepare data for insertion
                        data = (
                            batch_urls[j],                    # url
                            batch_chunk_numbers[j],          # chunk_number
                            batch_contents[j],               # content (store original, not contextual)
                            json.dumps({                     # metadata
                                "chunk_size": chunk_size,
                                **batch_metadatas[j]
                            }),
                            source_id,                       # source_id
                            owner_id,                        # owner_id
                            str(batch_embeddings[j])         # embedding (from contextual content)
                        )
                        
                        batch_data.append(data)
                    
                    # Insert batch into database
                    query = """
                        INSERT INTO crawled_pages (url, chunk_number, content, metadata, source_id, owner_id, embedding)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """
                    
                    await connection.executemany(query, batch_data)
                    total_processed += len(batch_data)
                    logger.debug(f"Inserted batch of {len(batch_data)} records")
                
                logger.info(f"Successfully inserted {total_processed} documents into database")
                
            except Exception as e:
                logger.error(f"Error during document insertion: {e}")
                raise

async def search_documents(
    pool: asyncpg.Pool, 
    query: str, 
    match_count: int = 10, 
    filter_metadata: Optional[Dict[str, Any]] = None,
    source_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Search for documents in the database using vector similarity with proper error handling.
    
    Args:
        pool: Database connection pool
        query: Search query text
        match_count: Maximum number of results to return
        filter_metadata: Optional metadata filter
        source_filter: Optional source ID filter
        
    Returns:
        List of matching documents
        
    Raises:
        EmbeddingError: If query embedding generation fails
    """
    try:
        query_embedding = create_embedding(query)
    except EmbeddingError as e:
        logger.error(f"Failed to create query embedding: {e}")
        raise
    
    try:
        async with pool.acquire() as connection:
            if filter_metadata:
                result = await connection.fetch(
                    'SELECT * FROM match_crawled_pages($1, $2, $3, $4)',
                    str(query_embedding), match_count, json.dumps(filter_metadata), source_filter
                )
            else:
                result = await connection.fetch(
                    'SELECT * FROM match_crawled_pages($1, $2, $3, $4)',
                    str(query_embedding), match_count, json.dumps({}), source_filter
                )
        
        documents = [dict(row) for row in result]
        logger.debug(f"Found {len(documents)} matching documents")
        return documents
        
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
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
    batch_size: int = 20
) -> None:
    """
    Add code examples to the code_examples table in batches with proper transaction handling.
    
    Args:
        pool: Database connection pool
        urls: List of URLs
        chunk_numbers: List of chunk numbers
        code_examples: List of code examples
        summaries: List of summaries for each code example
        metadatas: List of metadata dictionaries
        batch_size: Size of each batch for processing
    """
    if not urls:
        logger.warning("No URLs provided to add_code_examples_to_db")
        return
        
    unique_urls = list(set(urls))
    
    # Use transaction for consistency
    async with pool.acquire() as connection, connection.transaction():
            try:
                # Delete existing code examples for these URLs
                if unique_urls:
                    await connection.execute("DELETE FROM code_examples WHERE url = ANY($1)", unique_urls)
                    logger.info(f"Deleted existing code examples for {len(unique_urls)} URLs")

                # Process in batches
                total_items = len(urls)
                total_processed = 0
                
                for i in range(0, total_items, batch_size):
                    batch_end = min(i + batch_size, total_items)
                    batch_texts = [f"{code_examples[j]}\n\nSummary: {summaries[j]}" for j in range(i, batch_end)]
                    
                    try:
                        embeddings = create_embeddings_batch(batch_texts)
                    except EmbeddingError as e:
                        logger.error(f"Failed to create embeddings for code examples batch: {e}")
                        raise
                    
                    batch_data = []
                    for j, embedding in enumerate(embeddings):
                        idx = i + j
                        parsed_url = urlparse(urls[idx])
                        source_id = parsed_url.netloc or parsed_url.path
                        owner_id = "mcp_app_user"  # Default owner for MCP operations
                        
                        batch_data.append((
                            urls[idx],                       # url
                            chunk_numbers[idx],             # chunk_number
                            code_examples[idx],             # content
                            summaries[idx],                 # summary
                            json.dumps(metadatas[idx]),     # metadata
                            source_id,                      # source_id
                            owner_id,                       # owner_id
                            str(embedding)                  # embedding
                        ))
                    
                    # Insert batch into database
                    query = """
                        INSERT INTO code_examples (url, chunk_number, content, summary, metadata, source_id, owner_id, embedding)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """
                    
                    await connection.executemany(query, batch_data)
                    total_processed += len(batch_data)
                    logger.debug(f"Inserted batch of {len(batch_data)} code examples")
                
                logger.info(f"Successfully inserted {total_processed} code examples into database")
                
            except Exception as e:
                logger.error(f"Error during code examples insertion: {e}")
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
            owner_id = "mcp_app_user"  # Default owner for MCP operations
            
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
    source_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Search for code examples in the database using vector similarity with proper error handling.
    
    Args:
        pool: Database connection pool
        query: Search query text
        match_count: Maximum number of results to return
        filter_metadata: Optional metadata filter
        source_id: Optional source ID filter
        
    Returns:
        List of matching code examples
        
    Raises:
        EmbeddingError: If query embedding generation fails
    """
    enhanced_query = f"Code example for {query}\n\nSummary: Example code showing {query}"
    
    try:
        query_embedding = create_embedding(enhanced_query)
    except EmbeddingError as e:
        logger.error(f"Failed to create query embedding for code search: {e}")
        raise
    
    try:
        async with pool.acquire() as connection:
            # Construct query based on available filters
            if filter_metadata and source_id:
                result = await connection.fetch(
                    'SELECT * FROM match_code_examples($1, $2, $3, $4)',
                    str(query_embedding), match_count, json.dumps(filter_metadata), source_id
                )
            elif source_id:
                result = await connection.fetch(
                    'SELECT * FROM match_code_examples($1, $2, $3, $4)',
                    str(query_embedding), match_count, json.dumps({}), source_id
                )
            elif filter_metadata:
                result = await connection.fetch(
                    'SELECT * FROM match_code_examples($1, $2, $3, $4)',
                    str(query_embedding), match_count, json.dumps(filter_metadata), None
                )
            else:
                result = await connection.fetch(
                    'SELECT * FROM match_code_examples($1, $2, $3, $4)',
                    str(query_embedding), match_count, json.dumps({}), None
                )
        
        code_examples = [dict(row) for row in result]
        logger.debug(f"Found {len(code_examples)} matching code examples")
        return code_examples
        
    except Exception as e:
        logger.error(f"Error searching code examples: {e}")
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
        # Delete old crawled pages
        result1 = await connection.execute("""
            DELETE FROM crawled_pages 
            WHERE created_at < NOW() - INTERVAL '%s days'
        """, days)
        
        # Delete old code examples  
        result2 = await connection.execute("""
            DELETE FROM code_examples 
            WHERE created_at < NOW() - INTERVAL '%s days'
        """, days)
        
        # Delete orphaned sources
        result3 = await connection.execute("""
            DELETE FROM sources 
            WHERE source_id NOT IN (
                SELECT DISTINCT source_id FROM crawled_pages
                UNION
                SELECT DISTINCT source_id FROM code_examples
            )
        """)
        
        deleted_count = (
            int(result1.split()[-1]) + 
            int(result2.split()[-1]) + 
            int(result3.split()[-1])
        )
        
        logging.info(f"Cleaned up {deleted_count} old records")
        return deleted_count