"""
Utility functions for the Crawl4AI MCP server.
"""
import os
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple
import json
import asyncpg
from urllib.parse import urlparse
import openai
from openai import OpenAI
import re
import time

# -----------------------------------------------------------------------------
# OpenAI client configuration
# -----------------------------------------------------------------------------
# We separate the client used for embeddings (can point to any OpenAI-compatible
# server such as LM Studio) from the default OpenAI client that talks to
# api.openai.com for contextual embeddings.

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Embedding endpoint & model
EMBEDDING_URL = os.getenv("EMBEDDING_URL", "https://api.openai.com/v1/")
if not EMBEDDING_URL.endswith("/"):
    EMBEDDING_URL += "/"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-bge-m3")

# Contextual-embedding (chat) model – still uses OpenAI cloud by default
CONTEXTUAL_MODEL = os.getenv("CONTEXTUAL_MODEL", "gpt-4o-mini")

# Dedicated client for embeddings so we don't have to keep mutating global state
embedding_client = OpenAI(api_key=OPENAI_API_KEY, base_url=EMBEDDING_URL)

# Leave the default openai client pointing at api.openai.com for chat completions
openai.api_key = OPENAI_API_KEY

async def get_db_pool():
    """
    Get an asyncpg connection pool.
    """
    return await asyncpg.create_pool(
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        database=os.getenv("POSTGRES_DB"),
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
    )

def create_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Create embeddings for multiple texts in a single API call.
    
    Args:
        texts: List of texts to create embeddings for
        
    Returns:
        List of embeddings (each embedding is a list of floats)
    """
    if not texts:
        return []
    
    max_retries = 3
    retry_delay = 1.0  # Start with 1 second delay
    
    for retry in range(max_retries):
        try:
            response = embedding_client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            if retry < max_retries - 1:
                print(f"Error creating batch embeddings (attempt {retry + 1}/{max_retries}): {e}")
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"Failed to create batch embeddings after {max_retries} attempts: {e}")
                # Try creating embeddings one by one as fallback
                print("Attempting to create embeddings individually...")
                embeddings = []
                successful_count = 0
                
                for i, text in enumerate(texts):
                    try:
                        individual_response = embedding_client.embeddings.create(
                            model=EMBEDDING_MODEL,
                            input=[text]
                        )
                        embeddings.append(individual_response.data[0].embedding)
                        successful_count += 1
                    except Exception as individual_error:
                        print(f"Failed to create embedding for text {i}: {individual_error}")
                        # Add zero embedding as fallback
                        embeddings.append([0.0] * 1536)
                
                print(f"Successfully created {successful_count}/{len(texts)} embeddings individually")
                return embeddings
    return []

def create_embedding(text: str) -> List[float]:
    """
    Create an embedding for a single text using OpenAI's API.
    
    Args:
        text: Text to create an embedding for
        
    Returns:
        List of floats representing the embedding
    """
    model_choice = CONTEXTUAL_MODEL
    
    try:
        embeddings = create_embeddings_batch([text])
        return embeddings[0] if embeddings else [0.0] * 1536
    except Exception as e:
        print(f"Error creating embedding: {e}")
        # Return empty embedding if there's an error
        return [0.0] * 1536

def generate_contextual_embedding(full_document: str, chunk: str) -> Tuple[str, bool]:
    """
    Generate contextual information for a chunk within a document to improve retrieval.
    
    Args:
        full_document: The complete document text
        chunk: The specific chunk of text to generate context for
        
    Returns:
        Tuple containing:
        - The contextual text that situates the chunk within the document
        - Boolean indicating if contextual embedding was performed
    """
    model_choice = CONTEXTUAL_MODEL
    
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

        # Call the OpenAI API to generate contextual information
        response = openai.chat.completions.create(
            model=model_choice,
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
        print(f"Error generating contextual embedding: {e}. Using original chunk instead.")
        return chunk, False

def process_chunk_with_context(args):
    """
    Process a single chunk with contextual embedding.
    This function is designed to be used with concurrent.futures.
    
    Args:
        args: Tuple containing (url, content, full_document)
        
    Returns:
        Tuple containing:
        - The contextual text that situates the chunk within the document
        - Boolean indicating if contextual embedding was performed
    """
    url, content, full_document = args
    return generate_contextual_embedding(full_document, content)

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
    Add documents to the database table in batches.
    Deletes existing records with the same URLs before inserting to prevent duplicates.
    
    Args:
        pool: asyncpg connection pool
        urls: List of URLs
        chunk_numbers: List of chunk numbers
        contents: List of document contents
        metadatas: List of document metadata
        url_to_full_document: Dictionary mapping URLs to their full document content
        batch_size: Size of each batch for insertion
    """
    # Get unique URLs to delete existing records
    unique_urls = list(set(urls))
    
    # Delete existing records for these URLs in a single operation
    try:
        if unique_urls:
            async with pool.acquire() as connection:
                await connection.execute("DELETE FROM crawled_pages WHERE url = ANY($1)", unique_urls)
    except Exception as e:
        print(f"Batch delete failed: {e}. Trying one-by-one deletion as fallback.")
        # Fallback: delete records one by one
        async with pool.acquire() as connection:
            for url in unique_urls:
                try:
                    await connection.execute("DELETE FROM crawled_pages WHERE url = $1", url)
                except Exception as inner_e:
                    print(f"Error deleting record for URL {url}: {inner_e}")
                    # Continue with the next URL even if one fails
    
    # Check if MODEL_CHOICE is set for contextual embeddings
    use_contextual_embeddings = os.getenv("USE_CONTEXTUAL_EMBEDDINGS", "false") == "true"
    print(f"\n\nUse contextual embeddings: {use_contextual_embeddings}\n\n")
    
    # Process in batches to avoid memory issues
    for i in range(0, len(contents), batch_size):
        batch_end = min(i + batch_size, len(contents))
        
        # Get batch slices
        batch_urls = urls[i:batch_end]
        batch_chunk_numbers = chunk_numbers[i:batch_end]
        batch_contents = contents[i:batch_end]
        batch_metadatas = metadatas[i:batch_end]
        
        # Apply contextual embedding to each chunk if MODEL_CHOICE is set
        if use_contextual_embeddings:
            # Prepare arguments for parallel processing
            process_args = []
            for j, content in enumerate(batch_contents):
                url = batch_urls[j]
                full_document = url_to_full_document.get(url, "")
                process_args.append((url, content, full_document))
            
            # Process in parallel using ThreadPoolExecutor
            contextual_contents = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                # Submit all tasks and collect results
                future_to_idx = {executor.submit(process_chunk_with_context, arg): idx 
                                for idx, arg in enumerate(process_args)}
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result, success = future.result()
                        contextual_contents.append(result)
                        if success:
                            batch_metadatas[idx]["contextual_embedding"] = True
                    except Exception as e:
                        print(f"Error processing chunk {idx}: {e}")
                        # Use original content as fallback
                        contextual_contents.append(batch_contents[idx])
            
            # Sort results back into original order if needed
            if len(contextual_contents) != len(batch_contents):
                print(f"Warning: Expected {len(batch_contents)} results but got {len(contextual_contents)}")
                # Use original contents as fallback
                contextual_contents = batch_contents
        else:
            # If not using contextual embeddings, use original contents
            contextual_contents = batch_contents
        
        # Create embeddings for the entire batch at once
        batch_embeddings = create_embeddings_batch(contextual_contents)
        
        batch_data = []
        for j in range(len(contextual_contents)):
            # Extract metadata fields
            chunk_size = len(contextual_contents[j])
            
            # Extract source_id from URL
            parsed_url = urlparse(batch_urls[j])
            source_id = parsed_url.netloc or parsed_url.path
            
            # Prepare data for insertion
            data = {
                "url": batch_urls[j],
                "chunk_number": batch_chunk_numbers[j],
                "content": contextual_contents[j],  # Store original content
                "metadata": json.dumps({
                    "chunk_size": chunk_size,
                    **batch_metadatas[j]
                }),
                "source_id": source_id,  # Add source_id field
                "embedding": str(batch_embeddings[j])  # Use embedding from contextual content
            }
            
            batch_data.append(data)
        
        # Insert batch into database with retry logic
        max_retries = 3
        retry_delay = 1.0  # Start with 1 second delay
        
        for retry in range(max_retries):
            try:
                async with pool.acquire() as connection:
                    # Prepare the insert statement
                    # Convert dict to columns and values for the query
                    columns = batch_data[0].keys()
                    # The `on_conflict` part is removed because we are deleting first
                    query = f"""
                        INSERT INTO crawled_pages ({', '.join(columns)})
                        VALUES ({', '.join([f'${i+1}' for i in range(len(columns))])})
                    """
                    
                    await connection.executemany(query, [list(d.values()) for d in batch_data])

                break # Success
            except Exception as e:
                if retry < max_retries - 1:
                    print(f"Error inserting batch into DB (attempt {retry + 1}/{max_retries}): {e}")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    # Final attempt failed
                    print(f"Failed to insert batch after {max_retries} attempts: {e}")

async def search_documents(
    pool: asyncpg.Pool, 
    query: str, 
    match_count: int = 10, 
    filter_metadata: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Search for documents in the database using vector similarity.
    """
    query_embedding = create_embedding(query)
    
    try:
        async with pool.acquire() as connection:
            if filter_metadata:
                result = await connection.fetch(
                    'SELECT * FROM match_crawled_pages($1, $2, $3)',
                    str(query_embedding), match_count, json.dumps(filter_metadata)
                )
            else:
                result = await connection.fetch(
                    'SELECT * FROM match_crawled_pages($1, $2)',
                    str(query_embedding), match_count
                )
        return [dict(row) for row in result]
    except Exception as e:
        print(f"Error searching documents: {e}")
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


def generate_code_example_summary(code: str, context_before: str, context_after: str) -> str:
    """
    Generate a summary for a code example using its surrounding context.
    
    Args:
        code: The code example
        context_before: Context before the code
        context_after: Context after the code
        
    Returns:
        A summary of what the code example demonstrates
    """
    model_choice = CONTEXTUAL_MODEL
    
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
        response = openai.chat.completions.create(
            model=model_choice,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides concise code example summaries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=100
        )
        
        return response.choices[0].message.content.strip() if response.choices[0].message.content else "Code example for demonstration purposes."
    
    except Exception as e:
        print(f"Error generating code example summary: {e}")
        return "Code example for demonstration purposes."


async def add_code_examples_to_db(
    pool: asyncpg.Pool,
    urls: List[str],
    chunk_numbers: List[int],
    code_examples: List[str],
    summaries: List[str],
    metadatas: List[Dict[str, Any]],
    batch_size: int = 20
):
    """
    Add code examples to the code_examples table in batches.
    """
    if not urls:
        return
        
    unique_urls = list(set(urls))
    try:
        async with pool.acquire() as connection:
            await connection.execute("DELETE FROM code_examples WHERE url = ANY($1)", unique_urls)
    except Exception as e:
        print(f"Error deleting existing code examples: {e}")

    # Process in batches
    total_items = len(urls)
    for i in range(0, total_items, batch_size):
        batch_end = min(i + batch_size, total_items)
        batch_texts = [f"{code_examples[j]}\n\nSummary: {summaries[j]}" for j in range(i, batch_end)]
        
        embeddings = create_embeddings_batch(batch_texts)
        
        batch_data = []
        for j, embedding in enumerate(embeddings):
            idx = i + j
            parsed_url = urlparse(urls[idx])
            source_id = parsed_url.netloc or parsed_url.path
            
            batch_data.append({
                'url': urls[idx],
                'chunk_number': chunk_numbers[idx],
                'content': code_examples[idx],
                'summary': summaries[idx],
                'metadata': json.dumps(metadatas[idx]),
                'source_id': source_id,
                'embedding': str(embedding)
            })
        
        max_retries = 3
        retry_delay = 1.0
        
        for retry in range(max_retries):
            try:
                async with pool.acquire() as connection:
                    columns = batch_data[0].keys()
                    query = f"""
                        INSERT INTO code_examples ({', '.join(columns)})
                        VALUES ({', '.join([f'${k+1}' for k in range(len(columns))])})
                    """
                    # asyncpg's executemany is what we want here
                    await connection.executemany(query, [list(d.values()) for d in batch_data])
                break
            except Exception as e:
                if retry < max_retries - 1:
                    print(f"Error inserting batch into DB (attempt {retry + 1}/{max_retries}): {e}")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print(f"Failed to insert batch after {max_retries} attempts: {e}")

async def update_source_info(pool: asyncpg.Pool, source_id: str, summary: str, word_count: int):
    """
    Update or insert source information in the sources table.
    """
    try:
        async with pool.acquire() as connection:
            await connection.execute("""
                INSERT INTO sources (source_id, summary, total_word_count, updated_at)
                VALUES ($1, $2, $3, NOW())
                ON CONFLICT (source_id) DO UPDATE SET
                    summary = EXCLUDED.summary,
                    total_word_count = EXCLUDED.total_word_count,
                    updated_at = NOW();
            """, source_id, summary, word_count)
            print(f"Upserted source: {source_id}")
    except Exception as e:
        print(f"Error updating source {source_id}: {e}")

def extract_source_summary(source_id: str, content: str, max_length: int = 500) -> str:
    """
    Extract a summary for a source from its content using an LLM.
    
    This function uses the OpenAI API to generate a concise summary of the source content.
    
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
    
    # Get the model choice from environment variables
    model_choice = CONTEXTUAL_MODEL
    
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
        response = openai.chat.completions.create(
            model=model_choice,
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
        print(f"Error generating summary with LLM for {source_id}: {e}. Using default summary.")
        return default_summary


async def search_code_examples(
    pool: asyncpg.Pool,
    query: str, 
    match_count: int = 10, 
    filter_metadata: Optional[Dict[str, Any]] = None,
    source_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Search for code examples in the database using vector similarity.
    """
    enhanced_query = f"Code example for {query}\\n\\nSummary: Example code showing {query}"
    query_embedding = create_embedding(enhanced_query)
    
    try:
        async with pool.acquire() as connection:
            if source_id and filter_metadata:
                result = await connection.fetch(
                    'SELECT * FROM match_code_examples($1, $2, $3, $4)',
                    str(query_embedding), match_count, json.dumps(filter_metadata), source_id
                )
            elif source_id:
                 result = await connection.fetch(
                    'SELECT * FROM match_code_examples($1, $2, $3)',
                    str(query_embedding), match_count, source_id
                )
            elif filter_metadata:
                result = await connection.fetch(
                    'SELECT * FROM match_code_examples($1, $2, $3)',
                    str(query_embedding), match_count, json.dumps(filter_metadata)
                )
            else:
                result = await connection.fetch(
                    'SELECT * FROM match_code_examples($1, $2)',
                    str(query_embedding), match_count
                )
        return [dict(row) for row in result]
    except Exception as e:
        print(f"Error searching code examples: {e}")
        return []