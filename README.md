<h1 align="center">Crawl4AI RAG MCP Server</h1>

<p align="center">
  <em>Web Crawling and RAG Capabilities for AI Agents and AI Coding Assistants</em>
</p>

A powerful implementation of the [Model Context Protocol (MCP)](https://modelcontextprotocol.io) integrated with [Crawl4AI](https://crawl4ai.com) and [Supabase](https://supabase.com/) for providing AI agents and AI coding assistants with advanced web crawling and RAG capabilities.

With this MCP server, you can <b>scrape anything</b> and then <b>use that knowledge anywhere</b> for RAG.

The primary goal is to bring this MCP server into [Archon](https://github.com/coleam00/Archon) as I evolve it to be more of a knowledge engine for AI coding assistants to build AI agents. This first version of the Crawl4AI/RAG MCP server will be improved upon greatly soon, especially making it more configurable so you can use different embedding models and run everything locally with Ollama.

Consider this GitHub repository a testbed, hence why I haven't been super actively address issues and pull requests yet. I certainly will though as I bring this into Archon V2!

## Overview

This MCP server provides tools that enable AI agents to crawl websites, store content in a vector database (Supabase), and perform RAG over the crawled content. It follows the best practices for building MCP servers based on the [Mem0 MCP server template](https://github.com/coleam00/mcp-mem0/) I provided on my channel previously.

The server includes several advanced RAG strategies that can be enabled to enhance retrieval quality:
- **Contextual Embeddings** for enriched semantic understanding
- **Hybrid Search** combining vector and keyword search
- **Agentic RAG** for specialized code example extraction
- **Reranking** for improved result relevance using cross-encoder models
- **Knowledge Graph** for AI hallucination detection and repository code analysis

See the [Configuration section](#configuration) below for details on how to enable and configure these strategies.

## Vision

The Crawl4AI RAG MCP server is just the beginning. Here's where we're headed:

1. **Integration with Archon**: Building this system directly into [Archon](https://github.com/coleam00/Archon) to create a comprehensive knowledge engine for AI coding assistants to build better AI agents.

2. **Multiple Embedding Models**: Expanding beyond OpenAI to support a variety of embedding models, including the ability to run everything locally with Ollama for complete control and privacy.

3. **Advanced RAG Strategies**: Implementing sophisticated retrieval techniques like contextual retrieval, late chunking, and others to move beyond basic "naive lookups" and significantly enhance the power and precision of the RAG system, especially as it integrates with Archon.

4. **Enhanced Chunking Strategy**: Implementing a Context 7-inspired chunking approach that focuses on examples and creates distinct, semantically meaningful sections for each chunk, improving retrieval precision.

5. **Performance Optimization**: Increasing crawling and indexing speed to make it more realistic to "quickly" index new documentation to then leverage it within the same prompt in an AI coding assistant.

## Features

- **Smart URL Detection**: Automatically detects and handles different URL types (regular webpages, sitemaps, text files)
- **Recursive Crawling**: Follows internal links to discover content
- **Parallel Processing**: Efficiently crawls multiple pages simultaneously
- **Content Chunking**: Intelligently splits content by headers and size for better processing
- **Vector Search**: Performs RAG over crawled content, optionally filtering by data source for precision
- **Source Retrieval**: Retrieve sources available for filtering to guide the RAG process

## Tools

The server provides essential web crawling and search tools:

### Core Tools (Always Available)

1. **`crawl_single_page`**: Quickly crawl a single web page and store its content in the vector database
2. **`smart_crawl_url`**: Intelligently crawl a full website based on the type of URL provided (sitemap, llms-full.txt, or a regular webpage that needs to be crawled recursively)
3. **`get_available_sources`**: Get a list of all available sources (domains) in the database
4. **`perform_rag_query`**: Search for relevant content using semantic search with optional source filtering

### Conditional Tools

5. **`search_code_examples`** (requires `USE_AGENTIC_RAG=true`): Search specifically for code examples and their summaries from crawled documentation. This tool provides targeted code snippet retrieval for AI coding assistants.

### Knowledge Graph Tools (requires `USE_KNOWLEDGE_GRAPH=true`, see below)

6. **`parse_github_repository`**: Parse a GitHub repository into a Neo4j knowledge graph, extracting classes, methods, functions, and their relationships for hallucination detection
7. **`check_ai_script_hallucinations`**: Analyze Python scripts for AI hallucinations by validating imports, method calls, and class usage against the knowledge graph
8. **`query_knowledge_graph`**: Explore and query the Neo4j knowledge graph with commands like `repos`, `classes`, `methods`, and custom Cypher queries

## Prerequisites

- [Docker/Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [Python 3.12+](https://www.python.org/downloads/) if running the MCP server directly through uv
- [OpenAI API key](https://platform.openai.com/api-keys) (for generating embeddings)
- [Neo4j](https://neo4j.com/) (optional, for knowledge graph functionality) - see [Knowledge Graph Setup](#knowledge-graph-setup) section

## Installation & Setup

The recommended way to run the server is with Docker Compose, which orchestrates the MCP server, PostgreSQL database, and Neo4j.

> **🔒 Security Note**: This version includes important security improvements including Docker secrets management, database access controls, and non-root container execution. See [FIXES_IMPLEMENTED.md](FIXES_IMPLEMENTED.md) for complete details.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/coleam00/mcp-crawl4ai-rag.git
    cd mcp-crawl4ai-rag
    ```

2.  **Create and configure your secrets (Recommended for Production):**
    ```bash
    # Create secrets directory
    mkdir -p secrets
    
    # Set your credentials (use strong passwords for production)
    echo "your_postgres_username" > secrets/postgres_user.txt
    echo "your_secure_postgres_password" > secrets/postgres_password.txt
    echo "your_secure_neo4j_password" > secrets/neo4j_password.txt
    
    # Secure the secrets directory
    chmod 600 secrets/*.txt
    ```

3.  **Create and configure your `.env` file:**
    Create a `.env` file in the project root by copying the example:
    ```bash
    cp .env.example .env
    ```
    Now, edit the `.env` file and fill in your `OPENAI_API_KEY`. For production deployments, avoid putting passwords in the `.env` file and use the secrets method above.

4.  **Run with Docker Compose:**
    ```bash
    docker-compose up --build
    ```
    This command will build the MCP server image, start all the services, and automatically set up the PostgreSQL database with the necessary tables and functions from `crawled_pages.sql`.

## Migration to 1024-Dimensional Embeddings

If you're upgrading from a previous version that used 1536-dimensional embeddings, you'll need to run a migration to ensure compatibility with the current BGE-M3 embedding model (which natively produces 1024-dimensional vectors).

### Quick Migration (Recommended)

**⚠️ This will delete all existing crawled data**

```bash
# Connect to your PostgreSQL database and run the migration
psql -h localhost -U your_postgres_user -d crawl4rag -f migrate_to_1024_dimensions.sql
```

Or if using Docker:
```bash
docker-compose exec postgres psql -U your_postgres_user -d crawl4rag -f /app/migrate_to_1024_dimensions.sql
```

### Verify Migration

After running the migration, verify everything is working correctly:

```bash
# Run the verification script
python verify_embedding_dimensions.py
```

This script will:
- ✅ Test that embeddings are generated with 1024 dimensions
- ✅ Verify database schema is correctly configured
- ✅ Test embedding serialization/storage

### What Changed

- **Embedding dimensions**: Changed from 1536 → 1024 to match BGE-M3 model output
- **Database schema**: Updated `vector(1536)` → `vector(1024)` in both `crawled_pages` and `code_examples` tables
- **Embedding generation**: Added dimension normalization to ensure consistent 1024-dimensional output

The system will now work optimally with the BGE-M3 embedding model and provide better performance with the correct vector dimensions.

## Configuration

Your `.env` file controls the server configuration.

```
# MCP Server Configuration
HOST=0.0.0.0
PORT=8051
TRANSPORT=sse

# OpenAI API Key (required for contextual embeddings)
OPENAI_API_KEY=your_openai_api_key

# Embedding configuration (for /v1/embeddings endpoint)
# Must end with a trailing slash if pointing at a custom OpenAI-compatible server (e.g., LM Studio, Ollama, Groq)
EMBEDDING_URL=http://localhost:1234/v1/
# The model identifier served by the embedding endpoint above
EMBEDDING_MODEL=text-embedding-bge-m3

# LLM used for summaries and contextual embeddings (chat-completion)
CONTEXTUAL_MODEL=gpt-4o-mini

# RAG Strategies (set to "true" or "false", default to "false")
USE_CONTEXTUAL_EMBEDDINGS=false
USE_HYBRID_SEARCH=false
USE_AGENTIC_RAG=false
USE_RERANKING=false
USE_KNOWLEDGE_GRAPH=false

# Database Configuration (Docker secrets recommended for production)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=CHANGEME
POSTGRES_PASSWORD=CHANGEME
POSTGRES_DB=crawl4rag

# Database Performance Tuning
DB_MIN_CONNECTIONS=5
DB_MAX_CONNECTIONS=20

# OpenAI Configuration
OPENAI_MAX_RETRIES=3
OPENAI_RETRY_DELAY=1.0

# Performance Tuning
CODE_PROCESSING_BATCH_SIZE=5

# Neo4j Configuration (required for knowledge graph functionality)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=CHANGEME

# For production with Docker secrets, use these instead:
# POSTGRES_USER_FILE=/run/secrets/postgres_user
# POSTGRES_PASSWORD_FILE=/run/secrets/postgres_password
# NEO4J_PASSWORD_FILE=/run/secrets/neo4j_password
```

### RAG Strategy Options

The Crawl4AI RAG MCP server supports four powerful RAG strategies that can be enabled independently:

#### 1. **USE_CONTEXTUAL_EMBEDDINGS**
When enabled, this strategy enhances each chunk's embedding with additional context from the entire document. The system passes both the full document and the specific chunk to an LLM (configured via `CONTEXTUAL_MODEL`) to generate enriched context that gets embedded alongside the chunk content.

- **When to use**: Enable this when you need high-precision retrieval where context matters, such as technical documentation where terms might have different meanings in different sections.
- **Trade-offs**: Slower indexing due to LLM calls for each chunk, but significantly better retrieval accuracy.
- **Cost**: Additional LLM API calls during indexing.

#### 2. **USE_HYBRID_SEARCH**
Combines traditional keyword search with semantic vector search to provide more comprehensive results. The system performs both searches in parallel and intelligently merges results, prioritizing documents that appear in both result sets.

- **When to use**: Enable this when users might search using specific technical terms, function names, or when exact keyword matches are important alongside semantic understanding.
- **Trade-offs**: Slightly slower search queries but more robust results, especially for technical content.
- **Cost**: No additional API costs, just computational overhead.

#### 3. **USE_AGENTIC_RAG**
Enables specialized code example extraction and storage. When crawling documentation, the system identifies code blocks (≥300 characters), extracts them with surrounding context, generates summaries, and stores them in a separate vector database table specifically designed for code search.

- **When to use**: Essential for AI coding assistants that need to find specific code examples, implementation patterns, or usage examples from documentation.
- **Trade-offs**: Significantly slower crawling due to code extraction and summarization, requires more storage space.
- **Cost**: Additional LLM API calls for summarizing each code example.
- **Benefits**: Provides a dedicated `search_code_examples` tool that AI agents can use to find specific code implementations.

#### 4. **USE_RERANKING**
Applies cross-encoder reranking to search results after initial retrieval. Uses a lightweight cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to score each result against the original query, then reorders results by relevance.

- **When to use**: Enable this when search precision is critical and you need the most relevant results at the top. Particularly useful for complex queries where semantic similarity alone might not capture query intent.
- **Trade-offs**: Adds ~100-200ms to search queries depending on result count, but significantly improves result ordering.
- **Cost**: No additional API costs - uses a local model that runs on CPU.
- **Benefits**: Better result relevance, especially for complex queries. Works with both regular RAG search and code example search.

#### 5. **USE_KNOWLEDGE_GRAPH**
Enables AI hallucination detection and repository analysis using Neo4j knowledge graphs. When enabled, the system can parse GitHub repositories into a graph database and validate AI-generated code against real repository structures. (NOT fully compatible with Docker yet, I'd recommend running through uv)

- **When to use**: Enable this for AI coding assistants that need to validate generated code against real implementations, or when you want to detect when AI models hallucinate non-existent methods, classes, or incorrect usage patterns.
- **Trade-offs**: Requires Neo4j setup and additional dependencies. Repository parsing can be slow for large codebases, and validation requires repositories to be pre-indexed.
- **Cost**: No additional API costs for validation, but requires Neo4j infrastructure (can use free local installation or cloud AuraDB).
- **Benefits**: Provides three powerful tools: `parse_github_repository` for indexing codebases, `check_ai_script_hallucinations` for validating AI-generated code, and `query_knowledge_graph` for exploring indexed repositories.

You can now tell the AI coding assistant to add a Python GitHub repository to the knowledge graph like:

"Add https://github.com/pydantic/pydantic-ai.git to the knowledge graph"

Make sure the repo URL ends with .git.

You can also have the AI coding assistant check for hallucinations with scripts it just created, or you can manually run the command:

```
python knowledge_graphs/ai_hallucination_detector.py [full path to your script to analyze]
```

### Recommended Configurations

**For general documentation RAG:**
```
USE_CONTEXTUAL_EMBEDDINGS=false
USE_HYBRID_SEARCH=true
USE_AGENTIC_RAG=false
USE_RERANKING=true
```

**For AI coding assistant with code examples:**
```
USE_CONTEXTUAL_EMBEDDINGS=true
USE_HYBRID_SEARCH=true
USE_AGENTIC_RAG=true
USE_RERANKING=true
USE_KNOWLEDGE_GRAPH=false
```

**For AI coding assistant with hallucination detection:**
```
USE_CONTEXTUAL_EMBEDDINGS=true
USE_HYBRID_SEARCH=true
USE_AGENTIC_RAG=true
USE_RERANKING=true
USE_KNOWLEDGE_GRAPH=true
```

**For fast, basic RAG:**
```
USE_CONTEXTUAL_EMBEDDINGS=false
USE_HYBRID_SEARCH=true
USE_AGENTIC_RAG=false
USE_RERANKING=false
USE_KNOWLEDGE_GRAPH=false
```

## Running the Server

### Using Docker

```bash
docker run --env-file .env -p 8051:8051 mcp/crawl4ai-rag
```

### Using Python

```bash
uv run src/crawl4ai_mcp.py
```

The server will start and listen on the configured host and port.

## Integration with MCP Clients

### SSE Configuration

Once you have the server running with SSE transport, you can connect to it using this configuration:

```json
{
  "mcpServers": {
    "crawl4ai-rag": {
      "transport": "sse",
      "url": "http://localhost:8051/sse"
    }
  }
}
```

> **Note for Windsurf users**: Use `serverUrl` instead of `url` in your configuration:
> ```json
> {
>   "mcpServers": {
>     "crawl4ai-rag": {
>       "transport": "sse",
>       "serverUrl": "http://localhost:8051/sse"
>     }
>   }
> }
> ```
>
> **Note for Docker users**: Use `host.docker.internal` instead of `localhost` if your client is running in a different container. This will apply if you are using this MCP server within n8n!

> **Note for Claude Code users**: 
```
claude mcp add-json crawl4ai-rag '{"type":"http","url":"http://localhost:8051/sse"}' --scope user
```

### Stdio Configuration

Add this server to your MCP configuration for Claude Desktop, Windsurf, or any other MCP client:

```json
{
  "mcpServers": {
    "crawl4ai-rag": {
      "command": "python",
      "args": ["path/to/crawl4ai-mcp/src/crawl4ai_mcp.py"],
      "env": {
        "TRANSPORT": "stdio",
        "OPENAI_API_KEY": "your_openai_api_key",
        "POSTGRES_HOST": "localhost",
        "POSTGRES_PORT": "5432", 
        "POSTGRES_USER": "postgres",
        "POSTGRES_PASSWORD": "postgres",
        "POSTGRES_DB": "crawl4rag",
        "USE_KNOWLEDGE_GRAPH": "false",
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "your_neo4j_password"
      }
    }
  }
}
```

### Docker with Stdio Configuration

```json
{
  "mcpServers": {
    "crawl4ai-rag": {
      "command": "docker",
      "args": ["run", "--rm", "-i", 
               "-e", "TRANSPORT", 
               "-e", "OPENAI_API_KEY", 
               "-e", "POSTGRES_HOST",
               "-e", "POSTGRES_PORT",
               "-e", "POSTGRES_USER",
               "-e", "POSTGRES_PASSWORD",
               "-e", "POSTGRES_DB",
               "-e", "USE_KNOWLEDGE_GRAPH",
               "-e", "NEO4J_URI",
               "-e", "NEO4J_USER",
               "-e", "NEO4J_PASSWORD",
               "mcp/crawl4ai"],
      "env": {
        "TRANSPORT": "stdio",
        "OPENAI_API_KEY": "your_openai_api_key",
        "POSTGRES_HOST": "localhost",
        "POSTGRES_PORT": "5432",
        "POSTGRES_USER": "postgres", 
        "POSTGRES_PASSWORD": "postgres",
        "POSTGRES_DB": "crawl4rag",
        "USE_KNOWLEDGE_GRAPH": "false",
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "your_neo4j_password"
      }
    }
  }
}
```

## Knowledge Graph Architecture

The knowledge graph system stores repository code structure in Neo4j with the following components:

### Core Components (`knowledge_graphs/` folder):

- **`parse_repo_into_neo4j.py`**: Clones and analyzes GitHub repositories, extracting Python classes, methods, functions, and imports into Neo4j nodes and relationships
- **`ai_script_analyzer.py`**: Parses Python scripts using AST to extract imports, class instantiations, method calls, and function usage
- **`knowledge_graph_validator.py`**: Validates AI-generated code against the knowledge graph to detect hallucinations (non-existent methods, incorrect parameters, etc.)
- **`hallucination_reporter.py`**: Generates comprehensive reports about detected hallucinations with confidence scores and recommendations
- **`query_knowledge_graph.py`**: Interactive CLI tool for exploring the knowledge graph (functionality now integrated into MCP tools)

### Knowledge Graph Schema:

The Neo4j database stores code structure as:

**Nodes:**
- `Repository`: GitHub repositories
- `File`: Python files within repositories  
- `Class`: Python classes with methods and attributes
- `Method`: Class methods with parameter information
- `Function`: Standalone functions
- `Attribute`: Class attributes

**Relationships:**
- `Repository` -[:CONTAINS]-> `File`
- `File` -[:DEFINES]-> `Class`
- `File` -[:DEFINES]-> `Function`
- `Class` -[:HAS_METHOD]-> `Method`
- `Class` -[:HAS_ATTRIBUTE]-> `Attribute`

### Workflow:

1. **Repository Parsing**: Use `parse_github_repository` tool to clone and analyze open-source repositories
2. **Code Validation**: Use `check_ai_script_hallucinations` tool to validate AI-generated Python scripts
3. **Knowledge Exploration**: Use `query_knowledge_graph` tool to explore available repositories, classes, and methods

## Security Considerations

> **🔒 Important**: This version includes significant security improvements:
> - Docker secrets for credential management
> - Database Row Level Security (RLS) with proper authorization
> - Non-root container execution
> - Input validation and parameterized queries
> - Comprehensive error handling and logging

For production deployments:
- Always use Docker secrets for credentials
- Configure proper network security policies
- Implement API rate-limiting
- Regular security scanning and updates
- Monitor logs for suspicious activity

## Building Your Own Server

This implementation provides a foundation for building more complex MCP servers with web crawling capabilities. To build your own:

1. Add your own tools by creating methods with the `@mcp.tool()` decorator
2. Create your own lifespan function to add your own dependencies
3. Modify the `utils.py` file for any helper functions you need
4. Extend the crawling capabilities by adding more specialized crawlers

## Troubleshooting

### Common Issues

1. **Database Connection Errors**: Ensure PostgreSQL is running and credentials are correct
2. **Neo4j Connection Issues**: Verify Neo4j is running and accessible at the configured URI
3. **OpenAI API Errors**: Check your API key and rate limits
4. **Docker Permission Issues**: Ensure proper file permissions for secrets directory

### Logs and Monitoring

The system now includes comprehensive structured logging. Check Docker logs for detailed error information:

```bash
docker-compose logs crawl4rag
```

## Performance Tuning

For large-scale deployments, consider:

- Adjusting `DB_MIN_CONNECTIONS` and `DB_MAX_CONNECTIONS` based on load
- Tuning `CODE_PROCESSING_BATCH_SIZE` for memory usage optimization
- Configuring `OPENAI_MAX_RETRIES` and `OPENAI_RETRY_DELAY` for reliability
- Using dedicated database instances for production workloads