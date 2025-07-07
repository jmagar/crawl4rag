# Crawl4RAG MCP

A modern, **self-hostable** Retrieval-Augmented Generation (RAG) stack built on

* **[FastMCP 2.x](https://gofastmcp.com/)** – lightweight server framework for Model-Context-Protocol.
* **[Crawl4AI 0.6](https://github.com/unclecode/crawl4ai)** – async, Playwright-powered crawler that outputs clean Markdown and embeddings.
* **PostgreSQL + pgvector** – vector store (1024-dim) for semantic search.
* **Sentence-Transformers Cross-Encoder** – optional result re-ranking.
* Optional **Neo4j knowledge-graph** utilities for hallucination detection.

The project is a full rewrite of
[coleam00/mcp-crawl4ai-rag](https://github.com/coleam00/mcp-crawl4ai-rag) using
latest library capabilities, streamable-HTTP transport and direct Postgres
storage (no Supabase).

---

## Features

| Capability | Details |
|------------|---------|
| Web crawling | `crawl_single_page` & `smart_crawl_url` tools use Crawl4AI 0.6 (`arun_many`, `adeep_crawl`) with intelligent batching and Markdown generation. |
| Vector DB | pgvector (`vector(1024)`) with IVFFLAT ANN indexes. |
| Embeddings | Local HTTP embedding server (`text-embedding-bge-m3`) for bulk chunks, OpenAI (`gpt-4o-mini`) only for *contextual* embeddings. |
| RAG search | Vector similarity + hybrid keyword option + CrossEncoder re-ranking. |
| Knowledge graph (optional) | Import GitHub repos into Neo4j, run queries & validate Python scripts for hallucinations. |
| Transport | **streamable-HTTP** (the recommended FastMCP transport). |
| Modular codebase | `crawl4rag_mcp/` package with small, focused modules. |

---

## Quick-start

```bash
# 1. Clone
$ git clone https://github.com/your-org/crawl4rag-mcp.git
$ cd crawl4rag-mcp

# 2. Python env
$ python -m venv .venv && source .venv/bin/activate

# 3. Install deps
$ pip install -r requirements.txt

# 4. Create Postgres db & enable pgvector
$ createdb crawl4rag
$ psql -d crawl4rag -c 'CREATE EXTENSION IF NOT EXISTS "vector";'

# 5. Apply schema
$ psql -d crawl4rag -f - <<SQL
$(python - <<'PY'
from crawl4rag_mcp.db.vector import SCHEMA_SQL
print(SCHEMA_SQL)
PY
)
SQL

# 6. Run local embedding server (example using Ollama)
$ ollama run bge-base-en-v1  # or start your own server returning OpenAI-style /embeddings

# 7. Copy `.env.example` → `.env` and adjust Postgres / embedding URLs.

# 8. Start the server
$ python crawl4rag-mcp.py  # listens on http://0.0.0.0:8051/mcp/
```

### Try it

```bash
# Crawl a page
curl -X POST http://localhost:8051/mcp/crawl_single_page_tool \
     -d '{"url": "https://docs.python.org/3/"}'

# Ask a question
curl -X POST http://localhost:8051/mcp/perform_rag_query_tool \
     -d '{"query": "what is a context manager?"}'
```

---

## Project structure

```
crawl4rag_mcp/
 ├─ server.py            # FastMCP instance
 ├─ db/                  # Postgres helpers & vector store
 ├─ tools/               # FastMCP tools
 │    ├─ crawl.py
 │    ├─ rag.py
 │    ├─ kg.py           # optional knowledge-graph tools
 │    └─ __init__.py     # tool registrations
 ├─ lifespan.py          # (future) long-lived resources
 └─ …
```

---

## Environment variables

| Var | Default | Purpose |
|-----|---------|---------|
| `HOST` | `0.0.0.0` | IP for FastMCP |
| `PORT` | `8051` | HTTP port |
| `POSTGRES_*` | see `.env` | DB credentials |
| `EMBEDDING_URL` | | Local embedding endpoint (OpenAI-compatible) |
| `EMBEDDING_MODEL` | `text-embedding-bge-m3` | model name passed to server |
| `USE_RERANKING` | `true/false` | enable CrossEncoder rerank |
| `RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | sentence-transformer model |
| `USE_KNOWLEDGE_GRAPH` | `false` | enable Neo4j tools |
| `NEO4J_URI` etc. | | Neo4j credentials |

---

## Further reading

* [`docs/performance_tuning.md`](performance_tuning.md) – Postgres & embedding optimisation tips.
* [`docs/memory_layer.md`](memory_layer.md) – design sketch for a future persistent memory bank.

Pull requests are welcome ✨