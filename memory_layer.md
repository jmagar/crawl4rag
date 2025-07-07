# Persistent Memory Layer – Design Notes

> This document sketches a **future extension** that adds long-term,
> cross-session memory to the Crawl4RAG MCP server.  It is *not* yet implemented
> in code, but the schema and API below align with FastMCP paradigms so that
> integration will be straightforward.

---

## Motivation

LLM agents often need to **remember** facts across conversations: decisions,
preferences, vectors of past interactions.  Storing those embeddings alongside
the RAG corpus would mix operational concerns and bloat the primary index.
Instead we separate *agent memory* into its own lightweight table.

---

## Schema (`memory_items`)

```sql
CREATE TABLE memory_items (
    id           BIGSERIAL PRIMARY KEY,
    created_at   TIMESTAMPTZ DEFAULT now(),
    user_id      TEXT,            -- optional multi-tenant key
    embedding    VECTOR(1024),
    text         TEXT NOT NULL,
    metadata     JSONB DEFAULT '{}'
);

CREATE INDEX memory_items_embedding_idx ON memory_items
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 50);
```

* `user_id` allows isolating memory per human / agent.  
* `metadata` can store tags such as `source_tool`, `conversation_id`.

---

## FastMCP Tools (planned)

| Tool | Purpose |
|------|---------|
| `memory_store(text, metadata={})` | Embed the text and insert into `memory_items`. |
| `memory_recall(query, k=5)` | Vector search + hybrid keyword search across the table. |
| `memory_prune(days=30)` | Periodic job: delete or compress old items. |

### Example usage

```python
@mcp.tool()
async def memory_store_tool(ctx: Context, text: str, metadata: dict | None = None):
    ...

@mcp.tool()
async def memory_recall_tool(ctx: Context, query: str, k: int = 5):
    ...
```

The recall tool can be chained inside agent prompts so the LLM can decide when
to fetch memories.

---

## Retrieval Strategy

1. Embed the `query` using the same 1024-dim model (bge-m3).  
2. Cosine ANN search (ivfflat).  
3. **Optional** keyword filter (`ILIKE '%foo%'`) for precision.  
4. Cross-encoder re-rank when enabled.

---

## Privacy / Security

* Memory rows reference the `user_id`; ACLs can be added via PostgreSQL RLS.  
* Agents must supply `session_user_id` in the tool args or via `ctx.session`.

---

## Garbage collection

Without pruning, memory will grow unbounded.  Options:

* **Time-based TTL** – delete items older than *N* days.  
* **LRU via score** – keep top-`m` similarity hits and drop others.  
* **Embedding clustering** – merge near-duplicates into centroids.

---

## Next steps

1. Finalise schema & apply migration.  
2. Implement store/recall tools under `crawl4rag_mcp/tools/memory.py`.  
3. Add unit tests & load tests.

---

*Prepared by @your-name · July 2025*