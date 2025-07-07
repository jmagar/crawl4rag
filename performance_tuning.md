# Performance Tuning Guide – Crawl4RAG MCP

This guide collects practical tips for maximising throughput and minimising
latency when crawling, indexing and querying documentation.

---

## PostgreSQL / pgvector

| Setting | Recommended | Why |
|---------|-------------|-----|
| `shared_buffers` | 512-2048 MB (25 % RAM) | Keep hot pages in RAM (vector & metadata). |
| `work_mem` | 64 MB | Enough memory for similarity sort and JSON aggregation. |
| `maintenance_work_mem` | 256 MB | Faster `CREATE INDEX`. |
| `effective_io_concurrency` | 200 (SSD) | Async prefetch for ANN scans. |
| `max_parallel_workers_per_gather` | 4 | Parallel vector similarity on larger result windows. |

### Index parameters

```sql
-- higher `lists` → better recall, slower build & more RAM
CREATE INDEX crawled_pages_embedding_idx
ON crawled_pages USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

Values between **100-200** offer a good recall/latency trade-off for 100 k –
1 M vectors.

### Vacuum / analyse

Schedule `VACUUM (ANALYZE) crawled_pages;` nightly to keep planner stats fresh.

---

## Embedding throughput

### Local server (bge-m3)

* Run on GPU if available (6 GB VRAM≈ 45 k embeds/s).  
* Batch **64** inputs for peak utilisation.
* Use **HTTP keep-alive** (httpx default) and put the server on localhost.

### OpenAI contextual embeddings

* Batch **≤ 8** documents to stay under 16 k token limit.  
* Employ exponential back-off (3 retries, delay 1 s → 4 s → 9 s).

---

## Crawl4AI

| Parameter | Default | Tip |
|-----------|---------|-----|
| `BrowserConfig.headless` | `True` | Keep headless in prod; use `headless=False` only for debugging. |
| `CrawlerRunConfig.cache_mode` | `BYPASS` | Switch to `ENABLED` when re-indexing periodically. |
| `max_concurrent_tasks` | 10 | Increase gradually; each task ≈ 120 MB RAM. |
| `scan_full_page` | `False` | Enable only for infinite-scroll sites – big perf cost. |

---

## FastMCP

* Run with **Uvicorn workers = 1** (I/O bound).  
  Scale **horizontally** via multiple containers behind an nginx LB.
* Enable **gzip** / **brotli** on the LB for large JSON responses.

---

## Hardware sizing cheat-sheet

| Load | CPU | RAM | Storage |
|------|-----|-----|---------|
| Personal dev (≤10 k docs) | 2 vCPU | 8 GB | 10 GB SSD |
| Team server (100 k docs, 5 agents) | 4 vCPU | 16 GB | 50 GB SSD + 20 GB GPU VRAM |
| SaaS (1 M docs) | 8-16 vCPU | 64-128 GB | 250 GB NVMe + A100 40 GB |

---

## Monitoring

* **pg_stat_statements** – inspect slow queries.  
* **Prometheus + Grafana** – CPU / RAM / PG metrics.
* FastMCP emits progress & log events → forward to ELK.

---

Happy crawling! 🎉