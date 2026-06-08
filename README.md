# KG-RAG — Jira Ticket Assistant

A RAG (Retrieval-Augmented Generation) system for querying Jira support tickets using **ChromaDB** (semantic + BM25 search) and **Neo4j** (graph database), with a **Streamlit** chat interface.

---

## Architecture

```
            ┌───────────────────────────────────────────────┐
            │                   Streamlit UI                │
            │              http://localhost:8501            │
            └──────────────────────┬────────────────────────┘
                                   │
                          ┌────────▼─────────┐
                          │   RAG Pipeline   │
                          │   (qa.py)        │
                          └──┬─────────────┬─┘
                             │             │
                ┌────────────▼────┐    ┌───▼─────────────┐
                │   ChromaDB      │    │     Neo4j       │
                │ Semantic + BM25 │    │  Graph filters  │
                │ port 8000       │    │  port 7687      │
                └─────────────────┘    └─────────────────┘
```

### Retrieval Modes
| Mode | What runs | Best for |
|---|---|---|
| `Semantic + BM25` | ChromaDB embeddings + BM25 | Vague, semantic queries |
| `Graph (Neo4j)` | Neo4j graph traversal | Structured filters, exact lookups |
| `Semantic + BM25 + Graph` | Both combined | Complex queries needing full context |

---

## Dataset

This repository includes a **9k sample** of the full dataset for demonstration purposes.

The full dataset contains **100k+ Jira support tickets** across multiple products, regions and categories.

To request the full dataset contact:

**Odysseas Liagouris**
📧 [odyliagouris@gmail.com](mailto:odyliagouris@gmail.com)
🐙 [github.com/Liagos](https://github.com/Liagos)

---

## ⏱️ Performance Notes

### First run — ingestion

Ingestion runs once on first startup. Expected times for the **9k sample**:

| Step | Time |
|---|---|
| Building document corpus | ~30 sec |
| OpenAI embeddings | ~3-5 min |
| ChromaDB upload | ~1 min |
| Neo4j ingestion | ~2-3 min |
| BM25 index build | ~30 sec |
| **Total** | **~7-10 min** |

For the full **100k dataset** expect **~45-60 min** total.

> ℹ️ Ingestion only runs once — data persists in Docker volumes across restarts.

### First query — model initialization

The first time you send a question in Streamlit, expect a **30-60 second delay** while the reranker model (`BAAI/bge-reranker-base`) loads into memory. Subsequent queries are fast.

The HuggingFace cache volume in `docker-compose.yml` ensures the model is downloaded once and reused on every restart.

---

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- OpenAI API key
- Jira tickets JSON file (9k sample included, full dataset available on request)

---

## Quick Start

### 1 — Clone the repository

```bash
git clone https://github.com/Liagos/kg-rag.git
cd kg-rag
```

### 2 — Configure environment

```bash
cp .env.example .env
```

Edit `.env` with your values:

```bash
# Dataset — 9k sample included
json_file=data/raw/support_tickets.json

# OpenAI (required)
OPEN_API_KEY=sk-...

# Anthropic (optional — only needed for Claude models)
ANTHROPIC_API_KEY=sk-ant-...

# ChromaDB — leave as-is for Docker
CHROMA_HOST=chromadb
CHROMA_PORT=8000

# Neo4j
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=yourpassword
NEO4J_DATABASE=neo4j

# LLM
llm=gpt-4.1
embedding_model=text-embedding-3-small
```

> ⚠️ Do NOT add inline comments to `.env` — they cause parsing issues.

### 3 — Start the databases

```bash
docker-compose up -d neo4j chromadb
```

Wait for both to be healthy:
```bash
docker-compose ps
```

You should see `healthy` for both `neo4j` and `chromadb`.

### 4 — Build the RAG container

```bash
docker-compose build rag
```

This builds the image without starting it or running ingestion. Run this once after cloning or whenever you update the code.

To rebuild from scratch (no cache):
```bash
docker-compose build --no-cache rag
```
---

## Ingestion

> 💡 **Tip:** Use `docker run -it` for ingestion — the `-it` flag allocates a TTY so you see live progress bars for embeddings and Neo4j ingestion.

### Ingest both databases (recommended)

```bash
docker run -it --rm \
  --network kg-rag_default \
  --env-file .env \
  -e CHROMA_HOST=chromadb \
  -e CHROMA_PORT=8000 \
  -e NEO4J_URI=bolt://neo4j:7687 \
  --entrypoint /app/.venv/bin/python \
  -v $(pwd)/data:/app/data \
  kg-rag-rag:latest \
  -m kg_rag.rag.ingest --target both
```

You will see:
```
--- ChromaDB + BM25 ---
📄 Total documents: 9000
🧠 Generating embeddings...
  Embedding:  45%|████████▌    | 17/35 [01:12<01:22]
🚀 Ingesting into ChromaDB...
  ChromaDB ingest: 100%|████████| 35/35 [00:28<00:00]
🔎 Building BM25 index...
✅ Hybrid (ChromaDB + BM25) index ready

--- Neo4j ---
Ingesting 9000 tickets into Neo4j (18 batches)...
  Neo4j ingestion: 100%|████████| 18/18 [01:45<00:00]
✅ Neo4j ingest complete — 9000 tickets
```

### Ingest ChromaDB only

```bash
docker run -it --rm \
  --network kg-rag_default \
  --env-file .env \
  -e CHROMA_HOST=chromadb \
  -e CHROMA_PORT=8000 \
  --entrypoint /app/.venv/bin/python \
  -v $(pwd)/data:/app/data \
  kg-rag-rag:latest \
  -m kg_rag.rag.ingest --target chroma
```

### Ingest Neo4j only

```bash
docker run -it --rm \
  --network kg-rag_default \
  --env-file .env \
  -e NEO4J_URI=bolt://neo4j:7687 \
  -e CHROMA_HOST=chromadb \
  -e CHROMA_PORT=8000 \
  --entrypoint /app/.venv/bin/python \
  -v $(pwd)/data:/app/data \
  kg-rag-rag:latest \
  -m kg_rag.rag.ingest --target neo4j
```

### Rebuild BM25 index only (no re-embedding)

```bash
docker run -it --rm \
  --network kg-rag_default \
  --env-file .env \
  --entrypoint /app/.venv/bin/python \
  -v $(pwd)/data:/app/data \
  kg-rag-rag:latest \
  -m kg_rag.rag.ingest --bm25-only
```

> ℹ️ Ingestion is idempotent — running it again won't create duplicates. ChromaDB uses upsert and Neo4j uses MERGE.

---

## Start the App

After ingestion is complete start the full stack:

```bash
docker-compose up -d
```

Open the Streamlit UI:
```
http://localhost:8501
```

---

## Verify Ingestion

```bash
# check ChromaDB document count
docker exec rag /app/.venv/bin/python -c "
from kg_rag.vectorstore.chroma_db import ChromaTicketStore
print('ChromaDB documents:', ChromaTicketStore().count())
"

# check Neo4j node count
docker exec rag /app/.venv/bin/python -c "
from kg_rag.vectorstore.neo4j_store import Neo4jTicketStore
s = Neo4jTicketStore()
with s.driver.session() as session:
    r = session.run('MATCH (n) RETURN labels(n)[0] AS label, COUNT(n) AS count ORDER BY count DESC')
    for row in r:
        print(f'{row[\"label\"]}: {row[\"count\"]}')
s.close()
"
```

---

## Re-ingestion

To wipe and start fresh:

```bash
# stop everything and delete volumes
docker-compose down -v

# start databases
docker-compose up -d neo4j chromadb

# re-ingest with progress bars
docker run -it --rm \
  --network kg-rag_default \
  --env-file .env \
  -e CHROMA_HOST=chromadb \
  -e CHROMA_PORT=8000 \
  -e NEO4J_URI=bolt://neo4j:7687 \
  -v $(pwd)/data:/app/data \
  kg-rag-rag:latest \
  /app/.venv/bin/python -m kg_rag.rag.ingest --target both

# start the app
docker-compose up -d rag
```

---

## Neo4j Browser

Open `http://localhost:7474` and login with:
- Username: `neo4j`
- Password: value of `NEO4J_PASSWORD` in your `.env`

Useful queries:
```cypher
// overview
MATCH (n)
RETURN labels(n)[0] AS label, COUNT(n) AS count
ORDER BY count DESC

// latest tickets
MATCH (t:Ticket)-[:ABOUT]->(p:Product)
RETURN t.ticket_id, t.subject, t.priority, t.created_at
ORDER BY t.created_at DESC
LIMIT 10

// see the graph
MATCH (n)-[r]->(m)
RETURN n, r, m
LIMIT 100
```

---

## Evaluation

Generate an eval dataset and compare ChromaDB vs Neo4j answer quality:

```bash
# generate questions from your tickets
docker run -it --rm \
  --network kg-rag_default \
  --env-file .env \
  --entrypoint /app/.venv/bin/python \
  -v $(pwd)/data:/app/data \
  kg-rag-rag:latest \
  -m kg_rag.evaluation.prepare_eval_dataset \
    --source data/raw/support_tickets_sample.json \
    --output data/evaluation/eval_dataset.jsonl

# run evaluation
docker run -it --rm \
  --network kg-rag_default \
  --env-file .env \
  --entrypoint /app/.venv/bin/python \
  -e CHROMA_HOST=chromadb \
  -e NEO4J_URI=bolt://neo4j:7687 \
  -v $(pwd)/data:/app/data \
  kg-rag-rag:latest \
  -m kg_rag.evaluation.qa_eval \
    --dataset data/evaluation/eval_dataset.jsonl \
    --modes chroma neo4j \
    --models gpt-4.1 \
    --judge-model gpt-4.1 \
    --limit 20 \
    --output data/evaluation/results/qa_eval_results.jsonl
```

---

## Local Development (without Docker)

```bash
# install dependencies
uv sync

# update .env for local dev
CHROMA_HOST=localhost
NEO4J_URI=bolt://localhost:7687

# ingest
python -m kg_rag.rag.ingest --target both

# start UI
streamlit run src/kg_rag/app.py

# or CLI
python main.py
```

---

## Project Structure

```
kg-rag/
├── src/kg_rag/
│   ├── app.py                       # Streamlit UI
│   ├── config.py                    # Settings (pydantic)
│   ├── models.py                    # JiraTicket dataclass
│   ├── embeddings/
│   │   └── embedder.py              # OpenAI / local embeddings
│   ├── evaluation/
│   │   ├── prepare_eval_dataset.py  # Generate eval questions
│   │   └── qa_eval.py               # ChromaDB vs Neo4j evaluation
│   ├── query_understanding/
│   │   └── filters.py               # LLM filter extraction
│   ├── rag/
│   │   ├── ingest.py                # Ingestion pipeline
│   │   ├── qa.py                    # Main ask() function
│   │   └── transform.py             # build_text(), build_metadata()
│   ├── retrievers/
│   │   ├── hybrid.py                # RRF fusion (semantic + BM25)
│   │   ├── semantic.py              # ChromaDB retriever
│   │   ├── bm25.py                  # BM25 retriever
│   │   └── reranker.py              # Cross-encoder reranker
│   └── vectorstore/
│       ├── chroma_db.py             # ChromaDB client
│       └── neo4j_store.py           # Neo4j client + queries
├── data/
│   ├── raw/                         # Ticket JSON files (9k sample included)
│   ├── evaluation/                  # Eval datasets
│   └── cached/                      # BM25 index cache
├── Dockerfile
├── Dockerfile.chromadb
├── docker-compose.yml
├── entrypoint.sh
├── .env.example
└── pyproject.toml
```

---

## Environment Variables Reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `json_file` | ✅ | — | Path to tickets JSON file |
| `OPEN_API_KEY` | ✅ | — | OpenAI API key |
| `ANTHROPIC_API_KEY` | ❌ | — | Anthropic API key (Claude models) |
| `CHROMA_HOST` | ✅ | `localhost` | ChromaDB host |
| `CHROMA_PORT` | ✅ | `8000` | ChromaDB port |
| `NEO4J_URI` | ✅ | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USER` | ✅ | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | ✅ | — | Neo4j password |
| `NEO4J_DATABASE` | ✅ | `neo4j` | Neo4j database name |
| `llm` | ✅ | `gpt-4.1` | LLM model for generation |
| `embedding_model` | ✅ | `text-embedding-3-small` | OpenAI embedding model |

---

## Supported Models

**Generation:**
- `gpt-4.1` (default)
- `gpt-4o`
- `claude-sonnet-4-6`
- `claude-opus-4-6`

**Embeddings:**
- `text-embedding-3-small` (default, faster)
- `text-embedding-3-large` (higher quality)

---

## Troubleshooting

**Progress bars not showing**
Use `docker run -it` for ingestion — the `-it` flag enables TTY which makes progress bars visible.

**Neo4j connection refused**
Wait 30-60 seconds after starting — Neo4j takes time to initialize.
Check: `docker-compose logs neo4j`

**ChromaDB empty after restart**
Verify the volume is mounted at `/data`:
`docker exec chromadb ls /data/`

**Rate limit errors during embedding**
The embedder retries automatically with exponential backoff. If persistent reduce batch size by editing `ingest_chroma(batch_size=64)` in `ingest.py`.

**Out of memory during ingestion**
Reduce batch size: edit `ingest_chroma(batch_size=64)` in `ingest.py`.

**App starts but returns no results**
Run the verify commands above. Check `json_file` in `.env` matches the actual file location.

**First query is slow**
Expected — the reranker model loads on first query (~30-60 sec). Subsequent queries are fast. The HuggingFace cache volume handles this on restarts.
