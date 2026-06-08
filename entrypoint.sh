#!/bin/bash
set -e

PYTHON=/app/.venv/bin/python

echo "============================================"
echo "  KG-RAG — Jira Ticket Assistant"
echo "============================================"

# ---------------------------------------------------------------------------
# Wait for Neo4j
# ---------------------------------------------------------------------------
echo "⏳ Waiting for Neo4j..."
until $PYTHON -c "
import os
from neo4j import GraphDatabase
uri      = os.getenv('NEO4J_URI', 'bolt://neo4j:7687')
user     = os.getenv('NEO4J_USER', 'neo4j')
password = os.getenv('NEO4J_PASSWORD', 'yourpassword')
driver   = GraphDatabase.driver(uri, auth=(user, password))
driver.verify_connectivity()
driver.close()
" 2>/dev/null; do
    echo "   Neo4j not ready — retrying in 3s..."
    sleep 3
done
echo "✅ Neo4j ready"

# ---------------------------------------------------------------------------
# Wait for ChromaDB
# ---------------------------------------------------------------------------
echo "⏳ Waiting for ChromaDB..."
until $PYTHON -c "
import os, urllib.request
host = os.getenv('CHROMA_HOST', 'chromadb')
port = os.getenv('CHROMA_PORT', '8000')
urllib.request.urlopen(f'http://{host}:{port}/api/v2/heartbeat')
" 2>/dev/null; do
    echo "   ChromaDB not ready — retrying in 3s..."
    sleep 3
done
echo "✅ ChromaDB ready"

# ---------------------------------------------------------------------------
# Auto-ingest if no data found
# ---------------------------------------------------------------------------
DOC_COUNT=$($PYTHON -c "
from kg_rag.vectorstore.chroma_db import ChromaTicketStore
col = ChromaTicketStore()
print(col.count())
" 2>/dev/null || echo "0")

if [ "$DOC_COUNT" -eq "0" ]; then
    echo ""
    echo "📭 No data found in ChromaDB."

    JSON_FILE=$(printenv json_file || echo "data/raw/support_tickets.json")

    if [ -f "$JSON_FILE" ]; then
        echo "📂 Dataset found at $JSON_FILE — starting ingestion..."
        $PYTHON -m kg_rag.rag.ingest --target both
        echo "✅ Ingestion complete"
    else
        echo "⚠️  No dataset found at $JSON_FILE"
        echo "   Starting app without data — queries will return no results."
    fi
else
    echo "✅ ChromaDB has $DOC_COUNT documents — skipping ingestion"
fi

# ---------------------------------------------------------------------------
# Start Streamlit
# ---------------------------------------------------------------------------
echo ""
echo "🚀 Starting Streamlit..."
echo "   Open http://localhost:8501 in your browser"
echo ""

exec $PYTHON -m streamlit run src/kg_rag/app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false