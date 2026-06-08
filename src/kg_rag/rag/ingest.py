import json
import logging
import numpy as np
from pathlib import Path
from typing import Literal

from datetime import timedelta
from kg_rag.config import settings
from tqdm_loggable.auto import tqdm
from tqdm_loggable.tqdm_logging import tqdm_logging
from kg_rag.utils.read_json import read_json, load_tickets
from kg_rag.vectorstore.chroma_db import ChromaTicketStore
from kg_rag.vectorstore.neo4j_store import Neo4jTicketStore
from kg_rag.embeddings.embedder import EmbeddingModel
from kg_rag.retrievers.hybrid import HybridRetriever
from kg_rag.rag.transform import build_text, build_metadata

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

tqdm_logging.set_level(logging.INFO)
tqdm_logging.set_log_rate(timedelta(seconds=5))


def _artifact_path() -> Path:
    source = Path(settings.json_file).stem   # e.g. "support_tickets"
    return (
        Path(__file__).resolve().parents[3]
        / "data" / "modified"
        / f"{source}_as_docs.json"
    )


ARTIFACT_PATH = _artifact_path()

hybrid_retriever = HybridRetriever()


# =========================================================
# UTIL
# =========================================================

def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


# =========================================================
# LOAD OR CREATE DOCUMENTS
# =========================================================

def load_or_create_documents() -> list[dict]:
    if ARTIFACT_PATH.exists():
        logger.info("📂 Loading cached documents from %s", ARTIFACT_PATH)
        with open(ARTIFACT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    logger.info("⚙️ Building document corpus...")
    raw = read_json(settings.json_file)
    docs = [
        {
            "id":       t["ticket_id"],
            "document": build_text(t),
            "metadata": build_metadata(t),
        }
        for t in raw
    ]

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACT_PATH, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    logger.info("💾 Saved %d documents → %s", len(docs), ARTIFACT_PATH)
    return docs


# =========================================================
# CHROMA + BM25 INGEST
# =========================================================

def ingest_chroma(batch_size: int = 256) -> None:
    docs = load_or_create_documents()
    if not docs:
        logger.warning("⚠️ No documents found.")
        return

    logger.info("📄 Total documents: %d", len(docs))

    embedder   = EmbeddingModel(backend="openai")
    collection = ChromaTicketStore()

    logger.info("🚀 Embedding and ingesting into ChromaDB...")

    batches = list(range(0, len(docs), batch_size))

    for i in tqdm(batches, desc="  ChromaDB ingest", unit="batch"):
        batch_docs = docs[i : i + batch_size]
        texts      = [d["document"] for d in batch_docs]

        # embed small batch
        batch_embeddings = embedder.embed(texts, batch_size=batch_size)
        batch_embeddings = _normalize(np.array(batch_embeddings))

        # upload immediately — free memory after each batch
        collection.add(
            ids       =[d["id"]       for d in batch_docs],
            documents =[d["document"] for d in batch_docs],
            embeddings=batch_embeddings.tolist(),
            metadatas =[d["metadata"] for d in batch_docs],
        )

        del batch_embeddings  # ← explicitly free memory

    # BM25
    logger.info("🔎 Building BM25 index...")
    hybrid_retriever.index(docs)
    logger.info("✅ Hybrid (ChromaDB + BM25) index ready — %d documents", len(docs))

# =========================================================
# BM25 ONLY
# =========================================================

def build_bm25_index() -> None:
    docs = load_or_create_documents()
    if not docs:
        logger.warning("⚠️ No documents found.")
        return

    logger.info("🔎 Building BM25 index for %d docs...", len(docs))
    hybrid_retriever.bm25.index(docs, force_rebuild=True)
    logger.info("✅ BM25 ready")


# =========================================================
# NEO4J INGEST
# =========================================================

def ingest_neo4j() -> None:
    tickets = load_tickets()
    if not tickets:
        logger.warning("⚠️ No tickets found.")
        return

    logger.info("📄 Total tickets: %d", len(tickets))

    neo4j = Neo4jTicketStore()
    neo4j.driver.verify_connectivity()
    logger.info("✅ Neo4j connection verified")

    neo4j.create_schema()
    neo4j.ingest(tickets)
    neo4j.close()

    logger.info("🎉 Neo4j ingest complete — %d tickets", len(tickets))


# =========================================================
# COMBINED PIPELINE
# =========================================================

def run_pipeline(target: Literal["chroma", "neo4j", "both"] = "both") -> None:
    if target in ("chroma", "both"):
        logger.info("--- ChromaDB + BM25 ---")
        ingest_chroma()

    if target in ("neo4j", "both"):
        logger.info("--- Neo4j ---")
        ingest_neo4j()


# =========================================================
# ENTRYPOINT
# =========================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest tickets into ChromaDB / Neo4j")
    parser.add_argument(
        "--target",
        choices=["chroma", "neo4j", "both"],
        default="both",
        help="Which database(s) to ingest into (default: both)",
    )
    parser.add_argument(
        "--bm25-only",
        action="store_true",
        help="Rebuild only the BM25 index without re-embedding",
    )
    args = parser.parse_args()

    if args.bm25_only:
        build_bm25_index()
    else:
        run_pipeline(target=args.target)