import os
import json
import re
import logging
import hashlib
from pathlib import Path
from rank_bm25 import BM25Okapi
from kg_rag.config import settings
from kg_rag.retrievers.schemas import RetrievedDocument

logger = logging.getLogger(__name__)


def _cache_path() -> Path:
    """Dynamic cache path based on source file — avoids stale cache on dataset change."""
    source = Path(settings.json_file).stem   # e.g. "support_tickets"
    return (
        Path(__file__).resolve().parents[3]
        / "data" / "cached"
        / f"bm25_cache_{source}.json"
    )


CACHE_PATH = _cache_path()


def tokenize(text: str) -> list[str]:
    text = text.lower()
    text = text.replace("-", " ")
    text = text.replace("/", " ")
    tokens = re.findall(r"[a-z0-9_]+", text)
    return tokens


class BM25Retriever:

    def __init__(self):
        self.documents = []
        self.corpus    = []
        self.bm25      = None
        self._load()

    def _load(self):
        if os.path.exists(CACHE_PATH):
            logger.info("📦 Loading BM25 from disk...")
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.documents = data["documents"]
            self.corpus    = data["corpus"]
            self.bm25      = BM25Okapi(self.corpus)

            cached_count = data.get("count", 0)
            if cached_count != len(self.documents):
                logger.warning(
                    "⚠️ BM25 cache may be stale — cached %d docs but loaded %d. "
                    "Consider rebuilding with --bm25-only.",
                    cached_count,
                    len(self.documents),
                )

            logger.info("✅ BM25 loaded — %d documents", len(self.documents))
        else:
            logger.warning("⚠️ No BM25 cache found — run ingest to build index")

    def _save(self):
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "documents": self.documents,
                    "corpus":    self.corpus,
                    "count":     len(self.documents),
                },
                f,
                ensure_ascii=False,
            )
        logger.info("💾 BM25 saved — %d documents", len(self.documents))

    def _match_filters(self, doc: dict, filters: dict) -> bool:
        if not filters:
            return True

        meta = doc.get("metadata", {})

        for k, v in filters.items():
            meta_val = meta.get(k)

            if isinstance(v, dict):
                if "$gte" in v and (meta_val is None or meta_val < v["$gte"]):
                    return False
                if "$gt"  in v and (meta_val is None or meta_val <= v["$gt"]):
                    return False
                if "$lt"  in v and (meta_val is None or meta_val >= v["$lt"]):
                    return False
                if "$lte" in v and (meta_val is None or meta_val > v["$lte"]):
                    return False

            elif isinstance(v, list):
                if meta_val not in v:
                    return False

            else:
                if meta_val != v:
                    return False

        return True

    def index(self, documents: list[dict], force_rebuild: bool = False):
        if self.bm25 is not None and not force_rebuild:
            logger.info("♻️ BM25 already loaded — skipping rebuild")
            return

        logger.info("🔨 Building BM25 index for %d documents...", len(documents))
        self.documents = documents or []
        self.corpus    = [tokenize(doc["document"]) for doc in self.documents]
        self.bm25      = BM25Okapi(self.corpus)
        self._save()
        logger.info("✅ BM25 index ready")

    def retrieve(self, query: str, k: int = 5, filters: dict = None) -> list[RetrievedDocument]:
        if self.bm25 is None:
            logger.warning("⚠️ BM25 index not loaded — returning empty results")
            return []

        if not query.strip():
            return []

        tokenized_query = tokenize(query)
        scores          = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        results = []

        for i in ranked_indices:
            doc = self.documents[i]

            if not self._match_filters(doc, filters):
                continue

            results.append(
                RetrievedDocument(
                    id=doc["id"],
                    content=doc["document"],
                    metadata=doc.get("metadata", {}),
                    score=float(scores[i]),
                )
            )

            if len(results) == k:
                break

        return results