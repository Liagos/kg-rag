import logging
from collections import defaultdict
from kg_rag.retrievers.semantic import SemanticRetriever
from kg_rag.retrievers.bm25 import BM25Retriever

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid Retriever using Reciprocal Rank Fusion (RRF)."""

    def __init__(self, rrf_k: int = 60):
        self.semantic = SemanticRetriever()
        self.bm25     = BM25Retriever()
        self.rrf_k    = rrf_k

    def index(self, documents: list[dict]):
        self.bm25.index(documents)

    def _rrf(self, rank: int) -> float:
        return 1.0 / (self.rrf_k + rank)

    def _doc_weight(self, doc) -> float:
        """Weight by priority and business impact."""
        meta     = doc.metadata if hasattr(doc, "metadata") else {}
        priority = meta.get("priority", "")
        impact   = meta.get("business_impact", "")

        weight = 1.0
        if priority == "critical":
            weight += 0.2
        elif priority == "high":
            weight += 0.1
        if impact == "critical":
            weight += 0.1

        return weight

    def _build_rank_map(self, results) -> dict:
        return {doc.id: rank for rank, doc in enumerate(results)}

    def retrieve(self, query: str, k: int = 5, filters: dict = None):
        if not query or not query.strip():
            return []

        fetch_k = min(max(200, k * 4), 500)

        semantic_results = self.semantic.retrieve(query, fetch_k, filters)
        bm25_results     = self.bm25.retrieve(query, fetch_k, filters)

        semantic_rank = self._build_rank_map(semantic_results)
        bm25_rank     = self._build_rank_map(bm25_results)

        logger.info(
            "Hybrid retrieve — semantic: %d, bm25: %d, union: %d",
            len(semantic_results),
            len(bm25_results),
            len(set(semantic_rank) | set(bm25_rank)),
        )

        scores: dict[str, float] = defaultdict(float)
        docs:   dict[str, object] = {}

        all_doc_ids = set(semantic_rank.keys()) | set(bm25_rank.keys())

        for doc_id in all_doc_ids:
            doc = (
                semantic_results[semantic_rank[doc_id]]
                if doc_id in semantic_rank
                else bm25_results[bm25_rank[doc_id]]
            )
            weight = self._doc_weight(doc)
            score  = 0.0

            if doc_id in semantic_rank:
                score += weight * self._rrf(semantic_rank[doc_id])
            if doc_id in bm25_rank:
                score += weight * self._rrf(bm25_rank[doc_id])

            scores[doc_id] = score
            docs[doc_id]   = doc

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

        return [docs[doc_id] for doc_id, _ in ranked if doc_id in docs]