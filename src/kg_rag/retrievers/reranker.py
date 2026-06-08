import logging
from sentence_transformers.cross_encoder import CrossEncoder
from typing import List, Dict, Any
from kg_rag.retrievers.schemas import RetrievedDocument

logger = logging.getLogger(__name__)

# module-level singleton — loaded once on first import
_reranker_model: CrossEncoder | None = None
_reranker_model_name: str = ""


def _get_model(model_name: str) -> CrossEncoder:
    global _reranker_model, _reranker_model_name
    if _reranker_model is None or _reranker_model_name != model_name:
        logger.info("Loading reranker model: %s", model_name)
        _reranker_model      = CrossEncoder(model_name)
        _reranker_model_name = model_name
        logger.info("✅ Reranker model loaded")
    return _reranker_model


class Reranker:

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model_name = model_name
        self.model      = _get_model(model_name)

    def rerank(
        self,
        query: str,
        documents: List[RetrievedDocument],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:

        if not documents or not query or not query.strip():
            return []

        top_k  = min(top_k, len(documents))
        pairs  = [(query, doc.content) for doc in documents]

        logger.info("Reranking %d candidates → top %d", len(documents), top_k)

        scores = self.model.predict(pairs, batch_size=64)

        if hasattr(scores, "tolist"):
            scores = scores.tolist()

        ranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        return [
            {"doc": doc, "score": float(score)}
            for doc, score in ranked[:top_k]
        ]