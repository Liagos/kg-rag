from typing import Optional

from kg_rag.vectorstore.chroma_db import ChromaTicketStore
from kg_rag.embeddings.embedder import EmbeddingModel
from kg_rag.retrievers.schemas import RetrievedDocument


def build_chroma_where(filters: Optional[dict]) -> Optional[dict]:
    if not filters:
        return None

    clauses = []

    for key, value in filters.items():

        # range filter — split each operator into its own clause
        if isinstance(value, dict):
            for op, op_val in value.items():
                clauses.append({key: {op: op_val}})

        # list filter
        elif isinstance(value, list):
            clauses.append({
                "$or": [{key: v} for v in value]
            })

        # simple equality
        else:
            clauses.append({key: value})

    if not clauses:
        return None

    if len(clauses) == 1:
        return clauses[0]

    return {"$and": clauses}


class SemanticRetriever:

    def __init__(self):
        self.collection = ChromaTicketStore()
        self.embedder = EmbeddingModel(backend="openai")

    def retrieve(
        self,
        query: str,
        k: int = 5,
        filters: Optional[dict] = None,
    ):

        # Guard against empty queries
        if not query or not query.strip():
            return []

        # Embed query
        query_embedding = self.embedder.embed_query(query)

        # Build valid Chroma filter
        where_clause = build_chroma_where(filters)

        # Query Chroma
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where_clause,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        docs = []

        # Chroma returns nested lists
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for i in range(len(ids)):

            distance = distances[i] if i < len(distances) else 1.0

            # Convert distance -> similarity score
            similarity_score = 1.0 / (1.0 + distance)

            docs.append(
                RetrievedDocument(
                    id=ids[i],
                    content=documents[i],
                    metadata=metadatas[i],
                    score=similarity_score,
                )
            )

        return docs