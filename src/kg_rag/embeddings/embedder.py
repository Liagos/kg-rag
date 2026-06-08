import sys
import time
import logging
from datetime import timedelta
from tqdm_loggable.auto import tqdm
from tqdm_loggable.tqdm_logging import tqdm_logging
from openai import OpenAI, RateLimitError
from typing import List, Literal
from sentence_transformers import SentenceTransformer

from kg_rag.config import settings

tqdm_logging.set_level(logging.INFO)
tqdm_logging.set_log_rate(timedelta(seconds=5))

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

BackendType = Literal["openai", "local"]


class EmbeddingModel:
    def __init__(self, backend: BackendType = "openai", model_name: str | None = None):
        self.backend = backend

        if backend == "openai":
            self.client = OpenAI(api_key=settings.OPEN_API_KEY)
            self.model_name = settings.embedding_model or "text-embedding-3-small"

        elif backend == "local":
            self.model_name = model_name or "all-MiniLM-L6-v2"
            self._model = SentenceTransformer(self.model_name)

        else:
            raise ValueError("Unknown backend")

    def _embed_batch_with_retry(
            self,
            batch: List[str],
            max_retries: int = 5,
    ) -> List[List[float]]:
        """Embed a single batch with exponential backoff on rate limit."""
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=batch,
                )
                return [item.embedding for item in response.data]

            except RateLimitError:
                if attempt == max_retries - 1:
                    raise
                wait = 2 ** attempt  # 1s, 2s, 4s, 8s, 16s
                logger.warning(
                    "⚠️ Rate limit hit — waiting %ds (retry %d/%d)",
                    wait, attempt + 1, max_retries,
                )
                time.sleep(wait)

    def embed(self, texts: List[str], batch_size: int = 256) -> List[List[float]]:

        # ----------------------------
        # OPENAI
        # ----------------------------
        if self.backend == "openai":
            all_embeddings = []
            batches = list(range(0, len(texts), batch_size))

            logger.info(
                "🚀 Embedding %d texts using OpenAI (%d batches)...",
                len(texts), len(batches),
            )

            for i in tqdm(batches,
                          desc="  Embedding",
                          unit="batch",
                          dynamic_ncols=True,
                          leave=True,
                          ):
                batch = texts[i: i + batch_size]
                embeddings = self._embed_batch_with_retry(batch)
                all_embeddings.extend(embeddings)

            logger.info("✅ Embedding complete — %d vectors", len(all_embeddings))
            return all_embeddings

        # ----------------------------
        # LOCAL
        # ----------------------------
        elif self.backend == "local":
            logger.info("🚀 Embedding %d texts locally...", len(texts))

            embeddings = self._model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            logger.info("✅ Local embedding complete — %d vectors", len(embeddings))
            return [vec.tolist() for vec in embeddings]

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""

        if self.backend == "openai":
            result = self._embed_batch_with_retry([query])
            return result[0]

        elif self.backend == "local":
            embedding = self._model.encode([query], convert_to_numpy=True)
            return embedding[0].tolist()

        else:
            raise ValueError(f"Unknown backend: {self.backend}")
