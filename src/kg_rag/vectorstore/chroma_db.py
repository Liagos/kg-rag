import re
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from kg_rag.config import settings
from pathlib import Path

# use HttpClient when connecting to Docker, PersistentClient for local file
if settings.CHROMA_HOST != "localhost":
    client = chromadb.HttpClient(
        host=settings.CHROMA_HOST,
        port=settings.CHROMA_PORT,
    )
else:
    CHROMA_PATH = Path(__file__).resolve().parents[1] / "vectorstore" / "chroma_store"
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

embedding_function = OpenAIEmbeddingFunction(
    api_key=settings.OPEN_API_KEY,
    model_name=settings.embedding_model,
)

# sanitize collection name
collection_name = re.sub(r"[^a-zA-Z0-9_-]", "_", Path(settings.json_file).stem).lower()

collection = client.get_or_create_collection(
    name=collection_name,
    embedding_function=embedding_function,
    metadata={"hnsw:space": "cosine"},
)


def ChromaTicketStore():
    return collection
