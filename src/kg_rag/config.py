from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    json_file: str
    # OpenAI
    OPEN_API_KEY: str

    # Anthropic
    ANTHROPIC_API_KEY: str

    # Model
    # llm: str = 'gpt-4.1'
    llm: str

    # Embedding Model
    embedding_model: str

    # Neo4j
    NEO4J_URI:      str
    NEO4J_USER:     str
    NEO4J_PASSWORD: str
    NEO4J_DATABASE: str

    # ChromaDB
    CHROMA_HOST: str
    CHROMA_PORT: int

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
    )


settings = Settings()
