from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Load from `.env` in the project root (optional keys use defaults below)."""

    # Ollama — run `ollama serve` and `ollama pull <OLLAMA_MODEL>` locally.
    OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"
    OLLAMA_MODEL: str = "llama3.2"
    OLLAMA_NUM_PREDICT: int = 2048

    CHROMA_PERSIST_PATH: str = "./chroma_db"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_RESULTS: int = 4

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        extra="ignore",
    )


settings = Settings()
