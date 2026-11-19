"""Settings loaded from environment variables (or a .env file)."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    # Which LLM drives the agent: "gemini" or "openai"
    llm_provider: str = os.getenv("LLM_PROVIDER", "gemini")
    agent_mode: str = os.getenv("AGENT_MODE", "single")  # single | multi (supervisor + specialists)
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Which embedding model powers RAG: "local" (offline, keyword-based), "gemini" or "openai"
    embeddings_provider: str = os.getenv("EMBEDDINGS_PROVIDER", "local")
    gemini_embedding_model: str = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
    openai_embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    # Storage
    database_path: str = os.getenv("DATABASE_PATH", str(BASE_DIR / "clinic.db"))
    vector_dir: str = os.getenv("VECTOR_DIR", str(BASE_DIR / "vector_store"))
    docs_dir: str = os.getenv("DOCS_DIR", str(BASE_DIR / "data" / "clinic_docs"))

    # RAG settings
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "80"))
    top_k: int = int(os.getenv("TOP_K", "3"))


settings = Settings()
