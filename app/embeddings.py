"""Embedding models: turn text into vectors so similar meanings sit close together.

"local" is an offline, keyword-based embedding (hashing trick). It needs no API key,
which makes tests and demos run anywhere. "gemini" and "openai" give real semantic embeddings.
"""

import hashlib
import math
import re

from langchain_core.embeddings import Embeddings

from app.config import Settings

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "be", "to", "of", "and", "or", "in", "on", "at", "for",
    "with", "do", "does", "i", "you", "we", "my", "your", "it", "can", "what", "how", "when", "if",
}


class LocalHashingEmbeddings(Embeddings):
    """Bag-of-words vectors using the hashing trick. Not semantic, but fast and offline."""

    def __init__(self, size: int = 512):
        self.size = size

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.size
        words = [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]
        for w in words:
            stem = w[:-1] if w.endswith("s") and len(w) > 3 else w  # tiny plural normalisation
            idx = int(hashlib.md5(stem.encode()).hexdigest(), 16) % self.size
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def get_embeddings(settings: Settings) -> Embeddings:
    provider = settings.embeddings_provider.lower()
    if provider == "local":
        return LocalHashingEmbeddings()
    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(model=settings.gemini_embedding_model)
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=settings.openai_embedding_model)
    raise ValueError(f"Unknown EMBEDDINGS_PROVIDER: {settings.embeddings_provider}")
