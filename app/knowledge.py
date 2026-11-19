"""Knowledge base: an ETL pipeline that loads clinic documents into a vector database,
plus a retriever the agent uses for RAG.

Extract  -> read the .md / .txt files
Transform-> split into overlapping chunks, embed each chunk
Load     -> upsert into Chroma with a stable ID, so re-running never creates duplicates
"""

import hashlib
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class KnowledgeBase:
    def __init__(self, embeddings: Embeddings, persist_dir: str, collection: str = "clinic_docs"):
        self.store = Chroma(
            collection_name=collection,
            embedding_function=embeddings,
            persist_directory=persist_dir,
            collection_metadata={"hnsw:space": "cosine"},
        )

    # ---------- ETL ----------
    @staticmethod
    def extract(docs_dir: str) -> list[Document]:
        docs = []
        for path in sorted(Path(docs_dir).glob("*")):
            if path.suffix.lower() in {".md", ".txt"}:
                text = path.read_text(encoding="utf-8").strip()
                if text:
                    docs.append(Document(page_content=text, metadata={"source": path.name}))
        return docs

    @staticmethod
    def transform(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n## ", "\n\n", "\n", ". ", " "],
        )
        chunks = splitter.split_documents(docs)
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk"] = i
        return chunks

    @staticmethod
    def chunk_id(chunk: Document) -> str:
        """Same source + same text -> same ID. This makes loading idempotent."""
        key = f"{chunk.metadata['source']}::{chunk.page_content}"
        return hashlib.sha256(key.encode()).hexdigest()[:32]

    def load(self, chunks: list[Document]) -> int:
        if not chunks:
            return 0
        ids = [self.chunk_id(c) for c in chunks]
        unique = dict(zip(ids, chunks))  # drop exact duplicate chunks
        self.store.add_documents(list(unique.values()), ids=list(unique.keys()))
        return len(unique)

    def ingest(self, docs_dir: str, chunk_size: int = 500, chunk_overlap: int = 80) -> dict:
        docs = self.extract(docs_dir)
        chunks = self.transform(docs, chunk_size, chunk_overlap)
        loaded = self.load(chunks)
        return {"files": len(docs), "chunks": loaded, "total_in_store": self.count()}

    # ---------- retrieval ----------
    def count(self) -> int:
        return self.store._collection.count()

    def search(self, query: str, k: int = 3) -> list[tuple[Document, float]]:
        """Return the k most similar chunks with a relevance score (higher = more similar)."""
        if self.count() == 0:
            return []
        return self.store.similarity_search_with_relevance_scores(query, k=k)


if __name__ == "__main__":
    # Run the pipeline from the command line:  python -m app.knowledge
    from app.config import settings
    from app.embeddings import get_embeddings

    kb = KnowledgeBase(get_embeddings(settings), settings.vector_dir,
                       collection=f"clinic_docs_{settings.embeddings_provider}")
    print(kb.ingest(settings.docs_dir, settings.chunk_size, settings.chunk_overlap))
