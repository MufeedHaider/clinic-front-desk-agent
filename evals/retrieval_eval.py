"""Measure how often RAG retrieval finds the right document.

Run:  python -m evals.retrieval_eval
Uses whatever EMBEDDINGS_PROVIDER is set (local by default), so you can compare
keyword-based local embeddings with real semantic embeddings (gemini / openai).
"""

import json
import tempfile
from pathlib import Path

from app.config import settings
from app.embeddings import get_embeddings
from app.knowledge import KnowledgeBase
from app.tools import MIN_RELEVANCE

OFF_TOPIC = [
    "Who won the cricket match yesterday?",
    "What is the capital of France?",
    "Write me a poem about the sea",
]


def main() -> None:
    questions = json.loads((Path(__file__).parent / "retrieval_questions.json").read_text())
    kb = KnowledgeBase(get_embeddings(settings), tempfile.mkdtemp(), collection="eval_docs")
    kb.ingest(settings.docs_dir, settings.chunk_size, settings.chunk_overlap)

    hit1 = hit3 = 0
    for item in questions:
        sources = [d.metadata["source"] for d, _ in kb.search(item["q"], k=3)]
        hit1 += sources[:1] == [item["source"]]
        hit3 += item["source"] in sources
        if sources[:1] != [item["source"]]:
            print(f"  miss@1: {item['q']!r} -> {sources[0]} (expected {item['source']})")

    rejected = sum(kb.search(q, k=1)[0][1] < MIN_RELEVANCE for q in OFF_TOPIC)
    n = len(questions)
    print(f"\nEmbeddings: {settings.embeddings_provider}")
    print(f"Hit@1: {hit1}/{n} = {hit1 / n:.0%}")
    print(f"Hit@3: {hit3}/{n} = {hit3 / n:.0%}")
    print(f"Off-topic questions refused: {rejected}/{len(OFF_TOPIC)}")


if __name__ == "__main__":
    main()
