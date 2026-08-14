"""Build a ChromaDB index from the passage corpus.

Run from the repo root:

    python data/build_index.py

This embeds every passage with all-MiniLM-L6-v2 and stores it in a persistent
Chroma collection that the retriever reads at query time.
"""
import sys
from pathlib import Path

# Make the modules in src/ importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import chromadb  # noqa: E402

from config import CHROMA_DIR, COLLECTION_NAME  # noqa: E402
from corpus_loader import load_corpus  # noqa: E402
from embeddings import embed_texts  # noqa: E402


def build_index():
    passages = load_corpus()
    print(f"Loaded {len(passages)} passages.")

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Rebuild from scratch so re-running is idempotent.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    # Embed "title. text" for slightly better retrieval, but store just the text.
    embed_input = [
        f"{p['title']}. {p['text']}" if p["title"] else p["text"] for p in passages
    ]
    print("Embedding passages (first run downloads the model)...")
    vectors = embed_texts(embed_input)

    collection.add(
        ids=[p["id"] for p in passages],
        documents=[p["text"] for p in passages],
        metadatas=[{"title": p["title"]} for p in passages],
        embeddings=vectors,
    )
    print(f"Indexed {collection.count()} passages into '{COLLECTION_NAME}' at {CHROMA_DIR}")


if __name__ == "__main__":
    build_index()
