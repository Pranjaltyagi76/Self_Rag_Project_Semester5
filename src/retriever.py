"""Dense retriever over the ChromaDB index.

Public API:

    retrieve(query, k=5) -> list of {"id", "title", "text", "score"}

`score` is a cosine similarity in [0, 1] (higher is more relevant). Requires the
index to be built first via `python data/build_index.py`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import chromadb  # noqa: E402

from config import CHROMA_DIR, COLLECTION_NAME  # noqa: E402
from embeddings import embed_query  # noqa: E402

_collection = None


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        try:
            _collection = client.get_collection(COLLECTION_NAME)
        except Exception as e:
            raise RuntimeError(
                f"Chroma collection '{COLLECTION_NAME}' not found at {CHROMA_DIR}.\n"
                "Build the index first:  python data/build_index.py"
            ) from e
    return _collection


def retrieve(query, k=5):
    """Return the top-k passages most similar to `query`."""
    if not query or not query.strip():
        return []
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")

    collection = _get_collection()
    # Asking for more results than the collection holds is an error in some
    # Chroma versions, so cap k at the number of indexed passages.
    k = min(k, collection.count())
    if k == 0:
        return []

    q_vec = embed_query(query)
    res = collection.query(query_embeddings=[q_vec], n_results=k)

    docs = res["documents"][0]
    metas = res["metadatas"][0]
    ids = res["ids"][0]
    dists = res.get("distances", [[None] * len(docs)])[0]

    passages = []
    for i in range(len(docs)):
        dist = dists[i]
        meta = metas[i] or {}
        passages.append(
            {
                "id": ids[i],
                "title": meta.get("title", ""),
                "text": docs[i],
                # Chroma returns squared-L2 distance on normalized vectors;
                # convert to an approximate cosine similarity in [0, 1].
                "score": None if dist is None else max(0.0, 1.0 - dist / 2.0),
            }
        )
    return passages


if __name__ == "__main__":
    query = "Who wrote the play Hamlet?"
    print(f"Query: {query}\n")
    for p in retrieve(query, k=3):
        score = f"{p['score']:.3f}" if p["score"] is not None else "n/a"
        print(f"[{score}] {p['title']}: {p['text'][:80]}...")
