"""Embedding module using sentence-transformers all-MiniLM-L6-v2.

The model is loaded lazily and cached, so importing this module is cheap and the
(~80 MB) model only downloads the first time an embedding is actually requested.
Embeddings are L2-normalized, so cosine similarity equals a dot product.
"""
from config import EMBED_MODEL

_model = None


def get_model():
    """Load and cache the SentenceTransformer model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def embed_texts(texts):
    """Embed a list of strings -> list of vectors (each a list of floats)."""
    if isinstance(texts, str):
        texts = [texts]
    model = get_model()
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.tolist()


def embed_query(text):
    """Embed a single query string -> one vector (list of floats)."""
    return embed_texts([text])[0]


if __name__ == "__main__":
    vecs = embed_texts(["hello world", "self-reflective retrieval"])
    print(f"Embedded 2 texts, each of dimension {len(vecs[0])}")
