"""Shared configuration constants for the Self-RAG project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Where ChromaDB persists its index (gitignored).
CHROMA_DIR = str(ROOT / "chroma_db")

# Name of the Chroma collection holding our passages.
COLLECTION_NAME = "selfrag_corpus"

# Sentence-Transformers model used for embeddings (small, CPU-friendly).
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
