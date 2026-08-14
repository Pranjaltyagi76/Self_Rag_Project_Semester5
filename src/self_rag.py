"""Self-RAG orchestrator.

Combines the retriever, the citation-aware generator, and the four reflection
functions into an adaptive, self-critiquing pipeline:

    1. Retrieve?  -> decide whether external knowledge is needed
    2. IsRel      -> keep only relevant retrieved passages
    3. generate a grounded, cited answer
    4. IsSup + IsUse -> verify support/usefulness, regenerate once if weak

This file is built up step by step; this first version is the skeleton that
retrieves and generates a grounded answer. All dependencies are injectable so
the pipeline can be unit tested without an API key or a built index.

Public API:
    self_rag_answer(query, k=5) -> (answer, info)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generation import generate_with_context  # noqa: E402


def self_rag_answer(query, k=5, generate=None, retrieve_fn=None):
    """Skeleton pipeline: retrieve top-k, then generate a grounded answer."""
    if retrieve_fn is None:
        # Lazy import so mocked tests don't require chromadb.
        from retriever import retrieve as retrieve_fn

    passages = retrieve_fn(query, k=k)
    answer = generate_with_context(query, passages, generate=generate)

    info = {"query": query}
    return answer, info


if __name__ == "__main__":
    ans, info = self_rag_answer("Who wrote the play Hamlet?")
    print(f"Answer: {ans}")
    print(f"Info: {info}")
