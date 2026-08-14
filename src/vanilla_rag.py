"""Vanilla RAG baseline: ALWAYS retrieve a fixed top-k, then answer.

This is the comparison point for Self-RAG. It performs no adaptive-retrieval
decision and no self-critique -- it simply stuffs the top-k passages into the
prompt and generates an answer.

Public API:
    vanilla_rag_answer(query, k=5) -> (answer, info)

`info` is a dict with keys: retrieved, num_passages, passages.
The `generate` and `retrieve_fn` arguments are injectable for testing.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generation import generate_with_context  # noqa: E402


def vanilla_rag_answer(query, k=5, generate=None, retrieve_fn=None):
    """Always retrieve top-k passages and generate a cited answer from them."""
    if retrieve_fn is None:
        # Lazy import so tests can inject a mock retriever without needing chromadb.
        from retriever import retrieve as retrieve_fn

    passages = retrieve_fn(query, k=k)
    answer = generate_with_context(query, passages, generate=generate)

    info = {
        "retrieved": True,
        "num_passages": len(passages),
        "passages": passages,
    }
    return answer, info


if __name__ == "__main__":
    ans, info = vanilla_rag_answer("Who wrote the play Hamlet?")
    print(f"Answer: {ans}")
    print(f"Used {info['num_passages']} passages.")
