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

import llm as _llm  # noqa: E402
from generation import generate_with_context  # noqa: E402

# System prompt for answering directly from parametric knowledge (no retrieval).
_PARAMETRIC_SYSTEM = "Answer the question concisely from your own knowledge."


def _answer_without_context(query, generate):
    """Generate an answer using only the model's parametric knowledge."""
    generate = generate or _llm.generate
    return generate(query, system=_PARAMETRIC_SYSTEM, max_tokens=256)


def self_rag_answer(
    query,
    k=5,
    generate=None,
    retrieve_fn=None,
    needs_retrieval=None,
    is_relevant=None,
):
    """Adaptive pipeline: decide whether to retrieve, filter, then generate."""
    if needs_retrieval is None:
        from reflect import needs_retrieval
    if is_relevant is None:
        from reflect import is_relevant

    # Step 1 (Retrieve token): skip retrieval when no external knowledge is needed.
    if not needs_retrieval(query, generate=generate):
        answer = _answer_without_context(query, generate)
        return answer, {"query": query, "retrieved": False}

    if retrieve_fn is None:
        # Lazy import so mocked tests don't require chromadb.
        from retriever import retrieve as retrieve_fn

    passages = retrieve_fn(query, k=k)

    # Step 2 (IsRel token): keep only passages judged relevant.
    relevant = [p for p in passages if is_relevant(query, p, generate=generate)]

    # If nothing survived filtering, fall back to parametric knowledge.
    if not relevant:
        answer = _answer_without_context(query, generate)
        return answer, {"query": query, "retrieved": True, "passages_used": 0}

    answer = generate_with_context(query, relevant, generate=generate)

    info = {"query": query, "retrieved": True, "passages_used": len(relevant)}
    return answer, info


if __name__ == "__main__":
    ans, info = self_rag_answer("Who wrote the play Hamlet?")
    print(f"Answer: {ans}")
    print(f"Info: {info}")
