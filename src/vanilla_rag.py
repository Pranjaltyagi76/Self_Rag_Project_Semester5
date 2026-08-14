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

import llm as _llm  # noqa: E402
from retriever import retrieve  # noqa: E402


def _format_context(passages):
    return "\n".join(f"[{i}] {p['text']}" for i, p in enumerate(passages, start=1))


def vanilla_rag_answer(query, k=5, generate=None, retrieve_fn=None):
    """Always retrieve top-k passages and generate an answer from them."""
    generate = generate or _llm.generate
    retrieve_fn = retrieve_fn or retrieve

    passages = retrieve_fn(query, k=k)
    context = _format_context(passages)

    prompt = (
        "Answer the question using the sources below. Be concise.\n\n"
        f"Sources:\n{context}\n\n"
        f"Question: {query}\nAnswer:"
    )
    answer = generate(prompt, max_tokens=256)

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
