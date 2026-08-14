"""Self-RAG orchestrator.

Combines the retriever, the citation-aware generator, and the four reflection
functions into an adaptive, self-critiquing pipeline:

    1. Retrieve?  -> decide whether external knowledge is needed
    2. IsRel      -> keep only relevant retrieved passages
    3. generate a grounded, cited answer
    4. IsSup + IsUse -> verify support/usefulness, regenerate once if weak

All dependencies (generate, retrieve_fn, and the four reflection functions) are
injectable, so the pipeline can be unit tested without an API key or a built index.

Public API:
    self_rag_answer(query, k=5) -> (answer, info)

`info` is a uniform dict (see _make_info) describing the reflection decisions:
retrieved, num_retrieved, passages_used, support, usefulness, regenerated, passages.
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


def _make_info(
    query,
    retrieved,
    num_retrieved=0,
    passages_used=0,
    passages=None,
    support=None,
    usefulness=None,
    regenerated=False,
):
    """Build the uniform metadata dict returned from every pipeline path.

    Keys:
        query          the question asked
        retrieved      whether retrieval was performed (Retrieve token)
        num_retrieved  passages fetched before relevance filtering
        passages_used  passages kept after IsRel filtering and used to answer
        support        IsSup verdict: "fully" | "partially" | "no" | None
        usefulness     IsUse rating 1-5, or None
        regenerated    whether the strict-grounding regeneration ran
        passages       the passages actually used (for citations / the demo)
    """
    return {
        "query": query,
        "retrieved": retrieved,
        "num_retrieved": num_retrieved,
        "passages_used": passages_used,
        "support": support,
        "usefulness": usefulness,
        "regenerated": regenerated,
        "passages": passages or [],
    }


def self_rag_answer(
    query,
    k=5,
    generate=None,
    retrieve_fn=None,
    needs_retrieval=None,
    is_relevant=None,
    is_supported=None,
    usefulness=None,
):
    """Adaptive pipeline: decide whether to retrieve, filter, generate, verify."""
    if needs_retrieval is None:
        from reflect import needs_retrieval
    if is_relevant is None:
        from reflect import is_relevant
    if is_supported is None:
        from reflect import is_supported
    if usefulness is None:
        from reflect import usefulness

    # Step 1 (Retrieve token): skip retrieval when no external knowledge is needed.
    if not needs_retrieval(query, generate=generate):
        answer = _answer_without_context(query, generate)
        return answer, _make_info(query, retrieved=False)

    if retrieve_fn is None:
        # Lazy import so mocked tests don't require chromadb.
        from retriever import retrieve as retrieve_fn

    passages = retrieve_fn(query, k=k)

    # Step 2 (IsRel token): keep only passages judged relevant.
    relevant = [p for p in passages if is_relevant(query, p, generate=generate)]

    # If nothing survived filtering, fall back to parametric knowledge.
    if not relevant:
        answer = _answer_without_context(query, generate)
        return answer, _make_info(
            query, retrieved=True, num_retrieved=len(passages)
        )

    answer = generate_with_context(query, relevant, generate=generate)

    # Step 4 (IsSup + IsUse tokens): verify grounding and usefulness; if the
    # answer is unsupported or not useful, regenerate once with strict grounding.
    support = is_supported(query, answer, relevant, generate=generate)
    use = usefulness(query, answer, generate=generate)
    regenerated = False
    if support == "no" or use < 3:
        answer = generate_with_context(query, relevant, generate=generate, strict=True)
        support = is_supported(query, answer, relevant, generate=generate)
        use = usefulness(query, answer, generate=generate)
        regenerated = True

    info = _make_info(
        query,
        retrieved=True,
        num_retrieved=len(passages),
        passages_used=len(relevant),
        passages=relevant,
        support=support,
        usefulness=use,
        regenerated=regenerated,
    )
    return answer, info


if __name__ == "__main__":
    ans, info = self_rag_answer("Who wrote the play Hamlet?")
    print(f"Answer: {ans}")
    print(f"Info: {info}")
