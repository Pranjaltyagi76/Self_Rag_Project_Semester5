"""The Self-RAG reflection functions.

Each function turns one of the paper's reflection tokens into a small, parseable
LLM call. They all accept an optional `generate` callable so they can be unit
tested with a mock; by default they use the project's configured LLM.

    Retrieve -> needs_retrieval(query)
    IsRel    -> is_relevant(query, passage)
"""
import llm as _llm
from prompts import (
    ISREL_PROMPT,
    ISREL_SYSTEM,
    RETRIEVE_PROMPT,
    RETRIEVE_SYSTEM,
)


def _gen(generate):
    """Return the given generate callable, or the default LLM's."""
    return generate or _llm.generate


def needs_retrieval(query, generate=None):
    """Retrieve token: True if the question needs external factual lookup."""
    generate = _gen(generate)
    out = generate(
        RETRIEVE_PROMPT.format(query=query),
        system=RETRIEVE_SYSTEM,
        max_tokens=4,
    )
    return "YES" in out.strip().upper()


def _passage_text(passage):
    """Accept either a raw string or a {'title', 'text'} dict."""
    if isinstance(passage, dict):
        title = passage.get("title", "")
        text = passage.get("text", "")
        return f"{title}. {text}".strip(". ").strip() if title else text
    return str(passage)


def is_relevant(query, passage, generate=None):
    """IsRel token: True if the passage is relevant to the question."""
    generate = _gen(generate)
    out = generate(
        ISREL_PROMPT.format(query=query, passage=_passage_text(passage)),
        system=ISREL_SYSTEM,
        max_tokens=4,
    )
    return "IRRELEVANT" not in out.strip().upper() and "RELEVANT" in out.strip().upper()


if __name__ == "__main__":
    for q in ["Who painted the Mona Lisa?", "Write a short poem about the sea."]:
        print(f"{needs_retrieval(q)!s:>5}  <-  {q}")
