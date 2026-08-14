"""The Self-RAG reflection functions.

Each function turns one of the paper's reflection tokens into a small, parseable
LLM call. They all accept an optional `generate` callable so they can be unit
tested with a mock; by default they use the project's configured LLM.

    Retrieve -> needs_retrieval(query)
    IsRel    -> is_relevant(query, passage)
    IsSup    -> is_supported(query, answer, passages)
    IsUse    -> usefulness(query, answer)
"""
import re

import llm as _llm
from prompts import (
    ISREL_PROMPT,
    ISREL_SYSTEM,
    ISSUP_PROMPT,
    ISSUP_SYSTEM,
    ISUSE_PROMPT,
    ISUSE_SYSTEM,
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


def _format_passages(passages):
    """Number a list of passages (strings or dicts) for the support prompt."""
    return "\n".join(
        f"[{i}] {_passage_text(p)}" for i, p in enumerate(passages, start=1)
    )


def is_supported(query, answer, passages, generate=None):
    """IsSup token: how well the answer is grounded in the passages.

    Returns one of "fully", "partially", or "no".
    """
    generate = _gen(generate)
    out = generate(
        ISSUP_PROMPT.format(
            query=query, passages=_format_passages(passages), answer=answer
        ),
        system=ISSUP_SYSTEM,
        max_tokens=4,
    ).strip().upper()

    if "FULL" in out:
        return "fully"
    if "PARTIAL" in out:
        return "partially"
    return "no"


def usefulness(query, answer, generate=None):
    """IsUse token: integer usefulness rating from 1 to 5.

    Defaults to 3 if the model returns no parseable digit.
    """
    generate = _gen(generate)
    out = generate(
        ISUSE_PROMPT.format(query=query, answer=answer),
        system=ISUSE_SYSTEM,
        max_tokens=4,
    )
    match = re.search(r"[1-5]", out)
    return int(match.group()) if match else 3


if __name__ == "__main__":
    for q in ["Who painted the Mona Lisa?", "Write a short poem about the sea."]:
        print(f"{needs_retrieval(q)!s:>5}  <-  {q}")
