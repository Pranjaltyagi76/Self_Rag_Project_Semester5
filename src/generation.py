"""Citation-aware answer generation from retrieved passages.

Shared by both the vanilla RAG baseline and the Self-RAG pipeline. Passages are
numbered [1], [2], ... and the model is asked to cite them inline. In `strict`
mode the model must answer only from the sources (used on Self-RAG's regenerate
step to improve grounding).

Public API:
    generate_with_context(query, passages, strict=False) -> answer_text
    format_sources(passages) -> numbered string
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import llm as _llm  # noqa: E402
from prompts import GEN_PROMPT, GEN_STRICTNESS, GEN_SYSTEM  # noqa: E402


def format_sources(passages):
    """Number passages (dicts with 'title'/'text', or strings) as sources."""
    lines = []
    for i, p in enumerate(passages, start=1):
        if isinstance(p, dict):
            title = p.get("title", "")
            text = p.get("text", "")
            body = f"{title}: {text}" if title else text
        else:
            body = str(p)
        lines.append(f"[{i}] {body}")
    return "\n".join(lines)


def generate_with_context(query, passages, generate=None, strict=False, max_tokens=256):
    """Generate a concise, cited answer grounded in `passages`."""
    generate = generate or _llm.generate
    prompt = GEN_PROMPT.format(
        strictness=GEN_STRICTNESS if strict else "",
        sources=format_sources(passages),
        query=query,
    )
    return generate(prompt, system=GEN_SYSTEM, max_tokens=max_tokens)


if __name__ == "__main__":
    demo = [
        {"title": "Hamlet", "text": "Hamlet is a tragedy written by William Shakespeare."},
    ]
    print(generate_with_context("Who wrote Hamlet?", demo))
