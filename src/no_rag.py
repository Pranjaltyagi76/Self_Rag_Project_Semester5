"""No-RAG baseline: answer from the model's parametric knowledge only.

The lower bound in our three-way comparison. It never retrieves, so any gap
between this and the RAG systems shows what retrieval actually contributes.

Public API:
    no_rag_answer(query) -> (answer, info)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import llm as _llm  # noqa: E402

NO_RAG_SYSTEM = "Answer the question concisely from your own knowledge."


def no_rag_answer(query, generate=None, **kwargs):
    """Answer without any retrieval. Extra kwargs (e.g. k) are ignored."""
    generate = generate or _llm.generate
    answer = generate(query, system=NO_RAG_SYSTEM, max_tokens=256)
    info = {"query": query, "retrieved": False, "passages_used": 0, "passages": []}
    return answer, info


if __name__ == "__main__":
    ans, _ = no_rag_answer("Who wrote the play Hamlet?")
    print(ans)
