"""End-to-end wiring tests for the vanilla RAG pipeline.

These use injected mock retriever/generator functions, so they run without a
Groq API key or a built ChromaDB index -- they verify the pipeline logic:
retrieve top-k -> build cited context -> generate -> return answer + metadata.

Run:  pytest tests/test_vanilla_rag.py   (or: python tests/test_vanilla_rag.py)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vanilla_rag import vanilla_rag_answer  # noqa: E402

# A tiny fake corpus keyed by nothing in particular; the mock returns it for any query.
FAKE_PASSAGES = [
    {"id": "1", "title": "Hamlet", "text": "Hamlet was written by William Shakespeare."},
    {"id": "2", "title": "Paris", "text": "The Eiffel Tower is in Paris, France."},
]

SAMPLE_QUERIES = [
    "Who wrote the play Hamlet?",
    "Where is the Eiffel Tower?",
    "What is the capital of France?",
]


def _mock_retrieve(query, k=5):
    return FAKE_PASSAGES[:k]


def _make_mock_generate(record):
    def _gen(prompt, system=None, **kwargs):
        record["prompt"] = prompt
        record["system"] = system
        return "A concise cited answer [1]."

    return _gen


def test_returns_answer_and_metadata_for_each_query():
    for q in SAMPLE_QUERIES:
        record = {}
        answer, info = vanilla_rag_answer(
            q, k=2, generate=_make_mock_generate(record), retrieve_fn=_mock_retrieve
        )
        assert isinstance(answer, str) and answer, "answer must be a non-empty string"
        assert info["retrieved"] is True, "vanilla RAG always retrieves"
        assert info["num_retrieved"] == 2, "should retrieve exactly k passages"
        assert info["passages_used"] == 2, "vanilla RAG uses every retrieved passage"
        assert len(info["passages"]) == 2


def test_context_includes_numbered_sources_and_citation_instruction():
    record = {}
    vanilla_rag_answer(
        "Who wrote Hamlet?",
        k=2,
        generate=_make_mock_generate(record),
        retrieve_fn=_mock_retrieve,
    )
    prompt = record["prompt"]
    assert "[1]" in prompt and "[2]" in prompt, "sources should be numbered"
    assert "Cite every source" in prompt, "should instruct the model to cite"
    assert "Shakespeare" in prompt, "retrieved passage text should be in the prompt"


def test_respects_k():
    record = {}
    _, info = vanilla_rag_answer(
        "anything", k=1, generate=_make_mock_generate(record), retrieve_fn=_mock_retrieve
    )
    assert info["num_retrieved"] == 1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASSED: {name}")
    print("All vanilla RAG wiring tests passed.")
