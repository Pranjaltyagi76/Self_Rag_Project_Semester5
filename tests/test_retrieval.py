"""Sanity checks for the retrieval stack (corpus loader + index + retriever).

Run from the repo root:

    python data/build_index.py      # build the index once
    pytest tests/test_retrieval.py  # or: python tests/test_retrieval.py

These are lightweight checks, not a full evaluation: they confirm the corpus
loads, the index builds, and the retriever returns sensibly ranked passages.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data"))

from corpus_loader import load_corpus  # noqa: E402
from build_index import build_index  # noqa: E402
from retriever import retrieve  # noqa: E402


def _ensure_index():
    """Build the index if it is missing so the tests are self-contained."""
    try:
        retrieve("test", k=1)
    except RuntimeError:
        build_index()


def test_corpus_loads():
    corpus = load_corpus()
    assert len(corpus) > 0, "corpus should not be empty"
    for p in corpus:
        assert p["id"] and p["text"], "each passage needs an id and text"


def test_retrieve_returns_k_results():
    _ensure_index()
    results = retrieve("Where is the Eiffel Tower located?", k=3)
    assert len(results) == 3, "should return exactly k passages"
    assert all("text" in r for r in results)


def test_scores_are_sorted_descending():
    _ensure_index()
    results = retrieve("What is photosynthesis?", k=5)
    scores = [r["score"] for r in results if r["score"] is not None]
    assert scores == sorted(scores, reverse=True), "results should be ranked by score"


def test_relevant_passage_is_top_ranked():
    _ensure_index()
    results = retrieve("Who wrote the play Hamlet?", k=3)
    top_text = (results[0]["title"] + " " + results[0]["text"]).lower()
    assert "shakespeare" in top_text, "top result should mention Shakespeare"


if __name__ == "__main__":
    # Allow running without pytest.
    _ensure_index()
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASSED: {name}")
    print("All retrieval sanity checks passed.")
