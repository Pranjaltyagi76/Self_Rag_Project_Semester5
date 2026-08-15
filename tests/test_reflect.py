"""Tests for the four reflection functions.

Each function takes an injectable `generate`, so these run without an API key.
They pin down the *parsing* of the critic's replies, which is where the failure
modes live: a model that answers "IRRELEVANT" must not be read as relevant, and
a stray "10" must not be read as a usefulness score of 1.

Run:  pytest tests/test_reflect.py   (or: python tests/test_reflect.py)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reflect import is_relevant, is_supported, needs_retrieval, usefulness  # noqa: E402


def replying(text):
    """A mock generate() that always returns `text`."""
    return lambda *args, **kwargs: text


def test_needs_retrieval_parses_yes_and_no():
    assert needs_retrieval("q", generate=replying("YES")) is True
    assert needs_retrieval("q", generate=replying("yes")) is True
    assert needs_retrieval("q", generate=replying("NO")) is False


def test_is_relevant_does_not_confuse_irrelevant_with_relevant():
    # "IRRELEVANT" contains the substring "RELEVANT" -- the classic trap here.
    assert is_relevant("q", "p", generate=replying("RELEVANT")) is True
    assert is_relevant("q", "p", generate=replying("IRRELEVANT")) is False


def test_is_relevant_accepts_dict_or_string_passages():
    passage = {"title": "Hamlet", "text": "A tragedy by Shakespeare."}
    assert is_relevant("q", passage, generate=replying("RELEVANT")) is True
    assert is_relevant("q", "plain string", generate=replying("RELEVANT")) is True


def test_is_supported_maps_all_three_verdicts():
    assert is_supported("q", "a", ["p"], generate=replying("FULLY")) == "fully"
    assert is_supported("q", "a", ["p"], generate=replying("PARTIALLY")) == "partially"
    assert is_supported("q", "a", ["p"], generate=replying("NO")) == "no"
    # Anything unparseable is treated as unsupported rather than assumed good.
    assert is_supported("q", "a", ["p"], generate=replying("???")) == "no"


def test_usefulness_reads_single_digit_ratings():
    for digit in range(1, 6):
        assert usefulness("q", "a", generate=replying(str(digit))) == digit
    assert usefulness("q", "a", generate=replying("Score 4 of 5")) == 4


def test_usefulness_ignores_multi_digit_numbers():
    # Regression: a bare [1-5] search read "10" as 1, scoring a good answer as
    # the worst possible and triggering a needless regeneration.
    assert usefulness("q", "a", generate=replying("Rating: 10")) == 3
    assert usefulness("q", "a", generate=replying("15")) == 3


def test_usefulness_falls_back_to_neutral_on_garbage():
    assert usefulness("q", "a", generate=replying("no idea")) == 3


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASSED: {name}")
    print("All reflection tests passed.")
