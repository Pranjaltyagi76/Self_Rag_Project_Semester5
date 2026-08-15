"""Evaluation dataset loaders.

Two eval sets are supported:

  "sample"  a small bundled set aligned with data/corpus/sample_passages.jsonl.
            Runs out of the box and includes questions that need no retrieval,
            so the adaptive-retrieval behaviour is visible in the results.

  "popqa"   the real PopQA benchmark (Mallen et al.), as used by the Self-RAG
            paper. Downloaded via HuggingFace `datasets` and cached locally as
            JSONL. Requires a corpus large enough to answer it (see README).

Every example is normalised to:

    {"id": str, "question": str, "answers": [str, ...], "needs_retrieval": bool|None}

`needs_retrieval` is a human-labelled hint available only in the sample set; it
is used to report how well the Retrieve decision agrees with human judgement.
"""
import json
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
DATA_DIR = EVAL_DIR / "data"
SAMPLE_PATH = DATA_DIR / "sample_eval.jsonl"
POPQA_CACHE = DATA_DIR / "popqa_subset.jsonl"

POPQA_HF_NAME = "akariasai/PopQA"


def _read_jsonl(path):
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {e}") from e
    return examples


def _write_jsonl(path, examples):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def load_sample(n=None):
    """Load the bundled sample eval set."""
    if not SAMPLE_PATH.exists():
        raise FileNotFoundError(f"Sample eval set not found: {SAMPLE_PATH}")
    examples = [
        {
            "id": str(ex["id"]),
            "question": ex["question"],
            "answers": list(ex["answers"]),
            "needs_retrieval": ex.get("needs_retrieval"),
        }
        for ex in _read_jsonl(SAMPLE_PATH)
    ]
    return examples[:n] if n else examples


def load_popqa(n=500):
    """Load a PopQA subset, downloading and caching it on first use."""
    if POPQA_CACHE.exists():
        cached = _read_jsonl(POPQA_CACHE)
        if len(cached) >= n:
            return cached[:n]

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(
            "PopQA needs the `datasets` package (pip install datasets), or place a "
            f"pre-downloaded subset at {POPQA_CACHE}"
        ) from e

    ds = load_dataset(POPQA_HF_NAME, split="test")
    examples = []
    for row in ds.select(range(min(n, len(ds)))):
        # PopQA stores gold answers as a JSON-encoded list of strings.
        answers = row["possible_answers"]
        if isinstance(answers, str):
            answers = json.loads(answers)
        examples.append(
            {
                "id": str(row["id"]),
                "question": row["question"],
                "answers": list(answers),
                "needs_retrieval": None,
            }
        )

    _write_jsonl(POPQA_CACHE, examples)
    return examples


def load_eval_set(name="sample", n=None):
    """Load an eval set by name: "sample" or "popqa"."""
    name = name.lower()
    if name == "sample":
        return load_sample(n)
    if name == "popqa":
        return load_popqa(n or 500)
    raise ValueError(f"Unknown eval set: {name!r} (expected 'sample' or 'popqa')")


if __name__ == "__main__":
    examples = load_eval_set("sample")
    needs = sum(1 for e in examples if e["needs_retrieval"])
    print(f"Loaded {len(examples)} sample questions ({needs} labelled as needing retrieval).")
    for ex in examples[:3]:
        print(f"  [{ex['id']}] {ex['question']}  -> {ex['answers']}")
