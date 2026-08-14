"""Load the passage corpus from a JSONL file into a list of dicts.

Each line in the JSONL file is a JSON object with keys: id, title, text.
Returns a list of {"id": str, "title": str, "text": str}.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "data" / "corpus" / "sample_passages.jsonl"


def load_corpus(path=DEFAULT_CORPUS):
    """Read passages from a JSONL file. Skips blank lines."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Corpus file not found: {path}\n"
            "Expected a JSONL file with one {id, title, text} object per line."
        )

    passages = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {e}") from e
            passages.append(
                {
                    "id": str(obj["id"]),
                    "title": obj.get("title", ""),
                    "text": obj["text"],
                }
            )
    return passages


if __name__ == "__main__":
    corpus = load_corpus()
    print(f"Loaded {len(corpus)} passages from {DEFAULT_CORPUS}")
    for p in corpus[:3]:
        print(f"  [{p['id']}] {p['title']}: {p['text'][:60]}...")
