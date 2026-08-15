"""Evaluation harness: run all three systems over an eval set and score them.

Systems compared:
    no_rag       parametric knowledge only (lower bound)
    vanilla_rag  always retrieve top-k
    self_rag     adaptive retrieval + self-critique

Usage (from the repo root):

    python eval/run_eval.py                      # sample set, all systems
    python eval/run_eval.py --dataset popqa --n 200
    python eval/run_eval.py --systems self_rag --k 3

Results are written to eval/results/results_<dataset>.json, holding per-example
predictions plus the reflection metadata that reports/plots are built from.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from eval_data import load_eval_set  # noqa: E402
from metrics import score_prediction  # noqa: E402

RESULTS_DIR = ROOT / "eval" / "results"
SYSTEM_NAMES = ["no_rag", "vanilla_rag", "self_rag"]


def get_systems():
    """Return {name: callable(query, k=...) -> (answer, info)}.

    Imported lazily so a partial environment can still run a subset of systems.
    """
    from no_rag import no_rag_answer
    from self_rag import self_rag_answer
    from vanilla_rag import vanilla_rag_answer

    return {
        "no_rag": no_rag_answer,
        "vanilla_rag": vanilla_rag_answer,
        "self_rag": self_rag_answer,
    }


def run_system(name, system_fn, examples, k=5, verbose=True):
    """Run one system over every example and score its predictions."""
    records = []
    for i, ex in enumerate(examples, start=1):
        started = time.time()
        try:
            answer, info = system_fn(ex["question"], k=k)
            error = None
        except Exception as e:  # keep the sweep alive if one call fails
            answer, info, error = "", {}, f"{type(e).__name__}: {e}"

        scores = score_prediction(answer, ex["answers"])
        records.append(
            {
                "id": ex["id"],
                "question": ex["question"],
                "gold_answers": ex["answers"],
                "needs_retrieval_label": ex.get("needs_retrieval"),
                "prediction": answer,
                "accuracy": scores["accuracy"],
                "exact_match": scores["exact_match"],
                "retrieved": info.get("retrieved"),
                "num_retrieved": info.get("num_retrieved", 0),
                "passages_used": info.get("passages_used", 0),
                "support": info.get("support"),
                "usefulness": info.get("usefulness"),
                "regenerated": info.get("regenerated", False),
                "latency_s": round(time.time() - started, 3),
                "error": error,
            }
        )
        if verbose:
            mark = "x" if error else ("+" if scores["accuracy"] else "-")
            print(f"  [{name}] {i}/{len(examples)} {mark} {ex['question'][:50]}")
    return records


def run_eval(dataset="sample", n=None, k=5, systems=None, verbose=True):
    """Run the selected systems over the dataset and return the results dict."""
    examples = load_eval_set(dataset, n)
    chosen = systems or SYSTEM_NAMES
    available = get_systems()

    print(f"Evaluating {len(examples)} questions from '{dataset}' (k={k})")
    results = {"dataset": dataset, "k": k, "num_examples": len(examples), "systems": {}}
    for name in chosen:
        if name not in available:
            raise ValueError(f"Unknown system: {name!r} (expected one of {SYSTEM_NAMES})")
        print(f"\nRunning {name}...")
        results["systems"][name] = run_system(
            name, available[name], examples, k=k, verbose=verbose
        )
    return results


def save_results(results, out_path=None):
    out_path = Path(out_path) if out_path else RESULTS_DIR / f"results_{results['dataset']}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved results to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Run the Self-RAG evaluation.")
    parser.add_argument("--dataset", default="sample", choices=["sample", "popqa"])
    parser.add_argument("--n", type=int, default=None, help="limit number of questions")
    parser.add_argument("--k", type=int, default=5, help="passages to retrieve")
    parser.add_argument("--systems", nargs="+", default=None, choices=SYSTEM_NAMES)
    parser.add_argument("--out", default=None, help="output JSON path")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    results = run_eval(
        dataset=args.dataset,
        n=args.n,
        k=args.k,
        systems=args.systems,
        verbose=not args.quiet,
    )
    save_results(results, args.out)


if __name__ == "__main__":
    main()
