"""Summary statistics over an eval results file.

Beyond raw accuracy, these are the numbers that show *why* Self-RAG differs from
vanilla RAG:

  retrieval_rate     % of questions where the system retrieved at all. Vanilla RAG
                     is 100% by construction; Self-RAG should be lower, which is
                     the paper's adaptive-retrieval claim.
  retrieval_f1       agreement of the Retrieve decision with the human
                     `needs_retrieval` label (sample eval set only).
  support_rate       % of answered-with-context questions the critic judged
                     fully or partially supported (IsSup), i.e. grounding quality.
  regeneration_rate  % of questions where the strict-grounding retry fired.
  avg_passages_used  mean passages surviving the IsRel filter.

Usage:
    python eval/stats.py                              # reads sample results
    python eval/stats.py eval/results/results_popqa.json
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "eval" / "results" / "results_sample.json"


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else 0.0


def _retrieval_agreement(records):
    """Precision/recall/F1 of the Retrieve decision against human labels."""
    labelled = [r for r in records if r.get("needs_retrieval_label") is not None]
    if not labelled:
        return None

    tp = sum(1 for r in labelled if r["retrieved"] and r["needs_retrieval_label"])
    fp = sum(1 for r in labelled if r["retrieved"] and not r["needs_retrieval_label"])
    fn = sum(1 for r in labelled if not r["retrieved"] and r["needs_retrieval_label"])
    tn = sum(1 for r in labelled if not r["retrieved"] and not r["needs_retrieval_label"])

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": 100.0 * precision,
        "recall": 100.0 * recall,
        "f1": 100.0 * f1,
        "agreement": 100.0 * (tp + tn) / len(labelled),
        "n_labelled": len(labelled),
    }


def summarize_system(records):
    """Compute all summary statistics for one system's per-example records."""
    n = len(records)
    if n == 0:
        return {}

    retrieved = [r for r in records if r.get("retrieved")]
    grounded = [r for r in retrieved if r.get("support") is not None]
    supported = [r for r in grounded if r["support"] in ("fully", "partially")]
    fully = [r for r in grounded if r["support"] == "fully"]

    summary = {
        "n": n,
        "accuracy": 100.0 * sum(r["accuracy"] for r in records) / n,
        "exact_match": 100.0 * sum(r["exact_match"] for r in records) / n,
        "retrieval_rate": 100.0 * len(retrieved) / n,
        "avg_passages_used": _mean([r.get("passages_used", 0) for r in records]),
        "regeneration_rate": 100.0 * sum(1 for r in records if r.get("regenerated")) / n,
        "avg_latency_s": _mean([r.get("latency_s") for r in records]),
        "errors": sum(1 for r in records if r.get("error")),
        # Support stats are only defined where the critic actually judged an answer.
        "support_rate": 100.0 * len(supported) / len(grounded) if grounded else None,
        "fully_supported_rate": 100.0 * len(fully) / len(grounded) if grounded else None,
        "n_judged": len(grounded),
    }

    agreement = _retrieval_agreement(records)
    if agreement:
        summary["retrieval_decision"] = agreement
    return summary


def summarize(results):
    """Summarize every system in a results dict."""
    return {name: summarize_system(recs) for name, recs in results["systems"].items()}


def format_table(summaries):
    """Render the headline comparison as a fixed-width text table."""
    cols = [
        ("System", lambda s: None),
        ("Acc%", lambda s: s["accuracy"]),
        ("EM%", lambda s: s["exact_match"]),
        ("Retr%", lambda s: s["retrieval_rate"]),
        ("Psg", lambda s: s["avg_passages_used"]),
        ("Supp%", lambda s: s["support_rate"]),
        ("Regen%", lambda s: s["regeneration_rate"]),
    ]
    header = f"{cols[0][0]:<13}" + "".join(f"{c[0]:>9}" for c in cols[1:])
    lines = [header, "-" * len(header)]
    for name, s in summaries.items():
        row = f"{name:<13}"
        for _, getter in cols[1:]:
            value = getter(s)
            row += f"{'n/a':>9}" if value is None else f"{value:>9.1f}"
        lines.append(row)
    return "\n".join(lines)


def load_results(path=None):
    path = Path(path) if path else DEFAULT_RESULTS
    if not path.exists():
        raise FileNotFoundError(
            f"No results at {path}. Run the evaluation first:\n"
            "    python eval/run_eval.py"
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    results = load_results(path)
    summaries = summarize(results)

    print(f"Dataset: {results['dataset']}  |  {results['num_examples']} questions  |  k={results['k']}\n")
    print(format_table(summaries))

    for name, s in summaries.items():
        decision = s.get("retrieval_decision")
        if decision:
            print(
                f"\n{name} retrieve-decision vs human labels: "
                f"agreement {decision['agreement']:.1f}%  "
                f"(P {decision['precision']:.1f} / R {decision['recall']:.1f} / "
                f"F1 {decision['f1']:.1f}, n={decision['n_labelled']})"
            )


if __name__ == "__main__":
    main()
