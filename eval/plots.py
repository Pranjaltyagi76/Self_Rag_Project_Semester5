"""Comparison charts and results table for the report.

Reads an eval results JSON and writes, by default into docs/ so the figures are
tracked alongside the report:

    docs/fig_accuracy.png        answer quality (accuracy + exact match)
    docs/fig_retrieval_rate.png  the headline: how often each system retrieves
    docs/fig_support.png         grounding quality of the RAG systems
    docs/RESULTS.md              the same numbers as a markdown table

Usage:
    python eval/plots.py                              # reads sample results
    python eval/plots.py eval/results/results_popqa.json
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

from stats import load_results, summarize  # noqa: E402

DOCS_DIR = ROOT / "docs"

# Categorical palette: one fixed hue per system, never cycled or reassigned, so a
# system keeps its colour across every figure. Validated for colour-vision
# deficiency separation against the light chart surface.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#dedcd6"
SYSTEM_COLORS = {
    "no_rag": "#2a78d6",       # blue
    "vanilla_rag": "#eb6834",  # orange
    "self_rag": "#1baf7a",     # aqua
}
SYSTEM_LABELS = {
    "no_rag": "No-RAG",
    "vanilla_rag": "Vanilla RAG",
    "self_rag": "Self-RAG",
}


def _style_axes(ax, ylabel="percent"):
    """Recessive grid and axes: horizontal rules only, no box around the plot."""
    ax.set_facecolor(SURFACE)
    ax.set_ylim(0, 105)
    ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)


def _label_bars(ax, bars, fmt="{:.0f}"):
    """Direct value labels: identity and magnitude never depend on colour alone."""
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            fmt.format(height),
            (bar.get_x() + bar.get_width() / 2, height),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            fontsize=8.5,
            color=INK,
        )


def _order(summaries):
    """Keep a stable system order regardless of dict insertion order."""
    return [name for name in SYSTEM_LABELS if name in summaries]


def plot_accuracy(summaries, out_dir):
    """Grouped bars: answer quality per system, across both metrics."""
    import matplotlib.pyplot as plt

    names = _order(summaries)
    metrics = [("accuracy", "Accuracy"), ("exact_match", "Exact match")]

    fig, ax = plt.subplots(figsize=(6.4, 3.8), facecolor=SURFACE)
    width = 0.24
    gap = 0.02  # surface gap keeps adjacent bars from touching
    for i, name in enumerate(names):
        offset = (i - (len(names) - 1) / 2) * (width + gap)
        positions = [j + offset for j in range(len(metrics))]
        values = [summaries[name][key] for key, _ in metrics]
        bars = ax.bar(
            positions, values, width, label=SYSTEM_LABELS[name],
            color=SYSTEM_COLORS[name],
        )
        _label_bars(ax, bars)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([label for _, label in metrics], color=INK, fontsize=10)
    ax.set_title("Answer quality by system", color=INK, fontsize=12, pad=12, loc="left")
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_MUTED, ncols=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.12))

    path = out_dir / "fig_accuracy.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_retrieval_rate(summaries, out_dir):
    """The headline chart: Self-RAG retrieves less often than vanilla RAG."""
    import matplotlib.pyplot as plt

    names = _order(summaries)
    values = [summaries[name]["retrieval_rate"] for name in names]

    fig, ax = plt.subplots(figsize=(5.6, 3.6), facecolor=SURFACE)
    bars = ax.bar(
        range(len(names)), values, 0.42,
        color=[SYSTEM_COLORS[n] for n in names],
    )
    _label_bars(ax, bars, fmt="{:.0f}%")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([SYSTEM_LABELS[n] for n in names], color=INK, fontsize=10)
    ax.set_title(
        "How often each system retrieves", color=INK, fontsize=12, pad=12, loc="left"
    )
    _style_axes(ax, ylabel="questions retrieved for (%)")

    path = out_dir / "fig_retrieval_rate.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_support(summaries, out_dir):
    """Grounding quality, for the systems whose answers the critic judged."""
    import matplotlib.pyplot as plt

    names = [n for n in _order(summaries) if summaries[n].get("support_rate") is not None]
    if not names:
        return None

    fig, ax = plt.subplots(figsize=(5.6, 3.6), facecolor=SURFACE)
    width = 0.26
    gap = 0.02
    series = [("support_rate", "Supported"), ("fully_supported_rate", "Fully supported")]
    alphas = [1.0, 0.55]
    for i, (key, label) in enumerate(series):
        offset = (i - (len(series) - 1) / 2) * (width + gap)
        positions = [j + offset for j in range(len(names))]
        values = [summaries[n][key] for n in names]
        # Hue carries system identity here, so the measure is encoded by position
        # and opacity instead -- each system keeps its own colour in both bars.
        bars = ax.bar(
            positions, values, width,
            color=[SYSTEM_COLORS[n] for n in names],
            alpha=alphas[i],
        )
        _label_bars(ax, bars)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([SYSTEM_LABELS[n] for n in names], color=INK, fontsize=10)
    ax.set_title(
        "Grounding quality (IsSup)", color=INK, fontsize=12, pad=12, loc="left"
    )
    _style_axes(ax, ylabel="judged answers (%)")

    # Neutral legend handles: opacity distinguishes the two measures, so the
    # legend must not claim any single system's hue.
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=INK_MUTED, alpha=alphas[i], label=label)
        for i, (_, label) in enumerate(series)
    ]
    ax.legend(handles=handles, frameon=False, fontsize=9, labelcolor=INK_MUTED,
              ncols=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))

    path = out_dir / "fig_support.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def write_results_table(results, summaries, out_dir):
    """Write the markdown results table used in the report (and as the table view)."""
    def cell(value, suffix=""):
        return "n/a" if value is None else f"{value:.1f}{suffix}"

    lines = [
        "# Results",
        "",
        f"Dataset: **{results['dataset']}** · {results['num_examples']} questions · k={results['k']}",
        "",
        "| System | Accuracy % | Exact match % | Retrieval rate % | Avg passages used | Supported % | Regenerated % |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in _order(summaries):
        s = summaries[name]
        lines.append(
            f"| {SYSTEM_LABELS[name]} | {cell(s['accuracy'])} | {cell(s['exact_match'])} "
            f"| {cell(s['retrieval_rate'])} | {cell(s['avg_passages_used'])} "
            f"| {cell(s['support_rate'])} | {cell(s['regeneration_rate'])} |"
        )

    for name in _order(summaries):
        decision = summaries[name].get("retrieval_decision")
        if decision:
            lines += [
                "",
                f"**{SYSTEM_LABELS[name]} retrieve-decision vs human labels:** "
                f"agreement {decision['agreement']:.1f}% "
                f"(precision {decision['precision']:.1f}, recall {decision['recall']:.1f}, "
                f"F1 {decision['f1']:.1f}, n={decision['n_labelled']})",
            ]

    lines += [
        "",
        "![Answer quality](fig_accuracy.png)",
        "",
        "![Retrieval rate](fig_retrieval_rate.png)",
        "",
        "![Grounding quality](fig_support.png)",
        "",
    ]

    path = out_dir / "RESULTS.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def make_all(results_path=None, out_dir=DOCS_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(results_path)
    summaries = summarize(results)

    written = [
        plot_accuracy(summaries, out_dir),
        plot_retrieval_rate(summaries, out_dir),
        plot_support(summaries, out_dir),
        write_results_table(results, summaries, out_dir),
    ]
    for path in written:
        if path:
            print(f"Wrote {path}")
    return written


if __name__ == "__main__":
    make_all(sys.argv[1] if len(sys.argv) > 1 else None)
