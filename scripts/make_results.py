"""Build the results table and Success@k / Telling@k curves from the conversations in the output directory.

Usage (from the repo root): python scripts/make_results.py [--output_dir output] [--results_dir results]"""
import argparse
import csv
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "interactivetutoring"))
from evaluate import evaluate_problems  # noqa: E402
from utils import read_jsonl  # noqa: E402

RUN_FILE_RE = re.compile(r"^(nextstep|fewshot)_(\d+)shot\.jsonl$")
METRIC_COLUMNS = ["Success@5", "Telling@5", "Success@10", "Telling@10"]

# Reference rows from Macina et al. (2023), in METRIC_COLUMNS order (percent)
PAPER_ROWS = [
    ("NextStep (paper)", ["25", "0", "25", "0"]),
    ("ChatGPT (paper)", ["29", "16", "53", "32"]),
    ("Flan-T5 780M (paper)", ["30", "<1", "39", "4"]),
    ("Ground truth (paper)", ["59", "<1", "82", "4"]),
]

# Fixed colour and marker per condition (categorical slots 1-4), so a condition keeps its colour
# whichever subset of runs exists
STYLES = {
    "nextstep_0shot": ("#2a78d6", "o"),
    "fewshot_0shot": ("#eb6834", "s"),
    "fewshot_2shot": ("#1baf7a", "^"),
    "fewshot_5shot": ("#eda100", "D"),
}
REFERENCE_COLOR = "#8a8984"
SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--test_file", type=str, default="data/test.jsonl")
    parser.add_argument("--max_k", type=int, default=10)
    return parser.parse_args()


def condition_label(run_name: str) -> str:
    """Human-readable name of a run, e.g. fewshot_2shot -> 'GPT-6 Luna 2-shot'."""
    teacher, shots = RUN_FILE_RE.match(run_name + ".jsonl").groups()
    return "NextStep (ours)" if teacher == "nextstep" else f"GPT-6 Luna {shots}-shot"


def find_runs(output_dir: str):
    """Return the run names ({teacher}_{shots}shot) with an output file, NextStep first, then by shots."""
    names = [f[:-len(".jsonl")] for f in os.listdir(output_dir) if RUN_FILE_RE.match(f)]
    return sorted(names, key=lambda n: (not n.startswith("nextstep"), int(RUN_FILE_RE.match(n + ".jsonl").group(2))))


def evaluate_run(path: str, run_name: str, max_k: int):
    """Evaluate one output file; the conversation key is the record's model_name, or else the run name."""
    data = read_jsonl(path)
    model_name = data[0].get("model_name", run_name) if data else run_name
    success, telling = evaluate_problems(data, model_name, max_k)
    return len(data), success, telling


def pct(value: float) -> str:
    return f"{value * 100:.1f}"


def write_tables(rows, results_dir: str):
    """Write results_table.csv and results_table.md; rows are (condition, n, [metric strings in percent])."""
    with open(os.path.join(results_dir, "results_table.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["condition", "n_dialogues", *METRIC_COLUMNS])
        for condition, n, values in rows:
            writer.writerow([condition, n, *values])
    lines = ["| Condition | n | " + " | ".join(METRIC_COLUMNS) + " |",
             "|---|---:|" + "---:|" * len(METRIC_COLUMNS)]
    for condition, n, values in rows:
        lines.append(f"| {condition} | {n} | " + " | ".join(f"{v}%" for v in values) + " |")
    with open(os.path.join(results_dir, "results_table.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return "\n".join(lines)


def spread_labels(ys, min_gap):
    """Nudge end-of-line label positions apart so that no two are closer than min_gap."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    placed = list(ys)
    for prev, cur in zip(order, order[1:]):
        placed[cur] = max(placed[cur], placed[prev] + min_gap)
    return placed


def plot_curves(curves, reference, max_k: int, path: str):
    """Plot Success@k and Telling@k (k = 1..max_k) side by side, one line per condition.

    curves: {run_name: (success, telling)}; reference: (label, success, telling) drawn as a dashed grey line."""
    ks = list(range(1, max_k + 1))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True, facecolor=SURFACE)
    for ax, (metric_index, title) in zip(axes, [(0, "Success@k"), (1, "Telling@k")]):
        ax.set_facecolor(SURFACE)
        label_items = []
        ref_label, *ref_metrics = reference
        ref_values = [v * 100 for v in ref_metrics[metric_index][1:]]
        ax.plot(ks, ref_values, color=REFERENCE_COLOR, linewidth=2, linestyle=(0, (4, 3)), label=ref_label)
        label_items.append((ref_values[-1], ref_label, TEXT_SECONDARY))
        for run_name, metrics in curves.items():
            color, marker = STYLES.get(run_name, (TEXT_SECONDARY, "o"))
            values = [v * 100 for v in metrics[metric_index][1:]]
            ax.plot(ks, values, color=color, linewidth=2, marker=marker, markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=condition_label(run_name))
            label_items.append((values[-1], condition_label(run_name), TEXT_PRIMARY))
        for y, (_, text, ink) in zip(spread_labels([y for y, _, _ in label_items], 5.5), label_items):
            ax.annotate(text, xy=(max_k, y), xytext=(8, 0), textcoords="offset points", va="center",
                        fontsize=8.5, color=ink, annotation_clip=False)
        ax.set_title(title, loc="left", fontsize=12, color=TEXT_PRIMARY, fontweight="bold")
        ax.set_xlabel("k (student turns)", color=TEXT_SECONDARY)
        ax.set_xticks(ks)
        ax.set_xlim(0.7, max_k + 0.3)
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=TEXT_SECONDARY, length=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=9,
               labelcolor=TEXT_SECONDARY, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 0.93, 1), w_pad=8)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


if __name__ == '__main__':
    args = get_args()
    os.makedirs(args.results_dir, exist_ok=True)

    rows, curves = [], {}
    for run_name in find_runs(args.output_dir):
        n, success, telling = evaluate_run(os.path.join(args.output_dir, run_name + ".jsonl"), run_name, args.max_k)
        curves[run_name] = (success, telling)
        rows.append((condition_label(run_name), n, [pct(success[5]), pct(telling[5]), pct(success[10]),
                                                    pct(telling[10])]))

    test = read_jsonl(args.test_file)
    gt_success, gt_telling = evaluate_problems(test, "", args.max_k, ground_truth=True)
    rows.append(("Ground truth (repo evaluator)", len(test),
                 [pct(gt_success[5]), pct(gt_telling[5]), pct(gt_success[10]), pct(gt_telling[10])]))
    rows.extend((name, "", values) for name, values in PAPER_ROWS)

    print(write_tables(rows, args.results_dir))
    plot_curves(curves, ("Human ground truth", gt_success, gt_telling), args.max_k,
                os.path.join(args.results_dir, "curves.png"))
    print(f"\nWrote {args.results_dir}/results_table.csv, results_table.md and curves.png")
