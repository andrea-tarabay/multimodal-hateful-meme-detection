"""
visualize_results_simple.py

Simple CLI bar chart of MUTE test metrics for all three student models.

Input (one per model, produced by evaluate.py):
    <results_dir>/test_metrics_memeblip2_only_best.json
    <results_dir>/test_metrics_context_only_best.json
    <results_dir>/test_metrics_context_distillation_best.json

Output:
    <results_dir>/test_metrics_simple.png  (or --out path)

Usage:
    python visualize_results_simple.py [--results-dir <path>] [--out <path>]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


DEFAULT_RESULTS_DIR = Path(
    "/mnt/course-ee-559/rcp-caas-ee-559-g03/scratch-g03/finetune-joelle/"
    "old_runs/mute_from_0524-200949_lr1e-5_wd0.01_dropout0.5_beta0.0_lambdakd0.25_temp1.0/"
    "results"
)


parser = argparse.ArgumentParser(description="Plot MUTE test metrics bar chart.")
parser.add_argument(
    "--results-dir",
    type=Path,
    default=DEFAULT_RESULTS_DIR,
    help="Folder containing test_metrics_*_best.json files.",
)
parser.add_argument(
    "--out",
    type=Path,
    default=None,
    help="Output PNG path. Default: results_dir/test_metrics_simple.png",
)
args = parser.parse_args()

RESULTS_DIR = args.results_dir
OUT = args.out or (RESULTS_DIR / "test_metrics_simple.png")

if not RESULTS_DIR.exists():
    print(f"Results directory not found: {RESULTS_DIR}", file=sys.stderr)
    sys.exit(1)

print(f"Loading results from: {RESULTS_DIR}")

COLORS = {
    "memeblip2_only": "#E76F51",
    "context_only": "#457B9D",
    "context_distillation": "#2A9D8F",
}

LABELS = {
    "memeblip2_only": "BLIP-2 Only",
    "context_only": "Context Only",
    "context_distillation": "Context Distillation",
}

MODEL_KEYS = ["memeblip2_only", "context_only", "context_distillation"]


def load_metrics(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_metrics(metrics):
    """
    Our MUTE evaluate.py saves:
        accuracy, precision, recall, f1

    The old visualization expected:
        accuracy, macro_precision, macro_recall, macro_f1

    So we map them safely.
    """
    return {
        "accuracy": metrics.get("accuracy", 0.0),
        "macro_precision": metrics.get("macro_precision", metrics.get("precision", 0.0)),
        "macro_recall": metrics.get("macro_recall", metrics.get("recall", 0.0)),
        "macro_f1": metrics.get("macro_f1", metrics.get("f1", 0.0)),
    }


full = {}

for key in MODEL_KEYS:
    path = RESULTS_DIR / f"test_metrics_{key}_best.json"
    if not path.exists():
        print(f"Missing metrics file: {path}", file=sys.stderr)
        sys.exit(1)

    full[key] = normalize_metrics(load_metrics(path))


GRID_COL = "#2E3148"
TEXT_COL = "black"

fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_alpha(0)
ax.set_facecolor("none")

for spine in ax.spines.values():
    spine.set_edgecolor(GRID_COL)

ax.tick_params(colors=TEXT_COL, labelsize=12)
ax.xaxis.label.set_color(TEXT_COL)
ax.yaxis.label.set_color(TEXT_COL)
ax.grid(color=GRID_COL, linewidth=0.6, alpha=0.7)

metric_names = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
metric_labels = ["Accuracy", "Precision", "Recall", "F1"]

x = np.arange(len(metric_names))
w = 0.22

for i, key in enumerate(MODEL_KEYS):
    vals = [full[key][m] * 100 for m in metric_names]

    bars = ax.bar(
        x + (i - 1) * w,
        vals,
        w,
        color=COLORS[key],
        alpha=0.88,
        label=LABELS[key],
        edgecolor="none",
        zorder=3,
    )

    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{v:.1f}",
            ha="center",
            va="bottom",
            color=TEXT_COL,
            fontsize=11,
            fontweight="bold",
        )

ax.set_xticks(x)
ax.set_xticklabels(metric_labels, color=TEXT_COL, fontsize=13)

all_values = [full[k][m] * 100 for k in MODEL_KEYS for m in metric_names]
ymin = max(0, min(all_values) - 5)
ymax = min(100, max(all_values) + 5)
ax.set_ylim(ymin, ymax)

ax.set_ylabel("Score (%)", color=TEXT_COL, fontsize=13)
ax.set_title(
    "MUTE Test Metrics — Student Models",
    color=TEXT_COL,
    fontsize=16,
    fontweight="bold",
    pad=10,
)

legend = ax.legend(
    facecolor="none",
    edgecolor=GRID_COL,
    labelcolor=TEXT_COL,
    fontsize=12,
    loc="upper right",
)
legend.get_frame().set_alpha(0)

OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=150, bbox_inches="tight", transparent=True)
print(f"Saved → {OUT}")
plt.close()
