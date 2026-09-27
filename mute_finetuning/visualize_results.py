"""
visualize_results.py

Dashboard visualization for Stage 2 MUTE fine-tuning results.


Expected Stage 2 files:
    results/test_metrics_memeblip2_only_best.json
    results/test_metrics_context_only_best.json
    results/test_metrics_context_distillation_best.json

    results/test_predictions_memeblip2_only_best.csv
    results/test_predictions_context_only_best.csv
    results/test_predictions_context_distillation_best.csv

Optional teacher metrics:
    results/teacher_metrics/teacher_metrics.json

Output:
    results_dashboard_mute.png
"""

import json
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import auc, precision_recall_curve, roc_curve

try:
    from config import PROJECT_ROOT, RESULTS_DIR
except Exception:
    # Fallback for running inside the RunAI job if config import fails.
    PROJECT_ROOT = Path("/scratch/finetune-joelle")
    RESULTS_DIR = PROJECT_ROOT / "results"


# ============================================================
# 1. Paths / switches
# ============================================================

BASE = Path(PROJECT_ROOT)
RESULTS_DIR = Path(RESULTS_DIR)

INCLUDE_TEACHER = True
TEACHER_METRICS_PATH = RESULTS_DIR / "teacher_metrics" / "teacher_metrics.json"

OUTPUT_PATH = BASE / "results_dashboard_mute.png"


# ============================================================
# 2. Palette / labels
# ============================================================

COLORS = {
    "memeblip2_only": "#E76F51",
    "context_only": "#457B9D",
    "context_distillation": "#2A9D8F",
    "teacher": "#8338EC",
}

LABELS = {
    "memeblip2_only": "BLIP-2 Only",
    "context_only": "Context Only",
    "context_distillation": "Context Distillation",
    "teacher": "LLM Teacher*",
}

MODEL_KEYS = ["memeblip2_only", "context_only", "context_distillation"]


# ============================================================
# 3. Loading helpers
# ============================================================

def load_metrics(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_student_metrics(metrics):
    """
    Stage 2 evaluate.py saves: accuracy, precision, recall, f1, auc,
    confusion_matrix. The Stage 1 plot expects macro_* names too.
    """
    out = dict(metrics)

    if "macro_precision" not in out:
        out["macro_precision"] = out.get("precision", 0.0)
    if "macro_recall" not in out:
        out["macro_recall"] = out.get("recall", 0.0)
    if "macro_f1" not in out:
        out["macro_f1"] = out.get("f1", 0.0)
    if "test_loss" not in out:
        out["test_loss"] = out.get("loss", None)

    return out


def normalize_teacher_metrics(metrics):
    """
    Supports both Stage 2 teacher_metrics.json and older Stage 1 teacher metric
    naming conventions.
    """
    if "valid_only_accuracy" in metrics:
        return {
            "accuracy": metrics["valid_only_accuracy"],
            "macro_precision": metrics["valid_only_macro_precision"],
            "macro_recall": metrics["valid_only_macro_recall"],
            "macro_f1": metrics["valid_only_macro_f1"],
            "confusion_matrix": metrics["valid_only_confusion_matrix_labels_0_1"],
        }

    out = dict(metrics)
    if "macro_precision" not in out:
        out["macro_precision"] = out.get("precision", 0.0)
    if "macro_recall" not in out:
        out["macro_recall"] = out.get("recall", 0.0)
    if "macro_f1" not in out:
        out["macro_f1"] = out.get("f1", 0.0)
    return out


def prediction_prob_column(df):
    if "prob_hateful" in df.columns:
        return "prob_hateful"
    if "prob_hate" in df.columns:
        return "prob_hate"

    raise KeyError(
        "Prediction CSV must contain either 'prob_hateful' or 'prob_hate'. "
        f"Available columns: {list(df.columns)}"
    )


def prediction_label_column(df):
    if "label" in df.columns:
        return "label"
    if "true_label" in df.columns:
        return "true_label"

    raise KeyError(
        "Prediction CSV must contain either 'label' or 'true_label'. "
        f"Available columns: {list(df.columns)}"
    )


def load_student_results():
    full = {}
    preds = {}

    for key in MODEL_KEYS:
        metrics_path = RESULTS_DIR / f"test_metrics_{key}_best.json"
        predictions_path = RESULTS_DIR / f"test_predictions_{key}_best.csv"

        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
        if not predictions_path.exists():
            raise FileNotFoundError(f"Missing prediction file: {predictions_path}")

        full[key] = normalize_student_metrics(load_metrics(metrics_path))
        preds[key] = pd.read_csv(predictions_path)

    return full, preds


def maybe_load_teacher(full):
    if not INCLUDE_TEACHER:
        return full, list(MODEL_KEYS)

    if not TEACHER_METRICS_PATH.exists():
        print(f"[Warning] Teacher metrics not found, skipping teacher: {TEACHER_METRICS_PATH}")
        return full, list(MODEL_KEYS)

    full["teacher"] = normalize_teacher_metrics(load_metrics(TEACHER_METRICS_PATH))
    return full, MODEL_KEYS + ["teacher"]


def per_class_pr(cm):
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    p0 = tn / (tn + fn) if (tn + fn) else 0
    r0 = tn / (tn + fp) if (tn + fp) else 0
    p1 = tp / (tp + fp) if (tp + fp) else 0
    r1 = tp / (tp + fn) if (tp + fn) else 0
    return p0, r0, p1, r1


# ============================================================
# 4. Load data
# ============================================================

print("================ visualize_results.py ================")
print("PROJECT_ROOT:", PROJECT_ROOT)
print("RESULTS_DIR:", RESULTS_DIR)
print("OUTPUT_PATH:", OUTPUT_PATH)

full, preds = load_student_results()
full, ALL_KEYS = maybe_load_teacher(full)


# ============================================================
# 5. Figure layout
# ============================================================

fig = plt.figure(figsize=(22, 28), facecolor="#0F1117")
fig.patch.set_facecolor("#0F1117")

gs = gridspec.GridSpec(
    4,
    3,
    figure=fig,
    hspace=0.50,
    wspace=0.35,
    top=0.93,
    bottom=0.04,
    left=0.07,
    right=0.97,
)

PANEL_BG = "#1A1D27"
GRID_COL = "#2E3148"
TEXT_COL = "#E8EAF6"
SUB_COL = "#9FA8DA"
TITLE_COL = "#FFFFFF"


def style_ax(ax, title, subtitle=""):
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)
    ax.tick_params(colors=TEXT_COL, labelsize=9)
    ax.xaxis.label.set_color(TEXT_COL)
    ax.yaxis.label.set_color(TEXT_COL)
    ax.grid(color=GRID_COL, linewidth=0.6, alpha=0.7)
    ax.set_title(title, color=TITLE_COL, fontsize=12, fontweight="bold", pad=8)
    if subtitle:
        ax.text(
            0.5,
            1.02,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            color=SUB_COL,
            fontsize=8,
        )


# ============================================================
# 6. Header
# ============================================================

n_test = len(preds[MODEL_KEYS[0]])
fig.text(
    0.5,
    0.965,
    "MUTE Bengali Meme Detection — Stage 2 Fine-Tuning Dashboard",
    ha="center",
    va="center",
    color=TITLE_COL,
    fontsize=20,
    fontweight="bold",
)
fig.text(
    0.5,
    0.951,
    f"MUTE fine-tuning from FHM checkpoints · Student models: test set n={n_test}"
    + (" · * Teacher: train split, valid predictions only" if "teacher" in ALL_KEYS else ""),
    ha="center",
    va="center",
    color=SUB_COL,
    fontsize=9,
)


# ============================================================
# 7. Row 0 — Metric comparison bars
# ============================================================

ax_bar = fig.add_subplot(gs[0, :2])
style_ax(
    ax_bar,
    "Test Metrics — Student Models" + (" & LLM Teacher*" if "teacher" in ALL_KEYS else ""),
    "Higher is better"
    + (" · * Teacher evaluated on train split — not directly comparable" if "teacher" in ALL_KEYS else ""),
)

metric_names = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
metric_labels = ["Accuracy", "Macro\nPrecision", "Macro\nRecall", "Macro\nF1"]
x = np.arange(len(metric_names))
w = 0.17 if len(ALL_KEYS) == 4 else 0.22
center_offset = (len(ALL_KEYS) - 1) / 2

for i, key in enumerate(ALL_KEYS):
    vals = [full[key][m] * 100 for m in metric_names]
    is_teacher = key == "teacher"
    bars = ax_bar.bar(
        x + (i - center_offset) * w,
        vals,
        w,
        color=COLORS[key],
        alpha=0.88,
        label=LABELS[key],
        hatch="////" if is_teacher else "",
        edgecolor=COLORS[key] if is_teacher else "none",
        linewidth=0.8,
        zorder=3,
    )
    for bar, v in zip(bars, vals):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{v:.1f}",
            ha="center",
            va="bottom",
            color=TEXT_COL,
            fontsize=7,
            fontweight="bold",
        )

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(metric_labels, color=TEXT_COL, fontsize=10)
ax_bar.set_ylim(0, 100)
ax_bar.set_ylabel("Score (%)", color=TEXT_COL)
ax_bar.legend(facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL, fontsize=9, loc="upper right")


# ============================================================
# 8. Row 0 Col 2 — Test loss if available
# ============================================================

ax_loss = fig.add_subplot(gs[0, 2])
style_ax(ax_loss, "Validation/Test Loss", "Shown only if saved in metric files")

losses = [full[k].get("test_loss", None) for k in MODEL_KEYS]
if all(v is not None for v in losses):
    bars_l = ax_loss.bar(
        range(3),
        losses,
        color=[COLORS[k] for k in MODEL_KEYS],
        alpha=0.88,
        zorder=3,
    )
    for bar, v in zip(bars_l, losses):
        ax_loss.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.03,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            color=TEXT_COL,
            fontsize=8.5,
            fontweight="bold",
        )
    ax_loss.set_ylabel("Loss", color=TEXT_COL)
else:
    ax_loss.text(
        0.5,
        0.5,
        "Loss not saved\nby current evaluate.py",
        ha="center",
        va="center",
        color=SUB_COL,
        fontsize=12,
        transform=ax_loss.transAxes,
    )
    ax_loss.set_yticks([])

ax_loss.set_xticks(range(3))
ax_loss.set_xticklabels([LABELS[k].replace(" ", "\n") for k in MODEL_KEYS], color=TEXT_COL, fontsize=8)


# ============================================================
# 9. Row 1 — ROC curves
# ============================================================

ax_roc = fig.add_subplot(gs[1, 0])
style_ax(ax_roc, "ROC Curves — Student Models")
ax_roc.plot([0, 1], [0, 1], "--", color="#555877", linewidth=1, label="Random (AUC=0.50)")

for key in MODEL_KEYS:
    df = preds[key]
    prob_col = prediction_prob_column(df)
    label_col = prediction_label_column(df)
    fpr, tpr, _ = roc_curve(df[label_col], df[prob_col])
    ax_roc.plot(
        fpr,
        tpr,
        color=COLORS[key],
        linewidth=2,
        label=f"{LABELS[key]} (AUC={auc(fpr, tpr):.3f})",
    )

ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.legend(facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL, fontsize=7.5, loc="lower right")


# ============================================================
# 10. Row 1 — Precision-Recall curves
# ============================================================

ax_pr = fig.add_subplot(gs[1, 1])
style_ax(ax_pr, "Precision-Recall Curves — Student Models")

label_col_base = prediction_label_column(preds[MODEL_KEYS[0]])
base_rate = preds[MODEL_KEYS[0]][label_col_base].mean()
ax_pr.axhline(base_rate, linestyle="--", color="#555877", linewidth=1, label=f"Baseline (p={base_rate:.2f})")

for key in MODEL_KEYS:
    df = preds[key]
    prob_col = prediction_prob_column(df)
    label_col = prediction_label_column(df)
    prec, rec, _ = precision_recall_curve(df[label_col], df[prob_col])
    ax_pr.plot(
        rec,
        prec,
        color=COLORS[key],
        linewidth=2,
        label=f"{LABELS[key]} (AUC={auc(rec, prec):.3f})",
    )

ax_pr.set_xlabel("Recall")
ax_pr.set_ylabel("Precision")
ax_pr.legend(facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL, fontsize=7.5, loc="upper right")


# ============================================================
# 11. Row 1 Col 2 — Hate-score distribution
# ============================================================

ax_dist = fig.add_subplot(gs[1, 2])
style_ax(ax_dist, "Hate-Score Distributions", "Context Distillation model · MUTE test set")

df_cd = preds["context_distillation"]
prob_col = prediction_prob_column(df_cd)
label_col = prediction_label_column(df_cd)
bins = np.linspace(0, 1, 35)

for lbl, clr, name in [(0, "#457B9D", "Not Hateful"), (1, "#E76F51", "Hateful")]:
    subset = df_cd[df_cd[label_col] == lbl][prob_col]
    ax_dist.hist(
        subset,
        bins=bins,
        alpha=0.65,
        color=clr,
        label=f"{name} (n={len(subset)})",
        density=True,
    )

ax_dist.axvline(0.5, linestyle="--", color=TEXT_COL, linewidth=1, alpha=0.6)
ax_dist.set_xlabel("P(hateful)")
ax_dist.set_ylabel("Density")
ax_dist.legend(facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL, fontsize=8)


# ============================================================
# 12. Row 2 — Confusion matrices
# ============================================================

cmap_purple = LinearSegmentedColormap.from_list("dark_purple", ["#0F1117", "#3A0075", "#BF80FF"])
cmap_blues = LinearSegmentedColormap.from_list("dark_blue", ["#0F1117", "#1A4B8C", "#4FC3F7"])
cmap_oranges = LinearSegmentedColormap.from_list("dark_orange", ["#0F1117", "#7B3200", "#FFB347"])
cmap_teals = LinearSegmentedColormap.from_list("dark_teal", ["#0F1117", "#0A5045", "#4DD0C4"])

CMAPS = {
    "memeblip2_only": cmap_oranges,
    "context_only": cmap_blues,
    "context_distillation": cmap_teals,
    "teacher": cmap_purple,
}

gs_cm = gridspec.GridSpecFromSubplotSpec(1, len(ALL_KEYS), subplot_spec=gs[2, :], wspace=0.38)

for col_i, key in enumerate(ALL_KEYS):
    ax_cm = fig.add_subplot(gs_cm[0, col_i])
    cm = np.array(full[key]["confusion_matrix"])
    total = cm.sum()
    cm_norm = cm / max(total, 1)

    ax_cm.set_facecolor(PANEL_BG)
    ax_cm.imshow(cm_norm, cmap=CMAPS[key], vmin=0, vmax=0.5)

    for r in range(2):
        for c in range(2):
            color = "white" if cm_norm[r, c] < 0.25 else PANEL_BG
            ax_cm.text(
                c,
                r,
                f"{cm[r, c]}\n({cm_norm[r, c] * 100:.1f}%)",
                ha="center",
                va="center",
                color=color,
                fontsize=9,
                fontweight="bold",
            )

    acc = full[key]["accuracy"] * 100
    note = "\n(train split)" if key == "teacher" else ""
    ax_cm.set_title(
        f"{LABELS[key]}{note}\nAcc = {acc:.1f}%",
        color=COLORS[key],
        fontsize=9,
        fontweight="bold",
        pad=6,
    )
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["Pred:\nNot Hate", "Pred:\nHate"], color=TEXT_COL, fontsize=7)
    ax_cm.set_yticklabels(["True:\nNot Hate", "True:\nHate"], color=TEXT_COL, fontsize=7, rotation=90, va="center")
    ax_cm.tick_params(length=0)

    for spine in ax_cm.spines.values():
        spine.set_edgecolor(COLORS[key])
        spine.set_linewidth(1.5)


# ============================================================
# 13. Row 3 — Per-class P/R
# ============================================================

ax_cls = fig.add_subplot(gs[3, :2])
style_ax(
    ax_cls,
    "Per-Class Precision & Recall"
    + ("  (students: test set · teacher*: train split)" if "teacher" in ALL_KEYS else ""),
    "Class 0 = Not Hateful · Class 1 = Hateful",
)

sub_labels = ["Class 0\nPrecision", "Class 0\nRecall", "Class 1\nPrecision", "Class 1\nRecall"]
x_cls = np.arange(4)
w4 = 0.17 if len(ALL_KEYS) == 4 else 0.22
center_offset = (len(ALL_KEYS) - 1) / 2

for i, key in enumerate(ALL_KEYS):
    p0, r0, p1, r1 = per_class_pr(full[key]["confusion_matrix"])
    vals = [p0 * 100, r0 * 100, p1 * 100, r1 * 100]
    is_teacher = key == "teacher"

    bars_c = ax_cls.bar(
        x_cls + (i - center_offset) * w4,
        vals,
        w4,
        color=COLORS[key],
        alpha=0.88,
        label=LABELS[key],
        hatch="////" if is_teacher else "",
        edgecolor=COLORS[key] if is_teacher else "none",
        linewidth=0.8,
        zorder=3,
    )

    for bar, v in zip(bars_c, vals):
        ax_cls.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{v:.1f}",
            ha="center",
            va="bottom",
            color=TEXT_COL,
            fontsize=7,
            fontweight="bold",
        )

ax_cls.set_xticks(x_cls)
ax_cls.set_xticklabels(sub_labels, color=TEXT_COL, fontsize=10)
ax_cls.set_ylim(0, 100)
ax_cls.set_ylabel("Score (%)", color=TEXT_COL)
ax_cls.axvline(1.5, color=GRID_COL, linewidth=1.5, linestyle="--")
ax_cls.legend(facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL, fontsize=9, loc="upper right")


# ============================================================
# 14. Row 3 Col 2 — Radar chart
# ============================================================

ax_radar = fig.add_subplot(gs[3, 2], polar=True)
ax_radar.set_facecolor(PANEL_BG)

radar_metrics = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
radar_labels = ["Accuracy", "Precision", "Recall", "F1"]
N = len(radar_metrics)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

ax_radar.spines["polar"].set_color(GRID_COL)
ax_radar.tick_params(colors=TEXT_COL)
ax_radar.set_xticks(angles[:-1])
ax_radar.set_xticklabels(radar_labels, color=TEXT_COL, fontsize=9)
ax_radar.set_ylim(0, 100)
ax_radar.set_yticks([20, 40, 60, 80, 100])
ax_radar.set_yticklabels(["20", "40", "60", "80", "100"], color=SUB_COL, fontsize=7)
ax_radar.yaxis.grid(True, color=GRID_COL, linewidth=0.6)
ax_radar.xaxis.grid(True, color=GRID_COL, linewidth=0.6)

for key in ALL_KEYS:
    vals = [full[key][m] * 100 for m in radar_metrics]
    vals += vals[:1]
    is_teacher = key == "teacher"

    ax_radar.plot(
        angles,
        vals,
        color=COLORS[key],
        linewidth=2,
        linestyle="--" if is_teacher else "-",
        label=LABELS[key],
    )
    ax_radar.fill(angles, vals, color=COLORS[key], alpha=0.10)

ax_radar.set_title("Metric Radar", color=TITLE_COL, fontsize=10, fontweight="bold", pad=18)
ax_radar.legend(
    facecolor=PANEL_BG,
    edgecolor=GRID_COL,
    labelcolor=TEXT_COL,
    fontsize=7.5,
    loc="upper right",
    bbox_to_anchor=(1.4, 1.15),
)


# ============================================================
# 15. Save
# ============================================================

plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved -> {OUTPUT_PATH}")
plt.close()
