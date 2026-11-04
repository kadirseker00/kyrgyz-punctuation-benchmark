"""Regenerate the manuscript figures (R1.5: every panel has both axis titles and tick labels).

Usage: PYTHONPATH=src .venv/bin/python -m kg_punctuation.paper_figures [--output-dir DIR]

Inputs are the canonical v2 benchmark outputs, so the figures carry the same numbers as the
manuscript tables:
- reports/v2/data_stats.json                                     -> label_distribution
- reports/v2/main_panel.csv                                      -> model_comparison_f1
- experiments/v2_main/external_last/predictions_test_<source>.csv -> external_confusion
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "paper/R2/ACM_KG_puctuation/figures"

# Validated with the dataviz palette checker (light surface): comma/period pass CVD
# and contrast; macro-F1/comma likewise. "No punctuation" is deliberately neutral.
COLOR_S = "#c8c7c2"
COLOR_C = "#2a78d6"
COLOR_P = "#eb6834"
COLOR_MACRO = "#4a3aa7"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"

DOMAINS = [("wikipedia_ky", "Kyrgyz Wikipedia"), ("kyrgyznews", "Kyrgyz News")]
CLASS_TICKS = ["S\n(none)", "C\n(comma)", "P\n(period)"]

# Strongest member of each family per domain (by within-domain macro-F1 in reports/v2/main_panel.csv),
# plus our checkpoints and the external one: (display name, is ours, candidate jobs).
F1_GROUPS = [
    ("Classical", False, ["logreg_balanced", "sgd_log_balanced"]),
    ("CNN", False, ["cnn_small", "cnn_deep"]),
    ("Kyrgyz encoder", False, ["kyrgyzbert", "kyrgyzbert2"]),
    ("KG-Punct-XLMR", True, ["xlmr"]),
]
DISPLAY = {"logreg_balanced": "Balanced LR", "sgd_log_balanced": "Balanced SGD-LR", "cnn_small": "CNN 2×128",
           "cnn_deep": "CNN 4×256", "kyrgyzbert": "KyrgyzBERT", "kyrgyzbert2": "KyrgyzBERT2", "xlmr": "KG-Punct-XLMR"}
SHORT = {"kyrgyznews": "news", "wikipedia_ky": "wiki"}


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.edgecolor": INK_MUTED,
        "axes.labelcolor": INK,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,  # embed TrueType fonts (ACM production requirement)
        "ps.fonttype": 42,
    })


def _save(fig: plt.Figure, output_dir: Path, name: str) -> list[str]:
    paths = []
    for ext in ("pdf", "png"):
        path = output_dir / f"{name}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def label_distribution(output_dir: Path) -> list[str]:
    stats = json.loads((ROOT / "reports/v2/data_stats.json").read_text())
    fig, ax = plt.subplots(figsize=(6.5, 2.3))
    names = [name for _, name in DOMAINS]
    y = np.arange(len(DOMAINS))[::-1]
    for yi, (source_id, _) in zip(y, DOMAINS):
        left = 0.0
        for label, color, text_color in (("S", COLOR_S, INK), ("C", COLOR_C, "white"), ("P", COLOR_P, INK)):
            share = 100 * stats[source_id]["core_share"][label]
            ax.barh(yi, share, left=left, height=0.62, color=color, edgecolor="white", linewidth=1.5)
            text = f"{label} {share:.1f}%" if label == "S" else f"{share:.1f}%"
            if share >= 7:
                ax.text(left + share / 2, yi, text, ha="center", va="center", color=text_color, fontsize=8.5)
            else:  # too narrow for an inside label: place it directly above its own segment
                ax.text(left + share / 2, yi + 0.36, text, ha="center", va="bottom", color=INK, fontsize=8.5)
            left += share
    ax.set_yticks(y, names)
    ax.set_ylim(-0.55, len(DOMAINS) - 0.3)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of tokens (%)")
    ax.set_ylabel("Corpus")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (COLOR_S, COLOR_C, COLOR_P)]
    ax.legend(handles, ["S (no punctuation)", "C (comma)", "P (period)"], ncol=3, frameon=False,
              loc="lower center", bbox_to_anchor=(0.5, 1.0))
    return _save(fig, output_dir, "label_distribution")


def model_comparison_f1(output_dir: Path) -> list[str]:
    panel = list(csv.DictReader((ROOT / "reports/v2/main_panel.csv").open()))

    def score(job: str, test: str, key: str) -> float:
        return 100 * float(next(r[key] for r in panel if r["job"] == job and r["test"] == test))

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    width = 0.38
    for ax, (domain, title) in zip(axes, DOMAINS):
        entries = []
        for _, ours, prefixes in F1_GROUPS:
            best = max(prefixes, key=lambda p: score(f"{p}_{SHORT[domain]}", domain, "macro_f1"))
            entries.append((DISPLAY[best], ours, f"{best}_{SHORT[domain]}"))
        entries.append(("KG-Punct-XLMR-Mixed", True, "xlmr_mixed"))
        entries.append(("KyrgyzBERT-PR (zero-shot)", False, "external_last"))
        x = np.arange(len(entries))
        macro = [score(job, domain, "macro_f1") for _, _, job in entries]
        comma = [score(job, domain, "C_f1") for _, _, job in entries]
        for offset, values, color in ((-width / 2, macro, COLOR_MACRO), (width / 2, comma, COLOR_C)):
            bars = ax.bar(x + offset, values, width, color=color, edgecolor="white", linewidth=1)
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.1f}",
                        ha="center", va="bottom", fontsize=6, color=INK)
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x, [name for name, _, _ in entries], rotation=45, ha="right", rotation_mode="anchor", fontsize=7.5)
        for tick, (_, ours, _) in zip(ax.get_xticklabels(), entries):
            if ours:
                tick.set_fontweight("bold")
        ax.set_ylim(0, 100)
        ax.set_ylabel("F1 score (%)")
        ax.set_xlabel("Model")
        ax.yaxis.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (COLOR_MACRO, COLOR_C)]
    fig.legend(handles, ["Macro-F1", "Comma F1"], ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=2.5)
    return _save(fig, output_dir, "model_comparison_f1")


def external_confusion(output_dir: Path) -> list[str]:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), layout="constrained")
    image = None
    for ax, (domain, title) in zip(axes, DOMAINS):
        path = ROOT / "experiments/v2_main/external_last" / f"predictions_test_{domain}.csv"
        core = {"S": 0, "C": 1, "P": 2, "Q": 2, "E": 2}  # v2 core view
        counts = np.zeros((3, 3))
        for row in csv.DictReader(path.open(encoding="utf-8")):
            counts[core[row["gold"]], core[row["prediction"]]] += 1
        shares = 100 * counts / counts.sum(axis=1, keepdims=True)
        image = ax.imshow(shares, cmap="Blues", vmin=0, vmax=100)
        for i in range(3):
            for j in range(3):
                color = "white" if shares[i, j] > 55 else INK
                ax.text(j, i, f"{shares[i, j]:.1f}%\n({int(counts[i, j]):,})", ha="center", va="center",
                        fontsize=7.5, color=color)
        ax.set_xticks(range(3), CLASS_TICKS, fontsize=8)
        ax.set_yticks(range(3), CLASS_TICKS, fontsize=8)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(title)
        ax.spines[:].set_visible(False)
        ax.tick_params(length=0)
    fig.get_layout_engine().set(w_pad=0.12)
    cbar = fig.colorbar(image, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label("Share of true-label row (%)")
    cbar.outline.set_visible(False)
    return _save(fig, output_dir, "external_confusion")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _style()
    for build in (label_distribution, model_comparison_f1, external_confusion):
        for path in build(args.output_dir):
            print(path)


if __name__ == "__main__":
    main()
