from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
SOURCE_METADATA = [
    ROOT / "data/processed/source_level/wikipedia_ky.metadata.json",
    ROOT / "data/processed/source_level/kyrgyznews.metadata.json",
]


def write_dataset_report(output_dir: Path) -> list[str]:
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    rows = [_load_metadata(path) for path in SOURCE_METADATA if path.exists()]
    if not rows:
        raise FileNotFoundError("No source metadata files found.")

    summary_json = tables_dir / "dataset_source_summary.json"
    summary_csv = tables_dir / "dataset_source_summary.csv"
    class_plot = figures_dir / "dataset_label_distribution.png"
    token_plot = figures_dir / "dataset_token_counts.png"
    markdown = output_dir / "dataset_report.md"

    summary_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(rows, summary_csv)
    _plot_label_distribution(rows, class_plot)
    _plot_token_counts(rows, token_plot)
    _write_markdown(rows, markdown)

    return [str(summary_json), str(summary_csv), str(class_plot), str(token_plot), str(markdown)]


def _load_metadata(path: Path) -> dict[str, object]:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    labels = metadata["labels"]
    tokens = metadata["tokens"]
    return {
        "source_id": metadata["source_id"],
        "domain": metadata["domain"],
        "documents": metadata["documents"],
        "tokens": tokens,
        "S": labels["S"],
        "C": labels["C"],
        "P": labels["P"],
        "S_ratio": round(labels["S"] / tokens, 6),
        "C_ratio": round(labels["C"] / tokens, 6),
        "P_ratio": round(labels["P"] / tokens, 6),
        "runtime_seconds": metadata["runtime_seconds"],
        "raw_path": metadata["raw_path"],
        "output_path": metadata["output_path"],
    }


def _write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    fieldnames = [
        "source_id",
        "domain",
        "documents",
        "tokens",
        "S",
        "C",
        "P",
        "S_ratio",
        "C_ratio",
        "P_ratio",
        "runtime_seconds",
        "raw_path",
        "output_path",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_label_distribution(rows: list[dict[str, object]], output_path: Path) -> None:
    labels = ["S_ratio", "C_ratio", "P_ratio"]
    x = range(len(rows))
    bottom = [0.0] * len(rows)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    for label, color in zip(labels, colors):
        values = [float(row[label]) for row in rows]
        ax.bar(x, values, bottom=bottom, label=label.replace("_ratio", ""), color=color)
        bottom = [base + value for base, value in zip(bottom, values)]

    ax.set_xticks(list(x))
    ax.set_xticklabels([str(row["source_id"]) for row in rows])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Label ratio")
    ax.set_title("S/C/P label distribution by source")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_token_counts(rows: list[dict[str, object]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar([str(row["source_id"]) for row in rows], [int(row["tokens"]) for row in rows], color="#4C78A8")
    ax.set_ylabel("Tokens")
    ax.set_title("Processed token counts by source")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_markdown(rows: list[dict[str, object]], output_path: Path) -> None:
    lines = [
        "# Dataset Source Report",
        "",
        "| Source | Domain | Documents | Tokens | S % | C % | P % | Build seconds |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['source_id']} | {row['domain']} | {row['documents']} | {row['tokens']} | "
            f"{float(row['S_ratio']) * 100:.2f} | {float(row['C_ratio']) * 100:.2f} | "
            f"{float(row['P_ratio']) * 100:.2f} | {row['runtime_seconds']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- `wikipedia_ky` gives formal encyclopedic prose and a higher comma/period share.",
            "- `kyrgyznews` gives contemporary news prose and a more space-dominant label distribution.",
            "- These two sources support within-domain and cross-domain APR comparisons.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

