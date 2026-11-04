from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
import platform
import re
import time
from typing import Any
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer

from kg_punctuation.hf_token_experiment import (
    LABELS,
    AprTokenDataset,
    evaluate,
    load_documents,
)


def evaluate_hf_cross_domain(
    model_dir: Path,
    target_dataset_path: Path,
    output_dir: Path,
    source_name: str,
    target_name: str,
    max_documents: int = 1000,
    max_length: int = 128,
    batch_size: int = 8,
    seed: int = 42,
) -> dict[str, Any]:
    safe_source = re.sub(r"[^A-Za-z0-9_.-]+", "_", source_name)
    safe_target = re.sub(r"[^A-Za-z0-9_.-]+", "_", target_name)
    run_id = f"cross_{safe_source}_to_{safe_target}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    run_dir = output_dir / run_id
    figures_dir = run_dir / "figures"
    tables_dir = run_dir / "tables"
    for path in [figures_dir, tables_dir]:
        path.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    documents, load_stats = load_documents(target_dataset_path, max_documents=max_documents, seed=seed)
    target_train_docs, target_temp_docs = train_test_split(documents, test_size=0.35, random_state=seed)
    target_valid_docs, target_test_docs = train_test_split(target_temp_docs, test_size=25 / 35, random_state=seed)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)

    test_dataset = AprTokenDataset(target_test_docs, tokenizer, max_length=max_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    metrics = evaluate(model, test_loader, device)
    save_outputs(metrics=metrics, figures_dir=figures_dir, tables_dir=tables_dir)

    metadata = {
        "run_id": run_id,
        "model_family": "hf_cross_domain_evaluation",
        "source_name": source_name,
        "target_name": target_name,
        "model_dir": str(model_dir),
        "target_dataset_path": str(target_dataset_path),
        "output_dir": str(run_dir),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "seed": seed,
        "device": str(device),
        "hyperparameters": {
            "max_documents": max_documents,
            "max_length": max_length,
            "batch_size": batch_size,
        },
        "load_stats": load_stats,
        "target_split": {
            "train_documents_reserved": len(target_train_docs),
            "validation_documents_reserved": len(target_valid_docs),
            "test_documents": len(target_test_docs),
            "test_sequences": len(test_dataset),
            "test_label_counts": dict(Counter(label for doc in target_test_docs for label in doc["labels"])),
        },
        "metrics": {"hf_cross_domain": metrics},
        "runtime_seconds_total": round(time.perf_counter() - start, 3),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "torch": torch.__version__,
        },
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(metadata, run_dir / "report.md")
    return {"run_id": run_id, "output_dir": str(run_dir)}


def save_outputs(metrics: dict[str, Any], figures_dir: Path, tables_dir: Path) -> None:
    report = classification_report(metrics["y_true"], metrics["y_pred"], labels=LABELS, zero_division=0, output_dict=True)
    (tables_dir / "hf_cross_domain_classification_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    cm = confusion_matrix(metrics["y_true"], metrics["y_pred"], labels=LABELS)
    np.savetxt(tables_dir / "hf_cross_domain_confusion_matrix.csv", cm, delimiter=",", fmt="%d")
    plot_confusion_matrix(cm, figures_dir / "hf_cross_domain_confusion_matrix.png")

    slim_metrics = {key: value for key, value in metrics.items() if key not in {"y_true", "y_pred"}}
    (tables_dir / "summary_metrics.csv").write_text(
        "model,accuracy,macro_precision,macro_recall,macro_f1,weighted_f1,comma_f1\n"
        f"hf_cross_domain,{slim_metrics['accuracy']},{slim_metrics['macro_precision']},"
        f"{slim_metrics['macro_recall']},{slim_metrics['macro_f1']},{slim_metrics['weighted_f1']},"
        f"{slim_metrics['per_class']['C']['f1']}\n",
        encoding="utf-8",
    )
    metrics.clear()
    metrics.update(slim_metrics)


def plot_confusion_matrix(cm: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS).plot(
        ax=ax,
        cmap="Blues",
        colorbar=False,
        values_format="d",
    )
    ax.set_title("Cross-domain confusion matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_report(metadata: dict[str, Any], output_path: Path) -> None:
    metrics = metadata["metrics"]["hf_cross_domain"]
    lines = [
        f"# HF Cross-Domain APR Report: {metadata['run_id']}",
        "",
        f"- Source domain: `{metadata['source_name']}`",
        f"- Target domain: `{metadata['target_name']}`",
        f"- Model directory: `{metadata['model_dir']}`",
        f"- Target dataset: `{metadata['target_dataset_path']}`",
        f"- Device: `{metadata['device']}`",
        f"- Runtime seconds: {metadata['runtime_seconds_total']}",
        f"- Target documents loaded before split: {metadata['load_stats']['documents_loaded']}",
        f"- Target test documents: {metadata['target_split']['test_documents']}",
        "",
        "## Test Metrics",
        "",
        "| Model | Accuracy | S F1 | C F1 | P F1 | Macro-F1 |",
        "|---|---:|---:|---:|---:|---:|",
        (
            "| hf_cross_domain | "
            f"{metrics['accuracy']} | "
            f"{metrics['per_class']['S']['f1']} | "
            f"{metrics['per_class']['C']['f1']} | "
            f"{metrics['per_class']['P']['f1']} | "
            f"{metrics['macro_f1']} |"
        ),
        "",
        "## Notes",
        "",
        "- This is zero-shot cross-domain evaluation of a fine-tuned source-domain checkpoint.",
        "- The target train/validation split is reserved only to select the same deterministic target test split.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
