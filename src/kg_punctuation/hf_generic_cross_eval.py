from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import platform
import time
from typing import Any, Iterable
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer

from kg_punctuation.data.labeling import resolve_label_set
from kg_punctuation.hf_generic_experiment import (
    TokenClassificationDataset,
    evaluate,
    evaluate_canonical,
    load_documents,
    select_device,
)


def evaluate_generic_cross_domain(
    model_dir: Path,
    target_dataset_path: Path,
    output_dir: Path,
    source_name: str,
    target_name: str,
    max_documents: int = 5000,
    max_length: int = 128,
    batch_size: int = 8,
    seed: int = 42,
    split_seed: int | None = None,
    label_set: str | Iterable[str] = "extended",
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    benchmark = None
    if manifest_path is not None:
        from kg_punctuation.benchmark_v2 import Benchmark

        if label_set not in ("core", "extended"):
            raise ValueError("Canonical runs require a core/extended label view.")
        benchmark = Benchmark(manifest_path)
        split_seed = benchmark.manifest["seed"]
    labels = list(resolve_label_set(label_set))
    split_seed = seed if split_seed is None else split_seed
    run_id = (
        f"cross_{source_name}_to_{target_name}_"
        f"{'-'.join(labels).lower()}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    )
    run_dir = output_dir / run_id
    figures_dir = run_dir / "figures"
    tables_dir = run_dir / "tables"
    for path in (figures_dir, tables_dir):
        path.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    if benchmark is None:
        documents, load_stats = load_documents(
            target_dataset_path, max_documents=max_documents, seed=split_seed, labels=labels,
        )
        _, target_temp_docs = train_test_split(documents, test_size=0.35, random_state=split_seed)
        _, target_test_docs = train_test_split(target_temp_docs, test_size=25 / 35, random_state=split_seed)
    else:
        target_test_docs = benchmark.documents(target_name, "test", label_set)
        target_dataset_path = Path(benchmark.manifest["sources"][target_name]["data_file"])
        load_stats = {"protocol": "canonical_v2"}

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    label_to_id = {label: index for index, label in enumerate(labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    test_dataset = TokenClassificationDataset(
        target_test_docs, tokenizer, max_length, label_to_id,
    ) if benchmark is None else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size) if test_dataset is not None else None

    device = select_device()
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.to(device)
    if benchmark is not None and {int(k): v for k, v in model.config.id2label.items()} != id_to_label:
        raise ValueError("Checkpoint label mapping does not match the requested label view.")
    metrics = (
        evaluate(model, test_loader, device, labels, id_to_label) if benchmark is None else
        evaluate_canonical(model, tokenizer, target_test_docs, benchmark, target_name, "test",
                           label_set, max_length, batch_size, device, id_to_label,
                           tables_dir / "predictions.csv")
    )
    save_outputs(metrics, labels, figures_dir, tables_dir, preserve_predictions=benchmark is not None)

    metadata = {
        "run_id": run_id,
        "model_family": "hf_token_classification_generic_cross_domain",
        "label_set": labels,
        "source_name": source_name,
        "target_name": target_name,
        "model_dir": str(model_dir),
        "target_dataset_path": str(target_dataset_path),
        "output_dir": str(run_dir),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "seed": seed,
        "split_seed": split_seed,
        "device": str(device),
        "hyperparameters": {
            "max_documents": max_documents,
            "max_length": max_length,
            "batch_size": batch_size,
        },
        "load_stats": load_stats,
        "target_split": {
            "test_documents": len(target_test_docs),
            "test_sequences": len(test_dataset) if test_dataset is not None else None,
            "test_label_counts": count_labels(target_test_docs, labels),
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
    if benchmark is not None:
        metadata["benchmark"] = {
            "version": "v2", "manifest_path": str(manifest_path),
            "manifest_sha256": hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
            "label_view": label_set,
            "source_sha256": benchmark.manifest["sources"][target_name]["sha256"],
            "test_document_ids": [doc["document_id"] for doc in target_test_docs],
            "full_word_coverage": True, "subtoken": "first",
        }
        metadata["hyperparameters"].pop("max_documents")
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(metadata, run_dir / "report.md")
    return {"run_id": run_id, "output_dir": str(run_dir)}


def count_labels(documents: list[dict[str, list[str]]], labels: list[str]) -> dict[str, int]:
    counts = Counter(label for document in documents for label in document["labels"])
    return {label: counts[label] for label in labels}


def save_outputs(
    metrics: dict[str, Any],
    labels: list[str],
    figures_dir: Path,
    tables_dir: Path,
    preserve_predictions: bool = False,
) -> None:
    report = classification_report(
        metrics["y_true"],
        metrics["y_pred"],
        labels=labels,
        zero_division=0,
        output_dict=True,
    )
    (tables_dir / "classification_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not preserve_predictions:
        with (tables_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["index", "gold", "prediction"])
            writer.writerows(
                (index, gold, prediction)
                for index, (gold, prediction) in enumerate(
                    zip(metrics["y_true"], metrics["y_pred"])
                )
            )
    cm = confusion_matrix(metrics["y_true"], metrics["y_pred"], labels=labels)
    np.savetxt(tables_dir / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")
    plot_confusion_matrix(cm, labels, figures_dir / "confusion_matrix.png")

    fieldnames = [
        "model",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
    ]
    for label in labels:
        fieldnames.extend([f"{label}_precision", f"{label}_recall", f"{label}_f1", f"{label}_support"])
    row = {"model": "hf_cross_domain"}
    row.update({key: metrics[key] for key in fieldnames if key in metrics})
    for label in labels:
        for metric_name in ("precision", "recall", "f1", "support"):
            row[f"{label}_{metric_name}"] = metrics["per_class"][label][metric_name]
    with (tables_dir / "summary_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)
    metrics.clear()
    metrics.update({key: value for key, value in row.items() if key != "model"})
    metrics["per_class"] = {
        label: {
            metric_name: row[f"{label}_{metric_name}"]
            for metric_name in ("precision", "recall", "f1", "support")
        }
        for label in labels
    }


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(
        ax=ax,
        cmap="Blues",
        colorbar=False,
        values_format="d",
    )
    ax.set_title("Cross-domain token-classification confusion matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_report(metadata: dict[str, Any], output_path: Path) -> None:
    metrics = metadata["metrics"]["hf_cross_domain"]
    labels = metadata["label_set"]
    lines = [
        f"# Generic cross-domain APR report: {metadata['run_id']}",
        "",
        f"- Source checkpoint: `{metadata['source_name']}`",
        f"- Target domain: `{metadata['target_name']}`",
        f"- Model directory: `{metadata['model_dir']}`",
        f"- Target dataset: `{metadata['target_dataset_path']}`",
        f"- Labels: `{', '.join(labels)}`",
        f"- Device: `{metadata['device']}`",
        f"- Runtime seconds: {metadata['runtime_seconds_total']}",
        "",
        "## Test metrics",
        "",
        "| Label | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in labels:
        values = metrics["per_class"][label]
        lines.append(
            f"| `{label}` | {values['precision']} | {values['recall']} | "
            f"{values['f1']} | {values['support']} |"
        )
    lines.extend(
        [
            "",
            f"- Accuracy: {metrics['accuracy']}",
            f"- Macro precision: {metrics['macro_precision']}",
            f"- Macro recall: {metrics['macro_recall']}",
            f"- Macro-F1: {metrics['macro_f1']}",
            f"- Weighted-F1: {metrics['weighted_f1']}",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
