from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
import platform
import random
import re
import time
from typing import Any
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

from kg_punctuation.hf_token_experiment import (
    ID_TO_LABEL,
    LABELS,
    LABEL_TO_ID,
    AprTokenDataset,
    compute_class_weights,
    evaluate,
    load_documents,
    train_one_epoch,
)


def run_hf_mixed_experiment(
    wikipedia_dataset_path: Path,
    news_dataset_path: Path,
    output_dir: Path,
    model_id: str = "xlm-roberta-base",
    max_documents_per_source: int = 1000,
    max_length: int = 128,
    batch_size: int = 8,
    epochs: int = 3,
    learning_rate: float = 3e-5,
    seed: int = 42,
) -> dict[str, Any]:
    set_seed(seed)
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_id.split("/")[-1])
    run_id = f"mixed_{safe_model}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    run_dir = output_dir / run_id
    figures_dir = run_dir / "figures"
    tables_dir = run_dir / "tables"
    models_dir = run_dir / "models"
    for path in [figures_dir, tables_dir, models_dir]:
        path.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    wiki_split = load_split(
        dataset_path=wikipedia_dataset_path,
        source_name="wikipedia_ky",
        max_documents=max_documents_per_source,
        seed=seed,
    )
    news_split = load_split(
        dataset_path=news_dataset_path,
        source_name="kyrgyznews",
        max_documents=max_documents_per_source,
        seed=seed,
    )

    train_docs = wiki_split["train_docs"] + news_split["train_docs"]
    valid_docs = wiki_split["valid_docs"] + news_split["valid_docs"]
    random.Random(seed).shuffle(train_docs)
    random.Random(seed + 1).shuffle(valid_docs)

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    train_dataset = AprTokenDataset(train_docs, tokenizer, max_length=max_length)
    valid_dataset = AprTokenDataset(valid_docs, tokenizer, max_length=max_length)
    test_datasets = {
        "wikipedia_ky": AprTokenDataset(wiki_split["test_docs"], tokenizer, max_length=max_length),
        "kyrgyznews": AprTokenDataset(news_split["test_docs"], tokenizer, max_length=max_length),
    }

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loaders = {
        source: DataLoader(dataset, batch_size=batch_size)
        for source, dataset in test_datasets.items()
    }

    config = AutoConfig.from_pretrained(
        model_id,
        num_labels=len(LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_id,
        config=config,
        ignore_mismatched_sizes=True,
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    class_weights = compute_class_weights(train_docs).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)

    history = []
    best_macro_f1 = -1.0
    best_path = models_dir / "best_model.pt"
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        valid_metrics = evaluate(model, valid_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "valid_accuracy": valid_metrics["accuracy"],
            "valid_macro_f1": valid_metrics["macro_f1"],
            "valid_comma_f1": valid_metrics["per_class"]["C"]["f1"],
            "seconds": round(time.perf_counter() - epoch_start, 3),
        }
        history.append(row)
        if valid_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = valid_metrics["macro_f1"]
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device))
    raw_metrics = {
        source: evaluate(model, loader, device)
        for source, loader in test_loaders.items()
    }
    save_outputs(
        model=model,
        tokenizer=tokenizer,
        history=history,
        raw_metrics=raw_metrics,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
        models_dir=models_dir,
    )

    metrics = {
        source: slim_metrics(values)
        for source, values in raw_metrics.items()
    }
    metadata = {
        "run_id": run_id,
        "model_family": "hf_mixed_domain_token_classification",
        "model_id": model_id,
        "dataset_paths": {
            "wikipedia_ky": str(wikipedia_dataset_path),
            "kyrgyznews": str(news_dataset_path),
        },
        "output_dir": str(run_dir),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "seed": seed,
        "device": str(device),
        "hyperparameters": {
            "max_documents_per_source": max_documents_per_source,
            "max_length": max_length,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "class_weights": {label: round(float(class_weights[i].cpu()), 6) for i, label in enumerate(LABELS)},
        },
        "source_splits": {
            "wikipedia_ky": split_metadata(wiki_split),
            "kyrgyznews": split_metadata(news_split),
        },
        "combined_split": {
            "train_documents": len(train_docs),
            "validation_documents": len(valid_docs),
            "train_sequences": len(train_dataset),
            "validation_sequences": len(valid_dataset),
        },
        "history": history,
        "metrics": {"hf_mixed_domain": metrics},
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_split(dataset_path: Path, source_name: str, max_documents: int, seed: int) -> dict[str, Any]:
    documents, load_stats = load_documents(dataset_path, max_documents=max_documents, seed=seed)
    train_docs, temp_docs = train_test_split(documents, test_size=0.35, random_state=seed)
    valid_docs, test_docs = train_test_split(temp_docs, test_size=25 / 35, random_state=seed)
    return {
        "source_name": source_name,
        "dataset_path": str(dataset_path),
        "load_stats": load_stats,
        "train_docs": train_docs,
        "valid_docs": valid_docs,
        "test_docs": test_docs,
    }


def split_metadata(split: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_path": split["dataset_path"],
        "load_stats": split["load_stats"],
        "train_documents": len(split["train_docs"]),
        "validation_documents": len(split["valid_docs"]),
        "test_documents": len(split["test_docs"]),
        "train_label_counts": count_labels(split["train_docs"]),
        "validation_label_counts": count_labels(split["valid_docs"]),
        "test_label_counts": count_labels(split["test_docs"]),
    }


def count_labels(documents: list[dict[str, list[str]]]) -> dict[str, int]:
    return dict(Counter(label for doc in documents for label in doc["labels"]))


def save_outputs(
    model,
    tokenizer,
    history: list[dict[str, Any]],
    raw_metrics: dict[str, dict[str, Any]],
    figures_dir: Path,
    tables_dir: Path,
    models_dir: Path,
) -> None:
    model.save_pretrained(models_dir / "hf_model")
    tokenizer.save_pretrained(models_dir / "hf_model")

    with (tables_dir / "epoch_history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    with (tables_dir / "summary_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "target_domain",
                "model",
                "accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "weighted_f1",
                "s_f1",
                "comma_f1",
                "period_f1",
            ],
        )
        writer.writeheader()
        for source, metrics in raw_metrics.items():
            report = classification_report(
                metrics["y_true"],
                metrics["y_pred"],
                labels=LABELS,
                zero_division=0,
                output_dict=True,
            )
            (tables_dir / f"{source}_classification_report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            cm = confusion_matrix(metrics["y_true"], metrics["y_pred"], labels=LABELS)
            np.savetxt(tables_dir / f"{source}_confusion_matrix.csv", cm, delimiter=",", fmt="%d")
            plot_confusion_matrix(cm, source, figures_dir / f"{source}_confusion_matrix.png")
            slim = slim_metrics(metrics)
            writer.writerow(
                {
                    "target_domain": source,
                    "model": "hf_mixed_domain",
                    "accuracy": slim["accuracy"],
                    "macro_precision": slim["macro_precision"],
                    "macro_recall": slim["macro_recall"],
                    "macro_f1": slim["macro_f1"],
                    "weighted_f1": slim["weighted_f1"],
                    "s_f1": slim["per_class"]["S"]["f1"],
                    "comma_f1": slim["per_class"]["C"]["f1"],
                    "period_f1": slim["per_class"]["P"]["f1"],
                }
            )

    plot_history(history, figures_dir / "hf_mixed_domain_training_history.png")


def slim_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key not in {"y_true", "y_pred"}}


def plot_confusion_matrix(cm: np.ndarray, source: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS).plot(
        ax=ax,
        cmap="Blues",
        colorbar=False,
        values_format="d",
    )
    ax.set_title(f"KG-Punct-XLMR-Mixed on {source}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_history(history: list[dict[str, Any]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    epochs = [row["epoch"] for row in history]
    ax.plot(epochs, [row["valid_macro_f1"] for row in history], marker="o", label="valid macro-F1")
    ax.plot(epochs, [row["valid_comma_f1"] for row in history], marker="o", label="valid comma F1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1")
    ax.set_title("KG-Punct-XLMR-Mixed validation metrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_report(metadata: dict[str, Any], output_path: Path) -> None:
    metrics = metadata["metrics"]["hf_mixed_domain"]
    lines = [
        f"# Mixed-Domain HF Token APR Report: {metadata['run_id']}",
        "",
        f"- Model: `{metadata['model_id']}`",
        "- Paper name: `KG-Punct-XLMR-Mixed`",
        f"- Device: `{metadata['device']}`",
        f"- Total runtime seconds: {metadata['runtime_seconds_total']}",
        f"- Train documents: {metadata['combined_split']['train_documents']}",
        f"- Validation documents: {metadata['combined_split']['validation_documents']}",
        "",
        "## Test Metrics",
        "",
        "| Target Domain | Accuracy | S F1 | C F1 | P F1 | Macro-F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for source, values in metrics.items():
        lines.append(
            "| "
            f"{source} | "
            f"{values['accuracy']} | "
            f"{values['per_class']['S']['f1']} | "
            f"{values['per_class']['C']['f1']} | "
            f"{values['per_class']['P']['f1']} | "
            f"{values['macro_f1']} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This run trains one mixed-domain checkpoint on Wikipedia and KyrgyzNews train splits.",
            "- Test metrics are reported separately on the held-out Wikipedia and KyrgyzNews test splits.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
