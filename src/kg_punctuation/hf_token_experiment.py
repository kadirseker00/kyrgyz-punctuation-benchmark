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
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer


LABELS = ["S", "C", "P"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}
IGNORE_INDEX = -100


class AprTokenDataset(Dataset):
    def __init__(self, documents: list[dict[str, list[str]]], tokenizer, max_length: int):
        self.features = []
        for document in documents:
            encoding = tokenizer(
                document["tokens"],
                is_split_into_words=True,
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
            word_ids = encoding.word_ids()
            labels = []
            previous_word_id = None
            for word_id in word_ids:
                if word_id is None:
                    labels.append(IGNORE_INDEX)
                elif word_id != previous_word_id:
                    labels.append(LABEL_TO_ID[document["labels"][word_id]])
                else:
                    labels.append(IGNORE_INDEX)
                previous_word_id = word_id
            encoding["labels"] = labels
            self.features.append({key: torch.tensor(value, dtype=torch.long) for key, value in encoding.items()})

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.features[index]


def run_hf_token_experiment(
    dataset_path: Path,
    output_dir: Path,
    model_id: str,
    max_documents: int = 800,
    max_length: int = 128,
    batch_size: int = 8,
    epochs: int = 3,
    learning_rate: float = 3e-5,
    seed: int = 42,
) -> dict[str, Any]:
    set_seed(seed)
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_id.split("/")[-1])
    run_id = f"{dataset_path.stem}_{safe_model}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    run_dir = output_dir / run_id
    figures_dir = run_dir / "figures"
    tables_dir = run_dir / "tables"
    models_dir = run_dir / "models"
    for path in [figures_dir, tables_dir, models_dir]:
        path.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    documents, load_stats = load_documents(dataset_path, max_documents=max_documents, seed=seed)
    train_docs, temp_docs = train_test_split(documents, test_size=0.35, random_state=seed)
    valid_docs, test_docs = train_test_split(temp_docs, test_size=25 / 35, random_state=seed)

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    train_dataset = AprTokenDataset(train_docs, tokenizer, max_length=max_length)
    valid_dataset = AprTokenDataset(valid_docs, tokenizer, max_length=max_length)
    test_dataset = AprTokenDataset(test_docs, tokenizer, max_length=max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

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
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_INDEX)

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
    test_metrics = evaluate(model, test_loader, device)
    save_outputs(
        model=model,
        tokenizer=tokenizer,
        history=history,
        metrics=test_metrics,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
        models_dir=models_dir,
    )

    metadata = {
        "run_id": run_id,
        "model_family": "hf_token_classification",
        "model_id": model_id,
        "dataset_path": str(dataset_path),
        "output_dir": str(run_dir),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "seed": seed,
        "device": str(device),
        "hyperparameters": {
            "max_documents": max_documents,
            "max_length": max_length,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "class_weights": {label: round(float(class_weights[i].cpu()), 6) for i, label in enumerate(LABELS)},
        },
        "load_stats": load_stats,
        "split": {
            "train_documents": len(train_docs),
            "validation_documents": len(valid_docs),
            "test_documents": len(test_docs),
            "train_sequences": len(train_dataset),
            "validation_sequences": len(valid_dataset),
            "test_sequences": len(test_dataset),
        },
        "history": history,
        "metrics": {"hf_token_classifier": test_metrics},
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


def load_documents(dataset_path: Path, max_documents: int, seed: int) -> tuple[list[dict[str, list[str]]], dict[str, Any]]:
    rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines()]
    random.Random(seed).shuffle(rows)
    selected = rows[:max_documents]
    label_counts = Counter()
    token_count = 0
    documents = []
    for row in selected:
        tokens = []
        labels = []
        for token, label in zip(row["tokens"], row["labels"]):
            if label in LABEL_TO_ID:
                tokens.append(token)
                labels.append(label)
                label_counts[label] += 1
        if tokens:
            token_count += len(tokens)
            documents.append({"tokens": tokens, "labels": labels})
    return documents, {
        "documents_loaded": len(documents),
        "tokens_loaded": token_count,
        "labels": dict(label_counts),
    }


def compute_class_weights(documents: list[dict[str, list[str]]]) -> torch.Tensor:
    counts = Counter(label for document in documents for label in document["labels"])
    total = sum(counts.values())
    weights = [total / (len(LABELS) * max(1, counts[label])) for label in LABELS]
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(
    model,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    batches = 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        labels = batch.pop("labels")
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = loss_fn(outputs.logits.view(-1, len(LABELS)), labels.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu())
        batches += 1
    return total_loss / max(1, batches)


def evaluate(model, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            labels = batch.pop("labels")
            logits = model(**batch).logits
            predictions = logits.argmax(dim=-1)
            mask = labels != IGNORE_INDEX
            gold_ids = labels[mask].detach().cpu().tolist()
            pred_ids = predictions[mask].detach().cpu().tolist()
            y_true.extend(ID_TO_LABEL[index] for index in gold_ids)
            y_pred.extend(ID_TO_LABEL[index] for index in pred_ids)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABELS,
        zero_division=0,
    )
    macro = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "macro_precision": round(float(macro[0]), 6),
        "macro_recall": round(float(macro[1]), 6),
        "macro_f1": round(float(macro[2]), 6),
        "weighted_precision": round(float(weighted[0]), 6),
        "weighted_recall": round(float(weighted[1]), 6),
        "weighted_f1": round(float(weighted[2]), 6),
        "per_class": {
            label: {
                "precision": round(float(precision[i]), 6),
                "recall": round(float(recall[i]), 6),
                "f1": round(float(f1[i]), 6),
                "support": int(support[i]),
            }
            for i, label in enumerate(LABELS)
        },
        "y_true": y_true,
        "y_pred": y_pred,
    }


def save_outputs(
    model,
    tokenizer,
    history: list[dict[str, Any]],
    metrics: dict[str, Any],
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

    report = classification_report(metrics["y_true"], metrics["y_pred"], labels=LABELS, zero_division=0, output_dict=True)
    (tables_dir / "hf_token_classifier_classification_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    cm = confusion_matrix(metrics["y_true"], metrics["y_pred"], labels=LABELS)
    np.savetxt(tables_dir / "hf_token_classifier_confusion_matrix.csv", cm, delimiter=",", fmt="%d")
    plot_confusion_matrix(cm, figures_dir / "hf_token_classifier_confusion_matrix.png")
    plot_history(history, figures_dir / "hf_token_classifier_training_history.png")

    slim_metrics = {key: value for key, value in metrics.items() if key not in {"y_true", "y_pred"}}
    (tables_dir / "summary_metrics.csv").write_text(
        "model,accuracy,macro_precision,macro_recall,macro_f1,weighted_f1,comma_f1\n"
        f"hf_token_classifier,{slim_metrics['accuracy']},{slim_metrics['macro_precision']},"
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
    ax.set_title("HF token classifier confusion matrix")
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
    ax.set_title("HF token classifier validation metrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_report(metadata: dict[str, Any], output_path: Path) -> None:
    metrics = metadata["metrics"]["hf_token_classifier"]
    lines = [
        f"# HF Token APR Report: {metadata['run_id']}",
        "",
        f"- Model: `{metadata['model_id']}`",
        f"- Dataset: `{metadata['dataset_path']}`",
        f"- Device: `{metadata['device']}`",
        f"- Total runtime seconds: {metadata['runtime_seconds_total']}",
        f"- Documents loaded: {metadata['load_stats']['documents_loaded']}",
        f"- Tokens loaded: {metadata['load_stats']['tokens_loaded']}",
        "",
        "## Test Metrics",
        "",
        "| Model | Accuracy | S F1 | C F1 | P F1 | Macro-F1 |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| hf_token_classifier | {metrics['accuracy']} | {metrics['per_class']['S']['f1']} | "
            f"{metrics['per_class']['C']['f1']} | {metrics['per_class']['P']['f1']} | {metrics['macro_f1']} |"
        ),
        "",
        "## Epoch History",
        "",
        "| Epoch | Train loss | Valid macro-F1 | Valid comma F1 | Seconds |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in metadata["history"]:
        lines.append(
            f"| {row['epoch']} | {row['train_loss']} | {row['valid_macro_f1']} | "
            f"{row['valid_comma_f1']} | {row['seconds']} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

