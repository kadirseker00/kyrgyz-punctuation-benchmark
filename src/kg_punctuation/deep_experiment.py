from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
import platform
import random
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


LABELS = ["S", "C", "P"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
PAD_LABEL_ID = -100


class SequenceDataset(Dataset):
    def __init__(self, documents: list[dict[str, list[int]]], sequence_length: int):
        self.examples = []
        for document in documents:
            token_ids = document["token_ids"]
            label_ids = document["label_ids"]
            for start in range(0, len(token_ids), sequence_length):
                chunk_tokens = token_ids[start : start + sequence_length]
                chunk_labels = label_ids[start : start + sequence_length]
                if not chunk_tokens:
                    continue
                mask = [1] * len(chunk_tokens)
                pad_size = sequence_length - len(chunk_tokens)
                if pad_size:
                    chunk_tokens = chunk_tokens + [0] * pad_size
                    chunk_labels = chunk_labels + [PAD_LABEL_ID] * pad_size
                    mask = mask + [0] * pad_size
                self.examples.append(
                    {
                        "input_ids": torch.tensor(chunk_tokens, dtype=torch.long),
                        "labels": torch.tensor(chunk_labels, dtype=torch.long),
                        "mask": torch.tensor(mask, dtype=torch.bool),
                    }
                )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.examples[index]


class CnnTagger(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        sequence_length: int,
        embedding_dim: int,
        hidden_dim: int,
        num_labels: int,
        num_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(sequence_length, embedding_dim)
        padding = kernel_size // 2
        layers = []
        in_channels = embedding_dim
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Conv1d(in_channels, hidden_dim, kernel_size=kernel_size, padding=padding),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_channels = hidden_dim
        self.convs = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        embedded = self.embedding(input_ids) + self.position_embedding(positions)
        features = self.convs(embedded.transpose(1, 2)).transpose(1, 2)
        return self.classifier(self.dropout(features))


def run_deep_experiment(
    dataset_path: Path,
    output_dir: Path,
    max_documents: int = 3000,
    sequence_length: int = 128,
    vocab_size: int = 30000,
    embedding_dim: int = 128,
    hidden_dim: int = 128,
    num_layers: int = 2,
    kernel_size: int = 3,
    dropout: float = 0.25,
    batch_size: int = 64,
    epochs: int = 6,
    learning_rate: float = 0.001,
    seed: int = 42,
) -> dict[str, Any]:
    set_seed(seed)
    run_id = f"{dataset_path.stem}_cnn_{time.strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    run_dir = output_dir / run_id
    figures_dir = run_dir / "figures"
    tables_dir = run_dir / "tables"
    models_dir = run_dir / "models"
    for path in [figures_dir, tables_dir, models_dir]:
        path.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    raw_documents, load_stats = load_documents(dataset_path, max_documents=max_documents, seed=seed)
    train_docs, temp_docs = train_test_split(raw_documents, test_size=0.35, random_state=seed)
    valid_docs, test_docs = train_test_split(temp_docs, test_size=25 / 35, random_state=seed)

    vocabulary = build_vocabulary(train_docs, vocab_size=vocab_size)
    indexed_train = index_documents(train_docs, vocabulary)
    indexed_valid = index_documents(valid_docs, vocabulary)
    indexed_test = index_documents(test_docs, vocabulary)

    train_dataset = SequenceDataset(indexed_train, sequence_length=sequence_length)
    valid_dataset = SequenceDataset(indexed_valid, sequence_length=sequence_length)
    test_dataset = SequenceDataset(indexed_test, sequence_length=sequence_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = CnnTagger(
        vocab_size=len(vocabulary),
        sequence_length=sequence_length,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_labels=len(LABELS),
        num_layers=num_layers,
        kernel_size=kernel_size,
        dropout=dropout,
    ).to(device)
    class_weights = compute_class_weights(indexed_train).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=PAD_LABEL_ID)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    history = []
    best_macro_f1 = -1.0
    best_path = models_dir / "cnn_tagger_best.pt"
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        valid_metrics = evaluate_sequence_model(model, valid_loader, device)
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
    test_metrics = evaluate_sequence_model(model, test_loader, device)
    save_deep_outputs(
        model=model,
        vocabulary=vocabulary,
        history=history,
        metrics=test_metrics,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
        models_dir=models_dir,
    )

    metadata = {
        "run_id": run_id,
        "model_family": "cnn_sequence_tagger",
        "dataset_path": str(dataset_path),
        "output_dir": str(run_dir),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "seed": seed,
        "device": str(device),
        "hyperparameters": {
            "max_documents": max_documents,
            "sequence_length": sequence_length,
            "vocab_size_requested": vocab_size,
            "vocab_size_actual": len(vocabulary),
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "kernel_size": kernel_size,
            "dropout": dropout,
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
        "metrics": {"cnn_sequence_tagger": test_metrics},
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
        tokens = row["tokens"]
        labels = row["labels"]
        filtered_tokens = []
        filtered_labels = []
        for token, label in zip(tokens, labels):
            if label not in LABEL_TO_ID:
                continue
            filtered_tokens.append(token)
            filtered_labels.append(label)
            label_counts[label] += 1
        if filtered_tokens:
            token_count += len(filtered_tokens)
            documents.append({"tokens": filtered_tokens, "labels": filtered_labels})
    return documents, {
        "documents_loaded": len(documents),
        "tokens_loaded": token_count,
        "labels": dict(label_counts),
    }


def build_vocabulary(documents: list[dict[str, list[str]]], vocab_size: int) -> dict[str, int]:
    counter = Counter(token for document in documents for token in document["tokens"])
    vocabulary = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, _ in counter.most_common(max(0, vocab_size - len(vocabulary))):
        vocabulary[token] = len(vocabulary)
    return vocabulary


def index_documents(documents: list[dict[str, list[str]]], vocabulary: dict[str, int]) -> list[dict[str, list[int]]]:
    return [
        {
            "token_ids": [vocabulary.get(token, 1) for token in document["tokens"]],
            "label_ids": [LABEL_TO_ID[label] for label in document["labels"]],
        }
        for document in documents
    ]


def compute_class_weights(documents: list[dict[str, list[int]]]) -> torch.Tensor:
    counts = Counter(label for document in documents for label in document["label_ids"])
    total = sum(counts.values())
    weights = [total / (len(LABELS) * max(1, counts[index])) for index in range(len(LABELS))]
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    batches = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits.view(-1, len(LABELS)), labels.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu())
        batches += 1
    return total_loss / max(1, batches)


def evaluate_sequence_model(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    predictions = []
    gold = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].cpu().numpy()
            logits = model(input_ids).cpu().numpy()
            pred = logits.argmax(axis=-1)
            mask = labels != PAD_LABEL_ID
            predictions.extend(pred[mask].tolist())
            gold.extend(labels[mask].tolist())

    y_true = [LABELS[index] for index in gold]
    y_pred = [LABELS[index] for index in predictions]
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


def save_deep_outputs(
    model: nn.Module,
    vocabulary: dict[str, int],
    history: list[dict[str, Any]],
    metrics: dict[str, Any],
    figures_dir: Path,
    tables_dir: Path,
    models_dir: Path,
) -> None:
    torch.save(model.state_dict(), models_dir / "cnn_tagger_final.pt")
    (models_dir / "vocabulary.json").write_text(json.dumps(vocabulary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (tables_dir / "epoch_history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    report = classification_report(metrics["y_true"], metrics["y_pred"], labels=LABELS, zero_division=0, output_dict=True)
    (tables_dir / "cnn_sequence_tagger_classification_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    cm = confusion_matrix(metrics["y_true"], metrics["y_pred"], labels=LABELS)
    np.savetxt(tables_dir / "cnn_sequence_tagger_confusion_matrix.csv", cm, delimiter=",", fmt="%d")
    plot_confusion_matrix(cm, figures_dir / "cnn_sequence_tagger_confusion_matrix.png")
    plot_history(history, figures_dir / "cnn_sequence_tagger_training_history.png")

    slim_metrics = {key: value for key, value in metrics.items() if key not in {"y_true", "y_pred"}}
    (tables_dir / "summary_metrics.csv").write_text(
        "model,accuracy,macro_precision,macro_recall,macro_f1,weighted_f1,comma_f1\n"
        f"cnn_sequence_tagger,{slim_metrics['accuracy']},{slim_metrics['macro_precision']},"
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
    ax.set_title("CNN sequence tagger confusion matrix")
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
    ax.set_title("CNN sequence tagger validation metrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_report(metadata: dict[str, Any], output_path: Path) -> None:
    metrics = metadata["metrics"]["cnn_sequence_tagger"]
    lines = [
        f"# Deep APR Report: {metadata['run_id']}",
        "",
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
            f"| cnn_sequence_tagger | {metrics['accuracy']} | {metrics['per_class']['S']['f1']} | "
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
