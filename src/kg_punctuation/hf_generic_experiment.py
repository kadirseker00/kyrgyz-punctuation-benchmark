from __future__ import annotations

from collections import Counter
import csv
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import random
import re
import time
from typing import Any, Iterable
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

from kg_punctuation.data.labeling import resolve_label_set


IGNORE_INDEX = -100


class TokenClassificationDataset(Dataset):
    def __init__(
        self,
        documents: list[dict[str, list[str]]],
        tokenizer,
        max_length: int,
        label_to_id: dict[str, int],
        full_document_windows: bool = False,
    ):
        self.features = []
        for document in documents:
            if full_document_windows:
                self._append_windows(document, tokenizer, max_length, label_to_id)
                continue
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
                    labels.append(label_to_id[document["labels"][word_id]])
                else:
                    labels.append(IGNORE_INDEX)
                previous_word_id = word_id
            encoding["labels"] = labels
            self.features.append(
                {key: torch.tensor(value, dtype=torch.long) for key, value in encoding.items()}
            )

    def _append_windows(self, document, tokenizer, max_length, label_to_id):
        from kg_punctuation.benchmark_v2 import encode_training_windows

        tokens = document["tokens"]
        if not tokens or len(tokens) != len(document["labels"]):
            raise ValueError("Training documents must contain aligned, nonempty tokens and labels.")
        features = encode_training_windows(tokenizer, [document], label_to_id, max_length)
        observed = Counter(
            label for feature in features for label in feature["labels"] if label != IGNORE_INDEX
        )
        expected = Counter(label_to_id[label] for label in document["labels"])
        if observed != expected:
            raise ValueError("Training windows must supervise every word exactly once.")
        for feature in features:
            labels = feature["labels"]
            if all(label == IGNORE_INDEX for label in labels):
                continue
            encoding = tokenizer.pad(
                {key: value for key, value in feature.items() if key != "labels"},
                padding="max_length", max_length=max_length,
            )
            padding = [IGNORE_INDEX] * (len(encoding["input_ids"]) - len(labels))
            encoding["labels"] = labels + padding if tokenizer.padding_side == "right" else padding + labels
            self.features.append({key: torch.tensor(value, dtype=torch.long) for key, value in encoding.items()})

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.features[index]


def run_hf_experiment(
    dataset_path: Path,
    output_dir: Path,
    model_id: str = "xlm-roberta-base",
    max_documents: int = 1000,
    max_length: int = 128,
    batch_size: int = 8,
    epochs: int = 3,
    learning_rate: float = 3e-5,
    seed: int = 42,
    label_set: str | Iterable[str] = "core",
    split_seed: int | None = None,
    run_name: str | None = None,
    manifest_path: Path | None = None,
    source: str | None = None,
    train_documents: int | None = None,
    loss_weighting: str = "inverse_frequency",
    revision: str | None = None,
) -> dict[str, Any]:
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("Epochs, batch size and learning rate must be positive.")
    if loss_weighting not in ("inverse_frequency", "none"):
        raise ValueError("loss_weighting must be inverse_frequency or none.")
    benchmark = None
    manifest_hash = None
    if manifest_path is not None:
        from kg_punctuation.benchmark_v2 import Benchmark, view_labels

        if label_set not in ("core", "extended") or source is None:
            raise ValueError("Canonical runs require source and a core/extended label view.")
        benchmark = Benchmark(manifest_path)
        manifest_hash = hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
        split_seed = benchmark.manifest["seed"]
        labels = view_labels(label_set)
        if train_documents is None:
            train_documents = benchmark.manifest["main_train_documents"]
        if train_documents < 1:
            raise ValueError("train_documents must be positive.")
    else:
        if source is not None or train_documents is not None:
            raise ValueError("source/train_documents require manifest_path.")
        labels = list(resolve_label_set(label_set))
    split_seed = seed if split_seed is None else split_seed
    set_seed(seed)

    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_id.split("/")[-1])
    label_tag = "-".join(labels).lower()
    prefix = run_name or f"{dataset_path.stem}_{safe_model}_{label_tag}"
    run_id = f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    run_dir = output_dir / run_id
    print(f"AUDIT START {run_id}", flush=True)
    figures_dir = run_dir / "figures"
    tables_dir = run_dir / "tables"
    models_dir = run_dir / "models"
    for path in [figures_dir, tables_dir, models_dir]:
        path.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    if benchmark is None:
        documents, load_stats = load_documents(
            dataset_path, max_documents=max_documents, seed=split_seed, labels=labels,
        )
        train_docs, temp_docs = train_test_split(documents, test_size=0.35, random_state=split_seed)
        valid_docs, test_docs = train_test_split(temp_docs, test_size=25 / 35, random_state=split_seed)
    else:
        train_docs = benchmark.documents(source, "train", label_set, limit=train_documents)
        valid_docs = benchmark.documents(source, "validation", label_set)
        test_docs = benchmark.documents(source, "test", label_set)
        dataset_path = Path(benchmark.manifest["sources"][source]["data_file"])
        load_stats = {"protocol": "canonical_v2", "train_budget_documents": len(train_docs)}

    revision_kwargs = {"revision": revision} if revision is not None else {}
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, **revision_kwargs)
    label_to_id = {label: index for index, label in enumerate(labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    train_dataset = TokenClassificationDataset(
        train_docs, tokenizer, max_length, label_to_id, full_document_windows=benchmark is not None,
    )
    valid_dataset = TokenClassificationDataset(valid_docs, tokenizer, max_length, label_to_id) if benchmark is None else None
    test_dataset = TokenClassificationDataset(test_docs, tokenizer, max_length, label_to_id) if benchmark is None else None
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print(f"AUDIT DATA {run_id} train_documents={len(train_docs)} windows={len(train_dataset)} steps={len(train_loader)}", flush=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size) if valid_dataset is not None else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size) if test_dataset is not None else None

    config = AutoConfig.from_pretrained(
        model_id,
        num_labels=len(labels),
        id2label=id_to_label,
        label2id=label_to_id,
        **revision_kwargs,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_id,
        config=config,
        ignore_mismatched_sizes=True,
        **revision_kwargs,
    )
    device = select_device()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    class_weights = (compute_class_weights(train_docs, labels) if loss_weighting == "inverse_frequency"
                     else torch.ones(len(labels), dtype=torch.float32)).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_INDEX)

    history = []
    best_macro_f1 = -1.0
    best_path = models_dir / "best_model.pt"
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            len(labels),
            progress_prefix=f"{run_id} epoch={epoch}",
        )
        valid_metrics = (
            evaluate(model, valid_loader, device, labels, id_to_label) if benchmark is None else
            evaluate_canonical(model, tokenizer, valid_docs, benchmark, source, "validation",
                               label_set, max_length, batch_size, device, id_to_label,
                               tables_dir / f"validation_epoch_{epoch}.csv")
        )
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "valid_accuracy": valid_metrics["accuracy"],
            "valid_macro_precision": valid_metrics["macro_precision"],
            "valid_macro_recall": valid_metrics["macro_recall"],
            "valid_macro_f1": valid_metrics["macro_f1"],
            "seconds": round(time.perf_counter() - epoch_start, 3),
        }
        for label in labels:
            row[f"valid_{label}_f1"] = valid_metrics["per_class"][label]["f1"]
        history.append(row)
        print(f"AUDIT EPOCH {run_id} {json.dumps(row)}", flush=True)
        if valid_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = valid_metrics["macro_f1"]
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_start = time.perf_counter()
    test_metrics = (
        evaluate(model, test_loader, device, labels, id_to_label) if benchmark is None else
        evaluate_canonical(model, tokenizer, test_docs, benchmark, source, "test",
                           label_set, max_length, batch_size, device, id_to_label,
                           tables_dir / "predictions.csv")
    )
    test_seconds = time.perf_counter() - test_start
    save_outputs(
        model=model,
        tokenizer=tokenizer,
        history=history,
        metrics=test_metrics,
        labels=labels,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
        models_dir=models_dir,
        prediction_documents=test_docs if benchmark is not None else None,
    )

    metadata = {
        "run_id": run_id,
        "model_family": "hf_token_classification_generic",
        "model_id": model_id,
        "label_set": labels,
        "dataset_path": str(dataset_path),
        "output_dir": str(run_dir),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "seed": seed,
        "split_seed": split_seed,
        "device": str(device),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": max(history, key=lambda row: row["valid_macro_f1"])["epoch"],
        "model_revision": getattr(model.config, "_commit_hash", None),
        "tokenizer_revision": tokenizer.init_kwargs.get("_commit_hash"),
        "hyperparameters": {
            "max_documents": max_documents,
            "max_length": max_length,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "optimizer": "AdamW", "weight_decay": optimizer.defaults["weight_decay"],
            "scheduler": None,
            "loss_weighting": loss_weighting,
            "class_weights": {
                label: round(float(class_weights[index].cpu()), 6)
                for index, label in enumerate(labels)
            },
        },
        "load_stats": load_stats,
        "split": {
            "train_documents": len(train_docs),
            "validation_documents": len(valid_docs),
            "test_documents": len(test_docs),
            "train_sequences": len(train_dataset),
            "validation_sequences": len(valid_dataset) if valid_dataset is not None else None,
            "test_sequences": len(test_dataset) if test_dataset is not None else None,
            "train_label_counts": count_labels(train_docs, labels),
            "validation_label_counts": count_labels(valid_docs, labels),
            "test_label_counts": count_labels(test_docs, labels),
        },
        "history": history,
        "metrics": {"hf_token_classifier": test_metrics},
        "runtime_seconds_total": round(time.perf_counter() - start, 3),
        "runtime_seconds_test": round(test_seconds, 3),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "torch": torch.__version__,
            "transformers": version("transformers"),
            "numpy": np.__version__,
            "scikit-learn": version("scikit-learn"),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
    }
    if benchmark is not None:
        if hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest() != manifest_hash:
            raise ValueError("Canonical manifest changed during training.")
        metadata["benchmark"] = {
            "version": "v2", "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_hash,
            "source": source, "label_view": label_set,
            "source_sha256": benchmark.manifest["sources"][source]["sha256"],
            "train_document_ids": [doc["document_id"] for doc in train_docs],
            "validation_document_ids": [doc["document_id"] for doc in valid_docs],
            "test_document_ids": [doc["document_id"] for doc in test_docs],
            "full_word_coverage": True, "subtoken": "first",
            "training_windows": "one loss label per word in its owner window",
            "inference_windows": "nearest window centre; earlier window on ties",
        }
        metadata["hyperparameters"].pop("max_documents")
        metadata["hyperparameters"]["train_documents"] = train_documents
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(metadata, run_dir / "report.md")
    print(f"AUDIT DONE {run_id} seconds={metadata['runtime_seconds_total']} macro_f1={test_metrics['macro_f1']}", flush=True)
    return {"run_id": run_id, "output_dir": str(run_dir)}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_documents(
    dataset_path: Path,
    max_documents: int,
    seed: int,
    labels: list[str],
) -> tuple[list[dict[str, list[str]]], dict[str, Any]]:
    rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines()]
    random.Random(seed).shuffle(rows)
    selected = rows[:max_documents]
    allowed = set(labels)
    label_counts = Counter()
    token_count = 0
    documents = []
    for row in selected:
        tokens = []
        row_labels = []
        for token, label in zip(row["tokens"], row["labels"]):
            if label in allowed:
                tokens.append(token)
                row_labels.append(label)
                label_counts[label] += 1
        if tokens:
            token_count += len(tokens)
            documents.append({"tokens": tokens, "labels": row_labels})
    return documents, {
        "documents_loaded": len(documents),
        "tokens_loaded": token_count,
        "labels": {label: label_counts[label] for label in labels},
    }


def count_labels(documents: list[dict[str, list[str]]], labels: list[str]) -> dict[str, int]:
    counts = Counter(label for document in documents for label in document["labels"])
    return {label: counts[label] for label in labels}


def compute_class_weights(
    documents: list[dict[str, list[str]]],
    labels: list[str],
) -> torch.Tensor:
    counts = Counter(label for document in documents for label in document["labels"])
    total = sum(counts.values())
    weights = [total / (len(labels) * max(1, counts[label])) for label in labels]
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(
    model,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    num_labels: int,
    progress_prefix: str | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    batches = 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        labels = batch.pop("labels")
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = loss_fn(outputs.logits.view(-1, num_labels), labels.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu())
        batches += 1
        if progress_prefix and batches % 250 == 0:
            print(f"AUDIT PROGRESS {progress_prefix} step={batches}/{len(loader)}", flush=True)
    return total_loss / max(1, batches)


def evaluate(
    model,
    loader: DataLoader,
    device: torch.device,
    labels: list[str],
    id_to_label: dict[int, str],
) -> dict[str, Any]:
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            batch_labels = batch.pop("labels")
            predictions = model(**batch).logits.argmax(dim=-1)
            mask = batch_labels != IGNORE_INDEX
            gold_ids = batch_labels[mask].detach().cpu().tolist()
            pred_ids = predictions[mask].detach().cpu().tolist()
            y_true.extend(id_to_label[index] for index in gold_ids)
            y_pred.extend(id_to_label[index] for index in pred_ids)

    return classification_metrics(y_true, y_pred, labels)


def classification_metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict[str, Any]:

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    macro = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    weighted = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="weighted",
        zero_division=0,
    )
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
                "precision": round(float(precision[index]), 6),
                "recall": round(float(recall[index]), 6),
                "f1": round(float(f1[index]), 6),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        },
        "y_true": y_true,
        "y_pred": y_pred,
    }


def evaluate_canonical(model, tokenizer, docs, benchmark, source, split, view,
                       max_length, batch_size, device, id_to_label, prediction_path):
    from kg_punctuation.benchmark_v2 import predict_words, score, view_labels, write_predictions

    predictions = predict_words(
        model, tokenizer, docs, id_to_label, subtoken="first", max_length=max_length,
        batch_size=batch_size, device=device,
    )
    write_predictions(prediction_path, docs, predictions)
    verified = score(prediction_path, benchmark, source, split=split, view=view)
    metrics = classification_metrics(
        [label for doc in docs for label in doc["labels"]],
        [label for row in predictions for label in row], view_labels(view),
    )
    if abs(verified["macro_f1"] - metrics["macro_f1"]) > 1e-6:
        raise ValueError("Canonical scorer and saved metrics disagree.")
    return metrics


def save_outputs(
    model,
    tokenizer,
    history: list[dict[str, Any]],
    metrics: dict[str, Any],
    labels: list[str],
    figures_dir: Path,
    tables_dir: Path,
    models_dir: Path,
    prediction_documents: list[dict] | None = None,
) -> None:
    model.save_pretrained(models_dir / "hf_model")
    tokenizer.save_pretrained(models_dir / "hf_model")
    with (tables_dir / "epoch_history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

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
    if prediction_documents is None:
        with (tables_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["index", "gold", "prediction"])
            writer.writerows(
                (index, gold, prediction)
                for index, (gold, prediction) in enumerate(
                    zip(metrics["y_true"], metrics["y_pred"])
                )
            )
    else:
        from kg_punctuation.benchmark_v2 import write_predictions

        offset, predictions = 0, []
        for doc in prediction_documents:
            end = offset + len(doc["tokens"])
            predictions.append(metrics["y_pred"][offset:end])
            offset = end
        if offset != len(metrics["y_pred"]):
            raise ValueError("Prediction count differs from canonical documents.")
        write_predictions(tables_dir / "predictions.csv", prediction_documents, predictions)

    cm = confusion_matrix(metrics["y_true"], metrics["y_pred"], labels=labels)
    np.savetxt(tables_dir / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")
    plot_confusion_matrix(cm, labels, figures_dir / "confusion_matrix.png")
    plot_history(history, labels, figures_dir / "training_history.png")

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
    row = {
        "model": "hf_token_classifier",
        **{key: metrics[key] for key in fieldnames if key in metrics},
    }
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
    ax.set_title("Token-classification confusion matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_history(
    history: list[dict[str, Any]],
    labels: list[str],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    epochs = [row["epoch"] for row in history]
    ax.plot(epochs, [row["valid_macro_f1"] for row in history], marker="o", label="Macro-F1")
    if "C" in labels:
        ax.plot(epochs, [row["valid_C_f1"] for row in history], marker="o", label="Comma F1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation F1")
    ax.set_title("Validation metrics during fine-tuning")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_report(metadata: dict[str, Any], output_path: Path) -> None:
    metrics = metadata["metrics"]["hf_token_classifier"]
    labels = metadata["label_set"]
    lines = [
        f"# HF token-classification APR report: {metadata['run_id']}",
        "",
        f"- Model: `{metadata['model_id']}`",
        f"- Dataset: `{metadata['dataset_path']}`",
        f"- Labels: `{', '.join(labels)}`",
        f"- Device: `{metadata['device']}`",
        f"- Seed: `{metadata['seed']}`; split seed: `{metadata['split_seed']}`",
        f"- Total runtime seconds: {metadata['runtime_seconds_total']}",
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
