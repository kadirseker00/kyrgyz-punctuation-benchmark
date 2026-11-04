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
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer


LABELS = ["S", "C", "P"]
MODEL_TO_APR = {
    "O": "S",
    "COMMA": "C",
    "PERIOD": "P",
    "QUESTION": "P",
}


class EvalDataset(Dataset):
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
                    labels.append(-100)
                elif word_id != previous_word_id:
                    labels.append(LABELS.index(document["labels"][word_id]))
                else:
                    labels.append(-100)
                previous_word_id = word_id
            encoding["labels"] = labels
            self.features.append({key: torch.tensor(value, dtype=torch.long) for key, value in encoding.items()})

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.features[index]


def evaluate_hf_punctuation_model(
    dataset_path: Path,
    output_dir: Path,
    model_id: str,
    max_documents: int = 1000,
    max_length: int = 256,
    batch_size: int = 16,
    seed: int = 42,
) -> dict[str, Any]:
    run_id = f"{dataset_path.stem}_{safe_name(model_id)}_eval_{time.strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    run_dir = output_dir / run_id
    figures_dir = run_dir / "figures"
    tables_dir = run_dir / "tables"
    models_dir = run_dir / "models"
    for path in [figures_dir, tables_dir, models_dir]:
        path.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    documents, load_stats = load_documents(dataset_path, max_documents=max_documents, seed=seed)
    _, temp_docs = train_test_split(documents, test_size=0.35, random_state=seed)
    _, test_docs = train_test_split(temp_docs, test_size=25 / 35, random_state=seed)

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_id)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()

    dataset = EvalDataset(test_docs, tokenizer, max_length=max_length)
    loader = DataLoader(dataset, batch_size=batch_size)
    metrics = run_eval(model, loader, device)
    save_outputs(metrics, figures_dir, tables_dir)
    model.config.save_pretrained(models_dir / "hf_model_config")
    tokenizer.save_pretrained(models_dir / "hf_model_config")

    metadata = {
        "run_id": run_id,
        "model_family": "external_hf_punctuation_model",
        "model_id": model_id,
        "dataset_path": str(dataset_path),
        "output_dir": str(run_dir),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "seed": seed,
        "device": str(device),
        "label_mapping": MODEL_TO_APR,
        "hyperparameters": {
            "max_documents": max_documents,
            "max_length": max_length,
            "batch_size": batch_size,
        },
        "load_stats": load_stats,
        "split": {
            "test_documents": len(test_docs),
            "test_sequences": len(dataset),
        },
        "metrics": {"external_hf_punctuation": metrics},
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


def safe_name(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model_id.split("/")[-1])


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
            if label in LABELS:
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


def run_eval(model, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    id2label = model.config.id2label
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            labels = batch.pop("labels")
            logits = model(**batch).logits
            predictions = logits.argmax(dim=-1)
            mask = labels != -100
            gold_ids = labels[mask].detach().cpu().tolist()
            pred_ids = predictions[mask].detach().cpu().tolist()
            for gold_id, pred_id in zip(gold_ids, pred_ids):
                model_label = id2label[int(pred_id)]
                y_true.append(LABELS[int(gold_id)])
                y_pred.append(MODEL_TO_APR.get(model_label, "S"))

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


def save_outputs(metrics: dict[str, Any], figures_dir: Path, tables_dir: Path) -> None:
    report = classification_report(metrics["y_true"], metrics["y_pred"], labels=LABELS, zero_division=0, output_dict=True)
    (tables_dir / "external_hf_punctuation_classification_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    cm = confusion_matrix(metrics["y_true"], metrics["y_pred"], labels=LABELS)
    np.savetxt(tables_dir / "external_hf_punctuation_confusion_matrix.csv", cm, delimiter=",", fmt="%d")
    plot_confusion_matrix(cm, figures_dir / "external_hf_punctuation_confusion_matrix.png")
    slim = {key: value for key, value in metrics.items() if key not in {"y_true", "y_pred"}}
    (tables_dir / "summary_metrics.csv").write_text(
        "model,accuracy,macro_precision,macro_recall,macro_f1,weighted_f1,comma_f1\n"
        f"external_hf_punctuation,{slim['accuracy']},{slim['macro_precision']},"
        f"{slim['macro_recall']},{slim['macro_f1']},{slim['weighted_f1']},"
        f"{slim['per_class']['C']['f1']}\n",
        encoding="utf-8",
    )
    metrics.clear()
    metrics.update(slim)


def plot_confusion_matrix(cm: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS).plot(
        ax=ax,
        cmap="Blues",
        colorbar=False,
        values_format="d",
    )
    ax.set_title("External HF punctuation model confusion matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_report(metadata: dict[str, Any], output_path: Path) -> None:
    metrics = metadata["metrics"]["external_hf_punctuation"]
    lines = [
        f"# External HF Punctuation Evaluation: {metadata['run_id']}",
        "",
        f"- Model: `{metadata['model_id']}`",
        f"- Dataset: `{metadata['dataset_path']}`",
        f"- Device: `{metadata['device']}`",
        f"- Runtime seconds: {metadata['runtime_seconds_total']}",
        f"- Documents loaded before split: {metadata['load_stats']['documents_loaded']}",
        f"- Test documents: {metadata['split']['test_documents']}",
        "",
        "## Label Mapping",
        "",
        "- `O -> S`",
        "- `COMMA -> C`",
        "- `PERIOD -> P`",
        "- `QUESTION -> P`",
        "",
        "## Test Metrics",
        "",
        "| Model | Accuracy | S F1 | C F1 | P F1 | Macro-F1 |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| external_hf_punctuation | {metrics['accuracy']} | {metrics['per_class']['S']['f1']} | "
            f"{metrics['per_class']['C']['f1']} | {metrics['per_class']['P']['f1']} | {metrics['macro_f1']} |"
        ),
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

