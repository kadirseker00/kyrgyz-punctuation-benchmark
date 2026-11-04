from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import platform
import random
import time
from typing import Any
from uuid import uuid4

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer
from torch import nn
from torch.utils.data import DataLoader

from kg_punctuation.data.labeling import resolve_label_set
from kg_punctuation.hf_generic_experiment import (
    TokenClassificationDataset,
    compute_class_weights,
    evaluate,
    load_documents,
    select_device,
    set_seed,
    train_one_epoch,
    run_hf_experiment,
    classification_metrics,
)


def run_learning_curve(
    dataset_path: Path,
    output_dir: Path,
    model_id: str = "xlm-roberta-base",
    budgets: tuple[int, ...] = (250, 500, 1000, 2000, 3250),
    max_documents: int = 5000,
    max_length: int = 128,
    batch_size: int = 8,
    epochs: int = 3,
    learning_rate: float = 3e-5,
    seed: int = 42,
    split_seed: int = 42,
    label_set: str = "core",
    manifest_path: Path | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    if manifest_path is not None:
        return _run_canonical_curve(
            manifest_path, source, dataset_path, output_dir, model_id, budgets,
            max_length, batch_size, epochs, learning_rate, seed, label_set,
        )
    if source is not None:
        raise ValueError("source requires manifest_path.")
    labels = list(resolve_label_set(label_set))
    root = output_dir / f"learning_curve_{dataset_path.stem}_{'-'.join(labels).lower()}"
    root.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    documents, load_stats = load_documents(
        dataset_path,
        max_documents=max_documents,
        seed=split_seed,
        labels=labels,
    )
    train_docs, temp_docs = train_test_split(
        documents,
        test_size=0.35,
        random_state=split_seed,
    )
    valid_docs, test_docs = train_test_split(
        temp_docs,
        test_size=25 / 35,
        random_state=split_seed,
    )
    shuffled_train = list(train_docs)
    random.Random(split_seed).shuffle(shuffled_train)

    results = []
    for budget in budgets:
        if budget > len(shuffled_train):
            continue
        run_start = time.perf_counter()
        result = _run_budget(
            train_docs=shuffled_train[:budget],
            valid_docs=valid_docs,
            test_docs=test_docs,
            output_dir=root,
            model_id=model_id,
            labels=labels,
            max_length=max_length,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            seed=seed,
            split_seed=split_seed,
            budget=budget,
        )
        result["seconds"] = round(time.perf_counter() - run_start, 3)
        results.append(result)

    summary = {
        "dataset_path": str(dataset_path),
        "model_id": model_id,
        "label_set": labels,
        "budgets": list(budgets),
        "max_documents": max_documents,
        "seed": seed,
        "split_seed": split_seed,
        "load_stats": load_stats,
        "fixed_split": {
            "train_documents_available": len(train_docs),
            "validation_documents": len(valid_docs),
            "test_documents": len(test_docs),
            "test_label_counts": count_labels(test_docs, labels),
        },
        "results": results,
        "runtime_seconds_total": round(time.perf_counter() - start, 3),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "torch": torch.__version__,
        },
    }
    (root / "metadata.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_csv(results, root / "learning_curve.csv")
    _write_plot(results, root / "learning_curve")
    _write_markdown(summary, root / "report.md")
    return {"output_dir": str(root), "results": results}


def _run_canonical_curve(manifest_path, source, dataset_path, output_dir, model_id,
                         budgets, max_length, batch_size, epochs, learning_rate, seed, label_set):
    from kg_punctuation.benchmark_v2 import Benchmark, view_labels

    if source is None:
        raise ValueError("Canonical learning curves require a source.")
    if not budgets or any(b < 1 for b in budgets) or list(budgets) != sorted(set(budgets)):
        raise ValueError("Budgets must be strictly increasing positive document counts.")
    benchmark = Benchmark(manifest_path)
    available = len(benchmark.split_ids(source, "train"))
    feasible = [budget for budget in budgets if budget <= available]
    if not feasible:
        raise ValueError("No requested budget fits the canonical train pool.")
    root = output_dir / f"learning_curve_v2_{source}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    root.mkdir(parents=True, exist_ok=False)
    results = []
    start = time.perf_counter()
    summary = {
        "protocol": "canonical_v2", "dataset_path": str(dataset_path),
        "model_id": model_id, "label_set": view_labels(label_set),
        "seed": seed, "split_seed": benchmark.manifest["seed"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
        "budgets": list(budgets), "effective_budgets": feasible,
        "skipped_budgets": [budget for budget in budgets if budget > available],
        "train_documents_available": available,
        "hyperparameters": {"max_length": max_length, "batch_size": batch_size,
                            "epochs": epochs, "learning_rate": learning_rate},
        "results": results,
    }
    for budget in feasible:
        run = run_hf_experiment(
            dataset_path, root, model_id=model_id, max_length=max_length,
            batch_size=batch_size, epochs=epochs, learning_rate=learning_rate,
            seed=seed, label_set=label_set, run_name=f"v2_{source}_budget_{budget}",
            manifest_path=manifest_path, source=source, train_documents=budget,
        )
        run_dir = Path(run["output_dir"])
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        if metadata["benchmark"]["manifest_sha256"] != summary["manifest_sha256"]:
            raise ValueError("Manifest changed during the learning curve.")
        results.append({
            "budget": budget, "train_documents": metadata["split"]["train_documents"],
            "train_tokens": sum(metadata["split"]["train_label_counts"].values()),
            "seconds": metadata["runtime_seconds_total"], "run_dir": str(run_dir),
            **metadata["metrics"]["hf_token_classifier"],
        })
        summary["runtime_seconds_total"] = round(time.perf_counter() - start, 3)
        (root / "metadata.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        _write_csv(results, root / "learning_curve.csv")
    summary["sentence_overlap_audit"] = _audit_canonical_overlap(benchmark, source, label_set, results, root)
    (root / "metadata.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_plot(results, root / "learning_curve")
    _write_markdown(summary, root / "report.md")
    return {"output_dir": str(root), "results": results}


def _audit_canonical_overlap(benchmark, source, view, results, root):
    from kg_punctuation.benchmark_v2 import sentence_spans, view_labels

    test = benchmark.documents(source, "test", "extended")
    test_sentences = set()
    for doc in test:
        test_sentences.update(tuple(doc["tokens"][a:b]) for a, b in sentence_spans(doc["labels"]))
    earliest = {}
    for rank, doc in enumerate(benchmark.documents(source, "train", "extended"), start=1):
        for a, b in sentence_spans(doc["labels"]):
            sentence = tuple(doc["tokens"][a:b])
            if sentence in test_sentences:
                earliest.setdefault(sentence, rank)
    positions, seen_at = [], []
    for doc in test:
        ranks = [0] * len(doc["tokens"])
        for a, b in sentence_spans(doc["labels"]):
            rank = earliest.get(tuple(doc["tokens"][a:b]), 0)
            ranks[a:b] = [rank] * (b - a)
        positions.extend((source, doc["document_id"], str(i)) for i in range(len(ranks)))
        seen_at.extend(ranks)
    with (root / "sentence_overlap_mask.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "document_id", "word_index", "first_train_document_rank"])
        writer.writerows((*position, rank) for position, rank in zip(positions, seen_at))
    ranks = np.asarray(seen_at)
    fixed = ranks == 0
    rows = []
    for result in results:
        with (Path(result["run_dir"]) / "tables/predictions.csv").open(encoding="utf-8") as handle:
            predictions = list(csv.DictReader(handle))
        if [(r["source"], r["document_id"], r["word_index"]) for r in predictions] != positions:
            raise ValueError("Learning-curve prediction identities differ from the overlap mask.")
        gold = np.asarray([row["gold"] for row in predictions])
        predicted = np.asarray([row["prediction"] for row in predictions])
        seen = (ranks > 0) & (ranks <= result["budget"])
        subsets = {}
        for name, mask in (("seen_at_budget", seen), ("outside_seen_at_budget", ~seen), ("fixed_outside_full_pool_overlap", fixed)):
            if mask.any():
                values = classification_metrics(gold[mask].tolist(), predicted[mask].tolist(), view_labels(view))
                values.pop("y_true")
                values.pop("y_pred")
            else:
                values = None
            subsets[name] = {"tokens": int(mask.sum()), "metrics": values}
        rows.append({"budget": result["budget"], "subsets": subsets})
    return {
        "rule": "Exact token equality of completed sentences with at least 5 words; punctuation ignored in equality.",
        "limitation": "Outside-overlap does not rule out shorter phrases, paraphrases or unclosed sentence matches. Budget-specific subsets change; only the full-pool exclusion subset stays fixed.",
        "mask_path": str(root / "sentence_overlap_mask.csv"),
        "fixed_subset_tokens": int(fixed.sum()), "results": rows,
    }


def _run_budget(
    train_docs: list[dict[str, list[str]]],
    valid_docs: list[dict[str, list[str]]],
    test_docs: list[dict[str, list[str]]],
    output_dir: Path,
    model_id: str,
    labels: list[str],
    max_length: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    split_seed: int,
    budget: int,
) -> dict[str, Any]:
    set_seed(seed)
    run_dir = output_dir / f"budget_{budget}"
    run_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    label_to_id = {label: index for index, label in enumerate(labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    train_dataset = TokenClassificationDataset(train_docs, tokenizer, max_length, label_to_id)
    valid_dataset = TokenClassificationDataset(valid_docs, tokenizer, max_length, label_to_id)
    test_dataset = TokenClassificationDataset(test_docs, tokenizer, max_length, label_to_id)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    config = AutoConfig.from_pretrained(
        model_id,
        num_labels=len(labels),
        id2label=id_to_label,
        label2id=label_to_id,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_id,
        config=config,
        ignore_mismatched_sizes=True,
    )
    device = select_device()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    class_weights = compute_class_weights(train_docs, labels).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)

    history = []
    best_macro_f1 = -1.0
    best_state = None
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            len(labels),
        )
        valid_metrics = evaluate(model, valid_loader, device, labels, id_to_label)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "valid_macro_f1": valid_metrics["macro_f1"],
            }
        )
        if valid_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = valid_metrics["macro_f1"]
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)
    metrics = evaluate(model, test_loader, device, labels, id_to_label)
    metrics_without_predictions = {
        key: value for key, value in metrics.items() if key not in {"y_true", "y_pred"}
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "budget": budget,
                "seed": seed,
                "split_seed": split_seed,
                "label_set": labels,
                "device": str(device),
                "train_documents": len(train_docs),
                "validation_documents": len(valid_docs),
                "test_documents": len(test_docs),
                "train_tokens": sum(len(doc["tokens"]) for doc in train_docs),
                "history": history,
                "metrics": metrics_without_predictions,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    with (run_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "gold", "prediction"])
        writer.writerows(
            (index, gold, prediction)
            for index, (gold, prediction) in enumerate(zip(metrics["y_true"], metrics["y_pred"]))
        )
    return {
        "budget": budget,
        "train_documents": len(train_docs),
        "train_tokens": sum(len(doc["tokens"]) for doc in train_docs),
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "per_class": metrics["per_class"],
        "run_dir": str(run_dir),
    }


def count_labels(documents: list[dict[str, list[str]]], labels: list[str]) -> dict[str, int]:
    counts = Counter(label for document in documents for label in document["labels"])
    return {label: counts[label] for label in labels}


def _write_csv(results: list[dict[str, Any]], output_path: Path) -> None:
    fields = ["budget", "train_documents", "train_tokens", "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "seconds", "run_dir"]
    for label in results[0]["per_class"]:
        fields.extend([f"{label}_precision", f"{label}_recall", f"{label}_f1", f"{label}_support"])
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            row = {key: result.get(key) for key in fields}
            for label, values in result["per_class"].items():
                for metric_name in ("precision", "recall", "f1", "support"):
                    row[f"{label}_{metric_name}"] = values[metric_name]
            writer.writerow(row)


def _write_markdown(summary: dict[str, Any], output_path: Path) -> None:
    lines = [
        f"# Learning curve: {Path(summary['dataset_path']).stem}",
        "",
        f"- Model: `{summary['model_id']}`",
        f"- Labels: `{', '.join(summary['label_set'])}`",
        f"- Fixed split seed: `{summary['split_seed']}`",
        f"- Training seed: `{summary['seed']}`",
        "",
        "| Training documents | Training tokens | Accuracy | Macro precision | Macro recall | Macro-F1 | Weighted-F1 | Seconds |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in summary["results"]:
        lines.append(
            f"| {result['train_documents']} | {result['train_tokens']} | {result['accuracy']} | "
            f"{result['macro_precision']} | {result['macro_recall']} | {result['macro_f1']} | "
            f"{result['weighted_f1']} | {result['seconds']} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plot(results: list[dict[str, Any]], output_stem: Path) -> None:
    """Write the paper-ready learning-curve figure beside the run summary."""
    if not results:
        return
    import matplotlib.pyplot as plt

    documents = [result["train_documents"] for result in results]
    macro_f1 = [100 * result["macro_f1"] for result in results]
    weighted_f1 = [100 * result["weighted_f1"] for result in results]

    figure, axis = plt.subplots(figsize=(6.2, 3.8))
    axis.plot(
        documents,
        macro_f1,
        marker="o",
        linewidth=2.2,
        color="#4a3aa7",
        label="Macro-F1",
    )
    axis.plot(
        documents,
        weighted_f1,
        marker="s",
        linewidth=1.8,
        color="#eb6834",
        label="Weighted-F1",
    )
    axis.set_xlabel("Training documents")
    axis.set_ylabel("F1 score (%)")
    axis.set_xticks(documents)
    axis.tick_params(axis="x", labelrotation=20)
    axis.set_ylim(0, 100)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, loc="lower right")
    figure.tight_layout()
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)
