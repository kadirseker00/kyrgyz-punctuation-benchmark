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

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize


LABELS = ["S", "C", "P"]


def run_experiment(
    dataset_path: Path,
    output_dir: Path,
    max_tokens: int = 200000,
    seed: int = 42,
) -> dict[str, Any]:
    run_id = f"{dataset_path.stem}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    run_dir = output_dir / run_id
    figures_dir = run_dir / "figures"
    tables_dir = run_dir / "tables"
    models_dir = run_dir / "models"
    for path in [figures_dir, tables_dir, models_dir]:
        path.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    X, y, docs, load_stats = load_token_examples(dataset_path, max_tokens=max_tokens, seed=seed)
    split_start = time.perf_counter()
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.35,
        random_state=seed,
        stratify=y,
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=25 / 35,
        random_state=seed,
        stratify=y_temp,
    )
    split_seconds = time.perf_counter() - split_start

    models = {
        "logreg_balanced": LogisticRegression(
            max_iter=800,
            class_weight="balanced",
            solver="saga",
            random_state=seed,
        ),
        "sgd_log_balanced": SGDClassifier(
            loss="log_loss",
            alpha=1e-5,
            class_weight="balanced",
            max_iter=1000,
            tol=1e-3,
            random_state=seed,
        ),
    }

    all_metrics: dict[str, Any] = {}
    for name, estimator in models.items():
        model_start = time.perf_counter()
        pipeline = Pipeline(
            [
                ("vectorizer", FeatureHasher(n_features=2**18, input_type="dict")),
                ("classifier", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - model_start

        eval_start = time.perf_counter()
        metrics = evaluate_model(
            pipeline=pipeline,
            model_name=name,
            X_test=X_test,
            y_test=y_test,
            figures_dir=figures_dir,
            tables_dir=tables_dir,
        )
        metrics["fit_seconds"] = round(fit_seconds, 3)
        metrics["eval_seconds"] = round(time.perf_counter() - eval_start, 3)
        all_metrics[name] = metrics
        joblib.dump(pipeline, models_dir / f"{name}.joblib")

    metadata = {
        "run_id": run_id,
        "dataset_path": str(dataset_path),
        "output_dir": str(run_dir),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "seed": seed,
        "max_tokens": max_tokens,
        "documents_loaded": docs,
        "load_stats": load_stats,
        "split": {
            "train_tokens": len(y_train),
            "validation_tokens": len(y_valid),
            "test_tokens": len(y_test),
            "train_label_counts": dict(Counter(y_train)),
            "validation_label_counts": dict(Counter(y_valid)),
            "test_label_counts": dict(Counter(y_test)),
            "split_seconds": round(split_seconds, 3),
        },
        "runtime_seconds_total": round(time.perf_counter() - start, 3),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "metrics": all_metrics,
        "notes": [
            "R2 is included only as a diagnostic on numeric class ids; classification metrics are primary.",
            "This is a CPU-friendly classical baseline run, not the final deep learning architecture.",
        ],
    }

    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_summary_csv(all_metrics, tables_dir / "summary_metrics.csv")
    write_markdown_report(metadata, run_dir / "report.md")
    return {"run_id": run_id, "output_dir": str(run_dir)}


def load_token_examples(dataset_path: Path, max_tokens: int, seed: int) -> tuple[list[dict[str, Any]], list[str], int, dict[str, Any]]:
    rng = random.Random(seed)
    examples: list[dict[str, Any]] = []
    labels: list[str] = []
    docs = 0
    skipped = 0
    label_counts = Counter()

    with dataset_path.open("r", encoding="utf-8") as handle:
        rows = list(handle)
    rng.shuffle(rows)

    for line in rows:
        row = json.loads(line)
        tokens = row["tokens"]
        row_labels = row["labels"]
        docs += 1
        for index, (token, label) in enumerate(zip(tokens, row_labels)):
            if label not in LABELS:
                skipped += 1
                continue
            examples.append(token_features(tokens, index))
            labels.append(label)
            label_counts[label] += 1
            if len(labels) >= max_tokens:
                return examples, labels, docs, {
                    "tokens_loaded": len(labels),
                    "labels": dict(label_counts),
                    "skipped_tokens": skipped,
                }

    return examples, labels, docs, {
        "tokens_loaded": len(labels),
        "labels": dict(label_counts),
        "skipped_tokens": skipped,
    }


def token_features(tokens: list[str], index: int) -> dict[str, Any]:
    token = tokens[index]
    prev_token = tokens[index - 1] if index > 0 else "<BOS>"
    next_token = tokens[index + 1] if index + 1 < len(tokens) else "<EOS>"
    return {
        "token": token,
        "token_lower": token.lower(),
        "prev_token": prev_token.lower(),
        "next_token": next_token.lower(),
        "suffix_1": token[-1:],
        "suffix_2": token[-2:],
        "suffix_3": token[-3:],
        "prefix_1": token[:1],
        "prefix_2": token[:2],
        "prefix_3": token[:3],
        "len": len(token),
        "is_digit": token.isdigit(),
        "has_digit": any(char.isdigit() for char in token),
        "position_mod_10": index % 10,
    }


def evaluate_model(
    pipeline: Pipeline,
    model_name: str,
    X_test: list[dict[str, Any]],
    y_test: list[str],
    figures_dir: Path,
    tables_dir: Path,
) -> dict[str, Any]:
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test,
        y_pred,
        labels=LABELS,
        zero_division=0,
    )
    macro = precision_recall_fscore_support(
        y_test,
        y_pred,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )
    weighted = precision_recall_fscore_support(
        y_test,
        y_pred,
        labels=LABELS,
        average="weighted",
        zero_division=0,
    )

    y_test_num = np.array([LABELS.index(label) for label in y_test])
    y_pred_num = np.array([LABELS.index(label) for label in y_pred])
    r2 = r2_score(y_test_num, y_pred_num)

    report = classification_report(y_test, y_pred, labels=LABELS, zero_division=0, output_dict=True)
    report_path = tables_dir / f"{model_name}_classification_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    cm = confusion_matrix(y_test, y_pred, labels=LABELS)
    np.savetxt(tables_dir / f"{model_name}_confusion_matrix.csv", cm, delimiter=",", fmt="%d")
    plot_confusion_matrix(cm, model_name, figures_dir / f"{model_name}_confusion_matrix.png")

    curve_metrics = plot_curves(pipeline, X_test, y_test, model_name, figures_dir, tables_dir)

    return {
        "accuracy": round(float(accuracy), 6),
        "macro_precision": round(float(macro[0]), 6),
        "macro_recall": round(float(macro[1]), 6),
        "macro_f1": round(float(macro[2]), 6),
        "weighted_precision": round(float(weighted[0]), 6),
        "weighted_recall": round(float(weighted[1]), 6),
        "weighted_f1": round(float(weighted[2]), 6),
        "r2_numeric_label_diagnostic": round(float(r2), 6),
        "per_class": {
            label: {
                "precision": round(float(precision[i]), 6),
                "recall": round(float(recall[i]), 6),
                "f1": round(float(f1[i]), 6),
                "support": int(support[i]),
            }
            for i, label in enumerate(LABELS)
        },
        "curves": curve_metrics,
    }


def plot_confusion_matrix(cm: np.ndarray, model_name: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
    display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"{model_name} confusion matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_curves(
    pipeline: Pipeline,
    X_test: list[dict[str, Any]],
    y_test: list[str],
    model_name: str,
    figures_dir: Path,
    tables_dir: Path,
) -> dict[str, Any]:
    classifier = pipeline.named_steps["classifier"]
    if hasattr(classifier, "predict_proba"):
        scores = pipeline.predict_proba(X_test)
    elif hasattr(classifier, "decision_function"):
        raw_scores = pipeline.decision_function(X_test)
        scores = _softmax(raw_scores)
    else:
        return {"available": False}

    class_order = list(pipeline.named_steps["classifier"].classes_)
    ordered_scores = np.zeros((len(y_test), len(LABELS)))
    for index, label in enumerate(class_order):
        if label in LABELS:
            ordered_scores[:, LABELS.index(label)] = scores[:, index]

    y_bin = label_binarize(y_test, classes=LABELS)
    try:
        roc_auc = roc_auc_score(y_bin, ordered_scores, average="macro", multi_class="ovr")
    except ValueError:
        roc_auc = float("nan")

    fig, ax = plt.subplots(figsize=(6, 5))
    for i, label in enumerate(LABELS):
        RocCurveDisplay.from_predictions(y_bin[:, i], ordered_scores[:, i], name=label, ax=ax)
    ax.set_title(f"{model_name} ROC curves")
    fig.tight_layout()
    fig.savefig(figures_dir / f"{model_name}_roc_curves.png", dpi=180)
    plt.close(fig)

    pr_auc: dict[str, float] = {}
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, label in enumerate(LABELS):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], ordered_scores[:, i])
        pr_auc[label] = float(auc(recall, precision))
        PrecisionRecallDisplay(precision=precision, recall=recall).plot(ax=ax, name=label)
    ax.set_title(f"{model_name} precision-recall curves")
    fig.tight_layout()
    fig.savefig(figures_dir / f"{model_name}_precision_recall_curves.png", dpi=180)
    plt.close(fig)

    curve_path = tables_dir / f"{model_name}_curve_metrics.json"
    curve_metrics = {
        "available": True,
        "macro_roc_auc_ovr": None if np.isnan(roc_auc) else round(float(roc_auc), 6),
        "pr_auc": {label: round(value, 6) for label, value in pr_auc.items()},
    }
    curve_path.write_text(json.dumps(curve_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return curve_metrics


def _softmax(values: np.ndarray) -> np.ndarray:
    if values.ndim == 1:
        values = np.vstack([-values, values]).T
    shifted = values - values.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def write_summary_csv(metrics: dict[str, Any], output_path: Path) -> None:
    fields = [
        "model",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "comma_f1",
        "r2_numeric_label_diagnostic",
        "fit_seconds",
        "eval_seconds",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model, values in metrics.items():
            writer.writerow(
                {
                    "model": model,
                    "accuracy": values["accuracy"],
                    "macro_precision": values["macro_precision"],
                    "macro_recall": values["macro_recall"],
                    "macro_f1": values["macro_f1"],
                    "weighted_f1": values["weighted_f1"],
                    "comma_f1": values["per_class"]["C"]["f1"],
                    "r2_numeric_label_diagnostic": values["r2_numeric_label_diagnostic"],
                    "fit_seconds": values["fit_seconds"],
                    "eval_seconds": values["eval_seconds"],
                }
            )


def write_markdown_report(metadata: dict[str, Any], output_path: Path) -> None:
    lines = [
        f"# APR Baseline Report: {metadata['run_id']}",
        "",
        f"- Dataset: `{metadata['dataset_path']}`",
        f"- Max tokens: {metadata['max_tokens']}",
        f"- Documents loaded: {metadata['documents_loaded']}",
        f"- Total runtime seconds: {metadata['runtime_seconds_total']}",
        "",
        "## Split",
        "",
        f"- Train tokens: {metadata['split']['train_tokens']}",
        f"- Validation tokens: {metadata['split']['validation_tokens']}",
        f"- Test tokens: {metadata['split']['test_tokens']}",
        "",
        "## Summary Metrics",
        "",
        "| Model | Accuracy | Macro-F1 | Comma F1 | ROC-AUC | R2 diagnostic | Fit seconds |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model, values in metadata["metrics"].items():
        roc_auc = values["curves"].get("macro_roc_auc_ovr") if values.get("curves") else None
        lines.append(
            f"| {model} | {values['accuracy']} | {values['macro_f1']} | "
            f"{values['per_class']['C']['f1']} | {roc_auc} | "
            f"{values['r2_numeric_label_diagnostic']} | {values['fit_seconds']} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- R2 is not a primary classification metric; it is saved only because the user requested an R2-style diagnostic.",
            "- Primary metrics for the paper should be macro-F1 and class-wise precision/recall/F1, especially comma F1.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
