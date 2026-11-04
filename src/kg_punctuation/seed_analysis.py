from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score


def analyze_seed_runs(
    run_dirs: list[Path],
    output_path: Path,
    bootstrap_samples: int = 2000,
    seed: int = 123,
    prediction_source: str | None = None,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    if len(run_dirs) < 3:
        raise ValueError("At least three seed runs are required.")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive.")
    benchmark = None
    if prediction_source is not None:
        from kg_punctuation.benchmark_v2 import Benchmark, DEFAULT_MANIFEST

        manifest_path = manifest_path or DEFAULT_MANIFEST
        benchmark = Benchmark(manifest_path)
    loaded = [_load_run(path, prediction_source, benchmark) for path in run_dirs]
    reference = loaded[0]
    labels = reference["metadata"]["label_set"]
    gold = reference["gold"]
    if len({item["metadata"]["seed"] for item in loaded}) != len(loaded):
        raise ValueError("Seed stability requires distinct training seeds.")
    if len({item["metadata"]["run_id"] for item in loaded}) != len(loaded):
        raise ValueError("Run IDs must be distinct.")
    for item in loaded[1:]:
        for field in ("model_id", "hyperparameters", "benchmark"):
            if item["metadata"].get(field) != reference["metadata"].get(field):
                raise ValueError(f"Seed runs differ in controlled configuration: {field}.")
        if item["metadata"]["label_set"] != labels:
            raise ValueError("All seed runs must use the same ordered label set.")
        if item["positions"] != reference["positions"]:
            raise ValueError("All seed runs must have the same ordered test positions.")
        if item["gold"] != gold:
            raise ValueError("All seed runs must have the same gold labels.")
        if item["cluster_unit"] != reference["cluster_unit"] or item["clusters"] != reference["clusters"]:
            raise ValueError("All seed runs must have the same cluster assignments.")

    # Sum confusion matrices, not document F1s: the estimand remains token macro-F1.
    matrices = _cluster_confusions(
        gold, [item["predictions"] for item in loaded], labels, reference["clusters"]
    )
    if matrices.shape[1] < 2:
        raise ValueError("At least two independent test clusters are required.")
    draws = _bootstrap_scores(matrices, bootstrap_samples, np.random.default_rng(seed))
    per_run = []
    for index, item in enumerate(loaded):
        low, high = np.percentile(draws[index], [2.5, 97.5])
        per_run.append(
            {
                "run_dir": str(item["path"]),
                "run_id": item["metadata"]["run_id"],
                "seed": item["metadata"]["seed"],
                "prediction_sha256": item["prediction_sha256"],
                "metadata_sha256": item["metadata_sha256"],
                "macro_f1": float(f1_score(
                    gold, item["predictions"], labels=labels,
                    average="macro", zero_division=0,
                )),
                "macro_f1_ci_low": float(low),
                "macro_f1_ci_high": float(high),
                "accuracy": float(np.mean(np.array(gold) == np.array(item["predictions"]))),
            }
        )

    pairwise = []
    for index, item in enumerate(loaded[1:], start=1):
        low, high = np.percentile(draws[0] - draws[index], [2.5, 97.5])
        pairwise.append(
            {
                "reference_run_id": reference["metadata"]["run_id"],
                "comparison_run_id": item["metadata"]["run_id"],
                "reference_seed": reference["metadata"]["seed"],
                "comparison_seed": item["metadata"]["seed"],
                "macro_f1_delta_reference_minus_comparison": (
                    per_run[0]["macro_f1"] - per_run[index]["macro_f1"]
                ),
                "bootstrap_delta_ci_low": float(low),
                "bootstrap_delta_ci_high": float(high),
            }
        )

    result = {
        "analysis_version": 2,
        "labels": labels,
        "run_dirs": [str(path) for path in run_dirs],
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "bootstrap_unit": reference["cluster_unit"],
        "bootstrap_method": "paired cluster percentile, 95%, uniform clusters with replacement",
        "bootstrap_cluster_count": matrices.shape[1],
        "test_tokens": len(gold),
        "test_documents": len({position[:2] for position in reference["positions"]}),
        "ordered_test_sha256": hashlib.sha256(json.dumps(
            list(zip(reference["positions"], gold)), ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest(),
        "interpretation": (
            "Intervals quantify test-cluster sampling uncertainty conditional on each "
            "trained model. Across-seed standard deviation separately measures training "
            "variability. Pairwise intervals are unadjusted descriptive comparisons. "
            "Matching prediction IDs does not independently establish manifest coverage "
            "or identical training settings; these must be audited in the runner."
        ),
        "per_run": per_run,
        "pairwise_against_first_seed": pairwise,
        "summary": {
            "mean_macro_f1": float(np.mean([row["macro_f1"] for row in per_run])),
            "std_macro_f1": float(np.std([row["macro_f1"] for row in per_run], ddof=1)),
            "min_macro_f1": float(min(row["macro_f1"] for row in per_run)),
            "max_macro_f1": float(max(row["macro_f1"] for row in per_run)),
        },
    }
    if benchmark is not None:
        result["canonical_validation"] = {
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "prediction_source": prediction_source,
            "exact_test_positions_gold_groups_verified": True,
            "training_provenance_limit": (
                "Original main-panel metadata records source-file hashes and training "
                "document counts, not a split-manifest digest or actual training IDs. "
                "Training IDs used for configuration comparison are inferred from the "
                "current manifest and runner policy; they are not historical attestations."
            ),
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(result, output_path.with_suffix(".md"))
    return result


def _load_run(run_dir: Path, prediction_source: str | None = None, benchmark=None) -> dict[str, Any]:
    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    prediction_path = run_dir / "tables" / "predictions.csv"
    expected = None
    if prediction_source is not None:
        from kg_punctuation.benchmark_v2 import view_labels

        if benchmark is None or metadata.get("label_view") not in ("core", "extended"):
            raise ValueError("Main-panel analysis requires a manifest and recorded label_view.")
        prediction_path = run_dir / f"predictions_test_{prediction_source}.csv"
        source_hashes = {source: row["sha256"] for source, row in benchmark.manifest["sources"].items()}
        if metadata["manifest_sha256"] != source_hashes:
            raise ValueError("Main-panel source hashes differ from the canonical manifest.")
        metadata["run_id"] = metadata["job"]
        metadata["label_set"] = view_labels(metadata["label_view"])
        metadata["benchmark"] = {
            "source_sha256": source_hashes, "label_view": metadata["label_view"],
            "class_weights": metadata["class_weights"],
            "subtoken": metadata["subtoken"], "windowing": metadata["windowing"],
            "train_ids": {source: benchmark.split_ids(source, "train", count)
                          for source, count in metadata["train_documents"].items()},
        }
        docs = benchmark.documents(prediction_source, "test", metadata["label_view"])
        expected = [
            ((prediction_source, doc["document_id"], i), label, ("group", doc["group_id"]))
            for doc in docs for i, label in enumerate(doc["labels"])
        ]
    labels = metadata["label_set"]
    if not isinstance(labels, list) or not labels or len(set(labels)) != len(labels):
        raise ValueError("label_set must be a nonempty list of distinct labels.")
    gold, predictions, positions, clusters = [], [], [], []
    documents = {}
    required = {"source", "document_id", "word_index", "gold", "prediction"}
    with prediction_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"{prediction_path}: stable source/document_id/word_index columns required; "
                "legacy index-only predictions cannot be used for cluster inference."
            )
        has_groups = "group_id" in reader.fieldnames
        for row in reader:
            if not row["source"] or not row["document_id"]:
                raise ValueError("Source and document IDs must not be empty.")
            position = (row["source"], row["document_id"], int(row["word_index"]))
            if position[2] < 0:
                raise ValueError("word_index must be nonnegative.")
            if row["gold"] not in labels or row["prediction"] not in labels:
                raise ValueError("Gold and prediction labels must belong to label_set.")
            if has_groups and not row["group_id"]:
                raise ValueError("group_id must be present on every row when supplied.")
            cluster = ("group", row["group_id"]) if has_groups else position[:2]
            doc = position[:2]
            if doc in documents and documents[doc] != cluster:
                raise ValueError("A document cannot belong to multiple clusters.")
            documents[doc] = cluster
            positions.append(position)
            clusters.append(cluster)
            gold.append(row["gold"])
            predictions.append(row["prediction"])
    if not gold:
        raise ValueError("Prediction file must not be empty.")
    if len(set(positions)) != len(positions):
        raise ValueError("Duplicate test positions in prediction file.")
    if expected is not None and list(zip(positions, gold, clusters)) != expected:
        raise ValueError("Main-panel predictions do not cover the exact canonical test words/gold/groups.")
    return {
        "path": run_dir, "metadata": metadata, "gold": gold, "predictions": predictions,
        "positions": positions, "clusters": clusters,
        "cluster_unit": "group" if has_groups else "document",
        "prediction_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
        "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
    }


def _cluster_confusions(
    gold: list[str], predictions: list[list[str]], labels: list[str], clusters: list[tuple],
) -> np.ndarray:
    label_ids = {label: index for index, label in enumerate(labels)}
    cluster_ids = {cluster: index for index, cluster in enumerate(dict.fromkeys(clusters))}
    matrices = np.zeros((len(predictions), len(cluster_ids), len(labels), len(labels)), dtype=np.int64)
    cluster_index = np.asarray([cluster_ids[cluster] for cluster in clusters])
    gold_index = np.asarray([label_ids[label] for label in gold])
    for index, predicted in enumerate(predictions):
        pred_index = np.asarray([label_ids[label] for label in predicted])
        np.add.at(matrices[index], (cluster_index, gold_index, pred_index), 1)
    return matrices


def _macro_f1_from_confusions(matrices: np.ndarray) -> np.ndarray:
    numerator = 2.0 * np.diagonal(matrices, axis1=-2, axis2=-1)
    denominator = matrices.sum(axis=-1) + matrices.sum(axis=-2)
    scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
    return scores.mean(axis=-1)


def _bootstrap_scores(
    matrices: np.ndarray, samples: int, rng: np.random.Generator,
) -> np.ndarray:
    values = np.empty((matrices.shape[0], samples))
    cluster_count = matrices.shape[1]
    for index in range(samples):
        # Every model uses the same draw; all words in a cluster stay together.
        draw = rng.integers(0, cluster_count, size=cluster_count)
        values[:, index] = _macro_f1_from_confusions(matrices[:, draw].sum(axis=1))
    return values


def _write_markdown(result: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# Multi-seed stability and cluster uncertainty", "",
        f"- Labels: {', '.join(result['labels'])}",
        f"- Bootstrap samples: {result['bootstrap_samples']}",
        f"- Bootstrap unit: {result['bootstrap_unit']} ({result['bootstrap_cluster_count']} clusters)",
        f"- Test: {result['test_documents']} documents, {result['test_tokens']} tokens",
        f"- Ordered test SHA-256: {result['ordered_test_sha256']}", "",
        result["interpretation"], "",
        "| Seed | Macro-F1 | 95% cluster bootstrap CI | Accuracy |",
        "|---:|---:|---:|---:|",
    ]
    for row in result["per_run"]:
        lines.append(
            f"| {row['seed']} | {row['macro_f1']:.6f} | "
            f"[{row['macro_f1_ci_low']:.6f}, {row['macro_f1_ci_high']:.6f}] | "
            f"{row['accuracy']:.6f} |"
        )
    lines.extend([
        "", f"Mean macro-F1: {result['summary']['mean_macro_f1']:.6f}",
        f"Sample standard deviation: {result['summary']['std_macro_f1']:.6f}", "",
        "## Pairwise comparisons against the first seed", "",
        "| Reference seed | Comparison seed | Delta | Paired cluster bootstrap CI |",
        "|---:|---:|---:|---:|",
    ])
    for row in result["pairwise_against_first_seed"]:
        lines.append(
            f"| {row['reference_seed']} | {row['comparison_seed']} | "
            f"{row['macro_f1_delta_reference_minus_comparison']:.6f} | "
            f"[{row['bootstrap_delta_ci_low']:.6f}, {row['bootstrap_delta_ci_high']:.6f}] |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
