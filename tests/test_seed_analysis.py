import csv
import hashlib
import json

import numpy as np
import pytest
from sklearn.metrics import f1_score

from kg_punctuation.seed_analysis import (
    _bootstrap_scores,
    _cluster_confusions,
    analyze_seed_runs,
)


def make_runs(tmp_path, transform=None, groups=True):
    paths = []
    for seed in (42, 43, 44):
        path = tmp_path / str(seed)
        (path / "tables").mkdir(parents=True)
        (path / "metadata.json").write_text(json.dumps({
            "run_id": str(seed), "seed": seed, "label_set": ["S", "C", "P"],
        }))
        rows = [
            {"source": "news", "document_id": doc, "word_index": word,
             "group_id": group, "gold": gold, "prediction": pred}
            for doc, word, group, gold, pred in [
                ("a", 0, "g1", "S", "S"),
                ("a", 1, "g1", "C", "S"),
                ("b", 0, "g1", "P", "P"),
                ("c", 0, "g2", "S", "C"),
            ]
        ]
        if not groups:
            for row in rows:
                row.pop("group_id")
        if transform:
            transform(seed, rows)
        with (path / "tables" / "predictions.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        paths.append(path)
    return paths


def test_cluster_bootstrap_matches_explicit_resampling():
    # Unequal documents and absent E ensure this isn't an average of document F1s.
    gold = ["S", "C", "P", "S", "S", "C", "P"]
    predictions = [gold, ["C", "C", "S", "S", "C", "S", "P"]]
    labels = ["S", "C", "P", "E"]
    clusters = [("a",)] * 2 + [("b",)] * 4 + [("c",)]
    matrices = _cluster_confusions(gold, predictions, labels, clusters)
    actual = _bootstrap_scores(matrices, 40, np.random.default_rng(97))
    rng = np.random.default_rng(97)
    positions = [[0, 1], [2, 3, 4, 5], [6]]
    for sample in range(40):
        draw = rng.integers(0, 3, size=3)
        indices = [i for group in draw for i in positions[group]]
        for model, predicted in enumerate(predictions):
            expected = f1_score(
                [gold[i] for i in indices], [predicted[i] for i in indices],
                labels=labels, average="macro", zero_division=0,
            )
            assert actual[model, sample] == pytest.approx(expected)


@pytest.mark.parametrize("groups,unit,count", [(True, "group", 2), (False, "document", 3)])
def test_report_is_paired_reproducible_and_archived(tmp_path, groups, unit, count):
    runs = make_runs(tmp_path, groups=groups)
    output = tmp_path / "report.json"
    result = analyze_seed_runs(runs, output, bootstrap_samples=60)
    assert result == analyze_seed_runs(runs, output, bootstrap_samples=60)
    assert result == json.loads(output.read_text())
    assert result["bootstrap_unit"] == unit
    assert result["bootstrap_cluster_count"] == count
    assert result["test_documents"] == 3
    assert result["test_tokens"] == 4
    assert len(result["ordered_test_sha256"]) == 64
    assert "cluster bootstrap" in output.with_suffix(".md").read_text()
    for row in result["pairwise_against_first_seed"]:
        assert row["bootstrap_delta_ci_low"] == 0
        assert row["bootstrap_delta_ci_high"] == 0
        assert "mcnemar_exact_p" not in row


def test_same_gold_different_positions_rejected(tmp_path):
    def change(seed, rows):
        if seed == 43:
            rows[-1]["document_id"] = "different-document"
    runs = make_runs(tmp_path, change)
    with pytest.raises(ValueError, match="ordered test positions"):
        analyze_seed_runs(runs, tmp_path / "report.json")


@pytest.mark.parametrize("mutation,match", [
    (lambda rows: rows.append(dict(rows[0])), "Duplicate test positions"),
    (lambda rows: rows[0].update(prediction="OTHER"), "belong to label_set"),
    (lambda rows: rows[0].update(group_id=""), "every row"),
    (lambda rows: rows[1].update(group_id="g2"), "multiple clusters"),
    (lambda rows: rows[0].update(word_index=-1), "nonnegative"),
    (lambda rows: [row.pop("document_id") for row in rows], "stable source"),
])
def test_malformed_predictions_rejected(tmp_path, mutation, match):
    runs = make_runs(tmp_path, lambda seed, rows: mutation(rows) if seed == 42 else None)
    with pytest.raises(ValueError, match=match):
        analyze_seed_runs(runs, tmp_path / "report.json")


def test_cluster_assignment_changes_rejected(tmp_path):
    def change(seed, rows):
        if seed == 43:
            rows[-1]["group_id"] = "g3"
    runs = make_runs(tmp_path, change)
    with pytest.raises(ValueError, match="same cluster assignments"):
        analyze_seed_runs(runs, tmp_path / "report.json")


def test_gold_changes_rejected(tmp_path):
    def change(seed, rows):
        if seed == 43:
            rows[-1]["gold"] = "P"
    runs = make_runs(tmp_path, change)
    with pytest.raises(ValueError, match="same gold labels"):
        analyze_seed_runs(runs, tmp_path / "report.json")


def test_duplicate_seed_rejected(tmp_path):
    runs = make_runs(tmp_path)
    with pytest.raises(ValueError, match="distinct training seeds"):
        analyze_seed_runs([runs[0], runs[0], runs[2]], tmp_path / "report.json")


def test_different_training_protocol_is_rejected(tmp_path):
    runs = make_runs(tmp_path)
    path = runs[1] / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["hyperparameters"] = {"batch_size": 16}
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="controlled configuration"):
        analyze_seed_runs(runs, tmp_path / "report.json")


def test_one_cluster_cannot_estimate_uncertainty(tmp_path):
    runs = make_runs(tmp_path, lambda seed, rows: [row.update(group_id="g1") for row in rows])
    with pytest.raises(ValueError, match="two independent test clusters"):
        analyze_seed_runs(runs, tmp_path / "report.json")


def make_main_panel_runs(tmp_path):
    runs = make_runs(tmp_path)
    data = tmp_path / "data.jsonl"
    data.write_text("\n".join(json.dumps(row) for row in [
        {"document_id": "a", "tokens": ["one", "two"], "labels": ["S", "C"]},
        {"document_id": "b", "tokens": ["three"], "labels": ["P"]},
        {"document_id": "c", "tokens": ["four"], "labels": ["S"]},
        {"document_id": "train", "tokens": ["five"], "labels": ["P"]},
    ]) + "\n")
    digest = hashlib.sha256(data.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"sources": {"news": {
        "data_file": str(data), "sha256": digest,
        "group_of": {"a": "g1", "b": "g1", "c": "g2", "train": "g0"},
        "splits": {"train": ["train"], "validation": [], "test": ["a", "b", "c"]},
    }}}))
    for run in runs:
        metadata = {"job": run.name, "seed": int(run.name), "label_view": "core",
                    "manifest_sha256": {"news": digest}, "train_documents": {"news": 1},
                    "model_id": "test-model", "hyperparameters": {"epochs": 3},
                    "class_weights": {"S": 1, "C": 1, "P": 1},
                    "subtoken": "first", "windowing": "word windows"}
        (run / "metadata.json").write_text(json.dumps(metadata))
        (run / "tables/predictions.csv").rename(run / "predictions_test_news.csv")
    return runs, manifest


def test_main_panel_adapter_validates_manifest_without_changing_metadata(tmp_path):
    runs, manifest = make_main_panel_runs(tmp_path)
    before = [(run / "metadata.json").read_bytes() for run in runs]
    result = analyze_seed_runs(runs, tmp_path / "report.json", bootstrap_samples=20,
                               prediction_source="news", manifest_path=manifest)
    assert result["test_tokens"] == 4
    assert result["bootstrap_cluster_count"] == 2
    assert result["canonical_validation"]["exact_test_positions_gold_groups_verified"] is True
    assert result["canonical_validation"]["manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert [(run / "metadata.json").read_bytes() for run in runs] == before


@pytest.mark.parametrize("kind", ["missing_word", "source_hash", "changed_split"])
def test_main_panel_adapter_rejects_incompatible_artifacts(tmp_path, kind):
    runs, manifest = make_main_panel_runs(tmp_path)
    if kind == "missing_word":
        for run in runs:
            path = run / "predictions_test_news.csv"
            path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
    elif kind == "source_hash":
        path = runs[0] / "metadata.json"
        value = json.loads(path.read_text())
        value["manifest_sha256"]["news"] = "incorrect"
        path.write_text(json.dumps(value))
    else:
        value = json.loads(manifest.read_text())
        value["sources"]["news"]["splits"]["test"] = ["a", "b", "train"]
        manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="source hashes|exact canonical test"):
        analyze_seed_runs(runs, tmp_path / "report.json", bootstrap_samples=20,
                          prediction_source="news", manifest_path=manifest)
