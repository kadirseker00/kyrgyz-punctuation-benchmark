import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import BertConfig, BertForTokenClassification, PreTrainedTokenizerFast

from kg_punctuation import hf_generic_experiment as hf
from kg_punctuation.benchmark_v2 import Benchmark, score
from kg_punctuation.hf_generic_cross_eval import evaluate_generic_cross_domain
from kg_punctuation.learning_curve import run_learning_curve
from kg_punctuation.learning_curve import _audit_canonical_overlap


@pytest.fixture
def tiny_benchmark(tmp_path):
    vocab = {word: i for i, word in enumerate([
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "alpha", "beta", "gamma", "##s",
    ])}
    backend = Tokenizer(WordPiece(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]",
        cls_token="[CLS]", sep_token="[SEP]", model_max_length=32,
    )
    model_path = tmp_path / "tiny_model"
    tokenizer.save_pretrained(model_path)
    BertForTokenClassification(BertConfig(
        vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=24, max_position_embeddings=32,
        num_labels=5,
    )).save_pretrained(model_path)
    rows = [{
        "document_id": f"d{i}",
        "tokens": ["alpha", "betas", "gamma"] * 9,
        "labels": (["S", "C", "P", "Q", "E"] * 6)[:27],
    } for i in range(6)]
    data = tmp_path / "data.jsonl"
    data.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "version": "v2", "seed": 42, "main_train_documents": 2,
        "sources": {"tiny": {
            "data_file": str(data), "sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
            "group_of": {f"d{i}": f"tiny:g{i}" for i in range(6)},
            "splits": {"train": ["d0", "d1"], "validation": ["d2", "d3"], "test": ["d4", "d5"]},
        }},
    }))
    return tokenizer, model_path, data, manifest


def test_training_windows_keep_tail_words_and_count_each_once(tiny_benchmark):
    tokenizer, _, _, manifest = tiny_benchmark
    docs = Benchmark(manifest).documents("tiny", "train", "extended")
    labels = {label: i for i, label in enumerate(["S", "C", "P", "Q", "E"])}
    dataset = hf.TokenClassificationDataset(docs, tokenizer, 10, labels, full_document_windows=True)
    observed = Counter(value.item() for row in dataset for value in row["labels"] if value != -100)
    expected = Counter(labels[label] for doc in docs for label in doc["labels"])
    assert observed == expected
    assert sum(observed.values()) == 54
    assert len(dataset) > len(docs)
    assert all(row["input_ids"].shape == (10,) for row in dataset)


@pytest.mark.parametrize("loss_weighting", ["inverse_frequency", "none"])
def test_canonical_train_and_cross_archive_complete_predictions(tiny_benchmark, tmp_path, monkeypatch, loss_weighting):
    _, model_path, data, manifest = tiny_benchmark
    monkeypatch.setattr(hf, "select_device", lambda: torch.device("cpu"))
    monkeypatch.setattr("kg_punctuation.hf_generic_cross_eval.select_device", lambda: torch.device("cpu"))
    run = hf.run_hf_experiment(
        data, tmp_path / "runs", model_id=str(model_path), max_length=10, batch_size=4,
        epochs=1, label_set="extended", manifest_path=manifest, source="tiny",
        loss_weighting=loss_weighting,
    )
    run_dir = Path(run["output_dir"])
    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["best_epoch"] == 1
    assert metadata["hyperparameters"]["loss_weighting"] == loss_weighting
    if loss_weighting == "none":
        assert set(metadata["hyperparameters"]["class_weights"].values()) == {1.0}
    assert metadata["benchmark"]["train_document_ids"] == ["d0", "d1"]
    assert metadata["benchmark"]["full_word_coverage"] is True
    assert "max_documents" not in metadata["hyperparameters"]
    assert metadata["split"]["train_sequences"] > 2
    with (run_dir / "tables" / "predictions.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 54
    assert rows[-1]["word_index"] == "26"
    assert rows[-1]["group_id"] == "tiny:g5"
    checked = score(run_dir / "tables" / "predictions.csv", Benchmark(manifest), "tiny", view="extended")
    assert checked["macro_f1"] == pytest.approx(metadata["metrics"]["hf_token_classifier"]["macro_f1"], abs=1e-6)
    cross = evaluate_generic_cross_domain(
        run_dir / "models" / "hf_model", data, tmp_path / "cross", "tiny", "tiny",
        max_length=10, batch_size=4, label_set="extended", manifest_path=manifest,
    )
    cross_file = Path(cross["output_dir"]) / "tables" / "predictions.csv"
    assert cross_file.read_bytes() == (run_dir / "tables" / "predictions.csv").read_bytes()


def test_zero_train_budget_is_rejected(tiny_benchmark, tmp_path):
    _, model_path, data, manifest = tiny_benchmark
    with pytest.raises(ValueError, match="train_documents must be positive"):
        hf.run_hf_experiment(
            data, tmp_path / "runs", model_id=str(model_path), manifest_path=manifest,
            source="tiny", train_documents=0,
        )


def test_word_logits_match_canonical_labels_through_overlapping_windows(tiny_benchmark):
    from kg_punctuation.benchmark_v2 import predict_words
    from kg_punctuation.rare_calibration import LABELS, infer_logits

    tokenizer, model_path, _, manifest = tiny_benchmark
    model = BertForTokenClassification.from_pretrained(model_path)
    model.config.id2label = dict(enumerate(LABELS))
    docs = Benchmark(manifest).documents("tiny", "test", "extended")
    scores = infer_logits(model, tokenizer, docs, torch.device("cpu"), max_length=10, batch_size=4)
    expected = predict_words(model, tokenizer, docs, model.config.id2label, max_length=10,
                             batch_size=4, device=torch.device("cpu"))
    assert scores.shape == (54, 5)
    assert [LABELS[int(i)] for i in scores.argmax(axis=1)] == [x for d in expected for x in d]


def test_curve_uses_nested_training_and_identical_test_positions(tiny_benchmark, tmp_path, monkeypatch):
    _, model_path, data, manifest = tiny_benchmark
    monkeypatch.setattr(hf, "select_device", lambda: torch.device("cpu"))
    result = run_learning_curve(
        data, tmp_path / "curves", model_id=str(model_path), budgets=(1, 2, 3),
        max_length=10, batch_size=4, epochs=1, label_set="extended",
        manifest_path=manifest, source="tiny",
    )
    root = Path(result["output_dir"])
    summary = json.loads((root / "metadata.json").read_text())
    assert summary["effective_budgets"] == [1, 2]
    assert summary["skipped_budgets"] == [3]
    positions = []
    for index, row in enumerate(result["results"], start=1):
        run_dir = Path(row["run_dir"])
        metadata = json.loads((run_dir / "metadata.json").read_text())
        assert metadata["benchmark"]["train_document_ids"] == ["d0", "d1"][:index]
        assert (run_dir / "models" / "hf_model" / "config.json").exists()
        with (run_dir / "tables" / "predictions.csv").open() as handle:
            positions.append([(r["document_id"], r["word_index"], r["gold"]) for r in csv.DictReader(handle)])
    assert positions[0] == positions[1]
    assert (root / "learning_curve.pdf").exists()
    assert (root / "learning_curve.csv").exists()


def test_overlap_audit_keeps_one_fixed_subset_across_budgets(tmp_path):
    words_a = [f"a{i}" for i in range(6)]
    words_b = [f"b{i}" for i in range(6)]
    labels = ["S"] * 5 + ["P"]
    test_doc = {"document_id": "t", "tokens": words_a + words_b + ["tail"] * 3,
                "labels": labels * 2 + ["S"] * 3}

    class SmallBenchmark:
        def documents(self, source, split, view):
            if split == "test":
                return [test_doc]
            return [{"tokens": words_a, "labels": labels}, {"tokens": words_b, "labels": labels}]

    results = []
    for budget in (1, 2):
        run_dir = tmp_path / str(budget)
        (run_dir / "tables").mkdir(parents=True)
        with (run_dir / "tables/predictions.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["source", "document_id", "word_index", "gold", "prediction"])
            writer.writerows(("tiny", "t", i, label, label) for i, label in enumerate(test_doc["labels"]))
        results.append({"budget": budget, "run_dir": str(run_dir)})
    audit = _audit_canonical_overlap(SmallBenchmark(), "tiny", "core", results, tmp_path)
    assert audit["fixed_subset_tokens"] == 3
    assert [row["subsets"]["seen_at_budget"]["tokens"] for row in audit["results"]] == [6, 12]
    assert [row["subsets"]["fixed_outside_full_pool_overlap"]["tokens"] for row in audit["results"]] == [3, 3]
