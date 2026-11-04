"""Validation-selected Q/E logit offsets using the canonical word-window policy."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from kg_punctuation.benchmark_v2 import (Benchmark, DEFAULT_MANIFEST, ROOT, fit_long_words,
                                        owner_windows, score, word_windows, write_predictions)

LABELS = ("S", "C", "P", "Q", "E")
GRID = (0., .5, 1., 1.5, 2., 3., 4., 6., 8.)


def infer_logits(model, tokenizer, docs, device, max_length=128, batch_size=8):
    """One first-subtoken score vector per word, from its canonical owner window."""
    budget = max_length - tokenizer.num_special_tokens_to_add(pair=False)
    jobs, owners, fitted, arrays = [], [], [], []
    id_to_label = {int(k): v for k, v in model.config.id2label.items()}
    columns = [next(i for i, value in id_to_label.items() if value == label) for label in LABELS]
    for d, doc in enumerate(docs):
        words, counts = fit_long_words(tokenizer, doc["tokens"], budget)
        windows = word_windows(counts, budget)
        owners.append((len(jobs), owner_windows(windows, len(words))))
        fitted.append(words)
        arrays.append(np.full((len(words), len(LABELS)), np.nan, dtype=np.float32))
        jobs.extend((d, start, end) for start, end in windows)
    model.eval()
    for offset in range(0, len(jobs), batch_size):
        selected = jobs[offset:offset + batch_size]
        encoding = tokenizer([fitted[d][a:b] for d, a, b in selected], is_split_into_words=True,
                             truncation=False, padding=True, return_tensors="pt")
        if encoding["input_ids"].shape[1] > max_length:
            raise ValueError("Canonical inference window exceeds its budget")
        with torch.no_grad():
            values = model(**{k: v.to(device) for k, v in encoding.items()}).logits.cpu().numpy()
        for batch, (d, start, end) in enumerate(selected):
            positions = {}
            for pos, wid in enumerate(encoding.word_ids(batch)):
                if wid is not None:
                    positions.setdefault(wid, pos)
            if len(positions) != end - start:
                raise ValueError("A word produced no subtoken")
            first, owner = owners[d]
            for wid, pos in positions.items():
                if first + owner[start + wid] == offset + batch:
                    arrays[d][start + wid] = values[batch, pos, columns]
    if any(not np.isfinite(a).all() for a in arrays):
        raise ValueError("Missing or nonfinite word scores")
    return np.concatenate(arrays)


def class_metrics(gold, predicted):
    confusion = np.bincount(np.asarray(gold) * 5 + predicted, minlength=25).reshape(5, 5)
    tp = np.diag(confusion)
    support, counts = confusion.sum(axis=1), confusion.sum(axis=0)
    precision = np.divide(tp, counts, out=np.zeros(5), where=counts != 0)
    recall = np.divide(tp, support, out=np.zeros(5), where=support != 0)
    f1 = np.divide(2 * tp, support + counts, out=np.zeros(5), where=(support + counts) != 0)
    return {"macro_f1": float(f1.mean()), "per_class": {
        label: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]),
                "support": int(support[i]), "predicted": int(counts[i])} for i, label in enumerate(LABELS)}}


def select_offsets(validation_logits, validation_gold, grid=GRID):
    """Fit exclusively on validation; ties prefer the smallest total penalty."""
    if validation_logits.shape != (len(validation_gold), 5) or not len(validation_gold):
        raise ValueError("Validation scores must be a nonempty [words, 5] array")
    if not np.isfinite(validation_logits).all() or any(x < 0 for x in grid):
        raise ValueError("Scores must be finite and penalties nonnegative")
    trials = []
    for q, e in itertools.product(grid, repeat=2):
        adjusted = validation_logits + np.asarray([0, 0, 0, -q, -e])
        metric = class_metrics(validation_gold, adjusted.argmax(axis=1))["macro_f1"]
        trials.append({"q_penalty": q, "e_penalty": e, "validation_macro_f1": metric})
    best = max(trials, key=lambda r: (r["validation_macro_f1"], -r["q_penalty"] - r["e_penalty"],
                                      -r["q_penalty"], -r["e_penalty"]))
    return {"selection_split": "validation", "objective": "five-label macro-F1",
            "grid": list(grid), "selected": best, "trials": trials}


def predict(logits, selection):
    fitted = selection["selected"]
    return (logits + np.asarray([0, 0, 0, -fitted["q_penalty"], -fitted["e_penalty"]])).argmax(axis=1)


def run(model_path, source, output, device="cpu", threads=4):
    torch.set_num_threads(threads)
    output.mkdir(parents=True, exist_ok=True)
    benchmark = Benchmark(DEFAULT_MANIFEST)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_path).to(device)
    validation = benchmark.documents(source, "validation", "extended")
    vgold = np.asarray([LABELS.index(l) for doc in validation for l in doc["labels"]])
    vlogits = infer_logits(model, tokenizer, validation, device)
    selection = select_offsets(vlogits, vgold)
    selection.update(source=source, model_path=str(model_path),
                     manifest_sha256=hashlib.sha256(DEFAULT_MANIFEST.read_bytes()).hexdigest(),
                     code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    # Persist the choice before reading test labels or producing test scores.
    selection_path = output / "validation_selection.json"
    selection_path.write_text(json.dumps(selection, indent=2) + "\n")
    np.savez_compressed(output / "validation_logits.npz", logits=vlogits, gold=vgold)
    test = benchmark.documents(source, "test", "extended")
    logits = infer_logits(model, tokenizer, test, device)
    np.savez_compressed(output / "test_logits.npz", logits=logits)
    result = {"selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
              "selection": selection, "conditions": {}}
    for condition, ids in (("raw", logits.argmax(axis=1)), ("validation_offsets", predict(logits, selection))):
        predictions, start = [], 0
        for doc in test:
            end = start + len(doc["labels"])
            predictions.append([LABELS[int(i)] for i in ids[start:end]])
            start = end
        path = output / f"predictions_{condition}.csv"
        write_predictions(path, test, predictions)
        result["conditions"][condition] = score(path, benchmark, source, view="extended")
        result["conditions"][condition]["prediction_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"source": source, "selected": selection["selected"],
                      "scores": {k: v["macro_f1"] for k, v in result["conditions"].items()}}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", choices=("kyrgyznews", "wikipedia_ky"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run(args.model, args.source, args.output, args.device)
