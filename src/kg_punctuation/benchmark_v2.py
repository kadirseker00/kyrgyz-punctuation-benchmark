"""Canonical v2 benchmark: de-duplicated document splits and shared full-document evaluation (T14).

Every model family is trained on documents from the same ordered train pool and scored on the
same ordered test words, identified by (source, document_id, word_index). ``word_index`` indexes
the ``tokens`` list of the v2 source file and is identical for the core and extended views.

Build the manifest (no GPU):
    PYTHONPATH=src .venv/bin/python -m kg_punctuation.benchmark_v2 build-manifest
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "data/processed/splits/canonical_v2.json"
DEFAULT_SOURCES = {
    "kyrgyznews": ROOT / "data/processed/v2/kyrgyznews.jsonl",
    "wikipedia_ky": ROOT / "data/processed/v2/wikipedia_ky.jsonl",
}
SPLIT_SIZES = {"test": 1250, "validation": 500}
MAIN_TRAIN_DOCUMENTS = 3200
MIN_SENTENCE_TOKENS = 5
NEAR_DUP_MIN_OVERLAP = 0.5
SENTENCE_END = {"P", "Q", "E"}

VIEWS = {
    # v2 core: question and exclamation marks end a sentence, so they project to P.
    "core": ({"S": "S", "C": "C", "P": "P", "Q": "P", "E": "P"}, ["S", "C", "P"]),
    "extended": ({"S": "S", "C": "C", "P": "P", "Q": "Q", "E": "E"}, ["S", "C", "P", "Q", "E"]),
}
PREDICTION_FIELDS = ["source", "document_id", "group_id", "word_index", "gold", "prediction"]


# --------------------------------------------------------------------------- labels

def project(labels: Iterable[str], view: str) -> list[str]:
    mapping, _ = VIEWS[view]
    return [mapping[label] for label in labels]


def view_labels(view: str) -> list[str]:
    return list(VIEWS[view][1])


# --------------------------------------------------------------------------- manifest

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sentence_spans(labels: Sequence[str]) -> list[tuple[int, int]]:
    """[start, end) word spans that end with a sentence-final label and have >= MIN_SENTENCE_TOKENS words."""
    spans, start = [], 0
    for index, label in enumerate(labels):
        if label in SENTENCE_END:
            if index + 1 - start >= MIN_SENTENCE_TOKENS:
                spans.append((start, index + 1))
            start = index + 1
    return spans


def _sentences(doc: dict[str, Any]) -> set[tuple[str, ...]]:
    return {tuple(doc["tokens"][a:b]) for a, b in sentence_spans(doc["labels"])}


class _UnionFind:
    def __init__(self, items: Iterable[str]):
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def deduplicate(docs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str], dict[str, list[str]]]:
    """Drop exact duplicates (keep the smallest document_id) and group near-duplicates.

    Two remaining documents are near-duplicates when the tokens of their shared sentences cover
    at least NEAR_DUP_MIN_OVERLAP of the shorter document. Groups are connected components.
    """
    by_tokens: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for doc in docs:
        by_tokens[tuple(doc["tokens"])].append(doc["document_id"])
    keep = {min(ids) for ids in by_tokens.values()}
    dropped = sorted(doc["document_id"] for doc in docs if doc["document_id"] not in keep)
    kept = [doc for doc in docs if doc["document_id"] in keep]

    postings: dict[tuple[str, ...], list[str]] = defaultdict(list)
    length = {}
    for doc in kept:
        length[doc["document_id"]] = len(doc["tokens"])
        for sentence in _sentences(doc):
            postings[sentence].append(doc["document_id"])
    shared: dict[tuple[str, str], int] = defaultdict(int)
    for sentence, ids in postings.items():
        if len(ids) < 2:
            continue
        ids = sorted(ids)
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                shared[(a, b)] += len(sentence)
    union = _UnionFind(length)
    for (a, b), tokens in shared.items():
        if tokens / min(length[a], length[b]) >= NEAR_DUP_MIN_OVERLAP:
            union.union(a, b)
    groups: dict[str, list[str]] = defaultdict(list)
    for doc_id in sorted(length):
        groups[union.find(doc_id)].append(doc_id)
    return kept, dropped, {root: sorted(ids) for root, ids in groups.items()}


def assign_splits(groups: dict[str, list[str]], seed: int) -> dict[str, list[str]]:
    order = sorted(groups)
    random.Random(seed).shuffle(order)
    splits = {"test": [], "validation": [], "train": []}
    for root in order:
        for name in ("test", "validation"):
            if len(splits[name]) < SPLIT_SIZES[name]:
                splits[name].extend(groups[root])
                break
        else:
            splits["train"].extend(groups[root])
    return splits


def _residual_overlap(docs: dict[str, dict[str, Any]], train_ids: list[str], eval_ids: list[str]) -> dict[str, Any]:
    train_sentences = set().union(*(_sentences(docs[i]) for i in train_ids)) if train_ids else set()
    total_words = seen_words = total = seen = 0
    for doc_id in eval_ids:
        doc = docs[doc_id]
        total_words += len(doc["tokens"])
        for a, b in sentence_spans(doc["labels"]):
            total += 1
            if tuple(doc["tokens"][a:b]) in train_sentences:
                seen += 1
                seen_words += b - a
    return {"sentences": total, "sentences_seen_in_train": seen,
            "sentences_seen_pct": round(100 * seen / max(1, total), 3),
            "words": total_words, "words_in_seen_sentences_pct": round(100 * seen_words / max(1, total_words), 3)}


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_manifest(sources: dict[str, Path] = DEFAULT_SOURCES, output_path: Path = DEFAULT_MANIFEST, seed: int = 42) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "version": "v2",
        "seed": seed,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "code_commit": _git_commit(),
        "label_views": {name: {"mapping": mapping, "labels": labels} for name, (mapping, labels) in VIEWS.items()},
        "split_sizes": SPLIT_SIZES,
        "main_train_documents": MAIN_TRAIN_DOCUMENTS,
        "near_dup_rule": {"min_sentence_tokens": MIN_SENTENCE_TOKENS, "sentence_end_labels": sorted(SENTENCE_END),
                          "min_shared_sentence_token_fraction_of_shorter_doc": NEAR_DUP_MIN_OVERLAP,
                          "exact_duplicate_representative": "smallest document_id"},
        "sources": {},
    }
    for name, path in sources.items():
        raw = _read_jsonl(path)
        kept, dropped, groups = deduplicate(raw)
        splits = assign_splits(groups, seed)
        docs = {doc["document_id"]: doc for doc in kept}
        group_of = {doc_id: root for root, ids in groups.items() for doc_id in ids}
        for a in splits:
            for b in splits:
                if a < b:
                    assert not {group_of[i] for i in splits[a]} & {group_of[i] for i in splits[b]}, (name, a, b)
        if len(splits["train"]) < MAIN_TRAIN_DOCUMENTS:
            raise ValueError(f"{name}: train pool {len(splits['train'])} < main budget {MAIN_TRAIN_DOCUMENTS}")
        sizes = sorted((len(ids) for ids in groups.values()), reverse=True)
        main_train = splits["train"][:MAIN_TRAIN_DOCUMENTS]
        manifest["sources"][name] = {
            "data_file": str(path.relative_to(ROOT)),
            "sha256": _sha256(path),
            "pool_documents": len(raw),
            "dropped_exact": dropped,
            "near_dup_groups": {root: ids for root, ids in groups.items() if len(ids) > 1},
            "group_of": {doc_id: f"{name}:{root}" for doc_id, root in group_of.items()},
            "group_size_summary": {"groups": len(groups), "multi_document_groups": sum(s > 1 for s in sizes),
                                   "largest_groups": sizes[:10]},
            "splits": splits,
            "split_documents": {k: len(v) for k, v in splits.items()},
            "split_words": {k: sum(len(docs[i]["tokens"]) for i in v) for k, v in splits.items()},
            "residual_sentence_overlap": {
                "test_vs_main_train": _residual_overlap(docs, main_train, splits["test"]),
                "test_vs_full_train_pool": _residual_overlap(docs, splits["train"], splits["test"]),
                "validation_vs_main_train": _residual_overlap(docs, main_train, splits["validation"]),
            },
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return manifest


# --------------------------------------------------------------------------- data access

class Benchmark:
    """Read access to the canonical splits; verifies each data file against its manifest hash."""

    def __init__(self, manifest_path: Path = DEFAULT_MANIFEST, verify_hash: bool = True):
        self.manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        self._docs: dict[str, dict[str, dict[str, Any]]] = {}
        self._verify = verify_hash

    def _source_docs(self, source: str) -> dict[str, dict[str, Any]]:
        if source not in self._docs:
            meta = self.manifest["sources"][source]
            path = ROOT / meta["data_file"]
            if self._verify and _sha256(path) != meta["sha256"]:
                raise ValueError(f"{path} does not match the manifest sha256")
            self._docs[source] = {row["document_id"]: row for row in _read_jsonl(path)}
        return self._docs[source]

    def split_ids(self, source: str, split: str, limit: int | None = None) -> list[str]:
        ids = self.manifest["sources"][source]["splits"][split]
        if limit is not None:
            if split != "train":
                raise ValueError("limit is only defined for the ordered train pool")
            if limit > len(ids):
                raise ValueError(f"{source} train pool has {len(ids)} documents, requested {limit}")
            ids = ids[:limit]
        return list(ids)

    def documents(self, source: str, split: str, view: str = "core", limit: int | None = None) -> list[dict[str, Any]]:
        docs = self._source_docs(source)
        group_of = self.manifest["sources"][source].get("group_of", {})
        return [{"source": source, "document_id": doc_id, "group_id": group_of.get(doc_id, f"{source}:{doc_id}"),
                 "tokens": list(docs[doc_id]["tokens"]), "labels": project(docs[doc_id]["labels"], view)}
                for doc_id in self.split_ids(source, split, limit)]


# --------------------------------------------------------------------------- windowed inference

def word_windows(subword_counts: Sequence[int], budget: int) -> list[tuple[int, int]]:
    """Overlapping [start, end) word windows whose subword total fits ``budget``.

    Each window is maximal from its start; the next window starts at the current window's middle
    word, so every word except near document edges is seen with context on both sides.
    """
    n = len(subword_counts)
    if n == 0:
        return []
    if max(subword_counts) > budget:
        raise ValueError(f"a single word needs {max(subword_counts)} subwords > budget {budget}")
    windows, start = [], 0
    while True:
        end, used = start, 0
        while end < n and used + subword_counts[end] <= budget:
            used += subword_counts[end]
            end += 1
        windows.append((start, end))
        if end >= n:
            return windows
        start = max(start + 1, start + (end - start) // 2)


def fit_long_words(tokenizer, words: Sequence[str], budget: int) -> tuple[list[str], list[int]]:
    """Return (words, subword counts) where any word needing more than ``budget`` sub-tokens is cut
    to its longest character prefix that fits. Such words are parsing debris (e.g. flattened wiki
    tables); the cut is label-independent and leaves every other word untouched.
    """
    words = list(words)
    counts = [len(ids) or 1 for ids in tokenizer(words, add_special_tokens=False)["input_ids"]]
    for i, count in enumerate(counts):
        if count <= budget:
            continue
        word = words[i]
        lo, hi = 1, len(word)
        while lo < hi:  # longest prefix whose sub-token count fits the budget
            mid = (lo + hi + 1) // 2
            if len(tokenizer([word[:mid]], add_special_tokens=False)["input_ids"][0]) <= budget:
                lo = mid
            else:
                hi = mid - 1
        words[i] = word[:lo]
        counts[i] = len(tokenizer([words[i]], add_special_tokens=False)["input_ids"][0]) or 1
    return words, counts


def owner_windows(windows: Sequence[tuple[int, int]], n_words: int) -> list[int]:
    """For each word, the index of the window whose centre is closest (ties -> earlier window)."""
    owner = [-1] * n_words
    best = [float("inf")] * n_words
    for w, (start, end) in enumerate(windows):
        centre = (start + end - 1) / 2
        for i in range(start, end):
            distance = abs(i - centre)
            if distance < best[i]:
                best[i], owner[i] = distance, w
    if -1 in owner:
        raise AssertionError("uncovered word in windowing")
    return owner


def predict_words(model, tokenizer, docs: Sequence[dict[str, Any]], id2label: dict[int, str],
                  subtoken: str = "first", max_length: int = 128, batch_size: int = 16,
                  label_map: dict[str, str] | None = None, device=None) -> list[list[str]]:
    """Predict one label per word for every word of every document (no truncation).

    ``subtoken`` selects which sub-token's logits represent a word ("first" for our models,
    "last" for checkpoints trained that way). ``label_map`` projects model label names onto the
    evaluation inventory (e.g. {"QUESTION": "P", ...}); unmapped names raise.
    """
    import torch

    if subtoken not in ("first", "last"):
        raise ValueError(subtoken)
    budget = max_length - tokenizer.num_special_tokens_to_add(pair=False)
    jobs = []  # (doc index, window start, window end)
    owners = []
    fitted = []
    for d, doc in enumerate(docs):
        words, counts = fit_long_words(tokenizer, doc["tokens"], budget)
        fitted.append(words)
        windows = word_windows(counts, budget)
        owners.append((len(jobs), owner_windows(windows, len(words))))
        jobs.extend((d, s, e) for s, e in windows)

    window_preds: list[list[str]] = []
    model.eval()
    for i in range(0, len(jobs), batch_size):
        batch_jobs = jobs[i:i + batch_size]
        words = [fitted[d][s:e] for d, s, e in batch_jobs]
        encoding = tokenizer(words, is_split_into_words=True, truncation=False, padding=True, return_tensors="pt")
        if encoding["input_ids"].shape[1] > max_length:
            raise AssertionError("window exceeds max_length; subword counting is inconsistent")
        inputs = {k: v.to(device) if device is not None else v for k, v in encoding.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted = logits.argmax(dim=-1).cpu().tolist()
        for b, (d, s, e) in enumerate(batch_jobs):
            word_ids = encoding.word_ids(b)
            position: dict[int, int] = {}
            for p, wid in enumerate(word_ids):
                if wid is None:
                    continue
                if subtoken == "last" or wid not in position:
                    position[wid] = p
            if len(position) != e - s:
                raise AssertionError("a word produced no sub-token")
            labels = []
            for wid in range(e - s):
                name = id2label[predicted[b][position[wid]]]
                if label_map is not None:
                    name = label_map[name]
                labels.append(name)
            window_preds.append(labels)

    results = []
    for d, doc in enumerate(docs):
        first_job, owner = owners[d]
        labels = []
        for word, w in enumerate(owner):
            start = jobs[first_job + w][1]
            labels.append(window_preds[first_job + w][word - start])
        results.append(labels)
    return results


def encode_training_windows(tokenizer, docs: Sequence[dict[str, Any]], label2id: dict[str, int],
                            max_length: int = 128, ignore_index: int = -100) -> list[dict[str, Any]]:
    """Encode full documents as the same overlapping windows used by ``predict_words``.

    Each word carries its label exactly once in the whole document: on the first sub-token of the
    word inside its owner window (the window whose centre is closest). All other positions get
    ``ignore_index``, so overlapping context never double-counts a word in the loss.
    Returns unpadded features ``{"input_ids", "attention_mask", "labels"}`` (lists of ints).
    """
    budget = max_length - tokenizer.num_special_tokens_to_add(pair=False)
    features = []
    for doc in docs:
        words, counts = fit_long_words(tokenizer, doc["tokens"], budget)
        windows = word_windows(counts, budget)
        owner = owner_windows(windows, len(words))
        for w, (start, end) in enumerate(windows):
            encoding = tokenizer(words[start:end], is_split_into_words=True, truncation=False)
            if len(encoding["input_ids"]) > max_length:
                raise AssertionError("training window exceeds max_length")
            labels, previous = [], None
            for wid in encoding.word_ids():
                if wid is not None and wid != previous and owner[start + wid] == w:
                    labels.append(label2id[doc["labels"][start + wid]])
                else:
                    labels.append(ignore_index)
                previous = wid
            features.append({"input_ids": list(encoding["input_ids"]),
                             "attention_mask": list(encoding["attention_mask"]), "labels": labels})
    return features


# --------------------------------------------------------------------------- predictions and scoring

def write_predictions(path: Path, docs: Sequence[dict[str, Any]], predictions: Sequence[Sequence[str]]) -> None:
    if len(docs) != len(predictions):
        raise ValueError("one prediction list per document is required")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(PREDICTION_FIELDS)
        for doc, preds in zip(docs, predictions):
            if len(preds) != len(doc["tokens"]):
                raise ValueError(f"{doc['document_id']}: {len(preds)} predictions for {len(doc['tokens'])} words")
            group_id = doc.get("group_id", f"{doc['source']}:{doc['document_id']}")
            for index, (gold, pred) in enumerate(zip(doc["labels"], preds)):
                writer.writerow([doc["source"], doc["document_id"], group_id, index, gold, pred])


def score(path: Path, benchmark: Benchmark, source: str, split: str = "test", view: str = "core",
          prediction_map: dict[str, str] | None = None) -> dict[str, Any]:
    """Score a prediction CSV; asserts the exact ordered (source, id, word_index, gold) sequence."""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    docs = benchmark.documents(source, split, view)
    expected = [(source, d["document_id"], i, g) for d in docs for i, g in enumerate(d["labels"])]
    with Path(path).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(expected):
        raise AssertionError(f"{path}: {len(rows)} rows, expected {len(expected)} test words")
    y_true, y_pred = [], []
    for row, key in zip(rows, expected):
        got = (row["source"], row["document_id"], int(row["word_index"]), project([row["gold"]], view)[0])
        if got != key:
            raise AssertionError(f"{path}: row {got} does not match manifest key {key}")
        pred = row["prediction"] if prediction_map is None else prediction_map[row["prediction"]]
        y_true.append(key[3])
        y_pred.append(project([pred], view)[0])
    labels = view_labels(view)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    macro = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    return {
        "source": source, "split": split, "view": view, "words": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro[0]), "macro_recall": float(macro[1]), "macro_f1": float(macro[2]),
        "per_class": {lab: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                      for i, lab in enumerate(labels)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-manifest")
    build.add_argument("--output", type=Path, default=DEFAULT_MANIFEST)
    build.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.command == "build-manifest":
        manifest = build_manifest(output_path=args.output, seed=args.seed)
        for name, meta in manifest["sources"].items():
            print(name, json.dumps({k: meta[k] for k in ("pool_documents", "split_documents", "split_words", "group_size_summary")}))
            print("  dropped_exact", len(meta["dropped_exact"]), "residual", json.dumps(meta["residual_sentence_overlap"]))


if __name__ == "__main__":
    main()
