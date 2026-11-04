"""Audit train->test duplication under the exact document splits used by the experiments.

Reproduces the split of hf_token_experiment / deep_experiment / hf_generic_experiment
(shuffle with seed, take the first N documents, sklearn train_test_split 0.35 then 25/35)
and the learning-curve budgets, then counts test material that also occurs in training:
identical documents, and test sentences (>= MIN_SENT tokens, period-delimited on gold
labels) that occur verbatim in the training documents.

Usage: PYTHONPATH=src .venv/bin/python scripts/leakage_audit.py
Writes reports/r1/leakage_audit.{json,md}.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/source_level"
MIN_SENT = 5
CORE = ("S", "C", "P")


def load(path: Path, max_documents: int, seed: int = 42, labels: tuple[str, ...] = CORE) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    random.Random(seed).shuffle(rows)
    docs = []
    for row in rows[:max_documents]:
        kept = [(t, l) for t, l in zip(row["tokens"], row["labels"]) if l in labels]
        if kept:
            docs.append({"id": row["document_id"], "tokens": [t for t, _ in kept], "labels": [l for _, l in kept]})
    return docs


def split(docs: list[dict], seed: int = 42) -> tuple[list[dict], list[dict], list[dict]]:
    train, temp = train_test_split(docs, test_size=0.35, random_state=seed)
    valid, test = train_test_split(temp, test_size=25 / 35, random_state=seed)
    return train, valid, test


SENTENCE_END = {"P", "Q", "E"}


def sentences(doc: dict) -> list[tuple[str, ...]]:
    out, cur = [], []
    for token, label in zip(doc["tokens"], doc["labels"]):
        cur.append(token)
        if label in SENTENCE_END:
            out.append(tuple(cur))
            cur = []
    if cur:
        out.append(tuple(cur))
    return [s for s in out if len(s) >= MIN_SENT]


def overlap(train: list[dict], test: list[dict]) -> dict:
    train_docs = {tuple(d["tokens"]) for d in train}
    train_sents = {s for d in train for s in sentences(d)}
    test_sents = [s for d in test for s in sentences(d)]
    test_tokens = sum(len(d["tokens"]) for d in test)
    seen = [s for s in test_sents if s in train_sents]
    identical = sum(tuple(d["tokens"]) in train_docs for d in test)
    return {
        "train_documents": len(train),
        "test_documents": len(test),
        "test_documents_identical_to_train": identical,
        "test_documents_identical_pct": round(100 * identical / len(test), 2),
        "test_sentences": len(test_sents),
        "test_sentences_seen_in_train": len(seen),
        "test_sentences_seen_pct": round(100 * len(seen) / max(1, len(test_sents)), 2),
        "test_tokens": test_tokens,
        "test_tokens_in_seen_sentences_pct": round(100 * sum(map(len, seen)) / test_tokens, 2),
    }


def corpus_duplicates(path: Path) -> dict:
    rows = [tuple(json.loads(line)["tokens"]) for line in path.read_text(encoding="utf-8").splitlines()]
    return {"documents": len(rows), "exact_duplicate_documents": len(rows) - len(set(rows))}


def main() -> None:
    report: dict = {"min_sentence_tokens": MIN_SENT, "split": "shuffle(seed=42)[:N] -> train_test_split 0.35 / (25/35), random_state=42"}
    report["corpus"] = {name: corpus_duplicates(DATA / f"{name}.jsonl") for name in ("kyrgyznews", "wikipedia_ky", "kyrgyznews_20k")}
    report["benchmark_splits"] = {}
    for name in ("kyrgyznews", "wikipedia_ky"):
        for n in (1000, 5000):
            train, _, test = split(load(DATA / f"{name}.jsonl", n))
            report["benchmark_splits"][f"{name}@{n}"] = overlap(train, test)
    # Learning curve (learning_curve.py): same split on 20K, then budgets from a seed-42 shuffle of train.
    train, _, test = split(load(DATA / "kyrgyznews_20k.jsonl", 20000))
    shuffled = list(train)
    random.Random(42).shuffle(shuffled)
    report["learning_curve"] = {str(b): overlap(shuffled[:b], test) for b in (250, 1000, 2500, 5000, 10000, 13000)}

    out = ROOT / "reports/r1"
    out.mkdir(parents=True, exist_ok=True)
    (out / "leakage_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Train→test leakage audit", "", f"Split: `{report['split']}`; sentences ≥ {MIN_SENT} tokens, period-delimited on gold labels.", "",
             "## Corpus exact duplicates", "", "| File | Documents | Exact duplicate documents |", "|---|---|---|"]
    lines += [f"| {k} | {v['documents']} | {v['exact_duplicate_documents']} |" for k, v in report["corpus"].items()]
    header = "| Setting | Test docs identical to a train doc | Test sentences seen in train | Test tokens in seen sentences |"
    lines += ["", "## Benchmark splits", "", header, "|---|---|---|---|"]
    for k, v in report["benchmark_splits"].items():
        lines.append(f"| {k} | {v['test_documents_identical_to_train']}/{v['test_documents']} ({v['test_documents_identical_pct']}%) | "
                     f"{v['test_sentences_seen_in_train']}/{v['test_sentences']} ({v['test_sentences_seen_pct']}%) | {v['test_tokens_in_seen_sentences_pct']}% |")
    lines += ["", "## Learning curve (kyrgyznews_20k, fixed test split)", "", header.replace("Setting", "Train budget"), "|---|---|---|---|"]
    for k, v in report["learning_curve"].items():
        lines.append(f"| {k} | {v['test_documents_identical_to_train']}/{v['test_documents']} ({v['test_documents_identical_pct']}%) | "
                     f"{v['test_sentences_seen_in_train']}/{v['test_sentences']} ({v['test_sentences_seen_pct']}%) | {v['test_tokens_in_seen_sentences_pct']}% |")
    (out / "leakage_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
