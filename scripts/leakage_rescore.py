"""Re-score archived XLM-R checkpoints separately on test tokens whose sentence also
occurs verbatim in training ("seen") and on all other test tokens ("unseen").

Uses the exact split and truncation of hf_token_experiment (max_length 128, no windowing).
Usage: PYTHONPATH=src .venv/bin/python scripts/leakage_rescore.py
Writes reports/r1/leakage_rescore.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from leakage_audit import load, sentences, split  # noqa: E402

from kg_punctuation.hf_token_experiment import AprTokenDataset, evaluate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "kyrgyznews": "experiments/kyrgyznews_xlm-roberta-base_20260706_161433_33e18870",
    "wikipedia_ky": "experiments/wikipedia_ky_xlm-roberta-base_20260706_150352_79bbb655",
}
LABELS = ["S", "C", "P"]
TOKENIZER_DIR = ROOT / RUNS["kyrgyznews"] / "models/hf_model"  # xlm-roberta-base tokenizer
LC_ROOT = ROOT / "experiments/r1_learning_curve/learning_curve_kyrgyznews_20k_s-c-p"
EXTENDED = {
    "extended_news_5k": ("kyrgyznews_extended", "kyrgyznews_extended", "experiments/r1_extended/r1_extended_news_xlmr_5k_20260928_155546_edbf6641"),
    "extended_wiki_5k": ("wikipedia_ky_extended", "wikipedia_ky_extended", "experiments/r1_extended/r1_extended_wiki_xlmr_5k_20260928_155546_6039de32"),
}


def truncated_flags(test: list[dict], train: list[dict], tokenizer) -> list[bool]:
    train_sents = {s for d in train for s in sentences(d)}
    flags = []
    for d in test:
        word_ids = tokenizer(d["tokens"], is_split_into_words=True, truncation=True, max_length=128).word_ids()
        doc_flags = seen_flags(d, train_sents)
        previous = None
        for wid in word_ids:
            if wid is not None and wid != previous:
                flags.append(doc_flags[wid])
            previous = wid
    return flags


def split_scores(pred_csv: Path, flags: list[bool], labels: list[str]) -> dict:
    import csv
    rows = list(csv.DictReader(pred_csv.open(encoding="utf-8")))
    assert len(rows) == len(flags), (pred_csv, len(rows), len(flags))
    yt = [r["gold"] for r in rows]
    yp = [r["prediction"] for r in rows]
    global LABELS
    saved, LABELS = LABELS, labels
    out = {"all": metrics(yt, yp),
           "seen": metrics([a for a, f in zip(yt, flags) if f], [b for b, f in zip(yp, flags) if f]),
           "unseen": metrics([a for a, f in zip(yt, flags) if not f], [b for b, f in zip(yp, flags) if not f])}
    LABELS = saved
    return out


def seen_flags(doc: dict, train_sents: set) -> list[bool]:
    flags, cur = [], []
    for token, label in zip(doc["tokens"], doc["labels"]):
        cur.append(token)
        if label in ("P", "Q", "E"):
            hit = len(cur) >= 5 and tuple(cur) in train_sents
            flags.extend([hit] * len(cur))
            cur = []
    hit = len(cur) >= 5 and tuple(cur) in train_sents
    flags.extend([hit] * len(cur))
    return flags


def metrics(y_true: list[str], y_pred: list[str]) -> dict:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    macro = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    return {"tokens": len(y_true), "macro_f1": round(float(macro[2]), 6),
            **{f"{lab}_f1": round(float(f[i]), 6) for i, lab in enumerate(LABELS)},
            **{f"{lab}_support": int(s[i]) for i, lab in enumerate(LABELS)}}


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    report = {}
    for name, run in RUNS.items():
        train, _, test = split(load(ROOT / f"data/processed/source_level/{name}.jsonl", 1000))
        train_sents = {s for d in train for s in sentences(d)}
        model_dir = ROOT / run / "models/hf_model"
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        model = AutoModelForTokenClassification.from_pretrained(model_dir).to(device)
        docs = [{"tokens": d["tokens"], "labels": d["labels"]} for d in test]
        result = evaluate(model, DataLoader(AprTokenDataset(docs, tokenizer, max_length=128), batch_size=8), device)
        flags = []
        for d in test:  # first sub-token of each word that survives truncation, in loader order
            word_ids = tokenizer(d["tokens"], is_split_into_words=True, truncation=True, max_length=128).word_ids()
            doc_flags = seen_flags(d, train_sents)
            previous = None
            for wid in word_ids:
                if wid is not None and wid != previous:
                    flags.append(doc_flags[wid])
                previous = wid
        assert len(flags) == len(result["y_true"]), (len(flags), len(result["y_true"]))
        yt, yp = result["y_true"], result["y_pred"]
        report[name] = {
            "run": run,
            "all": metrics(yt, yp),
            "seen": metrics([a for a, f in zip(yt, flags) if f], [b for b, f in zip(yp, flags) if f]),
            "unseen": metrics([a for a, f in zip(yt, flags) if not f], [b for b, f in zip(yp, flags) if not f]),
        }
        print(name, json.dumps({k: v for k, v in report[name].items() if k != "run"}))

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, use_fast=True)
    import random
    train, _, test = split(load(ROOT / "data/processed/source_level/kyrgyznews_20k.jsonl", 20000))
    shuffled = list(train)
    random.Random(42).shuffle(shuffled)
    report["learning_curve"] = {}
    for budget in (250, 1000, 2500, 5000, 10000, 13000):
        flags = truncated_flags(test, shuffled[:budget], tokenizer)
        report["learning_curve"][str(budget)] = split_scores(LC_ROOT / f"budget_{budget}/predictions.csv", flags, ["S", "C", "P"])
        r = report["learning_curve"][str(budget)]
        print("lc", budget, {k: (v["tokens"], v["macro_f1"], v["C_f1"]) for k, v in r.items()})

    five = ("S", "C", "P", "Q", "E")
    for key, (file, _, run) in EXTENDED.items():
        path = ROOT / f"data/processed/source_level/{file}.jsonl"
        if not path.exists():
            print("missing", path); continue
        train, _, test = split(load(path, 5000, labels=five))
        flags = truncated_flags(test, train, tokenizer)
        report[key] = split_scores(ROOT / run / "tables/predictions.csv", flags, list(five))
        print(key, {k: (v["tokens"], v["macro_f1"], v.get("Q_f1"), v.get("E_f1")) for k, v in report[key].items()})
    (ROOT / "reports/r1/leakage_rescore.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
