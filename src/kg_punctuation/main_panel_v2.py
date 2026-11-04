"""v2 main panel (T14): every model family trained on the same canonical train documents and
scored on the same full test documents of both sources (within-domain and zero-shot cross-domain).

    PYTHONPATH=src .venv/bin/python -m kg_punctuation.main_panel_v2 run-all [--only JOB ...]

Jobs are skipped when their scores.json exists. Progress lines "PANEL <status> <job>" go to
stdout (and experiments/v2_main/panel.log) so a watcher can follow the queue.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np

from kg_punctuation import benchmark_v2 as bv

ROOT = bv.ROOT
OUT = ROOT / "experiments/v2_main"
SOURCES = ["kyrgyznews", "wikipedia_ky"]
LABELS = bv.view_labels("core")
SEED = 42
TRAIN_LIMIT = bv.MAIN_TRAIN_DOCUMENTS
ENCODERS = {
    "xlmr": {"model_id": "xlm-roberta-base", "tokenizer_kwargs": {}},
    "kyrgyzbert": {"model_id": "metinovadilet/KyrgyzBert", "tokenizer_kwargs": {}},
    "kyrgyzbert2": {"model_id": "metinovadilet/KyrgyzBert2", "tokenizer_kwargs": {"add_prefix_space": True}},
}
ENCODER_HP = {"epochs": 3, "learning_rate": 3e-5, "batch_size": 8, "max_length": 128}
CNN_CONFIGS = {  # the two v1 CNN configurations, now fixed across domains
    "cnn_small": {"sequence_length": 128, "embedding_dim": 192, "hidden_dim": 192, "num_layers": 2,
                  "kernel_size": 3, "dropout": 0.25, "epochs": 20, "learning_rate": 8e-4, "vocab_size": 50000, "batch_size": 64},
    "cnn_deep": {"sequence_length": 256, "embedding_dim": 192, "hidden_dim": 256, "num_layers": 4,
                 "kernel_size": 3, "dropout": 0.25, "epochs": 30, "learning_rate": 8e-4, "vocab_size": 50000, "batch_size": 64},
}
EXTERNAL = {"model_id": "Zarinaaa/kyrgyzbert-kyrgyz-punctuation",
            "label_map": {"O": "S", "COMMA": "C", "PERIOD": "P", "QUESTION": "Q"}}


def log(message: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {message}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "panel.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def seed_everything(seed: int = SEED) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def device():
    import torch

    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def class_weights(docs: list[dict[str, Any]]) -> list[float]:
    counts = Counter(label for doc in docs for label in doc["labels"])
    total = sum(counts.values())
    return [total / (len(LABELS) * max(1, counts[label])) for label in LABELS]


def finish(job_dir: Path, bench: bv.Benchmark, predict: Callable[[list[dict]], list[list[str]]],
           metadata: dict[str, Any]) -> None:
    """Predict and score the test split of both sources; write predictions, scores, metadata."""
    scores = {}
    for target in SOURCES:
        docs = bench.documents(target, "test", "core")
        path = job_dir / f"predictions_test_{target}.csv"
        bv.write_predictions(path, docs, predict(docs))
        scores[target] = bv.score(path, bench, target, "test", "core")
    metadata["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    (job_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (job_dir / "scores.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")


def base_metadata(job: str, family: str, train_sources: dict[str, int], bench: bv.Benchmark) -> dict[str, Any]:
    return {"job": job, "family": family, "seed": SEED, "train_documents": train_sources,
            "manifest_code_commit": bench.manifest["code_commit"],
            "manifest_sha256": {s: bench.manifest["sources"][s]["sha256"] for s in SOURCES},
            "label_view": "core", "started_at": time.strftime("%Y-%m-%d %H:%M:%S %z")}


# --------------------------------------------------------------------------- classical

def run_classical(job: str, name: str, source: str, bench: bv.Benchmark) -> None:
    from sklearn.feature_extraction import FeatureHasher
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.pipeline import Pipeline

    from kg_punctuation.experiment import token_features

    estimators = {
        "logreg_balanced": LogisticRegression(max_iter=800, class_weight="balanced", solver="saga", random_state=SEED),
        "sgd_log_balanced": SGDClassifier(loss="log_loss", alpha=1e-5, class_weight="balanced", max_iter=1000,
                                          tol=1e-3, random_state=SEED),
    }
    train = bench.documents(source, "train", "core", limit=TRAIN_LIMIT)
    X = [token_features(d["tokens"], i) for d in train for i in range(len(d["tokens"]))]
    y = [label for d in train for label in d["labels"]]
    pipeline = Pipeline([("vectorizer", FeatureHasher(n_features=2**18, input_type="dict")), ("classifier", estimators[name])])
    start = time.perf_counter()
    pipeline.fit(X, y)
    fit_seconds = time.perf_counter() - start

    def predict(docs):
        return [list(pipeline.predict([token_features(d["tokens"], i) for i in range(len(d["tokens"]))])) for d in docs]

    job_dir = OUT / job
    job_dir.mkdir(parents=True, exist_ok=True)
    meta = base_metadata(job, "classical", {source: len(train)}, bench)
    meta.update({"model": name, "train_words": len(y), "fit_seconds": round(fit_seconds, 1),
                 "features": "kg_punctuation.experiment.token_features, FeatureHasher 2**18",
                 "split_level": "document (canonical v2)"})
    finish(job_dir, bench, predict, meta)


# --------------------------------------------------------------------------- CNN

def run_cnn(job: str, config_name: str, source: str, bench: bv.Benchmark) -> None:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    from kg_punctuation.deep_experiment import CnnTagger, SequenceDataset, build_vocabulary

    cfg = CNN_CONFIGS[config_name]
    seed_everything()
    dev = device()
    label_id = {label: i for i, label in enumerate(LABELS)}
    train = bench.documents(source, "train", "core", limit=TRAIN_LIMIT)
    valid = bench.documents(source, "validation", "core")
    vocab = build_vocabulary(train, vocab_size=cfg["vocab_size"])

    def index(docs):
        return [{"token_ids": [vocab.get(t, 1) for t in d["tokens"]], "label_ids": [label_id[l] for l in d["labels"]]} for d in docs]

    loader = DataLoader(SequenceDataset(index(train), cfg["sequence_length"]), batch_size=cfg["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(SEED))
    model = CnnTagger(len(vocab), cfg["sequence_length"], cfg["embedding_dim"], cfg["hidden_dim"], len(LABELS),
                      cfg["num_layers"], cfg["kernel_size"], cfg["dropout"]).to(dev)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights(train)).to(dev), ignore_index=-100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])

    def predict(docs):
        model.eval()
        out = []
        with torch.no_grad():
            for d in docs:
                ids = [vocab.get(t, 1) for t in d["tokens"]]
                labels = []
                for s in range(0, len(ids), cfg["sequence_length"]):
                    chunk = torch.tensor([ids[s:s + cfg["sequence_length"]]], device=dev)
                    labels.extend(LABELS[i] for i in model(chunk).argmax(-1)[0].cpu().tolist())
                out.append(labels)
        return out

    history, best, best_state = [], -1.0, None
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total = 0.0
        for batch in loader:
            optimizer.zero_grad()
            logits = model(batch["input_ids"].to(dev))
            labels = batch["labels"].to(dev)
            labels = labels.masked_fill(~batch["mask"].to(dev), -100)
            loss = criterion(logits.view(-1, len(LABELS)), labels.view(-1))
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu())
        macro = _macro(valid, predict(valid))
        history.append({"epoch": epoch, "train_loss": round(total / max(1, len(loader)), 6), "valid_macro_f1": round(macro, 6)})
        if macro > best:
            best, best_state = macro, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    job_dir = OUT / job
    job_dir.mkdir(parents=True, exist_ok=True)
    meta = base_metadata(job, "cnn", {source: len(train)}, bench)
    meta.update({"config": config_name, "hyperparameters": cfg, "vocab_size_actual": len(vocab), "history": history,
                 "best_valid_macro_f1": best, "note": "CnnTagger includes position embeddings in both configurations"})
    finish(job_dir, bench, predict, meta)


def _macro(docs, preds) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score([l for d in docs for l in d["labels"]], [l for p in preds for l in p], labels=LABELS,
                          average="macro", zero_division=0))


# --------------------------------------------------------------------------- encoders

def _collate(pad_id: int):
    import torch

    def collate(features):
        width = max(len(f["input_ids"]) for f in features)
        return {
            "input_ids": torch.tensor([f["input_ids"] + [pad_id] * (width - len(f["input_ids"])) for f in features]),
            "attention_mask": torch.tensor([f["attention_mask"] + [0] * (width - len(f["attention_mask"])) for f in features]),
            "labels": torch.tensor([f["labels"] + [-100] * (width - len(f["labels"])) for f in features]),
        }
    return collate


def run_encoder(job: str, encoder: str, train_spec: dict[str, int], bench: bv.Benchmark, save_model: bool,
                seed: int = SEED) -> None:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

    spec = ENCODERS[encoder]
    hp = ENCODER_HP
    seed_everything(seed)
    dev = device()
    label2id = {label: i for i, label in enumerate(LABELS)}
    id2label = {i: label for label, i in label2id.items()}
    train = [d for source, n in train_spec.items() for d in bench.documents(source, "train", "core", limit=n)]
    valid = [d for source in train_spec for d in bench.documents(source, "validation", "core")]
    tokenizer = AutoTokenizer.from_pretrained(spec["model_id"], use_fast=True, **spec["tokenizer_kwargs"])
    features = bv.encode_training_windows(tokenizer, train, label2id, max_length=hp["max_length"])
    loader = DataLoader(features, batch_size=hp["batch_size"], shuffle=True, collate_fn=_collate(tokenizer.pad_token_id),
                        generator=torch.Generator().manual_seed(seed))
    config = AutoConfig.from_pretrained(spec["model_id"], num_labels=len(LABELS), id2label=id2label, label2id=label2id)
    model = AutoModelForTokenClassification.from_pretrained(spec["model_id"], config=config, ignore_mismatched_sizes=True).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp["learning_rate"])
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights(train)).to(dev), ignore_index=-100)

    def predict(docs):
        return bv.predict_words(model, tokenizer, docs, id2label, subtoken="first", max_length=hp["max_length"],
                                batch_size=32, device=dev)

    history, best, best_state = [], -1.0, None
    for epoch in range(1, hp["epochs"] + 1):
        model.train()
        start, total = time.perf_counter(), 0.0
        for step, batch in enumerate(loader, 1):
            batch = {k: v.to(dev) for k, v in batch.items()}
            labels = batch.pop("labels")
            optimizer.zero_grad()
            loss = loss_fn(model(**batch).logits.view(-1, len(LABELS)), labels.view(-1))
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu())
            if step % 500 == 0:
                log(f"PANEL PROGRESS {job} epoch {epoch} step {step}/{len(loader)}")
        macro = _macro(valid, predict(valid))
        history.append({"epoch": epoch, "train_loss": round(total / max(1, len(loader)), 6),
                        "valid_macro_f1": round(macro, 6), "seconds": round(time.perf_counter() - start, 1)})
        log(f"PANEL EPOCH {job} {history[-1]}")
        if macro > best:
            best = macro
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    job_dir = OUT / job
    job_dir.mkdir(parents=True, exist_ok=True)
    if save_model:
        model.save_pretrained(job_dir / "hf_model")
        tokenizer.save_pretrained(job_dir / "hf_model")
    meta = base_metadata(job, "encoder", train_spec, bench)
    meta["seed"] = seed
    meta.update({"encoder": encoder, "model_id": spec["model_id"], "hyperparameters": hp, "train_windows": len(features),
                 "class_weights": dict(zip(LABELS, class_weights(train))), "history": history,
                 "best_valid_macro_f1": best, "subtoken": "first", "windowing": "benchmark_v2 overlapping word windows"})
    finish(job_dir, bench, predict, meta)


def run_external(job: str, subtoken: str, bench: bv.Benchmark) -> None:
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    dev = device()
    tokenizer = AutoTokenizer.from_pretrained(EXTERNAL["model_id"], use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(EXTERNAL["model_id"]).to(dev)
    id2label = {int(k): v for k, v in model.config.id2label.items()}

    def predict(docs):
        return bv.predict_words(model, tokenizer, docs, id2label, subtoken=subtoken, max_length=128, batch_size=32,
                                label_map=EXTERNAL["label_map"], device=dev)

    job_dir = OUT / job
    job_dir.mkdir(parents=True, exist_ok=True)
    meta = base_metadata(job, "external", {}, bench)
    meta.update({"model_id": EXTERNAL["model_id"], "subtoken": subtoken, "label_map": EXTERNAL["label_map"],
                 "id2label": id2label, "note": "zero-shot; QUESTION->Q projected to P in the core view"})
    finish(job_dir, bench, predict, meta)


# --------------------------------------------------------------------------- queue

def jobs() -> dict[str, Callable[[bv.Benchmark], None]]:
    short = {"kyrgyznews": "news", "wikipedia_ky": "wiki"}
    half = TRAIN_LIMIT // 2
    plan: dict[str, Callable[[bv.Benchmark], None]] = {}
    plan["external_last"] = lambda b: run_external("external_last", "last", b)
    plan["external_first"] = lambda b: run_external("external_first", "first", b)
    for source in SOURCES:
        for enc in ENCODERS:
            name = f"{enc}_{short[source]}"
            plan[name] = (lambda b, n=name, e=enc, s=source: run_encoder(n, e, {s: TRAIN_LIMIT}, b, e == "xlmr"))
    for source in SOURCES:
        for seed in (43, 44):  # seed 42 is the main-panel row itself
            name = f"xlmr_{short[source]}_s{seed}"
            plan[name] = (lambda b, n=name, s=source, sd=seed: run_encoder(n, "xlmr", {s: TRAIN_LIMIT}, b, False, sd))
    plan["xlmr_mixed"] = lambda b: run_encoder("xlmr_mixed", "xlmr", {s: TRAIN_LIMIT for s in SOURCES}, b, True)
    plan["xlmr_mixed_matched"] = lambda b: run_encoder("xlmr_mixed_matched", "xlmr", {s: half for s in SOURCES}, b, True)
    for source in SOURCES:
        for cfg in CNN_CONFIGS:
            name = f"{cfg}_{short[source]}"
            plan[name] = (lambda b, n=name, c=cfg, s=source: run_cnn(n, c, s, b))
    for source in SOURCES:
        for model_name in ("sgd_log_balanced", "logreg_balanced"):
            name = f"{model_name}_{short[source]}"
            plan[name] = (lambda b, n=name, m=model_name, s=source: run_classical(n, m, s, b))
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run-all")
    run.add_argument("--only", nargs="*")
    run.add_argument("--smoke", action="store_true",
                     help="16 train docs, 1 epoch, output to experiments/v2_smoke (full test sets are still scored)")
    sub.add_parser("list")
    args = parser.parse_args()
    global OUT, TRAIN_LIMIT
    if getattr(args, "smoke", False):
        OUT = ROOT / "experiments/v2_smoke"
        TRAIN_LIMIT = 16
        ENCODER_HP["epochs"] = 1
        for cfg in CNN_CONFIGS.values():
            cfg["epochs"] = 1
    plan = jobs()
    if args.command == "list":
        print("\n".join(plan))
        return
    bench = bv.Benchmark()
    selected = args.only or list(plan)
    log(f"PANEL QUEUE {len(selected)} jobs: {' '.join(selected)}")
    for name in selected:
        if (OUT / name / "scores.json").exists():
            log(f"PANEL SKIP {name}")
            continue
        log(f"PANEL START {name}")
        start = time.perf_counter()
        try:
            plan[name](bench)
            scores = json.loads((OUT / name / "scores.json").read_text())
            summary = {t: round(100 * s["macro_f1"], 2) for t, s in scores.items()}
            log(f"PANEL DONE {name} {round(time.perf_counter() - start)}s macroF1={summary}")
        except Exception:  # noqa: BLE001 - keep the queue going, record the failure
            log(f"PANEL FAIL {name}\n{traceback.format_exc()}")
    log("PANEL QUEUE FINISHED")


if __name__ == "__main__":
    main()
