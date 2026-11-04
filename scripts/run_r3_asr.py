"""Expanded read-speech evaluation with full edit-alignment punctuation-event scores.

Common Voice uses every TSV row with nonempty reference words (no length/punctuation filter).
FLEURS uses its complete Kyrgyz test split and the punctuated raw_transcription field.
Neither dataset is described as spontaneous speech.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import hashlib
import json
from pathlib import Path
import csv
import tarfile
import time

import numpy as np
import torch
from transformers import AutoModelForCTC, AutoModelForTokenClassification, AutoTokenizer, Wav2Vec2Processor

from kg_punctuation.asr_evaluation import event_counts, event_metrics
from kg_punctuation.benchmark_v2 import ROOT, predict_words, project
from kg_punctuation.data.labeling import tokens_and_labels
from run_asr_pilot import decode_audio, normalized_words


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def prepare(dataset):
    if dataset == "common_voice":
        root = ROOT / "data/raw/common_voice/asr_pilot"
        with (root / "test.tsv").open() as f: source = list(csv.DictReader(f, delimiter="\t"))
        rows = [{"id": r["path"], "path": r["path"], "reference": r["sentence"],
                 "group_id": r["client_id"]} for r in source]
        archive, clip_dir = root / "ky_test_0.tar", root / "clips"
    else:
        root = ROOT / "data/raw/fleurs_ky"
        with (root / "test.tsv").open() as f: source = list(csv.reader(f, delimiter="\t"))
        rows = [{"id": r[1], "path": r[1], "reference": r[2], "group_id": r[0]} for r in source]
        archive, clip_dir = root / "test.tar.gz", root / "clips"
    rows = [r for r in rows if normalized_words(r["reference"])]
    if len({r["id"] for r in rows}) != len(rows): raise ValueError("Duplicate clip IDs")
    clip_dir.mkdir(parents=True, exist_ok=True)
    wanted = {r["path"] for r in rows if not (clip_dir / r["path"]).exists()}
    if wanted:
        with tarfile.open(archive) as tar:
            for member in tar:
                name = Path(member.name).name
                if name not in wanted or not member.isfile(): continue
                with tar.extractfile(member) as src, (clip_dir / name).open("wb") as dst:
                    dst.write(src.read())
    missing = [r["path"] for r in rows if not (clip_dir / r["path"]).is_file()]
    if missing: raise ValueError(f"Missing audio for {len(missing)} selected records")
    return rows, clip_dir, {"tsv_sha256": sha(root / "test.tsv"), "audio_archive_sha256": sha(archive),
                            "selected_clips": len(rows), "speech_type": "read speech",
                            "bootstrap_unit": "speaker" if dataset == "common_voice" else "source sentence ID (speaker IDs unavailable)"}


def decode(rows, clips, output, revision):
    config = {"model": "kyrgyz-ai/Wav2vec-Kyrgyz", "revision": revision, "batch_size": 4,
              "padding_policy": "Crop each CTC logit sequence to the acoustic output length of its unpadded waveform"}
    path = output / "decoding.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError("Existing hypotheses use a different decoding protocol")
    path.write_text(json.dumps(config, indent=2) + "\n")
    cache_path = output / "hypotheses.jsonl"
    cache = {r["id"]: r for r in map(json.loads, cache_path.read_text().splitlines())} if cache_path.exists() else {}
    for row in rows:
        if row["id"] in cache and cache[row["id"]]["reference"] != row["reference"]:
            raise ValueError("Cached reference mismatch")
    pending = [r for r in rows if r["id"] not in cache]
    if pending:
        processor = Wav2Vec2Processor.from_pretrained(config["model"], revision=revision)
        model = AutoModelForCTC.from_pretrained(config["model"], revision=revision).eval()
        with cache_path.open("a") as handle:
            for start in range(0, len(pending), 4):
                batch = pending[start:start + 4]
                waves = [decode_audio(clips / r["path"]) for r in batch]
                if any(not len(w) for w in waves): raise ValueError("Empty decoded audio")
                inputs = processor(waves, sampling_rate=16000, padding=True, return_tensors="pt")
                with torch.no_grad(): logits = model(**inputs).logits
                lengths = model._get_feat_extract_output_lengths(torch.tensor([len(w) for w in waves]))
                for row, wave, values, length in zip(batch, waves, logits, lengths):
                    text = processor.decode(values[:int(length)].argmax(dim=-1))
                    record = {**row, "asr": text, "duration_seconds": len(wave) / 16000}
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n"); cache[row["id"]] = record
                handle.flush()
                if start % 64 == 0: print(f"ASR {len(cache)}/{len(rows)}", flush=True)
        del model, processor; gc.collect()
    return [cache[r["id"]] for r in rows]


def bootstrap(records, samples=2000):
    groups = defaultdict(lambda: [Counter(), Counter()])
    for row in records:
        for i, condition in enumerate(("oracle_counts", "aligned_counts")):
            groups[row["group_id"]][i].update(row[condition])
    fields = [l + suffix for l in ("C", "P") for suffix in ("_tp", "_fp", "_fn")]
    values = np.array([[[counts[k] for k in fields] for counts in pair] for pair in groups.values()])
    rng = np.random.default_rng(123); deltas = []
    for _ in range(samples):
        summed = values[rng.integers(0, len(values), size=len(values))].sum(axis=0)
        scores = [event_metrics(Counter(dict(zip(fields, condition))))["punctuation_macro_f1"] for condition in summed]
        deltas.append(scores[0] - scores[1])
    return {"samples": samples, "seed": 123, "groups": len(values),
            "oracle_minus_asr_punctuation_macro_f1_95CI": np.quantile(deltas, [.025, .975]).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("common_voice", "fleurs"), required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(); torch.set_num_threads(args.threads)
    started = time.perf_counter()
    output = ROOT / "reports/r3/asr" / args.dataset; output.mkdir(parents=True, exist_ok=True)
    rows, clip_dir, provenance = prepare(args.dataset)
    old = json.loads((ROOT / "reports/r1/asr_pilot_v2.json").read_text())
    revision = old["provenance"]["asr_model_revision"]
    records = decode(rows, clip_dir, output, revision)
    model_path = ROOT / "experiments/v2_main/xlmr_mixed/hf_model"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_path).eval()
    gold_docs, asr_docs = [], []
    for row in records:
        words, labels = tokens_and_labels(row["reference"], lowercase=True, label_set="extended")
        words = [w.replace("ё", "е") for w in words]; labels = project(labels, "core")
        hypothesis = normalized_words(row["asr"])
        gold_docs.append({"tokens": words, "labels": labels})
        asr_docs.append({"tokens": hypothesis, "labels": ["S"] * len(hypothesis)})
    oracle = predict_words(model, tokenizer, gold_docs, model.config.id2label, batch_size=8, device=torch.device("cpu"))
    # Empty hypotheses have no punctuation events; keep them in full-reference scoring.
    nonempty = [i for i, doc in enumerate(asr_docs) if doc["tokens"]]
    predictions = predict_words(model, tokenizer, [asr_docs[i] for i in nonempty], model.config.id2label,
                                batch_size=8, device=torch.device("cpu"))
    predicted = dict(zip(nonempty, predictions))
    totals = {name: Counter() for name in ("oracle", "aligned", "lexically_anchored")}
    bins = defaultdict(Counter)
    with (output / "clips.jsonl").open("w") as handle:
        for i, (row, gold, asr, oracle_labels) in enumerate(zip(records, gold_docs, asr_docs, oracle)):
            args_tuple = gold["tokens"], gold["labels"], asr["tokens"], predicted.get(i, [])
            counts = {"oracle": event_counts(gold["tokens"], gold["labels"], gold["tokens"], oracle_labels),
                      "aligned": event_counts(*args_tuple),
                      "lexically_anchored": event_counts(*args_tuple, require_exact_anchor=True)}
            for key, value in counts.items(): totals[key].update(value)
            wer = counts["aligned"]["word_errors"] / len(gold["tokens"])
            bin_name = "WER=0" if wer == 0 else "0<WER<=25%" if wer <= .25 else "25%<WER<=50%" if wer <= .5 else "WER>50%"
            bins[bin_name].update(counts["aligned"])
            row.update(gold_words=gold["tokens"], gold_labels=gold["labels"], asr_words=asr["tokens"],
                       oracle_predictions=oracle_labels, asr_predictions=predicted.get(i, []),
                       oracle_counts=dict(counts["oracle"]), aligned_counts=dict(counts["aligned"]))
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    result = {"dataset": args.dataset, "clips": len(records), "provenance": provenance,
              "asr_revision": revision, "apr_checkpoint": str(model_path.relative_to(ROOT)),
              "evaluation_code_sha256": sha(ROOT / "src/kg_punctuation/asr_evaluation.py"),
              "runner_sha256": sha(__file__), "clips_sha256": sha(output / "clips.jsonl"),
              "metrics": {k: event_metrics(v) for k, v in totals.items()},
              "WER_bins": {k: event_metrics(v) for k, v in bins.items()}, "paired_bootstrap": bootstrap(records),
              "runtime_seconds": time.perf_counter() - started,
              "interpretation": "C/P punctuation-event F1 over a complete minimum-edit word alignment. Substituted words retain aligned boundaries in the aligned condition; deleted/inserted marks are FN/FP. The lexically_anchored condition additionally requires an exact anchor word for TP. S is excluded from punctuation F1. These are declared alignment-based and joint lexical/punctuation measures, not the original conditional three-label macro-F1, native-speaker acceptability or subtitle usability. Both evaluated sources are read speech."}
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("dataset", "clips", "metrics", "paired_bootstrap")}, indent=2), flush=True)


if __name__ == "__main__": main()
