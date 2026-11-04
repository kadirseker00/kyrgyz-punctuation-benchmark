"""Run a reproducible Common Voice Kyrgyz ASR-to-APR pilot."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
import json
from difflib import SequenceMatcher
from pathlib import Path
import re
import tarfile
import time

import av
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import torch
from transformers import (
    AutoModelForCTC,
    AutoModelForTokenClassification,
    AutoTokenizer,
    Wav2Vec2Processor,
)

from kg_punctuation.benchmark_v2 import project
from kg_punctuation.data.labeling import tokens_and_labels
from kg_punctuation.hf_generic_experiment import select_device


LABELS = ["S", "C", "P"]
WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", flags=re.UNICODE)
PUNCTUATION_RE = re.compile(r"[,.?!:;]")


def normalized_words(text: str) -> list[str]:
    text = text.lower().replace("ё", "е")
    return [match.group(0) for match in WORD_RE.finditer(text)]


def edit_distance(source: list[str], target: list[str]) -> int:
    previous = list(range(len(target) + 1))
    for index, source_word in enumerate(source, start=1):
        current = [index] + [0] * len(target)
        for target_index, target_word in enumerate(target, start=1):
            current[target_index] = min(
                previous[target_index] + 1,
                current[target_index - 1] + 1,
                previous[target_index - 1] + (source_word != target_word),
            )
        previous = current
    return previous[-1]


def decode_audio(path: Path, sample_rate: int = 16_000) -> np.ndarray:
    container = av.open(str(path))
    resampler = av.audio.resampler.AudioResampler(
        format="s16",
        layout="mono",
        rate=sample_rate,
    )
    chunks = []
    for frame in container.decode(audio=0):
        for resampled in resampler.resample(frame):
            values = resampled.to_ndarray()
            if values.ndim == 2:
                values = values[0]
            chunks.append(values.astype(np.float32) / 32768.0)
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


def select_rows(transcript_path: Path, limit: int) -> list[dict[str, str]]:
    rows = list(csv.DictReader(transcript_path.open(encoding="utf-8"), delimiter="\t"))
    selected = []
    for row in rows:
        if 4 <= len(normalized_words(row["sentence"])) <= 12 and PUNCTUATION_RE.search(
            row["sentence"]
        ):
            selected.append(row)
        if len(selected) == limit:
            break
    return selected


def extract_clips(
    archive_path: Path,
    clip_dir: Path,
    rows: list[dict[str, str]],
) -> None:
    clip_dir.mkdir(parents=True, exist_ok=True)
    wanted = {row["path"] for row in rows}
    missing = {name for name in wanted if not (clip_dir / name).exists()}
    if not missing:
        return
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            if member.name not in missing:
                continue
            source = archive.extractfile(member)
            if source is None:
                continue
            with source, (clip_dir / Path(member.name).name).open("wb") as target:
                target.write(source.read())


def run_asr(
    rows: list[dict[str, str]],
    clip_dir: Path,
    model_name: str,
    device: torch.device,
    batch_size: int,
) -> list[str]:
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = AutoModelForCTC.from_pretrained(model_name).to(device).eval()
    ASR_CONFIG_HOLDER["config"] = model.config
    predictions = []
    batch = []
    for row in rows:
        batch.append(decode_audio(clip_dir / row["path"]))
        if len(batch) == batch_size:
            predictions.extend(_decode_batch(processor, model, batch, device))
            batch = []
    if batch:
        predictions.extend(_decode_batch(processor, model, batch, device))
    return predictions


def _decode_batch(processor, model, arrays, device: torch.device) -> list[str]:
    inputs = processor(
        arrays,
        sampling_rate=16_000,
        return_tensors="pt",
        padding=True,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    return processor.batch_decode(torch.argmax(logits, dim=-1))


def predict_punctuation(
    tokenizer,
    model,
    words: list[str],
    device: torch.device,
) -> list[str]:
    if not words:
        return []
    encoding = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )
    model_inputs = {key: value.to(device) for key, value in encoding.items()}
    with torch.no_grad():
        ids = model(**model_inputs).logits.argmax(dim=-1)[0].cpu().tolist()
    labels = [None] * len(words)
    id_to_label = model.config.id2label
    for index, word_id in enumerate(encoding.word_ids(batch_index=0)):
        if word_id is None or word_id >= len(labels) or labels[word_id] is not None:
            continue
        labels[word_id] = id_to_label.get(ids[index], id_to_label.get(str(ids[index])))
    return [label or "S" for label in labels]


def _macro_f1(gold: list[str], prediction: list[str]) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(gold, prediction, labels=LABELS, average="macro", zero_division=0))


def paired_bootstrap(clips: list[dict], samples: int, seed: int) -> dict:
    """Clip-level paired bootstrap of macro-F1(oracle) - macro-F1(ASR) on the aligned positions."""
    rng = np.random.default_rng(seed)
    per_clip = []
    for clip in clips:
        gold = [clip["gold_labels"][g] for g, _ in clip["aligned_pairs"]]
        oracle = [clip["oracle_predictions"][g] for g, _ in clip["aligned_pairs"]]
        asr = [clip["asr_predictions"][a] for _, a in clip["aligned_pairs"]]
        per_clip.append((gold, oracle, asr))

    def delta(indices) -> float:
        gold = [l for i in indices for l in per_clip[i][0]]
        oracle = [l for i in indices for l in per_clip[i][1]]
        asr = [l for i in indices for l in per_clip[i][2]]
        return _macro_f1(gold, oracle) - _macro_f1(gold, asr)

    point = delta(range(len(per_clip)))
    draws = sorted(delta(rng.integers(0, len(per_clip), len(per_clip))) for _ in range(samples))
    return {"point": round(point, 6), "ci95": [round(draws[int(0.025 * samples)], 6), round(draws[int(0.975 * samples) - 1], 6)],
            "samples": samples, "seed": seed, "unit": "clip"}


def provenance(args, asr_model_name: str, apr_model) -> dict:
    import hashlib

    digest = hashlib.sha256()
    with Path(args.audio_tar).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return {"audio_tar_sha256": digest.hexdigest(),
            "transcript_tsv_sha256": hashlib.sha256(Path(args.transcript_tsv).read_bytes()).hexdigest(),
            "asr_model": asr_model_name,
            "asr_model_revision": getattr(ASR_CONFIG_HOLDER.get("config"), "_commit_hash", None),
            "apr_model_path": str(args.apr_model)}


ASR_CONFIG_HOLDER: dict = {}


def score(gold: list[str], prediction: list[str]) -> dict:
    report = classification_report(
        gold,
        prediction,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": round(float(accuracy_score(gold, prediction)), 6),
        "macro_precision": round(float(report["macro avg"]["precision"]), 6),
        "macro_recall": round(float(report["macro avg"]["recall"]), 6),
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 6),
        "weighted_f1": round(float(report["weighted avg"]["f1-score"]), 6),
        "per_class": {
            label: {
                "precision": round(float(report[label]["precision"]), 6),
                "recall": round(float(report[label]["recall"]), 6),
                "f1": round(float(report[label]["f1-score"]), 6),
                "support": int(report[label]["support"]),
            }
            for label in LABELS
        },
    }


def write_markdown(path: Path, result: dict) -> None:
    lines = [
        "# Common Voice Kyrgyz ASR-Domain APR Pilot",
        "",
        f"- Clips: {result['clips']}",
        f"- Reference tokens: {result['reference_tokens']}",
        f"- ASR model: `{result['asr_model']}`",
        f"- APR model: `{result['apr_model']}`",
        f"- ASR word error rate: {100 * result['asr_word_error_rate']:.2f}%",
        f"- Matched token coverage: {100 * result['matched_token_coverage']:.2f}%",
        "",
        "| Condition | Accuracy | Macro-F1 | Weighted-F1 | Comma F1 | Period F1 | Tokens |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in (
        ("oracle_text_metrics", "Oracle reference words"),
        ("asr_aligned_metrics", "ASR output, exact-word aligned"),
        ("oracle_on_aligned_positions_metrics", "Oracle, same aligned positions (paired)"),
    ):
        metrics = result[key]
        lines.append(
            f"| {label} | {100 * metrics['accuracy']:.2f} | "
            f"{100 * metrics['macro_f1']:.2f} | "
            f"{100 * metrics['weighted_f1']:.2f} | "
            f"{100 * metrics['per_class']['C']['f1']:.2f} | "
            f"{100 * metrics['per_class']['P']['f1']:.2f} | "
            f"{metrics['evaluated_tokens']} |"
        )
    lines.extend(["", "## Sample ASR outputs", ""])
    for sample in result["samples"]:
        lines.extend(
            [
                f"- `{sample['path']}` "
                f"({sample['word_errors']}/{sample['word_count']} edits, "
                f"{sample['matched_words']} matched words)",
                f"  - Reference: {sample['reference']}",
                f"  - ASR: {sample['asr']}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-tar", type=Path, required=True)
    parser.add_argument("--transcript-tsv", type=Path, required=True)
    parser.add_argument("--clip-dir", type=Path, required=True)
    parser.add_argument("--asr-model", default="kyrgyz-ai/Wav2vec-Kyrgyz")
    parser.add_argument("--apr-model", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-clips", type=Path, help="JSONL archive of every clip (reference, ASR, alignment, predictions)")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=123)
    args = parser.parse_args()

    start = time.perf_counter()
    rows = select_rows(args.transcript_tsv, args.limit)
    extract_clips(args.audio_tar, args.clip_dir, rows)
    device = select_device() if args.device == "auto" else torch.device(args.device)
    asr_outputs = run_asr(
        rows,
        args.clip_dir,
        args.asr_model,
        device,
        args.batch_size,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.apr_model, use_fast=True)
    apr_model = AutoModelForTokenClassification.from_pretrained(args.apr_model).to(device).eval()

    oracle_gold, oracle_prediction = [], []
    aligned_gold, aligned_prediction = [], []
    paired_oracle_prediction = []  # oracle predictions at exactly the ASR-aligned reference positions
    clip_records = []
    reference_tokens = 0
    word_errors = 0
    matched_tokens = 0
    samples = []

    for row, asr_text in zip(rows, asr_outputs):
        gold_words, gold_labels = tokens_and_labels(
            row["sentence"],
            lowercase=True,
            label_set="extended",
        )
        gold_labels = project(gold_labels, "core")  # v2 core view: ? and ! end a sentence (-> P)
        gold_words = [word.replace("ё", "е") for word in gold_words]
        asr_words = normalized_words(asr_text)

        oracle_labels = predict_punctuation(tokenizer, apr_model, gold_words, device)
        oracle_prediction.extend(oracle_labels)
        oracle_gold.extend(gold_labels)

        word_errors += edit_distance(gold_words, asr_words)
        reference_tokens += len(gold_words)
        asr_labels = predict_punctuation(tokenizer, apr_model, asr_words, device)
        matched = 0
        pairs = []
        for tag, gold_start, gold_end, asr_start, asr_end in SequenceMatcher(
            None,
            gold_words,
            asr_words,
            autojunk=False,
        ).get_opcodes():
            if tag != "equal":
                continue
            for gold_index, asr_index in zip(
                range(gold_start, gold_end),
                range(asr_start, asr_end),
            ):
                if asr_index < len(asr_labels):
                    aligned_gold.append(gold_labels[gold_index])
                    aligned_prediction.append(asr_labels[asr_index])
                    paired_oracle_prediction.append(oracle_labels[gold_index])
                    pairs.append([gold_index, asr_index])
                    matched += 1
        matched_tokens += matched
        clip_records.append({
            "path": row["path"], "reference": row["sentence"], "asr": asr_text,
            "gold_words": gold_words, "gold_labels": gold_labels, "oracle_predictions": oracle_labels,
            "asr_words": asr_words, "asr_predictions": asr_labels, "aligned_pairs": pairs,
            "word_errors": edit_distance(gold_words, asr_words),
        })
        if len(samples) < 10:
            samples.append(
                {
                    "path": row["path"],
                    "reference": row["sentence"],
                    "asr": asr_text,
                    "word_errors": edit_distance(gold_words, asr_words),
                    "word_count": len(gold_words),
                    "matched_words": matched,
                }
            )

    result = {
        "task": "Common Voice Kyrgyz ASR-domain punctuation pilot",
        "audio_source": str(args.audio_tar),
        "transcript_source": str(args.transcript_tsv),
        "asr_model": args.asr_model,
        "apr_model": str(args.apr_model),
        "clips": len(rows),
        "reference_tokens": reference_tokens,
        "matched_tokens": matched_tokens,
        "asr_word_error_rate": round(word_errors / reference_tokens, 6),
        "matched_token_coverage": round(matched_tokens / reference_tokens, 6),
        "oracle_text_metrics": score(oracle_gold, oracle_prediction),
        "asr_aligned_metrics": score(aligned_gold, aligned_prediction),
        "oracle_on_aligned_positions_metrics": score(aligned_gold, paired_oracle_prediction),
        "paired_difference_bootstrap": paired_bootstrap(clip_records, args.bootstrap_samples, args.bootstrap_seed),
        "label_view": "v2 core (S/C/P; ? and ! -> P)",
        "provenance": provenance(args, asr_model_name=args.asr_model, apr_model=apr_model),
        "label_counts_oracle": dict(Counter(oracle_gold)),
        "label_counts_asr_aligned": dict(Counter(aligned_gold)),
        "device": str(device),
        "runtime_seconds": round(time.perf_counter() - start, 3),
        "samples": samples,
    }
    result["oracle_text_metrics"]["evaluated_tokens"] = len(oracle_gold)
    result["asr_aligned_metrics"]["evaluated_tokens"] = len(aligned_gold)
    result["oracle_on_aligned_positions_metrics"]["evaluated_tokens"] = len(aligned_gold)
    if args.output_clips:
        args.output_clips.parent.mkdir(parents=True, exist_ok=True)
        with args.output_clips.open("w", encoding="utf-8") as handle:
            for record in clip_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown(args.output_md, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
