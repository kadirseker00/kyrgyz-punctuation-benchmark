"""Verify expanded ASR archives, add descriptive error strata and create tables."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from kg_punctuation.asr_evaluation import event_counts, event_metrics
from kg_punctuation.benchmark_v2 import Benchmark, DEFAULT_MANIFEST, ROOT
from kg_punctuation.overlap_sensitivity import sentences


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def edit_distance(a, b):
    # Independent rolling-row implementation; does not use the scorer's alignment.
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def main():
    benchmark = Benchmark(DEFAULT_MANIFEST)
    pool = {words for source in benchmark.manifest["sources"]
            for doc in benchmark.documents(source, "train", "extended", limit=3200)
            for _, _, words in sentences(doc)}
    result = {"reporter_sha256": sha(Path(__file__)), "manifest_sha256": sha(DEFAULT_MANIFEST), "datasets": {}}
    flat, bins_table = [], []
    for dataset in ("common_voice", "fleurs"):
        root = ROOT / "reports/r3/asr" / dataset
        metric_path = root / "metrics.json"; old = json.loads(metric_path.read_text())
        clips = [json.loads(line) for line in (root / "clips.jsonl").read_text().splitlines()]
        assert len(clips) == old["clips"] == len({r["id"] for r in clips})
        totals = {k: Counter() for k in ("oracle", "aligned", "lexically_anchored")}
        groups = defaultdict(lambda: {"clips": 0, "oracle": Counter(), "aligned": Counter()})
        overlap_words, overlap_clips = 0, 0
        for row in clips:
            ref, gold, hyp, pred = (row[k] for k in ("gold_words", "gold_labels", "asr_words", "asr_predictions"))
            counts = {"oracle": event_counts(ref, gold, ref, row["oracle_predictions"]),
                      "aligned": event_counts(ref, gold, hyp, pred),
                      "lexically_anchored": event_counts(ref, gold, hyp, pred, require_exact_anchor=True)}
            assert dict(counts["oracle"]) == row["oracle_counts"]
            assert dict(counts["aligned"]) == row["aligned_counts"]
            assert edit_distance(ref, hyp) == counts["aligned"]["word_errors"]
            for condition in totals: totals[condition].update(counts[condition])
            wer = counts["aligned"]["word_errors"] / len(ref)
            category = "WER=0" if wer == 0 else "0<WER<=25%" if wer <= .25 else "25%<WER<=50%" if wer <= .5 else "WER>50%"
            group = groups[category]; group["clips"] += 1
            for condition in ("oracle", "aligned"): group[condition].update(counts[condition])
            seen = sum(end - start for start, end, words in sentences({"tokens": ref, "labels": gold}) if words in pool)
            overlap_words += seen; overlap_clips += bool(seen)
        metrics = {k: event_metrics(v) for k, v in totals.items()}
        assert metrics == old["metrics"]
        durations = [r["duration_seconds"] for r in clips]; lengths = [len(r["gold_words"]) for r in clips]
        stats = {"audio_minutes": sum(durations) / 60, "seconds_median": float(np.median(durations)),
                 "seconds_max": max(durations), "words_median": float(np.median(lengths)), "words_max": max(lengths),
                 "clips_over_12_words": sum(n > 12 for n in lengths),
                 "exact_sentence_overlap_with_APR_main_train_words": overlap_words,
                 "clips_with_exact_sentence_overlap": overlap_clips,
                 "speech_type": "read speech"}
        source = dict(old["provenance"])
        if dataset == "fleurs":
            provenance_path = root / "dataset_provenance.json"
            if not provenance_path.exists():
                provenance_path.write_text((ROOT / "data/raw/fleurs_ky/provenance.json").read_text())
            source.update(json.loads(provenance_path.read_text()))
        else: source.update(dataset="Common Voice 21.0", split="Kyrgyz test", license="CC0-1.0")
        bins = {k: {"clips": v["clips"], **{c: event_metrics(v[c]) for c in ("oracle", "aligned")}}
                for k, v in groups.items()}
        result["datasets"][dataset] = {"clips": len(clips), "archive_sha256": sha(root / "clips.jsonl"),
                                       "metrics_sha256": sha(metric_path), "provenance": source,
                                       "statistics": stats, "metrics": metrics, "WER_bins": bins,
                                       "paired_bootstrap": old["paired_bootstrap"]}
        for condition, m in metrics.items():
            flat.append({"dataset": dataset, "condition": condition, "clips": len(clips),
                         **{k: v for k, v in m.items() if k != "per_class"},
                         **{f"{l}_{k}": v for l, values in m["per_class"].items() for k, v in values.items()}})
        for category, group in bins.items():
            bins_table.append({"dataset": dataset, "WER_bin": category, "clips": group["clips"],
                               "reference_words": group["aligned"]["reference_words"],
                               "oracle_CP_macro_f1": group["oracle"]["punctuation_macro_f1"],
                               "ASR_CP_macro_f1": group["aligned"]["punctuation_macro_f1"]})
    output = ROOT / "reports/r3/asr"
    (output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    for name, rows in (("comparison.csv", flat), ("error_strata.csv", bins_table)):
        with (output / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    md = ["# Expanded read-speech evaluation", "", "All source test rows with nonempty reference words are included. No length, punctuation, WER or outcome selection. C/P event macro-F1 excludes S and scores the complete minimum-edit alignment, including deletions (FN) and insertions (FP). Substituted words may retain aligned punctuation boundaries; the stricter lexical-anchor condition additionally requires exact word identity for TP. Neither condition is a native acceptability measure.", "",
          "| Dataset | Clips | Ref. words | WER | Oracle C/P F1 | ASR aligned C/P F1 | Exact-anchor C/P F1 |", "|---|---:|---:|---:|---:|---:|---:|"]
    tex = [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{4pt}",
           r"\caption{Expanded read-speech test (\%). C/P macro-F1 averages comma and period event F1, excludes S, and covers all reference words and hypothesis events. Aligned scoring retains boundaries on substitutions and counts deleted/inserted marks as FN/FP; exact-anchor scoring additionally requires a correct anchor word for TP. These scores differ from the three-label conditional pilot.}",
           r"\label{tab:asr-expanded}", r"\begin{tabular}{lrrrrrr}", r"\toprule",
           r"Dataset & Clips & Ref. words & WER & Oracle & ASR aligned & Exact anchor \\", r"\midrule"]
    notes = []
    for dataset, group in result["datasets"].items():
        name = "Common Voice" if dataset == "common_voice" else "FLEURS"
        m = group["metrics"]; aligned = m["aligned"]
        values = [100 * aligned["WER"]] + [100 * m[c]["punctuation_macro_f1"] for c in ("oracle", "aligned", "lexically_anchored")]
        cells = [f"{group['clips']:,}", f"{aligned['reference_words']:,}"] + [f"{v:.2f}" for v in values]
        md.append(f"| {name} | " + " | ".join(cells) + " |")
        tex.append(name + " & " + " & ".join(cells) + r" \\")
        notes += ["", f"{name} source statistics: `{json.dumps(group['statistics'])}`. Paired-bootstrap oracle minus ASR interval: `{group['paired_bootstrap']}`.", ""]
    tex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    table = ROOT / "paper/R3/ACM_KG_puctuation/tables/asr_expanded_r3.tex"
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text("\n".join(tex) + "\n")
    md.extend(notes)
    md += ["The intervals resample Common Voice speakers and FLEURS source-sentence IDs (speaker IDs unavailable). WER strata are descriptive: they also differ in content and speakers, so their differences are not a causal dose-response estimate. Both datasets are read speech, not spontaneous dialogue or continuous long-form audio. The ASR checkpoint's original Common Voice release overlap has not been audited. Exact-sentence matches with the APR training pools are counted above; absence of such matches would not rule out pretraining contamination or paraphrases. No listening review, multi-reference annotation, latency or subtitle usability measurement was performed.", "", "Decoding uses the pinned Wav2Vec checkpoint, batches of four, and crops padded CTC logits to each unpadded waveform's acoustic output length. The historical 128-clip pilot used a different padding/alignment policy; its WER and F1 are retained as historical results and not treated as a directly comparable expanded estimate."]
    (output / "README.md").write_text("\n".join(md) + "\n")
    print(json.dumps({k: v["statistics"] for k, v in result["datasets"].items()}, indent=2))


if __name__ == "__main__": main()
