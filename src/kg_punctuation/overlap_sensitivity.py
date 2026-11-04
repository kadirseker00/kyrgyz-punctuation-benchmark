"""Re-score the canonical main panel on one shared sentence-overlap exclusion mask."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import io
import itertools
import json
from pathlib import Path

from kg_punctuation.benchmark_v2 import Benchmark, DEFAULT_MANIFEST, ROOT

LABELS = ("S", "C", "P")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sentences(doc):
    """Independent implementation of completed >=5-word sentence matching."""
    start = 0
    for end, label in enumerate(doc["labels"], 1):
        if label in ("P", "Q", "E"):
            if end - start >= 5:
                yield start, end, tuple(doc["tokens"][start:end])
            start = end


def core(label):
    if label not in ("S", "C", "P", "Q", "E"):
        raise ValueError(f"Unknown label: {label}")
    return "P" if label in ("Q", "E") else label


def build_mask(docs, union_sentences, own_sentences):
    rows = []
    for doc in docs:
        excluded, own = set(), set()
        for start, end, words in sentences(doc):
            if words in union_sentences:
                excluded.update(range(start, end))
            if words in own_sentences:
                own.update(range(start, end))
        for index, gold in enumerate(doc["labels"]):
            rows.append({"source": doc["source"], "document_id": doc["document_id"],
                         "group_id": doc["group_id"], "word_index": str(index),
                         "gold": core(gold), "keep": index not in excluded,
                         "own_source_overlap": index in own})
    return rows


def metrics(counts):
    total = sum(counts.values())
    if not total:
        raise ValueError("Cannot score an empty subset")
    per_class = {}
    for label in LABELS:
        tp = counts[label, label]
        gold = sum(n for (a, b), n in counts.items() if a == label)
        predicted = sum(n for (a, b), n in counts.items() if b == label)
        per_class[label] = {"precision": tp / predicted if predicted else 0.,
                            "recall": tp / gold if gold else 0.,
                            "f1": 2 * tp / (gold + predicted) if gold + predicted else 0.,
                            "support": gold, "predicted": predicted}
    return {"words": total, "accuracy": sum(counts[x, x] for x in LABELS) / total,
            **{f"macro_{key}": sum(per_class[x][key] for x in LABELS) / 3
               for key in ("precision", "recall", "f1")}, "per_class": per_class}


def rescore(path, mask):
    all_counts, filtered_counts = Counter(), Counter()
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row, expected in itertools.zip_longest(csv.DictReader(handle), mask):
            if row is None or expected is None:
                raise ValueError(f"Prediction length differs from canonical mask: {path}")
            identity = {k: row[k] for k in ("source", "document_id", "group_id", "word_index")}
            if any(identity[k] != expected[k] for k in identity) or core(row["gold"]) != expected["gold"]:
                raise ValueError(f"Prediction identity/gold differs from canonical mask: {path}")
            pair = core(row["gold"]), core(row["prediction"])
            all_counts[pair] += 1
            if expected["keep"]:
                filtered_counts[pair] += 1
    return {"all": metrics(all_counts), "filtered": metrics(filtered_counts)}


def run(manifest=DEFAULT_MANIFEST, registry=ROOT / "reports/v2/main_panel.csv",
        output=ROOT / "reports/r3/overlap", tables=ROOT / "paper/R3/ACM_KG_puctuation/tables"):
    benchmark = Benchmark(manifest)
    output, tables = Path(output), Path(tables)
    output.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    pools = {source: {words for doc in benchmark.documents(source, "train", "extended", limit=3200)
                      for _, _, words in sentences(doc)} for source in benchmark.manifest["sources"]}
    union = set().union(*pools.values())
    masks, source_stats = {}, {}
    for source in pools:
        mask = build_mask(benchmark.documents(source, "test", "extended"), union, pools[source])
        masks[source] = mask
        path = output / f"mask_{source}.csv.gz"
        with path.open("wb") as binary:
            with gzip.GzipFile(filename="", fileobj=binary, mode="wb", mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(mask[0]))
                    writer.writeheader(); writer.writerows(mask)
        source_stats[source] = {"total_words": len(mask), "retained_words": sum(r["keep"] for r in mask),
                                "own_source_overlap_words": sum(r["own_source_overlap"] for r in mask),
                                "mask_path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                                "mask_sha256": digest(path)}
    with Path(registry).open(newline="", encoding="utf-8") as handle:
        registered = list(csv.DictReader(handle))
    results = []
    for row in registered:
        path = ROOT / "experiments/v2_main" / row["job"] / f"predictions_test_{row['test']}.csv"
        values = rescore(path, masks[row["test"]])
        for name in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
            if abs(values["all"][name] - float(row[name])) > 1e-9:
                raise ValueError(f"Registry mismatch: {path}: {name}")
        for label in LABELS:
            for name in ("precision", "recall", "f1", "support"):
                if abs(values["all"]["per_class"][label][name] - float(row[f"{label}_{name}"])) > 1e-9:
                    raise ValueError(f"Registry per-class mismatch: {path}: {label}/{name}")
        results.append({"job": row["job"], "model": row["model"], "train": row["train"],
                        "test": row["test"], "setting": row["setting"],
                        "prediction_path": str(path.relative_to(ROOT)), "prediction_sha256": digest(path),
                        **values})
    rankings = {}
    for source in pools:
        selected = [r for r in results if r["test"] == source and r["setting"] != "transfer"]
        orders = {subset: [r["job"] for r in sorted(selected, key=lambda r: -r[subset]["macro_f1"])]
                  for subset in ("all", "filtered")}
        rankings[source] = {**orders, "unchanged": orders["all"] == orders["filtered"]}
    report = {"manifest_sha256": digest(manifest), "registry_sha256": digest(registry),
              "analysis_code_sha256": digest(__file__), "sources": source_stats, "rankings": rankings,
              "rule": "Completed >=5-word sentences; token equality ignoring punctuation; union of both main 3200-document training subsets; one fixed mask per target for every model.",
              "limitations": "Post-hoc sensitivity analysis, not retraining or proof of no contamination. Short phrases, paraphrases and unfinished sentences are not excluded. The external checkpoint's training corpus is unknown. Rankings are point estimates.",
              "results": results}
    (output / "main_panel_sensitivity.json").write_text(json.dumps(report, indent=2) + "\n")
    flat = []
    for row in results:
        for subset in ("all", "filtered"):
            m = row[subset]
            flat.append({"job": row["job"], "test": row["test"], "setting": row["setting"],
                         "subset": subset, **{k: v for k, v in m.items() if k != "per_class"},
                         **{f"{label}_{k}": v for label, d in m["per_class"].items() for k, v in d.items()}})
    with (output / "main_panel_sensitivity.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    comparison = [r for r in results if r["test"] == "kyrgyznews" and r["setting"] != "transfer"]
    md = ["# Main-panel sentence-overlap sensitivity", "", report["rule"], "",
          "| Model | News full → fixed | Wikipedia full → fixed |", "|---|---:|---:|"]
    tex = [r"\begin{table}[t]", r"\centering\small", r"\caption{Main-panel sensitivity to residual sentence overlap. The same exclusion mask is used for all models on each target: 212,034 news and 432,943 Wikipedia words remain. Scores are macro-F1 (\%). The external model's training data are unknown.}",
           r"\label{tab:overlap-sensitivity}", r"\begin{tabular}{lrrrr}", r"\toprule",
           r"& \multicolumn{2}{c}{News} & \multicolumn{2}{c}{Wikipedia} \\",
           r"Model & Full & Fixed & Full & Fixed \\", r"\midrule"]
    for news in comparison:
        wiki = next(r for r in results if r["model"] == news["model"] and r["test"] == "wikipedia_ky" and r["setting"] != "transfer")
        numbers = [100 * r[s]["macro_f1"] for r in (news, wiki) for s in ("all", "filtered")]
        md.append(f"| {news['model']} | {numbers[0]:.2f} → {numbers[1]:.2f} | {numbers[2]:.2f} → {numbers[3]:.2f} |")
        tex.append(news["model"] + " & " + " & ".join(f"{x:.2f}" for x in numbers) + r" \\")
    tex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (tables / "overlap_sensitivity_r3.tex").write_text("\n".join(tex) + "\n")
    md.extend(["", "Main-panel rankings unchanged: " + str({s: r["unchanged"] for s, r in rankings.items()}),
               "", report["limitations"], "", "All 34 full-test rows and their per-class metrics match the archived registry."])
    (output / "main_panel_sensitivity.md").write_text("\n".join(md) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--registry", type=Path, default=ROOT / "reports/v2/main_panel.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/r3/overlap")
    parser.add_argument("--tables", type=Path, default=ROOT / "paper/R3/ACM_KG_puctuation/tables")
    args = parser.parse_args()
    result = run(args.manifest, args.registry, args.output, args.tables)
    print(json.dumps({"sources": result["sources"], "rankings": result["rankings"]}, indent=2))
