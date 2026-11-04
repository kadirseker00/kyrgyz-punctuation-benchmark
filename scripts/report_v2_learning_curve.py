"""Independently verify canonical curve predictions and sentence-overlap masks."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kg_punctuation.benchmark_v2 import Benchmark, DEFAULT_MANIFEST, ROOT


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sentences(doc):
    start = 0
    for end, label in enumerate(doc["labels"], 1):
        if label in {"P", "Q", "E"}:
            if end - start >= 5:
                yield start, end, tuple(doc["tokens"][start:end])
            start = end


def metrics(counts):
    classes = {}
    for label in ("S", "C", "P"):
        tp = counts[label, label]
        gold = sum(n for (g, p), n in counts.items() if g == label)
        pred = sum(n for (g, p), n in counts.items() if p == label)
        classes[label] = {"precision": tp / pred if pred else 0,
                          "recall": tp / gold if gold else 0,
                          "f1": 2 * tp / (gold + pred) if gold + pred else 0,
                          "support": gold}
    total = sum(counts.values())
    return {"tokens": total, "macro_f1": sum(r["f1"] for r in classes.values()) / 3,
            "accuracy": sum(counts[k, k] for k in classes) / total,
            "weighted_f1": sum(r["f1"] * r["support"] for r in classes.values()) / total,
            "per_class": classes}


def main():
    state = json.loads((ROOT / "experiments/v2_audits/queue_status.json").read_text())
    if state["jobs"]["learning_curve"]["status"] != "complete":
        raise ValueError("Learning curve is not complete.")
    paths = list((ROOT / "experiments/v2_audits/learning_curve").glob("*/metadata.json"))
    if len(paths) != 1:
        raise ValueError("Select the intended learning curve explicitly before reporting multiple runs.")
    summary_path = paths[0]
    summary = json.loads(summary_path.read_text())
    manifest_hash = digest(DEFAULT_MANIFEST)
    assert summary["manifest_sha256"] == manifest_hash == state["manifest_sha256"]
    benchmark = Benchmark()
    source = "kyrgyznews"
    test = benchmark.documents(source, "test", "extended")
    train = benchmark.documents(source, "train", "extended")
    test_sentences = {value for doc in test for _, _, value in sentences(doc)}
    first_seen = {}
    for rank, doc in enumerate(train, 1):
        for _, _, value in sentences(doc):
            if value in test_sentences:
                first_seen.setdefault(value, rank)
    expected, ranks = [], []
    for doc in test:
        doc_ranks = [0] * len(doc["tokens"])
        for start, end, value in sentences(doc):
            doc_ranks[start:end] = [first_seen.get(value, 0)] * (end - start)
        expected.extend((source, doc["document_id"], str(i), doc["group_id"],
                         "P" if label in ("Q", "E") else label)
                        for i, label in enumerate(doc["labels"]))
        ranks.extend(doc_ranks)
    mask_path = summary_path.parent / "sentence_overlap_mask.csv"
    with mask_path.open() as handle:
        saved = list(csv.DictReader(handle))
    assert [(r["source"], r["document_id"], r["word_index"]) for r in saved] == [r[:3] for r in expected]
    assert [int(r["first_train_document_rank"]) for r in saved] == ranks
    assert len(expected) == 220839 and ranks.count(0) == 194050
    rows = []
    audit = {r["budget"]: r for r in summary["sentence_overlap_audit"]["results"]}
    for result in summary["results"]:
        budget = result["budget"]
        run = Path(result["run_dir"])
        meta_path = run / "metadata.json"
        meta = json.loads(meta_path.read_text())
        assert meta["benchmark"]["manifest_sha256"] == manifest_hash
        assert meta["benchmark"]["train_document_ids"] == benchmark.split_ids(source, "train", budget)
        for split in ("validation", "test"):
            assert meta["benchmark"][f"{split}_document_ids"] == benchmark.split_ids(source, split)
        assert meta["seed"] == summary["seed"] == 42
        for key, value in summary["hyperparameters"].items():
            assert meta["hyperparameters"][key] == value
        assert meta["label_set"] == ["S", "C", "P"]
        assert result["train_tokens"] == sum(len(doc["tokens"]) for doc in train[:budget])
        path = run / "tables/predictions.csv"
        with path.open() as handle:
            predicted = list(csv.DictReader(handle))
        actual = [(r["source"], r["document_id"], r["word_index"], r["group_id"], r["gold"]) for r in predicted]
        assert actual == expected
        assert all(r["prediction"] in ("S", "C", "P") for r in predicted)
        masks = {"all": [True] * len(ranks),
                 "fixed_outside_full_pool_overlap": [rank == 0 for rank in ranks],
                 "seen_at_budget": [0 < rank <= budget for rank in ranks],
                 "outside_seen_at_budget": [not 0 < rank <= budget for rank in ranks]}
        computed = {}
        for name, mask in masks.items():
            values = metrics(Counter((r["gold"], r["prediction"]) for r, keep in zip(predicted, mask) if keep))
            archived = result if name == "all" else audit[budget]["subsets"][name]["metrics"]
            for metric in ("macro_f1", "accuracy", "weighted_f1"):
                assert abs(values[metric] - archived[metric]) < 1e-6, (budget, name, metric)
            for label in ("S", "C", "P"):
                for metric in ("precision", "recall", "f1", "support"):
                    assert abs(values["per_class"][label][metric] - archived["per_class"][label][metric]) < 1e-6
            computed[name] = values
        rows.append({"budget": budget, "train_tokens": result["train_tokens"],
                     "run_dir": str(run.relative_to(ROOT)), "metadata_sha256": digest(meta_path),
                     "predictions_sha256": digest(path), "best_epoch": meta["best_epoch"],
                     "runtime_seconds_total": meta["runtime_seconds_total"],
                     "runtime_seconds_test": meta["runtime_seconds_test"], "subsets": computed})
    assert [r["budget"] for r in rows] == [250, 1000, 2500, 3200, 5000, 10000, 13000]
    output = ROOT / "reports/v2"
    report = {"manifest_sha256": manifest_hash, "summary_sha256": digest(summary_path),
              "mask_sha256": digest(mask_path), "source_summary": str(summary_path.relative_to(ROOT)),
              "verification": "Independent sentence matching and confusion counts; exact test IDs/gold/groups; nested train IDs.",
              "overlap_rule": summary["sentence_overlap_audit"]["rule"],
              "overlap_limitations": summary["sentence_overlap_audit"]["limitation"],
              "test_documents": len(test), "test_tokens": len(expected), "train_pool_documents": len(train),
              "fixed_subset_tokens": ranks.count(0), "runs": rows}
    (output / "learning_curve_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    flat = []
    for row in rows:
        a = row["subsets"]["all"]
        fixed = row["subsets"]["fixed_outside_full_pool_overlap"]
        flat.append({"budget": row["budget"], "train_tokens": row["train_tokens"],
                     "macro_f1": a["macro_f1"], "comma_f1": a["per_class"]["C"]["f1"],
                     "weighted_f1": a["weighted_f1"], "fixed_subset_macro_f1": fixed["macro_f1"],
                     "fixed_subset_comma_f1": fixed["per_class"]["C"]["f1"],
                     "seen_test_tokens": row["subsets"]["seen_at_budget"]["tokens"],
                     "runtime_seconds": row["runtime_seconds_total"]})
    with (output / "learning_curve_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    table = [r"\begin{table}[t]", r"\centering",
             r"\caption{News learning curve on the same 1{,}250 test documents (220{,}839 words). Fixed-subset scores exclude the same 26{,}789 words in sentences found anywhere in the full training pool, leaving 194{,}050 words. All scores are percentages; each budget uses one training seed.}",
             r"\label{tab:learning-curve}", r"\small", r"\setlength{\tabcolsep}{5pt}",
             r"\begin{tabular}{rrrrr}", r"\toprule",
             r"Training documents & Macro-F1 & Comma F1 & Weighted-F1 & Fixed-subset macro-F1 \\", r"\midrule"]
    md = ["# Verified canonical v2 News learning curve", "",
          "| Train documents | Macro-F1 (%) | Comma F1 (%) | Fixed-subset macro-F1 (%) | Runtime (min) |",
          "|---:|---:|---:|---:|---:|"]
    for r in flat:
        table.append(" & ".join([f"{r['budget']:,}".replace(",", "{,}")]
                                + [f"{100*r[k]:.2f}" for k in ("macro_f1", "comma_f1", "weighted_f1", "fixed_subset_macro_f1")]) + r" \\")
        md.append(f"| {r['budget']} | {100*r['macro_f1']:.2f} | {100*r['comma_f1']:.2f} | {100*r['fixed_subset_macro_f1']:.2f} | {r['runtime_seconds']/60:.2f} |")
    table.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    for directory in (output / "tables", ROOT / "paper/R2/ACM_KG_puctuation/tables"):
        directory.mkdir(exist_ok=True, parents=True)
        (directory / "learning_curve_v2.tex").write_text("\n".join(table) + "\n")
    (output / "learning_curve_audit.md").write_text("\n".join(md) + "\n\n" + report["overlap_limitations"] + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), layout="constrained")
    for axis, metric, limits, title in zip(axes, ("macro_f1", "comma_f1"), ((80, 90), (60, 80)), ("Macro-F1", "Comma F1")):
        for key, label, color, marker in ((metric, "Full test", "#007D79", "o"),
                                        ("fixed_subset_" + metric, "Fixed subset", "#C15B2A", "s")):
            axis.plot([r["budget"] for r in flat], [100*r[key] for r in flat], color=color, marker=marker, markersize=4, linewidth=1.5, label=label)
        axis.set_xscale("log")
        axis.set_xticks([250, 1000, 3200, 13000], ["250", "1,000", "3,200", "13,000"])
        axis.minorticks_off()
        axis.set_ylim(*limits)
        axis.set_xlabel("Training documents")
        axis.set_ylabel("F1 score (%)")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(fontsize=8, loc="lower right", frameon=False)
        axis.tick_params(labelsize=8)
    directory = ROOT / "paper/R2/ACM_KG_puctuation/figures"
    for suffix in ("pdf", "png"):
        fig.savefig(directory / f"learning_curve_news.{suffix}", dpi=220)
    plt.close(fig)
    print("\n".join(md))


if __name__ == "__main__":
    main()
