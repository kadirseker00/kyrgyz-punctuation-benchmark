"""Independently verify the rare-label ablation and render its paper table."""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from kg_punctuation.benchmark_v2 import Benchmark, DEFAULT_MANIFEST, ROOT, score

LABELS = ("S", "C", "P", "Q", "E")


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def independent(path):
    with path.open(newline="") as f: rows = list(csv.DictReader(f))
    gold, pred = [r["gold"] for r in rows], [r["prediction"] for r in rows]
    precision, recall, f1, support = precision_recall_fscore_support(gold, pred, labels=LABELS, zero_division=0)
    counts = Counter(pred)
    return {"macro_f1": float(f1.mean()), "words": len(rows), "per_class": {
        l: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]),
            "support": int(support[i]), "predicted": counts[l]} for i, l in enumerate(LABELS)}}


def main():
    output = ROOT / "reports/r3/rare"; output.mkdir(parents=True, exist_ok=True)
    benchmark = Benchmark(DEFAULT_MANIFEST)
    status_path = ROOT / "experiments/r3/rare_unweighted/status.json"
    if not status_path.exists(): status_path = ROOT / "paper/R3/reproducibility/portable_rare_status.json"
    status = json.loads(status_path.read_text())
    result = {"manifest_sha256": sha(DEFAULT_MANIFEST), "reporter_sha256": sha(__file__),
              "interpretation": "Single-seed exploratory comparisons. All runs use the same canonical documents, schedule and base-model revision; best epoch selected by five-label validation macro-F1. Offsets selected exclusively on validation, never by test performance. Offsets adjust decisions, not probability calibration. No inference of reliable exclamation restoration or causal effect across all random seeds.",
              "sources": {}}
    flat = []
    for source, domain in (("kyrgyznews", "news"), ("wikipedia_ky", "wiki")):
        baseline = list((ROOT / f"experiments/v2_audits/extended_{domain}").glob("*/metadata.json"))
        assert len(baseline) == 1
        baseline_path = baseline[0]; weighted = json.loads(baseline_path.read_text())
        new = status["jobs"][source]; assert new["state"] == "complete"
        run_dir = Path(new["run_dir"])
        if not run_dir.is_absolute(): run_dir = ROOT / run_dir
        unweighted = json.loads((run_dir / "metadata.json").read_text())
        for key in ("model_id", "model_revision", "seed", "best_epoch"):
            if key != "best_epoch": assert weighted[key] == unweighted[key]
        for key in ("max_length", "batch_size", "epochs", "learning_rate", "optimizer", "weight_decay", "scheduler", "train_documents"):
            assert weighted["hyperparameters"][key] == unweighted["hyperparameters"][key]
        for key in ("manifest_sha256", "source_sha256", "train_document_ids", "validation_document_ids", "test_document_ids"):
            assert weighted["benchmark"][key] == unweighted["benchmark"][key]
        for key, value in weighted["environment"].items():
            if key != "platform": assert value == unweighted["environment"][key], key
        assert set(unweighted["hyperparameters"]["class_weights"].values()) == {1.}
        calibration = ROOT / f"experiments/r3/calibration/weighted_{domain}"
        selection = json.loads((calibration / "validation_selection.json").read_text())
        cache = np.load(calibration / "validation_logits.npz")
        trials = []
        for q, e in itertools.product(selection["grid"], repeat=2):
            pred = (cache["logits"] + [0, 0, 0, -q, -e]).argmax(axis=1)
            f1 = precision_recall_fscore_support(cache["gold"], pred, labels=range(5), zero_division=0)[2].mean()
            trials.append((float(f1), -(q + e), -q, -e, q, e))
        best = max(trials)
        assert (best[-2], best[-1]) == (selection["selected"]["q_penalty"], selection["selected"]["e_penalty"])
        assert abs(best[0] - selection["selected"]["validation_macro_f1"]) < 1e-12
        with (baseline_path.parent / "tables/predictions.csv").open() as a, (calibration / "predictions_raw.csv").open() as b:
            for old_row, new_row in itertools.zip_longest(csv.DictReader(a), csv.DictReader(b)):
                assert old_row == new_row, "Raw checkpoint predictions must reproduce the archived baseline"
        paths = {"weighted_CE": calibration / "predictions_raw.csv",
                 "weighted_CE_validation_offsets": calibration / "predictions_validation_offsets.csv",
                 "unweighted_CE": run_dir / "tables/predictions.csv"}
        conditions = {}
        for condition, path in paths.items():
            canonical = score(path, benchmark, source, view="extended")
            values = independent(path)
            assert abs(values["macro_f1"] - canonical["macro_f1"]) < 1e-12
            for l in LABELS:
                for metric in ("precision", "recall", "f1", "support"):
                    assert abs(values["per_class"][l][metric] - canonical["per_class"][l][metric]) < 1e-12
            conditions[condition] = {"prediction_path": str(path.relative_to(ROOT)), "prediction_sha256": sha(path), **values}
            flat.append({"source": source, "condition": condition, "macro_f1": values["macro_f1"],
                         **{f"{l}_{k}": v for l, metrics in values["per_class"].items() for k, v in metrics.items()}})
        result["sources"][source] = {"weighted_metadata": str(baseline_path.relative_to(ROOT)),
                                     "weighted_metadata_sha256": sha(baseline_path),
                                     "unweighted_metadata": str((run_dir / "metadata.json").relative_to(ROOT)),
                                     "unweighted_metadata_sha256": sha(run_dir / "metadata.json"),
                                     "weighted_best_epoch": weighted["best_epoch"], "unweighted_best_epoch": unweighted["best_epoch"],
                                     "weighted_class_weights": weighted["hyperparameters"]["class_weights"],
                                     "weighted_environment": weighted["environment"],
                                     "unweighted_environment": unweighted["environment"],
                                     "environment_note": "Library versions and arm64 processor match. Historical weighted runs used macOS 27.0; new unweighted runs used 27.0.1. MPS deterministic algorithms were disabled in both. The OS patch difference and single-seed design limit a strict loss-only causal interpretation.",
                                     "validation_selection": selection["selected"],
                                     "validation_selection_sha256": sha(calibration / "validation_selection.json"),
                                     "conditions": conditions}
    (output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    with (output / "comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(flat)
    md = ["# Rare-label loss and decision-offset comparison", "", result["interpretation"], "",
          "| Source | Condition | Macro-F1 | Q P/R/F1 | Q predicted/support | E P/R/F1 | E predicted/support |", "|---|---|---:|---|---|---|---|"]
    tex = [r"\begin{table}[t]", r"\centering\footnotesize", r"\setlength{\tabcolsep}{3pt}",
           r"\caption{Five-label controls on the same canonical test sets (\%, seed 42). Each model uses 3,200 training documents. Offsets are selected on validation only. Pred./gold show rare-mark prediction counts and test support. Macro-F1 averages S/C/P/Q/E.}",
           r"\label{tab:rare-controls}", r"\begin{tabular}{llrrrrrrrrr}", r"\toprule",
           r"& & & \multicolumn{4}{c}{Question} & \multicolumn{4}{c}{Exclamation} \\",
           r"Source & Condition & Macro-F1 & Pr & Re & F1 & Pred./gold & Pr & Re & F1 & Pred./gold \\", r"\midrule"]
    names = {"weighted_CE": "Weighted CE", "weighted_CE_validation_offsets": "Weighted + offsets", "unweighted_CE": "Unweighted CE"}
    for source, group in result["sources"].items():
        for condition, values in group["conditions"].items():
            source_name = "News" if source == "kyrgyznews" else "Wikipedia"
            cells = []
            for l in ("Q", "E"):
                v = values["per_class"][l]
                cells += [f"{100*v[k]:.2f}" for k in ("precision", "recall", "f1")] + [f"{v['predicted']}/{v['support']}"]
            tex.append(f"{source_name} & {names[condition]} & {100*values['macro_f1']:.2f} & " + " & ".join(cells) + r" \\")
            md.append(f"| {source_name} | {names[condition]} | {100*values['macro_f1']:.2f} | {'/'.join(cells[:3])} | {cells[3]} | {'/'.join(cells[4:7])} | {cells[7]} |")
    md.extend(["", "Library versions match, but historical weighted runs used macOS 27.0 and new unweighted runs 27.0.1 on arm64. MPS deterministic algorithms were disabled in both. This environment difference and the single seed limit a strict causal interpretation of changing the loss."])
    tex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    table = ROOT / "paper/R3/ACM_KG_puctuation/tables/rare_controls_r3.tex"
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text("\n".join(tex) + "\n")
    (output / "README.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__": main()
