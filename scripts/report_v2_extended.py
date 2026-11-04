"""Verify completed five-label runs and generate the R2 extended-audit table."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from kg_punctuation.benchmark_v2 import Benchmark, DEFAULT_MANIFEST, ROOT, score


def main():
    queue_path = ROOT / "experiments/v2_audits/queue_status.json"
    queue = json.loads(queue_path.read_text())
    manifest_hash = hashlib.sha256(DEFAULT_MANIFEST.read_bytes()).hexdigest()
    if queue["manifest_sha256"] != manifest_hash:
        raise ValueError("Queue and current manifest differ.")
    benchmark = Benchmark()
    rows = []
    jobs = [
        ("extended_news", "kyrgyznews", "kyrgyznews"),
        ("extended_wiki", "wikipedia_ky", "wikipedia_ky"),
        ("cross_news_to_wiki", "kyrgyznews", "wikipedia_ky"),
        ("cross_wiki_to_news", "wikipedia_ky", "kyrgyznews"),
    ]
    for job, source, target in jobs:
        state = queue["jobs"][job]
        if state["status"] != "complete":
            raise ValueError(f"Incomplete job: {job}")
        run = Path(state["run_dir"])
        metadata_path = run / "metadata.json"
        predictions = run / "tables/predictions.csv"
        metadata = json.loads(metadata_path.read_text())
        checked = score(predictions, benchmark, target, view="extended")
        if metadata["benchmark"]["manifest_sha256"] != manifest_hash:
            raise ValueError(f"Run manifest mismatch: {job}")
        labels = ["S", "C", "P", "Q", "E"]
        if metadata["label_set"] != labels:
            raise ValueError(f"Run label mismatch: {job}")
        with predictions.open(newline="") as handle:
            counts = Counter((r["gold"], r["prediction"]) for r in csv.DictReader(handle))
        classes = {}
        for label in labels:
            tp = counts[label, label]
            support = sum(n for (gold, pred), n in counts.items() if gold == label)
            predicted = sum(n for (gold, pred), n in counts.items() if pred == label)
            classes[label] = {
                "precision": tp / predicted if predicted else 0.0,
                "recall": tp / support if support else 0.0,
                "f1": 2 * tp / (support + predicted) if support + predicted else 0.0,
                "support": support, "predicted": predicted,
            }
        macro_f1 = sum(row["f1"] for row in classes.values()) / len(labels)
        archived = next(iter(metadata["metrics"].values()))
        if abs(macro_f1 - checked["macro_f1"]) > 1e-10 or abs(macro_f1 - archived["macro_f1"]) > 1e-6:
            raise ValueError(f"Macro-F1 mismatch: {job}")
        for label, values in classes.items():
            for key in ("precision", "recall", "f1", "support"):
                if abs(values[key] - archived["per_class"][label][key]) > 1e-6:
                    raise ValueError(f"Class metric mismatch: {job}/{label}/{key}")
        rows.append({
            "job": job, "source": source, "target": target,
            "run_dir": str(run.relative_to(ROOT)),
            "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
            "predictions_sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(),
            "test_tokens": sum(counts.values()), "macro_f1": macro_f1,
            "per_class": classes, "best_epoch": metadata.get("best_epoch"),
            "runtime_seconds_total": metadata.get("runtime_seconds_total"),
            "runtime_seconds_test": metadata.get("runtime_seconds_test"),
        })
    output = ROOT / "reports/v2"
    output.mkdir(parents=True, exist_ok=True)
    report = {"manifest_sha256": manifest_hash, "labels": labels,
              "verification": "Exact canonical test coverage plus independent confusion-count metrics",
              "runs": rows}
    (output / "extended_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    fields = ["job", "source", "target", "label", "precision", "recall", "f1", "support", "predicted"]
    with (output / "extended_per_class.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for label, values in row["per_class"].items():
                writer.writerow({key: row[key] for key in ("job", "source", "target")} | {"label": label} | values)

    names = {"kyrgyznews": "News", "wikipedia_ky": "Wikipedia"}
    table = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Canonical v2 five-label audit (\%). Each source model uses 3{,}200 training documents, seed 42, and full-word evaluation. Pr/Re denote precision/recall. Test supports for \texttt{Q}/\texttt{E} are 364/71 on News and 187/117 on Wikipedia. Transfer uses no target-domain fine-tuning.}",
        r"\label{tab:extended}", r"\small", r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{llrrrrrrr}", r"\toprule",
        r"Train & Test & Macro-F1 & \texttt{Q} Pr & \texttt{Q} Re & \texttt{Q} F1 & \texttt{E} Pr & \texttt{E} Re & \texttt{E} F1 \\",
        r"\midrule",
    ]
    markdown = ["# Canonical v2 extended audit", "", "All metrics below are percentages.", "",
                "| Train | Test | Macro-F1 | Q P/R/F1 | E P/R/F1 |", "|---|---|---:|---:|---:|"]
    for row in rows:
        q, e = row["per_class"]["Q"], row["per_class"]["E"]
        values = [row["macro_f1"]] + [c[k] for c in (q, e) for k in ("precision", "recall", "f1")]
        table.append(" & ".join([names[row["source"]], names[row["target"]]] + [f"{100*v:.2f}" for v in values]) + r" \\")
        markdown.append(f"| {names[row['source']]} | {names[row['target']]} | {100*row['macro_f1']:.2f} | "
                        + " / ".join(f"{100*q[k]:.2f}" for k in ("precision", "recall", "f1")) + " | "
                        + " / ".join(f"{100*e[k]:.2f}" for k in ("precision", "recall", "f1")) + " |")
    table.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    for directory in (output / "tables", ROOT / "paper/R2/ACM_KG_puctuation/tables"):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "extended_v2.tex").write_text("\n".join(table) + "\n")
    (output / "extended_audit.md").write_text("\n".join(markdown) + "\n")
    print("\n".join(markdown))


if __name__ == "__main__":
    main()
