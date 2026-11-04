"""Per-class precision/recall/F1 tables for every run in the model comparison (R1.2).

The run registry is ``reports/tables/model_comparison.csv``; each row points to an
archived run under ``experiments/``, whose sklearn classification report is read
unchanged. No model is re-evaluated here.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LABELS = ["S", "C", "P"]
AVERAGES = {"macro avg": "macro", "weighted avg": "weighted"}

# Display names used in the paper (kg_punct_apr.tex, Tables results/news-results).
DISPLAY_NAMES = {
    "logreg_balanced": "Balanced LR",
    "sgd_log_balanced": "Balanced SGD-LR",
    "cnn_sequence_tagger": "CNN",
    "cnn_position_tagger": "Position-aware CNN",
    "metinovadilet/KyrgyzBert": "KyrgyzBERT",
    "metinovadilet/KyrgyzBert2": "KyrgyzBERT2",
    "Zarinaaa/kyrgyzbert-kyrgyz-punctuation": "KyrgyzBERT-PR (zero-shot)",
}
DOMAIN_NAMES = {"wikipedia_ky": "Wikipedia", "kyrgyznews": "News"}


def write_per_class_report(
    comparison_csv: Path = ROOT / "reports/tables/model_comparison.csv",
    output_dir: Path = ROOT / "reports",
) -> list[str]:
    rows = list(csv.DictReader(comparison_csv.open(encoding="utf-8")))
    records = [_record(row) for row in rows]

    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    long_csv = tables_dir / "per_class_metrics.csv"
    latex = tables_dir / "per_class_metrics.tex"
    markdown = output_dir / "per_class_results.md"

    _write_long_csv(records, long_csv)
    latex.write_text(_latex_table(records), encoding="utf-8")
    markdown.write_text(_markdown(records), encoding="utf-8")
    return [str(long_csv), str(latex), str(markdown)]


def _record(row: dict[str, str]) -> dict[str, object]:
    report_path = _find_report(row)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    # Guard against a stale registry: the report must reproduce the registry numbers.
    for key, value in (("accuracy", report["accuracy"]), ("macro_f1", report["macro avg"]["f1-score"])):
        if abs(float(row[key]) - value) > 1e-5:
            raise ValueError(f"{report_path}: {key}={value} != model_comparison {row[key]}")
    if abs(float(row["comma_f1"]) - report["C"]["f1-score"]) > 1e-5:
        raise ValueError(f"{report_path}: comma F1 mismatch with model_comparison")

    setting, test_domain, model = _describe(row)
    return {
        "setting": setting,
        "test_domain": test_domain,
        "model": model,
        "family": row["family"],
        "run": row["run"],
        "report_path": str(report_path.relative_to(ROOT)),
        "accuracy": report["accuracy"],
        "metrics": {
            name: report[name] for name in LABELS + list(AVERAGES)
        },
    }


def _find_report(row: dict[str, str]) -> Path:
    tables = ROOT / "experiments" / row["run"] / "tables"
    candidates = sorted(tables.glob("*_classification_report.json"))
    if row["dataset"] == "mixed_domain":
        domain = "wikipedia_ky" if row["model"].endswith("Wikipedia") else "kyrgyznews"
        candidates = [path for path in candidates if path.name.startswith(domain)]
    elif len(candidates) > 1:
        candidates = [path for path in candidates if path.name.startswith(row["model"] + "_")]
    if len(candidates) != 1:
        raise FileNotFoundError(f"Ambiguous or missing classification report for {row['run']}: {candidates}")
    return candidates[0]


def _describe(row: dict[str, str]) -> tuple[str, str, str]:
    dataset, model = row["dataset"], row["model"]
    if dataset in DOMAIN_NAMES:
        name = DISPLAY_NAMES.get(model)
        if name is None:  # xlm-roberta-base fine-tuned on this domain = our checkpoint
            name = "KG-Punct-XLMR-" + ("Wiki" if dataset == "wikipedia_ky" else "News")
        return "within-domain", DOMAIN_NAMES[dataset], name
    checkpoint, target = [part.strip() for part in model.split("->")]
    setting = "cross-domain" if dataset == "cross_domain" else "mixed-domain"
    return setting, target.replace("KyrgyzNews", "News"), checkpoint


def _write_long_csv(records: list[dict[str, object]], output_path: Path) -> None:
    fields = [
        "setting", "test_domain", "model", "family", "class",
        "precision", "recall", "f1", "support", "run", "report_path",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            for name, metrics in record["metrics"].items():
                writer.writerow({
                    "setting": record["setting"],
                    "test_domain": record["test_domain"],
                    "model": record["model"],
                    "family": record["family"],
                    "class": AVERAGES.get(name, name),
                    "precision": f"{metrics['precision']:.6f}",
                    "recall": f"{metrics['recall']:.6f}",
                    "f1": f"{metrics['f1-score']:.6f}",
                    "support": int(metrics["support"]),
                    "run": record["run"],
                    "report_path": record["report_path"],
                })


def _pct(value: float) -> str:
    return f"{100 * value:.2f}"


def _cells(record: dict[str, object]) -> list[str]:
    metrics = record["metrics"]
    cells = []
    for label in ["C", "P", "S"]:
        cells += [_pct(metrics[label][key]) for key in ("precision", "recall", "f1-score")]
    cells += [_pct(metrics["macro avg"][key]) for key in ("precision", "recall")]
    return cells


def _panels(records: list[dict[str, object]]) -> list[tuple[str, list[tuple[str, dict[str, object]]]]]:
    panels = []
    for domain in ("Wikipedia", "News"):
        rows = [r for r in records if r["setting"] == "within-domain" and r["test_domain"] == domain]
        rows.sort(key=lambda r: r["family"] == "external_apr_checkpoint")  # stable: external last
        panels.append((f"Within-domain: {domain} test split", [(r["model"], r) for r in rows]))
    transfer = [r for r in records if r["setting"] != "within-domain"]
    panels.append((
        "Cross-domain (zero-shot) and mixed-domain",
        [(f"{r['model']} $\\rightarrow$ {r['test_domain']}", r) for r in transfer],
    ))
    return panels


def _latex_table(records: list[dict[str, object]]) -> str:
    lines = [
        "% Generated by `kg_punctuation.cli report-per-class`; do not edit by hand.",
        "% Source: reports/tables/per_class_metrics.csv",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Per-class precision (Pr), recall (Re), and F1 (\\%) for every model on both",
        "test splits and in the cross-/mixed-domain settings. \\texttt{C}: comma, \\texttt{P}:",
        "period, \\texttt{S}: no punctuation. The last two columns are macro-averaged",
        "precision and recall; macro-F1 is given in Tables~\\ref{tab:results},",
        "\\ref{tab:news-results}, and~\\ref{tab:crossdomain}.}",
        "\\label{tab:per-class}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{lrrrrrrrrrrr}",
        "\\toprule",
        " & \\multicolumn{3}{c}{Comma (\\texttt{C})} & \\multicolumn{3}{c}{Period (\\texttt{P})}"
        " & \\multicolumn{3}{c}{Space (\\texttt{S})} & \\multicolumn{2}{c}{Macro} \\\\",
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\\cmidrule(lr){11-12}",
        "Model & Pr & Re & F1 & Pr & Re & F1 & Pr & Re & F1 & Pr & Re \\\\",
    ]
    for title, rows in _panels(records):
        lines.append("\\midrule")
        lines.append(f"\\multicolumn{{12}}{{l}}{{\\textit{{{title}}}}} \\\\")
        for name, record in rows:
            lines.append(" & ".join([name] + _cells(record)) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    return "\n".join(lines)


def _markdown(records: list[dict[str, object]]) -> str:
    header = "| Model | C Pr | C Re | C F1 | P Pr | P Re | P F1 | S Pr | S Re | S F1 | Macro Pr | Macro Re |"
    lines = [
        "# Per-Class Precision / Recall / F1",
        "",
        "Generated by `PYTHONPATH=src .venv/bin/python -m kg_punctuation.cli report-per-class`",
        "from the classification reports of the runs listed in `reports/tables/model_comparison.csv`.",
        "All values in %. Long-form data: `reports/tables/per_class_metrics.csv`.",
    ]
    for title, rows in _panels(records):
        lines += ["", "## " + title, "", header, "|" + "---|" * 12]
        for name, record in rows:
            name = name.replace("$\\rightarrow$", "→")
            lines.append("| " + " | ".join([name] + _cells(record)) + " |")
    lines += ["", "## Source runs", ""]
    for record in records:
        lines.append(f"- {record['model']} / {record['test_domain']}: `{record['report_path']}`")
    return "\n".join(lines) + "\n"
