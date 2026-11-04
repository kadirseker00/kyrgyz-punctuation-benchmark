"""Census of core-label errors and a deterministic, unannotated stratified audit pack."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

from kg_punctuation.benchmark_v2 import Benchmark, DEFAULT_MANIFEST, ROOT
from kg_punctuation.overlap_sensitivity import build_mask, digest, rescore

SEED = "20261104"
PER_STRATUM = 12
LABELS = ("S", "C", "P")


def main():
    benchmark = Benchmark(DEFAULT_MANIFEST)
    output = ROOT / "reports/r3/errors"; output.mkdir(parents=True, exist_ok=True)
    report = {"manifest_sha256": digest(DEFAULT_MANIFEST), "code_sha256": digest(__file__),
              "sampling": {"seed": SEED, "per_stratum": PER_STRATUM,
                           "rule": "Lowest SHA-256(seed/source/document_id/word_index) within each of six off-diagonal confusion strata and one correct stratum, per source; all eligible full-test positions included in the sampling frame. This post-hoc audit pack is not human annotated."},
              "sources": {}}
    samples, summary_rows = [], []
    for source in ("kyrgyznews", "wikipedia_ky"):
        job = "xlmr_news" if source == "kyrgyznews" else "xlmr_wiki"
        path = ROOT / f"experiments/v2_main/{job}/predictions_test_{source}.csv"
        # Canonical identity and complete-word coverage checks before any analysis.
        docs = benchmark.documents(source, "test", "extended")
        frame = build_mask(docs, set(), set())
        rescore(path, frame)
        with path.open(newline="") as f: rows = list(csv.DictReader(f))
        by_document = {d["document_id"]: d for d in docs}
        confusion = Counter((r["gold"], r["prediction"]) for r in rows)
        errors = sum(n for (a, b), n in confusion.items() if a != b)
        strata = defaultdict(list)
        for row in rows:
            label = "correct" if row["gold"] == row["prediction"] else f"{row['gold']}->{row['prediction']}"
            key = f"{SEED}/{source}/{row['document_id']}/{row['word_index']}"
            strata[label].append((hashlib.sha256(key.encode()).hexdigest(), row))
        for label, candidates in sorted(strata.items()):
            for rank, (key, row) in enumerate(sorted(candidates, key=lambda x: x[0])[:PER_STRATUM], 1):
                doc = by_document[row["document_id"]]; i = int(row["word_index"])
                start, end = max(0, i - 24), min(len(doc["tokens"]), i + 25)
                samples.append({"sample_id": key[:16], "source": source,
                                "document_id": row["document_id"], "group_id": row["group_id"],
                                "word_index": i, "stratum": label, "rank": rank,
                                "stratum_population": len(candidates), "context_start": start,
                                "context": " ".join(doc["tokens"][start:end]),
                                "focus_word": doc["tokens"][i], "focus_offset": i - start,
                                "reference_label": row["gold"], "prediction_label": row["prediction"],
                                "rater_1_acceptable_labels": "", "rater_2_acceptable_labels": "",
                                "adjudicated_acceptable_labels": "", "notes": ""})
        matrix = [[confusion[a, b] for b in LABELS] for a in LABELS]
        report["sources"][source] = {"prediction_path": str(path.relative_to(ROOT)),
                                     "prediction_sha256": digest(path), "words": len(rows),
                                     "errors": errors, "confusion_labels": LABELS,
                                     "confusion_gold_rows_predicted_columns": matrix,
                                     "stratum_population": {k: len(v) for k, v in strata.items()},
                                     "sampled_positions": sum(min(PER_STRATUM, len(v)) for v in strata.values())}
        for a in LABELS:
            for b in LABELS:
                if a != b:
                    summary_rows.append({"source": source, "gold": a, "prediction": b,
                                         "count": confusion[a, b], "share_of_errors": confusion[a, b] / errors})
    with (output / "confusion_census.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0])); writer.writeheader(); writer.writerows(summary_rows)
    with (output / "annotation_pack.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(samples[0])); writer.writeheader(); writer.writerows(samples)
    report["annotation_pack_sha256"] = digest(output / "annotation_pack.csv")
    report["human_judgments_collected"] = 0
    (output / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    md = ["# Systematic error census", "", "Every canonical core-label test word of the source-specific seed-42 XLM-R checkpoint is included. Matrix rows are reference labels; columns are predictions. Counts indicate disagreement with one source-derived reference, not adjudicated linguistic errors.", "",
          "| Reference → prediction | News count (% of errors) | Wikipedia count (% of errors) |", "|---|---:|---:|"]
    tex = [r"\begin{table}[t]", r"\centering\small", r"\caption{Census of all core-label reference disagreements for the source-specific seed-42 XLM-R checkpoints. Parentheses give the share of each source's disagreements. These are single-reference counts, not human judgments of acceptability.}",
           r"\label{tab:error-census}", r"\begin{tabular}{lrr}", r"\toprule", r"Reference $\to$ prediction & News & Wikipedia \\", r"\midrule"]
    for a in LABELS:
        for b in LABELS:
            if a == b: continue
            selected = [next(r for r in summary_rows if r["source"] == s and r["gold"] == a and r["prediction"] == b)
                        for s in ("kyrgyznews", "wikipedia_ky")]
            cells = [f"{r['count']:,} ({r['share_of_errors']:.1%})" for r in selected]
            md.append(f"| {a} → {b} | {' | '.join(cells)} |")
            tex.append(f"{a} $\\to$ {b} & " + " & ".join(c.replace("%", r"\%") for c in cells) + r" \\")
    totals = [report["sources"][s]["errors"] for s in ("kyrgyznews", "wikipedia_ky")]
    tex.extend([r"\midrule", "All disagreements & " + " & ".join(f"{x:,}" for x in totals) + r" \\",
                r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    table = ROOT / "paper/R3/ACM_KG_puctuation/tables/error_census_r3.tex"
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text("\n".join(tex) + "\n")
    md += ["", f"The separate audit pack has {len(samples)} positions, at most 12 per stratum per source, chosen by a fixed hash order. Correct cases are also sampled. Because strata have different sizes, its raw sample proportions must not be used to estimate corpus-wide acceptability.", "", "No native-speaker annotation has been collected. Suggested protocol: two independent fluent Kyrgyz raters inspect the word context and mark all acceptable labels S/C/P before seeing the model/reference columns; use the full document when context is insufficient; adjudicate disagreements; report multi-reference scores separately from the canonical benchmark. Blank rating columns are intentional."]
    (output / "README.md").write_text("\n".join(md) + "\n")
    print(json.dumps({"samples": len(samples), "sources": report["sources"]}, indent=2))


if __name__ == "__main__": main()
