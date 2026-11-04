"""Deterministic selection of qualitative examples for the v2 paper (R1.4, T04).

Rule: for each source, using the within-domain XLM-R
predictions on the canonical test split, consider passages of two consecutive complete
core-label sentences, each at least five words and totaling 20-45 words. Neither sentence
may occur verbatim in the main training documents. Take, in manifest test order,
  (a) the first passage that contains a comma and is predicted entirely correctly, and
  (b) the first passage with exactly one error whose (gold -> predicted) type is the most
      frequent error type of that model on the whole test split.
Revision on 2026-10-01: use consecutive sentences and only alphabetic Cyrillic tokens,
excluding phone numbers, numerical fragments, and split website names. These criteria were added after
the original examples were inspected; this is illustrative, not preregistered evaluation.
To avoid boundaries after initials/abbreviations, the token immediately before the passage
(if present) and each sentence-final token must have at least three letters.
Writes reports/r1/qualitative_examples_v2.json with Cyrillic, BGN/PCGN romanization and
provenance. English translations are added by hand in the paper and checked separately.

Usage: PYTHONPATH=src .venv/bin/python scripts/select_examples_v2.py [--panel-dir experiments/v2_main]
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from kg_punctuation import benchmark_v2 as bv
from kg_punctuation.translit import romanize

ROOT = bv.ROOT
MARK = {"S": "", "C": ",", "P": ".", "Q": "?", "E": "!"}
JOBS = {"kyrgyznews": "xlmr_news", "wikipedia_ky": "xlmr_wiki"}


def render(words, labels):
    return " ".join(w + MARK[l] for w, l in zip(words, labels))


def write_tables(examples, directory):
    translations = json.loads((ROOT / "reports/r1/qualitative_translations_v2.json").read_text(encoding="utf-8"))
    english = {(e["document_id"], tuple(e["word_span"])): e["translation"] for e in translations["examples"]}
    for source, name, label in (("kyrgyznews", "News", "qualitative"),
                                ("wikipedia_ky", "Wikipedia", "qualitative-wiki")):
        lines = [r"\begin{table*}[tp]", r"\centering",
                 rf"\caption{{Two-sentence {name} passages from the canonical test split. Input has no punctuation or sentence breaks; Gold and Model show the reference and archived KG-Punct-XLMR predictions. Punctuation is bold; an extra predicted mark is underlined. English translations convey the reference meaning. The revised selection rule is described in Section~\ref{{sec:qualitative}}.}}",
                 rf"\label{{tab:{label}}}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
                 r"\renewcommand{\arraystretch}{1.15}",
                 r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.07\textwidth}>{\raggedright\arraybackslash}p{0.44\textwidth}>{\raggedright\arraybackslash}p{0.44\textwidth}@{}}",
                 r"\toprule", r" & Kyrgyz (Cyrillic) & Romanization (BGN/PCGN) \\", r"\midrule"]
        for ex in (e for e in examples if e["source"] == source):
            title = "Exact reference match" if ex["kind"] == "correct" else "One extra comma"
            lines.append(rf"\multicolumn{{3}}{{@{{}}l}}{{\textbf{{{title}}}}} \\")
            words = ex["input_cyrillic"].split()
            for row, labels in (("Input", ["S"] * len(words)), ("Gold", ex["gold_labels"]), ("Model", ex["prediction_labels"])):
                def format_words(latin):
                    out = []
                    for word, gold, predicted in zip(words, ex["gold_labels"], labels):
                        mark = MARK[predicted]
                        if mark:
                            mark = rf"\textbf{{{mark}}}"
                            if row == "Model" and predicted != gold:
                                mark = rf"\underline{{{mark}}}"
                        out.append((romanize(word) if latin else word) + mark)
                    return " ".join(out)
                lines.append(row + r" & \ky{" + format_words(False) + "} & " + format_words(True) + r" \\")
            translation = english[(ex["document_id"], tuple(ex["word_span"]))]
            lines.append(r"\multicolumn{3}{@{}>{\raggedright\arraybackslash}p{0.97\textwidth}@{}}{\emph{English:} " + translation + r"} \\")
            lines.append(r"\midrule")
        lines[-1] = r"\bottomrule"
        lines.extend((r"\end{tabular}", r"\end{table*}"))
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"qualitative_{source}_v2.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-dir", type=Path, default=ROOT / "experiments/v2_main")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/r1/qualitative_examples_v2.json")
    parser.add_argument("--tables-dir", type=Path, help="Also render bilingual paper tables using the checked-in draft translations")
    args = parser.parse_args()
    bench = bv.Benchmark()
    result = {"rule": __doc__.split("Rule")[1].split("Writes")[0].strip(), "examples": []}
    for source, job in JOBS.items():
        path = args.panel_dir / job / f"predictions_test_{source}.csv"
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        bv.score(path, bench, source)  # asserts identity and coverage
        errors = Counter((r["gold"], r["prediction"]) for r in rows if r["gold"] != r["prediction"])
        top_error = errors.most_common(1)[0][0]
        train = bench.documents(source, "train", "core", limit=bv.MAIN_TRAIN_DOCUMENTS)
        seen = {tuple(d["tokens"][a:b]) for d in train for a, b in bv.sentence_spans(d["labels"])}
        by_doc = defaultdict(list)
        for r in rows:
            by_doc[r["document_id"]].append(r)
        docs = {d["document_id"]: d for d in bench.documents(source, "test", "core")}
        picked = {}
        for doc_id in bench.split_ids(source, "test"):
            doc, preds = docs[doc_id], [r["prediction"] for r in by_doc[doc_id]]
            start, spans = 0, []
            for end, label in enumerate(doc["labels"], 1):
                if label == "P":
                    spans.append((start, end))
                    start = end
            for first, second in zip(spans, spans[1:]):
                boundary_indices = [first[1] - 1, second[1] - 1]
                if first[0]:
                    boundary_indices.append(first[0] - 1)
                if any(len(doc["tokens"][i]) < 3 for i in boundary_indices):
                    continue
                if any(b - a < 5 or tuple(doc["tokens"][a:b]) in seen for a, b in (first, second)):
                    continue
                span = (first[0], second[1])
                words = doc["tokens"][span[0]:span[1]]
                if not 20 <= len(words) <= 45:
                    continue
                if not all(all(c.isalpha() and "\u0400" <= c <= "\u04ff" for c in word) for word in words):
                    continue
                gold, pred = doc["labels"][span[0]:span[1]], preds[span[0]:span[1]]
                wrong = [(g, p) for g, p in zip(gold, pred) if g != p]
                kind = None
                if not wrong and "C" in gold and "correct" not in picked:
                    kind = "correct"
                elif len(wrong) == 1 and wrong[0] == top_error and "error" not in picked:
                    kind = "error"
                if kind:
                    picked[kind] = {
                        "source": source, "kind": kind, "document_id": doc_id, "word_span": list(span),
                        "sentence_spans": [list(first), list(second)],
                        "gold_labels": gold, "prediction_labels": pred,
                        "error_type": None if kind == "correct" else f"{top_error[0]}->{top_error[1]}",
                        "input_cyrillic": " ".join(words), "gold_cyrillic": render(words, gold),
                        "model_cyrillic": render(words, pred),
                        "input_latin": romanize(" ".join(words)), "gold_latin": romanize(render(words, gold)),
                        "model_latin": romanize(render(words, pred)),
                        "error_type_count": errors[top_error], "all_error_counts": {f"{g}->{p}": c for (g, p), c in errors.items()},
                    }
            if len(picked) == 2:
                break
        if set(picked) != {"correct", "error"}:
            raise ValueError(f"No eligible pair of examples for {source}")
        result["examples"].extend(picked[k] for k in ("correct", "error"))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.tables_dir:
        write_tables(result["examples"], args.tables_dir)
    for ex in result["examples"]:
        print(ex["source"], ex["kind"], ex["error_type"], ex["document_id"], ex["word_span"])
        print("   gold ", ex["gold_cyrillic"]); print("   model", ex["model_cyrillic"]); print("   latin", ex["gold_latin"])


if __name__ == "__main__":
    main()
