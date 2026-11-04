"""Dataset statistics for the canonical v2 benchmark (paper Tables data / extended-data, Figure labeldist).

Usage: PYTHONPATH=src .venv/bin/python scripts/v2_data_stats.py  ->  reports/v2/data_stats.json
"""

from __future__ import annotations

import json
from collections import Counter

from kg_punctuation import benchmark_v2 as bv


def main() -> None:
    bench = bv.Benchmark()
    out = {}
    for source, meta in bench.manifest["sources"].items():
        splits = {"train_main": bench.documents(source, "train", "extended", limit=bv.MAIN_TRAIN_DOCUMENTS),
                  "validation": bench.documents(source, "validation", "extended"),
                  "test": bench.documents(source, "test", "extended")}
        used = [d for docs in splits.values() for d in docs]
        ext = Counter(label for d in used for label in d["labels"])
        core = Counter(label for d in used for label in bv.project(d["labels"], "core"))
        words = sum(ext.values())
        out[source] = {
            "pool_documents": meta["pool_documents"],
            "exact_duplicates_dropped": len(meta["dropped_exact"]),
            "groups": meta["group_size_summary"]["groups"],
            "multi_document_groups": meta["group_size_summary"]["multi_document_groups"],
            "largest_group": meta["group_size_summary"]["largest_groups"][0],
            "train_pool_documents": meta["split_documents"]["train"],
            "documents": {k: len(v) for k, v in splits.items()},
            "words": {k: sum(len(d["tokens"]) for d in v) for k, v in splits.items()},
            "benchmark_documents": len(used),
            "benchmark_words": words,
            "core_share": {k: core[k] / words for k in ("S", "C", "P")},
            "extended_counts": {k: ext[k] for k in ("S", "C", "P", "Q", "E")},
            "extended_share": {k: ext[k] / words for k in ("Q", "E")},
            "test_extended_counts": dict(Counter(l for d in splits["test"] for l in d["labels"])),
            "residual_overlap_test_vs_main_train_pct": meta["residual_sentence_overlap"]["test_vs_main_train"]["words_in_seen_sentences_pct"],
        }
    path = bv.ROOT / "reports/v2/data_stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
