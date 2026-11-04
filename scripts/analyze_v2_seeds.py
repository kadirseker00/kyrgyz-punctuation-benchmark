"""Analyze the main panel's original seed-42 row and its seed-43/44 reruns."""
from pathlib import Path
import json

from kg_punctuation.benchmark_v2 import ROOT
from kg_punctuation.seed_analysis import analyze_seed_runs


def main():
    for training_source, suffix in (("kyrgyznews", "news"), ("wikipedia_ky", "wiki")):
        runs = [ROOT / "experiments/v2_main" / f"xlmr_{suffix}{tag}" for tag in ("", "_s43", "_s44")]
        for target in ("kyrgyznews", "wikipedia_ky"):
            output = ROOT / "reports/v2" / f"seed_bootstrap_{training_source}_to_{target}.json"
            result = analyze_seed_runs(runs, output, prediction_source=target)
            print(json.dumps({"training_source": training_source, "target": target,
                              "summary": result["summary"], "clusters": result["bootstrap_cluster_count"]}), flush=True)


if __name__ == "__main__":
    main()
