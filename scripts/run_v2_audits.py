"""Run the extended-label, transfer, learning-curve and seed audits on the canonical v2 manifest (one GPU job at a time)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from kg_punctuation.benchmark_v2 import Benchmark, DEFAULT_MANIFEST, ROOT
from kg_punctuation.hf_generic_cross_eval import evaluate_generic_cross_domain
from kg_punctuation.hf_generic_experiment import run_hf_experiment
from kg_punctuation.learning_curve import run_learning_curve
from kg_punctuation.seed_analysis import analyze_seed_runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "cross", "curve", "seeds"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source", choices=("kyrgyznews", "wikipedia_ky"), required=True)
    parser.add_argument("--target", choices=("kyrgyznews", "wikipedia_ky"))
    parser.add_argument("--view", choices=("core", "extended"), default="extended")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/v2_audits")
    parser.add_argument("--model-id", default="xlm-roberta-base")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--train-documents", type=int)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--budgets", type=int, nargs="+", default=[250, 1000, 2500, 5000, 10000, 13000])
    args = parser.parse_args()
    benchmark = Benchmark(args.manifest)
    dataset = ROOT / benchmark.manifest["sources"][args.source]["data_file"]
    common = dict(
        manifest_path=args.manifest, source=args.source, label_set=args.view,
        max_length=args.max_length, batch_size=args.batch_size,
    )
    training = dict(model_id=args.model_id, epochs=args.epochs, learning_rate=args.learning_rate)
    if args.command == "train":
        result = run_hf_experiment(
            dataset, args.output, **common, **training, seed=args.seed,
            train_documents=args.train_documents, run_name=f"v2_{args.source}_{args.view}",
        )
    elif args.command == "cross":
        if args.model_dir is None or args.target is None:
            parser.error("cross requires --model-dir and --target")
        target = ROOT / benchmark.manifest["sources"][args.target]["data_file"]
        common.pop("source")
        result = evaluate_generic_cross_domain(
            args.model_dir, target, args.output, args.source, args.target,
            **common, seed=args.seed,
        )
    elif args.command == "curve":
        result = run_learning_curve(
            dataset, args.output, **common, **training, seed=args.seed,
            budgets=tuple(args.budgets),
        )
    else:
        if len(args.seeds) < 3 or len(set(args.seeds)) != len(args.seeds):
            parser.error("seeds requires at least three distinct training seeds")
        runs = []
        for seed in args.seeds:
            run = run_hf_experiment(
                dataset, args.output, **common, **training, seed=seed,
                train_documents=args.train_documents,
                run_name=f"v2_{args.source}_{args.view}_seed_{seed}",
            )
            runs.append(Path(run["output_dir"]))
            print(json.dumps(run), flush=True)
        result = analyze_seed_runs(
            runs, args.output / f"seed_analysis_{args.source}_{args.view}_{runs[0].name}.json",
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
