"""Train the two matched unweighted-loss controls, one accelerator job at a time."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time

from kg_punctuation.benchmark_v2 import Benchmark, DEFAULT_MANIFEST, ROOT
from kg_punctuation.hf_generic_experiment import run_hf_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/r3/rare_unweighted")
    parser.add_argument("--sources", nargs="+", choices=("kyrgyznews", "wikipedia_ky"),
                        default=["kyrgyznews", "wikipedia_ky"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        benchmark = Benchmark(DEFAULT_MANIFEST)
        status_path = args.output / "status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {"jobs": {}}
        code = ROOT / "src/kg_punctuation/hf_generic_experiment.py"
        for source in args.sources:
            job = status["jobs"].get(source, {})
            if job.get("state") == "complete":
                meta = json.loads((Path(job["run_dir"]) / "metadata.json").read_text())
                assert meta["hyperparameters"]["loss_weighting"] == "none"
                assert meta["benchmark"]["manifest_sha256"] == hashlib.sha256(DEFAULT_MANIFEST.read_bytes()).hexdigest()
                assert meta["benchmark"]["train_document_ids"] == benchmark.split_ids(source, "train")[:3200]
                print(f"SKIP complete {source}", flush=True)
                continue
            domain = "news" if source == "kyrgyznews" else "wiki"
            baselines = list((ROOT / f"experiments/v2_audits/extended_{domain}").glob("*/metadata.json"))
            if len(baselines) != 1:
                raise ValueError(f"Expected exactly one weighted baseline for {source}")
            baseline = json.loads(baselines[0].read_text())
            revision = baseline["model_revision"]
            status["jobs"][source] = {
                "state": "running", "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "baseline_metadata": str(baselines[0].relative_to(ROOT)),
                "model_revision": revision, "runner_sha256": hashlib.sha256(code.read_bytes()).hexdigest(),
                "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            }
            status_path.write_text(json.dumps(status, indent=2) + "\n")
            try:
                run = run_hf_experiment(
                    ROOT / benchmark.manifest["sources"][source]["data_file"], args.output,
                    model_id="xlm-roberta-base", max_length=128, batch_size=8, epochs=3,
                    learning_rate=3e-5, seed=42, label_set="extended",
                    manifest_path=DEFAULT_MANIFEST, source=source, train_documents=3200,
                    loss_weighting="none", revision=revision, run_name=f"r3_{source}_unweighted",
                )
            except BaseException as error:
                status["jobs"][source].update(state="failed", error=repr(error))
                status_path.write_text(json.dumps(status, indent=2) + "\n")
                raise
            status["jobs"][source].update(state="complete", run_dir=run["output_dir"],
                                          finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
            status_path.write_text(json.dumps(status, indent=2) + "\n")


if __name__ == "__main__":
    main()
