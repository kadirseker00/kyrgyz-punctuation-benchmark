"""Sequential, restartable audit queue; reserves one process at a time for MPS."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from kg_punctuation.benchmark_v2 import DEFAULT_MANIFEST, ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/v2_audits")
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(root)


def execute(root):
    manifest_hash = hashlib.sha256(DEFAULT_MANIFEST.read_bytes()).hexdigest()
    status_path = root / "queue_status.json"
    state = json.loads(status_path.read_text()) if status_path.exists() else {
        "manifest_sha256": manifest_hash, "jobs": {},
    }
    if state["manifest_sha256"] != manifest_hash:
        raise ValueError("Manifest changed since this queue was created.")
    state.update(pid=os.getpid(), status="running")

    def save():
        state["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
        temporary.replace(status_path)

    def completed_run(job, cross=False):
        paths = list((root / job).glob("*/metadata.json"))
        valid = []
        for path in paths:
            metadata = json.loads(path.read_text())
            predictions = path.parent / "tables/predictions.csv"
            if metadata.get("benchmark", {}).get("manifest_sha256") != manifest_hash or not predictions.exists():
                continue
            if not cross and not (path.parent / "models/hf_model/config.json").exists():
                continue
            valid.append(path.parent)
        if len(valid) != 1:
            raise ValueError(f"Expected one completed run for {job}, got {valid}")
        return valid[0]

    jobs = [
        ("extended_news", ["train", "--source", "kyrgyznews", "--view", "extended", "--train-documents", "3200"]),
        ("extended_wiki", ["train", "--source", "wikipedia_ky", "--view", "extended", "--train-documents", "3200"]),
        ("cross_news_to_wiki", ["cross", "--source", "kyrgyznews", "--target", "wikipedia_ky", "--view", "extended"]),
        ("cross_wiki_to_news", ["cross", "--source", "wikipedia_ky", "--target", "kyrgyznews", "--view", "extended"]),
        ("learning_curve", ["curve", "--source", "kyrgyznews", "--view", "core", "--budgets", "250", "1000", "2500", "3200", "5000", "10000", "13000"]),
    ]
    save()
    for job, flags in jobs:
        if state["jobs"].get(job, {}).get("status") == "complete":
            print(f"QUEUE SKIP {job}", flush=True)
            continue
        try:
            if job.startswith("cross_"):
                dependency = "extended_news" if job == "cross_news_to_wiki" else "extended_wiki"
                flags += ["--model-dir", str(completed_run(dependency) / "models/hf_model")]
            command = [sys.executable, "-u", str(ROOT / "scripts/run_v2_audits.py"), *flags,
                       "--output", str(root / job), "--batch-size", "8"]
            state["jobs"][job] = {"status": "running", "command": command,
                                  "started_at": time.strftime("%Y-%m-%d %H:%M:%S %z")}
            state["current_job"] = job
            save()
            print(f"QUEUE START {job}", flush=True)
            with (root / f"{job}.log").open("a") as log:
                child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                state["jobs"][job]["pid"] = child.pid
                save()
                code = child.wait()
            if code:
                raise RuntimeError(f"{job} exited {code}; see {root / (job + '.log')}")
            if job != "learning_curve":
                state["jobs"][job]["run_dir"] = str(completed_run(job, cross=job.startswith("cross_")))
            state["jobs"][job].update(status="complete", finished_at=time.strftime("%Y-%m-%d %H:%M:%S %z"))
            print(f"QUEUE DONE {job}", flush=True)
        except Exception as error:
            state["jobs"].setdefault(job, {}).update(status="failed", error=str(error))
            print(f"QUEUE FAIL {job}: {error}", flush=True)
        save()
    state["status"] = "complete" if all(j["status"] == "complete" for j in state["jobs"].values()) else "needs_attention"
    state["current_job"] = None
    save()
    print(f"QUEUE FINISHED {state['status']}", flush=True)


if __name__ == "__main__":
    main()
