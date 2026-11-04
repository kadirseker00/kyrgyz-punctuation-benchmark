"""Publish the three KG-Punct-XLMR checkpoints to the Hugging Face Hub and verify them.

Each repository gets the four model files from experiments/v2_main/<job>/hf_model plus the model card
and punctuate.py from model_cards/<repo>/, in one commit. Afterwards the Hub's LFS SHA-256 of every
weight file and the bytes of every small file are compared with the local copies, and the result is
written to reports/r3/hf_release.json.

    .venv/bin/python scripts/upload_checkpoints.py [--namespace kadirseker]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
JOBS = {"xlmr_news": "kg-punct-xlmr-news", "xlmr_wiki": "kg-punct-xlmr-wiki", "xlmr_mixed": "kg-punct-xlmr-mixed"}
MODEL_FILES = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]
CARD_FILES = ["README.md", "punctuate.py"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def local_files(job: str, repo: str) -> dict[str, Path]:
    files = {name: ROOT / f"experiments/v2_main/{job}/hf_model/{name}" for name in MODEL_FILES}
    files.update({name: ROOT / f"model_cards/{repo}/{name}" for name in CARD_FILES})
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", default="kadirseker")
    args = parser.parse_args()
    api = HfApi()
    user = api.whoami()["name"]
    if user != args.namespace:
        raise SystemExit(f"logged in as {user!r}, not {args.namespace!r}; rerun with --namespace {user}")

    inventory = {Path(e["local_path"]).parent.name: {f["name"]: f["sha256"] for f in e["files"]}
                 for e in json.loads((ROOT / "paper/R3/reproducibility/checkpoint_inventory.json").read_text())}
    report = {"date": time.strftime("%Y-%m-%d %H:%M:%S %z"), "user": user, "repositories": []}

    for job, repo in JOBS.items():
        repo_id = f"{args.namespace}/{repo}"
        files = local_files(job, repo)
        hashes = {name: sha256(path) for name, path in files.items()}
        for name in MODEL_FILES:  # weights must be the inventoried, paper-scored files
            if hashes[name] != inventory[job][name]:
                raise SystemExit(f"{job}/{name}: local hash differs from checkpoint_inventory.json")

        print(f"[{repo_id}] creating repository and uploading ...", flush=True)
        api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
        commit = api.create_commit(
            repo_id, repo_type="model",
            operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path)) for name, path in files.items()],
            commit_message="Release KG-Punct-XLMR checkpoint (canonical v2 benchmark, seed 42)",
        )

        info = api.model_info(repo_id, revision=commit.oid, files_metadata=True)
        remote = {s.rfilename: s for s in info.siblings}
        checks = {}
        for name, path in files.items():
            sibling = remote.get(name)
            if sibling is None:
                checks[name] = "missing"
            elif sibling.lfs is not None:
                checks[name] = "ok" if sibling.lfs.sha256 == hashes[name] else "sha256 mismatch"
            else:
                downloaded = Path(hf_hub_download(repo_id, name, revision=commit.oid))
                checks[name] = "ok" if sha256(downloaded) == hashes[name] else "content mismatch"
        status = "PASS" if all(v == "ok" for v in checks.values()) else "FAIL"
        print(f"[{repo_id}] commit {commit.oid} · verification {status} {checks}", flush=True)
        report["repositories"].append({"repo_id": repo_id, "url": f"https://huggingface.co/{repo_id}",
                                       "commit": commit.oid, "private": info.private, "status": status,
                                       "sha256": hashes, "checks": checks})

    out = ROOT / "reports/r3/hf_release.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}")
    if any(r["status"] != "PASS" for r in report["repositories"]):
        raise SystemExit("verification failed")


if __name__ == "__main__":
    main()
