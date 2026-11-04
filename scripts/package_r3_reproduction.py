"""Build a local, reviewable reproduction snapshot; performs no upload or publication."""
from __future__ import annotations

import argparse
import bz2
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import zipfile

from kg_punctuation.benchmark_v2 import Benchmark, DEFAULT_MANIFEST, ROOT


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "paper/R3/reproducibility")
    args = parser.parse_args(); output = args.output; output.mkdir(parents=True, exist_ok=True)
    benchmark = Benchmark(DEFAULT_MANIFEST)
    state_path = ROOT / "experiments/r3/rare_unweighted/status.json"
    state = json.loads(state_path.read_text())
    for source in ("kyrgyznews", "wikipedia_ky"):
        if state["jobs"].get(source, {}).get("state") != "complete":
            raise ValueError(f"The reproduction candidate requires a completed unweighted control: {source}")
    if not (ROOT / "reports/r3/rare/comparison.json").is_file():
        raise ValueError("Generate and verify the rare-control comparison before packaging")
    sources = []
    raw_paths = {"kyrgyznews": ROOT / "data/raw/kyrgyznews/KG_news_256k.csv",
                 "wikipedia_ky": ROOT / "data/raw/wikipedia_ky/kywiki-latest-pages-articles.xml.bz2"}
    for source, spec in benchmark.manifest["sources"].items():
        data = ROOT / spec["data_file"]
        sources.append({"source": source, "processed_path": spec["data_file"],
                        "processed_sha256": sha(data), "raw_path": str(raw_paths[source].relative_to(ROOT)),
                        "raw_bytes": raw_paths[source].stat().st_size, "raw_sha256": sha(raw_paths[source])})
    sources[0].update(upstream="https://huggingface.co/datasets/the-cramer-project/Kyrgyz_News_Corpus",
                      upstream_revision="a7e5ac9c28b86f07c1ad4db1d9c095deb03e46c9",
                      upstream_file="KG_news_256k.csv", upstream_license="CC-BY-NC-4.0",
                      upstream_lfs_sha256="8cfef5b387084921a9609081714138084dfddc187e1a0a5468ad85e8c2bac828")
    assert sources[0]["raw_sha256"] == sources[0]["upstream_lfs_sha256"]
    sources[1].update(upstream="https://dumps.wikimedia.org/kywiki/",
                      upstream_snapshot_date="Not retained in original download provenance; do not infer from filename or modification time",
                      reconstruction="Use the bundled frozen processed pool for exact benchmark identity; a new latest dump is not an exact substitute",
                      upstream_license="CC-BY-SA-4.0, subject to page-specific notices; attribution file included",
                      upstream_terms="https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use")
    (output / "input_provenance.json").write_text(json.dumps({"sources": sources}, indent=2) + "\n")

    # Preserve page/revision links for the complete frozen Wikipedia pool.
    wiki = ROOT / benchmark.manifest["sources"]["wikipedia_ky"]["data_file"]
    ids = {json.loads(line)["document_id"] for line in wiki.open()}
    attribution = []
    with bz2.open(raw_paths["wikipedia_ky"], "rb") as handle:
        for _, element in ET.iterparse(handle, events=("end",)):
            if not element.tag.endswith("}page"): continue
            ns = element.tag.rsplit("}", 1)[0] + "}"
            page_id = element.findtext(ns + "id")
            if f"wiki_{page_id}" in ids:
                revision = element.find(ns + "revision")
                rid = revision.findtext(ns + "id")
                attribution.append({"document_id": f"wiki_{page_id}", "title": element.findtext(ns + "title"),
                                    "revision_id": rid, "revision_timestamp": revision.findtext(ns + "timestamp"),
                                    "revision_url": f"https://ky.wikipedia.org/w/index.php?oldid={rid}",
                                    "history_url": f"https://ky.wikipedia.org/w/index.php?curid={page_id}&action=history"})
            element.clear()
    assert {r["document_id"] for r in attribution} == ids
    with (output / "wikipedia_attribution.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(attribution[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(attribution)

    checkpoints = []
    paths = [ROOT / f"experiments/v2_main/{job}/hf_model" for job in ("xlmr_news", "xlmr_wiki", "xlmr_mixed", "xlmr_mixed_matched")]
    paths += [p.parent / "models/hf_model" for d in ("extended_news", "extended_wiki")
              for p in (ROOT / "experiments/v2_audits" / d).glob("*/metadata.json")]
    paths += [p.parent / "models/hf_model" for p in (ROOT / "experiments/r3/rare_unweighted").glob("*/metadata.json")]
    for path in paths:
        if not path.is_dir(): continue
        files = [{"name": f.name, "bytes": f.stat().st_size, "sha256": sha(f)} for f in sorted(path.iterdir()) if f.is_file()]
        checkpoints.append({"local_path": str(path.relative_to(ROOT)), "files": files,
                            "base_model": "FacebookAI/xlm-roberta-base", "base_license": "MIT",
                            "release_status": "Local only; weights are not included in the code/data archive. Author must confirm downstream data obligations and publication before redistribution."})
    (output / "checkpoint_inventory.json").write_text(json.dumps(checkpoints, indent=2) + "\n")
    metadata_paths = list((ROOT / "experiments/v2_main").glob("*/metadata.json"))
    metadata_paths += list((ROOT / "experiments/v2_audits").glob("*/*/metadata.json"))
    metadata_paths += list((ROOT / "experiments/r3/rare_unweighted").glob("*/metadata.json"))
    configurations = []
    for path in sorted(metadata_paths):
        # Preserve the original hash while making the copied configuration portable.
        portable = path.read_text().replace(str(ROOT), ".")
        configurations.append({"original_path": str(path.relative_to(ROOT)), "original_sha256": sha(path),
                               "configuration": json.loads(portable)})
    (output / "training_configurations.json").write_text(json.dumps(configurations, indent=2) + "\n")
    portable_state = json.loads(state_path.read_text().replace(str(ROOT) + "/", ""))
    (output / "portable_rare_status.json").write_text(json.dumps(portable_state, indent=2) + "\n")
    freeze = subprocess.check_output([str(ROOT / ".venv/bin/python"), "-m", "pip", "freeze"], text=True)
    (output / "requirements-frozen.txt").write_text(freeze)

    paths = [ROOT / "pyproject.toml", DEFAULT_MANIFEST]
    for directory, pattern in (("src", "*.py"), ("scripts", "*.py"), ("tests", "*.py")):
        paths += list((ROOT / directory).rglob(pattern))
    for directory in ("reports/r1", "reports/v2", "reports/r3"):
        paths += [p for p in (ROOT / directory).rglob("*") if p.is_file() and p.suffix in (".json", ".jsonl", ".csv", ".md", ".gz")]
    paths += [ROOT / spec["data_file"] for spec in benchmark.manifest["sources"].values()]
    paths += metadata_paths
    paths += list((ROOT / "experiments/v2_main").glob("*/predictions_test_*.csv"))
    paths += list((ROOT / "experiments/v2_audits").glob("*/*/tables/predictions.csv"))
    paths += list((ROOT / "experiments/r3/rare_unweighted").glob("*/tables/predictions.csv"))
    paths += list((ROOT / "experiments/r3/calibration").glob("*/predictions_*.csv"))
    paths += list((ROOT / "experiments/r3/calibration").glob("*/validation_selection.json"))
    paths += list((ROOT / "experiments/r3/calibration").glob("*/validation_logits.npz"))
    paths += list((ROOT / "experiments/r3/calibration").glob("*/metrics.json"))
    paths += [p for p in output.iterdir() if p.is_file() and p.suffix in (".md", ".json", ".csv", ".txt", ".py", ".sh") and p.name != "snapshot_manifest.json"]
    paths = sorted(set(paths))
    file_sources = {str(p.relative_to(ROOT)): p for p in paths}
    # setuptools reads the root README; use the R3 instructions in the release
    # without modifying the user's independently edited development README.
    file_sources["README.md"] = output / "README.md"
    manifest = {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "includes_working_tree_snapshot": True,
                "release_status": "LOCAL REVIEW CANDIDATE — not published, no stable public URL assigned",
                "files": [{"path": name, "bytes": p.stat().st_size, "sha256": sha(p)} for name, p in sorted(file_sources.items())]}
    (output / "snapshot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    file_sources[str((output / "snapshot_manifest.json").relative_to(ROOT))] = output / "snapshot_manifest.json"
    archive = output / "KG_punctuation_R3_reproduction_candidate.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for name, path in sorted(file_sources.items()): bundle.write(path, name)
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        for entry in manifest["files"]:
            assert hashlib.sha256(bundle.read(entry["path"])).hexdigest() == entry["sha256"]
    print(json.dumps({"archive": str(archive), "bytes": archive.stat().st_size, "sha256": sha(archive),
                      "files": len(file_sources), "wikipedia_attribution_rows": len(attribution), "checkpoint_inventories": len(checkpoints)}, indent=2))


if __name__ == "__main__": main()
