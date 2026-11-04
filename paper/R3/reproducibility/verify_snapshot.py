"""Verify the extracted candidate against its file manifest, then load both pools."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
manifest = json.loads((Path(__file__).parent / "snapshot_manifest.json").read_text())
for entry in manifest["files"]:
    path = ROOT / entry["path"]
    assert path.is_file(), f"Missing file: {entry['path']}"
    assert path.stat().st_size == entry["bytes"], f"Wrong length: {entry['path']}"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], f"Wrong hash: {entry['path']}"
from kg_punctuation.benchmark_v2 import Benchmark, DEFAULT_MANIFEST
benchmark = Benchmark(DEFAULT_MANIFEST)
for source in benchmark.manifest["sources"]:
    assert len(benchmark.documents(source, "train", "extended", limit=3200)) == 3200
    assert len(benchmark.documents(source, "validation", "extended")) == 500
    assert len(benchmark.documents(source, "test", "extended")) == 1250
print(f"Verified {len(manifest['files'])} files and both canonical source pools.")
