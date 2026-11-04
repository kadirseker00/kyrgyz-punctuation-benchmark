# R3 reproduction candidate

Reproduction notes for *Automatic Punctuation Restoration for Kyrgyz: A Two-Source Benchmark and Evaluation Study*. The code is public under MIT; the frozen text pools and archived predictions are in the Zenodo reproduction archive; the three XLM-R checkpoints are on the Hugging Face Hub (see the repository README).

The candidate ZIP contains a snapshot of the source code, tests, canonical manifest, two frozen processed source pools, archived word predictions, training configurations, and result reports. It excludes raw audio, raw text dumps, model weights, private collaboration history, and credentials. `snapshot_manifest.json` lists every included file's SHA-256. The ZIP includes current working-tree code; its identity is the file manifest, not just the recorded Git commit.

## Environment and verification

The observed environment is Python 3.12.14 on macOS arm64. `requirements-frozen.txt` pins installed packages, including Torch 2.12.1, Transformers 5.13.0, NumPy 2.5.0, scikit-learn 1.9.0 and PyAV 18.1.0. The captured package list is a tested-platform snapshot, not a cross-platform lockfile. MPS runs did not enable deterministic algorithms, so bit-identical retraining is not promised.

Extract at an empty location, change to its root, then run:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r paper/R3/reproducibility/requirements-frozen.txt
.venv/bin/python -m pip install --no-deps -e .
export PYTHONPATH=src
.venv/bin/python paper/R3/reproducibility/verify_snapshot.py
.venv/bin/python -m pytest -q
.venv/bin/python -m kg_punctuation.overlap_sensitivity
.venv/bin/python scripts/analyze_r3_errors.py
.venv/bin/python scripts/report_r3_asr.py
.venv/bin/python scripts/report_r3_rare.py
```

The overlap scorer verifies every prediction identity/gold label and all 34 original registry rows before generating the shared-mask comparison. The error report uses the complete within-source test predictions. The ASR report verifies archived clip counts, punctuation-event counts and WER; re-decoding audio requires the separately obtained source files and checkpoint. Rare-control scores are in `reports/r3/rare`; their report script uses the included original run metadata and the portable status copy. The report does not pretend that retraining took place in the extracted clean environment.

## Data identity and rights

`input_provenance.json` records original and processed hashes. Exact benchmark reconstruction can begin from the two bundled frozen JSONL pools and the bundled manifest. A moving `latest` Wikipedia URL is **not** an exact replacement for the original dump: its date was not retained in the original download record. `wikipedia_attribution.csv` supplies page titles, revision URLs and history URLs for all 5,000 processed Wikipedia documents. The source builder remains available to rebuild and compare hashes when the exact raw files are available.

The original News CSV is available from the [Kyrgyz News Corpus](https://huggingface.co/datasets/the-cramer-project/Kyrgyz_News_Corpus/tree/a7e5ac9c28b86f07c1ad4db1d9c095deb03e46c9) at the recorded revision; its LFS hash matches the local raw file. The upstream card declares **CC-BY-NC-4.0**. The processed News pool and its reference-bearing predictions retain the source attribution and noncommercial restrictions; they are not relicensed as code.

Wikipedia-derived content is subject to the upstream **CC-BY-SA-4.0** terms and page-specific notices. See the [Wikimedia terms](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use) and the attribution file. Reference-bearing Wikipedia predictions remain derived content. Common Voice references/audio are CC0; FLEURS references/audio are CC-BY-4.0. Their provenance and original dataset citations are retained. Dataset licenses are separate from the Python code and from third-party models. No blanket license is assigned by this local preparation step.

The code is released under MIT. Data and derived data keep the source terms above; the released checkpoints are CC-BY-NC-4.0 (see `DATA_LICENSES.md`).

## Retraining

From the frozen pools, retain `data/processed/splits/canonical_v2.json` rather than silently rebuilding a new split. The manifest hashes are checked at load time. For the established main panel and earlier audits:

```bash
export PYTHONPATH=src
.venv/bin/python -m kg_punctuation.main_panel_v2 run-all
.venv/bin/python -m kg_punctuation.v2_report
.venv/bin/python scripts/run_v2_queue.py
.venv/bin/python scripts/analyze_v2_seeds.py
.venv/bin/python scripts/train_r3_unweighted.py
```

These commands are expensive and write new experiment outputs. The historical main-panel runner resolves model names from the Hub and did not retain their commit revisions in its metadata. This is a provenance limitation: no exact historical base revision is invented here, and name-only downloads cannot establish bit-identical reruns. Archived predictions permit exact score checks; the locally inventoried fine-tuned XLM-R files permit checkpoint-based inference when provided. Older configurations and current cache snapshots alone do not establish every historical initialization. New R3 runs explicitly record/pin revisions. The R3 unweighted runner explicitly pins the original XLM-R revision `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`, uses the same ordered 3,200 training IDs, 500 validation and 1,250 test documents per source, and runs seed 42, three epochs, batch 8, learning rate 3e-5. Validation macro-F1 selects each epoch. The weighted and unweighted variants use the same runner, differing in the CE weight vector. This single-seed control is exploratory.

Validation offsets subtract Q/E penalties from the weighted checkpoint's logits. Grid: 0, 0.5, 1, 1.5, 2, 3, 4, 6, 8 for each mark. Five-label validation macro-F1 selects the pair; ties prefer least total correction, then least Q and E correction. The choice is persisted before test reading. This adjusts decisions and does not establish probability calibration. Example:

```bash
.venv/bin/python -m kg_punctuation.rare_calibration \
  --model <weighted-checkpoint-directory> --source kyrgyznews \
  --output experiments/r3/calibration/weighted_news --device cpu
```

`checkpoint_inventory.json` records the local XLM-R checkpoint files and hashes. Weights are **not** in the ZIP; the News, Wiki and Mixed XLM-R checkpoints are on the Hugging Face Hub. Base XLM-R metadata declares MIT, but source-data terms and author release approval must be checked before redistributing derivatives. KyrgyzBERT2 weights are excluded because its license metadata is `other`. Model names in the paper identify checkpoints, not new architectures.

## Expanded speech evaluation

The complete Common Voice 21.0 Kyrgyz test contains 1,615 eligible nonempty references; the pinned FLEURS `ky_kg` test contains 977. Both are **read speech**. FLEURS raw punctuated references are used, not ASR-generated punctuation. Exact source paths, revisions and hashes are recorded in `reports/r3/asr/comparison.json`. Download FLEURS with:

```python
from huggingface_hub import hf_hub_download
revision = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
hf_hub_download("google/fleurs", "data/ky_kg/test.tsv", repo_type="dataset", revision=revision)
hf_hub_download("google/fleurs", "data/ky_kg/audio/test.tar.gz", repo_type="dataset", revision=revision)
```

Copy these into `data/raw/fleurs_ky/`; place Common Voice's `test.tsv` and `ky_test_0.tar` in `data/raw/common_voice/asr_pilot/`. After restoring the mixed APR checkpoint:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_r3_asr.py --dataset common_voice
PYTHONPATH=src .venv/bin/python scripts/run_r3_asr.py --dataset fleurs
PYTHONPATH=src .venv/bin/python scripts/report_r3_asr.py
```

The new C/P event metric excludes S and covers all alignment columns, including recognition errors and gaps. Its scores are not interchangeable with the historical three-label score on exactly recognized words. No speech-model training-set overlap clearance, spontaneous conversation test, human acceptability judgment or subtitle usability claim is made. Search leads and remaining gaps are documented in `reports/r3/data_search/README.md`.
