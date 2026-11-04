"""Assemble the clean public code repository for the Kyrgyz APR benchmark.

Source of truth is the hash-verified reproduction candidate ZIP. The public repository takes its
code, tests, scripts, canonical split manifest, reports and reproducibility metadata, and leaves out the
two frozen text pools and the archived predictions (`data/processed/v2/`, `experiments/`), which are
archived separately on Zenodo. Review correspondence, the collaboration log, assistant instructions,
manuscript sources and git history are not part of the ZIP and therefore never reach the public tree.
Added on top: MIT license for the code, a data/model license notice, the Hugging Face model cards and
the release scripts.

    python scripts/build_public_repo.py [--output release/kyrgyz-punctuation-benchmark]
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ZIP = ROOT / "paper/R3/reproducibility/KG_punctuation_R3_reproduction_candidate.zip"
EXCLUDE_PREFIXES = ("experiments/", "data/processed/v2/")
EXCLUDE_FILES = {"reports/r3/response_quote_audit.json"}  # concerns the journal response, not the benchmark
EXTRA = ["model_cards/punctuate.py", "model_cards/kg-punct-xlmr-news", "model_cards/kg-punct-xlmr-wiki",
         "model_cards/kg-punct-xlmr-mixed", "scripts/build_model_cards.py", "scripts/upload_checkpoints.py",
         "scripts/build_public_repo.py"]
FORBIDDEN = ("COLLAB", "CLAUDE.md", "AGENTS.md", "decision", "desicion", "Response_from", "R3reviews",
             "cover_letter", "response_round", ".tex", ".bib")

HF = "https://huggingface.co/kadirseker"
TITLE = "Automatic Punctuation Restoration for Kyrgyz: A Two-Source Benchmark and Evaluation Study"

MIT = """MIT License

Copyright (c) 2026 Abdulkadir Şeker

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

DATA_LICENSES = f"""# Data and model licenses

The MIT license in `LICENSE` covers the source code only (`src/`, `tests/`, `scripts/`, `model_cards/*.py`,
`paper/R3/reproducibility/verify_snapshot.py`). Data, derived data and model weights keep their own terms:

| Material | Where | Terms |
|---|---|---|
| Kyrgyz News Corpus (the-cramer-project/Kyrgyz_News_Corpus, revision `a7e5ac9c28b86f07c1ad4db1d9c095deb03e46c9`) and everything derived from its text | Zenodo archive (frozen pool, predictions); text excerpts in `reports/` | CC-BY-NC-4.0 |
| Kyrgyz Wikipedia and everything derived from its text | Zenodo archive; text excerpts in `reports/`; page/revision attribution in `paper/R3/reproducibility/wikipedia_attribution.csv` | CC-BY-SA-4.0 and page-specific notices |
| Common Voice 21.0 Kyrgyz test references | `reports/r1/`, `reports/r3/asr/` | CC0-1.0 |
| FLEURS `ky_kg` test references | `reports/r3/asr/` | CC-BY-4.0 |
| KG-Punct-XLMR-News / -Wiki / -Mixed weights | {HF}/kg-punct-xlmr-news, -wiki, -mixed | CC-BY-NC-4.0 (base model `xlm-roberta-base`: MIT) |

The canonical split manifest (`data/processed/splits/canonical_v2.json`) contains document identifiers,
duplicate groups and hashes only. KyrgyzBERT2-based checkpoints are not distributed because the base
model's license metadata is `other`.
"""

README = f"""# Kyrgyz Automatic Punctuation Restoration Benchmark

Code, canonical split, evaluation protocol and result reports for

> Abdulkadir Şeker. *{TITLE}*. ACM Transactions on Asian and Low-Resource Language Information Processing
> (under review).

The benchmark turns punctuated Kyrgyz text from two written sources (Kyrgyz Wikipedia and a Kyrgyz news corpus)
into word-level labels — `S` (no mark), `C` (comma), `P` (sentence end; `Q`/`E` kept separate in the extended
audit) — with document-level splits after near-duplicate grouping. Every model writes one prediction per test word,
and the scorer rejects any prediction file whose word identities or gold labels differ from the manifest.

## Released models

| Checkpoint | Training data | Hugging Face |
|---|---|---|
| KG-Punct-XLMR-News | 3,200 news documents | <{HF}/kg-punct-xlmr-news> |
| KG-Punct-XLMR-Wiki | 3,200 Wikipedia documents | <{HF}/kg-punct-xlmr-wiki> |
| KG-Punct-XLMR-Mixed | 3,200 + 3,200 documents | <{HF}/kg-punct-xlmr-mixed> |

These are standard `xlm-roberta-base` token-classification fine-tunes, the exact weights scored in the paper's main
table. `model_cards/punctuate.py` restores punctuation with the benchmark's windowed inference:

```bash
pip install torch transformers
python model_cards/punctuate.py --model kadirseker/kg-punct-xlmr-mixed "бүгүн аба ырайы жылуу болот бирок кечинде жамгыр жаашы мүмкүн"
```

## Repository contents

| Path | Contents |
|---|---|
| `src/kg_punctuation/` | data building and labeling, canonical v2 benchmark, models, scoring, overlap/rare-label/ASR analyses |
| `tests/` | unit and integration tests |
| `scripts/` | experiment runners and report generators |
| `data/processed/splits/canonical_v2.json` | canonical split manifest (ordered document IDs, duplicate groups, source hashes) |
| `reports/` | every reported number: main panel, seeds and bootstrap, learning curve, extended labels, overlap sensitivity, rare-label controls, ASR, error census |
| `paper/R3/reproducibility/` | provenance, Wikipedia attribution, frozen requirements, training configurations, checkpoint inventory, snapshot verifier |
| `model_cards/` | Hugging Face model cards and the inference script |

The frozen text pools (`data/processed/v2/*.jsonl`) and all archived word predictions (`experiments/`) are not in
this repository; they are in the reproduction archive on Zenodo, <https://doi.org/10.5281/zenodo.23959428>. Extracting that archive at
the repository root restores the full snapshot, which `paper/R3/reproducibility/verify_snapshot.py` checks file by
file.

## Setup and checks

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r paper/R3/reproducibility/requirements-frozen.txt
.venv/bin/python -m pip install --no-deps -e .
PYTHONPATH=src .venv/bin/python -m pytest -q
```

With the Zenodo archive extracted, the reported tables are regenerated from the archived predictions with:

```bash
export PYTHONPATH=src
.venv/bin/python paper/R3/reproducibility/verify_snapshot.py
.venv/bin/python -m kg_punctuation.overlap_sensitivity
.venv/bin/python scripts/analyze_r3_errors.py
.venv/bin/python scripts/report_r3_asr.py
.venv/bin/python scripts/report_r3_rare.py
```

`paper/R3/reproducibility/README.md` documents provenance, retraining commands, the speech evaluation and the known
limitations (the original Wikipedia dump date and the main panel's base-model revisions were not recorded, so
bit-identical retraining of the historical panel is not claimed).

## Licenses

Code: MIT (`LICENSE`). Data, derived data and model weights keep their source terms — News CC-BY-NC-4.0, Wikipedia
CC-BY-SA-4.0, released checkpoints CC-BY-NC-4.0; see `DATA_LICENSES.md`.

## Citation

```bibtex
@article{{seker2026kgpunct,
  author  = {{Abdulkadir {{\\c{{S}}}}eker}},
  title   = {{{TITLE}}},
  journal = {{ACM Transactions on Asian and Low-Resource Language Information Processing}},
  year    = {{2026}},
  note    = {{Under review}}
}}
```
"""

GITIGNORE = """.venv/
__pycache__/
*.egg-info/
.pytest_cache/
data/raw/
data/processed/v2/
experiments/
models/
.claude/
"""

# The packaged README predates the release; its two status paragraphs are replaced, nothing else changes.
OLD_INTRO = ("This is a **local review package** for *" + TITLE + "*. It has not been published. The private "
             "development repository and this local archive do not yet satisfy the reviewers' request for stable "
             "public access.")
NEW_INTRO = ("Reproduction notes for *" + TITLE + "*. The code is public under MIT; the frozen text pools and archived "
             "predictions are in the Zenodo reproduction archive; the three XLM-R checkpoints are on the Hugging Face "
             "Hub (see the repository README).")
OLD_RELEASE = ("Before public release, the author must choose a code license and confirm the scope of the data release. "
               "No release has been approved or uploaded by this script. The historical cancellation of HF publication "
               "is preserved.")
NEW_RELEASE = ("The code is released under MIT. Data and derived data keep the source terms above; the released "
               "checkpoints are CC-BY-NC-4.0 (see `DATA_LICENSES.md`).")
OLD_WEIGHTS = "Weights are **not** in the ZIP."
NEW_WEIGHTS = "Weights are **not** in the ZIP; the News, Wiki and Mixed XLM-R checkpoints are on the Hugging Face Hub."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "release/kyrgyz-punctuation-benchmark")
    args = parser.parse_args()
    out: Path = args.output
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; choose a new output directory")
    out.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ZIP) as archive:
        for name in archive.namelist():
            if name.endswith("/") or name.startswith(EXCLUDE_PREFIXES) or name in EXCLUDE_FILES:
                continue
            target = out / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
    for rel in EXTRA:
        source = ROOT / rel
        if source.is_dir():
            shutil.copytree(source, out / rel)
        else:
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, out / rel)

    notes = out / "paper/R3/reproducibility/README.md"
    text = notes.read_text(encoding="utf-8")
    for old, new in [(OLD_INTRO, NEW_INTRO), (OLD_RELEASE, NEW_RELEASE), (OLD_WEIGHTS, NEW_WEIGHTS)]:
        if text.count(old) != 1:
            raise SystemExit(f"expected passage not found once in {notes}: {old[:60]}...")
        text = text.replace(old, new)
    notes.write_text(text, encoding="utf-8")

    (out / "README.md").write_text(README, encoding="utf-8")
    (out / "LICENSE").write_text(MIT, encoding="utf-8")
    (out / "DATA_LICENSES.md").write_text(DATA_LICENSES, encoding="utf-8")
    (out / ".gitignore").write_text(GITIGNORE, encoding="utf-8")

    files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    leaked = [f for f in files if any(token in f for token in FORBIDDEN)]
    if leaked:
        raise SystemExit(f"forbidden files in public tree: {leaked}")
    print(f"{len(files)} files, {sum((out / f).stat().st_size for f in files) / 1e6:.1f} MB -> {out}")


if __name__ == "__main__":
    main()
