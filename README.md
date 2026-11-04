# Kyrgyz Automatic Punctuation Restoration Benchmark

Code, canonical split, evaluation protocol and result reports for

> Abdulkadir Şeker. *Automatic Punctuation Restoration for Kyrgyz: A Two-Source Benchmark and Evaluation Study*. ACM Transactions on Asian and Low-Resource Language Information Processing
> (under review).

The benchmark turns punctuated Kyrgyz text from two written sources (Kyrgyz Wikipedia and a Kyrgyz news corpus)
into word-level labels — `S` (no mark), `C` (comma), `P` (sentence end; `Q`/`E` kept separate in the extended
audit) — with document-level splits after near-duplicate grouping. Every model writes one prediction per test word,
and the scorer rejects any prediction file whose word identities or gold labels differ from the manifest.

## Released models

| Checkpoint | Training data | Hugging Face |
|---|---|---|
| KG-Punct-XLMR-News | 3,200 news documents | <https://huggingface.co/kadirseker/kg-punct-xlmr-news> |
| KG-Punct-XLMR-Wiki | 3,200 Wikipedia documents | <https://huggingface.co/kadirseker/kg-punct-xlmr-wiki> |
| KG-Punct-XLMR-Mixed | 3,200 + 3,200 documents | <https://huggingface.co/kadirseker/kg-punct-xlmr-mixed> |

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
this repository; they are in the reproduction archive on Zenodo: **DOI to be added**. Extracting that archive at
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
@article{seker2026kgpunct,
  author  = {Abdulkadir {\c{S}}eker},
  title   = {Automatic Punctuation Restoration for Kyrgyz: A Two-Source Benchmark and Evaluation Study},
  journal = {ACM Transactions on Asian and Low-Resource Language Information Processing},
  year    = {2026},
  note    = {Under review}
}
```
