"""Build the Hugging Face model cards for the three released KG-Punct-XLMR checkpoints.

Every number is read from the archived registries, never typed by hand:
  reports/v2/main_panel.csv                    full-test scores (main panel)
  reports/r3/overlap/main_panel_sensitivity.csv fixed overlap-mask scores (T17)
  reports/r3/asr/comparison.json               read-speech ASR scores of the mixed checkpoint (T19)
  experiments/v2_main/<job>/metadata.json       training configuration and epoch history

    python scripts/build_model_cards.py [--namespace kadirseker] [--code-url URL]
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model_cards"

MODELS = {
    "xlmr_news": ("kg-punct-xlmr-news", "KG-Punct-XLMR-News", "kyrgyznews",
                  "3,200 Kyrgyz News training documents"),
    "xlmr_wiki": ("kg-punct-xlmr-wiki", "KG-Punct-XLMR-Wiki", "wikipedia_ky",
                  "3,200 Kyrgyz Wikipedia training documents"),
    "xlmr_mixed": ("kg-punct-xlmr-mixed", "KG-Punct-XLMR-Mixed", "mixed",
                   "3,200 Kyrgyz News + 3,200 Kyrgyz Wikipedia training documents"),
}
SOURCE_NAME = {"kyrgyznews": "Kyrgyz News", "wikipedia_ky": "Kyrgyz Wikipedia"}
TITLE = "Automatic Punctuation Restoration for Kyrgyz: A Two-Source Benchmark and Evaluation Study"


def pct(x: float) -> str:
    return f"{100 * float(x):.2f}"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def setting(train: str, test: str) -> str:
    if train == "mixed":
        return "mixed-source"
    return "in-source" if train == test else "cross-source transfer"


def card(job: str, namespace: str, code_url: str | None) -> str:
    repo, name, train, train_desc = MODELS[job]
    meta = json.loads((ROOT / f"experiments/v2_main/{job}/metadata.json").read_text())
    panel = {r["test"]: r for r in rows(ROOT / "reports/v2/main_panel.csv") if r["job"] == job}
    masked = {(r["test"], r["subset"]): r for r in rows(ROOT / "reports/r3/overlap/main_panel_sensitivity.csv")
              if r["job"] == job}
    assert set(panel) == set(SOURCE_NAME), job

    hp = meta["hyperparameters"]
    history = meta["history"]
    best = max(history, key=lambda h: h["valid_macro_f1"])
    assert abs(best["valid_macro_f1"] - meta["best_valid_macro_f1"]) < 1e-6

    index = []
    for test in SOURCE_NAME:
        r = panel[test]
        index.append(f"""  - task:
      type: token-classification
      name: Automatic Punctuation Restoration
    dataset:
      name: {SOURCE_NAME[test]} test split (canonical v2)
      type: self-supervised
    metrics:
    - type: f1
      name: macro-F1 (S/C/P)
      value: {pct(r['macro_f1'])}
    - type: accuracy
      value: {pct(r['accuracy'])}""")

    table = ["| Test source | Setting | Words | Accuracy | Macro-P | Macro-R | Macro-F1 | Comma F1 | Period F1 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for test in SOURCE_NAME:
        r = panel[test]
        table.append(f"| {SOURCE_NAME[test]} | {setting(train, test)} | {int(r['words']):,} | {pct(r['accuracy'])} | "
                     f"{pct(r['macro_precision'])} | {pct(r['macro_recall'])} | **{pct(r['macro_f1'])}** | "
                     f"{pct(r['C_f1'])} | {pct(r['P_f1'])} |")
    overlap = ["| Test source | Full test macro-F1 | Overlap-masked macro-F1 | Words kept |",
               "|---|---:|---:|---:|"]
    for test in SOURCE_NAME:
        full, kept = masked[(test, "all")], masked[(test, "filtered")]
        overlap.append(f"| {SOURCE_NAME[test]} | {pct(full['macro_f1'])} | {pct(kept['macro_f1'])} | {int(kept['words']):,} |")

    epochs = ", ".join(f"epoch {h['epoch']}: {pct(h['valid_macro_f1'])}" for h in history)
    weights = ", ".join(f"`{k}`={v:.3f}" for k, v in meta["class_weights"].items())
    selection = (f"The released weights are the epoch with the best validation macro-F1 "
                 f"(epoch {best['epoch']}; per-epoch validation macro-F1 — {epochs}).")
    if job == "xlmr_mixed":
        selection += (" Training became unstable after the first epoch in this run (validation macro-F1 fell "
                      "sharply in epochs 2–3); the released checkpoint is the epoch-1 model selected on "
                      "validation, and all reported test scores are for exactly these weights.")

    asr = ""
    if job == "xlmr_mixed":
        comp = json.loads((ROOT / "reports/r3/asr/comparison.json").read_text())["datasets"]
        cv, fl = comp["common_voice"], comp["fleurs"]
        asr = f"""
### Read-speech ASR transcripts

This checkpoint was also applied to wav2vec2 ASR output for the complete Common Voice 21.0 Kyrgyz test
set ({cv['clips']:,} clips) and the FLEURS `ky_kg` test set ({fl['clips']:,} clips); both are **read speech**.
Comma/period event macro-F1 over a full minimum-edit alignment (recognition errors included):

| Test set | Reference transcript | ASR transcript (aligned) | ASR, exact-word anchors only |
|---|---:|---:|---:|
| Common Voice 21.0 | {pct(cv['metrics']['oracle']['punctuation_macro_f1'])} | {pct(cv['metrics']['aligned']['punctuation_macro_f1'])} | {pct(cv['metrics']['lexically_anchored']['punctuation_macro_f1'])} |
| FLEURS | {pct(fl['metrics']['oracle']['punctuation_macro_f1'])} | {pct(fl['metrics']['aligned']['punctuation_macro_f1'])} | {pct(fl['metrics']['lexically_anchored']['punctuation_macro_f1'])} |

These numbers use a different (event-level, C/P only) metric from the text benchmark above and are not
directly comparable to it. Spontaneous, conversational and long-form speech were **not** evaluated.
"""

    code = (f"Code, canonical split manifests and the evaluation protocol: <{code_url}>." if code_url else
            "The benchmark code, canonical split manifests and evaluation protocol accompany the paper's "
            "reproduction package.")

    return f"""---
language:
- ky
license: cc-by-nc-4.0
base_model: FacebookAI/xlm-roberta-base
library_name: transformers
pipeline_tag: token-classification
tags:
- kyrgyz
- punctuation-restoration
- token-classification
- xlm-roberta
- low-resource
metrics:
- f1
- accuracy
model-index:
- name: {name}
  results:
{chr(10).join(index)}
---

# {name}

{name} is `xlm-roberta-base` fully fine-tuned for **Kyrgyz automatic punctuation restoration** (APR)
on {train_desc}. It is one of the checkpoints evaluated in *{TITLE}*. It is a standard XLM-R
token-classification fine-tune — not a new architecture; the paper's contribution is the benchmark and
the controlled evaluation.

Given lowercased Kyrgyz words without punctuation, the model predicts, for each word, which mark follows it:

| Label | Meaning |
|---|---|
| `S` | no comma or period after the word |
| `C` | comma |
| `P` | sentence end (period; question and exclamation marks are mapped to `P` in this core label set) |

## Usage

Long texts must be split into overlapping windows exactly as in the benchmark; `punctuate.py` in this
repository does this (it is a self-contained copy of the benchmark inference code):

```bash
pip install torch transformers
python punctuate.py --model {namespace}/{repo} "бүгүн аба ырайы жылуу болот бирок кечинде жамгыр жаашы мүмкүн"
```

```python
from transformers import AutoModelForTokenClassification, AutoTokenizer
from punctuate import restore

tokenizer = AutoTokenizer.from_pretrained("{namespace}/{repo}")
model = AutoModelForTokenClassification.from_pretrained("{namespace}/{repo}").eval()
print(restore(model, tokenizer, "бүгүн аба ырайы жылуу болот бирок кечинде жамгыр жаашы мүмкүн"))
```

Input is lowercased and any existing punctuation is discarded; output is lowercased (capitalization is not
restored). Each word is labelled from the logits of its first sub-token, in the window whose centre is
closest to it (126 sub-tokens per window).

## Evaluation

Canonical v2 benchmark: document-level splits after near-duplicate grouping, shared test sets for all models,
seed 42. Scores in %; macro averages over `S`/`C`/`P`.

{chr(10).join(table)}

**Residual-overlap check.** Removing every test word inside a complete sentence (≥5 words) that also occurs
in either source's main training subset gives one fixed mask per test source for all models:

{chr(10).join(overlap)}
{asr}
## Training

| | |
|---|---|
| Base model | `xlm-roberta-base` |
| Training data | {train_desc} ({meta['train_windows']:,} training windows) |
| Epochs / batch size / learning rate | {hp['epochs']} / {hp['batch_size']} / {hp['learning_rate']:g} |
| Max length | {hp['max_length']} sub-tokens, overlapping word windows, each word labelled once |
| Loss | class-weighted cross-entropy ({weights}) |
| Seed | {meta['seed']} |
| Model selection | validation macro-F1 |

{selection}

## Training data and license

- **Kyrgyz News Corpus** — the-cramer-project/Kyrgyz_News_Corpus (`KG_news_256k.csv`, revision
  `a7e5ac9c28b86f07c1ad4db1d9c095deb03e46c9`), CC-BY-NC-4.0.
- **Kyrgyz Wikipedia** — kywiki pages-articles dump, CC-BY-SA-4.0 with page-specific notices.

Labels are derived automatically from the punctuation already present in these texts. The weights are released
under **CC-BY-NC-4.0** for research and other non-commercial use, following the most restrictive source
license; users are responsible for compliance with the source terms. The base model `xlm-roberta-base` is MIT
licensed. {code}

## Limitations

- Trained and tested on two written sources (Wikipedia and news); robustness is established only between
  these two sources, not for spoken, conversational or other genres.
- Only commas and sentence ends; question and exclamation marks are not distinguished, colons, semicolons,
  quotes and dashes are not restored.
- Evaluation uses a single reference punctuation; some disagreements with the reference may be acceptable
  alternatives.
- Scores are from a single training run (seed 42).

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", default="kadirseker")
    parser.add_argument("--code-url")
    args = parser.parse_args()
    for job, (repo, *_rest) in MODELS.items():
        target = OUT / repo
        target.mkdir(parents=True, exist_ok=True)
        (target / "README.md").write_text(card(job, args.namespace, args.code_url), encoding="utf-8")
        shutil.copyfile(OUT / "punctuate.py", target / "punctuate.py")
        print(target / "README.md")


if __name__ == "__main__":
    main()
