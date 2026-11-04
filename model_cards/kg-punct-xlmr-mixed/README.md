---
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
- name: KG-Punct-XLMR-Mixed
  results:
  - task:
      type: token-classification
      name: Automatic Punctuation Restoration
    dataset:
      name: Kyrgyz News test split (canonical v2)
      type: self-supervised
    metrics:
    - type: f1
      name: macro-F1 (S/C/P)
      value: 85.09
    - type: accuracy
      value: 94.70
  - task:
      type: token-classification
      name: Automatic Punctuation Restoration
    dataset:
      name: Kyrgyz Wikipedia test split (canonical v2)
      type: self-supervised
    metrics:
    - type: f1
      name: macro-F1 (S/C/P)
      value: 80.03
    - type: accuracy
      value: 88.64
---

# KG-Punct-XLMR-Mixed

KG-Punct-XLMR-Mixed is `xlm-roberta-base` fully fine-tuned for **Kyrgyz automatic punctuation restoration** (APR)
on 3,200 Kyrgyz News + 3,200 Kyrgyz Wikipedia training documents. It is one of the checkpoints evaluated in *Automatic Punctuation Restoration for Kyrgyz: A Two-Source Benchmark and Evaluation Study*. It is a standard XLM-R
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
python punctuate.py --model kadirseker/kg-punct-xlmr-mixed "бүгүн аба ырайы жылуу болот бирок кечинде жамгыр жаашы мүмкүн"
```

```python
from transformers import AutoModelForTokenClassification, AutoTokenizer
from punctuate import restore

tokenizer = AutoTokenizer.from_pretrained("kadirseker/kg-punct-xlmr-mixed")
model = AutoModelForTokenClassification.from_pretrained("kadirseker/kg-punct-xlmr-mixed").eval()
print(restore(model, tokenizer, "бүгүн аба ырайы жылуу болот бирок кечинде жамгыр жаашы мүмкүн"))
```

Input is lowercased and any existing punctuation is discarded; output is lowercased (capitalization is not
restored). Each word is labelled from the logits of its first sub-token, in the window whose centre is
closest to it (126 sub-tokens per window).

## Evaluation

Canonical v2 benchmark: document-level splits after near-duplicate grouping, shared test sets for all models,
seed 42. Scores in %; macro averages over `S`/`C`/`P`.

| Test source | Setting | Words | Accuracy | Macro-P | Macro-R | Macro-F1 | Comma F1 | Period F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Kyrgyz News | mixed-source | 220,839 | 94.70 | 79.70 | 93.12 | **85.09** | 69.62 | 88.43 |
| Kyrgyz Wikipedia | mixed-source | 443,185 | 88.64 | 74.78 | 89.19 | **80.03** | 64.31 | 82.48 |

**Residual-overlap check.** Removing every test word inside a complete sentence (≥5 words) that also occurs
in either source's main training subset gives one fixed mask per test source for all models:

| Test source | Full test macro-F1 | Overlap-masked macro-F1 | Words kept |
|---|---:|---:|---:|
| Kyrgyz News | 85.09 | 84.95 | 212,034 |
| Kyrgyz Wikipedia | 80.03 | 79.93 | 432,943 |

### Read-speech ASR transcripts

This checkpoint was also applied to wav2vec2 ASR output for the complete Common Voice 21.0 Kyrgyz test
set (1,615 clips) and the FLEURS `ky_kg` test set (977 clips); both are **read speech**.
Comma/period event macro-F1 over a full minimum-edit alignment (recognition errors included):

| Test set | Reference transcript | ASR transcript (aligned) | ASR, exact-word anchors only |
|---|---:|---:|---:|
| Common Voice 21.0 | 75.08 | 61.36 | 48.52 |
| FLEURS | 75.52 | 61.81 | 49.63 |

These numbers use a different (event-level, C/P only) metric from the text benchmark above and are not
directly comparable to it. Spontaneous, conversational and long-form speech were **not** evaluated.

## Training

| | |
|---|---|
| Base model | `xlm-roberta-base` |
| Training data | 3,200 Kyrgyz News + 3,200 Kyrgyz Wikipedia training documents (45,895 training windows) |
| Epochs / batch size / learning rate | 3 / 8 / 3e-05 |
| Max length | 128 sub-tokens, overlapping word windows, each word labelled once |
| Loss | class-weighted cross-entropy (`S`=0.406, `C`=4.555, `P`=3.125) |
| Seed | 42 |
| Model selection | validation macro-F1 |

The released weights are the epoch with the best validation macro-F1 (epoch 1; per-epoch validation macro-F1 — epoch 1: 82.38, epoch 2: 6.59, epoch 3: 30.06). Training became unstable after the first epoch in this run (validation macro-F1 fell sharply in epochs 2–3); the released checkpoint is the epoch-1 model selected on validation, and all reported test scores are for exactly these weights.

## Training data and license

- **Kyrgyz News Corpus** — the-cramer-project/Kyrgyz_News_Corpus (`KG_news_256k.csv`, revision
  `a7e5ac9c28b86f07c1ad4db1d9c095deb03e46c9`), CC-BY-NC-4.0.
- **Kyrgyz Wikipedia** — kywiki pages-articles dump, CC-BY-SA-4.0 with page-specific notices.

Labels are derived automatically from the punctuation already present in these texts. The weights are released
under **CC-BY-NC-4.0** for research and other non-commercial use, following the most restrictive source
license; users are responsible for compliance with the source terms. The base model `xlm-roberta-base` is MIT
licensed. The benchmark code, canonical split manifests and evaluation protocol accompany the paper's reproduction package.

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
@article{seker2026kgpunct,
  author  = {Abdulkadir {\c{S}}eker},
  title   = {Automatic Punctuation Restoration for Kyrgyz: A Two-Source Benchmark and Evaluation Study},
  journal = {ACM Transactions on Asian and Low-Resource Language Information Processing},
  year    = {2026},
  note    = {Under review}
}
```
