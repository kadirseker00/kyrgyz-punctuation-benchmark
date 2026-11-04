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
- name: KG-Punct-XLMR-Wiki
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
      value: 79.20
    - type: accuracy
      value: 92.12
  - task:
      type: token-classification
      name: Automatic Punctuation Restoration
    dataset:
      name: Kyrgyz Wikipedia test split (canonical v2)
      type: self-supervised
    metrics:
    - type: f1
      name: macro-F1 (S/C/P)
      value: 82.81
    - type: accuracy
      value: 90.83
---

# KG-Punct-XLMR-Wiki

KG-Punct-XLMR-Wiki is `xlm-roberta-base` fully fine-tuned for **Kyrgyz automatic punctuation restoration** (APR)
on 3,200 Kyrgyz Wikipedia training documents. It is one of the checkpoints evaluated in *Automatic Punctuation Restoration for Kyrgyz: A Two-Source Benchmark and Evaluation Study*. It is a standard XLM-R
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
python punctuate.py --model kadirseker/kg-punct-xlmr-wiki "бүгүн аба ырайы жылуу болот бирок кечинде жамгыр жаашы мүмкүн"
```

```python
from transformers import AutoModelForTokenClassification, AutoTokenizer
from punctuate import restore

tokenizer = AutoTokenizer.from_pretrained("kadirseker/kg-punct-xlmr-wiki")
model = AutoModelForTokenClassification.from_pretrained("kadirseker/kg-punct-xlmr-wiki").eval()
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
| Kyrgyz News | cross-source transfer | 220,839 | 92.12 | 73.36 | 89.06 | **79.20** | 63.11 | 78.57 |
| Kyrgyz Wikipedia | in-source | 443,185 | 90.83 | 78.36 | 89.73 | **82.81** | 67.98 | 85.65 |

**Residual-overlap check.** Removing every test word inside a complete sentence (≥5 words) that also occurs
in either source's main training subset gives one fixed mask per test source for all models:

| Test source | Full test macro-F1 | Overlap-masked macro-F1 | Words kept |
|---|---:|---:|---:|
| Kyrgyz News | 79.20 | 79.02 | 212,034 |
| Kyrgyz Wikipedia | 82.81 | 82.62 | 432,943 |

## Training

| | |
|---|---|
| Base model | `xlm-roberta-base` |
| Training data | 3,200 Kyrgyz Wikipedia training documents (31,731 training windows) |
| Epochs / batch size / learning rate | 3 / 8 / 3e-05 |
| Max length | 128 sub-tokens, overlapping word windows, each word labelled once |
| Loss | class-weighted cross-entropy (`S`=0.419, `C`=3.989, `P`=2.741) |
| Seed | 42 |
| Model selection | validation macro-F1 |

The released weights are the epoch with the best validation macro-F1 (epoch 3; per-epoch validation macro-F1 — epoch 1: 81.05, epoch 2: 82.33, epoch 3: 83.64).

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
