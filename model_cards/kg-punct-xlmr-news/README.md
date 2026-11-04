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
- name: KG-Punct-XLMR-News
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
      value: 86.44
    - type: accuracy
      value: 95.43
  - task:
      type: token-classification
      name: Automatic Punctuation Restoration
    dataset:
      name: Kyrgyz Wikipedia test split (canonical v2)
      type: self-supervised
    metrics:
    - type: f1
      name: macro-F1 (S/C/P)
      value: 70.51
    - type: accuracy
      value: 83.44
---

# KG-Punct-XLMR-News

KG-Punct-XLMR-News is `xlm-roberta-base` fully fine-tuned for **Kyrgyz automatic punctuation restoration** (APR)
on 3,200 Kyrgyz News training documents. It is one of the checkpoints evaluated in *Automatic Punctuation Restoration for Kyrgyz: A Two-Source Benchmark and Evaluation Study*. It is a standard XLM-R
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
python punctuate.py --model kadirseker/kg-punct-xlmr-news "бүгүн аба ырайы жылуу болот бирок кечинде жамгыр жаашы мүмкүн"
```

```python
from transformers import AutoModelForTokenClassification, AutoTokenizer
from punctuate import restore

tokenizer = AutoTokenizer.from_pretrained("kadirseker/kg-punct-xlmr-news")
model = AutoModelForTokenClassification.from_pretrained("kadirseker/kg-punct-xlmr-news").eval()
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
| Kyrgyz News | in-source | 220,839 | 95.43 | 81.75 | 92.91 | **86.44** | 72.13 | 89.50 |
| Kyrgyz Wikipedia | cross-source transfer | 443,185 | 83.44 | 68.09 | 76.22 | **70.51** | 51.66 | 69.48 |

**Residual-overlap check.** Removing every test word inside a complete sentence (≥5 words) that also occurs
in either source's main training subset gives one fixed mask per test source for all models:

| Test source | Full test macro-F1 | Overlap-masked macro-F1 | Words kept |
|---|---:|---:|---:|
| Kyrgyz News | 86.44 | 86.26 | 212,034 |
| Kyrgyz Wikipedia | 70.51 | 70.87 | 432,943 |

## Training

| | |
|---|---|
| Base model | `xlm-roberta-base` |
| Training data | 3,200 Kyrgyz News training documents (14,164 training windows) |
| Epochs / batch size / learning rate | 3 / 8 / 3e-05 |
| Max length | 128 sub-tokens, overlapping word windows, each word labelled once |
| Loss | class-weighted cross-entropy (`S`=0.382, `C`=6.413, `P`=4.376) |
| Seed | 42 |
| Model selection | validation macro-F1 |

The released weights are the epoch with the best validation macro-F1 (epoch 2; per-epoch validation macro-F1 — epoch 1: 84.57, epoch 2: 86.88, epoch 3: 86.62).

## Training data and license

- **Kyrgyz News Corpus** — the-cramer-project/Kyrgyz_News_Corpus (`KG_news_256k.csv`, revision
  `a7e5ac9c28b86f07c1ad4db1d9c095deb03e46c9`), CC-BY-NC-4.0.
- **Kyrgyz Wikipedia** — kywiki pages-articles dump, CC-BY-SA-4.0 with page-specific notices.

Labels are derived automatically from the punctuation already present in these texts. The weights are released
under **CC-BY-NC-4.0** for research and other non-commercial use, following the most restrictive source
license; users are responsible for compliance with the source terms. The base model `xlm-roberta-base` is MIT
licensed. Code, canonical split manifest and evaluation protocol: <https://github.com/kadirseker00/kyrgyz-punctuation-benchmark> (MIT). Frozen text pools and archived predictions: <https://doi.org/10.5281/zenodo.23959428>.

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
