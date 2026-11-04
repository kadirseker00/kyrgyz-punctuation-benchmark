# Main-panel sentence-overlap sensitivity

Completed >=5-word sentences; token equality ignoring punctuation; union of both main 3200-document training subsets; one fixed mask per target for every model.

| Model | News full → fixed | Wikipedia full → fixed |
|---|---:|---:|
| Balanced LR | 74.24 → 73.84 | 66.29 → 66.06 |
| Balanced SGD-LR | 77.12 → 76.83 | 70.59 → 70.34 |
| CNN 2$\times$128 | 71.36 → 70.87 | 67.60 → 67.22 |
| CNN 4$\times$256 | 72.30 → 71.82 | 68.71 → 68.31 |
| KyrgyzBERT | 76.80 → 76.51 | 75.42 → 75.20 |
| KyrgyzBERT2 | 82.60 → 82.23 | 79.55 → 79.25 |
| KG-Punct-XLMR | 86.44 → 86.26 | 82.81 → 82.62 |
| KyrgyzBERT-PR | 44.94 → 45.00 | 40.80 → 40.91 |
| KG-Punct-XLMR-Mixed (3,200+3,200) | 85.09 → 84.95 | 80.03 → 79.93 |
| KG-Punct-XLMR-Mixed (1,600+1,600) | 86.12 → 85.96 | 81.32 → 81.18 |

Main-panel rankings unchanged: {'kyrgyznews': True, 'wikipedia_ky': True}

Post-hoc sensitivity analysis, not retraining or proof of no contamination. Short phrases, paraphrases and unfinished sentences are not excluded. The external checkpoint's training corpus is unknown. Rankings are point estimates.

All 34 full-test rows and their per-class metrics match the archived registry.
