# Multi-seed stability and cluster uncertainty

- Labels: S, C, P
- Bootstrap samples: 2000
- Bootstrap unit: group (1243 clusters)
- Test: 1250 documents, 443185 tokens
- Ordered test SHA-256: cb963bd1282a5fb2572f1b0b5e36b8ea5a03f4ec3105a37d9f3b8bedc14575ed

Intervals quantify test-cluster sampling uncertainty conditional on each trained model. Across-seed standard deviation separately measures training variability. Pairwise intervals are unadjusted descriptive comparisons. Matching prediction IDs does not independently establish manifest coverage or identical training settings; these must be audited in the runner.

| Seed | Macro-F1 | 95% cluster bootstrap CI | Accuracy |
|---:|---:|---:|---:|
| 42 | 0.828070 | [0.819149, 0.835646] | 0.908318 |
| 43 | 0.827458 | [0.818534, 0.834947] | 0.909277 |
| 44 | 0.827975 | [0.819295, 0.835465] | 0.908061 |

Mean macro-F1: 0.827834
Sample standard deviation: 0.000329

## Pairwise comparisons against the first seed

| Reference seed | Comparison seed | Delta | Paired cluster bootstrap CI |
|---:|---:|---:|---:|
| 42 | 43 | 0.000612 | [-0.001648, 0.002781] |
| 42 | 44 | 0.000095 | [-0.001872, 0.002043] |
