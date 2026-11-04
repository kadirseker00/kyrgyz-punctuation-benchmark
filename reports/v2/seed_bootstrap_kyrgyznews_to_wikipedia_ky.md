# Multi-seed stability and cluster uncertainty

- Labels: S, C, P
- Bootstrap samples: 2000
- Bootstrap unit: group (1243 clusters)
- Test: 1250 documents, 443185 tokens
- Ordered test SHA-256: cb963bd1282a5fb2572f1b0b5e36b8ea5a03f4ec3105a37d9f3b8bedc14575ed

Intervals quantify test-cluster sampling uncertainty conditional on each trained model. Across-seed standard deviation separately measures training variability. Pairwise intervals are unadjusted descriptive comparisons. Matching prediction IDs does not independently establish manifest coverage or identical training settings; these must be audited in the runner.

| Seed | Macro-F1 | 95% cluster bootstrap CI | Accuracy |
|---:|---:|---:|---:|
| 42 | 0.705093 | [0.692258, 0.715717] | 0.834444 |
| 43 | 0.708848 | [0.697367, 0.718407] | 0.843923 |
| 44 | 0.693366 | [0.682562, 0.702880] | 0.824042 |

Mean macro-F1: 0.702436
Sample standard deviation: 0.008076

## Pairwise comparisons against the first seed

| Reference seed | Comparison seed | Delta | Paired cluster bootstrap CI |
|---:|---:|---:|---:|
| 42 | 43 | -0.003755 | [-0.007122, -0.000464] |
| 42 | 44 | 0.011727 | [0.008232, 0.014850] |
