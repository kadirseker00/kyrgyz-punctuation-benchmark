# Multi-seed stability and cluster uncertainty

- Labels: S, C, P
- Bootstrap samples: 2000
- Bootstrap unit: group (1152 clusters)
- Test: 1250 documents, 220839 tokens
- Ordered test SHA-256: 98e93a60e635e74c64758114dfbea06fa8b6b5bd52cca35311e641c6698c3dc6

Intervals quantify test-cluster sampling uncertainty conditional on each trained model. Across-seed standard deviation separately measures training variability. Pairwise intervals are unadjusted descriptive comparisons. Matching prediction IDs does not independently establish manifest coverage or identical training settings; these must be audited in the runner.

| Seed | Macro-F1 | 95% cluster bootstrap CI | Accuracy |
|---:|---:|---:|---:|
| 42 | 0.791990 | [0.786416, 0.797535] | 0.921205 |
| 43 | 0.797158 | [0.791813, 0.802458] | 0.925688 |
| 44 | 0.788581 | [0.783290, 0.793912] | 0.920390 |

Mean macro-F1: 0.792576
Sample standard deviation: 0.004318

## Pairwise comparisons against the first seed

| Reference seed | Comparison seed | Delta | Paired cluster bootstrap CI |
|---:|---:|---:|---:|
| 42 | 43 | -0.005167 | [-0.007372, -0.002660] |
| 42 | 44 | 0.003409 | [0.001070, 0.005834] |
