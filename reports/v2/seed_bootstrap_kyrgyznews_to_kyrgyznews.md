# Multi-seed stability and cluster uncertainty

- Labels: S, C, P
- Bootstrap samples: 2000
- Bootstrap unit: group (1152 clusters)
- Test: 1250 documents, 220839 tokens
- Ordered test SHA-256: 98e93a60e635e74c64758114dfbea06fa8b6b5bd52cca35311e641c6698c3dc6

Intervals quantify test-cluster sampling uncertainty conditional on each trained model. Across-seed standard deviation separately measures training variability. Pairwise intervals are unadjusted descriptive comparisons. Matching prediction IDs does not independently establish manifest coverage or identical training settings; these must be audited in the runner.

| Seed | Macro-F1 | 95% cluster bootstrap CI | Accuracy |
|---:|---:|---:|---:|
| 42 | 0.864388 | [0.855019, 0.872679] | 0.954324 |
| 43 | 0.863670 | [0.855181, 0.871712] | 0.953726 |
| 44 | 0.847067 | [0.838572, 0.854881] | 0.944761 |

Mean macro-F1: 0.858375
Sample standard deviation: 0.009800

## Pairwise comparisons against the first seed

| Reference seed | Comparison seed | Delta | Paired cluster bootstrap CI |
|---:|---:|---:|---:|
| 42 | 43 | 0.000718 | [-0.001684, 0.003159] |
| 42 | 44 | 0.017321 | [0.014609, 0.019979] |
