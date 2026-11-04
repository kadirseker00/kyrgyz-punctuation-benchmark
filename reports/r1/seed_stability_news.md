# Multi-seed stability and significance

- Labels: `S, C, P`
- Bootstrap samples: 2000

| Seed | Macro-F1 | 95% bootstrap CI | Accuracy |
|---:|---:|---:|---:|
| 42 | 0.851409 | [0.841721, 0.860689] | 0.945402 |
| 43 | 0.843716 | [0.833992, 0.853394] | 0.939943 |
| 44 | 0.822193 | [0.811851, 0.831298] | 0.927226 |

Mean macro-F1: 0.839106
Sample standard deviation: 0.015144

## Pairwise comparisons against the first seed

| Reference seed | Comparison seed | Delta | Bootstrap CI | McNemar exact p |
|---:|---:|---:|---:|---:|
| 42 | 43 | 0.007693 | [0.000360, 0.015205] | 0.000649551 |
| 42 | 44 | 0.029216 | [0.021236, 0.036836] | 1.77457e-24 |
