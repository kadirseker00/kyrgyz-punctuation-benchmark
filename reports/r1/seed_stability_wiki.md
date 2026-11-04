# Multi-seed stability and significance

- Labels: `S, C, P`
- Bootstrap samples: 2000

| Seed | Macro-F1 | 95% bootstrap CI | Accuracy |
|---:|---:|---:|---:|
| 42 | 0.796782 | [0.788005, 0.805568] | 0.895384 |
| 43 | 0.802327 | [0.793291, 0.811164] | 0.902107 |
| 44 | 0.761427 | [0.752657, 0.770136] | 0.864594 |

Mean macro-F1: 0.786845
Sample standard deviation: 0.022187

## Pairwise comparisons against the first seed

| Reference seed | Comparison seed | Delta | Bootstrap CI | McNemar exact p |
|---:|---:|---:|---:|---:|
| 42 | 43 | -0.005545 | [-0.012251, 0.001203] | 0.000218914 |
| 42 | 44 | 0.035355 | [0.028598, 0.042502] | 2.44255e-49 |
