# Rare-label loss and decision-offset comparison

Single-seed exploratory comparisons. All runs use the same canonical documents, schedule and base-model revision; best epoch selected by five-label validation macro-F1. Offsets selected exclusively on validation, never by test performance. Offsets adjust decisions, not probability calibration. No inference of reliable exclamation restoration or causal effect across all random seeds.

| Source | Condition | Macro-F1 | Q P/R/F1 | Q predicted/support | E P/R/F1 | E predicted/support |
|---|---|---:|---|---|---|---|
| News | Weighted CE | 55.04 | 24.03/88.74/37.82 | 1344/364 | 3.79/28.17/6.68 | 528/71 |
| News | Weighted + offsets | 58.30 | 62.54/50.00/55.57 | 291/364 | 5.80/5.63/5.71 | 69/71 |
| News | Unweighted CE | 71.45 | 86.56/76.10/80.99 | 320/364 | 11.11/2.82/4.49 | 18/71 |
| Wikipedia | Weighted CE | 49.73 | 5.62/86.10/10.55 | 2865/187 | 2.57/32.48/4.76 | 1479/117 |
| Wikipedia | Weighted + offsets | 54.21 | 20.88/60.96/31.11 | 546/187 | 4.15/20.51/6.91 | 578/117 |
| Wikipedia | Unweighted CE | 64.05 | 52.43/57.75/54.96 | 206/187 | 50.00/2.56/4.88 | 6/117 |

Library versions match, but historical weighted runs used macOS 27.0 and new unweighted runs 27.0.1 on arm64. MPS deterministic algorithms were disabled in both. This environment difference and the single seed limit a strict causal interpretation of changing the loss.
