# Train→test leakage audit

Split: `shuffle(seed=42)[:N] -> train_test_split 0.35 / (25/35), random_state=42`; sentences ≥ 5 tokens, period-delimited on gold labels.

## Corpus exact duplicates

| File | Documents | Exact duplicate documents |
|---|---|---|
| kyrgyznews | 5000 | 479 |
| wikipedia_ky | 5000 | 1 |
| kyrgyznews_20k | 20000 | 759 |

## Benchmark splits

| Setting | Test docs identical to a train doc | Test sentences seen in train | Test tokens in seen sentences |
|---|---|---|---|
| kyrgyznews@1000 | 8/250 (3.2%) | 312/2118 (14.73%) | 12.45% |
| kyrgyznews@5000 | 145/1250 (11.6%) | 4181/11216 (37.28%) | 33.25% |
| wikipedia_ky@1000 | 0/250 (0.0%) | 272/5564 (4.89%) | 2.29% |
| wikipedia_ky@5000 | 0/1250 (0.0%) | 2086/30843 (6.76%) | 3.37% |

## Learning curve (kyrgyznews_20k, fixed test split)

| Train budget | Test docs identical to a train doc | Test sentences seen in train | Test tokens in seen sentences |
|---|---|---|---|
| 250 | 22/5000 (0.44%) | 848/57433 (1.48%) | 1.44% |
| 1000 | 46/5000 (0.92%) | 2589/57433 (4.51%) | 3.95% |
| 2500 | 71/5000 (1.42%) | 4527/57433 (7.88%) | 7.03% |
| 5000 | 114/5000 (2.28%) | 7248/57433 (12.62%) | 11.43% |
| 10000 | 186/5000 (3.72%) | 10727/57433 (18.68%) | 17.18% |
| 13000 | 224/5000 (4.48%) | 12167/57433 (21.18%) | 19.44% |
