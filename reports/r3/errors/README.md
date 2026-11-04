# Systematic error census

Every canonical core-label test word of the source-specific seed-42 XLM-R checkpoint is included. Matrix rows are reference labels; columns are predictions. Counts indicate disagreement with one source-derived reference, not adjudicated linguistic errors.

| Reference → prediction | News count (% of errors) | Wikipedia count (% of errors) |
|---|---:|---:|
| S → C | 5,598 (55.5%) | 22,115 (54.4%) |
| S → P | 2,282 (22.6%) | 9,242 (22.7%) |
| C → S | 650 (6.4%) | 2,752 (6.8%) |
| C → P | 752 (7.5%) | 2,340 (5.8%) |
| P → S | 221 (2.2%) | 1,481 (3.6%) |
| P → C | 584 (5.8%) | 2,702 (6.6%) |

The separate audit pack has 168 positions, at most 12 per stratum per source, chosen by a fixed hash order. Correct cases are also sampled. Because strata have different sizes, its raw sample proportions must not be used to estimate corpus-wide acceptability.

No native-speaker annotation has been collected. Suggested protocol: two independent fluent Kyrgyz raters inspect the word context and mark all acceptable labels S/C/P before seeing the model/reference columns; use the full document when context is insufficient; adjudicate disagreements; report multi-reference scores separately from the canonical benchmark. Blank rating columns are intentional.
