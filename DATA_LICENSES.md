# Data and model licenses

The MIT license in `LICENSE` covers the source code only (`src/`, `tests/`, `scripts/`, `model_cards/*.py`,
`paper/R3/reproducibility/verify_snapshot.py`). Data, derived data and model weights keep their own terms:

| Material | Where | Terms |
|---|---|---|
| Kyrgyz News Corpus (the-cramer-project/Kyrgyz_News_Corpus, revision `a7e5ac9c28b86f07c1ad4db1d9c095deb03e46c9`) and everything derived from its text | Zenodo archive (frozen pool, predictions); text excerpts in `reports/` | CC-BY-NC-4.0 |
| Kyrgyz Wikipedia and everything derived from its text | Zenodo archive; text excerpts in `reports/`; page/revision attribution in `paper/R3/reproducibility/wikipedia_attribution.csv` | CC-BY-SA-4.0 and page-specific notices |
| Common Voice 21.0 Kyrgyz test references | `reports/r1/`, `reports/r3/asr/` | CC0-1.0 |
| FLEURS `ky_kg` test references | `reports/r3/asr/` | CC-BY-4.0 |
| KG-Punct-XLMR-News / -Wiki / -Mixed weights | https://huggingface.co/kadirseker/kg-punct-xlmr-news, -wiki, -mixed | CC-BY-NC-4.0 (base model `xlm-roberta-base`: MIT) |

The canonical split manifest (`data/processed/splits/canonical_v2.json`) contains document identifiers,
duplicate groups and hashes only. KyrgyzBERT2-based checkpoints are not distributed because the base
model's license metadata is `other`.
