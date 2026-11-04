# Expanded read-speech evaluation

All source test rows with nonempty reference words are included. No length, punctuation, WER or outcome selection. C/P event macro-F1 excludes S and scores the complete minimum-edit alignment, including deletions (FN) and insertions (FP). Substituted words may retain aligned punctuation boundaries; the stricter lexical-anchor condition additionally requires exact word identity for TP. Neither condition is a native acceptability measure.

| Dataset | Clips | Ref. words | WER | Oracle C/P F1 | ASR aligned C/P F1 | Exact-anchor C/P F1 |
|---|---:|---:|---:|---:|---:|---:|
| Common Voice | 1,615 | 10,910 | 33.26 | 75.08 | 61.36 | 48.52 |
| FLEURS | 977 | 17,303 | 37.69 | 75.52 | 61.81 | 49.63 |

Common Voice source statistics: `{"audio_minutes": 132.99308333333335, "seconds_median": 4.859, "seconds_max": 9.899, "words_median": 7.0, "words_max": 12, "clips_over_12_words": 0, "exact_sentence_overlap_with_APR_main_train_words": 8, "clips_with_exact_sentence_overlap": 1, "speech_type": "read speech"}`. Paired-bootstrap oracle minus ASR interval: `{'samples': 2000, 'seed': 123, 'groups': 148, 'oracle_minus_asr_punctuation_macro_f1_95CI': [0.11372871976252441, 0.1590253595210931]}`.


FLEURS source statistics: `{"audio_minutes": 194.775, "seconds_median": 11.1, "seconds_max": 36.6, "words_median": 16.0, "words_max": 52, "clips_over_12_words": 771, "exact_sentence_overlap_with_APR_main_train_words": 0, "clips_with_exact_sentence_overlap": 0, "speech_type": "read speech"}`. Paired-bootstrap oracle minus ASR interval: `{'samples': 2000, 'seed': 123, 'groups': 350, 'oracle_minus_asr_punctuation_macro_f1_95CI': [0.11555335382591668, 0.15831490478279403]}`.

The intervals resample Common Voice speakers and FLEURS source-sentence IDs (speaker IDs unavailable). WER strata are descriptive: they also differ in content and speakers, so their differences are not a causal dose-response estimate. Both datasets are read speech, not spontaneous dialogue or continuous long-form audio. The ASR checkpoint's original Common Voice release overlap has not been audited. Exact-sentence matches with the APR training pools are counted above; absence of such matches would not rule out pretraining contamination or paraphrases. No listening review, multi-reference annotation, latency or subtitle usability measurement was performed.

Decoding uses the pinned Wav2Vec checkpoint, batches of four, and crops padded CTC logits to each unpadded waveform's acoustic output length. The historical 128-clip pilot used a different padding/alignment policy; its WER and F1 are retained as historical results and not treated as a directly comparable expanded estimate.
