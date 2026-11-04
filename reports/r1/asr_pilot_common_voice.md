# Common Voice Kyrgyz ASR-Domain APR Pilot

- Clips: 128
- Reference tokens: 868
- ASR model: `kyrgyz-ai/Wav2vec-Kyrgyz`
- APR model: `KG-Punct-XLMR-Mixed` (experiments/mixed_xlm-roberta-base_20260707_140845_f4e9d371/models/hf_model)
- ASR word error rate: 32.26%
- Matched token coverage for ASR-aligned punctuation scoring: 69.93%
- Runtime seconds: 18.041

## Punctuation metrics

| Condition | Accuracy | Macro-F1 | Weighted-F1 | Comma F1 | Period F1 | Evaluated tokens |
|---|---:|---:|---:|---:|---:|---:|
| Oracle Common Voice transcript | 91.36 | 69.86 | 91.24 | 34.62 | 79.82 | 868 |
| ASR output, exact-word aligned | 89.62 | 65.33 | 89.80 | 25.00 | 76.65 | 607 |

## Sample ASR outputs

- `common_voice_ky_17589356.mp3` (1/8 WER edits, 7 matched words)
  - Reference: Эртең дагы башка бирөө бул маселе менен алектенет.
  - ASR: эртең дагы башка бирөө бул маселе менен айлектенет
- `common_voice_ky_18391692.mp3` (2/4 WER edits, 3 matched words)
  - Reference: Бардыгы мурдагыдай эле болот.
  - ASR: бардыгы мурдагы да эле болот
- `common_voice_ky_17936429.mp3` (3/5 WER edits, 2 matched words)
  - Reference: Нурболоттун энесин таба албай койдук.
  - ASR: нырболоддун энесин таба албайкпайт н
- `common_voice_ky_17454364.mp3` (3/8 WER edits, 5 matched words)
  - Reference: Бул чырдын арты менен Темир Сариев отставкага кеткен.
  - ASR: юобул черден арты менен темир сариев аставкага кеткен
- `common_voice_ky_18160877.mp3` (3/6 WER edits, 3 matched words)
  - Reference: Бирок ошол сол тарабы иштебей жатат.
  - ASR: биролк ошол сот тарабы иштибей жатат
- `common_voice_ky_19920799.mp3` (4/5 WER edits, 1 matched words)
  - Reference: Чет өлкөлөрдө бул модага айланган.
  - ASR: чет өркөлөрдү булзмо дага айыланган
- `common_voice_ky_17830306.mp3` (0/7 WER edits, 7 matched words)
  - Reference: Жыйынтыгында сот коргоочу тараптын өтүнүчүн четке какты.
  - ASR: жыйынтыгында сот коргоочу тараптын өтүнүчүн четке какты
- `common_voice_ky_17830308.mp3` (0/4 WER edits, 4 matched words)
  - Reference: Эч ким жок, көрдүңөрбү?
  - ASR: эч ким жок көрдүңөрбү
- `common_voice_ky_19571750.mp3` (0/8 WER edits, 8 matched words)
  - Reference: Ал биз менен келип таанышкан биринчи киши болду.
  - ASR: ал биз менен келип таанышкан биринчи киши болду
- `common_voice_ky_19571751.mp3` (1/9 WER edits, 8 matched words)
  - Reference: Мен юрист катары бул жеке аныктаманы мыйзамсыз деп эсептейм.
  - ASR: мен юрист катары бул жеге аныктаманы мыйзамсыз деп эсептейм
