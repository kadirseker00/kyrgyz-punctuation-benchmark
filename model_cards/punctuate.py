"""Restore commas and periods in unpunctuated Kyrgyz text with a KG-Punct-XLMR checkpoint.

Self-contained copy of the benchmark inference path (``kg_punctuation.benchmark_v2.predict_words``):
the same lowercasing/word pattern as the data pipeline, overlapping 126-subword word windows,
each word scored once in the window whose centre is closest, first-subtoken logits.

    python punctuate.py --model kadirseker/kg-punct-xlmr-mixed "бүгүн аба ырайы жакшы болот деп күтүлүүдө"
"""

from __future__ import annotations

import argparse
import re
import sys
from typing import Sequence

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", flags=re.UNICODE)
REPLACEMENTS = {"“": '"', "”": '"', "‘": "'", "’": "'",
                "–": "-", "—": "-", "…": "."}
PUNCT = {"S": "", "C": ",", "P": "."}


def words_from_text(text: str) -> list[str]:
    """Lowercased word tokens, as in the benchmark; any punctuation in the input is discarded."""
    for source, target in REPLACEMENTS.items():
        text = text.replace(source, target)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return WORD_RE.findall(text)


def fit_long_words(tokenizer, words: Sequence[str], budget: int) -> tuple[list[str], list[int]]:
    words = list(words)
    counts = [len(ids) or 1 for ids in tokenizer(words, add_special_tokens=False)["input_ids"]]
    for i, count in enumerate(counts):
        if count <= budget:
            continue
        word, lo, hi = words[i], 1, len(words[i])
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if len(tokenizer([word[:mid]], add_special_tokens=False)["input_ids"][0]) <= budget:
                lo = mid
            else:
                hi = mid - 1
        words[i] = word[:lo]
        counts[i] = len(tokenizer([words[i]], add_special_tokens=False)["input_ids"][0]) or 1
    return words, counts


def word_windows(counts: Sequence[int], budget: int) -> list[tuple[int, int]]:
    windows, start, n = [], 0, len(counts)
    while n:
        end, used = start, 0
        while end < n and used + counts[end] <= budget:
            used += counts[end]
            end += 1
        windows.append((start, end))
        if end >= n:
            break
        start = max(start + 1, start + (end - start) // 2)
    return windows


def owner_windows(windows: Sequence[tuple[int, int]], n_words: int) -> list[int]:
    owner, best = [-1] * n_words, [float("inf")] * n_words
    for w, (start, end) in enumerate(windows):
        centre = (start + end - 1) / 2
        for i in range(start, end):
            if abs(i - centre) < best[i]:
                best[i], owner[i] = abs(i - centre), w
    return owner


@torch.no_grad()
def predict_labels(model, tokenizer, words: Sequence[str], max_length: int = 128,
                   batch_size: int = 16) -> list[str]:
    """One label (S/C/P) per word; long inputs are covered by overlapping windows."""
    if not words:
        return []
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    budget = max_length - tokenizer.num_special_tokens_to_add(pair=False)
    fitted, counts = fit_long_words(tokenizer, words, budget)
    windows = word_windows(counts, budget)
    owner = owner_windows(windows, len(fitted))
    window_labels: list[list[str]] = []
    device = next(model.parameters()).device
    for i in range(0, len(windows), batch_size):
        batch = windows[i:i + batch_size]
        enc = tokenizer([fitted[s:e] for s, e in batch], is_split_into_words=True,
                        truncation=False, padding=True, return_tensors="pt")
        predicted = model(**{k: v.to(device) for k, v in enc.items()}).logits.argmax(-1).cpu().tolist()
        for b, (s, e) in enumerate(batch):
            first: dict[int, int] = {}
            for position, wid in enumerate(enc.word_ids(b)):
                if wid is not None and wid not in first:
                    first[wid] = position
            window_labels.append([id2label[predicted[b][first[wid]]] for wid in range(e - s)])
    return [window_labels[w][i - windows[w][0]] for i, w in enumerate(owner)]


def restore(model, tokenizer, text: str) -> str:
    words = words_from_text(text)
    labels = predict_labels(model, tokenizer, words)
    return " ".join(word + PUNCT[label] for word, label in zip(words, labels))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="kadirseker/kg-punct-xlmr-mixed")
    parser.add_argument("text", nargs="*", help="input text (default: read stdin)")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(args.model).eval()
    text = " ".join(args.text) if args.text else sys.stdin.read()
    print(restore(model, tokenizer, text))


if __name__ == "__main__":
    main()
