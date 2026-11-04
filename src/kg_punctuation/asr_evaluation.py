"""Punctuation-event scoring over a full minimum-edit word alignment, including gaps."""
from __future__ import annotations

from collections import Counter

MARKS = ("C", "P")


def align_words(reference, hypothesis):
    """Levenshtein alignment; ties prefer substitution, then deletion, then insertion."""
    n, m = len(reference), len(hypothesis)
    cost = [list(range(m + 1))] + [[i] + [0] * m for i in range(1, n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i][j] = min(cost[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]),
                             cost[i - 1][j] + 1, cost[i][j - 1] + 1)
    pairs, i, j = [], n, m
    while i or j:
        if i and j and reference[i - 1] == hypothesis[j - 1] and cost[i][j] == cost[i - 1][j - 1]:
            pairs.append((i - 1, j - 1)); i -= 1; j -= 1
        elif i and j and cost[i][j] == cost[i - 1][j - 1] + 1:
            pairs.append((i - 1, j - 1)); i -= 1; j -= 1
        elif i and cost[i][j] == cost[i - 1][j] + 1:
            pairs.append((i - 1, None)); i -= 1
        else:
            pairs.append((None, j - 1)); j -= 1
    return list(reversed(pairs)), cost[n][m]


def event_counts(reference_words, reference_labels, hypothesis_words, prediction_labels,
                 require_exact_anchor=False):
    if len(reference_words) != len(reference_labels) or len(hypothesis_words) != len(prediction_labels):
        raise ValueError("Every input word must have exactly one punctuation label")
    if any(l not in ("S", "C", "P") for l in reference_labels + prediction_labels):
        raise ValueError("This evaluation uses the core S/C/P inventory")
    pairs, edits = align_words(reference_words, hypothesis_words)
    counts = Counter()
    for i, j in pairs:
        gold = reference_labels[i] if i is not None else None
        prediction = prediction_labels[j] if j is not None else None
        anchor = i is not None and j is not None and reference_words[i] == hypothesis_words[j]
        match = gold in MARKS and gold == prediction and (anchor or not require_exact_anchor)
        if match:
            counts[gold + "_tp"] += 1
        else:
            if gold in MARKS: counts[gold + "_fn"] += 1
            if prediction in MARKS: counts[prediction + "_fp"] += 1
        if anchor: counts["matched_words"] += 1
    counts.update(reference_words=len(reference_words), hypothesis_words=len(hypothesis_words), word_errors=edits)
    return counts


def event_metrics(counts):
    per_class = {}
    for label in MARKS:
        tp, fp, fn = (counts[label + suffix] for suffix in ("_tp", "_fp", "_fn"))
        per_class[label] = {"precision": tp / (tp + fp) if tp + fp else 0.,
                            "recall": tp / (tp + fn) if tp + fn else 0.,
                            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
                            "support": tp + fn, "predicted": tp + fp}
    tp = sum(counts[l + "_tp"] for l in MARKS)
    fp = sum(counts[l + "_fp"] for l in MARKS)
    fn = sum(counts[l + "_fn"] for l in MARKS)
    return {"punctuation_macro_f1": sum(x["f1"] for x in per_class.values()) / 2,
            "punctuation_micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
            "per_class": per_class, "reference_words": counts["reference_words"],
            "hypothesis_words": counts["hypothesis_words"], "matched_words": counts["matched_words"],
            "word_errors": counts["word_errors"],
            "WER": counts["word_errors"] / counts["reference_words"] if counts["reference_words"] else None}
