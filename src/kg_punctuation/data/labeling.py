from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


LABEL_SPACE = "S"
LABEL_COMMA = "C"
LABEL_PERIOD = "P"
LABEL_QUESTION = "Q"
LABEL_EXCLAMATION = "E"
LABEL_COLON = "COLON"
LABEL_SEMICOLON = "SEMICOLON"

CORE_LABELS = (LABEL_SPACE, LABEL_COMMA, LABEL_PERIOD)
EXTENDED_LABELS = (
    LABEL_SPACE,
    LABEL_COMMA,
    LABEL_PERIOD,
    LABEL_QUESTION,
    LABEL_EXCLAMATION,
)
ALL_LABELS = EXTENDED_LABELS + (LABEL_COLON, LABEL_SEMICOLON)
LABEL_SET_ALIASES = {
    "core": CORE_LABELS,
    "extended": EXTENDED_LABELS,
    "all": ALL_LABELS,
}

WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", flags=re.UNICODE)


@dataclass(frozen=True)
class LabeledToken:
    token: str
    label: str


def normalize_text(text: str, lowercase: bool = True) -> str:
    """Normalize common punctuation variants while preserving comma and period."""
    replacements = {
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": ".",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower() if lowercase else text


def resolve_label_set(label_set: str | Iterable[str] | None = None) -> tuple[str, ...]:
    """Resolve a named or explicit label set while preserving label order."""
    if label_set is None:
        return CORE_LABELS
    if isinstance(label_set, str):
        try:
            return LABEL_SET_ALIASES[label_set.lower()]
        except KeyError as exc:
            raise ValueError(f"Unknown label set: {label_set}") from exc
    labels = tuple(label_set)
    if not labels or labels[0] != LABEL_SPACE:
        raise ValueError("Every label set must begin with the no-punctuation label 'S'.")
    unknown = set(labels) - set(ALL_LABELS)
    if unknown:
        raise ValueError(f"Unknown labels: {sorted(unknown)}")
    return labels


def iter_labeled_tokens(
    text: str,
    lowercase: bool = True,
    label_set: str | Iterable[str] | None = None,
) -> Iterable[LabeledToken]:
    """Yield one punctuation label for each word-like token in punctuated text."""
    labels = resolve_label_set(label_set)
    text = normalize_text(text, lowercase=lowercase)
    matches = list(WORD_RE.finditer(text))

    for index, match in enumerate(matches):
        token = match.group(0)
        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        between = text[match.end() : next_start]
        yield LabeledToken(token=token, label=_label_from_separator(between, labels))


def _label_from_separator(separator: str, labels: Iterable[str] = CORE_LABELS) -> str:
    """Map punctuation between tokens to the first supported label.

    The order makes mixed spans such as ``?!`` deterministic and keeps the
    historical period-over-comma behavior for the core three-class setup.
    Unsupported marks fall back to ``S`` so the core benchmark remains
    backward compatible.
    """
    supported = set(labels)
    for mark, label in (
        ("?", LABEL_QUESTION),
        ("!", LABEL_EXCLAMATION),
        (".", LABEL_PERIOD),
        (",", LABEL_COMMA),
        (":", LABEL_COLON),
        (";", LABEL_SEMICOLON),
    ):
        if mark in separator and label in supported:
            return label
    return LABEL_SPACE


def tokens_and_labels(
    text: str,
    lowercase: bool = True,
    label_set: str | Iterable[str] | None = None,
) -> tuple[list[str], list[str]]:
    pairs = list(iter_labeled_tokens(text, lowercase=lowercase, label_set=label_set))
    return [pair.token for pair in pairs], [pair.label for pair in pairs]
