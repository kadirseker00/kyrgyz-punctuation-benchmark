"""BGN/PCGN (1979) romanization of Kyrgyz Cyrillic, used for the paper's example tables (R1.4)."""

from __future__ import annotations

_BASE = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "ё": "yo", "ж": "j", "з": "z", "и": "i",
    "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "ң": "ng", "о": "o", "ө": "ö", "п": "p",
    "р": "r", "с": "s", "т": "t", "у": "u", "ү": "ü", "ф": "f", "х": "kh", "ц": "ts", "ч": "ch",
    "ш": "sh", "щ": "shch", "ъ": "”", "ы": "y", "ь": "’", "э": "e", "ю": "yu", "я": "ya",
}
_VOWELS = set("аеёиоөуүыэюя")


def romanize(text: str) -> str:
    """Romanize Kyrgyz Cyrillic; ``е`` is ``ye`` word-initially and after vowels, ``ъ`` or ``ь``."""
    out = []
    previous = ""
    for char in text:
        lower = char.lower()
        if lower == "е":
            latin = "ye" if (not previous or not previous.isalpha() or previous in _VOWELS or previous in "ъь") else "e"
        else:
            latin = _BASE.get(lower, char)
        if char != lower and latin:
            latin = latin[0].upper() + latin[1:]
        out.append(latin)
        previous = lower
    return "".join(out)
