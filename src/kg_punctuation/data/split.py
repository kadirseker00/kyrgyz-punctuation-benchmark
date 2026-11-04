from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Sequence


@dataclass(frozen=True)
class DatasetSplit:
    train: list[PathLike]
    validation: list[PathLike]
    test: list[PathLike]


PathLike = str


def split_items(
    items: Sequence[PathLike],
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> DatasetSplit:
    total = train_ratio + validation_ratio + test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1.0")

    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)

    train_end = round(len(shuffled) * train_ratio)
    validation_end = train_end + round(len(shuffled) * validation_ratio)

    return DatasetSplit(
        train=shuffled[:train_end],
        validation=shuffled[train_end:validation_end],
        test=shuffled[validation_end:],
    )

