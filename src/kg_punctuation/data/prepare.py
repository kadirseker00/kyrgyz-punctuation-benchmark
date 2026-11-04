from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from kg_punctuation.data.labeling import resolve_label_set, tokens_and_labels
from kg_punctuation.data.split import DatasetSplit, split_items


def prepare_dataset(
    input_dir: Path,
    output_dir: Path,
    seed: int = 42,
    train_ratio: float = 0.65,
    validation_ratio: float = 0.10,
    test_ratio: float = 0.25,
    label_set: str = "core",
) -> dict[str, dict[str, int]]:
    documents = sorted(str(path) for path in input_dir.glob("**/*.txt"))
    if not documents:
        raise FileNotFoundError(f"No .txt files found under {input_dir}")

    split = split_items(
        documents,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    labels = resolve_label_set(label_set)
    stats = {
        "train": _write_split(split.train, output_dir / "train.jsonl", labels),
        "validation": _write_split(split.validation, output_dir / "validation.jsonl", labels),
        "test": _write_split(split.test, output_dir / "test.jsonl", labels),
    }
    (output_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "label_metadata.json").write_text(
        json.dumps({"label_set": list(labels)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return stats


def _write_split(paths: list[str], output_path: Path, labels: tuple[str, ...]) -> dict[str, int]:
    stats = Counter({"documents": 0, "tokens": 0})
    stats.update({label: 0 for label in labels})
    with output_path.open("w", encoding="utf-8") as handle:
        for path in paths:
            text = Path(path).read_text(encoding="utf-8")
            tokens, row_labels = tokens_and_labels(text, label_set=labels)
            if not tokens:
                continue
            row = {"source": path, "tokens": tokens, "labels": row_labels}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats["documents"] += 1
            stats["tokens"] += len(tokens)
            stats.update(row_labels)
    return dict(stats)
