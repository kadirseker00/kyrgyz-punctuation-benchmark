from __future__ import annotations

from collections import Counter
import bz2
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re
import time
import xml.etree.ElementTree as ET

from kg_punctuation.data.labeling import resolve_label_set, tokens_and_labels


ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    domain: str
    raw_path: Path
    output_path: Path


SOURCES = {
    "kyrgyznews": SourceSpec(
        source_id="kyrgyznews",
        domain="news",
        raw_path=ROOT / "data/raw/kyrgyznews/KG_news_256k.csv",
        output_path=ROOT / "data/processed/source_level/kyrgyznews.jsonl",
    ),
    "wikipedia_ky": SourceSpec(
        source_id="wikipedia_ky",
        domain="wikipedia",
        raw_path=ROOT / "data/raw/wikipedia_ky/kywiki-latest-pages-articles.xml.bz2",
        output_path=ROOT / "data/processed/source_level/wikipedia_ky.jsonl",
    ),
}


def build_source_dataset(
    source_id: str,
    max_documents: int | None = None,
    min_tokens: int = 20,
    label_set: str = "core",
    output_path: Path | None = None,
) -> dict[str, object]:
    if source_id not in SOURCES:
        raise ValueError(f"Unknown source_id: {source_id}")

    spec = SOURCES[source_id]
    labels = resolve_label_set(label_set)
    destination = output_path or _default_output_path(spec, label_set)
    start = time.perf_counter()
    destination.parent.mkdir(parents=True, exist_ok=True)

    if source_id == "kyrgyznews":
        rows = _iter_kyrgyznews(spec.raw_path)
    elif source_id == "wikipedia_ky":
        rows = _iter_wikipedia(spec.raw_path)
    else:
        raise ValueError(f"No builder for {source_id}")

    stats = Counter({"documents": 0, "tokens": 0})
    stats.update({label: 0 for label in labels})
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            tokens, row_labels = tokens_and_labels(row["text"], label_set=labels)
            if len(tokens) < min_tokens:
                continue
            output = {
                "source_id": spec.source_id,
                "domain": spec.domain,
                "document_id": row["document_id"],
                "title": row.get("title", ""),
                "source": row.get("source", ""),
                "script": "cyrillic",
                "tokens": tokens,
                "labels": row_labels,
            }
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")
            stats["documents"] += 1
            stats["tokens"] += len(tokens)
            stats.update(row_labels)
            if max_documents and stats["documents"] >= max_documents:
                break

    metadata = {
        "source_id": spec.source_id,
        "domain": spec.domain,
        "raw_path": str(spec.raw_path),
        "output_path": str(destination),
        "label_set": list(labels),
        "documents": stats["documents"],
        "tokens": stats["tokens"],
        "labels": {label: stats[label] for label in labels},
        "min_tokens": min_tokens,
        "max_documents": max_documents,
        "runtime_seconds": round(time.perf_counter() - start, 3),
    }
    destination.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


def _default_output_path(spec: SourceSpec, label_set: str) -> Path:
    if label_set == "core":
        return spec.output_path
    return spec.output_path.with_name(f"{spec.output_path.stem}_{label_set}.jsonl")


def _iter_kyrgyznews(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        sample = handle.read(8192)
        handle.seek(0)
        dialect = csv.Sniffer().sniff(sample)
        reader = csv.DictReader(handle, dialect=dialect)
        fieldnames = reader.fieldnames or []
        for index, row in enumerate(reader):
            text = _first_nonempty(row, ["text", "content", "article", "news", "body", "description"])
            if not text:
                text = _longest_text_field(row)
            title = _first_nonempty(row, ["title", "headline", "name"])
            source = _first_nonempty(row, ["source", "url", "link"])
            yield {
                "document_id": str(row.get("id") or row.get("ID") or f"kyrgyznews_{index:06d}"),
                "title": title,
                "source": source,
                "text": _clean_plain_text(text),
                "fieldnames": fieldnames,
            }


def _iter_wikipedia(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)

    namespace = ""
    with bz2.open(path, "rb") as handle:
        context = ET.iterparse(handle, events=("start", "end"))
        for event, elem in context:
            if event == "start" and elem.tag.endswith("mediawiki"):
                namespace = elem.tag.removesuffix("mediawiki")
            if event == "end" and elem.tag == f"{namespace}page":
                title = elem.findtext(f"{namespace}title") or ""
                ns = elem.findtext(f"{namespace}ns") or ""
                page_id = elem.findtext(f"{namespace}id") or title
                text = elem.findtext(f"{namespace}revision/{namespace}text") or ""
                elem.clear()
                if ns != "0" or _is_wiki_redirect(text):
                    continue
                yield {
                    "document_id": f"wiki_{page_id}",
                    "title": title,
                    "source": "kywiki",
                    "text": _clean_wiki_text(text),
                }


def _first_nonempty(row: dict[str, str], names: list[str]) -> str:
    lower_map = {key.lower(): value for key, value in row.items() if key}
    for name in names:
        value = lower_map.get(name.lower())
        if value and value.strip():
            return value.strip()
    return ""


def _longest_text_field(row: dict[str, str]) -> str:
    values = [value.strip() for value in row.values() if isinstance(value, str) and value.strip()]
    return max(values, key=len, default="")


def _clean_plain_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _is_wiki_redirect(text: str) -> bool:
    return text.lstrip().lower().startswith("#redirect") or text.lstrip().startswith("#БАГЫТ")


def _clean_wiki_text(text: str) -> str:
    try:
        import mwparserfromhell

        text = mwparserfromhell.parse(text).strip_code()
    except Exception:
        text = re.sub(r"\{\{.*?\}\}", " ", text, flags=re.DOTALL)
        text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)

    text = re.sub(r"<ref[^>]*>.*?</ref>", " ", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\[\w+:[^\]]+\]", " ", text)
    text = re.sub(r"={2,}[^=]+={2,}", " ", text)
    return re.sub(r"\s+", " ", text).strip()
