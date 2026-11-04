from __future__ import annotations

import argparse
from pathlib import Path

from kg_punctuation.data.build_sources import build_source_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kg-punct",
        description="Kyrgyz automatic punctuation restoration utilities.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_source = subparsers.add_parser(
        "build-source",
        help="Convert a raw source dataset into source-level APR JSONL.",
    )
    build_source.add_argument("source_id", choices=["kyrgyznews", "wikipedia_ky"])
    build_source.add_argument("--max-documents", type=int, default=None)
    build_source.add_argument("--min-tokens", type=int, default=20)
    build_source.add_argument("--output-path", type=Path, default=None)
    build_source.add_argument(
        "--label-set",
        choices=["core", "extended", "all"],
        default="core",
        help="core=S/C/P; extended adds question and exclamation labels.",
    )
    build_source.set_defaults(func=_build_source)

    return parser


def _build_source(args: argparse.Namespace) -> None:
    metadata = build_source_dataset(
        source_id=args.source_id,
        max_documents=args.max_documents,
        min_tokens=args.min_tokens,
        label_set=args.label_set,
        output_path=args.output_path,
    )
    print(f"source={metadata['source_id']} documents={metadata['documents']} tokens={metadata['tokens']}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
