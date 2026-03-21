from __future__ import annotations

import argparse
from pathlib import Path

from .build_dictionary import build_ngram_range
from .constants import DEFAULT_FLUSH_THRESHOLD, DEFAULT_MAX_N, DEFAULT_MIN_N, DEFAULT_TEXT_KEY
from .reader import iter_input_files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train character-level n-gram dictionaries from JSONL corpora.")
    parser.add_argument("--input", nargs="+", required=True, help="Input JSONL file(s) or directory path(s).")
    parser.add_argument("--output", required=True, help="Output directory for n-gram dictionary files.")
    parser.add_argument("--min-n", type=int, default=DEFAULT_MIN_N, help="Minimum n-gram order.")
    parser.add_argument("--max-n", type=int, default=DEFAULT_MAX_N, help="Maximum n-gram order.")
    parser.add_argument("--text-key", default=DEFAULT_TEXT_KEY, help="JSON key used for text extraction.")
    parser.add_argument(
        "--flush-threshold",
        type=int,
        default=DEFAULT_FLUSH_THRESHOLD,
        help="Unique n-gram threshold for chunk flush.",
    )
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help="Merge with existing output files if they already exist.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_files: list[Path] = []
    for raw in args.input:
        for file_path in iter_input_files(raw):
            input_files.append(file_path)

    if not input_files:
        parser.error("No JSONL files found from --input")

    outputs = build_ngram_range(
        input_paths=input_files,
        output_dir=args.output,
        min_n=args.min_n,
        max_n=args.max_n,
        text_key=args.text_key,
        flush_threshold=args.flush_threshold,
        merge_existing=args.merge_existing,
    )

    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
