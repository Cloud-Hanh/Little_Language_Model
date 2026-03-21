from __future__ import annotations

import argparse
from pathlib import Path


def cut_jsonl_head(input_path: str | Path, output_path: str | Path, max_bytes: int) -> int:
    src = Path(input_path)
    dst = Path(output_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with src.open("rb") as fin, dst.open("wb") as fout:
        for line in fin:
            line_len = len(line)
            if total + line_len > max_bytes:
                break
            fout.write(line)
            total += line_len
    return total


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a small demo JSONL file by byte-size head slicing.")
    parser.add_argument("--input", required=True, help="Source JSONL file")
    parser.add_argument("--output", required=True, help="Destination JSONL file")
    parser.add_argument("--max-bytes", type=int, default=10 * 1024 * 1024, help="Maximum output bytes")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    written = cut_jsonl_head(args.input, args.output, args.max_bytes)
    print(f"Generated: {args.output}, size={written} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
