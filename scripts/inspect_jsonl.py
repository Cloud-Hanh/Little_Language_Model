from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl_line_range(input_path: str | Path, start: int, end: int) -> list[tuple[int, dict]]:
    if start < 1 or end < start:
        raise ValueError("Require 1 <= start <= end")

    src = Path(input_path)
    rows: list[tuple[int, dict]] = []
    with src.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if line_no < start:
                continue
            if line_no > end:
                break
            raw = line.strip()
            if not raw:
                continue
            item = json.loads(raw)
            if isinstance(item, dict):
                rows.append((line_no, item))
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect JSONL records in an explicit line range.")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--start", type=int, required=True, help="Start line number (1-based, inclusive)")
    parser.add_argument("--end", type=int, required=True, help="End line number (1-based, inclusive)")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON objects")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = read_jsonl_line_range(args.input, args.start, args.end)

    for line_no, item in rows:
        if args.pretty:
            payload = json.dumps(item, ensure_ascii=False, indent=2)
        else:
            payload = json.dumps(item, ensure_ascii=False)
        print(f"line {line_no}: {payload}")

    if not rows:
        print("No records found in the requested range.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
