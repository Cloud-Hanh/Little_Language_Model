from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class SearchResult:
    line_number: int
    file_path: str
    char_position: int
    context_before: str
    context_after: str
    full_line: str
    matched_char: str
    json_key: str | None = None
    json_value: Any | None = None


class JSONLCharSearcher:
    def __init__(self, file_path: str | Path, encoding: str = "utf-8"):
        self.file_path = Path(file_path)
        self.encoding = encoding
        self.results: list[SearchResult] = []

    def search(
        self,
        target_char: str,
        context_size: int = 50,
        case_sensitive: bool = True,
        include_json_path: bool = True,
    ) -> list[SearchResult]:
        self.results.clear()

        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        with self.file_path.open("r", encoding=self.encoding) as handle:
            for line_num, line in enumerate(handle, start=1):
                line = line.rstrip("\n")
                self._search_in_line(
                    line=line,
                    line_num=line_num,
                    target_char=target_char,
                    context_size=context_size,
                    case_sensitive=case_sensitive,
                    include_json_path=include_json_path,
                )

        return self.results

    def _search_in_line(
        self,
        line: str,
        line_num: int,
        target_char: str,
        context_size: int,
        case_sensitive: bool,
        include_json_path: bool,
    ) -> None:
        search_line = line if case_sensitive else line.lower()
        search_target = target_char if case_sensitive else target_char.lower()

        for match in re.finditer(re.escape(search_target), search_line):
            char_pos = match.start()
            start_context = max(0, char_pos - context_size)
            end_context = min(len(line), char_pos + context_size + 1)

            json_key = None
            json_value = None
            if include_json_path:
                try:
                    payload = json.loads(line)
                    json_key, json_value = self._find_char_in_json(payload, target_char)
                except json.JSONDecodeError:
                    pass

            self.results.append(
                SearchResult(
                    line_number=line_num,
                    file_path=str(self.file_path),
                    char_position=char_pos,
                    context_before=line[start_context:char_pos],
                    context_after=line[char_pos + 1 : end_context],
                    full_line=line,
                    matched_char=line[char_pos],
                    json_key=json_key,
                    json_value=json_value,
                )
            )

    def _find_char_in_json(self, data: Any, target_char: str) -> tuple[str | None, Any | None]:
        if isinstance(data, dict):
            for key, value in data.items():
                key_str = json.dumps(key, ensure_ascii=False)
                value_str = json.dumps(value, ensure_ascii=False)
                if target_char in key_str:
                    return str(key), None
                if target_char in value_str:
                    return str(key), value
        return None, None

    def print_results(self, show_full_line: bool = False) -> None:
        if not self.results:
            print("No matches found")
            return

        print(f"Found {len(self.results)} matches")
        print("=" * 60)

        for idx, result in enumerate(self.results, start=1):
            if idx > 10:
                print(f"... {len(self.results) - 10} more results not shown ...")
                break

            print(f"Result #{idx}")
            print(f"  position: line {result.line_number}, char {result.char_position}")
            if result.json_key:
                print(f"  json key: {result.json_key}")
            print(f"  before: {result.context_before}")
            print(f"  char: [{result.matched_char}]")
            print(f"  after: {result.context_after}")

            if show_full_line:
                if len(result.full_line) > 200:
                    print(f"  full: {result.full_line[:200]}...")
                else:
                    print(f"  full: {result.full_line}")
            print("-" * 40)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search a character in JSONL files with context.")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--target-char", required=True, help="Target character to search")
    parser.add_argument("--context-size", type=int, default=30, help="Context window size")
    parser.add_argument("--show-full-line", action="store_true", help="Print full matching lines")
    parser.add_argument("--ignore-case", action="store_true", help="Case-insensitive search")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    searcher = JSONLCharSearcher(args.input)
    searcher.search(
        target_char=args.target_char,
        context_size=args.context_size,
        case_sensitive=not args.ignore_case,
        include_json_path=True,
    )
    searcher.print_results(show_full_line=args.show_full_line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
