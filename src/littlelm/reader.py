from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator


def iter_input_files(path_or_dir: str | Path, pattern: str = "*.jsonl") -> Iterator[Path]:
    target = Path(path_or_dir)
    if target.is_file():
        yield target
        return

    if target.is_dir():
        for file_path in sorted(target.rglob(pattern)):
            if file_path.is_file():
                yield file_path
        return

    raise FileNotFoundError(f"Input path does not exist: {target}")


def iter_jsonl(path: str | Path, on_error: str = "warn") -> Iterator[dict]:
    """
    Iterate JSON objects from a JSONL file.

    on_error:
    - "warn": print and skip bad lines
    - "raise": raise json.JSONDecodeError or TypeError
    - "skip": silently skip bad lines
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"JSONL file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
                if not isinstance(item, dict):
                    raise TypeError("JSONL line is not a JSON object")
                yield item
            except (json.JSONDecodeError, TypeError) as exc:
                if on_error == "raise":
                    raise
                if on_error == "warn":
                    print(f"{file_path} line {line_no} skipped: {exc}")


def iter_texts_from_jsonl(path: str | Path, text_key: str = "Content", on_error: str = "warn") -> Iterator[str]:
    for item in iter_jsonl(path, on_error=on_error):
        value = item.get(text_key)
        if value is None:
            continue
        if isinstance(value, str):
            yield value
        else:
            yield str(value)
