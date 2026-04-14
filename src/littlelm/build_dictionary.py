from __future__ import annotations

import heapq
import json
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

from .constants import DEFAULT_FLUSH_THRESHOLD, DEFAULT_TEXT_KEY
from .reader import iter_jsonl
from .utils import ensure_parent_dir, safe_replace

# 空白分隔符（片段内部不跨空白构建 n-gram）
_SPACE_RE = re.compile(r"[ \u3000\xa0\r\n]+")
# 句末标点作为边界：n-gram 不跨句末标点，但标点本身保留为独立片段供 1-gram 统计
_SENT_BOUNDARY_RE = re.compile(r"([。！？…]+)")


def split_text(text: str) -> list[str]:
    if text is None:
        return []
    if not isinstance(text, str):
        text = str(text)
    fragments: list[str] = []
    # 先按句末标点切割，capture group 让标点留在结果里成为独立片段
    for part in _SENT_BOUNDARY_RE.split(text):
        if not part:
            continue
        # 再按空白切割
        for sub in _SPACE_RE.split(part):
            if sub:
                fragments.append(sub)
    return fragments


def iter_ngrams_from_text(text: str, gram_size: int) -> Iterator[str]:
    if gram_size < 1:
        raise ValueError("gram_size must be >= 1")
    for part in split_text(text):
        if len(part) < gram_size:
            continue
        for idx in range(len(part) - gram_size + 1):
            yield part[idx : idx + gram_size]


def count_ngrams_in_file(file_path: str | Path, gram_size: int, text_key: str = DEFAULT_TEXT_KEY) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for item in iter_jsonl(file_path, on_error="warn"):
        text = item.get(text_key)
        if text is None:
            continue
        for ngram in iter_ngrams_from_text(str(text), gram_size):
            counts[ngram] += 1
    return dict(counts)


def flush_counts_to_chunk(
    counts: dict[str, int],
    chunk_dir: str | Path,
    gram_size: int,
    chunk_index: int,
) -> Path | None:
    if not counts:
        return None

    chunk_dir_path = Path(chunk_dir)
    chunk_dir_path.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        dir=str(chunk_dir_path),
        prefix=f"{gram_size}gram_{chunk_index}_",
        suffix=".jsonl",
    ) as handle:
        for ngram in sorted(counts):
            handle.write(json.dumps({"ngram": ngram, "count": int(counts[ngram])}, ensure_ascii=False))
            handle.write("\n")
        return Path(handle.name)


def _read_next_valid_json_line(handle, source: Path) -> dict | None:
    while True:
        line = handle.readline()
        if not line:
            return None
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            print(f"Skipping malformed line while merging ({source}): {exc}")
            continue
        if not isinstance(item, dict):
            continue
        return item


def merge_chunk_files(chunk_files: Iterable[str | Path], output_file: str | Path) -> None:
    output_path = ensure_parent_dir(output_file)
    chunk_paths = [Path(p) for p in chunk_files if p is not None]

    if not chunk_paths:
        output_path.write_text("", encoding="utf-8")
        return

    readers = []
    heap: list[tuple[str, int, int]] = []

    try:
        for idx, path in enumerate(chunk_paths):
            handle = path.open("r", encoding="utf-8")
            readers.append(handle)
            first = _read_next_valid_json_line(handle, path)
            if first is None:
                continue
            heapq.heappush(heap, (str(first.get("ngram", "")), int(first.get("count", 0) or 0), idx))

        tmp_output = output_path.with_suffix(output_path.suffix + ".tmp")
        with tmp_output.open("w", encoding="utf-8", newline="") as out:
            current_ngram: str | None = None
            current_count = 0

            while heap:
                ngram, count, idx = heapq.heappop(heap)
                if current_ngram is None:
                    current_ngram = ngram
                    current_count = count
                elif ngram == current_ngram:
                    current_count += count
                else:
                    out.write(json.dumps({"ngram": current_ngram, "count": current_count}, ensure_ascii=False))
                    out.write("\n")
                    current_ngram = ngram
                    current_count = count

                nxt = _read_next_valid_json_line(readers[idx], chunk_paths[idx])
                if nxt is not None:
                    heapq.heappush(heap, (str(nxt.get("ngram", "")), int(nxt.get("count", 0) or 0), idx))

            if current_ngram is not None:
                out.write(json.dumps({"ngram": current_ngram, "count": current_count}, ensure_ascii=False))
                out.write("\n")

        safe_replace(tmp_output, output_path)
    finally:
        for handle in readers:
            handle.close()


def build_ngram_dictionary(
    input_paths: Iterable[str | Path],
    gram_size: int,
    output_file: str | Path,
    text_key: str = DEFAULT_TEXT_KEY,
    flush_threshold: int = DEFAULT_FLUSH_THRESHOLD,
    merge_existing: bool = False,
) -> Path:
    if gram_size < 1:
        raise ValueError("gram_size must be >= 1")
    if flush_threshold < 1:
        raise ValueError("flush_threshold must be >= 1")

    output_path = Path(output_file)
    ensure_parent_dir(output_path)

    counts: dict[str, int] = {}
    chunk_files: list[Path] = []
    chunk_index = 0

    def add_ngram(ngram: str, inc: int = 1) -> None:
        if not ngram:
            return
        counts[ngram] = counts.get(ngram, 0) + inc

    def maybe_flush(force: bool = False) -> None:
        nonlocal chunk_index
        if not counts:
            return
        if not force and len(counts) < flush_threshold:
            return
        chunk = flush_counts_to_chunk(counts, output_path.parent, gram_size, chunk_index)
        chunk_index += 1
        if chunk is not None:
            chunk_files.append(chunk)
        counts.clear()

    if merge_existing and output_path.exists():
        for item in iter_jsonl(output_path, on_error="warn"):
            add_ngram(str(item.get("ngram", "")), int(item.get("count", 0) or 0))
            maybe_flush()

    for source in input_paths:
        source_path = Path(source)
        for item in iter_jsonl(source_path, on_error="warn"):
            text = item.get(text_key)
            if text is None:
                continue
            for ngram in iter_ngrams_from_text(str(text), gram_size):
                add_ngram(ngram)
            maybe_flush()

    maybe_flush(force=True)

    try:
        merge_chunk_files(chunk_files, output_path)
    finally:
        for chunk in chunk_files:
            if chunk.exists():
                chunk.unlink()

    return output_path


def build_ngram_range(
    input_paths: Iterable[str | Path],
    output_dir: str | Path,
    min_n: int = 1,
    max_n: int = 10,
    text_key: str = DEFAULT_TEXT_KEY,
    flush_threshold: int = DEFAULT_FLUSH_THRESHOLD,
    merge_existing: bool = False,
) -> list[Path]:
    if min_n < 1 or max_n < min_n:
        raise ValueError("Require 1 <= min_n <= max_n")

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    source_files = [Path(p) for p in input_paths]
    if not source_files:
        raise ValueError("No input files provided")

    outputs: list[Path] = []
    for gram_size in range(min_n, max_n + 1):
        out = output_dir_path / f"{gram_size}-gram.jsonl"
        outputs.append(
            build_ngram_dictionary(
                input_paths=source_files,
                gram_size=gram_size,
                output_file=out,
                text_key=text_key,
                flush_threshold=flush_threshold,
                merge_existing=merge_existing,
            )
        )
    return outputs
