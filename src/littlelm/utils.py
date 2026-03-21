from __future__ import annotations

import errno
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Iterable, Mapping, Any


def ensure_parent_dir(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def safe_replace(src: str | Path, dst: str | Path, retries: int = 8, base_sleep: float = 0.05) -> None:
    """Atomically replace dst with src, retrying around transient Windows file locks."""
    src_path = Path(src)
    dst_path = Path(dst)
    last_error: Exception | None = None

    for i in range(retries):
        try:
            os.replace(src_path, dst_path)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(base_sleep * (2**i))
        except OSError as exc:
            last_error = exc
            if getattr(exc, "errno", None) in (errno.EACCES, errno.EPERM):
                time.sleep(base_sleep * (2**i))
            else:
                raise

    if last_error is not None:
        raise last_error


def write_jsonl_records(path: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    target = ensure_parent_dir(path)
    with target.open("w", encoding="utf-8", newline="") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def atomic_write_text(path: str | Path, content: str, encoding: str = "utf-8") -> None:
    target = ensure_parent_dir(path)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding=encoding,
        delete=False,
        dir=str(target.parent),
        suffix=".tmp",
    ) as temp:
        temp.write(content)
        temp_path = Path(temp.name)
    safe_replace(temp_path, target)
