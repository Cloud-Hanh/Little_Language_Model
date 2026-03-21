from __future__ import annotations

from pathlib import Path

from inspect_jsonl import read_jsonl_line_range


def test_read_line_range_respects_start_and_end(tmp_path: Path) -> None:
    sample = tmp_path / "sample.jsonl"
    sample.write_text(
        """{"id":1}\n{"id":2}\n{"id":3}\n{"id":4}\n""",
        encoding="utf-8",
    )

    rows = read_jsonl_line_range(sample, start=2, end=3)

    assert len(rows) == 2
    assert rows[0][0] == 2
    assert rows[1][0] == 3
    assert rows[0][1]["id"] == 2
    assert rows[1][1]["id"] == 3
