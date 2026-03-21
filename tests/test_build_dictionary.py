from __future__ import annotations

import json
from pathlib import Path

from littlelm.build_dictionary import build_ngram_range


def test_build_dictionary_from_tiny_corpus(tmp_path: Path) -> None:
    source = Path("examples/tiny_corpus.jsonl")
    output_dir = tmp_path / "dictionary"

    outputs = build_ngram_range(
        input_paths=[source],
        output_dir=output_dir,
        min_n=1,
        max_n=3,
        text_key="text",
        flush_threshold=50,
    )

    assert len(outputs) == 3
    for output in outputs:
        assert output.exists()
        found_positive = False
        with output.open("r", encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                assert set(item.keys()) == {"ngram", "count"}
                assert isinstance(item["ngram"], str)
                assert isinstance(item["count"], int)
                if item["count"] > 0:
                    found_positive = True
        assert found_positive
