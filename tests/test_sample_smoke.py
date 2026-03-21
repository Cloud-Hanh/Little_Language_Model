from __future__ import annotations

from pathlib import Path

from littlelm.build_dictionary import build_ngram_range
from littlelm.sample import generate_text


def test_sample_smoke(tmp_path: Path) -> None:
    source = Path("examples/tiny_corpus.jsonl")
    output_dir = tmp_path / "dictionary"

    build_ngram_range(
        input_paths=[source],
        output_dir=output_dir,
        min_n=1,
        max_n=3,
        text_key="text",
        flush_threshold=50,
    )

    seed = "午后"
    generated = generate_text(
        seed=seed,
        ngrams_count_folder=output_dir,
        max_n=3,
        max_steps=20,
        debug=False,
    )

    assert isinstance(generated, str)
    assert generated
    assert generated.startswith(seed)
