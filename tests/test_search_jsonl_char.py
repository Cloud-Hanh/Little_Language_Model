from __future__ import annotations

from pathlib import Path

from search_jsonl_char import JSONLCharSearcher, SearchResult


def test_search_jsonl_character_finds_matches(tmp_path: Path) -> None:
    sample = tmp_path / "sample.jsonl"
    sample.write_text(
        """{"text":"abc"}\n{"text":"a午c"}\n{"text":"xyz"}\n""",
        encoding="utf-8",
    )

    searcher = JSONLCharSearcher(sample)
    matches = searcher.search("午")

    assert len(matches) == 1
    match = matches[0]
    assert isinstance(match, SearchResult)
    assert match.line_number == 2
    assert "a午c" in match.full_line


def test_print_results_no_global_dependency(capsys, tmp_path: Path) -> None:
    sample = tmp_path / "sample.jsonl"
    sample.write_text('{"text":"a午c"}\n', encoding="utf-8")

    searcher = JSONLCharSearcher(sample)
    searcher.search("午")
    searcher.print_results()

    out = capsys.readouterr().out
    assert "Found 1 matches" in out
    assert "line 1" in out
