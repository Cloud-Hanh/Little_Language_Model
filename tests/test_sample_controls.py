from __future__ import annotations

from pathlib import Path

import pytest

from littlelm.build_dictionary import build_ngram_range
from littlelm.sample import (
    apply_temperature,
    apply_top_k,
    apply_top_p,
    build_n_selection_distribution,
    generate_text,
    interactive_seed_input,
    normalize_distribution,
    parse_n_weights,
)


def _build_demo_dictionary(tmp_path: Path) -> Path:
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
    return output_dir


def test_distribution_controls_temperature_topk_topp() -> None:
    base = [("a", 0.6), ("b", 0.3), ("c", 0.1)]

    colder = apply_temperature(base, 0.5)
    hotter = apply_temperature(base, 2.0)

    prob_a_colder = dict(colder)["a"]
    prob_a_hotter = dict(hotter)["a"]
    assert prob_a_colder > prob_a_hotter

    top2 = apply_top_k(base, 2)
    assert [item[0] for item in top2] == ["a", "b"]

    topp = apply_top_p(base, 0.7)
    assert [item[0] for item in topp] == ["a", "b"]

    normalized = normalize_distribution(topp)
    assert pytest.approx(sum(prob for _, prob in normalized), rel=1e-9) == 1.0


def test_n_selection_modes() -> None:
    available = [1, 2, 3]

    uniform = build_n_selection_distribution(available, mode="uniform")
    probs_uniform = dict(uniform)
    assert pytest.approx(probs_uniform[1], rel=1e-9) == probs_uniform[2]
    assert pytest.approx(probs_uniform[2], rel=1e-9) == probs_uniform[3]

    fixed = build_n_selection_distribution(available, mode="fixed", fixed_n=2)
    assert fixed == [(2, 1.0)]

    manual = build_n_selection_distribution(
        available,
        mode="manual",
        manual_weights={1: 0.1, 2: 0.2, 3: 0.7},
    )
    probs_manual = dict(manual)
    assert probs_manual[3] > probs_manual[2] > probs_manual[1]


def test_parse_n_weights() -> None:
    weights = parse_n_weights("1:0.1,2:0.2,3:0.7")
    assert weights == {1: 0.1, 2: 0.2, 3: 0.7}


def test_verbose_debug_info_contains_required_fields(tmp_path: Path) -> None:
    dictionary = _build_demo_dictionary(tmp_path)

    text, infos = generate_text(
        seed="午后",
        ngrams_count_folder=dictionary,
        max_n=3,
        max_steps=5,
        verbosity="verbose",
        return_debug=True,
    )

    assert text.startswith("午后")
    if infos:
        first = infos[0]
        assert isinstance(first.chosen_n, int)
        assert first.top_candidates
        assert isinstance(first.chosen_char, str)


def test_interactive_seed_input_random_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    prompts: list[str] = []

    def fake_print(message: str) -> None:
        prompts.append(message)

    monkeypatch.setattr("littlelm.sample.get_random_prompt", lambda: "测试提示")
    monkeypatch.setattr("builtins.input", lambda _: "我的种子")

    seed = interactive_seed_input(prompt_mode="random", print_fn=fake_print)

    assert seed == "我的种子"
    assert any("测试提示" in item for item in prompts)


def test_interactive_seed_input_empty_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = iter(["", "", ""])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    seed = interactive_seed_input(prompt_mode="plain", default_seed="默认种子")
    assert seed == "默认种子"
