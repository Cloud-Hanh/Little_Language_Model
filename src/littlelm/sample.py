from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable

from .constants import (
    DEFAULT_END_CHARS,
    DEFAULT_N_SELECTION_MODE,
    DEFAULT_N_TEMPERATURE,
    DEFAULT_PROMPT_MODE,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_REPETITION_WINDOW,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    DEFAULT_VERBOSITY,
    RANDOM_SEED_PROMPTS,
)


@dataclass
class StepDebugInfo:
    step: int
    context: str
    n_distribution: list[tuple[int, float]]
    chosen_n: int
    top_candidates: list[tuple[str, float]]
    chosen_char: str
    current_text: str


@dataclass
class GenerationConfig:
    max_n: int = 10
    max_steps: int = 100
    end_chars: str = DEFAULT_END_CHARS
    temperature: float = DEFAULT_TEMPERATURE
    top_k: int = DEFAULT_TOP_K
    top_p: float = DEFAULT_TOP_P
    n_selection_mode: str = DEFAULT_N_SELECTION_MODE
    fixed_n: int | None = None
    n_weights: dict[int, float] | None = None
    n_temperature: float = DEFAULT_N_TEMPERATURE
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY
    repetition_window: int = DEFAULT_REPETITION_WINDOW


def infer_n_from_filename(filename: str | Path) -> int | None:
    base = Path(filename).stem.lower()
    patterns = [
        r"(\d+)\s*grams?",
        r"(\d+)[_-]?grams?",
        r"grams?[_-]?(\d+)",
        r"(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, base)
        if match:
            return int(match.group(1))
    return None


def infer_n_from_ngram_value(ngram_value: object) -> int:
    if isinstance(ngram_value, str):
        return len(ngram_value)
    if isinstance(ngram_value, list):
        return len(ngram_value)
    raise TypeError(f"Cannot infer n from ngram type: {type(ngram_value)}")


@lru_cache(maxsize=None)
def load_ngrams_count_from_folder(folder_path: str | Path) -> list[dict[str, int]]:
    folder = Path(folder_path).resolve()
    if not folder.is_dir():
        raise ValueError(f"Not a directory: {folder}")

    jsonl_files = sorted(path for path in folder.iterdir() if path.suffix.lower() == ".jsonl")
    if not jsonl_files:
        raise ValueError(f"No jsonl files found in {folder}")

    data_by_n: dict[int, dict[str, int]] = {}

    for file_path in jsonl_files:
        current: dict[str, int] = {}
        inferred_n = infer_n_from_filename(file_path)

        with file_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{file_path}:{line_no} invalid JSON: {exc}") from exc

                if not isinstance(item, dict) or "ngram" not in item or "count" not in item:
                    raise ValueError(f"{file_path}:{line_no} missing ngram/count")

                ngram = item["ngram"]
                if isinstance(ngram, list):
                    ngram = "".join(str(part) for part in ngram)
                if not isinstance(ngram, str):
                    ngram = str(ngram)

                count = int(item["count"])
                if count < 0:
                    raise ValueError(f"{file_path}:{line_no} count must be >= 0")

                current[ngram] = count

                if inferred_n is None:
                    inferred_n = infer_n_from_ngram_value(ngram)

        if inferred_n is None:
            continue

        data_by_n.setdefault(inferred_n, {}).update(current)

    if not data_by_n:
        raise ValueError(f"No valid ngram data loaded from {folder}")

    max_n = max(data_by_n)
    return [data_by_n.get(i, {}) for i in range(1, max_n + 1)]


def calculate_conditional_probability(ngram: str, previous_ngram: str, ngrams_count_folder: str | Path) -> float:
    ngrams_count = load_ngrams_count_from_folder(ngrams_count_folder)
    if len(ngram) != len(previous_ngram) + 1:
        raise ValueError("ngram length must be previous_ngram length + 1")

    n = len(ngram)
    if n - 1 >= len(ngrams_count) or n - 2 < 0:
        return 0.0

    count_ngram = ngrams_count[n - 1].get(ngram, 0)
    count_prev = ngrams_count[n - 2].get(previous_ngram, 0)
    if count_prev == 0:
        return 0.0
    return count_ngram / count_prev


def normalize_distribution(candidates: Iterable[tuple[str, float]]) -> list[tuple[str, float]]:
    items = [(key, float(value)) for key, value in candidates if float(value) > 0.0]
    total = sum(value for _, value in items)
    if total <= 0.0:
        return []
    return [(key, value / total) for key, value in items]


def apply_temperature(candidates: list[tuple[str, float]], temperature: float) -> list[tuple[str, float]]:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    norm = normalize_distribution(candidates)
    if not norm:
        return []

    adjusted = [(key, prob ** (1.0 / temperature)) for key, prob in norm]
    return normalize_distribution(adjusted)


def apply_top_k(candidates: list[tuple[str, float]], k: int) -> list[tuple[str, float]]:
    if k is None or k <= 0:
        return list(candidates)
    ranked = sorted(candidates, key=lambda item: item[1], reverse=True)
    return ranked[:k]


def apply_top_p(candidates: list[tuple[str, float]], p: float) -> list[tuple[str, float]]:
    if p <= 0 or p > 1:
        raise ValueError("top_p must be in (0, 1]")
    ranked = sorted(candidates, key=lambda item: item[1], reverse=True)
    if p >= 1.0:
        return ranked

    picked: list[tuple[str, float]] = []
    cumulative = 0.0
    for item in ranked:
        picked.append(item)
        cumulative += item[1]
        if cumulative >= p:
            break
    return picked


def apply_repetition_penalty(
    candidates: list[tuple[str, float]],
    recent_text: str,
    penalty: float,
    window: int = DEFAULT_REPETITION_WINDOW,
) -> list[tuple[str, float]]:
    """对最近生成过的字符降权，抑制重复。penalty=1.0 表示不惩罚。"""
    if penalty <= 1.0:
        return list(candidates)
    recent = set(recent_text[-window:])
    return [(char, prob / penalty if char in recent else prob) for char, prob in candidates]


def parse_n_weights(spec: str) -> dict[int, float]:
    weights: dict[int, float] = {}
    text = (spec or "").strip()
    if not text:
        raise ValueError("n-weights cannot be empty")

    parts = [part.strip() for part in text.split(",") if part.strip()]
    for part in parts:
        if ":" not in part:
            raise ValueError(f"Invalid n-weights token: {part}")
        key_str, value_str = part.split(":", 1)
        n_value = int(key_str.strip())
        weight = float(value_str.strip())
        if n_value < 1:
            raise ValueError("n in n-weights must be >= 1")
        if weight < 0:
            raise ValueError("weights in n-weights must be >= 0")
        weights[n_value] = weight

    if not weights:
        raise ValueError("No valid n-weights found")
    return weights


def generate_weights(length: int, peak: int) -> list[float]:
    if length < 0:
        raise ValueError("length must be >= 0")
    if peak < 1 or peak > length + 1:
        raise ValueError(f"peak must be in [1, {length + 1}]")

    weights: list[float] = []
    for k in range(1, length + 2):
        if k <= peak:
            if peak > 1:
                weight = math.exp(5 * (k - peak) / (peak - 1))
            else:
                weight = 1.0
        else:
            if length + 1 > peak:
                weight = math.exp(-5 * (k - peak) / (length + 2 - peak))
            else:
                weight = 1.0
        weights.append(weight)

    max_w = max(weights) if weights else 1.0
    if max_w <= 0:
        return [1.0 for _ in weights]

    return [w / max_w * 0.9 + 0.1 for w in weights]


def build_n_selection_distribution(
    available_n_values: list[int],
    mode: str,
    n_temperature: float = DEFAULT_N_TEMPERATURE,
    fixed_n: int | None = None,
    manual_weights: dict[int, float] | None = None,
) -> list[tuple[int, float]]:
    if not available_n_values:
        return []

    ordered_n = sorted(set(available_n_values))
    raw: list[tuple[str, float]] = []

    if mode == "uniform":
        raw = [(str(n), 1.0) for n in ordered_n]
    elif mode == "fixed":
        if fixed_n is None:
            raise ValueError("fixed-n is required when n-selection-mode=fixed")
        selected_n = fixed_n if fixed_n in ordered_n else max(ordered_n)
        raw = [(str(selected_n), 1.0)]
    elif mode == "manual":
        if not manual_weights:
            raise ValueError("n-weights is required when n-selection-mode=manual")
        raw = [(str(n), float(manual_weights.get(n, 0.0))) for n in ordered_n]
    elif mode == "weighted":
        max_k = max(ordered_n)
        weighted = generate_weights(max_k - 1, peak=min(3, max_k))
        raw = [(str(n), float(weighted[n - 1])) for n in ordered_n]
    else:
        raise ValueError(f"Unknown n-selection-mode: {mode}")

    dist = normalize_distribution(raw)
    if not dist:
        raise ValueError("n-selection distribution is empty")

    dist = apply_temperature(dist, n_temperature)
    return [(int(n_str), prob) for n_str, prob in normalize_distribution(dist)]


def sample_n_value(distribution: list[tuple[int, float]]) -> int:
    if not distribution:
        raise ValueError("Cannot sample n from empty distribution")
    n_values = [n for n, _ in distribution]
    probs = [p for _, p in distribution]
    return random.choices(n_values, weights=probs, k=1)[0]


def _candidate_char_distribution_for_n(
    context: str,
    chosen_n: int,
    ngrams_count: list[dict[str, int]],
) -> list[tuple[str, float]]:
    if chosen_n <= 1:
        total = sum(ngrams_count[0].values())
        if total <= 0:
            return []
        return [
            (ngram, count / total)
            for ngram, count in ngrams_count[0].items()
            if isinstance(ngram, str) and len(ngram) == 1 and count > 0
        ]

    prev = context[-(chosen_n - 1) :]
    count_prev = ngrams_count[chosen_n - 2].get(prev, 0)
    if count_prev <= 0:
        return []

    char_probs: dict[str, float] = {}
    for ngram, count_ngram in ngrams_count[chosen_n - 1].items():
        if not ngram.startswith(prev):
            continue
        char = ngram[-1]
        prob = count_ngram / count_prev
        char_probs[char] = char_probs.get(char, 0.0) + prob
    return list(char_probs.items())


def predict_next_char(
    context: str,
    ngrams_count_folder: str | Path,
    max_n: int = 10,
    debug: bool = False,
) -> tuple[str, StepDebugInfo | None]:
    if not isinstance(context, str):
        raise TypeError("context must be a string")

    ngrams_count = load_ngrams_count_from_folder(ngrams_count_folder)
    if not ngrams_count or not ngrams_count[0]:
        return "", None

    config = GenerationConfig(max_n=max_n)

    max_available = len(ngrams_count)
    max_k = min(len(context) + 1, config.max_n, max_available)
    if max_k < 1:
        return "", None

    available_n = list(range(1, max_k + 1))

    # 从分布中采样首选 n，然后确定性地逐级回退直到找到候选
    n_distribution = build_n_selection_distribution(available_n, mode="weighted")
    chosen_n = sample_n_value(n_distribution)

    raw_candidates: list[tuple[str, float]] = []
    for try_n in range(chosen_n, 0, -1):
        raw_candidates = normalize_distribution(
            _candidate_char_distribution_for_n(context, try_n, ngrams_count)
        )
        if raw_candidates:
            chosen_n = try_n
            break

    if not raw_candidates:
        return "", None

    final_candidates = sorted(raw_candidates, key=lambda item: item[1], reverse=True)
    chosen_char = random.choices([c for c, _ in final_candidates], weights=[p for _, p in final_candidates], k=1)[0]

    info = StepDebugInfo(
        step=0,
        context=context,
        n_distribution=n_distribution,
        chosen_n=chosen_n,
        top_candidates=final_candidates[:10],
        chosen_char=chosen_char,
        current_text=context + chosen_char,
    )

    if debug:
        print(f"Candidates (top 10): {final_candidates[:10]}")

    return chosen_char, info


# Backward-compat alias
predict_next_word = predict_next_char


def generate_text(
    seed: str,
    ngrams_count_folder: str | Path,
    max_n: int = 10,
    end_chars: str = DEFAULT_END_CHARS,
    max_steps: int = 100,
    debug: bool = False,
    temperature: float = DEFAULT_TEMPERATURE,
    top_k: int = DEFAULT_TOP_K,
    top_p: float = DEFAULT_TOP_P,
    n_selection_mode: str = DEFAULT_N_SELECTION_MODE,
    fixed_n: int | None = None,
    n_weights: dict[int, float] | None = None,
    n_temperature: float = DEFAULT_N_TEMPERATURE,
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
    repetition_window: int = DEFAULT_REPETITION_WINDOW,
    verbosity: str = DEFAULT_VERBOSITY,
    return_debug: bool = False,
) -> str | tuple[str, list[StepDebugInfo]]:
    if not isinstance(seed, str):
        raise TypeError("seed must be a string")

    config = GenerationConfig(
        max_n=max_n,
        max_steps=max_steps,
        end_chars=end_chars,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        n_selection_mode=n_selection_mode,
        fixed_n=fixed_n,
        n_weights=n_weights,
        n_temperature=n_temperature,
        repetition_penalty=repetition_penalty,
        repetition_window=repetition_window,
    )

    if config.temperature <= 0:
        raise ValueError("temperature must be > 0")
    if config.top_k < 0:
        raise ValueError("top-k must be >= 0")
    if config.top_p <= 0 or config.top_p > 1:
        raise ValueError("top-p must be in (0, 1]")
    if config.n_temperature <= 0:
        raise ValueError("n-temperature must be > 0")
    if config.repetition_penalty < 1.0:
        raise ValueError("repetition-penalty must be >= 1.0")
    if config.n_selection_mode == "fixed" and config.fixed_n is None:
        raise ValueError("fixed-n is required when n-selection-mode=fixed")
    if config.n_selection_mode == "manual" and not config.n_weights:
        raise ValueError("n-weights is required when n-selection-mode=manual")

    text = seed
    ngrams_count = load_ngrams_count_from_folder(ngrams_count_folder)
    max_available = len(ngrams_count)
    step_infos: list[StepDebugInfo] = []

    for step in range(1, config.max_steps + 1):
        max_k = min(len(text) + 1, config.max_n, max_available)
        if max_k < 1:
            break

        available_n = list(range(1, max_k + 1))

        # 从分布中采样首选 n，然后确定性地逐级回退直到找到候选
        n_distribution = build_n_selection_distribution(
            available_n,
            mode=config.n_selection_mode,
            n_temperature=config.n_temperature,
            fixed_n=config.fixed_n,
            manual_weights=config.n_weights,
        )
        chosen_n = sample_n_value(n_distribution)

        raw_candidates: list[tuple[str, float]] = []
        for try_n in range(chosen_n, 0, -1):
            raw_candidates = normalize_distribution(
                _candidate_char_distribution_for_n(text, try_n, ngrams_count)
            )
            if raw_candidates:
                chosen_n = try_n
                break

        if not raw_candidates:
            break

        candidates = normalize_distribution(raw_candidates)
        candidates = apply_repetition_penalty(candidates, text, config.repetition_penalty, config.repetition_window)
        candidates = normalize_distribution(candidates)
        candidates = apply_temperature(candidates, config.temperature)
        candidates = normalize_distribution(candidates)
        candidates = apply_top_k(candidates, config.top_k)
        candidates = normalize_distribution(candidates)
        candidates = apply_top_p(candidates, config.top_p)
        candidates = normalize_distribution(candidates)

        if not candidates:
            break

        ranked = sorted(candidates, key=lambda item: item[1], reverse=True)
        nxt = random.choices([c for c, _ in ranked], weights=[p for _, p in ranked], k=1)[0]
        if not nxt:
            break
        text += nxt

        info = StepDebugInfo(
            step=step,
            context=text[:-1],
            n_distribution=n_distribution,
            chosen_n=chosen_n,
            top_candidates=ranked[:10],
            chosen_char=nxt,
            current_text=text,
        )
        step_infos.append(info)

        if debug and verbosity == "verbose":
            print(f"[Step {step}] chosen n={chosen_n}, chosen char={nxt}")

        if nxt in config.end_chars:
            break

    if return_debug:
        return text, step_infos
    return text


def get_random_prompt() -> str:
    return random.choice(RANDOM_SEED_PROMPTS)


def interactive_seed_input(
    prompt_mode: str = DEFAULT_PROMPT_MODE,
    # default_seed: str = "我",
    input_fn: Callable[[str], str] | None = None,
) -> str:
    if input_fn is None:
        input_fn = input
        
    if prompt_mode == "random":
        random_prompt = get_random_prompt()
        seed=input_fn(random_prompt+(":" if not random_prompt.endswith("？") else "")).strip()
    if prompt_mode == "plain":
        seed=input_fn("请输入一个开头句子，作为生成的种子 (seed)。").strip()
    return seed
    # for _ in range(3):
        # seed = input_fn("请输入你的开头句子 / seed：\n> ").strip()
        # if seed:
        #     return seed
    #     print_fn("输入为空，请再试一次。")

    # print_fn(f"输入多次为空，使用默认 seed：{default_seed}")
    # return default_seed


def format_step_debug_info(info: StepDebugInfo) -> str:
    lines = [
        f"[Step {info.step}]",
        f"Context: \"{info.context}\"",
        "n-choice distribution:",
    ]
    for n_value, prob in sorted(info.n_distribution, key=lambda item: item[0], reverse=True):
        lines.append(f"  n={n_value}: {prob:.4f}")
    lines.append(f"chosen n: {info.chosen_n}")
    lines.append("")
    lines.append("top candidates:")
    for idx, (char, prob) in enumerate(info.top_candidates, start=1):
        lines.append(f"  {idx}. \"{char}\" -> {prob:.4f}")
    lines.append(f"chosen char: \"{info.chosen_char}\"")
    lines.append(f"current text: \"{info.current_text}\"")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample text from character-level n-gram dictionaries.")
    parser.add_argument("--dictionary", required=True, help="Dictionary folder that contains *-gram.jsonl files.")
    parser.add_argument("--seed", help="Seed text for generation.")
    parser.add_argument("--interactive", action="store_true", help="Enable interactive seed input.")
    parser.add_argument(
        "--prompt-mode",
        choices=["plain", "random"],
        default=DEFAULT_PROMPT_MODE,
        help="Prompt mode for interactive seed input.",
    )
    parser.add_argument("--max-n", type=int, default=10, help="Max n used for prediction.")
    parser.add_argument("--max-steps", type=int, default=100, help="Maximum generation steps.")
    parser.add_argument("--end-chars", default=DEFAULT_END_CHARS, help="Stop when generated char is in this set.")
    parser.add_argument(
        "--verbosity",
        choices=["quiet", "normal", "verbose"],
        default=DEFAULT_VERBOSITY,
        help="Output detail level.",
    )
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Character sampling temperature.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Keep only top-k characters (0 = disabled).")
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P, help="Keep minimal cumulative top-p nucleus.")
    parser.add_argument(
        "--n-selection-mode",
        choices=["weighted", "uniform", "fixed", "manual"],
        default=DEFAULT_N_SELECTION_MODE,
        help="How to choose n per step.",
    )
    parser.add_argument("--fixed-n", type=int, help="Fixed n value when n-selection-mode=fixed.")
    parser.add_argument("--n-weights", help="Manual n weights, for example: 1:0.1,2:0.2,3:0.7")
    parser.add_argument("--n-temperature", type=float, default=DEFAULT_N_TEMPERATURE, help="Temperature for n selection distribution.")
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=DEFAULT_REPETITION_PENALTY,
        help="Divide probability of recently-seen chars by this factor (1.0 = disabled, >1.0 = penalize).",
    )
    parser.add_argument(
        "--repetition-window",
        type=int,
        default=DEFAULT_REPETITION_WINDOW,
        help="Number of recent characters to consider for repetition penalty.",
    )
    parser.add_argument("--debug", action="store_true", help="Print debug candidate info.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.n_selection_mode == "fixed" and args.fixed_n is None:
        parser.error("--fixed-n is required when --n-selection-mode=fixed")
    if args.n_selection_mode == "manual" and not args.n_weights:
        parser.error("--n-weights is required when --n-selection-mode=manual")

    if args.seed:
        seed = args.seed
    elif args.interactive or not args.seed:
        seed = interactive_seed_input(prompt_mode=args.prompt_mode)
    else:
        parser.error("seed is required")

    manual_weights = parse_n_weights(args.n_weights) if args.n_weights else None

    verbosity = args.verbosity
    if args.debug and verbosity == "normal":
        verbosity = "verbose"

    generated, step_infos = generate_text(
        seed=seed,
        ngrams_count_folder=args.dictionary,
        max_n=args.max_n,
        end_chars=args.end_chars,
        max_steps=args.max_steps,
        debug=args.debug,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        n_selection_mode=args.n_selection_mode,
        fixed_n=args.fixed_n,
        n_weights=manual_weights,
        n_temperature=args.n_temperature,
        repetition_penalty=args.repetition_penalty,
        repetition_window=args.repetition_window,
        verbosity=verbosity,
        return_debug=True,
    )

    if verbosity == "quiet":
        print(generated)
        return 0

    if verbosity == "normal":
        print("Sampling configuration:")
        print(f"  seed: {seed}")
        print(f"  max_n: {args.max_n}")
        print(f"  max_steps: {args.max_steps}")
        print(f"  n_selection_mode: {args.n_selection_mode}")
        print("")
        print("Generated text:")
        print(generated)
        return 0

    print("Sampling configuration:")
    print(f"  seed: {seed}")
    print(f"  max_n: {args.max_n}")
    print(f"  max_steps: {args.max_steps}")
    print(f"  temperature: {args.temperature}")
    print(f"  top_k: {args.top_k}")
    print(f"  top_p: {args.top_p}")
    print(f"  n_selection_mode: {args.n_selection_mode}")
    print(f"  fixed_n: {args.fixed_n}")
    print(f"  n_temperature: {args.n_temperature}")
    print(f"  repetition_penalty: {args.repetition_penalty}")
    print(f"  repetition_window: {args.repetition_window}")
    print("")

    for info in step_infos:
        print(format_step_debug_info(info))
        print("")

    print("Generated text:")
    print(generated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
