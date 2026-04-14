# LittleLM

LittleLM is a character-level n-gram tiny language model toolkit.

It keeps a simple workflow:

1. Read text fields from JSONL corpora.
2. Split each text by whitespace into fragments.
3. Build character-level n-gram counts inside each fragment.
4. Save 1~N gram dictionaries as JSONL files.
5. Sample from generated dictionaries to produce text.

## Model type

LittleLM is a **character-level n-gram model**.

## Corpus format

- Input must be JSONL (one JSON object per line).
- Default text key is `Content`.
- You can override with `--text-key`.

See docs:

- `docs/corpus_format.md`
- `docs/training.md`
- `docs/tuning.md`

## Installation(be sure you are "in" the program folder)

```bash
pip install -e .
```

Optional dev dependencies:

```bash
pip install -e .[dev]
```

## Quick start

### 1. Train dictionary

```bash
python -m littlelm.train \
  --input examples/tiny_corpus.jsonl \
  --output artifacts/dictionary \
  --min-n 1 \
  --max-n 3 \
  --text-key text
```

### 2. Sample text

```bash
python -m littlelm.sample \
  --dictionary artifacts/dictionary \
  --seed 午后 \
  --max-n 3 \
  --max-steps 20
```

### 3. Interactive seed input (random prompt)

```bash
python -m littlelm.sample \
  --dictionary artifacts/dictionary \
  --interactive \
  --prompt-mode plain \
  --verbosity normal
```

### 4. Verbose step-by-step debug

```bash
python -m littlelm.sample \
  --dictionary artifacts/dictionary \
  --seed 我 \
  --verbosity verbose \
  --temperature 0.8 \
  --top-k 10
```

### 5. Fixed n selection

```bash
python -m littlelm.sample \
  --dictionary artifacts/dictionary \
  --seed 刚刚 \
  --n-selection-mode fixed \
  --fixed-n 3
```

### 6. Manual n weights

```bash
python -m littlelm.sample \
  --dictionary artifacts/dictionary \
  --seed 今天 \
  --n-selection-mode manual \
  --n-weights "1:0.1,2:0.2,3:0.7"
```

You can also use the unified entry:

```bash
python -m littlelm train --input examples/tiny_corpus.jsonl --output artifacts/dictionary --min-n 1 --max-n 10 --text-key text
python -m littlelm sample --dictionary artifacts/dictionary --max-n 10 --max-steps 20
```

## Demo data

`examples/tiny_corpus.jsonl` uses key `text`, so pass `--text-key text` when training with it.

## Output format

Each dictionary JSONL line:

```json
{"ngram": "示例", "count": 123}
```

## Repository structure

```text
src/littlelm/        # core package
scripts/             # helper scripts
examples/            # tiny demo data
docs/                # docs
tests/               # pytest tests
archive/             # legacy archived scripts
```

## Sample CLI options

- `--dictionary`: dictionary folder containing `*-gram.jsonl`.
- `--seed`: direct seed input. If provided, interactive prompt is skipped.
- `--interactive`: explicitly enable interactive seed input.
- `--prompt-mode {plain,random}`: interactive prompt style.
- `--verbosity {quiet,normal,verbose}`: output detail level.
- `--max-n`: max n-gram order used per step.
- `--max-steps`: max generated steps.
- `--end-chars`: stop characters.
- `--temperature`: character sampling temperature.
- `--top-k`: character top-k filter (`0` means disabled).
- `--top-p`: character top-p nucleus filter.
- `--n-selection-mode {weighted,uniform,fixed,manual}`: n-value sampling strategy.
- `--fixed-n`: fixed n when mode is `fixed`.
- `--n-weights`: manual n weights when mode is `manual`.
- `--n-temperature`: temperature on n-selection distribution.
- `--repetition-penalty`: divide probability of recently-seen characters by this factor (`1.0` = disabled, recommended range `1.2`–`1.5`).
- `--repetition-window`: number of recent characters considered for repetition penalty (default `20`).

## Verbosity modes

- `quiet`: only final generated text.
- `normal`: basic run config + final generated text.
- `verbose`: per-step structured output including:
  - current context
  - n-choice distribution
  - chosen n
  - top 10 candidate chars and probabilities
  - chosen char
  - current text

## Probability control order

Character sampling applies in this order:

1. raw conditional probabilities
2. `repetition-penalty`
3. `temperature`
4. `top-k`
5. `top-p`
6. renormalize and sample

For n-value selection:

- mode chooses base distribution (`weighted`, `uniform`, `fixed`, `manual`)
- then `n-temperature` adjusts sharpness
- then sample n

## Bug fixes in this refactor

- Fixed line-range reader bug from legacy jsonl reader script.
- Fixed global variable dependency in legacy JSONL character search printer.
- Removed hard-coded local Windows paths from runnable code.

## Improvements

- **Backoff chain**: when the sampled n-gram order yields no candidates, the model now deterministically backs off to n-1, n-2, … down to 1 instead of random retrying. This ensures the longest valid context is always used first.
- **Sentence boundary awareness**: training splits text on sentence-ending punctuation (`。！？…`) so n-grams never cross sentence boundaries. Punctuation characters are still counted as unigrams.
- **Repetition penalty**: new `--repetition-penalty` parameter down-weights characters that appeared recently in the generated text, reducing repetitive output. Use `--repetition-window` to control how many recent characters are considered.

## Contributing

See `CONTRIBUTING.md`.
