# Training

## Command

```bash
python -m littlelm.train \
  --input examples/tiny_corpus.jsonl \
  --output artifacts/dictionary \
  --min-n 1 \
  --max-n 3 \
  --text-key text
```

## Output

For each n in `[min_n, max_n]`, one JSONL file is produced:

- `1-gram.jsonl`
- `2-gram.jsonl`
- ...

Each line format:

```json
{"ngram": "示例", "count": 123}
```

## Directory suggestion

- `data/raw/` for local corpora (ignored)
- `data/processed/` for preprocessed corpora (ignored)
- `artifacts/dictionary/` for generated dictionaries (ignored)
