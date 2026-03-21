# Corpus Format

LittleLM consumes JSONL input.

## JSONL requirements

- One JSON object per line.
- UTF-8 encoding.
- Empty lines are ignored.

## Text field

- Default text field name is `Content`.
- You can override it with CLI option `--text-key`, for example `--text-key text`.

## Tokenization and n-gram behavior

LittleLM keeps legacy behavior:

1. Split text by whitespace into fragments.
2. For each fragment, build character-level sliding-window n-grams.
3. Do not cross fragment boundaries when generating n-grams.
