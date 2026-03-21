# Tuning

## Output verbosity

- `quiet`: print only final generated text.
- `normal`: print basic configuration and final text.
- `verbose`: print step-by-step structure:
	- step index
	- current context
	- n-choice distribution
	- chosen n
	- top 10 candidate chars and probabilities
	- chosen char
	- current text

## Interactive seed input

- If `--seed` is provided, it is used directly.
- If `--seed` is missing, sample enters interactive seed input.
- `--prompt-mode plain`: plain input prompt.
- `--prompt-mode random`: show a random writing prompt before input.

## Character sampling controls

- `--temperature`:
	- `< 1.0` more conservative
	- `> 1.0` more diverse
- `--top-k`: keep only highest-k candidate chars (`0` means disabled).
- `--top-p`: keep smallest set whose cumulative probability reaches p.

Character filtering order:

1. raw conditional probability
2. temperature scaling
3. top-k filtering
4. top-p filtering
5. renormalize and sample

## n-selection controls

- `--n-selection-mode weighted`: legacy-like weighted strategy.
- `--n-selection-mode uniform`: equal probability across available n.
- `--n-selection-mode fixed`: always use `--fixed-n`.
- `--n-selection-mode manual`: use `--n-weights`, for example `1:0.1,2:0.2,3:0.7`.
- `--n-temperature`: sharpness control for n-selection distribution.

Notes:

- In `fixed` mode, `--fixed-n` is required.
- In `manual` mode, `--n-weights` is required.
- Missing n values in manual weights are treated as zero and ignored after normalization.

## Example commands

```bash
python -m littlelm.sample --dictionary artifacts/dictionary --seed "今天" --max-n 3 --max-steps 20
python -m littlelm.sample --dictionary artifacts/dictionary --interactive --prompt-mode random --verbosity normal
python -m littlelm.sample --dictionary artifacts/dictionary --seed "我" --verbosity verbose --temperature 0.8 --top-k 10
python -m littlelm.sample --dictionary artifacts/dictionary --seed "刚刚" --n-selection-mode fixed --fixed-n 3
python -m littlelm.sample --dictionary artifacts/dictionary --interactive --prompt-mode random --n-selection-mode manual --n-weights "1:0.05,2:0.1,3:0.25,4:0.6"
```
