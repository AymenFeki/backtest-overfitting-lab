# Model-choice pilot: status on 2026-11-16

Pilot = one agent run of 20 rounds per world, on world 0 of `no_signal` and
`tsmom_medium`, compact history, one worked example in the prompt, up to two
repair attempts per round. The gate is about 80% usable rounds.

## Local models: none passes

| Model | Usable rounds | Usable at first attempt | Attempts per round | Look-ahead rejections | Distinct strategies | Seconds per round |
|---|---|---|---|---|---|---|
| qwen2.5-coder:7b (4.7 GB) | 4 of 40 (10%) | 3 | 2.85 | 19 | 4 | 39 |
| mimo-v2.6:9b-q4_K_M (6.6 GB) | 3 of 20 (15%) | 1 | 2.85 | 10 | 3 | 50 |
| qwen2.5-coder:14b (9.0 GB) | not run: out of memory, machine crashed | | | | | |
| rnj-1:8b (5.1 GB) | not run: Ollama 0.40.13 and 0.40.16 fail to load it | | | | | |

- mimo was stopped after its first world (`no_signal`) on request, so its row
  has 20 rounds, not 40.
- Earlier setups of qwen2.5-coder:7b with the growing transcript: 5 of 60 (8%)
  with the first prompt and no repairs, 3 of 40 (8%) with example and repairs.
- Typical failures of both models: output of the wrong shape (one column, or
  rows and columns mixed up), pandas arguments that do not exist (`keepdims`,
  `keep`, `axis` on `Rolling.sum`), and dividing by a full-sample sum, which
  is a real look-ahead and is correctly rejected.
- At these speeds the 1,200 rounds planned for Stage 4 would take about 13
  hours (7B) or 17 hours (mimo) and yield 10% to 15% usable strategies.
- With mimo loaded at context 8192, free memory was 17% to 21% and macOS
  reported its "warning" pressure level. It ran without incident under a
  watchdog, but it is at the edge of what this machine holds.
- Removed from disk with approval: qwen2.5-coder:14b, rnj-1:8b. Still
  installed: qwen2.5-coder:7b, mimo-v2.6:9b-q4_K_M.

## Haiku 5.5: pilot not run yet

It needs `ANTHROPIC_API_KEY` in the environment of the shell that runs it.
No API call has been made and nothing has been spent.

```bash
read -s "ANTHROPIC_API_KEY?Anthropic API key (hidden, press Enter): " && export ANTHROPIC_API_KEY
```

```bash
uv run python -m overfitlab pilot --provider anthropic --live
```

The code refuses any call that could take the total past $2.00. The result
lands in `results/pilot/claude-haiku-5-5__compact.csv` and the replies in
`cache/llm/`.

## Files

- `*__compact.csv`, `*__transcript.csv`: one row per round.
- Model replies: `cache/llm/*.jsonl` (the 7B file also holds the replies of
  the earlier prompt versions).
