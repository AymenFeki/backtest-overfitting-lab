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

## Claude Haiku 5.5: passes

Same pilot (2 worlds x 20 rounds), thinking switched off, key loaded with
`uv run --env-file .env`.

| Measure | Result |
|---|---|
| Usable rounds | 40 of 40 (100%), all at the first attempt |
| Look-ahead rejections | 0 |
| Distinct strategies | 38 of 40 |
| Seconds per round, including the backtest | 2.7 |
| Tokens per round | about 1,320 in, 390 out |
| Cost | $0.0144 for the two runs ($0.0072 per run, $0.00036 per round) |

Research-sample Sharpe of its strategies: median 0.11 and best 0.55 on the
no-signal world, median 0.60 and best 1.02 on the momentum world. Its
hypotheses refer to earlier rounds by number and result, so it does adapt.

Estimates from these measured rates, for the plan's tentative sizes:

| Runs | Rounds | Cost | Model time |
|---|---|---|---|
| Stage 4, experiment 1: 4 world types x 15 seeds x 20 rounds | 1,200 | about $0.43 | about 55 minutes |
| Stage 5, experiment 2: 2 arms x 15 seeds x 40 rounds | 1,200 | about $0.49 (longer tables) | about 55 minutes |
| Stage 5, experiment 3: 2 arms x 15 seeds x 20 rounds | 600 | about $0.23 | about 27 minutes |
| All three | 3,000 | about $1.15 | about 2.3 hours |

Spent so far: $0.0144 of the $2.00 cap. At the measured rate the rest pays
for about 275 runs of 20 rounds; Stage 4 needs 60. If every round needed all
three attempts with replies at the 600-token limit, Stage 4 would cost about
$1.80, still under the cap, which the code enforces before each call.

Not included: computing the true Sharpe of each agent strategy on fresh
simulated paths. That is local CPU time, not API cost, and is not measured yet.

To repeat the pilot without any API call (stored replies):

```bash
uv run python -m overfitlab pilot --provider anthropic
```

## Files

- `*__compact.csv`, `*__transcript.csv`: one row per round.
- Model replies: `cache/llm/*.jsonl` (the 7B file also holds the replies of
  the earlier prompt versions).
