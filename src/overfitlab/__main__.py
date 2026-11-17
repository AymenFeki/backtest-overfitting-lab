"""Command line: `uv run python -m overfitlab all` reruns every experiment."""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg
from .agent import run_agent
from .agent_experiment import agent_truth, generate, score_runs, summarise_agent
from .backtest import net_returns, sharpe
from .experiments import grid_scores, summarise, true_sharpes
from .llm import Model
from .plots import (
    AGENT_PANELS,
    AGENT_TITLE,
    LABEL,
    VARIANT_PANELS,
    VARIANT_TITLE,
    comparison_figure,
    fdr_power_figure,
)
from .registry import GAP_FILTERS
from .strategies import oracle
from .worlds import draw_shocks, simulate, to_prices, world

# Tables written to results/exp1_grid_tables.md: title -> {column: group of worlds}
TABLES = {
    "Plain worlds": {
        "No signal": "no_signal",
        "Barely": "all_barely",
        "Weak": "all_weak",
        "Medium": "all_medium",
        "Obvious": "all_obvious",
    },
    "Regime switching": {
        "No signal, plain": "no_signal",
        "No signal, regimes": "no_signal_regimes",
        "Medium, plain": "all_medium",
        "Medium, regimes": "all_medium_regimes",
        "Medium, regimes, signal rescaled": "all_medium_regimes_matched",
    },
    "Alpha decay": {
        "Obvious, plain": "all_obvious",
        "Obvious, decay": "all_obvious_decay",
    },
}


AGENT_TABLES = {
    "Grid search against the LLM agent": {
        "No signal, grid": "grid_no_signal",
        "No signal, agent": "agent_no_signal",
        "Medium, grid": "grid_all_medium",
        "Medium, agent": "agent_all_medium",
    },
    "LLM agent by signal type": {
        "Momentum": "agent_tsmom_medium",
        "Cross-sectional": "agent_xsmom_medium",
        "Mean reversion": "agent_meanrev_medium",
    },
}


def markdown_tables(points: pd.DataFrame, tables: dict = TABLES) -> str:
    """Each filter at its conventional cut: FDR / pFDR / power in percent."""
    points = points.set_index(["worlds", "filter"])
    lines = ["Cells are FDR / pFDR / power in percent; `-` means undefined.", ""]
    for title, columns in tables.items():
        lines += [f"## {title}", "", "| Filter | " + " | ".join(columns) + " |"]
        lines.append("|---" * (len(columns) + 1) + "|")
        for name, label in LABEL.items():
            cells = []
            for group in columns.values():
                row = points.loc[(group, name)]
                values = (row.fdr, row.pfdr, row.power)
                cells.append(
                    " / ".join("-" if np.isnan(v) else f"{100 * v:.1f}" for v in values)
                )
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def run_all(quick: bool, jobs: int) -> None:
    """Truth, Experiment 1 with grid search, tables and figure."""
    out = Path("results/quick" if quick else "results")
    (out / "figures").mkdir(parents=True, exist_ok=True)
    n_paths = 10 if quick else cfg.TRUTH_PATHS
    n_worlds = 20 if quick else cfg.N_WORLDS

    start = time.time()
    truth = true_sharpes(n_paths, jobs)
    truth.to_csv(out / "truth.csv", index=False, float_format="%.4f")
    print(f"truth: {n_paths} paths per world type, {time.time() - start:.0f}s")

    start = time.time()
    scores = grid_scores(n_worlds, jobs)
    print(f"scores: {n_worlds} worlds per world type, {time.time() - start:.0f}s")

    points = pd.concat(
        summarise(scores, truth, threshold, sweep=False)
        for threshold in cfg.TRUTH_THRESHOLDS
    )
    points.to_csv(out / "exp1_grid_points.csv", index=False, float_format="%.4f")
    curves = summarise(scores, truth, cfg.TRUE_SHARPE_MIN, sweep=True)
    curves.to_csv(out / "exp1_grid_curves.csv", index=False, float_format="%.4f")
    main_points = points[points.true_threshold == cfg.TRUE_SHARPE_MIN]
    fdr_power_figure(curves, main_points, out / "figures" / "exp1_grid_fdr_power.png")
    figure = out / "figures" / "exp1_grid_regimes_decay.png"
    comparison_figure(main_points, figure, VARIANT_PANELS, VARIANT_TITLE)
    (out / "exp1_grid_tables.md").write_text(markdown_tables(main_points))

    # CPCV with wider gaps around the test groups, on the first 50 worlds.
    gap_scores = grid_scores(5 if quick else 50, jobs, GAP_FILTERS)
    gaps = summarise(gap_scores, truth, cfg.TRUE_SHARPE_MIN, False, GAP_FILTERS)
    gaps.to_csv(out / "exp1_cpcv_gap.csv", index=False, float_format="%.4f")
    print(f"results written to {out}/")
    run_agent_experiment(quick, jobs)


def run_agent_experiment(quick: bool, jobs: int, stored_truth: bool = False) -> None:
    """Experiment 1 with the LLM agent, from stored replies (no model call)."""
    out = Path("results/quick" if quick else "results")
    model = Model(cfg.AGENT_PROVIDER, cfg.API_MODEL)
    n_seeds = 2 if quick else cfg.AGENT_SEEDS

    start = time.time()
    runs = score_runs(model, n_seeds, jobs)
    print(f"agent: {len(runs)} runs replayed and scored, {time.time() - start:.0f}s")
    start = time.time()
    if stored_truth:  # the truth depends on the strategies only, not the filters
        truth = pd.read_csv(out / "exp1_agent_truth.csv")
    else:
        truth = agent_truth(runs, 10 if quick else cfg.TRUTH_PATHS, jobs)
        truth.to_csv(out / "exp1_agent_truth.csv", index=False, float_format="%.4f")
    print(f"agent truth: {len(truth)} strategies, {time.time() - start:.0f}s")

    per_run = pd.DataFrame(
        {
            "world_type": cfg.SPECS[run["spec"]].name,
            "world": run["world"],
            "usable": len(run["rounds"]),
            "attempts": run["attempts"],
            "lookahead_rejections": run["lookahead"],
            "distinct": run["distinct"],
            "seconds": run["seconds"],
            "cost_usd": run["cost"],
        }
        for run in runs
    )
    per_run.to_csv(out / "exp1_agent_runs.csv", index=False, float_format="%.4f")
    points = pd.concat(
        summarise_agent(runs, truth, threshold) for threshold in cfg.TRUTH_THRESHOLDS
    )
    points.to_csv(out / "exp1_agent_points.csv", index=False, float_format="%.4f")

    # Side by side with the grid search on the same world types.
    grid = pd.read_csv(out / "exp1_grid_points.csv")
    both = pd.concat(
        [
            grid.assign(worlds="grid_" + grid.worlds),
            points.assign(worlds="agent_" + points.worlds),
        ]
    )
    both = both[both.true_threshold == cfg.TRUE_SHARPE_MIN]
    (out / "exp1_agent_tables.md").write_text(markdown_tables(both, AGENT_TABLES))
    figure = out / "figures" / "exp1_agent_vs_grid.png"
    comparison_figure(both, figure, AGENT_PANELS, AGENT_TITLE)
    print(f"agent results written to {out}/")


def calibrate(n_paths: int = 50) -> None:
    """Find the kappa that gives each target oracle Sharpe, by interpolation.

    Prints one table for the plain worlds (config.KAPPA) and one for worlds
    with regime switching (its "medium" column is config.KAPPA_REGIMES_MATCHED).
    Uses its own random stream.
    """
    kappas = np.array([0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1])
    seeds = [cfg.seed("calibrate", p) for p in range(n_paths)]
    shocks = draw_shocks(seeds, cfg.TRUTH_YEARS * cfg.PERIODS_PER_YEAR)
    for regimes in (False, True):
        print("with regime switching:" if regimes else "plain worlds:")
        for signal in cfg.KAPPA:
            measured = []
            for kappa in kappas:
                spec = cfg.WorldSpec(
                    "calibration", signal, "calibration", float(kappa), regimes
                )
                trader = oracle(spec)
                per_path = [
                    sharpe(net_returns(prices, trader.positions(prices)))
                    for prices in map(to_prices, simulate(spec, shocks))
                ]
                measured.append(np.mean(per_path))
            chosen = np.interp(list(cfg.STRENGTH_TARGETS.values()), measured, kappas)
            table = dict(
                zip(cfg.STRENGTH_TARGETS, chosen.round(4).tolist(), strict=True)
            )
            print(f'"{signal}": {table},')


PILOT_WORLDS = ["no_signal", "tsmom_medium"]


def pilot(model: Model, live: bool, history: str) -> None:
    """A small agent run to choose a model: one world of two types, 20 rounds.

    Writes one row per round to results/pilot/ and prints the share of usable
    rounds, look-ahead rejections, distinct strategies and the speed. Replies
    are stored, so a second run makes no model call.
    """
    rows, distinct, wall = [], 0, []
    for name in PILOT_WORLDS:
        spec_index = [s.name for s in cfg.SPECS].index(name)
        prices = world(cfg.SPECS[spec_index], 0)
        start = time.time()
        trials = run_agent(
            prices, model, seed=spec_index * 1000, live=live, history=history
        )
        wall.append((time.time() - start) / len(trials))
        usable = pd.DataFrame({t.round: t.returns for t in trials if t.status == "ok"})
        # Distinct: not a near copy (correlation above 0.99) of an earlier round.
        copies = np.triu(usable.corr().abs().to_numpy(), k=1) > 0.99
        distinct += int((~copies.any(axis=0)).sum())
        for trial in trials:
            research = None
            if trial.returns is not None:
                research = float(sharpe(trial.returns.iloc[: cfg.N_RESEARCH]))
            rows.append(
                {
                    "world": name,
                    "round": trial.round,
                    "status": trial.status.split(":")[0],
                    "attempts": len(trial.replies),
                    "lookahead_rejections": sum(
                        f.startswith("look-ahead") for f in trial.failures
                    ),
                    "research_sharpe": research,
                    "input_tokens": max(r.input_tokens for r in trial.replies),
                    "output_tokens": sum(r.output_tokens for r in trial.replies),
                    "seconds": sum(r.seconds for r in trial.replies),
                    "cost_usd": sum(r.cost_usd for r in trial.replies),
                }
            )
            print(rows[-1], flush=True)
    table = pd.DataFrame(rows)
    out = Path("results/pilot")
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / f"{model.name.replace(':', '_')}__{history}.csv", index=False)
    usable = table.status == "ok"
    print(f"\n{model.name}, {history} history")
    print(f"usable rounds: {usable.mean():.0%} of {len(table)}")
    print(f"usable at the first attempt: {(usable & (table.attempts == 1)).mean():.0%}")
    print(f"attempts per round: {table.attempts.mean():.2f}")
    print(f"look-ahead rejections: {table.lookahead_rejections.sum()}")
    print(f"distinct strategies: {distinct}")
    print(table.status.value_counts().to_string())
    half = len(table) // 2
    parts = (("first", table[:half]), ("second", table[half:]))
    for (label, part), seconds in zip(parts, wall, strict=True):
        speed = part.output_tokens.sum() / part.seconds.sum()
        print(
            f"{label} world: {seconds:.1f} s per round including backtests, "
            f"{speed:.1f} tokens per second"
        )
    print(f"largest prompt: {table.input_tokens.max()} tokens")
    print(
        f"tokens per round: {table.input_tokens.mean():.0f} in (largest call), "
        f"{table.output_tokens.mean():.0f} out"
    )
    print(f"cost: ${table.cost_usd.sum():.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="overfitlab")
    parser.add_argument(
        "command", choices=["all", "calibrate", "pilot", "generate", "agent"]
    )
    parser.add_argument("--quick", action="store_true", help="small run for a check")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--provider", default="ollama", help="ollama or anthropic")
    parser.add_argument("--model", help="model name; default from the config")
    parser.add_argument("--live", action="store_true", help="allow new model calls")
    parser.add_argument(
        "--stored-truth",
        action="store_true",
        help="agent: reuse exp1_agent_truth.csv instead of simulating it again",
    )
    parser.add_argument(
        "--history", default=cfg.AGENT_HISTORY, choices=["compact", "transcript"]
    )
    args = parser.parse_args()
    if args.command == "all":
        run_all(args.quick, args.jobs)
    elif args.command == "calibrate":
        calibrate()
    elif args.command == "generate":  # the only command that calls the API
        generate(Model(cfg.AGENT_PROVIDER, cfg.API_MODEL))
    elif args.command == "agent":
        run_agent_experiment(args.quick, args.jobs, args.stored_truth)
    else:
        default = cfg.LOCAL_MODEL if args.provider == "ollama" else cfg.API_MODEL
        pilot(Model(args.provider, args.model or default), args.live, args.history)


if __name__ == "__main__":
    main()
