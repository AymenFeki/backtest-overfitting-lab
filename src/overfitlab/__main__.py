"""Command line: `uv run python -m overfitlab all` reruns every experiment."""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg
from .backtest import net_returns, sharpe
from .experiments import grid_scores, summarise, true_sharpes
from .plots import fdr_power_figure
from .strategies import oracle
from .worlds import draw_shocks, simulate, to_prices


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
    print(f"results written to {out}/")


def calibrate(n_paths: int = 50) -> None:
    """Find the kappa that gives each target oracle Sharpe, by interpolation.

    Prints the table to paste into config.KAPPA. Uses its own random stream.
    """
    kappas = np.array([0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1])
    seeds = [cfg.seed("calibrate", p) for p in range(n_paths)]
    shocks = draw_shocks(seeds, cfg.TRUTH_YEARS * cfg.PERIODS_PER_YEAR)
    for signal in cfg.KAPPA:
        measured = []
        for kappa in kappas:
            spec = cfg.WorldSpec("calibration", signal, "calibration", float(kappa))
            trader = oracle(spec)
            per_path = [
                sharpe(net_returns(prices, trader.positions(prices)))
                for prices in map(to_prices, simulate(spec, shocks))
            ]
            measured.append(np.mean(per_path))
        chosen = np.interp(list(cfg.STRENGTH_TARGETS.values()), measured, kappas)
        table = dict(zip(cfg.STRENGTH_TARGETS, chosen.round(4).tolist(), strict=True))
        print(f'"{signal}": {table},')


def main() -> None:
    parser = argparse.ArgumentParser(prog="overfitlab")
    parser.add_argument("command", choices=["all", "calibrate"])
    parser.add_argument("--quick", action="store_true", help="small run for a check")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args()
    if args.command == "all":
        run_all(args.quick, args.jobs)
    else:
        calibrate()


if __name__ == "__main__":
    main()
