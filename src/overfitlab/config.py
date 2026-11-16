"""Every number that defines the experiments lives here."""

from dataclasses import dataclass

import numpy as np

# --- reproducibility -------------------------------------------------------
MASTER_SEED = 20261116
_STREAMS = {"world": 0, "truth": 1, "bootstrap": 2, "calibrate": 3, "romano_wolf": 4}


def seed(stream: str, *key: int) -> np.random.SeedSequence:
    """Independent random stream for (stream, key), all derived from MASTER_SEED.

    SeedSequence with a spawn_key is what `SeedSequence.spawn` produces
    internally; building it directly lets us regenerate world 137 without
    generating worlds 0..136 first.
    """
    return np.random.SeedSequence(MASTER_SEED, spawn_key=(_STREAMS[stream], *key))


# --- market ----------------------------------------------------------------
N_ASSETS = 10
N_DAYS = 2520  # 10 years of daily data per world
PERIODS_PER_YEAR = 252
FACTOR_VOL = 0.12  # annual vol of the common factor
IDIO_VOL = 0.20  # annual vol of each asset's own noise
BETA_RANGE = (0.6, 1.4)  # factor loadings, evenly spaced across assets
STUDENT_T_DF = 6  # fat tails: Student-t shocks
GARCH_ALPHA = 0.07  # reaction of variance to yesterday's squared shock
GARCH_BETA = 0.90  # persistence of variance
# With these three values the return's fourth moment is finite (theoretical
# kurtosis about 10, like daily ETF returns), so Sharpe estimates behave.

# --- planted signals -------------------------------------------------------
SIGNAL_LOOKBACK = {"tsmom": 60, "xsmom": 60, "meanrev": 5}
# Target: annualised Sharpe, after costs, of the oracle that trades the
# planted signal itself. Kappa values come from `python -m overfitlab calibrate`.
STRENGTH_TARGETS = {"barely": 0.3, "weak": 0.6, "medium": 1.0, "obvious": 2.0}
KAPPA = {
    "tsmom": {"barely": 0.0137, "weak": 0.0216, "medium": 0.0316, "obvious": 0.0534},
    "xsmom": {"barely": 0.0129, "weak": 0.0193, "medium": 0.0273, "obvious": 0.0452},
    "meanrev": {"barely": 0.0283, "weak": 0.039, "medium": 0.0536, "obvious": 0.0916},
}


@dataclass(frozen=True)
class WorldSpec:
    """One type of simulated market. `kappa` is the planted signal's strength."""

    name: str
    signal: str = "none"  # none | tsmom | xsmom | meanrev
    strength: str = "none"
    kappa: float = 0.0


# The null world is labelled "no_signal": pandas reads the text "null" in a
# CSV as a missing value.
SPECS = [WorldSpec("no_signal")] + [
    WorldSpec(f"{signal}_{strength}", signal, strength, KAPPA[signal][strength])
    for signal in KAPPA
    for strength in STRENGTH_TARGETS
]

# --- backtests and experiments ---------------------------------------------
COST_BPS = 5.0  # per unit of one-way turnover
RESEARCH_FRAC = 0.7  # the last 30% is the lockbox
N_RESEARCH = int(N_DAYS * RESEARCH_FRAC)
SCREEN_SHARPE = 1.0  # in-sample Sharpe a strategy needs before it is tested
TRUE_SHARPE_MIN = 0.2  # a "true" strategy earns at least this, long run, net
TRUTH_THRESHOLDS = (0.0, 0.2, 0.5)  # sensitivity check
TRUTH_PATHS, TRUTH_YEARS = 100, 100  # 10,000 simulated years per world type
N_WORLDS = 200
N_BOOTSTRAP = 2000
RW_RESAMPLES = 1000  # Romano-Wolf bootstrap resamples
RW_BLOCK = 20  # block length in days, about one half-life of a volatility shock
