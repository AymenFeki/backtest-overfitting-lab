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
    regimes: bool = False  # calm/crisis switching
    half_life_years: float = 0.0  # alpha decay; 0 = the signal never fades
    variant: str = "base"  # base | regimes | regimes_matched | decay


# The null world is labelled "no_signal": pandas reads the text "null" in a
# CSV as a missing value.
SPECS = [WorldSpec("no_signal")] + [
    WorldSpec(f"{signal}_{strength}", signal, strength, KAPPA[signal][strength])
    for signal in KAPPA
    for strength in STRENGTH_TARGETS
]

# --- regime switching and alpha decay (Stage 3) ----------------------------
# A hidden two-state Markov chain. Daily switching probabilities: calm spells
# last about 250 days, crises about 50, so about one day in six is a crisis.
REGIME_ENTER, REGIME_EXIT = 0.004, 0.02
# In a crisis the common factor's volatility triples and asset-specific
# volatility rises by half, which lifts the average correlation from about
# 0.26 to about 0.59.
CRISIS_FACTOR_VOL, CRISIS_IDIO_VOL = 3.0, 1.5
DECAY_HALF_LIFE_YEARS = 5.0  # the planted signal halves every 5 years
DEPLOYMENT_YEARS = 5  # with decay, truth is the Sharpe over the 5 years after

SPECS += [
    WorldSpec("no_signal_regimes", regimes=True, variant="regimes"),
    *(
        WorldSpec(
            f"{signal}_medium_regimes",
            signal,
            "medium",
            KAPPA[signal]["medium"],
            regimes=True,
            variant="regimes",
        )
        for signal in KAPPA
    ),
    *(
        WorldSpec(
            f"{signal}_obvious_decay",
            signal,
            "obvious",
            KAPPA[signal]["obvious"],
            half_life_years=DECAY_HALF_LIFE_YEARS,
            variant="decay",
        )
        for signal in KAPPA
    ),
]

# Regime switching makes returns noisier, so the same kappa gives a weaker
# signal (oracle Sharpe 0.64 to 0.92 instead of 1.0). These kappas are
# recalibrated with regimes on, so the oracle earns Sharpe 1.0 again: the
# "medium" column of the second table printed by `python -m overfitlab calibrate`.
KAPPA_REGIMES_MATCHED = {"tsmom": 0.0432, "xsmom": 0.0287, "meanrev": 0.0751}
# Appended last: a world's seed depends on its position in SPECS.
SPECS += [
    WorldSpec(
        f"{signal}_medium_regimes_matched",
        signal,
        "medium",
        kappa,
        regimes=True,
        variant="regimes_matched",
    )
    for signal, kappa in KAPPA_REGIMES_MATCHED.items()
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

# --- language-model agent (Stage 4) ----------------------------------------
AGENT_ROUNDS = 20  # strategies one agent may try in one world
LOCAL_MODEL = "qwen2.5-coder:7b"
OLLAMA_URL = "http://localhost:11434"
# First characters of each model's digest (`ollama list` shows them). A live
# call stops if the installed weights are a different version.
OLLAMA_DIGESTS = {"qwen2.5-coder:7b": "dae161e27b0e"}
OLLAMA_CONTEXT = 16384  # tokens the model can see; Ollama's default cuts silently
LLM_TEMPERATURE = 0.7  # some randomness, so 20 rounds give 20 different ideas
LLM_MAX_TOKENS = 600  # per reply
API_MODEL = "claude-haiku-5-5"
API_PRICE_PER_MTOK = {"claude-haiku-5-5": (0.10, 0.50)}  # dollars: input, output
API_BUDGET_USD = 2.00  # hard cap on all API calls together
