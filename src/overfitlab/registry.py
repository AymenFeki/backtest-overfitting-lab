"""The list of filters every experiment runs, in a fixed order."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd

from . import config as cfg
from .cpcv import cpcv
from .dsr import deflated_sharpe, deflated_sharpe_null
from .filters import holdout, naive_sharpe, walk_forward
from .multiple import bh_filter, by_filter, romano_wolf_filter
from .pbo import pbo_filter


@dataclass(frozen=True)
class Filter:
    name: str
    score: Callable[[pd.DataFrame], np.ndarray]
    cut: float  # conventional setting: accept if score > cut
    sweep: np.ndarray  # cuts used to trace the FDR-versus-power curve


_CONFIDENCE = 1 - np.array([0.5, 0.3, 0.2, 0.1, 0.05, 0.025, 0.01, 0.005, 1e-3, 1e-4])
FILTERS = [
    Filter("naive", naive_sharpe, cfg.SCREEN_SHARPE, np.linspace(0, 3, 31)),
    Filter("holdout", holdout, 0.95, _CONFIDENCE),
    Filter("walk_forward", walk_forward, 0.95, _CONFIDENCE),
    Filter("dsr", deflated_sharpe, 0.95, _CONFIDENCE),
    Filter("dsr_null", deflated_sharpe_null, 0.95, _CONFIDENCE),
    Filter("cpcv", cpcv, 0.0, np.linspace(0, 2, 21)),
    Filter("bh", bh_filter, 0.95, _CONFIDENCE),
    Filter("by", by_filter, 0.95, _CONFIDENCE),
    Filter("romano_wolf", romano_wolf_filter, 0.95, _CONFIDENCE),
    Filter("pbo", pbo_filter, 0.95, _CONFIDENCE),
]

# Sensitivity check: the same CPCV rule with wider gaps around the test groups.
GAP_FILTERS = [
    Filter(f"cpcv_gap_{gap}", partial(cpcv, gap=gap), 0.0, np.array([0.0]))
    for gap in (20, 60, 250)
]
