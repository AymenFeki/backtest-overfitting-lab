"""Run strategy code written by a language model, with guard rails.

Two layers:

1. `check_code` reads the code without running it (via its syntax tree) and
   rejects anything outside a small allowed set: only numpy and pandas, no
   file or network access, no names starting with an underscore.
2. `run_strategy` runs the code in a separate Python process with a time
   limit, so a crash or an endless loop cannot take the experiment down.

In that process the strategy must also pass the look-ahead test before its
weights are handed back. One automatic repair is made and no other: a NumPy
array of exactly the prices' shape is accepted and given the prices' index
and columns (missing weights count as zero, as in the engine). This guards against accidents. It is not a hardened
security boundary: do not use it for code from an untrusted person.
"""

import ast
import json
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from .lookahead import passes_lookahead_test

TIMEOUT_SECONDS = 20.0
_ALLOWED_IMPORTS = {"numpy", "pandas"}
_BANNED_NAMES = {
    "open",
    "exec",
    "eval",
    "compile",
    "getattr",
    "setattr",
    "delattr",
    "globals",
    "locals",
    "vars",
    "input",
    "breakpoint",
    "exit",
    "quit",
}
_BANNED_ATTRIBUTES = {
    "eval",
    "query",
    "save",
    "savez",
    "savez_compressed",
    "savetxt",
    "load",
    "loadtxt",
    "fromfile",
    "tofile",
    "memmap",
    "genfromtxt",
    "DataSource",
}
_ALLOWED_TO = {"to_numpy", "to_frame", "to_series", "to_numeric", "to_list"}
_BUILTINS = [
    "abs",
    "all",
    "any",
    "bool",
    "dict",
    "enumerate",
    "float",
    "int",
    "isinstance",
    "len",
    "list",
    "map",
    "max",
    "min",
    "pow",
    "print",
    "range",
    "reversed",
    "round",
    "set",
    "slice",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
    "ValueError",
    "TypeError",
    "ZeroDivisionError",
    "Exception",
    "True",
    "False",
    "None",
]


def check_code(source: str) -> str | None:
    """Reason the code is not allowed, or None if it may run."""
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return f"syntax error: {error.msg} (line {error.lineno})"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import | ast.ImportFrom):
            modules = (
                [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else [alias.name for alias in node.names]
            )
            for module in modules:
                if module.split(".")[0] not in _ALLOWED_IMPORTS:
                    return f"import of {module!r} is not allowed (numpy, pandas only)"
        elif isinstance(node, ast.Name):
            if node.id in _BANNED_NAMES or node.id.startswith("_"):
                return f"name {node.id!r} is not allowed"
        elif isinstance(node, ast.Attribute):
            name = node.attr
            banned = name in _BANNED_ATTRIBUTES or name.startswith(("_", "read_"))
            if banned or (name.startswith("to_") and name not in _ALLOWED_TO):
                return f"attribute {name!r} is not allowed"
    defined = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    if "positions" not in defined:
        return "no function named positions"
    return None


def _launch(work: Path, timeout: float) -> str:
    """Run the strategy file in `work` in its own process; return the status."""
    command = [sys.executable, "-m", "overfitlab.sandbox", str(work)]
    try:
        done = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=work,
            env={},  # the strategy process sees no API key or other secret
        )
    except subprocess.TimeoutExpired:
        return f"too slow: no result within {timeout:.0f} seconds"
    try:
        return json.loads(done.stdout.strip().splitlines()[-1])["status"]
    except (IndexError, ValueError, KeyError):
        return "error: the process crashed"


def run_strategy(
    source: str, prices: pd.DataFrame, timeout: float = TIMEOUT_SECONDS
) -> tuple[pd.DataFrame | None, str]:
    """Weights from `positions(prices)` and a status: "ok" or what went wrong."""
    reason = check_code(source)
    if reason:
        return None, f"not allowed: {reason}"
    with tempfile.TemporaryDirectory() as folder:
        work = Path(folder)
        (work / "strategy.py").write_text(source)
        np.save(work / "prices.npy", prices.to_numpy())
        status = _launch(work, timeout)
        if status != "ok":
            return None, status
        weights = np.load(work / "weights.npy", allow_pickle=False)
    return pd.DataFrame(weights, index=prices.index, columns=prices.columns), "ok"


def run_on_paths(
    sources: list[str], prices: np.ndarray, timeout: float = 600.0
) -> Iterator[np.ndarray | None]:
    """Weights of several accepted strategies on several price paths.

    `prices` has shape (paths, days, assets). Used to measure a strategy's
    true Sharpe on fresh data, so the look-ahead test is not repeated. One
    process per strategy; a strategy that fails here gives None.

    This is a generator: it hands over one strategy's weights at a time. The
    caller must use them and let them go before asking for the next, because
    one array is about 20 MB and a world type has hundreds of strategies.
    """
    with tempfile.TemporaryDirectory() as folder:
        work = Path(folder)
        np.save(work / "prices.npy", prices)
        for source in sources:
            (work / "strategy.py").write_text(source)
            usable = check_code(source) is None and _launch(work, timeout) == "ok"
            yield np.load(work / "weights.npy") if usable else None


def _safe_import(name: str, *args: object, **kwargs: object) -> object:
    if name.split(".")[0] not in _ALLOWED_IMPORTS:
        raise ImportError(f"import of {name!r} is not allowed")
    return __import__(name, *args, **kwargs)


def _load(source: str) -> object:
    """Execute the strategy file with a reduced set of built-in functions."""
    import builtins

    allowed = {name: getattr(builtins, name) for name in _BUILTINS}
    namespace = {"__builtins__": allowed | {"__import__": _safe_import}}
    namespace |= {"np": np, "pd": pd}
    exec(compile(source, "strategy.py", "exec"), namespace)  # noqa: S102
    return namespace["positions"]


def _child(work: Path) -> str:
    """Runs inside the separate process. Returns the status."""
    prices = np.load(work / "prices.npy")
    try:
        function = _load((work / "strategy.py").read_text())

        def positions(table: pd.DataFrame) -> pd.DataFrame:
            weights = np.asarray(function(table.copy()), dtype=float)
            if weights.shape != table.shape:
                raise ValueError(
                    f"weights have shape {weights.shape}, prices {table.shape}"
                )
            return pd.DataFrame(np.nan_to_num(weights, nan=0.0))

        def table(path: np.ndarray) -> pd.DataFrame:
            return pd.DataFrame(path, columns=[f"A{i}" for i in range(path.shape[1])])

        if prices.ndim == 3:  # many paths: weights only, see run_on_paths
            weights = np.stack([positions(table(path)).to_numpy() for path in prices])
        else:
            weights = positions(table(prices)).to_numpy()
            if not weights.any():
                return "error: all weights are zero, the strategy never trades"
            if np.isfinite(weights).all() and not passes_lookahead_test(
                positions, table(prices)
            ):
                return (
                    "look-ahead: weights for a day change when later prices change "
                    "(or the function is random)"
                )
        if not np.isfinite(weights).all():
            return "error: weights contain infinite values"
    except Exception as error:  # noqa: BLE001  any failure is reported to the agent
        return f"error: {type(error).__name__}: {str(error)[:300]}"
    np.save(work / "weights.npy", weights)
    return "ok"


if __name__ == "__main__":
    print(json.dumps({"status": _child(Path(sys.argv[1]))}))
