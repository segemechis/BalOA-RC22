"""
constraint_utils.py
====================
Shared constraint-handling utilities used by both BalOA (native Deb's-rule
support) and the 16 unconstrained competitor algorithms in competitors.py
(via scalar penalty wrapping).

Deb's feasibility rules (Deb, 2000):
    1. Any feasible solution is preferred over any infeasible solution.
    2. Between two feasible solutions, the one with better objective wins.
    3. Between two infeasible solutions, the one with smaller total
       constraint violation wins (objective value is ignored).

These rules are parameter-free: no penalty coefficient has to be tuned
or defended.
"""

from __future__ import annotations
from functools import cmp_to_key
from typing import Callable, Sequence
import numpy as np


def deb_worse(fa: float, cva: float, fb: float, cvb: float) -> bool:
    """True if solution A (fa, cva) is strictly worse than B (fb, cvb)
    under Deb's feasibility rules."""
    if cva == 0.0 and cvb == 0.0:
        return fa > fb
    if cva == 0.0:
        return False      # A feasible, B infeasible -> A is not worse
    if cvb == 0.0:
        return True        # A infeasible, B feasible -> A is worse
    return cva > cvb       # both infeasible -> larger violation is worse


def deb_best_index(F: Sequence[float], CV: Sequence[float],
                    idxs: Sequence[int]) -> int:
    """Return the index (from idxs) of the Deb-best entry in F/CV."""
    best = idxs[0]
    for i in idxs[1:]:
        if deb_worse(F[best], CV[best], F[i], CV[i]):
            best = i
    return best


def deb_sort_indices(F: Sequence[float], CV: Sequence[float],
                      idxs: Sequence[int]) -> list[int]:
    """Return idxs sorted best-to-worst under Deb's feasibility rules."""
    def _cmp(i, j):
        if deb_worse(F[i], CV[i], F[j], CV[j]):
            return 1
        if deb_worse(F[j], CV[j], F[i], CV[i]):
            return -1
        return 0
    return sorted(idxs, key=cmp_to_key(_cmp))


def make_penalized_objective(evaluate_fn: Callable[[np.ndarray], tuple[float, float]],
                              big: float) -> Callable[[np.ndarray], float]:
    """
    Wrap a (f, cv) evaluator into a single scalar objective such that
    plain minimisation of the wrapped scalar (using ordinary '<') is
    *exactly* equivalent to applying Deb's feasibility rules, provided
    `big` exceeds the worst possible feasible objective value anywhere
    in the search box (pressure_vessel_problem.PENALTY_BIG is sized for
    this already).

        feasible solution   (cv == 0):  scalar = f(x)            < big
        infeasible solution (cv  > 0):  scalar = big + cv(x)     >= big

    This lets every one of the 16 unconstrained competitor algorithms
    in competitors.py be used completely unmodified internally: they
    just minimise `wrapped(x)` with their existing scalar comparisons,
    and the resulting ranking reproduces Deb's rule precisely.
    """
    def wrapped(x: np.ndarray) -> float:
        f, cv = evaluate_fn(x)
        return f if cv <= 0.0 else big + cv
    return wrapped


def make_logging_objective(evaluate_fn: Callable[[np.ndarray], tuple[float, float]],
                            log: list,
                            big: float | None):
    """
    Wrap `evaluate_fn` so every (f, cv) pair it returns is also appended
    to `log`, in call order -- call order == evaluation order == FES
    order, since every algorithm in this project calls the objective
    exactly once per evaluation it counts toward its FES budget.

    This is PURE instrumentation: it reads `evaluate_fn(x)`'s result and
    appends to `log`, nothing else. It does not call any RNG, does not
    skip or add evaluations, and returns to the caller exactly what the
    unwrapped call would have returned. So for a fixed seed, wrapping an
    algorithm's objective with this function changes NOTHING about that
    algorithm's search trajectory or final result -- the f/cv/best_x it
    ends up reporting are bit-for-bit identical to an unwrapped run with
    the same seed. The only difference is that `log` now also holds the
    full per-evaluation history, which lets a convergence curve be
    reconstructed afterwards by applying Deb's feasibility rule to the
    raw log -- identically for every algorithm, including BalOA (which
    tracks raw (f, cv) natively) and the 16 penalty-wrapped competitors
    (which otherwise only ever see a single penalized scalar).

    `big`:
      - None  -> returns the raw (f, cv) tuple unchanged (for BalOA,
                 which consumes (f, cv) natively).
      - a float -> returns the same penalized scalar
                 `make_penalized_objective` would (for the 16
                 competitors, which consume a single scalar).
    """
    if big is None:
        def wrapped(x: np.ndarray):
            f, cv = evaluate_fn(x)
            log.append((f, cv))
            return f, cv
    else:
        def wrapped(x: np.ndarray) -> float:
            f, cv = evaluate_fn(x)
            log.append((f, cv))
            return f if cv <= 0.0 else big + cv
    return wrapped
