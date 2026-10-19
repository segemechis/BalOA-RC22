"""
rc22_friedman.py
=======================
Friedman omnibus rank test for the RC22 (Planetary Gear Train Design)
comparison -- all 17 algorithms (BalOA + 16 competitors) together.

Why this script exists
-----------------------
rc22_significance.py already answers "is BalOA significantly better
than competitor X", one competitor at a time (Wilcoxon rank-sum,
Holm-corrected, A12 effect size). That is NOT the same question as
"averaged over all 20 seeds, do the 17 algorithms differ at all as a
group, and how does each one rank relative to the others". The second
question is what the Friedman test answers, and it is the standard
complement to the pairwise tests in this literature (Derrac, Garcia,
Molina & Herrera, 2011, "A practical tutorial on the use of
nonparametric statistical tests...", Swarm and Evolutionary
Computation 1(1):3-18 -- same reference rc22_significance.py already
cites).

Method
------
1. For each seed (block), rank all 17 algorithms 1 (best) .. 17
   (worst) by Deb's feasibility rule -- NOT by raw f -- so an
   infeasible run is correctly ranked worse than every feasible run
   regardless of its f value, consistent with how every other script
   in this project orders solutions. Exact ties (identical f and cv)
   share the average rank of the tied block, the standard convention.
2. Average each algorithm's rank across all 20 seeds -> mean rank
   (lower is better). This is the headline number: "BalOA's mean
   rank across 20 seeds was X, the best/2nd-best/... of 17".
3. Friedman chi-square statistic on those ranks, testing the omnibus
   null "all 17 algorithms perform identically" -- plus the
   Iman-Davenport F-statistic, which Derrac et al. (2011) specifically
   recommend over the plain chi-square approximation (chi-square is
   known to be conservative for this many treatments/blocks). Both
   are reported; the F-statistic's p-value is the one to cite.

This script does NOT repeat the pairwise BalOA-vs-each-competitor
comparisons -- that is rc22_significance.py's job and is unchanged by
this one. Run both; they answer different questions.

Requires rc22_raw_runs.csv (produced by run_rc22_comparison.py), with
exactly one row per (algorithm, seed) for all 17 algorithms x 20
seeds -- i.e. a complete block design. A seed missing any algorithm's
run is dropped from the test (with a printed warning), since Friedman
requires a complete block to rank within.

Usage (run locally, per project rule #3):
    python rc22_friedman.py

Output:
    Results/rc22_friedman.csv   -- one row per algorithm: mean_rank,
        rank_position (1 = best), n_seeds_used
    Printed: the same table, plus the Friedman chi-square and
        Iman-Davenport F statistics with their p-values.
"""

from __future__ import annotations
import os
import csv

import numpy as np
from scipy.stats import chi2, f as f_dist

from constraint_utils import deb_worse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "Results"))
RAW_CSV = os.path.join(RESULTS_DIR, "rc22_raw_runs.csv")
OUT_CSV = os.path.join(RESULTS_DIR, "rc22_friedman.csv")

ALPHA = 0.05


def _load_by_seed(path: str) -> dict[int, dict[str, tuple[float, float]]]:
    """seed -> {algorithm: (f, cv)}."""
    data: dict[int, dict[str, tuple[float, float]]] = {}
    with open(path, "r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            seed = int(row["seed"])
            name = row["algorithm"]
            f_val = float(row["f_max_gear_error"])
            cv_val = float(row["constraint_violation"])
            data.setdefault(seed, {})[name] = (f_val, cv_val)
    return data


def _deb_ranks(cells: dict[str, tuple[float, float]]) -> dict[str, float]:
    """
    Rank every algorithm 1 (best) .. k (worst) under Deb's feasibility
    rule, with exact ties sharing the average rank of their tie group
    (standard mid-rank convention, matching scipy's own 'average' tie
    handling so these ranks feed a textbook Friedman statistic).
    """
    names = list(cells.keys())
    f = {n: cells[n][0] for n in names}
    cv = {n: cells[n][1] for n in names}

    # Full ordering best-to-worst via Deb's rule as a comparator.
    def cmp(a, b):
        if deb_worse(f[a], cv[a], f[b], cv[b]):
            return 1
        if deb_worse(f[b], cv[b], f[a], cv[a]):
            return -1
        return 0

    from functools import cmp_to_key
    ordered = sorted(names, key=cmp_to_key(cmp))

    ranks: dict[str, float] = {}
    i = 0
    k = len(ordered)
    while i < k:
        j = i
        # Extend the tie group while neighbours are mutually "not worse".
        while j + 1 < k and cmp(ordered[j], ordered[j + 1]) == 0:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0   # ranks are 1-based
        for name in ordered[i:j + 1]:
            ranks[name] = avg_rank
        i = j + 1
    return ranks


def main() -> None:
    if not os.path.exists(RAW_CSV):
        raise FileNotFoundError(
            f"{RAW_CSV} not found -- run run_rc22_comparison.py first."
        )

    by_seed = _load_by_seed(RAW_CSV)
    all_algorithms = sorted({name for cells in by_seed.values() for name in cells})
    k = len(all_algorithms)

    rank_rows = []   # one dict per seed-block: {algorithm: rank}
    skipped = []
    for seed in sorted(by_seed.keys()):
        cells = by_seed[seed]
        missing = [a for a in all_algorithms if a not in cells]
        if missing:
            skipped.append((seed, missing))
            continue
        rank_rows.append(_deb_ranks(cells))

    n = len(rank_rows)
    if n < 2:
        raise RuntimeError(
            f"Only {n} complete seed-block(s) available (need >=2 for a "
            f"Friedman test). Check rc22_raw_runs.csv has every algorithm "
            f"run on the same seeds."
        )

    mean_rank = {a: float(np.mean([rr[a] for rr in rank_rows])) for a in all_algorithms}
    sum_rank = {a: float(np.sum([rr[a] for rr in rank_rows])) for a in all_algorithms}

    # Friedman chi-square, using mean ranks R_bar_j (algebraically
    # identical to the textbook sum-of-ranks form 12/(Nk(k+1)) * sum
    # R_j^2 - 3N(k+1), but numerically stable -- that form squares the
    # raw rank SUMS, which overflows the intended scale for anything
    # beyond a handful of blocks/treatments):
    #   Q = 12N/(k(k+1)) * sum_j (R_bar_j - (k+1)/2)^2,   df = k-1
    grand_mean_rank = (k + 1) / 2.0
    ss = sum((mean_rank[a] - grand_mean_rank) ** 2 for a in all_algorithms)
    Q = (12.0 * n) / (k * (k + 1)) * ss
    df_chi2 = k - 1
    p_chi2 = float(chi2.sf(Q, df_chi2))

    # Iman-Davenport F-statistic -- preferred over the chi-square
    # approximation by Derrac et al. (2011) for exactly this setting
    # (moderate k, moderate N), since the chi-square test is overly
    # conservative here.
    if n * (k - 1) - Q == 0:
        F_id = float("inf")
    else:
        F_id = ((n - 1) * Q) / (n * (k - 1) - Q)
    df1, df2 = k - 1, (k - 1) * (n - 1)
    p_fid = float(f_dist.sf(F_id, df1, df2))

    order = sorted(all_algorithms, key=lambda a: mean_rank[a])
    rows = []
    for pos, name in enumerate(order, start=1):
        rows.append({
            "rank_position": pos,
            "algorithm": name,
            "mean_rank": mean_rank[name],
            "sum_rank": sum_rank[name],
            "n_seeds_used": n,
        })

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["rank_position", "algorithm", "mean_rank",
                                            "sum_rank", "n_seeds_used"])
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print("=" * 90)
    print("RC22: Friedman omnibus test across all 17 algorithms (Deb-rule ranks per seed)")
    print(f"Algorithms (k) = {k}   Seed-blocks used (N) = {n}   alpha = {ALPHA}")
    if skipped:
        print(f"NOTE: {len(skipped)} seed(s) skipped (incomplete block): "
              f"{[s for s, _ in skipped]}")
    print("=" * 90)
    header = f"{'Rank':>4} {'Algorithm':<10} {'Mean rank':>10} {'Sum rank':>10}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row['rank_position']:>4} {row['algorithm']:<10} "
              f"{row['mean_rank']:>10.3f} {row['sum_rank']:>10.1f}")

    print("")
    print(f"Friedman chi-square: Q = {Q:.4f}, df = {df_chi2}, p = {p_chi2:.6g}")
    print(f"Iman-Davenport F:    F = {F_id:.4f}, df = ({df1}, {df2}), p = {p_fid:.6g}"
          f"   <-- preferred statistic (Derrac et al., 2011)")
    verdict = "REJECTED" if p_fid < ALPHA else "NOT rejected"
    print(f"Null hypothesis (all {k} algorithms perform identically): {verdict} "
          f"at alpha={ALPHA}.")
    print("")
    print(f"Full rank table written to: {OUT_CSV}")
    print("(Pairwise BalOA-vs-each-competitor significance is a separate question "
          "-- see rc22_significance.py.)")


if __name__ == "__main__":
    main()
