"""
rc22_significance.py
=======================
Statistical significance testing for the RC22 (Planetary Gear Train
Design) comparison -- BalOA vs. each of the 16 competitor algorithms
in competitors.py.

Why this script exists
-----------------------
run_rc22_comparison.py's summary ranks algorithms by best-found value
(Deb's rule) and separately reports mean/std -- useful, but neither is
a significance test. With only 5 seeds, "BalOA's best was 0.525769
vs. HHO's 0.525967" is not distinguishable from noise. This script
runs the comparison the way it's actually done in the metaheuristics
literature (see Derrac, Garcia, Molina & Herrera, 2011, "A practical
tutorial on the use of nonparametric statistical tests as a
methodology for comparing evolutionary and swarm intelligence
algorithms", Swarm and Evolutionary Computation 1(1):3-18 -- the
standard reference this whole sub-field cites for exactly this
problem):

  1. Wilcoxon rank-sum test (Mann-Whitney U), BalOA vs. each
     competitor, on each algorithm's FEASIBLE run f-values only
     (an infeasible run isn't a comparable "how good was the
     objective" data point under Deb's rule -- mixing them in would
     bias the test). One-sided ("BalOA's f-distribution is
     stochastically smaller", i.e. better, since this is
     minimisation) and two-sided p-values are both reported.
  2. Holm-Bonferroni step-down correction across the 16 simultaneous
     BalOA-vs-X comparisons, to control the family-wise error rate
     (running 16 uncorrected tests at alpha=0.05 would give an
     inflated ~1-(0.95)^16 ≈ 56% chance of at least one false
     "significant" result by chance alone).
  3. Vargha-Delaney A12 effect size (Vargha & Delaney, 2000), the
     standard companion effect-size measure for this exact test in
     this literature: A12 = P(a random BalOA run's f < a random
     competitor run's f) + 0.5*P(tie). A12 > 0.5 favours BalOA;
     thresholds (Vargha & Delaney): negligible <0.56, small <0.64,
     medium <0.71, large >=0.71 (mirrored around 0.5 the other way
     for competitor-favouring effects).

Requires rc22_raw_runs.csv to already exist (produced by
run_rc22_comparison.py) -- run that script first, ideally with SEEDS
widened to 20+ for real statistical power; a 5-seed run will still
produce output here but the tests will be weak.

Usage (run locally, per project rule #3):
    python rc22_significance.py

Output:
    Results/rc22_significance.csv  -- one row per competitor, with
        n_baloa, n_competitor (feasible run counts used), U statistic,
        p_two_sided, p_one_sided (BalOA better), p_holm (corrected),
        significant_after_holm (alpha=0.05), A12, A12_interpretation
    Printed table, sorted by corrected p-value.
"""

from __future__ import annotations
import os
import csv

import numpy as np
from scipy.stats import mannwhitneyu

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'Results'))
RAW_CSV = os.path.join(RESULTS_DIR, 'rc22_raw_runs.csv')
OUT_CSV = os.path.join(RESULTS_DIR, 'rc22_significance.csv')

ALPHA = 0.05


def _load_feasible_f(path: str) -> dict[str, list[float]]:
    """algorithm -> list of f values from FEASIBLE runs only."""
    data: dict[str, list[float]] = {}
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            feasible = row["feasible"].strip().lower() in ("true", "1", "yes")
            if not feasible:
                continue
            name = row["algorithm"]
            data.setdefault(name, []).append(float(row["f_max_gear_error"]))
    return data


def _a12(x: list[float], y: list[float]) -> float:
    """
    Vargha-Delaney A12 (Vargha & Delaney, 2000): P(random x < random y)
    + 0.5*P(tie) -- i.e. the probability that a random BalOA run beats
    a random competitor run (smaller f = better, since this is
    minimisation), with ties split evenly. x = BalOA's f values,
    y = competitor's f values. A12 > 0.5 favours BalOA.

    scipy's mannwhitneyu(a, b).statistic is U_a, defined as the count
    of pairs where a > b (plus 0.5 per tie) -- i.e. U_a/(n_a*n_b) =
    P(a > b) + 0.5*P(tie), the OPPOSITE direction from what we want.
    So: P(x < y) + 0.5*P(tie) = 1 - [P(x > y) + 0.5*P(tie)]
                              = 1 - U_x / (n_x * n_y).
    """
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return float("nan")
    res = mannwhitneyu(x, y, alternative="two-sided")
    u_x = res.statistic
    return float(1.0 - u_x / (n1 * n2))


def _a12_interpretation(a12: float) -> str:
    if a12 != a12:  # NaN
        return "n/a"
    d = abs(a12 - 0.5)
    if d < 0.06:
        mag = "negligible"
    elif d < 0.14:
        mag = "small"
    elif d < 0.21:
        mag = "medium"
    else:
        mag = "large"
    direction = "BalOA better" if a12 > 0.5 else ("competitor better" if a12 < 0.5 else "tie")
    return f"{mag} ({direction})" if mag != "negligible" else "negligible"


def _holm_correction(pvals: list[float]) -> list[float]:
    """
    Holm-Bonferroni step-down correction. Returns adjusted p-values in
    the SAME order as the input (not sorted), suitable for direct
    comparison against alpha per-hypothesis.
    """
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        factor = m - rank
        val = min(1.0, pvals[idx] * factor)
        running_max = max(running_max, val)   # Holm requires monotonicity
        adjusted[idx] = running_max
    return adjusted


def main() -> None:
    if not os.path.exists(RAW_CSV):
        raise FileNotFoundError(
            f"{RAW_CSV} not found -- run run_rc22_comparison.py first."
        )

    feas = _load_feasible_f(RAW_CSV)
    if "BalOA" not in feas or len(feas["BalOA"]) < 2:
        raise RuntimeError(
            "Need at least 2 feasible BalOA runs in rc22_raw_runs.csv to "
            "run any significance test."
        )
    baloa_f = feas["BalOA"]
    competitors = [name for name in feas.keys() if name != "BalOA"]

    rows = []
    raw_two_sided = []
    for name in competitors:
        comp_f = feas[name]
        n_b, n_c = len(baloa_f), len(comp_f)
        if n_c < 2:
            rows.append({
                "competitor": name, "n_baloa": n_b, "n_competitor": n_c,
                "U": float("nan"), "p_two_sided": float("nan"),
                "p_one_sided_baloa_better": float("nan"),
                "A12": float("nan"), "A12_interpretation": "n/a (too few feasible runs)",
            })
            raw_two_sided.append(1.0)
            continue

        res_two = mannwhitneyu(baloa_f, comp_f, alternative="two-sided")
        res_one = mannwhitneyu(baloa_f, comp_f, alternative="less")  # BalOA f stochastically smaller = better
        a12 = _a12(baloa_f, comp_f)

        rows.append({
            "competitor": name,
            "n_baloa": n_b,
            "n_competitor": n_c,
            "U": float(res_two.statistic),
            "p_two_sided": float(res_two.pvalue),
            "p_one_sided_baloa_better": float(res_one.pvalue),
            "A12": a12,
            "A12_interpretation": _a12_interpretation(a12),
        })
        raw_two_sided.append(float(res_two.pvalue))

    p_holm = _holm_correction(raw_two_sided)
    for row, ph in zip(rows, p_holm):
        row["p_holm"] = ph
        row["significant_after_holm"] = bool(ph < ALPHA)

    rows.sort(key=lambda r: (r["p_holm"] if r["p_holm"] == r["p_holm"] else 1.0))

    fieldnames = ["competitor", "n_baloa", "n_competitor", "U",
                  "p_two_sided", "p_one_sided_baloa_better", "p_holm",
                  "significant_after_holm", "A12", "A12_interpretation"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print("=" * 100)
    print("RC22: BalOA vs. each competitor -- Wilcoxon rank-sum (Mann-Whitney U) "
          "on feasible runs, Holm-corrected")
    print(f"BalOA feasible runs used: n={len(baloa_f)}   alpha={ALPHA}")
    print("=" * 100)
    header = (f"{'Competitor':<10} {'n_B':>4} {'n_C':>4} {'p_2sided':>10} "
              f"{'p_1sided':>10} {'p_holm':>10} {'Sig?':>5} {'A12':>7}  Interpretation")
    print(header)
    print("-" * len(header))
    for row in rows:
        p2 = row["p_two_sided"]; p1 = row["p_one_sided_baloa_better"]; ph = row["p_holm"]
        a12 = row["A12"]
        p2_s = f"{p2:.6f}" if p2 == p2 else "n/a"
        p1_s = f"{p1:.6f}" if p1 == p1 else "n/a"
        ph_s = f"{ph:.6f}" if ph == ph else "n/a"
        a12_s = f"{a12:.3f}" if a12 == a12 else "n/a"
        sig = "YES" if row["significant_after_holm"] else "no"
        print(f"{row['competitor']:<10} {row['n_baloa']:>4} {row['n_competitor']:>4} "
              f"{p2_s:>10} {p1_s:>10} {ph_s:>10} {sig:>5} {a12_s:>7}  "
              f"{row['A12_interpretation']}")

    n_sig = sum(1 for r in rows if r["significant_after_holm"])
    print("")
    print(f"{n_sig} of {len(rows)} competitors show a statistically significant "
          f"difference from BalOA after Holm-Bonferroni correction (alpha={ALPHA}).")
    print(f"Full table written to: {OUT_CSV}")


if __name__ == "__main__":
    main()
