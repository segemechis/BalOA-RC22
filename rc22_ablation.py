"""
rc22_ablation.py
=======================
Component ablation study for BalOA on RC22 (Planetary Gear Train
Design): isolates the individual contribution of each named cultural
mechanism by disabling it on its own and re-running the same 20-seed
protocol used everywhere else in this project, then compares each
one-mechanism-disabled variant against the full/baseline BalOA
(already validated in Results/rc22_raw_runs.csv, Results/
checkpoints_rc22/BalOA.json) with a paired Wilcoxon signed-rank test
on the per-seed Deb-rule scalar (same seed -> same RNG stream for
both runs, so the comparison is paired, not independent).

Mechanisms ablated (one at a time; see the disable_* kwargs added to
BalOA's constructor in BalOA_algorithm.py -- every one of them
defaults to False, so the baseline/full BalOA is bit-for-bit
unchanged from every prior run in this project):

    No_Faacie          disable_faacie=True          no diversity-collapse purification
    No_Gondooro        disable_gondooro=True        no individual-stagnation escape
    No_Murra           disable_murra=True           no cyclic cross-Roga migration
    No_Bilbaana_Swap   disable_bilbaana_swap=True   lineage role never alternates
    No_Baallee         disable_baallee=True          no periodic G1 reseed ceremony
    No_Songo           disable_songo=True           archive stays permanently empty
                                                     (every consumer -- G4 Raaba,
                                                     Faaci'e's, Gondooro's and G9's
                                                     reseeding -- falls back to its
                                                     own already-implemented
                                                     "archive empty" uniform-random
                                                     behaviour; no other code path
                                                     needs touching for this one)
    No_Escalation      disable_escalation=True      Abba Gada never consulted on
                                                     sustained Roga failure (x_authority
                                                     is always the Roga's own best)
    No_Cowwajje        disable_cowwajje=True        no periodic full renewal (G9
                                                     always falls back to the ordinary
                                                     G8 Qulullo-style move instead)

The Full_BalOA baseline is NOT re-run -- it is read straight from the
existing, already cross-checked Results/checkpoints_rc22/BalOA.json
(produced by run_rc22_comparison.py), so the ablation comparison uses
the exact same validated numbers already reported elsewhere in the
paper, and running this script never duplicates that compute.

Usage (run locally, per project rule #3):
    python rc22_ablation.py

Outputs (Results folder, per the fixed project folder structure):
    Results/rc22_ablation.csv              -- one row per variant
    Results/rc22_ablation_run_log.txt      -- everything printed, saved verbatim
    Results/checkpoints_rc22_ablation/*.json  -- one checkpoint per variant,
                                                  same resumability guarantees
                                                  (per-seed, crash-safe) as
                                                  run_rc22_comparison.py
"""

from __future__ import annotations
import os
import json
import csv
import time
import datetime

import numpy as np
from scipy.stats import wilcoxon

import rc22_gear_train_problem as rc22
from BalOA_algorithm import BalOA

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
SEEDS = list(range(20))        # identical seed set used everywhere else
MAX_FES = 100000                # identical budget used everywhere else
ALPHA = 0.05

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "Results"))
CKPT_DIR = os.path.join(RESULTS_DIR, "checkpoints_rc22")               # for baseline reuse
ABL_CKPT_DIR = os.path.join(RESULTS_DIR, "checkpoints_rc22_ablation")  # new, variants only
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ABL_CKPT_DIR, exist_ok=True)

OUT_CSV = os.path.join(RESULTS_DIR, "rc22_ablation.csv")
LOG_TXT = os.path.join(RESULTS_DIR, "rc22_ablation_run_log.txt")
BASELINE_CKPT = os.path.join(CKPT_DIR, "BalOA.json")

VARIANTS: dict[str, dict] = {
    "No_Faacie":        {"disable_faacie": True},
    "No_Gondooro":      {"disable_gondooro": True},
    "No_Murra":         {"disable_murra": True},
    "No_Bilbaana_Swap": {"disable_bilbaana_swap": True},
    "No_Baallee":       {"disable_baallee": True},
    "No_Songo":         {"disable_songo": True},
    "No_Escalation":    {"disable_escalation": True},
    "No_Cowwajje":      {"disable_cowwajje": True},
}

# ---------------------------------------------------------------------
# Console + file logging (standing project requirement)
# ---------------------------------------------------------------------
_log_lines: list[str] = []


def log(msg: object = "") -> None:
    line = str(msg)
    print(line)
    _log_lines.append(line)
    with open(LOG_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(_log_lines) + "\n")


# ---------------------------------------------------------------------
# Checkpointing (same per-seed, crash-safe pattern as run_rc22_comparison.py)
# ---------------------------------------------------------------------
def _ckpt_path(variant: str) -> str:
    return os.path.join(ABL_CKPT_DIR, f"{variant}.json")


def _load_ckpt(variant: str) -> list[dict]:
    p = _ckpt_path(variant)
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_ckpt(variant: str, records: list[dict]) -> None:
    with open(_ckpt_path(variant), "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


def _load_baseline() -> dict[int, dict]:
    if not os.path.exists(BASELINE_CKPT):
        raise FileNotFoundError(
            f"{BASELINE_CKPT} not found -- run run_rc22_comparison.py first "
            f"(the ablation study compares against BalOA's already-validated "
            f"full-mechanism baseline, it does not recompute it)."
        )
    with open(BASELINE_CKPT, "r", encoding="utf-8") as f:
        ckpt = json.load(f)
    by_seed = {r["seed"]: r for r in ckpt if r["seed"] in SEEDS}
    missing = [s for s in SEEDS if s not in by_seed]
    if missing:
        raise RuntimeError(
            f"Baseline checkpoint is missing seed(s) {missing} -- re-run "
            f"run_rc22_comparison.py to complete it before running the ablation."
        )
    return by_seed


# ---------------------------------------------------------------------
# Single-run execution for a variant (no convergence-curve capture --
# the ablation study only needs final f/cv per seed, same Deb-rule
# comparisons used throughout this project)
# ---------------------------------------------------------------------
def run_variant_once(seed: int, disable_kwargs: dict) -> dict:
    t0 = time.time()
    opt = BalOA(func=rc22.evaluate, lb=rc22.LB, ub=rc22.UB, dim=rc22.DIM,
                max_fes=MAX_FES, **disable_kwargs)
    x_best, f_best, cv_best, _conv = opt.optimize(seed=seed)
    elapsed = time.time() - t0
    x_best = np.clip(np.asarray(x_best, dtype=float), rc22.LB, rc22.UB)
    return {
        "seed": seed,
        "f": float(f_best),
        "cv": float(cv_best),
        "feasible": bool(cv_best <= 0.0),
        "elapsed_sec": elapsed,
    }


def _deb_scalar(f: float, cv: float) -> float:
    """Same scalarisation convention used for the convergence curves and
    the Friedman ranks elsewhere in this project: feasible runs are
    compared on f directly, infeasible runs are pushed strictly worse
    than every feasible run by a large constant plus their violation,
    so a single signed paired difference is meaningful."""
    return f if cv <= 0.0 else rc22.PENALTY_BIG + cv


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    log("=" * 88)
    log("RC22: BalOA component ablation study (one named mechanism disabled at a time)")
    log(f"Started: {datetime.datetime.now().isoformat(timespec='seconds')}")
    log(f"Seeds: {SEEDS}   Max FES: {MAX_FES}   Variants: {len(VARIANTS)}")
    log("Baseline (Full_BalOA) is read from the already-validated "
        f"{BASELINE_CKPT}, not recomputed.")
    log("=" * 88)

    baseline_by_seed = _load_baseline()
    baseline_scalar = {s: _deb_scalar(r["f"], r["cv"]) for s, r in baseline_by_seed.items()}
    n_base_feas = sum(1 for r in baseline_by_seed.values() if r["cv"] <= 0.0)
    base_feas_f = [r["f"] for r in baseline_by_seed.values() if r["cv"] <= 0.0]
    log(f"[Full_BalOA] baseline loaded: {n_base_feas}/{len(SEEDS)} feasible, "
        f"mean f (feasible) = {np.mean(base_feas_f):.6f}")

    summary_rows: list[dict] = []
    summary_rows.append({
        "variant": "Full_BalOA",
        "mechanism_disabled": "(none -- baseline)",
        "n_feasible": n_base_feas,
        "n_runs": len(SEEDS),
        "mean_f_feasible": float(np.mean(base_feas_f)) if base_feas_f else float("nan"),
        "mean_deb_scalar": float(np.mean(list(baseline_scalar.values()))),
        "delta_mean_deb_scalar_vs_baseline": 0.0,
        "wilcoxon_p": float("nan"),
        "verdict": "baseline",
    })

    for variant, disable_kwargs in VARIANTS.items():
        ckpt = _load_ckpt(variant)
        have = {r["seed"] for r in ckpt}
        records = [r for r in ckpt if r["seed"] in SEEDS]
        missing = [s for s in SEEDS if s not in have]

        if not missing:
            log(f"[{variant}] checkpoint already covers all {len(SEEDS)} seed(s) -- skipping.")
        else:
            log(f"[{variant}] disabling: {list(disable_kwargs.keys())[0]}  "
                f"-- running {len(missing)} seed(s) {missing} ...")
            for seed in missing:
                t0 = time.time()
                rec = run_variant_once(seed, disable_kwargs)
                log(f"    seed={seed}  f={rec['f']:.6f}  feasible={rec['feasible']}  "
                    f"cv={rec['cv']:.6g}  ({time.time() - t0:.1f}s)")
                records = [r for r in records if r["seed"] != seed] + [rec]
                records.sort(key=lambda r: r["seed"])
                _save_ckpt(variant, records)
            log(f"[{variant}] done -- checkpoint saved ({len(records)} seed(s)).")

        by_seed = {r["seed"]: r for r in records}
        n_feas = sum(1 for r in records if r["feasible"])
        feas_f = [r["f"] for r in records if r["feasible"]]

        variant_scalar = {s: _deb_scalar(by_seed[s]["f"], by_seed[s]["cv"]) for s in SEEDS}
        diffs = [variant_scalar[s] - baseline_scalar[s] for s in SEEDS]  # >0 = variant worse
        mean_delta = float(np.mean(diffs))

        if all(d == 0 for d in diffs):
            p_val = float("nan")
            verdict = "identical to baseline on every seed"
        else:
            try:
                _, p_val = wilcoxon(diffs)
                p_val = float(p_val)
                if p_val < ALPHA:
                    verdict = ("SIGNIFICANTLY WORSE without this mechanism"
                               if mean_delta > 0 else
                               "significantly BETTER without this mechanism")
                else:
                    verdict = "no significant difference"
            except ValueError:
                # wilcoxon raises if the paired differences are degenerate
                # in a way it can't rank (e.g. too few non-zero diffs)
                p_val = float("nan")
                verdict = "not enough non-zero paired differences to test"

        mechanism_name = list(disable_kwargs.keys())[0]
        summary_rows.append({
            "variant": variant,
            "mechanism_disabled": mechanism_name,
            "n_feasible": n_feas,
            "n_runs": len(SEEDS),
            "mean_f_feasible": float(np.mean(feas_f)) if feas_f else float("nan"),
            "mean_deb_scalar": float(np.mean(list(variant_scalar.values()))),
            "delta_mean_deb_scalar_vs_baseline": mean_delta,
            "wilcoxon_p": p_val,
            "verdict": verdict,
        })

    fieldnames = ["variant", "mechanism_disabled", "n_feasible", "n_runs",
                  "mean_f_feasible", "mean_deb_scalar",
                  "delta_mean_deb_scalar_vs_baseline", "wilcoxon_p", "verdict"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in summary_rows:
            w.writerow(row)

    log("")
    log("=" * 88)
    log("ABLATION SUMMARY (paired Wilcoxon signed-rank vs. Full_BalOA, same 20 seeds)")
    log(f"Deb-rule scalar: feasible runs compared on f directly; infeasible runs are "
        f"PENALTY_BIG + cv (so lower is always better, and a positive delta below "
        f"means the ablated variant is WORSE than the full algorithm).")
    log("=" * 88)
    header = (f"{'Variant':<18} {'Feasible':>9} {'Mean f':>10} {'Delta scalar':>13} "
              f"{'p-value':>10}  Verdict")
    log(header)
    log("-" * len(header))
    for row in summary_rows:
        mf = row["mean_f_feasible"]
        mf_s = f"{mf:.6f}" if mf == mf else "nan"
        pv = row["wilcoxon_p"]
        pv_s = f"{pv:.4g}" if pv == pv else "n/a"
        log(f"{row['variant']:<18} {row['n_feasible']:>5}/{row['n_runs']:<3} "
            f"{mf_s:>10} {row['delta_mean_deb_scalar_vs_baseline']:>13.4g} "
            f"{pv_s:>10}  {row['verdict']}")

    log("")
    log(f"Full table written to: {OUT_CSV}")
    log(f"Full console log:      {LOG_TXT}")
    log("")
    log("Reading the table: a mechanism whose removal gives a significantly higher "
        "(worse) mean Deb-rule scalar -- small p-value, positive delta -- is one "
        "BalOA's performance genuinely depends on; a mechanism whose removal changes "
        "nothing significantly is a candidate to simplify or justify on grounds other "
        "than raw RC22 performance (e.g. interpretability, robustness on other problems).")


if __name__ == "__main__":
    main()
