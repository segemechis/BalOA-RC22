"""
run_rc22_comparison.py
=========================
Comparison runner for RC22 -- Planetary Gear Train Design (see
rc22_gear_train_problem.py): BalOA vs. all 16 competitor algorithms in
competitors.py, self-run (no literature comparison values used in any
ranking -- every number in the outputs comes from this script's own
executions; the source paper's reported f(x*)=0.52576870748 is quoted
only in rc22_gear_train_problem.py's docstring as a formula sanity
check, never used here as a result).

REVISION (convergence capture folded in): earlier revisions of this
project ran a SEPARATE script (rc22_convergence_capture.py) to capture
per-iteration convergence curves, because the original checkpoints only
kept each run's final best f/cv and threw away the per-evaluation
history. That meant two scripts could in principle drift out of sync.
This revision captures the full convergence curve for EVERY seed of
EVERY algorithm as part of THIS run, using the same
constraint_utils.make_logging_objective wrapper -- so there is now one
script and one pass that produces everything: the final per-run
summary stats AND the convergence curves, for all 20 seeds, together,
consistently. rc22_convergence_capture.py is retired; do not run it
anymore -- everything it did now happens here.

IMPORTANT correctness guarantee: make_logging_objective is pure
instrumentation. It does not call any RNG and does not add or skip
evaluations -- it just records each (f, cv) pair as it's returned and
passes the same value on unchanged. So for a fixed seed, every
algorithm's search trajectory and final (f, cv, best_x) are BIT-FOR-BIT
IDENTICAL to a run without the logging wrapper. Re-running a seed that
was already in an older (pre-convergence-capture) checkpoint reproduces
that exact same result -- nothing about your already-validated numbers
changes; the only new thing captured is the trajectory.

Built-in cross-check (no silent trust): for every seed that already had
a recorded result in the checkpoint ON DISK BEFORE this run started,
the newly (re)computed f/cv is compared against that old recorded
value. A mismatch beyond 1e-6 absolute difference prints a loud
"*** MISMATCH ***" warning naming the algorithm, seed, old value and
new value, rather than silently trusting the rerun. A clean run prints
"cross-check OK" for every such seed. This is a safety net on top of
the determinism guarantee above, not a substitute for it.

Usage (run locally, per project rule #3):
    python run_rc22_comparison.py

Outputs (all written into the sibling "Results" folder, per the fixed
project folder structure C:\\The Baalle System\\{Code,Figures,
Literature,Results}):
    Results/rc22_summary.csv          -- one row per algorithm (ranked)
    Results/rc22_raw_runs.csv         -- one row per (algorithm, seed) run
    Results/rc22_run_log.txt          -- everything printed to console,
                                          saved verbatim (standing
                                          project requirement)
    Results/checkpoints_rc22/*.json   -- one checkpoint per algorithm;
                                          each record now also carries
                                          a "convergence" field
    Results/convergence_rc22/*.json   -- one file per algorithm, a dict
                                          {"<seed>": [[fes, disp], ...]}
                                          for all captured seeds -- the
                                          same format
                                          rc22_convergence_capture.py
                                          used to produce, consumed
                                          as-is by make_rc22_figures.py

Resumability / seed-extension / crash safety:
    A checkpoint record is only considered COMPLETE if it has a
    non-empty "convergence" field. Any existing record missing that
    field (i.e. from before this revision) is treated as incomplete
    and is RE-RUN -- which, by the determinism guarantee above,
    reproduces the same f/cv/best_x while now also capturing the
    curve, and is verified against the old value via the cross-check
    above. The checkpoint file (and the matching convergence_rc22
    file) is rewritten to disk after EVERY SINGLE SEED, not just once
    per algorithm -- so an interruption partway through an algorithm's
    seed list (crash, timeout, Ctrl-C) loses at most the one seed in
    flight, never any already-finished seed.
"""

from __future__ import annotations
import os
import json
import csv
import time
import datetime
from functools import cmp_to_key

import numpy as np

import rc22_gear_train_problem as rc22
import competitors
from BalOA_algorithm import BalOA
from constraint_utils import make_logging_objective, deb_worse, deb_best_index

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
SEEDS = list(range(20))      # full 20 seeds, same seed set used for the
                              # statistical (Wilcoxon / A12) results
MAX_FES = 100000              # source paper's own budget for D<=10 (their Eq. 60); D=9 here
POP_SIZE = 30                  # competitors only -- BalOA's population (57) is
                                # fixed internally and ignores this
SAMPLE_EVERY = 50              # convergence curve downsampling: ~2000 points/seed
                                # instead of 100000; always includes the final point

MISMATCH_TOL = 1e-6             # cross-check tolerance on f and cv, absolute

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'Results'))
CKPT_DIR = os.path.join(RESULTS_DIR, 'checkpoints_rc22')
CONV_DIR = os.path.join(RESULTS_DIR, 'convergence_rc22')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(CONV_DIR, exist_ok=True)

SUMMARY_CSV = os.path.join(RESULTS_DIR, 'rc22_summary.csv')
RAW_CSV = os.path.join(RESULTS_DIR, 'rc22_raw_runs.csv')
LOG_TXT = os.path.join(RESULTS_DIR, 'rc22_run_log.txt')

ALL_ALGORITHMS = list(competitors.ALGORITHMS.keys()) + ['BalOA']


# ---------------------------------------------------------------------
# Console + file logging (standing project requirement: everything
# printed to console, saved verbatim -- flushed after every line, not
# buffered to the end, so a crash never loses log text that was already
# printed)
# ---------------------------------------------------------------------
_log_lines: list[str] = []


def log(msg: object = "") -> None:
    line = str(msg)
    print(line)
    _log_lines.append(line)
    with open(LOG_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(_log_lines) + "\n")


# ---------------------------------------------------------------------
# Checkpointing (main per-algorithm JSON -- now carries "convergence")
# ---------------------------------------------------------------------
def _checkpoint_path(name: str) -> str:
    return os.path.join(CKPT_DIR, f"{name}.json")


def _load_checkpoint(name: str) -> list[dict]:
    p = _checkpoint_path(name)
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_checkpoint(name: str, records: list[dict]) -> None:
    with open(_checkpoint_path(name), "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


def _record_complete(r: dict) -> bool:
    return bool(r.get("convergence"))


# ---------------------------------------------------------------------
# Convergence-curve companion files (Results/convergence_rc22/*.json,
# the same {"<seed>": [[fes, disp], ...]} format the retired
# rc22_convergence_capture.py produced -- kept so make_rc22_figures.py
# needs no changes)
# ---------------------------------------------------------------------
def _conv_path(name: str) -> str:
    return os.path.join(CONV_DIR, f"{name}.json")


def _save_conv(name: str, records: list[dict]) -> None:
    data = {str(r["seed"]): r["convergence"] for r in records if r.get("convergence")}
    with open(_conv_path(name), "w", encoding="utf-8") as f:
        json.dump(data, f)


# ---------------------------------------------------------------------
# Deb-best-so-far convergence curve, reconstructed from a raw
# evaluation-order (f, cv) log -- identical procedure for every
# algorithm (see constraint_utils.make_logging_objective's docstring).
# ---------------------------------------------------------------------
def _deb_best_so_far_curve(evlog: list[tuple[float, float]]) -> list[list[float]]:
    curve: list[list[float]] = []
    best_f, best_cv = np.inf, np.inf
    have_best = False
    n = len(evlog)
    for i, (f, cv) in enumerate(evlog, start=1):
        if not have_best or deb_worse(best_f, best_cv, f, cv):
            best_f, best_cv = f, cv
            have_best = True
        if i % SAMPLE_EVERY == 0 or i == n:
            display = best_f if best_cv <= 0.0 else rc22.PENALTY_BIG + best_cv
            curve.append([i, float(display)])
    return curve


# ---------------------------------------------------------------------
# Single-run execution (now with logging wrapper -> convergence field)
# ---------------------------------------------------------------------
def run_competitor_once(run_fn, seed: int) -> dict:
    evlog: list[tuple[float, float]] = []
    logged = make_logging_objective(rc22.evaluate, evlog, big=rc22.PENALTY_BIG)
    t0 = time.time()
    best_cost, best_pos, conv = run_fn(
        logged, rc22.LB, rc22.UB, rc22.DIM, MAX_FES, pop_size=POP_SIZE, seed=seed
    )
    elapsed = time.time() - t0
    best_pos = np.clip(np.asarray(best_pos, dtype=float), rc22.LB, rc22.UB)
    f_val, cv = rc22.evaluate(best_pos)
    p, m1, m3, N1, N2, N3, N4, N5, N6 = rc22._unpack(best_pos)
    return {
        "seed": seed,
        "f": float(f_val),
        "cv": float(cv),
        "feasible": bool(cv <= 0.0),
        "best_x_raw": best_pos.tolist(),
        "best_x_snapped": [p, m1, m3, N1, N2, N3, N4, N5, N6],
        "elapsed_sec": elapsed,
        "convergence": _deb_best_so_far_curve(evlog),
    }


def run_baloa_once(seed: int) -> dict:
    evlog: list[tuple[float, float]] = []
    logged = make_logging_objective(rc22.evaluate, evlog, big=None)
    t0 = time.time()
    opt = BalOA(func=logged, lb=rc22.LB, ub=rc22.UB, dim=rc22.DIM, max_fes=MAX_FES)
    x_best, f_best, cv_best, conv = opt.optimize(seed=seed)
    elapsed = time.time() - t0
    x_best = np.clip(np.asarray(x_best, dtype=float), rc22.LB, rc22.UB)
    p, m1, m3, N1, N2, N3, N4, N5, N6 = rc22._unpack(x_best)
    return {
        "seed": seed,
        "f": float(f_best),
        "cv": float(cv_best),
        "feasible": bool(cv_best <= 0.0),
        "best_x_raw": x_best.tolist(),
        "best_x_snapped": [p, m1, m3, N1, N2, N3, N4, N5, N6],
        "elapsed_sec": elapsed,
        "convergence": _deb_best_so_far_curve(evlog),
    }


# ---------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------
def _write_raw_csv(all_results: dict[str, list[dict]]) -> None:
    with open(RAW_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["algorithm", "seed", "f_max_gear_error", "feasible", "constraint_violation",
             "elapsed_sec"] + rc22.VAR_NAMES
        )
        for name, records in all_results.items():
            for r in records:
                w.writerow(
                    [name, r["seed"], r["f"], r["feasible"], r["cv"], r["elapsed_sec"]]
                    + list(r["best_x_snapped"])
                )


def _write_summary_csv(all_results: dict[str, list[dict]]) -> list[dict]:
    rows = []
    for name, records in all_results.items():
        n = len(records)
        feas = [r for r in records if r["feasible"]]
        n_feas = len(feas)
        fs_feas = [r["f"] for r in feas]
        best_f = min(fs_feas) if feas else float("nan")
        mean_f = float(np.mean(fs_feas)) if feas else float("nan")
        std_f = float(np.std(fs_feas)) if feas else float("nan")
        worst_f = max(fs_feas) if feas else float("nan")

        F = [r["f"] for r in records]
        CV = [r["cv"] for r in records]
        bidx = deb_best_index(F, CV, list(range(n))) if n else None

        rows.append({
            "algorithm": name,
            "n_runs": n,
            "n_feasible": n_feas,
            "feasibility_rate": (n_feas / n) if n else float("nan"),
            "best_f": best_f,
            "mean_f": mean_f,
            "std_f": std_f,
            "worst_f": worst_f,
            "deb_best_seed": records[bidx]["seed"] if bidx is not None else None,
            "deb_best_f": records[bidx]["f"] if bidx is not None else None,
            "deb_best_cv": records[bidx]["cv"] if bidx is not None else None,
        })

    # Rank algorithms by Deb's rule on each algorithm's own best run
    # (minimisation: feasible beats infeasible; among feasible, lower
    # f wins; among infeasible, lower constraint violation wins).
    def _cmp(a, b):
        fa, cva = a["deb_best_f"], a["deb_best_cv"]
        fb, cvb = b["deb_best_f"], b["deb_best_cv"]
        if deb_worse(fa, cva, fb, cvb):
            return 1
        if deb_worse(fb, cvb, fa, cva):
            return -1
        return 0

    rows.sort(key=cmp_to_key(_cmp))
    for i, row in enumerate(rows, start=1):
        row["rank"] = i

    fieldnames = ["rank", "algorithm", "n_runs", "n_feasible", "feasibility_rate",
                  "best_f", "mean_f", "std_f", "worst_f",
                  "deb_best_seed", "deb_best_f", "deb_best_cv"]
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    return rows


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    log("=" * 78)
    log("RC22 Planetary Gear Train Design comparison -- BalOA vs. 16 competitors")
    log(f"Started: {datetime.datetime.now().isoformat(timespec='seconds')}")
    log(f"Problem: {rc22.DIM}-D, minimise worst-case gear-ratio error "
        f"(fully discrete/integer design space, non-smooth minimax objective)")
    log(f"Seeds per algorithm: {SEEDS}   Max FES: {MAX_FES}   "
        f"Competitor pop size: {POP_SIZE}")
    log("Convergence curves are captured for every seed in this same pass "
        "(rc22_convergence_capture.py is retired -- do not run it).")
    log(f"Results folder: {RESULTS_DIR}")
    log("=" * 78)

    all_results: dict[str, list[dict]] = {}
    n_mismatches = 0
    n_crosschecked = 0

    for name in ALL_ALGORITHMS:
        ckpt = _load_checkpoint(name)
        old_by_seed = {r["seed"]: r for r in ckpt}          # pre-this-run snapshot, for cross-check
        have_complete = {r["seed"] for r in ckpt if _record_complete(r)}
        missing = [s for s in SEEDS if s not in have_complete]

        # Keep every already-complete record (has convergence data);
        # legacy/incomplete records for requested seeds will be replaced.
        records = [r for r in ckpt if r["seed"] in have_complete and r["seed"] in SEEDS]

        if not missing:
            log(f"[{name}] checkpoint already covers all {len(SEEDS)} requested "
                f"seed(s) with convergence data -- skipping.")
            all_results[name] = records
            continue

        n_legacy = sum(1 for s in missing if s in old_by_seed)
        if n_legacy:
            log(f"[{name}] {len(have_complete)} seed(s) already complete; "
                f"{len(missing)} seed(s) {missing} need (re)running "
                f"({n_legacy} of those have an old pre-convergence-capture "
                f"result that will be cross-checked for an identical "
                f"f/cv) ...")
        else:
            log(f"[{name}] {len(have_complete)} seed(s) already complete; "
                f"running {len(missing)} new seed(s) {missing} ...")

        for seed in missing:
            t0 = time.time()
            if name == "BalOA":
                rec = run_baloa_once(seed)
            else:
                rec = run_competitor_once(competitors.ALGORITHMS[name], seed)

            old = old_by_seed.get(seed)
            if old is not None:
                n_crosschecked += 1
                df = abs(rec["f"] - old["f"])
                dcv = abs(rec["cv"] - old["cv"])
                if df > MISMATCH_TOL or dcv > MISMATCH_TOL:
                    n_mismatches += 1
                    log(f"    seed={seed}  *** MISMATCH *** vs. existing checkpoint: "
                        f"old f={old['f']:.10g} cv={old['cv']:.10g}  |  "
                        f"new f={rec['f']:.10g} cv={rec['cv']:.10g}  "
                        f"(|df|={df:.3g}, |dcv|={dcv:.3g})")
                else:
                    log(f"    seed={seed}  f={rec['f']:.6f}  feasible={rec['feasible']}  "
                        f"cv={rec['cv']:.6g}  cross-check OK (matches existing checkpoint)  "
                        f"({time.time() - t0:.1f}s)")
            else:
                log(f"    seed={seed}  f={rec['f']:.6f}  feasible={rec['feasible']}  "
                    f"cv={rec['cv']:.6g}  ({time.time() - t0:.1f}s)")

            # Replace any legacy record for this seed, then save
            # IMMEDIATELY -- after every single seed, not once per
            # algorithm -- so an interruption loses at most the seed
            # currently in flight.
            records = [r for r in records if r["seed"] != seed] + [rec]
            records.sort(key=lambda r: r["seed"])
            _save_checkpoint(name, records)
            _save_conv(name, records)

        all_results[name] = records
        log(f"[{name}] done -- checkpoint + convergence file saved "
            f"({len(records)} total seed(s)).")

        _write_raw_csv(all_results)
        _write_summary_csv(all_results)
        log(f"[{name}] CSVs updated ({RAW_CSV}, {SUMMARY_CSV}).")

    log("")
    log("=" * 78)
    log("FINAL RANKING (Deb's feasibility rule, each algorithm's own best run)")
    log("=" * 78)
    ranked = _write_summary_csv(all_results)
    header = (f"{'Rank':>4}  {'Algorithm':<10}  {'Feasible':>9}  "
              f"{'Best f':>10}  {'Mean f':>10}  {'Std f':>10}")
    log(header)
    log("-" * len(header))
    for row in ranked:
        bf = row['best_f']; mf = row['mean_f']; sf = row['std_f']
        bf_s = f"{bf:.6f}" if bf == bf else "nan"       # bf==bf is False only for NaN
        mf_s = f"{mf:.6f}" if mf == mf else "nan"
        sf_s = f"{sf:.6f}" if sf == sf else "nan"
        log(f"{row['rank']:>4}  {row['algorithm']:<10}  "
            f"{row['n_feasible']:>4}/{row['n_runs']:<4}  "
            f"{bf_s:>10}  {mf_s:>10}  {sf_s:>10}")

    log("")
    log("=" * 78)
    log("CROSS-CHECK SUMMARY")
    log("=" * 78)
    if n_crosschecked == 0:
        log("No seeds had a pre-existing checkpoint value to cross-check against "
            "(nothing to compare -- e.g. a completely fresh run).")
    elif n_mismatches == 0:
        log(f"All {n_crosschecked} re-run seed(s) with a prior recorded result "
            f"matched that result to within {MISMATCH_TOL:g} (f and cv). "
            f"No mismatches.")
    else:
        log(f"*** {n_mismatches} of {n_crosschecked} re-run seed(s) MISMATCHED "
            f"their prior recorded result by more than {MISMATCH_TOL:g}. "
            f"See the '*** MISMATCH ***' lines above for which algorithm/seed "
            f"and by how much. Do not trust the summary/significance numbers "
            f"until this is understood. ***")

    log("")
    log(f"Finished: {datetime.datetime.now().isoformat(timespec='seconds')}")
    log(f"Raw per-run CSV:       {RAW_CSV}")
    log(f"Summary CSV:           {SUMMARY_CSV}")
    log(f"Convergence curves:    {CONV_DIR}/<algorithm>.json  (all {len(SEEDS)} seeds)")
    log(f"Full console log:      {LOG_TXT}")
    log("")
    log("Next step: run rc22_significance.py for Wilcoxon rank-sum p-values "
        "(Holm-Bonferroni corrected) and Vargha-Delaney A12 effect sizes, "
        "then make_rc22_figures.py for the figures.")


if __name__ == "__main__":
    main()
