"""
make_rc22_figures.py
=======================
Generates the script-derived (non-illustrative) figures for the RC22
results section, from the CSVs already in Results/ plus the
convergence captures from rc22_convergence_capture.py.

House style applied to every figure (per project requirement):
  * Vector output: PDF, with real embedded text (pdf.fonttype = 42,
    i.e. TrueType) -- not a rasterized image and not Type-3 bitmap
    fonts that print/editing software can mangle.
  * No titles anywhere (no ax.set_title, no fig.suptitle). The caption
    in the paper carries that information instead.
  * Everything bold: axis labels, tick labels, legend text, and any
    in-plot annotations. Grid lines are drawn with a heavier weight so
    they read clearly at print size, not just "on".

Figures produced (all written to Results/figures/):
  fig1_convergence.pdf        BalOA vs. a chosen subset of competitors,
                               mean Deb-rule display value (+/- std
                               band) across all captured seeds vs.
                               FES, log-y. Requires convergence_rc22/
                               *.json, which run_rc22_comparison.py
                               now writes directly as part of its main
                               pass (rc22_convergence_capture.py is
                               retired) -- skipped with a clear message
                               if missing.
  fig2_boxplot.pdf            Feasible-run f distribution, all 17
                               algorithms, sorted by mean, BalOA
                               highlighted.
  fig3_mean_std_bar.pdf       Mean +/- std bar chart, same ordering.
  fig4_feasibility.pdf        Feasibility rate (n_feasible/n_runs),
                               all 17 algorithms.
  fig5_effect_size_forest.pdf Vargha-Delaney A12 vs. BalOA per
                               competitor, Holm-significance marked,
                               from rc22_significance.csv.

Usage (run locally, per project rule #3):
    python make_rc22_figures.py
"""

from __future__ import annotations
import os
import csv
import json

import numpy as np
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# House style
# ---------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.linewidth": 1.4,
    "font.size": 11,
    "pdf.fonttype": 42,   # embed real (TrueType) vector text, not Type 3
    "ps.fonttype": 42,
    "savefig.transparent": False,
})

BALOA_COLOR = "#C0392B"     # BalOA always highlighted in this color
OTHER_COLOR = "#5B6B7A"
GRID_KW = dict(linewidth=0.9, alpha=0.55)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "Results"))
RAW_CSV = os.path.join(RESULTS_DIR, "rc22_raw_runs.csv")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "rc22_summary.csv")
SIG_CSV = os.path.join(RESULTS_DIR, "rc22_significance.csv")
CONV_DIR = os.path.join(RESULTS_DIR, "convergence_rc22")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Subset plotted in the convergence figure (edit freely) -- kept small
# so the figure stays legible; run_rc22_comparison.py saves convergence
# data for ALL 17 algorithms and all 20 seeds, so any subset can be
# swapped in later without re-running anything.
CONVERGENCE_SUBSET = ["BalOA", "HHO", "GTO", "GWO", "SMA", "GA"]


def _bold_ticks(ax) -> None:
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")


def _bold_legend(leg) -> None:
    if leg is None:
        return
    for text in leg.get_texts():
        text.set_fontweight("bold")


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------
def _load_raw() -> dict[str, list[float]]:
    """algorithm -> list of f values from FEASIBLE runs only."""
    data: dict[str, list[float]] = {}
    with open(RAW_CSV, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["feasible"].strip().lower() not in ("true", "1", "yes"):
                continue
            data.setdefault(row["algorithm"], []).append(float(row["f_max_gear_error"]))
    return data


def _load_summary() -> list[dict]:
    with open(SUMMARY_CSV, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_significance() -> list[dict]:
    with open(SIG_CSV, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------
# Figure 1 -- convergence
# ---------------------------------------------------------------------
def fig1_convergence() -> None:
    missing = [name for name in CONVERGENCE_SUBSET
               if not os.path.exists(os.path.join(CONV_DIR, f"{name}.json"))]
    if missing or not os.path.isdir(CONV_DIR):
        print(f"[fig1] skipped -- run run_rc22_comparison.py first "
              f"(missing: {missing or 'convergence_rc22/ folder itself'}).")
        return

    # Two panels: (a) the full log-scale curve (shows the early-search
    # drop, but the tail where all algorithms converge is too tightly
    # bunched to tell apart), and (b) a linear-scale zoom on that tail
    # region, so the late-FES ranking is actually readable.
    fig, (ax, axz) = plt.subplots(1, 2, figsize=(11.0, 4.4))

    curves = {}  # name -> (fes, mean_vals, std_vals, n_seeds)
    for name in CONVERGENCE_SUBSET:
        with open(os.path.join(CONV_DIR, f"{name}.json"), "r", encoding="utf-8") as f:
            data = json.load(f)   # {"<seed>": [[fes, disp], ...], ...}

        # Every seed shares the identical FES sampling grid (see capture
        # script's docstring), so stacking the display values column-wise
        # and averaging is exact -- not an interpolated approximation.
        seed_curves = list(data.values())
        fes = np.array([p[0] for p in seed_curves[0]])
        vals = np.array([[p[1] for p in curve] for curve in seed_curves])
        mean_vals = vals.mean(axis=0)
        std_vals = vals.std(axis=0)
        curves[name] = (fes, mean_vals, std_vals, len(seed_curves))

    # Zoom window: last quarter of the FES budget on the x-axis, and a
    # y-range tight around where the mean curves actually sit there
    # (computed from the data itself so this stays correct if the
    # algorithm subset or problem scale ever changes).
    any_fes = next(iter(curves.values()))[0]
    x_zoom_lo = 0.75 * any_fes[-1]
    tail_vals = []
    for fes, mean_vals, std_vals, _ in curves.values():
        tail_vals.extend(mean_vals[fes >= x_zoom_lo].tolist())
    y_zoom_lo = min(tail_vals) * 0.97
    y_zoom_hi = max(tail_vals) * 1.03

    for name, (fes, mean_vals, std_vals, n_seeds) in curves.items():
        color = BALOA_COLOR if name == "BalOA" else None
        lw = 2.6 if name == "BalOA" else 1.6
        line, = ax.plot(fes, mean_vals, label=f"{name} (n={n_seeds})",
                        color=color, linewidth=lw)
        # Clamp the lower band to a small positive epsilon: the y-axis is
        # log-scale, and mean-std can dip to <=0 early in the search when
        # the handful of seeds have wildly different penalized values.
        lower = np.maximum(mean_vals - std_vals, 1e-3)
        ax.fill_between(fes, lower, mean_vals + std_vals,
                        color=line.get_color(), alpha=0.15, linewidth=0)
        axz.plot(fes, mean_vals, color=line.get_color(), linewidth=lw + 0.4)

    ax.set_yscale("log")
    ax.set_xlabel("Function evaluations (FES)", fontweight="bold")
    # Split across two lines: the single-line version is long enough that
    # its rotated bounding box gets clipped by some PDF viewers/renderers
    # regardless of tight_layout/bbox_inches settings.
    ax.set_ylabel("Mean best-so-far objective +/- std\n(Deb-rule scalar)",
                  fontweight="bold", multialignment="center")
    ax.grid(True, which="both", **GRID_KW)
    _bold_ticks(ax)
    leg = ax.legend(frameon=False, loc="upper right")
    _bold_legend(leg)
    ax.text(0.03, 0.92, "(a)", transform=ax.transAxes, fontweight="bold",
            fontsize=13)

    # Mark the zoomed region on panel (a) and connect it to panel (b),
    # same convention as a standard "full view + zoomed inset" figure.
    rect_x = [x_zoom_lo, any_fes[-1], any_fes[-1], x_zoom_lo, x_zoom_lo]
    rect_y = [y_zoom_lo, y_zoom_lo, y_zoom_hi, y_zoom_hi, y_zoom_lo]
    ax.plot(rect_x, rect_y, color="black", linewidth=1.0, linestyle="--")
    ax.annotate("see (b)", xy=(x_zoom_lo, y_zoom_hi), xytext=(-55, 10),
                textcoords="offset points", fontweight="bold", fontsize=9)

    axz.set_xlim(x_zoom_lo, any_fes[-1])
    axz.set_ylim(y_zoom_lo, y_zoom_hi)
    axz.set_xlabel("Function evaluations (FES)", fontweight="bold")
    axz.set_ylabel("Mean best-so-far objective (zoomed)", fontweight="bold")
    axz.grid(True, which="both", **GRID_KW)
    _bold_ticks(axz)
    axz.text(0.03, 0.92, "(b)", transform=axz.transAxes, fontweight="bold",
             fontsize=13)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig1_convergence.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig1] saved -> {out}")


# ---------------------------------------------------------------------
# Figure 2 -- boxplot of feasible f, sorted by mean
# ---------------------------------------------------------------------
def fig2_boxplot() -> None:
    data = _load_raw()
    order = sorted(data.keys(), key=lambda k: np.mean(data[k]))
    values = [data[k] for k in order]

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    # NOTE: matplotlib >=3.9 renamed boxplot's `labels` kwarg to
    # `tick_labels`; set tick labels manually afterwards instead so this
    # works across matplotlib versions without a version check.
    bp = ax.boxplot(values, patch_artist=True, showfliers=True, widths=0.6)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order)
    for patch, name in zip(bp["boxes"], order):
        patch.set_facecolor(BALOA_COLOR if name == "BalOA" else "#D9DEE3")
        patch.set_edgecolor("black")
        patch.set_linewidth(1.2)
    for element in ("whiskers", "caps", "medians"):
        for line in bp[element]:
            line.set_color("black")
            line.set_linewidth(1.2)

    ax.set_ylabel("Feasible-run objective value f", fontweight="bold")
    ax.grid(True, axis="y", **GRID_KW)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    _bold_ticks(ax)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig2_boxplot.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"[fig2] saved -> {out}")


# ---------------------------------------------------------------------
# Figure 3 -- mean +/- std bar chart
# ---------------------------------------------------------------------
def fig3_mean_std_bar() -> None:
    rows = _load_summary()
    rows = sorted(rows, key=lambda r: float(r["mean_f"]))
    names = [r["algorithm"] for r in rows]
    means = [float(r["mean_f"]) for r in rows]
    stds = [float(r["std_f"]) for r in rows]
    colors = [BALOA_COLOR if n == "BalOA" else OTHER_COLOR for n in names]

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = np.arange(len(names))
    ax.bar(x, means, yerr=stds, capsize=3, color=colors, edgecolor="black",
           linewidth=1.0, error_kw={"elinewidth": 1.4, "capthick": 1.4})
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Mean objective value f (+/- std)", fontweight="bold")
    ax.grid(True, axis="y", **GRID_KW)
    _bold_ticks(ax)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig3_mean_std_bar.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"[fig3] saved -> {out}")


# ---------------------------------------------------------------------
# Figure 4 -- feasibility rate
# ---------------------------------------------------------------------
def fig4_feasibility() -> None:
    rows = _load_summary()
    rows = sorted(rows, key=lambda r: float(r["feasibility_rate"]))
    names = [r["algorithm"] for r in rows]
    rates = [float(r["feasibility_rate"]) * 100.0 for r in rows]
    colors = [BALOA_COLOR if n == "BalOA" else OTHER_COLOR for n in names]

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = np.arange(len(names))
    ax.bar(x, rates, color=colors, edgecolor="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Feasibility rate (%)", fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y", **GRID_KW)
    _bold_ticks(ax)
    for xi, r in zip(x, rates):
        ax.annotate(f"{r:.0f}%", (xi, r + 2), ha="center", fontweight="bold",
                    fontsize=9)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig4_feasibility.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"[fig4] saved -> {out}")


# ---------------------------------------------------------------------
# Figure 5 -- effect-size forest plot (A12 vs. BalOA, Holm significance)
# ---------------------------------------------------------------------
def fig5_effect_size_forest() -> None:
    rows = _load_significance()
    rows = sorted(rows, key=lambda r: float(r["A12"]))
    names = [r["competitor"] for r in rows]
    a12 = [float(r["A12"]) for r in rows]
    sig = [r["significant_after_holm"].strip().lower() == "true" for r in rows]

    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    y = np.arange(len(names))
    for yi, (val, s) in enumerate(zip(a12, sig)):
        ax.plot([0.5, val], [yi, yi], color="#9AA5AD", linewidth=1.3, zorder=1)
        ax.scatter([val], [yi],
                   marker="o",
                   s=70 if s else 55,
                   facecolor=BALOA_COLOR if s else "white",
                   edgecolor=BALOA_COLOR,
                   linewidth=1.6, zorder=2)

    ax.axvline(0.5, color="black", linewidth=1.3, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Vargha-Delaney A12 (BalOA vs. competitor)", fontweight="bold")
    ax.set_xlim(0.0, 1.05)
    ax.grid(True, axis="x", **GRID_KW)
    _bold_ticks(ax)

    # Legend built manually (filled = significant after Holm correction,
    # hollow = not significant at alpha=0.05) since the marker styling
    # above encodes significance rather than a plotted series.
    sig_marker = plt.Line2D([0], [0], marker="o", linestyle="", markersize=8,
                             markerfacecolor=BALOA_COLOR, markeredgecolor=BALOA_COLOR,
                             label="Significant (Holm, alpha=0.05)")
    nonsig_marker = plt.Line2D([0], [0], marker="o", linestyle="", markersize=8,
                                markerfacecolor="white", markeredgecolor=BALOA_COLOR,
                                label="Not significant")
    # Every stem line starts at x=0.5 and runs right, so there is no
    # y-position where an in-plot legend is guaranteed clear of them --
    # that's what caused the overlap before. Placed below the axes
    # instead (outside the data area entirely), so it can never overlap
    # a marker or line regardless of how the A12 values fall.
    leg = ax.legend(handles=[sig_marker, nonsig_marker], frameon=False,
                     loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    _bold_legend(leg)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig5_effect_size_forest.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig5] saved -> {out}")


def main() -> None:
    print("=" * 78)
    print("RC22 figure generation")
    print(f"Output folder: {FIG_DIR}")
    print("=" * 78)
    fig1_convergence()
    fig2_boxplot()
    fig3_mean_std_bar()
    fig4_feasibility()
    fig5_effect_size_forest()
    print("")
    print("Done.")


if __name__ == "__main__":
    main()
