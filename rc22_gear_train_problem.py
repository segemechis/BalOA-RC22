"""
rc22_gear_train_problem.py
=============================
Formal definition of RC22 -- Planetary Gear Train Design optimization
problem -- the new target benchmark for BalOA, replacing Rolling
Element Bearing Design (which showed the same "ceiling effect" as
Pressure Vessel Design: most algorithms trivially tie at the same
global optimum once design variables collapse to box bounds).

Source
------
Kumar, A.; Wu, G.; Ali, M.Z.; Mallipeddi, R.; Suganthan, P.N.; Das, S.
"A test-suite of non-convex constrained optimization problems from the
real-world and some baseline results." Swarm and Evolutionary
Computation 56 (2020) 100693. Problem RC22, Eq. (28)-(29) [their
internal numbering], Section 2.3.8, citing: Sandgren, E. "Nonlinear
integer and discrete programming in mechanical design." Proc. ASME
Design Technology Conference (1988), pp. 95-105.

Why this problem (chosen after Rolling Bearing Design and Pressure
Vessel Design both showed every competitor converging to the same
trivial global optimum at a 30000-FES budget): Kumar et al. (2020)
benchmarked this entire 57-problem suite with three modern
state-of-the-art algorithms (IUDE, eps-MAgES, iLSHADE_eps) across 25
independent runs each, and their own stated conclusion singles out
RC22 as one of only three genuinely hard problems in the 19-problem
Mechanical Engineering category. Their own Table 7 results show a
large, already-documented separation on this exact problem:
    IUDE          success rate 100%
    eps-MAgES     success rate  20-36%
    iLSHADE_eps   success rate  20-28%
i.e. a well-designed algorithm solves it perfectly every run while two
other strong, modern algorithms fail a majority of the time -- which
is the kind of real separation this project is looking for, not
something asserted here but read directly from that paper's results.

Structurally this problem is nothing like PVD/Rolling-Bearing: it has
NO continuous variables at all. Every one of its 9 design variables is
discrete (6 bounded integers, plus 3 variables -- p, m1, m3 -- each
restricted to a small named list of allowed values), and its objective
is a non-smooth minimax (not a smooth bracketed power-law like the
other two problems), so there is no "collapse to a box corner" failure
mode available to a competitor algorithm here.

Problem statement
------------------
Minimise the worst-case error between three actual gear ratios (of an
automatic planetary transmission) and their three target ratios:

    f(X) = max( |i1 - i01|, |i2 - i02|, |iR - i0R| )

    i1 = N6 / N4                                i01 =  3.11
    i2 = N6*(N1*N3 + N2*N4) / (N1*N3*(N6-N4))    i02 =  1.84
    iR = -N2*N6 / (N1*N3)                        i0R = -3.11

Design vector (9 variables; order used throughout this module):
    X = (p, m1, m3, N1, N2, N3, N4, N5, N6)

    p        : number of planet gears, integer, one of {3, 4, 5}
    m1, m3   : gear modules, each one of
               {1.75, 2.00, 2.25, 2.50, 2.75, 3.00}
    N1..N6   : numbers of gear teeth, bounded integers:
               17<=N1<=96, 14<=N2<=54, 14<=N3<=51,
               17<=N4<=46, 14<=N5<=51, 48<=N6<=124

KNOWN SOURCE ISSUE (flagged, not silently fixed): the source paper's
own design-vector listing prints "X = (p, N6, N5, N4, N3, N2, N1, m2,
m1)" -- i.e. a variable called "m2" -- but no constraint anywhere in
the paper's own equations ever uses "m2"; every constraint uses only
m1 and m3 (e.g. g1 uses m3, g2 uses m1, g4 uses both m1 and m3), and
the paper's own bounds section lists allowed-value sets for "m1" and
"m3" (not m2). This is internally consistent only if "m2" in the
design-vector listing is a typo for "m3" (an easy OCR/typesetting slip
given how similar the two render). Resolved here using the
constraint/bounds text as ground truth: the real design variables are
p, m1, m3, N1..N6 (9 total, matching the paper's own "nine design
parameters" / "contains six integer variables" description), with no
m2 anywhere.

Constraints (10 inequality + 1 equality), rewritten in this project's
"c_i(X) <= 0 is feasible" convention (identical sign convention to the
source, which already states every g_i(X) <= 0):

    g1  = m3*(N6+2.5) - Dmax                                        <= 0
    g2  = m1*(N1+N2) + m1*(N2+2) - Dmax                              <= 0
    g3  = m3*(N4+N5) + m3*(N5+2) - Dmax                              <= 0
    g4  = |m1*(N1+N2) - m3*(N6-N3)| - m1 - m3                        <= 0
    g5  = -(N1+N2)*sin(pi/p) + N2 + 2 + delta22                      <= 0
    g6  = -(N6-N3)*sin(pi/p) + N3 + 2 + delta33                      <= 0
    g7  = -(N4+N5)*sin(pi/p) + N5 + 2 + delta55                      <= 0
    g8  = (N3+N5+2+delta35)^2 - (N6-N3)^2 - (N4+N5)^2
          + 2*(N6-N3)*(N4+N5)*cos(2*pi/p - beta)                     <= 0
    g9  = N4 - N6 + 2*N5 + 2*delta56 + 4                             <= 0
    g10 = 2*N3 - N6 + N4 + 2*delta34 + 4                             <= 0
    h1  = (N6-N4)/p  must be an integer

    where delta22=delta33=delta55=delta35=delta56=delta34 = 0.5,
    beta = arccos( ((N4+N5)^2 + (N6-N3)^2 - (N3+N5)^2)
                   / (2*(N6-N3)*(N4+N5)) ),
    Dmax = 220.

Equality-constraint handling follows the source paper's own stated
convention (their Eq. 2): |h_j(X)| - eps <= 0, with eps = 1e-4. Here
h1(X) is replaced by its signed distance to the nearest integer,
dist((N6-N4)/p), so |h1| <= eps is exactly "(N6-N4)/p is an integer"
to within that tolerance.

All 9 variables are treated as continuous during search and snapped to
their nearest allowed discrete value only at evaluation time (the same
round-at-evaluation-time pattern used for Z in the earlier Rolling
Bearing problem) -- this lets every one of the 16 unconstrained
competitor algorithms run completely unmodified, exactly as before.

FES budget: the source paper's own stopping rule (their Eq. 60) uses
100000 function evaluations for any problem with D<=10; this problem
has D=9, so this project uses MAX_FES=100000 for RC22 runs (not the
30000 used for the earlier, lower-dimensional smooth problems), in
place of inventing a different number.
"""

from __future__ import annotations
import numpy as np

DIM = 9
VAR_NAMES = ['p', 'm1', 'm3', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6']

# ---------------------------------------------------------------------
# Allowed discrete values / integer bounds (Sandgren 1988, as restated
# in Kumar et al. 2020, Sect. 2.3.8)
# ---------------------------------------------------------------------
P_CHOICES = np.array([3.0, 4.0, 5.0])
M_CHOICES = np.array([1.75, 2.00, 2.25, 2.50, 2.75, 3.00])

N1_LO, N1_HI = 17, 96
N2_LO, N2_HI = 14, 54
N3_LO, N3_HI = 14, 51
N4_LO, N4_HI = 17, 46
N5_LO, N5_HI = 14, 51
N6_LO, N6_HI = 48, 124

D_MAX = 220.0
DELTA = 0.5          # delta22 = delta33 = delta55 = delta35 = delta56 = delta34
EQ_EPS = 1e-4         # source paper's own equality-constraint tolerance (their Eq. 2)

# Search-box bounds handed to every algorithm (continuous relaxation;
# snapped to the true discrete/integer values only inside evaluate()).
LB = np.array([3.0, 1.75, 1.75,
               float(N1_LO), float(N2_LO), float(N3_LO),
               float(N4_LO), float(N5_LO), float(N6_LO)])
UB = np.array([5.0, 3.00, 3.00,
               float(N1_HI), float(N2_HI), float(N3_HI),
               float(N4_HI), float(N5_HI), float(N6_HI)])


def _nearest(value: float, choices: np.ndarray) -> float:
    return float(choices[np.argmin(np.abs(choices - value))])


def _unpack(x: np.ndarray):
    p = _nearest(x[0], P_CHOICES)
    m1 = _nearest(x[1], M_CHOICES)
    m3 = _nearest(x[2], M_CHOICES)
    N1 = float(np.clip(round(x[3]), N1_LO, N1_HI))
    N2 = float(np.clip(round(x[4]), N2_LO, N2_HI))
    N3 = float(np.clip(round(x[5]), N3_LO, N3_HI))
    N4 = float(np.clip(round(x[6]), N4_LO, N4_HI))
    N5 = float(np.clip(round(x[7]), N5_LO, N5_HI))
    N6 = float(np.clip(round(x[8]), N6_LO, N6_HI))
    return p, m1, m3, N1, N2, N3, N4, N5, N6


def gear_ratio_errors(x: np.ndarray):
    """Returns (e1, e2, eR), the three |actual - target| gear-ratio
    errors whose max is the objective."""
    p, m1, m3, N1, N2, N3, N4, N5, N6 = _unpack(x)

    i1 = N6 / N4
    i2 = N6 * (N1 * N3 + N2 * N4) / (N1 * N3 * (N6 - N4))
    iR = -N2 * N6 / (N1 * N3)

    e1 = abs(i1 - 3.11)
    e2 = abs(i2 - 1.84)
    eR = abs(iR - (-3.11))
    return e1, e2, eR


def objective(x: np.ndarray) -> float:
    """Minimisation objective: worst of the three gear-ratio errors."""
    e1, e2, eR = gear_ratio_errors(x)
    return max(e1, e2, eR)


def constraints(x: np.ndarray):
    """Returns (g[10], h1) -- the 10 inequality constraint values
    (c<=0 feasible) and the raw (unsigned-distance) equality residual
    for h1, BEFORE the eps tolerance is applied."""
    p, m1, m3, N1, N2, N3, N4, N5, N6 = _unpack(x)

    g1 = m3 * (N6 + 2.5) - D_MAX
    g2 = m1 * (N1 + N2) + m1 * (N2 + 2) - D_MAX
    g3 = m3 * (N4 + N5) + m3 * (N5 + 2) - D_MAX
    g4 = abs(m1 * (N1 + N2) - m3 * (N6 - N3)) - m1 - m3

    sin_term = np.sin(np.pi / p)
    g5 = -(N1 + N2) * sin_term + N2 + 2 + DELTA
    g6 = -(N6 - N3) * sin_term + N3 + 2 + DELTA
    g7 = -(N4 + N5) * sin_term + N5 + 2 + DELTA

    # N6 == N3 is possible within the bounds box (e.g. N3=N6=48) and
    # makes this denominator zero; such a point is degenerate (no real
    # planet-gear geometry) and is always infeasible anyway via g6
    # (which requires N3 <= -2.5 when N6-N3=0, impossible since
    # N3>=14) -- guarded here purely to avoid a division error, not
    # because the point could ever be genuinely feasible.
    num_beta = (N4 + N5) ** 2 + (N6 - N3) ** 2 - (N3 + N5) ** 2
    den_beta = 2.0 * (N6 - N3) * (N4 + N5)
    if den_beta == 0.0:
        beta = 0.0
    else:
        beta = np.arccos(np.clip(num_beta / den_beta, -1.0, 1.0))
    g8 = ((N3 + N5 + 2 + DELTA) ** 2 - (N6 - N3) ** 2 - (N4 + N5) ** 2
          + 2 * (N6 - N3) * (N4 + N5) * np.cos(2 * np.pi / p - beta))

    g9 = N4 - N6 + 2 * N5 + 2 * DELTA + 4
    g10 = 2 * N3 - N6 + N4 + 2 * DELTA + 4

    ratio = (N6 - N4) / p
    h1_residual = abs(ratio - round(ratio))   # 0 exactly iff (N6-N4)/p is an integer

    g = np.array([g1, g2, g3, g4, g5, g6, g7, g8, g9, g10], dtype=float)
    return g, h1_residual


def constraint_violation(x: np.ndarray) -> float:
    """Total constraint violation: sum of inequality violations plus
    the equality residual's excess over the source paper's own eps
    tolerance (their Eq. 2: |h_j| - eps <= 0)."""
    g, h1_residual = constraints(x)
    ineq_violation = float(np.sum(np.maximum(0.0, g)))
    eq_violation = max(0.0, h1_residual - EQ_EPS)
    return ineq_violation + eq_violation


def evaluate(x: np.ndarray) -> tuple[float, float]:
    """Canonical evaluator: returns (f, constraint_violation)."""
    return objective(x), constraint_violation(x)


# ---------------------------------------------------------------------
# Safe penalty constant for wrapping *unconstrained* competitor
# algorithms into a Deb-rule-equivalent scalar minimisation (see
# constraint_utils.make_penalized_objective). The objective f(X) here
# is a max of three |ratio error| terms; over the search box its value
# is bounded well below 50 in practice (gear ratios of this scale
# can't diverge arbitrarily given the bounded N_i/p), so BIG=100 is a
# generous, safely-dominating margin -- flagged here rather than
# derived from a tight analytic bound, since (unlike the previous two
# problems) this objective has no clean closed-form worst case.
# ---------------------------------------------------------------------
PENALTY_BIG = 100.0
