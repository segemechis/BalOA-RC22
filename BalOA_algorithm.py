"""
Baalle Optimization Algorithm (BalOA)
=======================================
Inspired by the Baalle (Baallee) indigenous governance system of the
Gedeo people, southern Ethiopia.

NOTE ON NAMING: earlier scripts in this project used the placeholder
class name "BOA" before it was discovered that abbreviation was already
taken by a previously published algorithm. The correct, final name
used from this point forward is BalOA (Baalle Optimization Algorithm).

Cultural foundation
--------------------
The Gedeo Baalle system organises all male members into nine rotating
grade classes (Qadaaddo through Cowwajje) within three autonomous
territorial regions called Roga (Subbo Roga, Dhiibata Roga, Riqata Roga).
Leadership is exercised by the fifth grade (Luba), governed by two
alternating political bodies -- Bilbaana and Dhaalaana. The Songo
institutions serve as centres of governance, law (Seera), and conflict
resolution. Key mechanisms adapted here:

  Faaci'e   Confession / purification ritual  --  fires on diversity collapse
  Gondooro  "Never again" ritual              --  fires on individual stagnation
  Murra     Messenger / information carrier   --  cyclic cross-Roga migration
  Cowwajje  Return to childhood               --  full renewal after cycle
  Songo     Governance / cultural archive     --  quality-diversity elite store
  Bilbaana /
  Dhaalaana Alternating Luba lineages         --  structural diversity in elite

Cultural-consistency note (resolved in this revision)
-------------------------------------------------------
Each Roga is horizontally autonomous: by default its Raaba (G4) and
Luba (G5) agents search using their OWN Roga's current best solution,
never a constantly-shared global attractor. Abba Gada -- elected from
Luba across all three Rogas -- is only consulted by a given Roga once
that Roga's OWN best solution has failed to improve (under Deb's
feasibility rules) for `T_escalate` consecutive iterations. This models
the cultural claim that escalation to Abba Gada happens on Roga
failure, not continuously. The escalation flag clears as soon as that
Roga's own best improves again, or at the next Ya'a re-election.

This revision also fixes two implementation bugs identified during
verification:
  * G4 Raaba's random donor selection could previously include the
    acting agent's own index, reducing DE-style mutation diversity.
    Donors are now drawn only from the OTHER agents in the Roga.
  * G6 Yuuba previously trained against the Luba grade's best value
    from the START of the iteration (before G5 had been updated that
    same iteration). Yuuba now trains against Luba's current best,
    recomputed immediately after the G5 sub-loop completes.

Constraint handling (new in this revision)
---------------------------------------------
BalOA originally carried a dead `cv` (constraint-violation) parameter
throughout SongoArchive with no actual constraint handling anywhere.
This revision makes every comparison point in the algorithm genuinely
constraint-aware via Deb's feasibility rules (see constraint_utils.py):
SongoArchive admission/worst-tracking, main-loop greedy acceptance,
Ya'a (Abba Gada election), Murra (migration), Faaci'e, Gondooro,
Cowwajje, and global-best tracking all now compare (f, cv) pairs
instead of a bare scalar. `func` passed to BalOA must now return a
(f, cv) tuple -- see pressure_vessel_problem.evaluate for the canonical
example.

References
----------
[1] Desalegn Bekele (2019). AAU MA Thesis -- Gedeo indigenous communication.
[2] Golle, Zekareas & Orkaydo (2021). Kervan Journal 25/2, 127-152.
[3] Kippie Kanshie et al. (2008). History of the Gedeo Nation.
[4] Wikipedia: Gedeo people (Accessed 2026).

Author : Sena Gemechis File
         Faculty of Electrical and Computer Engineering,
         Jimma Institute of Technology, Jimma University, Ethiopia
"""

from __future__ import annotations
import numpy as np
from scipy.special import gamma as _sp_gamma

from constraint_utils import deb_worse, deb_best_index, deb_sort_indices


# ===========================================================================
# SONGO ARCHIVE
# ===========================================================================

class SongoArchive:
    """
    Quality-diversity archive modelled on the Songo institution.
    Admits a candidate solution if:
      (a) it Deb-dominates the archive worst, OR
      (b) its minimum distance to all archived solutions exceeds sigma_S.
    """

    def __init__(self, capacity: int, sigma_s: float,
                 dim: int, lb: np.ndarray, ub: np.ndarray) -> None:
        self.capacity = capacity
        self._thresh  = sigma_s * np.linalg.norm(ub - lb)
        self.dim      = dim
        self.lb       = lb.copy()
        self.ub       = ub.copy()
        self._X: list[np.ndarray] = []
        self._F: list[float]      = []
        self._CV: list[float]     = []

    def _worst_idx(self) -> int:
        w = 0
        for i in range(1, len(self._X)):
            if deb_worse(self._F[i], self._CV[i], self._F[w], self._CV[w]):
                w = i
        return w

    def update(self, x: np.ndarray, f: float, cv: float = 0.0) -> None:
        x = x.copy()
        if len(self._X) < self.capacity:
            self._X.append(x); self._F.append(f); self._CV.append(cv)
            return
        min_d = min(np.linalg.norm(x - s) for s in self._X)
        wi    = self._worst_idx()
        wf, wcv = self._F[wi], self._CV[wi]
        admit   = deb_worse(wf, wcv, f, cv) or (min_d > self._thresh)
        if admit:
            self._X[wi] = x; self._F[wi] = f; self._CV[wi] = cv

    def get_best(self) -> tuple[np.ndarray | None, float, float]:
        if not self._X:
            return None, np.inf, np.inf
        bi = deb_best_index(self._F, self._CV, list(range(len(self._X))))
        return self._X[bi].copy(), self._F[bi], self._CV[bi]

    def get_random(self) -> np.ndarray:
        if not self._X:
            return self.lb + np.random.rand(self.dim) * (self.ub - self.lb)
        return self._X[np.random.randint(len(self._X))].copy()

    def get_diverse(self, x_ref: np.ndarray) -> np.ndarray:
        """Return the archived solution most distant from x_ref."""
        if not self._X:
            return self.lb + np.random.rand(self.dim) * (self.ub - self.lb)
        dists = [np.linalg.norm(x_ref - s) for s in self._X]
        return self._X[int(np.argmax(dists))].copy()

    @property
    def empty(self) -> bool:
        return len(self._X) == 0


# ===========================================================================
# BAALLE OPTIMIZATION ALGORITHM
# ===========================================================================

class BalOA:
    """
    Baalle Optimization Algorithm (BalOA).

    Population : N_ROGA = 3 Rogas  x  N_PER_ROGA = 19 agents  =  57 total.

    Grade distribution per Roga (static assignment):
      G1 Qadaaddo  : 2 agents   local idx [0-1]   broad Levy exploration
      G2 Siida     : 2 agents   local idx [2-3]   territorial hardening
      G3 Lumaasa   : 2 agents   local idx [4-5]   silent preparation
      G4 Raaba     : 3 agents   local idx [6-8]   3-strategy competition
                                 (Sanachcha / Mosiisa / Ya'a candidate moves)
      G5 Luba      : 4 agents   local idx [9-12]  governing (Bilbaana:[9-10] / Dhaalaana:[11-12])
      G6 Yuuba     : 2 agents   local idx [13-14] senior apprenticeship
      G7 Guduro    : 2 agents   local idx [15-16] non-uniform refinement
      G8 Qulullo   : 1 agent    local idx [17]    opposition-based
      G9 Cowwajje  : 1 agent    local idx [18]    periodic renewal

    `func(x)` MUST return a (f, cv) tuple -- objective value and total
    constraint violation (0.0 for a feasible point). For an unconstrained
    problem, pass a wrapper returning (f(x), 0.0).
    """

    N_ROGA      = 3
    GRADE_SIZES = [2, 2, 2, 3, 4, 2, 2, 1, 1]
    GRADE_NAMES = ['Qadaaddo', 'Siida', 'Lumaasa', 'Raaba', 'Luba',
                   'Yuuba',    'Guduro', 'Qulullo', 'Cowwajje']
    N_PER_ROGA  = 19     # sum(GRADE_SIZES)
    N_TOTAL     = 57     # N_ROGA * N_PER_ROGA

    # ------------------------------------------------------------------
    def __init__(self,
                 func,
                 lb      : list | np.ndarray,
                 ub      : list | np.ndarray,
                 dim     : int,
                 max_fes : int,
                 *,
                 T_B        : int   = 25,   # Baallee Ceremony period
                 T_M        : int   = 8,    # Murra migration period
                 T_C        : int   = 50,   # Cowwajje renewal cycle
                 T_G        : int   = 15,   # Ya'a assembly (Abba Gada election)
                 T_L        : int   = 20,   # Bilbaana/Dhaalaana alternation
                 T_stag     : int   = 8,    # Gondooro individual-stagnation threshold
                 T_escalate : int   = 10,   # Roga-failure escalation threshold
                 N_S        : int   = 10,   # Songo archive capacity
                 D_min      : float = 0.05, # Faaci'e diversity threshold
                 sigma_s    : float = 0.05, # Songo novelty threshold
                 # ---- Ablation switches (all default False = full BalOA,
                 # bit-for-bit identical to every run before this revision).
                 # Each isolates exactly one named cultural mechanism; see
                 # rc22_ablation.py for the controlled comparison this
                 # enables.
                 disable_faacie        : bool = False,  # no diversity-collapse purification
                 disable_gondooro      : bool = False,  # no individual-stagnation escape
                 disable_murra         : bool = False,  # no cyclic cross-Roga migration
                 disable_bilbaana_swap : bool = False,  # lineage role never alternates
                 disable_baallee       : bool = False,  # no periodic G1 reseed ceremony
                 disable_songo         : bool = False,  # archive stays permanently empty
                 disable_escalation    : bool = False,  # Abba Gada never consulted on failure
                 disable_cowwajje      : bool = False,  # no periodic full renewal
                 ) -> None:

        self.func    = func
        self.lb      = np.asarray(lb, dtype=float)
        self.ub      = np.asarray(ub, dtype=float)
        self.dim     = dim
        self.max_fes = max_fes
        self.T_B  = T_B;  self.T_M = T_M;  self.T_C = T_C
        self.T_G  = T_G;  self.T_L = T_L;  self.T_stag = T_stag
        self.T_escalate = T_escalate
        self.N_S  = N_S;  self.D_min = D_min; self.sigma_s = sigma_s

        self.disable_faacie        = disable_faacie
        self.disable_gondooro      = disable_gondooro
        self.disable_murra         = disable_murra
        self.disable_bilbaana_swap = disable_bilbaana_swap
        self.disable_baallee       = disable_baallee
        self.disable_songo         = disable_songo
        self.disable_escalation    = disable_escalation
        self.disable_cowwajje      = disable_cowwajje

        # Pre-compute grade index ranges (local to one Roga)
        self._gr: list[list[int]] = []
        s = 0
        for sz in self.GRADE_SIZES:
            self._gr.append(list(range(s, s + sz))); s += sz
        # Luba (G5, grade index 4) split into Bilbaana and Dhaalaana
        luba = self._gr[4]
        self._bilbaana  : list[int] = luba[:2]   # local [9, 10]
        self._dhaalaana : list[int] = luba[2:]   # local [11, 12]

        self.fes             = 0
        self.iteration        = 0
        self.bilbaana_active  = True

    # ------------------------------------------------------------------
    # Core utilities
    # ------------------------------------------------------------------
    def _clip(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, self.lb, self.ub)

    def _eval(self, x: np.ndarray) -> tuple[float, float]:
        self.fes += 1
        f, cv = self.func(x)
        return float(f), float(cv)

    def _levy(self, alpha: float = 1.5) -> np.ndarray:
        """Mantegna algorithm for Levy flight step."""
        num = _sp_gamma(1 + alpha) * np.sin(np.pi * alpha / 2)
        den = _sp_gamma((1 + alpha) / 2) * alpha * 2 ** ((alpha - 1) / 2)
        sig = (num / den) ** (1 / alpha)
        u   = np.random.randn(self.dim) * sig
        v   = np.abs(np.random.randn(self.dim)) + 1e-12
        return u / v ** (1 / alpha)

    def _diversity(self, Xr: np.ndarray) -> float:
        """Normalised mean pairwise distance within a Roga."""
        diam = np.linalg.norm(self.ub - self.lb) + 1e-12
        n = self.N_PER_ROGA; acc = 0.0; cnt = 0
        for i in range(n):
            for j in range(i + 1, n):
                acc += np.linalg.norm(Xr[i] - Xr[j]); cnt += 1
        return acc / (cnt * diam) if cnt else 1.0

    def _tn(self) -> float:
        """Normalised search progress in [0, 1]."""
        return min(self.fes / self.max_fes, 1.0)

    def _best_idx(self, Fr: np.ndarray, CVr: np.ndarray,
                  idxs: list[int]) -> int:
        return deb_best_index(Fr, CVr, idxs)

    # ------------------------------------------------------------------
    # Grade operators  G1 - G9
    # ------------------------------------------------------------------

    def _G1_qadaaddo(self, x: np.ndarray, centroid: np.ndarray) -> np.ndarray:
        """Qadaaddo (G1) -- Broad Global Search."""
        tn    = self._tn()
        k_exp = (1 - tn) ** 2
        step  = self._levy(1.5)
        u     = 2 * np.random.rand(self.dim) - 1
        x_new = (centroid
                 + k_exp * 0.10 * step * (self.ub - self.lb)
                 + k_exp * 0.06 * np.sqrt(self.dim) * (self.ub - self.lb) * u)
        return self._clip(x_new)

    def _G2_siida(self, x: np.ndarray, x_best_roga: np.ndarray) -> np.ndarray:
        """Siida (G2) -- Territorial Hardening."""
        tn    = self._tn()
        F_    = 0.40 + 0.40 * tn
        r1    = np.random.rand()
        noise = F_ * (1 - tn) * 0.08 * np.random.randn(self.dim) * (self.ub - self.lb)
        return self._clip(x + r1 * (x_best_roga - x) + noise)

    def _G3_lumaasa(self, x: np.ndarray) -> np.ndarray:
        """Lumaasa (G3) -- Silent Preparation."""
        gam   = 0.10 * (1 - self._tn())
        u     = 2 * np.random.rand(self.dim) - 1
        return self._clip(x + gam * (self.ub - self.lb) * u)

    def _G4_raaba(self, x: np.ndarray, f_x: float, cv_x: float, li: int,
                  x_authority: np.ndarray, Xr: np.ndarray, Fr: np.ndarray,
                  CVr: np.ndarray, songo: SongoArchive
                  ) -> tuple[np.ndarray, float, float]:
        """
        Raaba (G4) -- Active Struggle: 3-strategy candidate competition.
          Sanachcha : current-to-p-best differential attack
          Mosiisa   : Songo-guided institutional knowledge
          Ya'a      : authority-centred coordination (own-Roga best,
                      or Abba Gada if this Roga has escalated -- see
                      x_authority, set by the caller)
        Three trial vectors are evaluated; the Deb-best is accepted.
        Uses up to 3 extra FEs per agent per call.

        BUGFIX (this revision): the 6 donor indices for the differential
        vectors are now drawn only from agents OTHER than the acting
        agent `li` -- previously `li` itself could be selected, silently
        reducing mutation diversity.
        """
        N   = self.N_PER_ROGA
        p   = max(2, int(0.20 * N))
        Fv  = float(np.clip(0.5 + 0.1 * np.random.randn(), 0.1, 0.9))
        CR  = 0.70
        xs  = songo.get_random()

        donor_pool = [j for j in range(N) if j != li]
        idxs = np.random.choice(donor_pool, 6, replace=False)
        r_   = [Xr[i] for i in idxs]

        # p-best (top 20 %, Deb-ranked)
        ranked = deb_sort_indices(Fr, CVr, list(range(N)))
        pbst   = Xr[ranked[np.random.randint(p)]]

        v1 = x + Fv * (pbst - x) + Fv * (r_[0] - r_[1])
        v2 = x + Fv * (xs   - x) + Fv * (r_[2] - r_[3])
        v3 = (x_authority + Fv * (r_[4] - r_[5])
              + 0.05 * np.random.randn(self.dim) * (self.ub - self.lb))

        best_x, best_f, best_cv = x.copy(), f_x, cv_x   # baseline (no extra FE)
        for v in [v1, v2, v3]:
            if self.fes >= self.max_fes:
                break
            mask = np.random.rand(self.dim) < CR
            mask[np.random.randint(self.dim)] = True
            trial = self._clip(np.where(mask, v, x))
            ft, cvt = self._eval(trial)
            if deb_worse(best_f, best_cv, ft, cvt):
                best_f, best_cv, best_x = ft, cvt, trial.copy()
        return best_x, best_f, best_cv

    def _G5_luba(self, x: np.ndarray, x_authority: np.ndarray,
                 bil_best: np.ndarray, dha_best: np.ndarray) -> np.ndarray:
        """Luba (G5) -- Governing Grade: Bilbaana / Dhaalaana Dual Lineage.
        `x_authority` is this Roga's own best unless the Roga has
        escalated to Abba Gada (see caller)."""
        r1, r2, r3 = np.random.rand(), np.random.rand(), np.random.rand()
        act   = bil_best if self.bilbaana_active else dha_best
        inact = dha_best if self.bilbaana_active else bil_best
        x_new = (x
                 + r1 * (x_authority - x)
                 + r2 * (act          - x)
                 + r3 * 0.15 * (inact - act))
        return self._clip(x_new)

    def _G6_yuuba(self, x: np.ndarray, x_luba_best: np.ndarray,
                  x_authority: np.ndarray) -> np.ndarray:
        """Yuuba (G6) -- Senior Apprenticeship.
        BUGFIX (this revision): `x_luba_best` is now passed in freshly
        recomputed AFTER the G5 sub-loop has run this same iteration
        (previously it reflected the previous iteration's Luba best)."""
        tn    = self._tn()
        x_int = (x
                 + np.random.rand() * (x_luba_best  - x)
                 + np.random.rand() * (x_authority   - x))
        perturb = ((1 - tn) ** 2 * 0.05
                   * np.random.randn(self.dim) * (self.ub - self.lb))
        return self._clip(x_int + perturb)

    def _G7_guduro(self, x: np.ndarray) -> np.ndarray:
        """Guduro (G7) -- Non-uniform mutation, decreasing amplitude."""
        tn  = self._tn()
        tau = max(1 - tn, 1e-6)
        b   = 2.0
        xn  = x.copy()
        for j in range(self.dim):
            rj = np.random.rand()
            if rj < 0.5:
                delta = (self.ub[j] - x[j]) * (1 - np.random.rand() ** (tau ** b))
                xn[j] = x[j] + delta
            else:
                delta = (x[j] - self.lb[j]) * (1 - np.random.rand() ** (tau ** b))
                xn[j] = x[j] - delta
        return self._clip(xn)

    def _G8_qulullo(self, x: np.ndarray, x_g_best: np.ndarray) -> np.ndarray:
        """Qulullo (G8) -- Opposition-Based Learning."""
        x_obl = self.lb + self.ub - x
        x_new = (x
                 + np.random.rand() * (x_g_best - x)
                 + np.random.rand() * (x_obl    - x))
        return self._clip(x_new)

    def _G9_cowwajje(self, x: np.ndarray, x_ag: np.ndarray, f_ag: float,
                     cv_ag: float, songo: SongoArchive,
                     Xr: np.ndarray, Fr: np.ndarray, CVr: np.ndarray
                     ) -> np.ndarray:
        """Cowwajje (G9) -- Return to Childhood: Periodic Renewal."""
        if not self.disable_songo:
            songo.update(x_ag, f_ag, cv_ag)

        n_abs = max(1, int(0.20 * self.N_PER_ROGA))
        worst_order = deb_sort_indices(Fr, CVr, list(range(self.N_PER_ROGA)))
        for wi in worst_order[-n_abs:]:
            g_wi = next(g for g, idxs in enumerate(self._gr) if wi in idxs)
            if g_wi == 4:
                continue
            if self.fes >= self.max_fes:
                break
            if not songo.empty and np.random.rand() < 0.6:
                Xr[wi] = self._clip(
                    songo.get_diverse(Xr[wi])
                    + 0.10 * np.random.randn(self.dim) * (self.ub - self.lb))
            else:
                Xr[wi] = self.lb + np.random.rand(self.dim) * (self.ub - self.lb)
            Fr[wi], CVr[wi] = self._eval(Xr[wi])

        if not songo.empty and np.random.rand() < 0.5:
            x_new = self._clip(
                songo.get_random()
                + 0.25 * np.random.randn(self.dim) * (self.ub - self.lb))
        else:
            x_new = self.lb + np.random.rand(self.dim) * (self.ub - self.lb)
        return x_new

    # ------------------------------------------------------------------
    # Special mechanisms
    # ------------------------------------------------------------------

    def _faaci_e(self, Xr: np.ndarray, Fr: np.ndarray, CVr: np.ndarray,
                 songo: SongoArchive) -> None:
        """Faaci'e -- diversity-triggered purification."""
        if self._diversity(Xr) < self.D_min:
            n_conf = max(2, int(0.30 * self.N_PER_ROGA))
            worst_order = deb_sort_indices(Fr, CVr, list(range(self.N_PER_ROGA)))
            for wi in worst_order[-n_conf:]:
                if self.fes >= self.max_fes:
                    break
                if not songo.empty and np.random.rand() < 0.6:
                    Xr[wi] = self._clip(
                        songo.get_random()
                        + 0.15 * np.random.randn(self.dim) * (self.ub - self.lb))
                else:
                    Xr[wi] = self.lb + np.random.rand(self.dim) * (self.ub - self.lb)
                Fr[wi], CVr[wi] = self._eval(Xr[wi])

    def _gondooro(self, Xr: np.ndarray, Fr: np.ndarray, CVr: np.ndarray,
                  stag: np.ndarray, songo: SongoArchive) -> None:
        """Gondooro -- individual stagnation escape."""
        for i in range(self.N_PER_ROGA):
            if stag[i] >= self.T_stag:
                if self.fes >= self.max_fes:
                    break
                if not songo.empty:
                    Xr[i] = self._clip(
                        songo.get_diverse(Xr[i])
                        + 0.20 * np.random.randn(self.dim) * (self.ub - self.lb))
                else:
                    Xr[i] = self.lb + np.random.rand(self.dim) * (self.ub - self.lb)
                Fr[i], CVr[i] = self._eval(Xr[i])
                stag[i] = 0

    def _murra(self, X: np.ndarray, F: np.ndarray, CV: np.ndarray) -> None:
        """Murra -- cyclic cross-Roga migration (Deb-aware)."""
        bidx = [self._best_idx(F[k], CV[k], list(range(self.N_PER_ROGA)))
                for k in range(self.N_ROGA)]
        bx = [X[k, bidx[k]].copy() for k in range(self.N_ROGA)]
        bf = [float(F[k, bidx[k]]) for k in range(self.N_ROGA)]
        bcv = [float(CV[k, bidx[k]]) for k in range(self.N_ROGA)]
        for k in range(self.N_ROGA):
            dest = (k + 1) % self.N_ROGA
            worst_order = deb_sort_indices(F[dest], CV[dest],
                                            list(range(self.N_PER_ROGA)))
            wi = worst_order[-1]
            if deb_worse(F[dest, wi], CV[dest, wi], bf[k], bcv[k]) and self.fes < self.max_fes:
                X[dest, wi] = bx[k]
                F[dest, wi], CV[dest, wi] = self._eval(X[dest, wi])

    def _ya_a(self, X: np.ndarray, F: np.ndarray, CV: np.ndarray
              ) -> tuple[np.ndarray, float, float]:
        """Ya'a Assembly -- Abba Gada election from Luba (G5), Deb-aware."""
        best_x: np.ndarray | None = None
        best_f, best_cv = np.inf, np.inf
        for k in range(self.N_ROGA):
            for i in self._gr[4]:       # G5 Luba
                if best_x is None or deb_worse(best_f, best_cv, F[k, i], CV[k, i]):
                    best_f, best_cv, best_x = float(F[k, i]), float(CV[k, i]), X[k, i].copy()
        if best_x is None:
            idx = np.unravel_index(int(np.argmin(F)), F.shape)
            best_x, best_f, best_cv = X[idx].copy(), float(F[idx]), float(CV[idx])
        return best_x, best_f, best_cv

    # ------------------------------------------------------------------
    # Main optimisation loop
    # ------------------------------------------------------------------

    def optimize(self, seed: int | None = None
                 ) -> tuple[np.ndarray, float, float, list[float]]:
        """
        Run BalOA optimisation.

        Returns
        -------
        x_best      : ndarray  -- best solution vector found
        f_best      : float    -- its objective value
        cv_best     : float    -- its constraint violation (0.0 if feasible)
        convergence : list     -- best-so-far objective recorded each
                                   iteration (Deb-best; only meaningful
                                   once cv_best == 0, flagged in practice
                                   by also tracking feasibility -- see
                                   the runner script for feasibility-rate
                                   reporting across independent runs)
        """
        if seed is not None:
            np.random.seed(seed)
        self.fes = 0; self.iteration = 0
        self.bilbaana_active = True

        # ---- Latin Hypercube initialisation per Roga -------------------
        X = np.zeros((self.N_ROGA, self.N_PER_ROGA, self.dim))
        for k in range(self.N_ROGA):
            for d in range(self.dim):
                p = np.random.permutation(self.N_PER_ROGA)
                X[k, :, d] = (self.lb[d]
                               + (p + np.random.rand(self.N_PER_ROGA))
                               / self.N_PER_ROGA
                               * (self.ub[d] - self.lb[d]))

        # ---- Initial fitness evaluation --------------------------------
        F  = np.zeros((self.N_ROGA, self.N_PER_ROGA))
        CV = np.zeros((self.N_ROGA, self.N_PER_ROGA))
        for k in range(self.N_ROGA):
            for i in range(self.N_PER_ROGA):
                F[k, i], CV[k, i] = self._eval(X[k, i])

        # ---- Songo archives (one per Roga) ----------------------------
        songos = [SongoArchive(self.N_S, self.sigma_s, self.dim, self.lb, self.ub)
                  for _ in range(self.N_ROGA)]
        if not self.disable_songo:
            for k in range(self.N_ROGA):
                bi = self._best_idx(F[k], CV[k], list(range(self.N_PER_ROGA)))
                songos[k].update(X[k, bi], F[k, bi], CV[k, bi])

        # ---- Abba Gada, lineage bests, stagnation counters ------------
        x_ag, f_ag, cv_ag = self._ya_a(X, F, CV)

        bil_best = [X[k, self._bilbaana[0]].copy()  for k in range(self.N_ROGA)]
        bil_f    = [float(F[k, self._bilbaana[0]])   for k in range(self.N_ROGA)]
        bil_cv   = [float(CV[k, self._bilbaana[0]])  for k in range(self.N_ROGA)]
        dha_best = [X[k, self._dhaalaana[0]].copy() for k in range(self.N_ROGA)]
        dha_f    = [float(F[k, self._dhaalaana[0]])  for k in range(self.N_ROGA)]
        dha_cv   = [float(CV[k, self._dhaalaana[0]]) for k in range(self.N_ROGA)]

        stag = np.zeros((self.N_ROGA, self.N_PER_ROGA), dtype=int)

        # ---- Per-Roga escalation state (NEW) ---------------------------
        roga_best_f  = [None] * self.N_ROGA
        roga_best_cv = [None] * self.N_ROGA
        roga_stall    = np.zeros(self.N_ROGA, dtype=int)
        roga_escalate = np.zeros(self.N_ROGA, dtype=bool)
        for k in range(self.N_ROGA):
            bi = self._best_idx(F[k], CV[k], list(range(self.N_PER_ROGA)))
            roga_best_f[k]  = float(F[k, bi])
            roga_best_cv[k] = float(CV[k, bi])

        # ---- Global best -----------------------------------------------
        idx0   = np.unravel_index(int(np.argmin(F)), F.shape)  # cheap seed; corrected below
        x_best, f_best, cv_best = X[idx0].copy(), float(F[idx0]), float(CV[idx0])
        for k in range(self.N_ROGA):
            bi = self._best_idx(F[k], CV[k], list(range(self.N_PER_ROGA)))
            if deb_worse(f_best, cv_best, F[k, bi], CV[k, bi]):
                f_best, cv_best, x_best = float(F[k, bi]), float(CV[k, bi]), X[k, bi].copy()
        conv: list[float] = [f_best]

        # ================================================================
        # Main optimisation loop
        # ================================================================
        while self.fes < self.max_fes:
            self.iteration += 1

            for k in range(self.N_ROGA):
                Xr = X[k]; Fr = F[k]; CVr = CV[k]; sg = songos[k]

                centroid      = Xr.mean(axis=0)
                x_br_idx      = self._best_idx(Fr, CVr, list(range(self.N_PER_ROGA)))
                x_br          = Xr[x_br_idx].copy()
                x_guduro_best = Xr[self._best_idx(Fr, CVr, self._gr[6])].copy()

                # This Roga's authority target: its own best, unless it
                # has escalated to Abba Gada after sustained failure.
                x_authority = x_ag if (roga_escalate[k] and not self.disable_escalation) else x_br

                for g in range(9):
                    # Luba (G5) best is recomputed fresh right before G6
                    # consumes it, so Yuuba trains on THIS iteration's
                    # Luba result, not last iteration's.
                    if g == 5:
                        x_luba_best = Xr[self._best_idx(Fr, CVr, self._gr[4])].copy()

                    for li in self._gr[g]:
                        if self.fes >= self.max_fes:
                            break
                        xi = Xr[li].copy(); fi = Fr[li]; cvi = CVr[li]

                        if g == 0:   # G1 Qadaaddo
                            xn = self._G1_qadaaddo(xi, centroid)
                            fn, cvn = self._eval(xn)

                        elif g == 1: # G2 Siida
                            xn = self._G2_siida(xi, x_br)
                            fn, cvn = self._eval(xn)

                        elif g == 2: # G3 Lumaasa
                            xn = self._G3_lumaasa(xi)
                            fn, cvn = self._eval(xn)

                        elif g == 3: # G4 Raaba (up to 3 extra FEs)
                            if self.fes + 3 <= self.max_fes:
                                xn, fn, cvn = self._G4_raaba(
                                    xi, fi, cvi, li, x_authority, Xr, Fr, CVr, sg)
                            else:
                                xn = self._G3_lumaasa(xi); fn, cvn = self._eval(xn)

                        elif g == 4: # G5 Luba
                            xn = self._G5_luba(xi, x_authority, bil_best[k], dha_best[k])
                            fn, cvn = self._eval(xn)
                            if li in self._bilbaana and deb_worse(bil_f[k], bil_cv[k], fn, cvn):
                                bil_f[k], bil_cv[k], bil_best[k] = fn, cvn, xn.copy()
                            elif li in self._dhaalaana and deb_worse(dha_f[k], dha_cv[k], fn, cvn):
                                dha_f[k], dha_cv[k], dha_best[k] = fn, cvn, xn.copy()

                        elif g == 5: # G6 Yuuba
                            xn = self._G6_yuuba(xi, x_luba_best, x_authority)
                            fn, cvn = self._eval(xn)

                        elif g == 6: # G7 Guduro
                            xn = self._G7_guduro(xi)
                            fn, cvn = self._eval(xn)

                        elif g == 7: # G8 Qulullo
                            xn = self._G8_qulullo(xi, x_guduro_best)
                            fn, cvn = self._eval(xn)

                        elif g == 8: # G9 Cowwajje
                            if (self.iteration % self.T_C == 0
                                    and self.fes + 1 <= self.max_fes
                                    and not self.disable_cowwajje):
                                xn = self._G9_cowwajje(xi, x_ag, f_ag, cv_ag, sg, Xr, Fr, CVr)
                                fn, cvn = self._eval(xn)
                            else:
                                xn = self._G8_qulullo(xi, x_br)
                                fn, cvn = self._eval(xn)

                        # ---------- greedy acceptance (Deb-aware) -------
                        if deb_worse(fi, cvi, fn, cvn):
                            Xr[li] = xn; Fr[li] = fn; CVr[li] = cvn
                            stag[k, li] = 0
                            if not self.disable_songo:
                                sg.update(xn, fn, cvn)
                        else:
                            stag[k, li] += 1

                if not self.disable_faacie:
                    self._faaci_e(Xr, Fr, CVr, sg)
                if not self.disable_gondooro:
                    self._gondooro(Xr, Fr, CVr, stag[k], sg)

                # ---- Update this Roga's own escalation state ----------
                new_bi = self._best_idx(Fr, CVr, list(range(self.N_PER_ROGA)))
                new_bf, new_bcv = float(Fr[new_bi]), float(CVr[new_bi])
                if deb_worse(roga_best_f[k], roga_best_cv[k], new_bf, new_bcv):
                    roga_best_f[k], roga_best_cv[k] = new_bf, new_bcv
                    roga_stall[k] = 0
                    roga_escalate[k] = False
                else:
                    roga_stall[k] += 1
                    if roga_stall[k] >= self.T_escalate:
                        roga_escalate[k] = True

            # ---- Periodic cross-Roga mechanisms -----------------------

            if self.iteration % self.T_M == 0 and not self.disable_murra:
                self._murra(X, F, CV)

            if self.iteration % self.T_G == 0:
                x_ag, f_ag, cv_ag = self._ya_a(X, F, CV)
                # A fresh Ya'a assembly resets escalation for all Rogas:
                # the newly (re-)elected Abba Gada gets a clean mandate.
                roga_stall[:]    = 0
                roga_escalate[:] = False

            if self.iteration % self.T_L == 0:
                if not self.disable_bilbaana_swap:
                    self.bilbaana_active = not self.bilbaana_active
                for k in range(self.N_ROGA):
                    for li in self._bilbaana:
                        if deb_worse(bil_f[k], bil_cv[k], F[k, li], CV[k, li]):
                            bil_f[k], bil_cv[k] = float(F[k, li]), float(CV[k, li])
                            bil_best[k] = X[k, li].copy()
                    for li in self._dhaalaana:
                        if deb_worse(dha_f[k], dha_cv[k], F[k, li], CV[k, li]):
                            dha_f[k], dha_cv[k] = float(F[k, li]), float(CV[k, li])
                            dha_best[k] = X[k, li].copy()

            if self.iteration % self.T_B == 0 and not self.disable_baallee:
                for k in range(self.N_ROGA):
                    sg = songos[k]
                    for li in self._gr[0]:
                        if self.fes >= self.max_fes:
                            break
                        if not sg.empty and np.random.rand() < 0.40:
                            X[k, li] = self._clip(
                                sg.get_random()
                                + 0.30 * np.random.randn(self.dim) * (self.ub - self.lb))
                        else:
                            X[k, li] = (self.lb
                                        + np.random.rand(self.dim) * (self.ub - self.lb))
                        F[k, li], CV[k, li] = self._eval(X[k, li])
                        stag[k, li] = 0

            # ---- Update global best (Deb-aware) ------------------------
            for k in range(self.N_ROGA):
                bi = self._best_idx(F[k], CV[k], list(range(self.N_PER_ROGA)))
                if deb_worse(f_best, cv_best, F[k, bi], CV[k, bi]):
                    f_best, cv_best, x_best = float(F[k, bi]), float(CV[k, bi]), X[k, bi].copy()
                    if not self.disable_songo:
                        songos[k].update(x_best, f_best, cv_best)

            conv.append(f_best)

        return x_best, f_best, cv_best, conv
