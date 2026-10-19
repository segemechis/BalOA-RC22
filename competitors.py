"""
competitors.py  –  Clean, standardised competitor implementations for ELD study.

Interface (identical for every algorithm):
    run_ALGO(func, lb, ub, dim, max_fes, pop_size=30, seed=None)
    -> (best_cost: float,
        best_position: ndarray shape (dim,),
        convergence_curve: list[float])   # one entry per iteration

    func     : callable(x: ndarray(dim,)) -> float   [minimise]
    lb, ub   : scalar or 1-D array of length dim
    dim      : int
    max_fes  : int   maximum function evaluations
    pop_size : int
    seed     : int | None
"""

import numpy as np
import math
from scipy.optimize import differential_evolution as _scipy_de

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _bounds(lb, ub, dim):
    lb = np.full(dim, lb, dtype=float) if np.isscalar(lb) else np.asarray(lb, dtype=float)
    ub = np.full(dim, ub, dtype=float) if np.isscalar(ub) else np.asarray(ub, dtype=float)
    return lb, ub

def _levy(dim, rng):
    beta = 1.5
    num   = math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
    den   = math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    sigma = (num / den) ** (1.0 / beta)
    u = rng.standard_normal(dim) * sigma
    v = rng.standard_normal(dim)
    return u / (np.abs(v) ** (1.0 / beta))


# ===========================================================================
# 1.  GA  –  Real-valued Genetic Algorithm (SBX crossover + poly mutation)
# ===========================================================================

def _sbx(p1, p2, lb, ub, eta, rng):
    u = rng.random(len(p1))
    beta = np.where(u <= 0.5,
                    (2.0 * u) ** (1.0 / (eta + 1)),
                    (0.5 / (1.0 - u)) ** (1.0 / (eta + 1)))
    c1 = np.clip(0.5 * ((1 + beta) * p1 + (1 - beta) * p2), lb, ub)
    c2 = np.clip(0.5 * ((1 - beta) * p1 + (1 + beta) * p2), lb, ub)
    return c1, c2

def _poly_mut(x, lb, ub, pm, eta, rng):
    x = x.copy()
    for k in range(len(x)):
        if rng.random() < pm:
            u = rng.random()
            d = ((2.0 * u) ** (1.0 / (eta + 1)) - 1.0 if u < 0.5
                 else 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta + 1)))
            x[k] = np.clip(x[k] + d * (ub[k] - lb[k]), lb[k], ub[k])
    return x

def run_GA(func, lb, ub, dim, max_fes, pop_size=30, seed=None):
    rng = np.random.default_rng(seed)
    lb, ub = _bounds(lb, ub, dim)
    eta_c, eta_m, pc, pm = 20, 20, 0.9, 1.0 / dim

    pop = rng.uniform(lb, ub, (pop_size, dim))
    fit = np.array([func(pop[i]) for i in range(pop_size)])
    fes = pop_size

    best_idx = int(np.argmin(fit))
    best_pos, best_cost = pop[best_idx].copy(), float(fit[best_idx])
    conv = [best_cost]

    while fes < max_fes:
        offspring = []
        while len(offspring) < pop_size:
            ia = rng.integers(pop_size, size=3)
            ib = rng.integers(pop_size, size=3)
            pa = pop[ia[np.argmin(fit[ia])]]
            pb = pop[ib[np.argmin(fit[ib])]]
            c1, c2 = (_sbx(pa, pb, lb, ub, eta_c, rng) if rng.random() < pc
                      else (pa.copy(), pb.copy()))
            offspring.extend([_poly_mut(c1, lb, ub, pm, eta_m, rng),
                               _poly_mut(c2, lb, ub, pm, eta_m, rng)])
        offspring = np.array(offspring[:pop_size])
        new_fit = np.empty(pop_size)
        done = False
        for i in range(pop_size):
            new_fit[i] = func(offspring[i])
            fes += 1
            if fes >= max_fes:
                done = True
                break
        # elitism: keep known best
        worst = int(np.argmax(new_fit))
        if best_cost < new_fit[worst]:
            offspring[worst] = best_pos.copy()
            new_fit[worst] = best_cost
        pop, fit = offspring, new_fit
        cur = int(np.argmin(fit))
        if fit[cur] < best_cost:
            best_cost, best_pos = float(fit[cur]), pop[cur].copy()
        conv.append(best_cost)
        if done:
            break

    return best_cost, best_pos, conv


# ===========================================================================
# 2.  PSO  –  Standard Particle Swarm Optimisation (inertia weight)
# ===========================================================================

def run_PSO(func, lb, ub, dim, max_fes, pop_size=30, seed=None):
    rng = np.random.default_rng(seed)
    lb, ub = _bounds(lb, ub, dim)
    w_max, w_min, c1, c2 = 0.9, 0.4, 2.0, 2.0
    max_iter = max_fes // pop_size
    v_max = 0.2 * (ub - lb)

    pos = rng.uniform(lb, ub, (pop_size, dim))
    vel = rng.uniform(-v_max, v_max, (pop_size, dim))
    fit = np.array([func(pos[i]) for i in range(pop_size)])
    fes = pop_size

    pbest_pos = pos.copy()
    pbest_fit = fit.copy()
    gbest_idx = int(np.argmin(fit))
    gbest_pos = pos[gbest_idx].copy()
    gbest_cost = float(fit[gbest_idx])
    conv = [gbest_cost]

    it = 0
    while fes < max_fes:
        it += 1
        w = w_max - (w_max - w_min) * it / max_iter
        r1 = rng.random((pop_size, dim))
        r2 = rng.random((pop_size, dim))
        vel = (w * vel
               + c1 * r1 * (pbest_pos - pos)
               + c2 * r2 * (gbest_pos - pos))
        vel = np.clip(vel, -v_max, v_max)
        pos = np.clip(pos + vel, lb, ub)

        done = False
        for i in range(pop_size):
            f = func(pos[i])
            fes += 1
            if f < pbest_fit[i]:
                pbest_fit[i] = f
                pbest_pos[i] = pos[i].copy()
                if f < gbest_cost:
                    gbest_cost = f
                    gbest_pos = pos[i].copy()
            if fes >= max_fes:
                done = True
                break
        conv.append(gbest_cost)
        if done:
            break

    return gbest_cost, gbest_pos, conv


# ===========================================================================
# 3.  DE  –  Differential Evolution  (via scipy)
# ===========================================================================

def run_DE(func, lb, ub, dim, max_fes, pop_size=30, seed=None):
    lb, ub = _bounds(lb, ub, dim)
    best_so_far = [np.inf]
    fes_count   = [0]
    conv = []

    def _tracked(x):
        fes_count[0] += 1
        v = func(x)
        if v < best_so_far[0]:
            best_so_far[0] = v
        return v

    def _cb(xk, convergence):
        conv.append(best_so_far[0])
        # BUGFIX: scipy's differential_evolution has no native
        # max-function-evaluations knob, so without this the run could
        # silently exceed max_fes by a full extra generation (or more).
        # Stopping the callback once the tracked FE count reaches the
        # budget bounds the overshoot to at most one generation
        # (actual_pop evaluations), matching every other algorithm here
        # that checks its FES cap before each evaluation.
        return fes_count[0] >= max_fes

    popsize_mult = max(1, pop_size // dim)
    actual_pop   = popsize_mult * dim
    bounds = list(zip(lb, ub))

    res = _scipy_de(
        _tracked, bounds,
        strategy='best1bin',
        maxiter=max(1, max_fes // actual_pop),
        popsize=popsize_mult,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=seed,
        callback=_cb,
        tol=0,
        init='latinhypercube',
        polish=False,
        # BUGFIX: the default polish=True runs an uncounted L-BFGS-B
        # local-search pass on the final result, using extra function
        # evaluations that are never counted against max_fes. That gave
        # DE an unbounded, unfair FES advantage over every other
        # algorithm in this comparison (and L-BFGS-B assumes a smooth
        # gradient, which is a mismatch for a penalized/Deb-rule-wrapped
        # constrained objective with a hard discontinuity at the
        # feasibility boundary). Disabled.
    )
    if not conv:
        conv = [best_so_far[0]]
    return float(best_so_far[0]), np.asarray(res.x).copy(), conv


# ===========================================================================
# 4.  GWO  –  Grey Wolf Optimiser (Mirjalili et al., 2014)
# ===========================================================================

def run_GWO(func, lb, ub, dim, max_fes, pop_size=30, seed=None):
    rng = np.random.default_rng(seed)
    lb, ub = _bounds(lb, ub, dim)
    max_iter = max_fes // pop_size

    pos = rng.uniform(lb, ub, (pop_size, dim))
    fit = np.array([func(pos[i]) for i in range(pop_size)])
    fes = pop_size

    sidx = np.argsort(fit)
    alpha_pos, alpha_score = pos[sidx[0]].copy(), float(fit[sidx[0]])
    beta_pos,  beta_score  = pos[sidx[1]].copy(), float(fit[sidx[1]])
    delta_pos, delta_score = pos[sidx[2]].copy(), float(fit[sidx[2]])
    conv = [alpha_score]

    it = 0
    while fes < max_fes:
        it += 1
        a = 2.0 - 2.0 * it / max_iter

        done = False
        for i in range(pop_size):
            r1, r2 = rng.random(dim), rng.random(dim)
            X1 = alpha_pos - (2*a*r1 - a) * np.abs(2*r2*alpha_pos - pos[i])
            r1, r2 = rng.random(dim), rng.random(dim)
            X2 = beta_pos  - (2*a*r1 - a) * np.abs(2*r2*beta_pos  - pos[i])
            r1, r2 = rng.random(dim), rng.random(dim)
            X3 = delta_pos - (2*a*r1 - a) * np.abs(2*r2*delta_pos - pos[i])

            pos[i] = np.clip((X1 + X2 + X3) / 3.0, lb, ub)
            f = func(pos[i]); fes += 1; fit[i] = f

            if f < alpha_score:
                delta_score, delta_pos = beta_score,  beta_pos.copy()
                beta_score,  beta_pos  = alpha_score, alpha_pos.copy()
                alpha_score, alpha_pos = f, pos[i].copy()
            elif f < beta_score:
                delta_score, delta_pos = beta_score, beta_pos.copy()
                beta_score,  beta_pos  = f, pos[i].copy()
            elif f < delta_score:
                delta_score, delta_pos = f, pos[i].copy()

            if fes >= max_fes:
                done = True; break

        conv.append(alpha_score)
        if done:
            break

    return alpha_score, alpha_pos, conv


# ===========================================================================
# 5.  MFO  –  Moth-Flame Optimisation (Mirjalili, 2015)
# ===========================================================================

def run_MFO(func, lb, ub, dim, max_fes, pop_size=30, seed=None):
    rng = np.random.default_rng(seed)
    lb, ub = _bounds(lb, ub, dim)
    max_iter = max_fes // pop_size
    b = 1.0

    moths = rng.uniform(lb, ub, (pop_size, dim))
    fit   = np.array([func(moths[i]) for i in range(pop_size)])
    fes   = pop_size

    best_idx  = int(np.argmin(fit))
    best_pos  = moths[best_idx].copy()
    best_cost = float(fit[best_idx])
    conv = [best_cost]

    it = 0
    while fes < max_fes:
        it += 1
        flame_no = max(1, round(pop_size - it * (pop_size - 1) / max_iter))

        sidx   = np.argsort(fit)
        flames = moths[sidx[:flame_no]].copy()

        done = False
        for i in range(pop_size):
            fi = min(i, flame_no - 1)
            fl = flames[fi]
            t  = rng.uniform(-1, 1, dim)
            D  = np.abs(fl - moths[i])
            moths[i] = np.clip(
                D * np.exp(b * t) * np.cos(2 * np.pi * t) + fl,
                lb, ub)
            f = func(moths[i]); fes += 1; fit[i] = f
            if f < best_cost:
                best_cost = float(f); best_pos = moths[i].copy()
            if fes >= max_fes:
                done = True; break

        conv.append(best_cost)
        if done:
            break

    return best_cost, best_pos, conv


# ===========================================================================
# 6.  WOA  –  Whale Optimisation Algorithm (Mirjalili & Lewis, 2016)
# ===========================================================================

def run_WOA(func, lb, ub, dim, max_fes, pop_size=30, seed=None):
    rng = np.random.default_rng(seed)
    lb, ub = _bounds(lb, ub, dim)
    max_iter = max_fes // pop_size
    b = 1.0

    pos = rng.uniform(lb, ub, (pop_size, dim))
    fit = np.array([func(pos[i]) for i in range(pop_size)])
    fes = pop_size

    best_idx  = int(np.argmin(fit))
    best_pos  = pos[best_idx].copy()
    best_cost = float(fit[best_idx])
    conv = [best_cost]

    it = 0
    while fes < max_fes:
        it += 1
        a  = 2.0 - 2.0 * it / max_iter
        a2 = -1.0 - it / max_iter

        done = False
        for i in range(pop_size):
            A = 2*a*rng.random(dim) - a
            C = 2*rng.random(dim)
            p = rng.random()

            if p < 0.5:
                if np.linalg.norm(A) < 1:
                    D = np.abs(C * best_pos - pos[i])
                    new_x = best_pos - A * D
                else:
                    ri = rng.integers(pop_size)
                    D  = np.abs(C * pos[ri] - pos[i])
                    new_x = pos[ri] - A * D
            else:
                l     = (a2 - 1) * rng.random() + 1
                D     = np.abs(best_pos - pos[i])
                new_x = D * np.exp(b * l) * np.cos(2*np.pi*l) + best_pos

            pos[i] = np.clip(new_x, lb, ub)
            f = func(pos[i]); fes += 1; fit[i] = f
            if f < best_cost:
                best_cost = float(f); best_pos = pos[i].copy()
            if fes >= max_fes:
                done = True; break

        conv.append(best_cost)
        if done:
            break

    return best_cost, best_pos, conv


# ===========================================================================
# 7.  SCA  –  Sine Cosine Algorithm (Mirjalili, 2016)
# ===========================================================================

def run_SCA(func, lb, ub, dim, max_fes, pop_size=30, seed=None):
    rng = np.random.default_rng(seed)
    lb, ub = _bounds(lb, ub, dim)
    max_iter = max_fes // pop_size
    a = 2.0

    pos = rng.uniform(lb, ub, (pop_size, dim))
    fit = np.array([func(pos[i]) for i in range(pop_size)])
    fes = pop_size

    best_idx  = int(np.argmin(fit))
    best_pos  = pos[best_idx].copy()
    best_cost = float(fit[best_idx])
    conv = [best_cost]

    it = 0
    while fes < max_fes:
        it += 1
        r1 = a - a * it / max_iter   # control parameter → 0
        r2 = 2 * np.pi * rng.random((pop_size, dim))
        r3 = 2 * rng.random((pop_size, dim))
        r4 = rng.random((pop_size, dim))

        mask = r4 < 0.5
        diff = np.abs(r3 * best_pos - pos)
        pos = np.where(mask,
                       pos + r1 * np.sin(r2) * diff,
                       pos + r1 * np.cos(r2) * diff)
        pos = np.clip(pos, lb, ub)

        done = False
        for i in range(pop_size):
            f = func(pos[i]); fes += 1
            if f < best_cost:
                best_cost = float(f); best_pos = pos[i].copy()
            if fes >= max_fes:
                done = True; break

        conv.append(best_cost)
        if done:
            break

    return best_cost, best_pos, conv


# ===========================================================================
# 8.  MVO  –  Multi-Verse Optimiser (Mirjalili et al., 2016)
# ===========================================================================

def run_MVO(func, lb, ub, dim, max_fes, pop_size=30, seed=None):
    rng = np.random.default_rng(seed)
    lb, ub = _bounds(lb, ub, dim)
    max_iter = max_fes // pop_size
    WEP_min, WEP_max = 0.2, 1.0

    pos = rng.uniform(lb, ub, (pop_size, dim))
    fit = np.array([func(pos[i]) for i in range(pop_size)])
    fes = pop_size

    best_idx  = int(np.argmin(fit))
    best_pos  = pos[best_idx].copy()
    best_cost = float(fit[best_idx])
    conv = [best_cost]

    it = 0
    while fes < max_fes:
        it += 1
        WEP = WEP_min + (WEP_max - WEP_min) * it / max_iter
        TDR = 1.0 - (it ** (1.0 / 6)) / (max_iter ** (1.0 / 6))

        # Normalised inflation rates (inverse of cost, proportional to fitness)
        min_f = fit.min()
        raw   = 1.0 / (1.0 + fit - min_f)
        cumul = np.cumsum(raw / raw.sum())

        done = False
        for i in range(pop_size):
            new_x = pos[i].copy()
            for j in range(dim):
                r1 = rng.random()
                # White-hole roulette selection
                if r1 < cumul[i]:
                    r = rng.random()
                    wh = min(int(np.searchsorted(cumul, r)), pop_size - 1)
                    new_x[j] = pos[wh, j]
                # Wormhole travel toward best universe
                r2 = rng.random()
                if r2 < WEP:
                    r3, r4 = rng.random(), rng.random()
                    span = ub[j] - lb[j]
                    if r3 <= 0.5:
                        new_x[j] = best_pos[j] + TDR * (span * r4 + lb[j])
                    else:
                        new_x[j] = best_pos[j] - TDR * (span * r4 + lb[j])
            pos[i] = np.clip(new_x, lb, ub)
            f = func(pos[i]); fes += 1; fit[i] = f
            if f < best_cost:
                best_cost = float(f); best_pos = pos[i].copy()
            if fes >= max_fes:
                done = True; break

        conv.append(best_cost)
        if done:
            break

    return best_cost, best_pos, conv


# ===========================================================================
# 9.  SSA  –  Salp Swarm Algorithm (Mirjalili et al., 2017)
# ===========================================================================

def run_SSA(func, lb, ub, dim, max_fes, pop_size=30, seed=None):
    rng = np.random.default_rng(seed)
    lb, ub = _bounds(lb, ub, dim)
    max_iter = max_fes // pop_size

    pos = rng.uniform(lb, ub, (pop_size, dim))
    fit = np.array([func(pos[i]) for i in range(pop_size)])
    fes = pop_size

    best_idx  = int(np.argmin(fit))
    food_pos  = pos[best_idx].copy()
    food_cost = float(fit[best_idx])
    conv = [food_cost]

    it = 0
    while fes < max_fes:
        it += 1
        c1 = 2.0 * np.exp(-(4.0 * it / max_iter) ** 2)

        done = False
        for i in range(pop_size):
            if i == 0:
                c2 = rng.random(dim)
                c3 = rng.random(dim)
                sign = np.where(c3 >= 0.5, 1.0, -1.0)
                pos[i] = np.clip(
                    food_pos + sign * c1 * ((ub - lb) * c2 + lb),
                    lb, ub)
            else:
                pos[i] = np.clip(0.5 * (pos[i] + pos[i - 1]), lb, ub)

            f = func(pos[i]); fes += 1; fit[i] = f
            if f < food_cost:
                food_cost = float(f); food_pos = pos[i].copy()
            if fes >= max_fes:
                done = True; break

        conv.append(food_cost)
        if done:
            break

    return food_cost, food_pos, conv


# ===========================================================================
# 10. HHO  –  Harris Hawks Optimisation (Heidari et al., 2019)
#     (Strategies 3 & 4 use single Y update; Z dive counted separately)
# ===========================================================================

def run_HHO(func, lb, ub, dim, max_fes, pop_size=30, seed=None):
    rng = np.random.default_rng(seed)
    lb, ub = _bounds(lb, ub, dim)
    max_iter = max_fes // pop_size

    pos = rng.uniform(lb, ub, (pop_size, dim))
    fit = np.array([func(pos[i]) for i in range(pop_size)])
    fes = pop_size

    best_idx     = int(np.argmin(fit))
    rabbit_pos   = pos[best_idx].copy()
    rabbit_cost  = float(fit[best_idx])
    conv = [rabbit_cost]

    it = 0
    while fes < max_fes:
        it += 1
        E1 = 2.0 * (1.0 - it / max_iter)

        done = False
        for i in range(pop_size):
            E0 = 2.0 * rng.random() - 1.0
            E  = E1 * E0
            J  = 2.0 * (1.0 - rng.random())

            if abs(E) >= 1:
                # Exploration
                ri = rng.integers(pop_size)
                q  = rng.random()
                if q >= 0.5:
                    new_x = (pos[ri]
                             - rng.random() * np.abs(pos[ri] - 2*rng.random()*pos[i]))
                else:
                    mean_p = np.mean(pos, axis=0)
                    new_x  = (rabbit_pos - mean_p
                              - rng.random() * (lb + rng.random() * (ub - lb)))
                new_x = np.clip(new_x, lb, ub)
                f = func(new_x); fes += 1
                if f < fit[i]:
                    pos[i] = new_x; fit[i] = f
                if f < rabbit_cost:
                    rabbit_cost = float(f); rabbit_pos = new_x.copy()

            else:
                r = rng.random()
                if r >= 0.5 and abs(E) >= 0.5:
                    new_x = np.clip(
                        rabbit_pos - E * np.abs(J * rabbit_pos - pos[i]), lb, ub)
                elif r >= 0.5 and abs(E) < 0.5:
                    new_x = np.clip(
                        rabbit_pos - E * np.abs(rabbit_pos - pos[i]), lb, ub)
                elif r < 0.5 and abs(E) >= 0.5:
                    Y = np.clip(rabbit_pos - E * np.abs(J * rabbit_pos - pos[i]), lb, ub)
                    fY = func(Y); fes += 1
                    if fes >= max_fes:
                        if fY < fit[i]:
                            pos[i] = Y; fit[i] = fY
                        if fY < rabbit_cost:
                            rabbit_cost = float(fY); rabbit_pos = Y.copy()
                        done = True; break
                    Z = np.clip(Y + rng.standard_normal(dim) * _levy(dim, rng), lb, ub)
                    fZ = func(Z); fes += 1
                    best_f = min(fY, fZ, fit[i])
                    if best_f == fY:
                        pos[i] = Y; fit[i] = fY
                    elif best_f == fZ:
                        pos[i] = Z; fit[i] = fZ
                    if fit[i] < rabbit_cost:
                        rabbit_cost = float(fit[i]); rabbit_pos = pos[i].copy()
                    if fes >= max_fes:
                        done = True; break
                    continue
                else:
                    mean_p = np.mean(pos, axis=0)
                    Y = np.clip(rabbit_pos - E * np.abs(J * rabbit_pos - mean_p), lb, ub)
                    fY = func(Y); fes += 1
                    if fes >= max_fes:
                        if fY < fit[i]:
                            pos[i] = Y; fit[i] = fY
                        if fY < rabbit_cost:
                            rabbit_cost = float(fY); rabbit_pos = Y.copy()
                        done = True; break
                    Z = np.clip(Y + rng.standard_normal(dim) * _levy(dim, rng), lb, ub)
                    fZ = func(Z); fes += 1
                    best_f = min(fY, fZ, fit[i])
                    if best_f == fY:
                        pos[i] = Y; fit[i] = fY
                    elif best_f == fZ:
                        pos[i] = Z; fit[i] = fZ
                    if fit[i] < rabbit_cost:
                        rabbit_cost = float(fit[i]); rabbit_pos = pos[i].copy()
                    if fes >= max_fes:
                        done = True; break
                    continue

                f = func(new_x); fes += 1
                if f < fit[i]:
                    pos[i] = new_x; fit[i] = f
                if f < rabbit_cost:
                    rabbit_cost = float(f); rabbit_pos = new_x.copy()

            if fes >= max_fes:
                done = True; break

        conv.append(rabbit_cost)
        if done:
            break

    return rabbit_cost, rabbit_pos, conv


# ===========================================================================
# 11. DBO  –  Dung Beetle Optimisation (Xue & Shen, 2023)
#     Generalised proportional groups (fixes hardcoded-index bug)
# ===========================================================================

def run_DBO(func, lb, ub, dim, max_fes, pop_size=30, seed=None):
    rng = np.random.default_rng(seed)
    lb, ub = _bounds(lb, ub, dim)
    max_iter = max_fes // pop_size

    pos = rng.uniform(lb, ub, (pop_size, dim))
    fit = np.array([func(pos[i]) for i in range(pop_size)])
    fes = pop_size

    sidx = np.argsort(fit)
    pos, fit = pos[sidx], fit[sidx]
    best_pos  = pos[0].copy()
    best_cost = float(fit[0])
    worst_pos = pos[-1].copy()
    conv = [best_cost]

    # Group sizes (proportional to pop_size)
    n_roll  = max(1, int(0.3  * pop_size))   # ball-rollers
    n_brood = max(1, int(0.15 * pop_size))   # brooders (tail)
    # foragers = rest
    eta = 3.0

    it = 0
    while fes < max_fes:
        it += 1
        R  = 1.0 - it / max_iter              # boundary shrink ratio
        Rb = np.exp(-eta * it / max_iter)     # obstacle probability

        done = False

        # -- Group 1: ball-rollers (indices 0..n_roll-1) --
        for i in range(n_roll):
            if fes >= max_fes: done = True; break
            # BUGFIX: the i==0 case previously multiplied pos[i] by
            # itself (`pos[i] if i == 0 else ...`) instead of using a
            # displacement vector, which is dimensionally wrong and lets
            # agent 0's step scale explosively with its own magnitude
            # rather than performing a bounded random walk. Both
            # branches now use a proper difference vector against the
            # current worst-known position (the canonical DBO obstacle
            # reference) when there is no previous ball-roller to
            # reference.
            ref = worst_pos if i == 0 else pos[i - 1]
            if rng.random() > Rb:
                # rolling straight (no obstacle)
                delta = rng.standard_normal(dim)
                pos[i] = pos[i] + 0.1 * np.abs(delta) * (pos[i] - ref)
            else:
                # dancing (obstacle hit)
                theta = rng.uniform(0.01, np.pi - 0.01)
                pos[i] = pos[i] + np.tan(theta) * np.abs(pos[i] - ref)
            pos[i] = np.clip(pos[i], lb, ub)
            f = func(pos[i]); fes += 1; fit[i] = f
            if f < best_cost: best_cost = float(f); best_pos = pos[i].copy()

        if done:
            conv.append(best_cost); break

        # -- Group 2: foragers (indices n_roll..pop_size-n_brood-1) --
        lb_loc = np.clip(best_pos - R * (ub - lb), lb, ub)
        ub_loc = np.clip(best_pos + R * (ub - lb), lb, ub)
        for i in range(n_roll, pop_size - n_brood):
            if fes >= max_fes: done = True; break
            C1 = rng.random(dim)
            C2 = rng.random(dim)
            pos[i] = pos[i] + C1 * (pos[i] - lb_loc) + C2 * (pos[i] - ub_loc)
            pos[i] = np.clip(pos[i], lb, ub)
            f = func(pos[i]); fes += 1; fit[i] = f
            if f < best_cost: best_cost = float(f); best_pos = pos[i].copy()
            # update worst
            wi = int(np.argmax(fit))
            worst_pos = pos[wi].copy()

        if done:
            conv.append(best_cost); break

        # -- Group 3: brooders (last n_brood indices) --
        for i in range(pop_size - n_brood, pop_size):
            if fes >= max_fes: done = True; break
            pos[i] = best_pos + (2.0 * rng.random(dim) - 1.0) * R * (ub - lb)
            pos[i] = np.clip(pos[i], lb, ub)
            f = func(pos[i]); fes += 1; fit[i] = f
            if f < best_cost: best_cost = float(f); best_pos = pos[i].copy()

        # sort for next iteration
        sidx = np.argsort(fit)
        pos, fit = pos[sidx], fit[sidx]
        worst_pos = pos[-1].copy()

        conv.append(best_cost)
        if done:
            break

    return best_cost, best_pos, conv


# ===========================================================================
# 12. AVOA  –  African Vultures Optimisation Algorithm (Abdollahzadeh et al., 2022)
#     Fixed: 1-D lb/ub, clip instead of wrap, standardised interface
# ===========================================================================

def run_AVOA(func, lb, ub, dim, max_fes, pop_size=30, seed=None):
    rng = np.random.default_rng(seed)
    lb, ub = _bounds(lb, ub, dim)
    max_iter = max_fes // pop_size
    alpha, betha = 0.8, 0.2
    p1, p2, p3   = 0.6, 0.4, 0.6
    Gama = 2.5

    pos = rng.uniform(lb, ub, (pop_size, dim))
    fit = np.array([func(pos[i]) for i in range(pop_size)])
    fes = pop_size

    sidx = np.argsort(fit)
    best_pos  = pos[sidx[0]].copy()
    best_cost = float(fit[sidx[0]])
    conv = [best_cost]

    def _roulette(probs):
        cs = np.cumsum(probs)
        r  = rng.random()
        return int(np.searchsorted(cs, r))

    it = 0
    while fes < max_fes:
        it += 1
        sidx = np.argsort(fit)
        pos  = pos[sidx]; fit = fit[sidx]
        V1, V2 = pos[0], pos[1]

        t3 = (rng.uniform(-2, 2) *
              ((np.sin((np.pi/2)*(it/max_iter))**Gama)
               + np.cos((np.pi/2)*(it/max_iter)) - 1))
        P1 = (2*rng.random() + 1) * (1 - it/max_iter) + t3
        F  = P1 * (2*rng.random() - 1)

        done = False
        for i in range(pop_size):
            rv = _roulette([alpha, betha])
            rand_vulture = V1 if rv == 0 else V2

            if abs(F) >= 1:
                # Exploration
                if rng.random() < p1:
                    new_x = (rand_vulture
                             - np.abs(2*rng.random()*rand_vulture - pos[i]) * F)
                else:
                    new_x = (rand_vulture - F
                             + rng.random() * ((ub - lb)*rng.random() + lb))
            else:
                # Exploitation
                if abs(F) < 0.5:
                    if rng.random() < p2:
                        A = (V1 - (V1 * pos[i]) / (V1 - pos[i]**2 + 1e-10) * F)
                        B = (V2 - (V2 * pos[i]) / (V2 - pos[i]**2 + 1e-10) * F)
                        new_x = (A + B) / 2.0
                    else:
                        lv   = _levy(dim, rng)
                        new_x = rand_vulture - np.abs(rand_vulture - pos[i]) * F * lv
                else:
                    if rng.random() < p3:
                        new_x = (np.abs(2*rng.random()*rand_vulture - pos[i])
                                 * (F + rng.random()) - (rand_vulture - pos[i]))
                    else:
                        s1 = rand_vulture * (rng.random() * pos[i] / (2*np.pi)) * np.cos(pos[i])
                        s2 = rand_vulture * (rng.random() * pos[i] / (2*np.pi)) * np.sin(pos[i])
                        new_x = rand_vulture - (s1 + s2)

            pos[i] = np.clip(new_x, lb, ub)
            f = func(pos[i]); fes += 1; fit[i] = f
            if f < best_cost: best_cost = float(f); best_pos = pos[i].copy()
            if fes >= max_fes: done = True; break

        conv.append(best_cost)
        if done:
            break

    return best_cost, best_pos, conv


# ===========================================================================
# 13. SMA  –  Slime Mould Algorithm (Li et al., 2020)
# ===========================================================================

def run_SMA(func, lb, ub, dim, max_fes, pop_size=30, seed=None):
    rng = np.random.default_rng(seed)
    lb, ub = _bounds(lb, ub, dim)
    max_iter = max_fes // pop_size
    z = 0.03

    pos = rng.uniform(lb, ub, (pop_size, dim))
    fit = np.array([func(pos[i]) for i in range(pop_size)])
    fes = pop_size

    best_idx  = int(np.argmin(fit))
    best_pos  = pos[best_idx].copy()
    best_cost = float(fit[best_idx])
    conv = [best_cost]

    it = 0
    while fes < max_fes:
        it += 1
        # vb oscillates, vc decreases
        a  = np.arctanh(-(it / max_iter) + 1.0 + 1e-10)
        vc = 1.0 - it / max_iter

        # Sort to assign weights
        sidx = np.argsort(fit)
        bF   = float(fit[sidx[0]])
        wF   = float(fit[sidx[-1]])

        W = np.ones(pop_size)
        for rank, idx in enumerate(sidx):
            r = rng.random()
            ratio = (bF - fit[idx]) / (bF - wF + 1e-10)
            ratio = max(ratio, 0.0)   # guard against float rounding
            if rank < pop_size // 2:
                W[idx] = 1 + r * np.log(ratio + 1)
            else:
                W[idx] = 1 - r * np.log(ratio + 1)

        done = False
        for i in range(pop_size):
            vb = rng.uniform(-a, a, dim)
            p  = np.tanh(abs(fit[i] - best_cost))
            r  = rng.random()

            if r < z:
                pos[i] = rng.uniform(lb, ub)
            elif r < p:
                # approach food
                A  = rng.integers(pop_size)
                B  = rng.integers(pop_size)
                pos[i] = np.clip(
                    best_pos + vb * (W[i] * pos[A] - pos[B]),
                    lb, ub)
            else:
                pos[i] = np.clip(vc * pos[i], lb, ub)

            f = func(pos[i]); fes += 1; fit[i] = f
            if f < best_cost: best_cost = float(f); best_pos = pos[i].copy()
            if fes >= max_fes: done = True; break

        conv.append(best_cost)
        if done:
            break

    return best_cost, best_pos, conv


# ===========================================================================
# 14. EO  –  Equilibrium Optimiser (Faramarzi et al., 2020)
# ===========================================================================

def run_EO(func, lb, ub, dim, max_fes, pop_size=30, seed=None):
    rng = np.random.default_rng(seed)
    lb, ub = _bounds(lb, ub, dim)
    max_iter = max_fes // pop_size
    a2 = 1.0
    GP = 0.5

    pos = rng.uniform(lb, ub, (pop_size, dim))
    fit = np.array([func(pos[i]) for i in range(pop_size)])
    fes = pop_size

    best_idx  = int(np.argmin(fit))
    best_pos  = pos[best_idx].copy()
    best_cost = float(fit[best_idx])
    conv = [best_cost]

    # Equilibrium pool: 4 best (slots 0-3) + their centroid (slot 4).
    # Only slots 0-3 are ever evaluated/replaced candidates; slot 4 is a
    # purely geometric construct (the mean of slots 0-3) used only as an
    # extra candidate source for Ceq, never compared against for
    # replacement.
    eq_pos = np.zeros((5, dim))
    eq_fit = np.full(4, np.inf)

    sidx = np.argsort(fit)
    for k in range(min(4, pop_size)):
        eq_pos[k] = pos[sidx[k]].copy()
        eq_fit[k] = fit[sidx[k]]
    eq_pos[4] = eq_pos[:4].mean(axis=0)

    it = 0
    while fes < max_fes:
        it += 1
        t = (1.0 - it / max_iter) ** (a2 * it / max_iter)

        done = False
        for i in range(pop_size):
            ci  = rng.integers(5)
            Ceq = eq_pos[ci]
            lam = rng.random(dim)
            F   = np.exp(-lam * t)
            r   = rng.random()
            GCP = 0.5 * r * np.ones(dim) if r >= GP else np.zeros(dim)
            G0  = GCP * (Ceq - lam * pos[i])
            G   = G0 * F
            pos[i] = np.clip(
                Ceq + (pos[i] - Ceq) * F + G / (lam + 1e-10) * (1.0 - F),
                lb, ub)
            f = func(pos[i]); fes += 1; fit[i] = f
            if f < best_cost: best_cost = float(f); best_pos = pos[i].copy()
            # BUGFIX: previously compared against eq_fit.max() which
            # included the centroid slot pinned at +inf forever (it was
            # never re-evaluated by any "lazy" path — none existed), so
            # this branch always fired and always overwrote the
            # centroid slot instead of ever updating the real top-4
            # pool. That froze slots 0-3 after initialisation and made
            # the equilibrium pool stop reflecting search progress.
            # Fixed: compare only against the worst of the 4 real,
            # evaluated candidates; the centroid (slot 4) is refreshed
            # geometrically whenever the real pool changes, never
            # itself evaluated or replaced.
            worst_pi = int(np.argmax(eq_fit))
            if f < eq_fit[worst_pi]:
                eq_pos[worst_pi] = pos[i].copy()
                eq_fit[worst_pi] = f
                eq_pos[4] = eq_pos[:4].mean(axis=0)
            if fes >= max_fes: done = True; break

        conv.append(best_cost)
        if done:
            break

    return best_cost, best_pos, conv


# ===========================================================================
# 15. AOA  –  Arithmetic Optimisation Algorithm (Abualigah et al., 2021)
# ===========================================================================

def run_AOA(func, lb, ub, dim, max_fes, pop_size=30, seed=None):
    rng = np.random.default_rng(seed)
    lb, ub = _bounds(lb, ub, dim)
    max_iter = max_fes // pop_size
    alpha   = 5.0
    mu      = 0.499
    MOA_min = 0.2
    MOA_max = 0.9

    pos = rng.uniform(lb, ub, (pop_size, dim))
    fit = np.array([func(pos[i]) for i in range(pop_size)])
    fes = pop_size

    best_idx  = int(np.argmin(fit))
    best_pos  = pos[best_idx].copy()
    best_cost = float(fit[best_idx])
    conv = [best_cost]

    it = 0
    while fes < max_fes:
        it += 1
        MOA = MOA_min + it * (MOA_max - MOA_min) / max_iter
        MOP = 1.0 - (it ** (1.0 / alpha)) / (max_iter ** (1.0 / alpha))

        done = False
        for i in range(pop_size):
            new_x = pos[i].copy()
            for j in range(dim):
                r = rng.random()
                if r > MOA:
                    # Exploration
                    r2 = rng.random()
                    if r2 > 0.5:
                        new_x[j] = best_pos[j] / (MOP + 1e-10) * ((ub[j] - lb[j]) * mu + lb[j])
                    else:
                        new_x[j] = best_pos[j] * MOP * ((ub[j] - lb[j]) * mu + lb[j])
                else:
                    # Exploitation
                    r3 = rng.random()
                    if r3 > 0.5:
                        new_x[j] = best_pos[j] - MOP * ((ub[j] - lb[j]) * mu + lb[j])
                    else:
                        new_x[j] = best_pos[j] + MOP * ((ub[j] - lb[j]) * mu + lb[j])

            pos[i] = np.clip(new_x, lb, ub)
            f = func(pos[i]); fes += 1; fit[i] = f
            if f < best_cost: best_cost = float(f); best_pos = pos[i].copy()
            if fes >= max_fes: done = True; break

        conv.append(best_cost)
        if done:
            break

    return best_cost, best_pos, conv


# ===========================================================================
# 16. GTO  –  Gorilla Troops Optimiser (Abdollahzadeh et al., 2021)
# ===========================================================================

def run_GTO(func, lb, ub, dim, max_fes, pop_size=30, seed=None):
    rng = np.random.default_rng(seed)
    lb, ub = _bounds(lb, ub, dim)
    max_iter = max_fes // pop_size
    p   = 0.03
    beta = 3.0
    GF  = 0.8

    pos = rng.uniform(lb, ub, (pop_size, dim))
    fit = np.array([func(pos[i]) for i in range(pop_size)])
    fes = pop_size

    best_idx    = int(np.argmin(fit))
    silverback  = pos[best_idx].copy()
    sil_cost    = float(fit[best_idx])
    conv = [sil_cost]

    it = 0
    while fes < max_fes:
        it += 1
        C = GF * (1.0 - it / max_iter)

        done = False
        for i in range(pop_size):
            r1 = rng.random()
            if r1 < p:
                # Random migration (exploration)
                new_x = rng.uniform(lb, ub)
            else:
                r2 = rng.random()
                L  = C * (2.0 * rng.random(dim) - 1.0)
                if r2 >= 0.5:
                    # Follow random gorilla
                    ri    = rng.integers(pop_size)
                    new_x = (1.0 - C) * pos[ri] + C * pos[i]
                else:
                    r3 = rng.random()
                    if r3 >= 0.5:
                        new_x = pos[i] - L * (pos[i] - silverback)
                    else:
                        ri    = rng.integers(pop_size)
                        new_x = silverback - (silverback - pos[i]) * L

            pos[i] = np.clip(new_x, lb, ub)
            f = func(pos[i]); fes += 1; fit[i] = f
            if f < sil_cost: sil_cost = float(f); silverback = pos[i].copy()
            if fes >= max_fes: done = True; break

        if not done:
            # Exploitation phase: follow silverback / compete
            for i in range(pop_size):
                if fes >= max_fes: done = True; break
                W = C * (2.0 * rng.random(dim) - 1.0)
                if rng.random() >= 0.5:
                    new_x = np.abs(pos[i] ** beta) ** (1.0 / beta) * W + silverback
                else:
                    new_x = (silverback
                             - (silverback * W - pos[i] * W)
                             * np.cos(2.0 * np.pi * rng.random(dim)))
                pos[i] = np.clip(new_x, lb, ub)
                f = func(pos[i]); fes += 1; fit[i] = f
                if f < sil_cost: sil_cost = float(f); silverback = pos[i].copy()

        conv.append(sil_cost)
        if done:
            break

    return sil_cost, silverback, conv


# ===========================================================================
# Registry
# ===========================================================================

ALGORITHMS = {
    'GA'  : run_GA,
    'PSO' : run_PSO,
    'DE'  : run_DE,
    'GWO' : run_GWO,
    'MFO' : run_MFO,
    'WOA' : run_WOA,
    'SCA' : run_SCA,
    'MVO' : run_MVO,
    'SSA' : run_SSA,
    'HHO' : run_HHO,
    'DBO' : run_DBO,
    'AVOA': run_AVOA,
    'SMA' : run_SMA,
    'EO'  : run_EO,
    'AOA' : run_AOA,
    'GTO' : run_GTO,
}
