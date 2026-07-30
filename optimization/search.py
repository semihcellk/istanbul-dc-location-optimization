"""
optimization/search.py

Outer-layer optimization for CFLP parameters (alpha and Q).
Responsible for finding the "knee point" of the cost-service tradeoff.
"""

import os
import sys
import json
import pandas as pd
import numpy as np

# Add project root to sys.path to allow module imports when running directly
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from model.cflp import solve_cflp, split_costs

# ── DATA LOADING ─────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

Q0_DEFAULT = 100_000.0   # reference capacity used to normalise opening costs

# Capacity search range. The lower bound must stay above the largest single
# neighborhood (Esenyurt/Yenikent, 118,479 people): with Q below that, no DC
# can absorb it and the "assign each neighborhood to exactly one DC"
# formulation has no feasible solution at all. The earlier range started at
# 50,000, so part of the 2D search space was infeasible — the exact solver
# would reject it outright, while the heuristic quietly dropped the
# oversized neighborhood and reported a misleadingly cheap objective.
Q_RANGE_DEFAULT = (125_000.0, 500_000.0)

def load_optimization_data():
    """Loads and prepares all data needed for the outer-layer optimization."""
    df_n = pd.read_csv(os.path.join(DATA_DIR, "neighborhoods.csv"))
    t = np.load(os.path.join(DATA_DIR, "travel_times.npy"))
    r = df_n["rent_per_m2"].values
    w = df_n["population"].values
    return w, t, r

def save_results(data, filename):
    """Saves optimization results to a JSON file for notebook integration."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f"Results saved to: {path}")

# ── LOSS FUNCTION ────────────────────────────────────────────────────────────

def pre_sample_ranges(w, t, r, alpha_range=(0.1, 10.0), Q_range=Q_RANGE_DEFAULT, steps=5,
                      Q0=Q0_DEFAULT, solver_method="greedy"):
    """
    Pre-samples the parameter space to find min/max for N and S for normalization.
    """
    alphas = np.linspace(alpha_range[0], alpha_range[1], steps)
    Qs = np.linspace(Q_range[0], Q_range[1], steps)
    n_values, s_values = [], []

    print(f"Pre-sampling {steps}x{steps} grid for normalization...")
    for a in alphas:
        for q in Qs:
            res = solve_cflp(w, t, r, a, q, Q0, method=solver_method)
            _, service_cost = split_costs(res, r, a, q, Q0)
            n_values.append(np.sum(res["y"]))
            s_values.append(service_cost)

    return (min(n_values), max(n_values)), (min(s_values), max(s_values))

def knee_point_loss(alpha, Q, w, t, r, Q0=Q0_DEFAULT, n_norm=(0, 1), s_norm=(0, 1), solver_method="greedy"):
    """Computes the Knee-Point Loss: L(alpha, Q) = N_hat + S_hat"""
    res = solve_cflp(w, t, r, alpha, Q, Q0, method=solver_method)
    n_open = np.sum(res["y"])
    _, service_cost = split_costs(res, r, alpha, Q, Q0)

    n_min, n_max = n_norm
    s_min, s_max = s_norm
    n_hat = (n_open - n_min) / (n_max - n_min) if n_max > n_min else 0.0
    s_hat = (service_cost - s_min) / (s_max - s_min) if s_max > s_min else 0.0

    return n_hat + s_hat, {"n_open": int(n_open), "service_cost": float(service_cost), "n_hat": float(n_hat), "s_hat": float(s_hat)}

# ── 1D LINE SEARCHES (generic, unit-testable) ───────────────────────────────
#
# Both routines assume f is unimodal on [a, b] and shrink the bracket until it
# is narrower than tol. They are written against a plain callable so they can
# be tested on analytic functions, independently of the CFLP solver.

def golden_section(f, a, b, tol=1e-3):
    """Golden-section line search. Returns the midpoint of the final bracket."""
    phi = (np.sqrt(5) - 1) / 2
    x1, x2 = b - phi * (b - a), a + phi * (b - a)
    f1, f2 = f(x1), f(x2)

    while (b - a) > tol:
        if f1 < f2:
            b, x2, f2 = x2, x1, f1
            x1 = b - phi * (b - a)
            f1 = f(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + phi * (b - a)
            f2 = f(x2)
    return (a + b) / 2


def fibonacci_search(f, a, b, tol=1e-3):
    """Fibonacci line search. Returns the midpoint of the final bracket.

    The number of evaluations is fixed up-front by the smallest Fibonacci
    number exceeding (b − a)/tol — this is what makes the method optimal in the
    minimax sense for a given evaluation budget, and it is why it needs one or
    two fewer calls than golden section for the same final bracket width.
    """
    fibs = [1, 1]
    while fibs[-1] < (b - a) / tol:
        fibs.append(fibs[-1] + fibs[-2])
    n = len(fibs) - 1

    x1 = a + (fibs[n - 2] / fibs[n]) * (b - a)
    x2 = a + (fibs[n - 1] / fibs[n]) * (b - a)
    f1, f2 = f(x1), f(x2)

    for k in range(n, 2, -1):
        if f1 < f2:
            b, x2, f2 = x2, x1, f1
            x1 = a + (fibs[k - 3] / fibs[k - 1]) * (b - a)
            f1 = f(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + (fibs[k - 2] / fibs[k - 1]) * (b - a)
            f2 = f(x2)
    return (a + b) / 2


# ── 1D SEARCH (alpha only) ──────────────────────────────────────────────────

def run_1d_search(w, t, r, Q_fixed=200000, method="golden", alpha_range=(0.1, 10.0),
                  n_norm=(0, 1), s_norm=(0, 1), tol=1e-3, solver_method="greedy"):
    """Performs 1D search for optimal alpha."""
    searchers = {"golden": golden_section, "fibonacci": fibonacci_search}
    if method not in searchers:
        raise ValueError(f"Unknown 1D method: {method!r}. Use 'golden' or 'fibonacci'.")

    history = []

    def f(alpha):
        loss, info = knee_point_loss(alpha, Q_fixed, w, t, r, n_norm=n_norm, s_norm=s_norm, solver_method=solver_method)
        history.append({"iter": len(history), "alpha": float(alpha), "L": float(loss), **info})
        return loss

    alpha_opt = searchers[method](f, alpha_range[0], alpha_range[1], tol)

    # Evaluate the returned point so L_opt corresponds to alpha_opt itself
    # rather than to the last probe of the bracketing loop.
    return {"alpha_opt": float(alpha_opt), "L_opt": float(f(alpha_opt)), "history": history}

# ── 2D SEARCH (alpha and Q) ─────────────────────────────────────────────────

def run_2d_search(w, t, r, method="nelder_mead", alpha_range=(0.1, 10.0), Q_range=Q_RANGE_DEFAULT,
                  n_norm=(0, 1), s_norm=(0, 1), tol=1e-3, max_iter=50, solver_method="greedy"):
    """Performs 2D search for optimal alpha and Q.

    Nelder-Mead runs in *normalised* coordinates u ∈ [0,1]² which are mapped
    onto the (α, Q) box. α spans ~10 units while Q spans ~450,000, so a simplex
    built directly on the raw parameters is dominated by the Q direction: the
    reflection/expansion steps barely move α and the ``tol`` convergence test
    (a distance in raw units) can never be met. Normalising puts both axes on
    the same footing and makes ``tol`` a meaningful stopping criterion.
    """
    if method not in ("grid", "nelder_mead"):
        raise ValueError(f"Unknown 2D method: {method!r}. Use 'grid' or 'nelder_mead'.")

    history = []

    def f(x):
        alpha = float(np.clip(x[0], alpha_range[0], alpha_range[1]))
        Q = float(np.clip(x[1], Q_range[0], Q_range[1]))
        loss, info = knee_point_loss(alpha, Q, w, t, r, n_norm=n_norm, s_norm=s_norm, solver_method=solver_method)
        history.append({"iter": len(history), "alpha": alpha, "Q": Q, "L": float(loss), **info})
        return loss

    if method == "grid":
        steps = 10
        alphas = np.linspace(alpha_range[0], alpha_range[1], steps)
        Qs = np.linspace(Q_range[0], Q_range[1], steps)
        best_l, best_params = np.inf, (0, 0)
        for a in alphas:
            for q in Qs:
                l = f([a, q])
                if l < best_l:
                    best_l, best_params = l, (a, q)
        return {"alpha_opt": float(best_params[0]), "Q_opt": float(best_params[1]), "L_opt": float(best_l), "history": history}

    # ── Nelder-Mead on the unit square ────────────────────────────────────────
    lo = np.array([alpha_range[0], Q_range[0]], dtype=float)
    hi = np.array([alpha_range[1], Q_range[1]], dtype=float)

    def to_params(u):
        """Map u ∈ [0,1]² onto the (α, Q) box, clipping to stay feasible."""
        return lo + np.clip(u, 0.0, 1.0) * (hi - lo)

    def g(u):
        return f(to_params(u))

    u0 = np.array([0.5, 0.5])
    simplex = [u0, u0 + np.array([0.25, 0.0]), u0 + np.array([0.0, 0.25])]
    values = [g(u) for u in simplex]

    for _ in range(max_iter):
        order = np.argsort(values)
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        if max(np.linalg.norm(simplex[k] - simplex[0]) for k in (1, 2)) < tol:
            break

        centroid = (simplex[0] + simplex[1]) / 2
        xr = centroid + 1.0 * (centroid - simplex[2])
        fr = g(xr)

        if values[0] <= fr < values[1]:
            simplex[2], values[2] = xr, fr
        elif fr < values[0]:
            xe = centroid + 2.0 * (centroid - simplex[2])
            fe = g(xe)
            simplex[2], values[2] = (xe, fe) if fe < fr else (xr, fr)
        else:
            xc = centroid + 0.5 * (centroid - simplex[2])
            fc = g(xc)
            if fc < values[2]:
                simplex[2], values[2] = xc, fc
            else:
                for i in range(1, 3):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    values[i] = g(simplex[i])

    best_idx = int(np.argmin(values))
    alpha_opt, Q_opt = to_params(simplex[best_idx])
    return {"alpha_opt": float(alpha_opt), "Q_opt": float(Q_opt),
            "L_opt": float(values[best_idx]), "history": history}

# ── SEARCH COMPARISON TABLE ──────────────────────────────────────────────────

def generate_search_comparison(res_golden, res_fib, res_nm, res_grid):
    """Generates a unified comparison CSV from the four search methods."""
    rows = [
        {
            "method": "Golden Section (1D)",
            "dimensions": "1D (α only)",
            "n_evaluations": len(res_golden["history"]),
            "final_L": round(res_golden["L_opt"], 6),
            "optimal_alpha": round(res_golden["alpha_opt"], 4),
            "optimal_Q": "fixed (200,000)",
        },
        {
            "method": "Fibonacci (1D)",
            "dimensions": "1D (α only)",
            "n_evaluations": len(res_fib["history"]),
            "final_L": round(res_fib["L_opt"], 6),
            "optimal_alpha": round(res_fib["alpha_opt"], 4),
            "optimal_Q": "fixed (200,000)",
        },
        {
            "method": "Nelder-Mead (2D)",
            "dimensions": "2D (α, Q)",
            "n_evaluations": len(res_nm["history"]),
            "final_L": round(res_nm["L_opt"], 6),
            "optimal_alpha": round(res_nm["alpha_opt"], 4),
            "optimal_Q": round(res_nm["Q_opt"], 0),
        },
        {
            "method": "Grid Search (2D)",
            "dimensions": "2D (α, Q)",
            "n_evaluations": len(res_grid["history"]),
            "final_L": round(res_grid["L_opt"], 6),
            "optimal_alpha": round(res_grid["alpha_opt"], 4),
            "optimal_Q": round(res_grid["Q_opt"], 0),
        },
    ]
    return pd.DataFrame(rows)

# ── MAIN ANALYSIS ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    w, t, r = load_optimization_data()
    print(f"Data loaded: {len(w)} neighborhoods.")

    if Q_RANGE_DEFAULT[0] < w.max():
        raise SystemExit(
            f"Q_RANGE_DEFAULT lower bound ({Q_RANGE_DEFAULT[0]:,.0f}) is below the "
            f"largest neighborhood ({w.max():,.0f}); every instance in that part of "
            "the search space is infeasible."
        )

    # 1. Normalization
    n_norm, s_norm = pre_sample_ranges(w, t, r, steps=4)
    print(f"N Range: {n_norm}, S Range: {s_norm}")
    
    # 2. 1D Comparison: Golden vs Fibonacci
    print("\n--- 1D Search Comparison ---")
    res_golden = run_1d_search(w, t, r, method="golden", n_norm=n_norm, s_norm=s_norm)
    res_fib = run_1d_search(w, t, r, method="fibonacci", n_norm=n_norm, s_norm=s_norm)
    
    print(f"Golden Search: alpha={res_golden['alpha_opt']:.4f}, Calls={len(res_golden['history'])}")
    print(f"Fibonacci Search: alpha={res_fib['alpha_opt']:.4f}, Calls={len(res_fib['history'])}")
    save_results(res_golden, "search_1d_golden.json")
    save_results(res_fib, "search_1d_fibonacci.json")
    
    # 3. 2D Comparison: Nelder-Mead vs Grid (10×10)
    print("\n--- 2D Search Comparison ---")
    res_nm = run_2d_search(w, t, r, method="nelder_mead", n_norm=n_norm, s_norm=s_norm, max_iter=20)
    res_grid = run_2d_search(w, t, r, method="grid", n_norm=n_norm, s_norm=s_norm)
    
    print(f"Nelder-Mead: alpha={res_nm['alpha_opt']:.4f}, Q={res_nm['Q_opt']:.0f}, Calls={len(res_nm['history'])}")
    print(f"Grid Search: alpha={res_grid['alpha_opt']:.4f}, Q={res_grid['Q_opt']:.0f}, Calls={len(res_grid['history'])}")
    save_results(res_nm, "search_2d_nelder_mead.json")
    save_results(res_grid, "search_2d_grid.json")

    # 4. Search Comparison Table
    print("\n--- Search Comparison Table ---")
    df_comp = generate_search_comparison(res_golden, res_fib, res_nm, res_grid)
    comp_path = os.path.join(RESULTS_DIR, "search_comparison.csv")
    df_comp.to_csv(comp_path, index=False)
    print(df_comp.to_string(index=False))
    print(f"\nSaved → {comp_path}")
