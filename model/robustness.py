"""
model/robustness.py

Robustness simulation for the CFLP greedy solution under demand uncertainty.

Interface
---------
run_robustness(w, t, r, alpha, Q, Q0, deltas, n_repeat, seed)

    w       : np.ndarray (n,)   — base demand weights
    t       : np.ndarray (n, n) — travel time matrix (hours)
    r       : np.ndarray (n,)   — rent per m² for each candidate location
    alpha   : float             — opening cost scaling parameter
    Q       : float             — DC capacity
    Q0      : float             — reference capacity
    deltas  : list[float]       — perturbation magnitudes to test (e.g. [0.05, 0.10, 0.20, 0.30])
    n_repeat: int               — number of Monte Carlo repeats per delta (default 30)
    seed    : int               — random seed for reproducibility

    Returns pd.DataFrame with columns:
        delta        — perturbation magnitude
        repeat       — repeat index (0-based)
        jaccard      — Jaccard similarity of opened DC set vs base solution
        n_open_dcs   — number of opened DCs in this run
        objective    — total cost of this run

Demand perturbation model
-------------------------
    w̃_i = w_i · ε_i,   ε_i ~ Uniform(1 − δ, 1 + δ)

Stability metric
----------------
    Jaccard(A, B) = |A ∩ B| / |A ∪ B|
    where A = set of open DC indices in base solution,
          B = set of open DC indices in perturbed solution.
"""

import os
import sys

import numpy as np
import pandas as pd

# Allow direct execution (python3 model/robustness.py) from the optimizasyon/ dir
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

from model.cflp import solve_cflp


def run_robustness(w, t, r, alpha, Q, Q0, deltas=None, n_repeat=30, seed=42):
    """Run robustness simulation and return results as a DataFrame."""
    if deltas is None:
        deltas = [0.05, 0.10, 0.20, 0.30]

    w = np.asarray(w, dtype=float)
    t = np.asarray(t, dtype=float)
    r = np.asarray(r, dtype=float)

    rng = np.random.default_rng(seed)

    # Base solution (unperturbed)
    base = solve_cflp(w, t, r, alpha, Q, Q0, method="greedy")
    base_open = set(np.where(base["y"] > 0.5)[0])

    rows = []
    for delta in deltas:
        for rep in range(n_repeat):
            eps = rng.uniform(1.0 - delta, 1.0 + delta, size=len(w))
            w_perturbed = w * eps

            result = solve_cflp(w_perturbed, t, r, alpha, Q, Q0, method="greedy")
            perturbed_open = set(np.where(result["y"] > 0.5)[0])

            union = base_open | perturbed_open
            intersection = base_open & perturbed_open
            jaccard = len(intersection) / len(union) if union else 1.0

            rows.append({
                "delta":      delta,
                "repeat":     rep,
                "jaccard":    round(jaccard, 6),
                "n_open_dcs": int(result["y"].sum()),
                "objective":  round(float(result["objective"]), 2),
            })

    return pd.DataFrame(rows)


# ── standalone runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import config

    print("Loading data...")
    w, t, r, _ = config.load_instance()

    # Operating point comes from the 1D parameter search — see config.py
    ALPHA, Q, Q0 = config.ALPHA, config.Q, config.Q0
    DELTAS = config.ROBUSTNESS_DELTAS
    N_REPEAT = config.ROBUSTNESS_REPEATS
    RESULTS_DIR = config.RESULTS_DIR

    print(f"Running robustness simulation: {len(DELTAS)} deltas × {N_REPEAT} repeats = {len(DELTAS)*N_REPEAT} runs")
    df_results = run_robustness(w, t, r, ALPHA, Q, Q0, deltas=DELTAS,
                                n_repeat=N_REPEAT, seed=config.RANDOM_SEED)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "robustness.csv")
    df_results.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")

    print("\nSummary (mean ± std Jaccard per delta):")
    summary = df_results.groupby("delta")["jaccard"].agg(["mean", "std"])
    summary.columns = ["jaccard_mean", "jaccard_std"]
    print(summary.to_string())
