"""
experiments/cbc_vs_greedy.py

CBC vs Greedy comparison across instance sizes.

Runs both solvers on subsets of the Istanbul neighborhood data, records
runtime, objective value, number of open DCs, and optimality gap.

Output: results/cbc_vs_greedy.csv
"""

import os
import sys

import numpy as np
import pandas as pd

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

import config
from model.cflp import solve_cflp

# ── parameters (single source of truth: config.py) ────────────────────────────

ALPHA = config.ALPHA
Q = config.Q
Q0 = config.Q0
CBC_TIME_LIMIT = config.CBC_TIME_LIMIT

# Instance sizes to test: small → medium → large
# CBC is skipped above CBC_MAX_N (too many variables for the time limit)
INSTANCE_SIZES = config.INSTANCE_SIZES
CBC_MAX_N = config.CBC_MAX_N
RESULTS_DIR = config.RESULTS_DIR

# ── data loading ──────────────────────────────────────────────────────────────

print("Loading data...")
w_full, t_full, r_full, df_n = config.load_instance()

print(f"Loaded {len(w_full)} neighborhoods, travel matrix {t_full.shape}")
print(f"Parameters: alpha={ALPHA}, Q={Q:,.0f}, Q0={Q0:,.0f}, CBC limit={CBC_TIME_LIMIT}s\n")

# ── run experiments ───────────────────────────────────────────────────────────

rows = []

for n in INSTANCE_SIZES:
    w = w_full[:n]
    t = t_full[:n, :n]
    r = r_full[:n]

    for method in ("greedy", "cbc"):
        if method == "cbc" and n > CBC_MAX_N:
            print(f"  n={n:4d}  {method:6s}  SKIPPED (n > {CBC_MAX_N})")
            continue

        print(f"  n={n:4d}  {method:6s}  ", end="", flush=True)
        res = solve_cflp(w, t, r, ALPHA, Q, Q0, method=method,
                         time_limit=CBC_TIME_LIMIT)

        # Ensure gap is a proper float (NaN if not available, not empty)
        gap_val = res["gap"]
        if gap_val is None:
            gap_val = float("nan")

        row = {
            "instance_size":   n,
            "method":          method,
            "runtime_s":       round(res["runtime"], 3),
            "objective_value": round(res["objective"], 2),
            "n_open_dcs":      int(res["y"].sum()),
            "gap":             float(gap_val),
            "status":          res["status"],
        }
        rows.append(row)
        gap_str = "nan" if np.isnan(row["gap"]) else f"{row['gap']:.4f}"
        print(f"runtime={row['runtime_s']:.2f}s  obj={row['objective_value']:.0f}  "
              f"n_open={row['n_open_dcs']}  gap={gap_str}  status={row['status']}")

# ── save results ──────────────────────────────────────────────────────────────

df_out = pd.DataFrame(rows)
os.makedirs(RESULTS_DIR, exist_ok=True)
out_path = os.path.join(RESULTS_DIR, "cbc_vs_greedy.csv")
df_out.to_csv(out_path, index=False)
print(f"\nSaved → {out_path}")

# ── summary table ──────────────────────────────────────────────────────────────

print("\nResults:")
print(df_out.to_string(index=False))

# Objective gap between methods (where both ran)
print("\nObjective ratio (greedy / CBC) for matched instances:")
grp = df_out.pivot(index="instance_size", columns="method", values="objective_value")
if "cbc" in grp and "greedy" in grp:
    grp["greedy_ratio"] = grp["greedy"] / grp["cbc"]
    matched = grp[["cbc", "greedy", "greedy_ratio"]].dropna()
    print(matched.to_string())
    print(f"\nMean greedy excess cost over CBC: "
          f"{100 * (matched['greedy_ratio'].mean() - 1):.2f}%")

    speed = df_out.pivot(index="instance_size", columns="method", values="runtime_s").dropna()
    print(f"Median CBC/greedy runtime factor: "
          f"{(speed['cbc'] / speed['greedy']).median():,.0f}×")
