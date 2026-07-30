"""
config.py

Single source of truth for the project's paths and for the operating point at
which the reported results are produced.

The operating point used to be copy-pasted into `model/robustness.py`,
`experiments/cbc_vs_greedy.py`, `generate_outputs.py` and the notebook, so
re-tuning α meant editing four files and hoping none was missed.
"""

import os

# ── paths ─────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")
MAPS_DIR = os.path.join(OUTPUTS_DIR, "maps")

# ── operating point ───────────────────────────────────────────────────────────
#
# ALPHA is the knee point of the cost/service trade-off found by the 1D
# parameter search (Golden Section and Fibonacci agree to 3 decimals with Q
# fixed at 200,000). Re-run `python optimization/search.py` and update this
# value if the model, the data or the loss function changes.
# See results/search_comparison.csv.

ALPHA = 1.9967      # opening-cost scaling factor  [m²·month]
Q = 200_000.0       # DC capacity                  [population]
Q0 = 100_000.0      # reference capacity           [population]

# Search bounds shared by the 1D and 2D outer-layer searches.
# The Q lower bound must exceed the largest neighborhood (118,479 people),
# otherwise no DC can serve it and the instance is infeasible by construction.
ALPHA_RANGE = (0.1, 10.0)
Q_RANGE = (125_000.0, 500_000.0)

# ── experiment settings ───────────────────────────────────────────────────────

CBC_TIME_LIMIT = 120        # seconds per CBC run
CBC_MAX_N = 400             # skip CBC above this instance size
INSTANCE_SIZES = [20, 50, 100, 200, 400, 890]

ROBUSTNESS_DELTAS = [0.05, 0.10, 0.20, 0.30]
ROBUSTNESS_REPEATS = 30
RANDOM_SEED = 42


def load_instance(travel_times="travel_times.npy"):
    """Load the full Istanbul instance as (w, t, r, df_neighborhoods)."""
    import numpy as np
    import pandas as pd

    df = pd.read_csv(os.path.join(PROCESSED_DIR, "neighborhoods.csv"))
    t = np.load(os.path.join(PROCESSED_DIR, travel_times))
    w = df["population"].values.astype(float)
    r = df["rent_per_m2"].values.astype(float)
    return w, t, r, df
