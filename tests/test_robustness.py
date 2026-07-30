"""
tests/test_robustness.py

Unit tests — Monte Carlo robustness simulation.

Run with pytest:
    python -m pytest tests/test_robustness.py -v
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.robustness import run_robustness


@pytest.fixture
def small_instance():
    rng = np.random.default_rng(99)
    n = 20
    w = rng.uniform(1_000, 15_000, size=n)
    t = rng.uniform(0.1, 2.0, size=(n, n))
    np.fill_diagonal(t, 0.0)
    r = rng.uniform(200, 800, size=n)
    return w, t, r


def test_output_shape_and_columns(small_instance):
    w, t, r = small_instance
    df = run_robustness(w, t, r, alpha=1.5, Q=100_000.0, Q0=100_000.0,
                        deltas=[0.05, 0.20], n_repeat=3)

    assert len(df) == 2 * 3
    assert list(df.columns) == ["delta", "repeat", "jaccard", "n_open_dcs", "objective"]
    assert set(df["delta"]) == {0.05, 0.20}


def test_jaccard_is_a_valid_similarity(small_instance):
    w, t, r = small_instance
    df = run_robustness(w, t, r, alpha=1.5, Q=100_000.0, Q0=100_000.0,
                        deltas=[0.10], n_repeat=5)

    assert df["jaccard"].between(0.0, 1.0).all()


def test_zero_perturbation_reproduces_the_base_solution(small_instance):
    """δ = 0 leaves demand untouched, so every run must match the base exactly."""
    w, t, r = small_instance
    df = run_robustness(w, t, r, alpha=1.5, Q=100_000.0, Q0=100_000.0,
                        deltas=[0.0], n_repeat=3)

    assert (df["jaccard"] == 1.0).all()


def test_stability_degrades_as_uncertainty_grows(small_instance):
    """Mean Jaccard similarity should not increase with larger δ."""
    w, t, r = small_instance
    df = run_robustness(w, t, r, alpha=1.5, Q=100_000.0, Q0=100_000.0,
                        deltas=[0.05, 0.30], n_repeat=15)

    means = df.groupby("delta")["jaccard"].mean()
    assert means[0.30] <= means[0.05] + 1e-9


def test_seed_makes_runs_reproducible(small_instance):
    w, t, r = small_instance
    kwargs = dict(alpha=1.5, Q=100_000.0, Q0=100_000.0, deltas=[0.15], n_repeat=4)

    first = run_robustness(w, t, r, seed=7, **kwargs)
    second = run_robustness(w, t, r, seed=7, **kwargs)
    different = run_robustness(w, t, r, seed=8, **kwargs)

    assert first.equals(second)
    assert not first["objective"].equals(different["objective"])
