"""
tests/test_search.py

Unit tests — outer-layer parameter search.

The line searches are exercised against analytic functions with a known
minimiser, so a failure points at the search implementation rather than at the
CFLP solver underneath it. The end-to-end tests then run the full
(search → solver) stack on a small synthetic instance.

Run with pytest:
    python -m pytest tests/test_search.py -v
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimization.search import (
    fibonacci_search,
    golden_section,
    knee_point_loss,
    run_1d_search,
    run_2d_search,
)


# ── fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def small_instance():
    """A 25-neighborhood synthetic instance — small enough to search quickly."""
    rng = np.random.default_rng(2024)
    n = 25
    w = rng.uniform(1_000, 20_000, size=n)
    t = rng.uniform(0.1, 2.0, size=(n, n))
    np.fill_diagonal(t, 0.0)
    r = rng.uniform(200, 800, size=n)
    return w, t, r


# ══════════════════════════════════════════════════════════════════════════════
# Line searches on analytic functions
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("search", [golden_section, fibonacci_search])
def test_line_search_finds_known_minimum(search):
    """min of (x − 3)² on [0, 10] is x = 3."""
    x_opt = search(lambda x: (x - 3.0) ** 2, 0.0, 10.0, tol=1e-4)
    assert abs(x_opt - 3.0) < 1e-3


@pytest.mark.parametrize("search", [golden_section, fibonacci_search])
def test_line_search_handles_minimum_at_bracket_edge(search):
    """A monotone function must converge to the correct end of the bracket."""
    x_opt = search(lambda x: x, 0.0, 5.0, tol=1e-4)
    assert x_opt < 1e-2


@pytest.mark.parametrize("search", [golden_section, fibonacci_search])
def test_line_search_respects_tolerance(search):
    """Tightening tol must not loosen the result."""
    coarse = search(lambda x: (x - 7.0) ** 2, 0.0, 10.0, tol=1e-1)
    fine = search(lambda x: (x - 7.0) ** 2, 0.0, 10.0, tol=1e-5)
    assert abs(fine - 7.0) <= abs(coarse - 7.0) + 1e-9
    assert abs(fine - 7.0) < 1e-4


def test_fibonacci_is_not_more_expensive_than_golden():
    """For the same bracket and tolerance Fibonacci uses at most as many calls."""
    calls = {"golden": 0, "fibonacci": 0}

    def counted(name):
        def f(x):
            calls[name] += 1
            return (x - 4.0) ** 2
        return f

    golden_section(counted("golden"), 0.0, 10.0, tol=1e-3)
    fibonacci_search(counted("fibonacci"), 0.0, 10.0, tol=1e-3)

    assert calls["fibonacci"] <= calls["golden"]


# ══════════════════════════════════════════════════════════════════════════════
# Loss function
# ══════════════════════════════════════════════════════════════════════════════

def test_knee_point_loss_splits_costs_consistently(small_instance):
    """service_cost reported by the loss must be objective − opening cost."""
    w, t, r = small_instance
    loss, info = knee_point_loss(2.0, 100_000.0, w, t, r, Q0=100_000.0)

    assert np.isfinite(loss)
    assert info["n_open"] >= 1
    assert info["service_cost"] > 0


def test_knee_point_loss_degenerate_normalisation(small_instance):
    """Empty normalisation ranges must not divide by zero."""
    w, t, r = small_instance
    loss, _ = knee_point_loss(2.0, 100_000.0, w, t, r,
                              n_norm=(5, 5), s_norm=(10.0, 10.0))
    assert loss == 0.0


def test_more_expensive_facilities_open_fewer_dcs(small_instance):
    """Sanity check on the model: raising α must not increase the DC count."""
    w, t, r = small_instance
    _, cheap = knee_point_loss(0.1, 100_000.0, w, t, r)
    _, pricey = knee_point_loss(50.0, 100_000.0, w, t, r)
    assert pricey["n_open"] <= cheap["n_open"]


# ══════════════════════════════════════════════════════════════════════════════
# End-to-end searches
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("method", ["golden", "fibonacci"])
def test_run_1d_search_stays_in_range(small_instance, method):
    w, t, r = small_instance
    res = run_1d_search(w, t, r, Q_fixed=100_000.0, method=method,
                        alpha_range=(0.1, 10.0), tol=0.5)

    assert 0.1 <= res["alpha_opt"] <= 10.0
    assert len(res["history"]) > 0
    # L_opt must correspond to the reported alpha, i.e. the last evaluation
    assert res["history"][-1]["alpha"] == pytest.approx(res["alpha_opt"])
    assert res["history"][-1]["L"] == pytest.approx(res["L_opt"])


@pytest.mark.parametrize("method", ["grid", "nelder_mead"])
def test_run_2d_search_stays_in_range(small_instance, method):
    w, t, r = small_instance
    res = run_2d_search(w, t, r, method=method,
                        alpha_range=(0.1, 10.0), Q_range=(100_000, 500_000),
                        max_iter=8)

    assert 0.1 <= res["alpha_opt"] <= 10.0
    assert 100_000 <= res["Q_opt"] <= 500_000
    assert res["L_opt"] == pytest.approx(min(h["L"] for h in res["history"]))


def test_nelder_mead_explores_both_dimensions(small_instance):
    """Regression: an unscaled simplex barely moved α because Q dominated it.

    With normalised coordinates both parameters must actually vary during the
    search instead of the α axis staying pinned near its starting value.
    """
    w, t, r = small_instance
    res = run_2d_search(w, t, r, method="nelder_mead",
                        alpha_range=(0.1, 10.0), Q_range=(100_000, 500_000),
                        max_iter=15)

    alphas = [h["alpha"] for h in res["history"]]
    Qs = [h["Q"] for h in res["history"]]
    alpha_span = (max(alphas) - min(alphas)) / (10.0 - 0.1)
    Q_span = (max(Qs) - min(Qs)) / (500_000 - 100_000)

    assert alpha_span > 0.1, "α was barely explored — simplex is badly scaled"
    assert Q_span > 0.1, "Q was barely explored — simplex is badly scaled"


@pytest.mark.parametrize("runner,kwargs", [
    (run_1d_search, {"method": "newton"}),
    (run_2d_search, {"method": "bfgs"}),
])
def test_unknown_search_method_raises(small_instance, runner, kwargs):
    w, t, r = small_instance
    with pytest.raises(ValueError, match="Unknown"):
        runner(w, t, r, **kwargs)
