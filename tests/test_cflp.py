"""
tests/test_cflp.py

Unit tests — CFLP solvers (CBC exact + greedy heuristic).

Run with pytest:
    python -m pytest tests/test_cflp.py -v
"""

import sys
import os
import numpy as np
import pytest

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.cflp import solve_cflp, opening_costs, split_costs


# ── Test 1: Trivial single DC ───────────────────────────────────────────────
# 3 neighborhoods, 1 candidate location → only feasible solution, is the objective correct?

def test_trivial_single_dc():
    """3 neighborhoods, 1 candidate DC — the only feasible solution must open it."""
    w = np.array([100.0, 200.0, 150.0])
    # t is (3, 1): each neighborhood has only one DC candidate
    t = np.array([
        [0.5],
        [1.0],
        [0.8],
    ])
    r = np.array([20.0])
    alpha = 1.0
    Q = 500.0   # enough for total demand 450
    Q0 = 100.0

    res = solve_cflp(w, t, r, alpha, Q, Q0, method="cbc", time_limit=30)

    # The single DC must be opened
    assert res["y"][0] == 1.0, "The only candidate DC must be opened"

    # All neighborhoods assigned to DC 0
    np.testing.assert_array_equal(res["z"][:, 0], [1.0, 1.0, 1.0])

    # Manually compute expected objective
    f_0 = alpha * r[0] * np.sqrt(Q / Q0)  # opening cost
    service = w[0] * t[0, 0] + w[1] * t[1, 0] + w[2] * t[2, 0]
    expected = f_0 + service

    assert abs(res["objective"] - expected) < 1e-4, (
        f"Objective mismatch: got {res['objective']}, expected {expected}"
    )
    assert res["gap"] < 1e-6, "Optimal solution should have gap ≈ 0"


# ── Test 2: Two candidates, one clearly cheaper ─────────────────────────────
# Does CBC correctly select the cheaper one?

def test_two_candidates_cheaper_wins():
    """2 candidate DCs — one is clearly cheaper in both rent and travel cost."""
    w = np.array([100.0, 100.0, 100.0])
    # DC 0: close to everyone, DC 1: far from everyone
    t = np.array([
        [0.1, 5.0],
        [0.2, 5.0],
        [0.1, 5.0],
    ])
    r = np.array([10.0, 100.0])   # DC 0 is much cheaper rent
    alpha = 1.0
    Q = 400.0   # enough for total demand 300
    Q0 = 100.0

    res = solve_cflp(w, t, r, alpha, Q, Q0, method="cbc", time_limit=30)

    # DC 0 should be open, DC 1 should be closed
    assert res["y"][0] == 1.0, "Cheaper DC (index 0) must be opened"
    assert res["y"][1] == 0.0, "Expensive DC (index 1) must remain closed"

    # All neighborhoods assigned to DC 0
    np.testing.assert_array_equal(res["z"][:, 0], [1.0, 1.0, 1.0])
    np.testing.assert_array_equal(res["z"][:, 1], [0.0, 0.0, 0.0])


# ── Test 3: Capacity forces multiple DCs ────────────────────────────────────
# Q is very low → are at least 2 DCs opened?

def test_capacity_forces_multiple_dcs():
    """Low capacity forces at least 2 DCs to open."""
    w = np.array([200.0, 200.0, 200.0, 200.0])
    # 3 candidate DCs, all equidistant
    t = np.array([
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ])
    r = np.array([10.0, 10.0, 10.0])
    alpha = 1.0
    Q = 450.0   # total demand = 800, one DC can hold at most 450 → need ≥ 2
    Q0 = 100.0

    res = solve_cflp(w, t, r, alpha, Q, Q0, method="cbc", time_limit=30)

    n_open = int(np.sum(res["y"]))
    assert n_open >= 2, f"Expected ≥ 2 DCs open due to capacity, got {n_open}"


# ── Test 4: All constraints satisfied ───────────────────────────────────────
# 10 neighborhoods, 5 candidate DCs — verify every constraint programmatically after solving

def test_all_constraints_satisfied():
    """10 neighborhoods, 5 candidate DCs — verify every MILP constraint holds."""
    np.random.seed(42)
    n_I, n_J = 10, 5

    w = np.random.uniform(50, 500, size=n_I)
    t = np.random.uniform(0.1, 3.0, size=(n_I, n_J))
    r = np.random.uniform(5, 50, size=n_J)
    alpha = 2.0
    Q = 1200.0
    Q0 = 100.0

    res = solve_cflp(w, t, r, alpha, Q, Q0, method="cbc", time_limit=60)

    y = res["y"]
    z = res["z"]

    # Constraint (1): Each neighborhood assigned to exactly one DC → Σ_j z[i,j] = 1  ∀i
    row_sums = z.sum(axis=1)
    np.testing.assert_array_almost_equal(
        row_sums, np.ones(n_I),
        err_msg="Constraint (1) violated: each neighborhood must be assigned to exactly one DC"
    )

    # Constraint (2): z[i,j] ≤ y[j]  ∀i,j — assignment only to open DCs
    for i in range(n_I):
        for j in range(n_J):
            assert z[i, j] <= y[j] + 1e-6, (
                f"Constraint (2) violated: z[{i},{j}]={z[i,j]} but y[{j}]={y[j]}"
            )

    # Constraint (3): Σ_i w[i]·z[i,j] ≤ Q·y[j]  ∀j — capacity constraint
    for j in range(n_J):
        load_j = float(np.dot(w, z[:, j]))
        capacity_j = Q * y[j]
        assert load_j <= capacity_j + 1e-6, (
            f"Constraint (3) violated at DC {j}: load={load_j:.1f} > capacity={capacity_j:.1f}"
        )

    # Constraint (4): Binary variables — y, z ∈ {0, 1}
    assert np.all((y == 0) | (y == 1)), "Constraint (4) violated: y must be binary"
    assert np.all((z == 0) | (z == 1)), "Constraint (4) violated: z must be binary"

    # Objective correctness: compute manually and compare
    f = alpha * r * np.sqrt(Q / Q0)
    opening_cost = float(f @ y)
    service_cost = float(np.sum(w[:, None] * t * z))
    expected_obj = opening_cost + service_cost

    assert abs(res["objective"] - expected_obj) < 1e-2, (
        f"Objective mismatch: solver={res['objective']:.4f}, computed={expected_obj:.4f}"
    )


# ── Test 5: CBC ≤ Greedy objective ───────────────────────────────────────────
# Run both on the same instance — CBC must be at least as good as greedy since it is exact

def test_cbc_at_least_as_good_as_greedy():
    """On the same instance, CBC (exact) should yield objective ≤ greedy (heuristic)."""
    np.random.seed(123)
    n_I, n_J = 8, 8   # square matrix (greedy expects n×n)

    w = np.random.uniform(100, 400, size=n_I)
    t = np.random.uniform(0.2, 2.5, size=(n_I, n_J))
    r = np.random.uniform(10, 40, size=n_J)
    alpha = 1.5
    Q = 1000.0
    Q0 = 100.0

    res_cbc = solve_cflp(w, t, r, alpha, Q, Q0, method="cbc", time_limit=60)
    res_greedy = solve_cflp(w, t, r, alpha, Q, Q0, method="greedy")

    assert res_cbc["objective"] <= res_greedy["objective"] + 1e-6, (
        f"CBC ({res_cbc['objective']:.4f}) should be ≤ Greedy ({res_greedy['objective']:.4f})"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Greedy heuristic
# ══════════════════════════════════════════════════════════════════════════════

# ── Test 6: Greedy picks the first DC on cost, not on array order ────────────
# Regression test. The heuristic used to start from assign_cost = +inf, which
# made every candidate score gain = +inf on the first iteration, so argmax
# always returned index 0 — the first DC was chosen by position in the array.

def test_greedy_first_pick_is_cost_based():
    """The first DC opened must be the cheapest one, even when it is not index 0."""
    n = 6
    t = np.full((n, n), 5.0)
    np.fill_diagonal(t, 0.0)
    t[:, 3] = 0.1     # DC 3 is close to everyone
    t[:, 0] = 4.0     # DC 0 is far from everyone
    r = np.array([1000.0, 50.0, 50.0, 1.0, 50.0, 50.0])   # DC 0 also has the worst rent
    w = np.full(n, 100.0)

    res = solve_cflp(w, t, r, alpha=1.0, Q=1000.0, Q0=100.0, method="greedy")

    assert res["y"][3] == 1.0, "The cheapest candidate (index 3) must be opened"
    assert res["y"][0] == 0.0, "The worst candidate (index 0) must not be opened"


# ── Test 7: Greedy respects capacity and assigns everyone ───────────────────

def test_greedy_solution_is_feasible():
    """Greedy output must satisfy the same constraints as the MILP."""
    rng = np.random.default_rng(7)
    n = 40

    w = rng.uniform(50, 500, size=n)
    t = rng.uniform(0.1, 3.0, size=(n, n))
    np.fill_diagonal(t, 0.0)
    r = rng.uniform(5, 50, size=n)
    Q = 2000.0

    res = solve_cflp(w, t, r, alpha=1.0, Q=Q, Q0=100.0, method="greedy")
    y, z = res["y"], res["z"]

    assert res["n_unassigned"] == 0, "every neighborhood must be served"
    np.testing.assert_array_almost_equal(
        z.sum(axis=1), np.ones(n),
        err_msg="each neighborhood must be assigned to exactly one DC",
    )
    assert np.all(z <= y[None, :] + 1e-9), "neighborhoods may only be assigned to open DCs"

    loads = w @ z
    assert np.all(loads <= Q + 1e-6), f"capacity exceeded: max load {loads.max():.1f} > Q={Q}"

    # Reported objective must match the assignment it returns
    opening, service = split_costs(res, r, 1.0, Q, 100.0)
    assert abs(opening - float(opening_costs(r, 1.0, Q, 100.0) @ y)) < 1e-6
    assert abs(service - float(np.sum(w[:, None] * t * z))) < 1e-6


# ── Test 8: Tight capacity forces the required number of DCs ────────────────

def test_greedy_opens_enough_dcs_under_tight_capacity():
    """With total demand ≫ Q, greedy must open at least ceil(Σw / Q) DCs."""
    n = 30
    rng = np.random.default_rng(11)
    w = np.full(n, 100.0)
    t = rng.uniform(0.5, 2.0, size=(n, n))
    np.fill_diagonal(t, 0.0)
    r = np.full(n, 10.0)
    Q = 400.0   # total demand 3000 → at least 8 DCs needed

    res = solve_cflp(w, t, r, alpha=1.0, Q=Q, Q0=100.0, method="greedy")

    assert res["n_unassigned"] == 0
    assert int(res["y"].sum()) >= int(np.ceil(w.sum() / Q))


# ── Test 9: Rectangular instances (demand points ≠ candidate sites) ─────────

def test_greedy_supports_rectangular_travel_matrix():
    """Greedy must handle n_I ≠ n_J, like the CBC path already does."""
    rng = np.random.default_rng(3)
    n_I, n_J = 12, 4

    w = rng.uniform(50, 200, size=n_I)
    t = rng.uniform(0.1, 2.0, size=(n_I, n_J))
    r = rng.uniform(10, 30, size=n_J)

    res = solve_cflp(w, t, r, alpha=1.0, Q=5000.0, Q0=100.0, method="greedy")

    assert res["y"].shape == (n_J,)
    assert res["z"].shape == (n_I, n_J)
    assert res["n_unassigned"] == 0


# ══════════════════════════════════════════════════════════════════════════════
# Input validation
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("method", ["cbc", "greedy"])
def test_rejects_mismatched_shapes(method):
    """A travel matrix that does not line up with w or r must fail loudly."""
    w = np.array([100.0, 200.0])
    t = np.ones((3, 2))          # 3 rows but only 2 demand points
    r = np.array([10.0, 10.0])

    with pytest.raises(ValueError, match="rows"):
        solve_cflp(w, t, r, 1.0, 1000.0, 100.0, method=method)


def test_rejects_demand_larger_than_capacity():
    """A neighborhood bigger than a whole DC makes the instance infeasible."""
    w = np.array([100.0, 5000.0])
    t = np.ones((2, 2))
    r = np.array([10.0, 10.0])

    with pytest.raises(ValueError, match="infeasible"):
        solve_cflp(w, t, r, 1.0, 1000.0, 100.0, method="greedy")


def test_rejects_unknown_method():
    w = np.array([100.0])
    t = np.ones((1, 1))
    r = np.array([10.0])

    with pytest.raises(ValueError, match="Unknown method"):
        solve_cflp(w, t, r, 1.0, 1000.0, 100.0, method="simulated_annealing")


# ── Opening-cost helper ─────────────────────────────────────────────────────

def test_opening_cost_formula():
    """f_j = α · r_j · √(Q/Q₀)"""
    r = np.array([100.0, 200.0])
    f = opening_costs(r, alpha=2.0, Q=400.0, Q0=100.0)
    np.testing.assert_allclose(f, [400.0, 800.0])   # 2 · r · √4 = 4r
