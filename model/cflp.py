"""
model/cflp.py

Capacitated Facility Location Problem (CFLP) solver.

Interface
---------
solve_cflp(w, t, r, alpha, Q, Q0, method="cbc", time_limit=120)

    w      : np.ndarray (n_I,)      — demand weights (population per neighborhood)
    t      : np.ndarray (n_I, n_J)  — travel times in hours; t[i,j] = time from i to DC at j.
                                      Square (n, n) for the Istanbul instance, where every
                                      neighborhood is also a candidate site, but any
                                      demand-points × candidate-sites matrix works.
    r      : np.ndarray (n_J,)      — commercial rent per m² for each candidate DC location
    alpha  : float                  — opening cost scaling parameter
    Q      : float                  — DC capacity in population units
    Q0     : float                  — reference capacity for cost normalization
    method : "cbc" | "greedy"
    time_limit : int                — solver time limit in seconds (CBC only)

    Returns dict:
        y             np.ndarray (n_J,)      — 1 if DC opened at location j, else 0
        z             np.ndarray (n_I, n_J)  — z[i,j]=1 if neighborhood i assigned to DC j
        objective     float              — total cost (opening + service)
        runtime       float              — wall-clock seconds
        gap           float              — optimality gap (0.0 for greedy)
        status        str                — solver status ("Optimal", "Heuristic", ...)
        n_unassigned  int                — neighborhoods left unserved (0 = feasible)

Module layout
-------------
  _solve_cbc    — exact MILP via PuLP/CBC  (Alperen)
  _solve_greedy — greedy-add heuristic     (Hasan)
  _reassign     — capacity-constrained assignment helper
"""

import time
import numpy as np


# ── public interface ──────────────────────────────────────────────────────────

def opening_costs(r, alpha, Q, Q0):
    """Per-candidate opening cost  f_j = α · r_j · √(Q / Q₀).

    Kept here so every caller (solver, parameter search, plotting scripts and
    the notebook) prices facilities with exactly the same formula.
    """
    return alpha * np.asarray(r, dtype=float) * np.sqrt(Q / Q0)


def split_costs(result, r, alpha, Q, Q0):
    """Split a solver result's objective into (opening_cost, service_cost)."""
    opening = float(opening_costs(r, alpha, Q, Q0) @ result["y"])
    return opening, float(result["objective"]) - opening


def solve_cflp(w, t, r, alpha, Q, Q0, method="cbc", time_limit=120):
    """Solve the Capacitated Facility Location Problem."""
    w = np.asarray(w, dtype=float)
    t = np.asarray(t, dtype=float)
    r = np.asarray(r, dtype=float)

    # ── input validation ──────────────────────────────────────────────────
    if t.ndim != 2:
        raise ValueError(f"t must be a 2-D matrix, got shape {t.shape}")
    if t.shape[0] != w.shape[0]:
        raise ValueError(
            f"t has {t.shape[0]} rows but w has {w.shape[0]} entries; "
            "rows of t must match the demand points."
        )
    if t.shape[1] != r.shape[0]:
        raise ValueError(
            f"t has {t.shape[1]} columns but r has {r.shape[0]} entries; "
            "columns of t must match the candidate DC locations."
        )
    if np.any(w < 0):
        raise ValueError("demand weights w must be non-negative")
    if Q <= 0 or Q0 <= 0:
        raise ValueError("capacities Q and Q0 must be strictly positive")
    if w.max(initial=0.0) > Q:
        raise ValueError(
            f"infeasible instance: max demand {w.max():,.0f} exceeds DC capacity Q={Q:,.0f}"
        )

    f = opening_costs(r, alpha, Q, Q0)

    if method == "cbc":
        return _solve_cbc(w, t, f, Q, time_limit)
    elif method == "greedy":
        return _solve_greedy(w, t, f, Q)
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'cbc' or 'greedy'.")


# ── CBC exact solver ──────────────────────────────────────────────────────────

def _solve_cbc(w, t, f, Q, time_limit):
    """Exact MILP via PuLP/CBC.

    Formulation
    -----------
    Decision variables:
        y[j] ∈ {0,1}    — 1 if DC opened at candidate location j
        z[i][j] ∈ {0,1} — 1 if neighborhood i is assigned to DC j

    Objective:
        min  Σ_j f[j]·y[j]  +  Σ_i Σ_j w[i]·t[i,j]·z[i][j]

    Constraints:
        (1) Σ_j z[i][j] = 1           ∀i   — each neighborhood assigned to exactly one DC
        (2) z[i][j] ≤ y[j]            ∀i,j — assignment only to open DCs
        (3) Σ_i w[i]·z[i][j] ≤ Q·y[j] ∀j   — capacity constraint
        (4) y[j], z[i][j] ∈ {0,1}           — binary variables

    If the solver hits *time_limit* before proving optimality, the best
    feasible solution found so far is returned with a non-zero gap.
    """
    import pulp

    n_I = len(w)          # number of neighborhoods (demand points)
    n_J = t.shape[1]      # number of candidate DC locations

    I = range(n_I)
    J = range(n_J)

    # ── define the problem ────────────────────────────────────────────────
    prob = pulp.LpProblem("CFLP", pulp.LpMinimize)

    # decision variables
    y = [pulp.LpVariable(f"y_{j}", cat=pulp.LpBinary) for j in J]
    z = [[pulp.LpVariable(f"z_{i}_{j}", cat=pulp.LpBinary) for j in J] for i in I]

    # objective: min Σ f_j·y_j + Σ_i Σ_j w_i·t_ij·z_ij
    prob += (
        pulp.lpSum(f[j] * y[j] for j in J)
        + pulp.lpSum(w[i] * t[i, j] * z[i][j] for i in I for j in J)
    ), "TotalCost"

    # constraint (1): each neighborhood assigned to exactly one DC
    for i in I:
        prob += pulp.lpSum(z[i][j] for j in J) == 1, f"Assign_{i}"

    # constraint (2): assignment only to open DCs
    for i in I:
        for j in J:
            prob += z[i][j] <= y[j], f"Link_{i}_{j}"

    # constraint (3): capacity constraint
    for j in J:
        prob += (
            pulp.lpSum(w[i] * z[i][j] for i in I) <= Q * y[j]
        ), f"Capacity_{j}"

    # ── solve with CBC ────────────────────────────────────────────────────
    solver = pulp.PULP_CBC_CMD(
        timeLimit=time_limit,
        msg=0,          # suppress solver output
    )

    t0 = time.time()
    prob.solve(solver)
    runtime = time.time() - t0

    # ── extract solution ──────────────────────────────────────────────────
    # "Not Solved" with a feasible incumbent is what CBC reports when the time
    # limit is hit: PuLP marks status=0 but the variables are populated, so any
    # status that produced variable values is accepted here.
    status = pulp.LpStatus[prob.status]

    if status == "Infeasible" or y[0].varValue is None:
        raise RuntimeError(
            f"CBC returned no usable solution (status={status!r}). "
            "The instance may be infeasible or the time limit too tight."
        )

    y_val = np.array([v.varValue if v.varValue is not None else 0.0 for v in y])
    z_val = np.array(
        [[z[i][j].varValue if z[i][j].varValue is not None else 0.0 for j in J] for i in I]
    )

    # round near-integer values produced by the solver
    y_val = np.round(y_val).astype(float)
    z_val = np.round(z_val).astype(float)

    objective = float(pulp.value(prob.objective)) if pulp.value(prob.objective) is not None else float("inf")

    # Optimality gap. A proven-optimal run has gap 0 by definition; otherwise we
    # use CBC's best bound if PuLP exposed it, and fall back to NaN ("unknown")
    # rather than silently reporting a zero gap for a time-limited run.
    best_bound = getattr(prob, "bestBound", None)
    if status == "Optimal":
        gap = 0.0
    elif best_bound is not None and objective not in (0.0, float("inf")):
        gap = abs(objective - best_bound) / abs(objective)
    else:
        gap = float("nan")

    return {
        "y":            y_val,
        "z":            z_val,
        "objective":    objective,
        "runtime":      runtime,
        "gap":          gap,
        "status":       status,
        "n_unassigned": int(np.sum(z_val.sum(axis=1) < 0.5)),
    }


# ── Greedy-Add Heuristic ──────────────────────────────────────────────────────

def _solve_greedy(w, t, f, Q):
    """Greedy-add heuristic for CFLP.

    Each iteration opens the DC candidate with the highest marginal gain:
        gain(j) = Σ_i w_i * max(0, current_cost_i − t_ij) − f_j

    After each opening, neighborhoods are re-assigned to open DCs respecting
    capacity Q. Stops when no candidate yields a positive gain and all
    neighborhoods are assigned.

    Unserved demand is priced at a finite penalty *UNSERVED* rather than at
    infinity. This matters: with an infinite baseline every candidate would
    score gain=+inf on the first iteration and ``np.argmax`` would always pick
    index 0 — i.e. the first DC would be chosen by array order instead of by
    cost. With the finite penalty the first iteration reduces to
    ``argmin_j (f_j + Σ_i w_i·t_ij)``, the candidate that is cheapest to serve
    the whole city from, which is the textbook greedy-add starting point.
    """
    t0 = time.time()
    n_I, n_J = t.shape

    # Penalty per unit of unserved demand: strictly larger than any travel time,
    # so opening a DC that can serve unassigned demand always looks attractive.
    UNSERVED = 2.0 * float(t.max()) + 1.0

    open_mask = np.zeros(n_J, dtype=bool)
    assign = np.full(n_I, -1, dtype=int)          # assign[i] = DC index for neighborhood i
    assign_cost = np.full(n_I, UNSERVED)          # current travel cost per neighborhood

    while True:
        # ── compute gain for every closed candidate DC (vectorized) ──────────
        # savings[j] = Σ_i w_i * max(0, assign_cost[i] − t[i,j])
        savings = (w[:, None] * np.maximum(0.0, assign_cost[:, None] - t)).sum(axis=0)
        gains = savings - f
        gains[open_mask] = -np.inf

        best_j = int(np.argmax(gains))
        best_gain = float(gains[best_j])

        if best_gain <= 0.0:
            # No improvement — stop if all neighborhoods are assigned
            if not np.any(assign == -1):
                break
            # Otherwise force-open the cheapest remaining DC to ensure feasibility
            closed = np.where(~open_mask)[0]
            if len(closed) == 0:
                break
            unassigned = np.where(assign == -1)[0]
            force_cost = (
                f[closed]
                + (w[unassigned, None] * t[np.ix_(unassigned, closed)]).sum(axis=0)
            )
            best_j = int(closed[np.argmin(force_cost)])

        open_mask[best_j] = True
        assign, assign_cost = _reassign(w, t, np.where(open_mask)[0], Q, UNSERVED)

    # ── build result arrays ───────────────────────────────────────────────────
    y = open_mask.astype(float)

    z = np.zeros((n_I, n_J))
    valid = assign >= 0
    z[np.where(valid)[0], assign[valid]] = 1.0

    open_cost = float(f @ y)
    valid_idx = np.where(valid)[0]
    service_cost = float(np.dot(w[valid_idx], t[valid_idx, assign[valid_idx]]))
    objective = open_cost + service_cost

    n_unassigned = int(np.sum(~valid))

    return {
        "y":            y,
        "z":            z,
        "objective":    float(objective),
        "runtime":      float(time.time() - t0),
        "gap":          0.0,
        "status":       "Heuristic" if n_unassigned == 0 else "Infeasible",
        "n_unassigned": n_unassigned,
    }


# ── Assignment helper ─────────────────────────────────────────────────────────

def _reassign(w, t, open_arr, Q, unserved_cost=np.inf):
    """Capacity-constrained greedy assignment over a fixed set of open DCs.

    Neighborhoods are processed in descending order of *regret* — the gap
    between their nearest and second-nearest open DC. A neighborhood with a
    large regret loses the most if its nearest DC fills up, so it is placed
    first. (Ordering by nearest-DC cost alone systematically serves the easy
    cases first and pushes the cost onto the constrained ones.)

    Parameters
    ----------
    w             : np.ndarray (n_I,)
    t             : np.ndarray (n_I, n_J)
    open_arr      : np.ndarray (k,)  — indices of currently open DCs
    Q             : float            — capacity per DC
    unserved_cost : float            — cost recorded for a neighborhood that no
                                       open DC has capacity for

    Returns
    -------
    assign      : np.ndarray (n_I,)  — DC index for each neighborhood (-1 if unassigned)
    assign_cost : np.ndarray (n_I,)  — travel cost per neighborhood (unserved_cost if unassigned)
    """
    n_I = t.shape[0]
    k = len(open_arr)
    assign = np.full(n_I, -1, dtype=int)
    load = np.zeros(k)               # load[idx] = population at open_arr[idx]

    t_open = t[:, open_arr]          # (n_I, k) — travel times to open DCs only

    if k == 1:
        order = np.argsort(t_open[:, 0])
    else:
        # regret = second-nearest minus nearest open DC; largest regret first
        two_best = np.partition(t_open, 1, axis=1)[:, :2]
        regret = two_best[:, 1] - two_best[:, 0]
        order = np.argsort(-regret)

    for i in order:
        dc_order = np.argsort(t_open[i])     # open DCs sorted by distance from i
        for idx in dc_order:
            if load[idx] + w[i] <= Q:
                assign[i] = open_arr[idx]
                load[idx] += w[i]
                break

    assign_cost = np.where(
        assign >= 0,
        t[np.arange(n_I), np.where(assign >= 0, assign, 0)],
        unserved_cost,
    )
    return assign, assign_cost
