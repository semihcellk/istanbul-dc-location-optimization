#!/usr/bin/env python3
"""
generate_outputs.py

Generates all static output files for the project:
  1. Interactive Folium maps (peak, off-peak, blended) with MarkerCluster
  2. search_comparison.csv from existing JSON results
  3. Enhanced Pareto curve with peak/off-peak/blended comparison
  4. Improved 4-panel results summary figure

Run from the project root:
    python generate_outputs.py
"""

import os
import sys
import json

# Fix Windows console encoding (cp1252 cannot render the box-drawing output)
if (sys.stdout.encoding or "").lower() != "utf-8" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")     # must be set before pyplot is imported
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster

# ── project paths ─────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import config
from model.cflp import solve_cflp, split_costs

DATA_DIR = config.PROCESSED_DIR
RESULTS_DIR = config.RESULTS_DIR
MAPS_DIR = config.MAPS_DIR
FIGS_DIR = config.FIGURES_DIR

os.makedirs(MAPS_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── data loading ──────────────────────────────────────────────────────────────

print("Loading data...")
w, t_blended, r, df_n = config.load_instance()
t_peak = np.load(os.path.join(DATA_DIR, "travel_times_peak.npy"))
t_offpeak = np.load(os.path.join(DATA_DIR, "travel_times_offpeak.npy"))

print(f"   {len(w)} neighborhoods loaded.")

# ── parameters (single source of truth: config.py) ────────────────────────────

ALPHA = config.ALPHA
Q = config.Q
Q0 = config.Q0

# ══════════════════════════════════════════════════════════════════════════════
# 1. INTERACTIVE FOLIUM MAPS
# ══════════════════════════════════════════════════════════════════════════════

def generate_map(t_matrix, scenario_name, alpha=ALPHA, Q=Q, Q0=Q0):
    """Generate an interactive Folium map with MarkerCluster and enhanced popups."""
    print(f"\n   Generating map: {scenario_name}...")
    
    result = solve_cflp(w, t_matrix, r, alpha, Q, Q0, method="greedy")
    y = result["y"]
    z = result["z"]
    
    open_dcs = np.where(y > 0.5)[0]
    n_open = len(open_dcs)
    print(f"   Open DCs: {n_open}, Objective: {result['objective']:.2f}")
    
    # Compute per-DC statistics
    dc_stats = {}
    for j in open_dcs:
        assigned = np.where(z[:, j] > 0.5)[0]
        total_pop = w[assigned].sum()
        dc_stats[j] = {
            "n_assigned": len(assigned),
            "total_pop": int(total_pop),
            "capacity_pct": round(100 * total_pop / Q, 1),
            "assigned_indices": assigned,
        }
    
    # Color palette for DC clusters
    np.random.seed(42)
    colors = []
    for _ in range(n_open):
        h = np.random.uniform(0, 360)
        s = np.random.uniform(40, 80)
        l = np.random.uniform(40, 60)
        # HSL to hex
        c = s / 100
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = l / 100 - c / 2
        if h < 60:    r_, g_, b_ = c, x, 0
        elif h < 120: r_, g_, b_ = x, c, 0
        elif h < 180: r_, g_, b_ = 0, c, x
        elif h < 240: r_, g_, b_ = 0, x, c
        elif h < 300: r_, g_, b_ = x, 0, c
        else:         r_, g_, b_ = c, 0, x
        rgb = (int((r_ + m) * 255), int((g_ + m) * 255), int((b_ + m) * 255))
        colors.append(f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}")
    
    # Create map
    m = folium.Map(
        location=[41.05, 28.97],
        zoom_start=11,
        tiles="CartoDB dark_matter",
    )
    
    # DC marker cluster
    dc_cluster = MarkerCluster(name="Distribution Centers", show=True)
    
    for idx, j in enumerate(open_dcs):
        color = colors[idx]
        stats = dc_stats[j]
        lat = df_n.iloc[j]["lat"]
        lon = df_n.iloc[j]["lon"]
        name = df_n.iloc[j]["name"]
        district = df_n.iloc[j]["district"]
        
        # Enhanced popup with statistics
        popup_html = f"""
        <div style="font-family: Arial; min-width: 200px;">
            <h4 style="margin:0; color:#e74c3c;">📍 DC: {name}</h4>
            <hr style="margin:4px 0;">
            <b>District:</b> {district}<br>
            <b>Assigned Neighborhoods:</b> {stats['n_assigned']}<br>
            <b>Total Population:</b> {stats['total_pop']:,}<br>
            <b>Capacity Usage:</b> {stats['capacity_pct']}%<br>
            <b>Rent (TL/m²):</b> {df_n.iloc[j]['rent_per_m2']:.0f}
        </div>
        """
        
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"DC: {name} ({district})",
            icon=folium.Icon(color="red", icon="info-sign"),
        ).add_to(dc_cluster)
        
        # Draw assignment lines
        for i in stats["assigned_indices"]:
            lat_i = df_n.iloc[i]["lat"]
            lon_i = df_n.iloc[i]["lon"]
            folium.PolyLine(
                [[lat_i, lon_i], [lat, lon]],
                color=color,
                weight=1,
                opacity=0.5,
            ).add_to(m)
            
            folium.CircleMarker(
                location=[lat_i, lon_i],
                radius=2,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.3,
                opacity=0.8,
            ).add_to(m)
    
    dc_cluster.add_to(m)
    
    # Add title
    title_html = f"""
    <div style="position: fixed; top: 10px; left: 60px; z-index: 1000;
         background: rgba(0,0,0,0.7); color: white; padding: 10px 15px;
         border-radius: 8px; font-family: Arial; font-size: 14px;">
        <b>Traffic-Aware DC Location — {scenario_name}</b><br>
        <span style="font-size: 12px;">Open DCs: {n_open} | Cost: {result['objective']:,.0f} TL | α={alpha} | Q={Q:,.0f}</span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m, result


print("\n═══ 1. GENERATING INTERACTIVE MAPS ═══")

scenarios = {
    "Peak": t_peak,
    "Off-Peak": t_offpeak,
    "Blended": t_blended,
}

map_results = {}
for scenario_name, t_matrix in scenarios.items():
    m, res = generate_map(t_matrix, scenario_name)
    filename = f"interactive_map_{scenario_name.lower().replace('-', '')}.html"
    path = os.path.join(MAPS_DIR, filename)
    m.save(path)
    map_results[scenario_name] = res
    print(f"   Saved → {path}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. SEARCH COMPARISON CSV (from existing JSON results)
# ══════════════════════════════════════════════════════════════════════════════

print("\n═══ 2. GENERATING SEARCH COMPARISON TABLE ═══")

def load_json(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

res_golden = load_json("search_1d_golden.json")
res_fib = load_json("search_1d_fibonacci.json")
res_nm = load_json("search_2d_nelder_mead.json")
res_grid = load_json("search_2d_grid.json")

if all([res_golden, res_fib, res_nm, res_grid]):
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
    df_comp = pd.DataFrame(rows)
    comp_path = os.path.join(RESULTS_DIR, "search_comparison.csv")
    df_comp.to_csv(comp_path, index=False)
    print(df_comp.to_string(index=False))
    print(f"\n   Saved → {comp_path}")
else:
    print("   WARNING: Some search result JSON files are missing. Skipping.")

# ══════════════════════════════════════════════════════════════════════════════
# 3. PARETO CURVE WITH PEAK / OFF-PEAK / BLENDED COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

print("\n═══ 3. GENERATING PARETO CURVES ═══")

plt.style.use("dark_background")

alphas_pareto = np.concatenate([
    np.linspace(0.1, 1.0, 5),
    np.linspace(1.0, 3.0, 10),
    np.linspace(3.0, 10.0, 8),
])

fig, ax = plt.subplots(figsize=(10, 6))

scenario_colors = {"Peak": "#e74c3c", "Off-Peak": "#3498db", "Blended": "#2ecc71"}
scenario_markers = {"Peak": "^", "Off-Peak": "v", "Blended": "o"}

knee_points = {}

for scenario_name, t_matrix in scenarios.items():
    n_opens = []
    service_costs = []
    
    for a in alphas_pareto:
        res = solve_cflp(w, t_matrix, r, a, Q, Q0, method="greedy")
        _, service_cost = split_costs(res, r, a, Q, Q0)
        n_opens.append(int(res["y"].sum()))
        service_costs.append(service_cost)
    
    # Knee of *this* sweep: L is normalised over the α grid plotted here, which
    # is not the same normaliser the outer-layer search uses (that one is
    # pre-sampled over the whole (α, Q) box). The two therefore land on slightly
    # different α — the star marks the knee of the curve you are looking at,
    # the circle marks the operating point the project actually reports.
    n_opens_arr = np.array(n_opens, dtype=float)
    s_arr = np.array(service_costs, dtype=float)
    n_hat = (n_opens_arr - n_opens_arr.min()) / (n_opens_arr.max() - n_opens_arr.min() + 1e-10)
    s_hat = (s_arr - s_arr.min()) / (s_arr.max() - s_arr.min() + 1e-10)
    L = n_hat + s_hat
    knee_idx = int(np.argmin(L))
    knee_points[scenario_name] = (n_opens[knee_idx], service_costs[knee_idx], alphas_pareto[knee_idx])

    color = scenario_colors[scenario_name]
    marker = scenario_markers[scenario_name]

    ax.scatter(n_opens, service_costs, c=color, marker=marker, s=50, alpha=0.7,
               label=f"{scenario_name}", zorder=3)

    # Sort by n_opens for line
    sort_idx = np.argsort(n_opens)
    ax.plot(np.array(n_opens)[sort_idx], np.array(service_costs)[sort_idx],
            color=color, alpha=0.4, linestyle="--", linewidth=1)

    # Mark knee point
    kx, ky, ka = knee_points[scenario_name]
    ax.scatter([kx], [ky], c=color, marker="*", s=300, edgecolors="white",
               linewidths=1.5, zorder=5, label=f"Sweep knee ({scenario_name}, α={ka:.2f})")

    # Mark the reported operating point on the blended curve
    if scenario_name == "Blended":
        res_op = solve_cflp(w, t_matrix, r, ALPHA, Q, Q0, method="greedy")
        _, service_op = split_costs(res_op, r, ALPHA, Q, Q0)
        ax.scatter([int(res_op["y"].sum())], [service_op], facecolors="none",
                   edgecolors="white", s=260, linewidths=2, zorder=6,
                   label=f"Operating point (α={ALPHA:.2f}, {int(res_op['y'].sum())} DCs)")

ax.set_xlabel("Number of Open DCs", fontsize=12)
ax.set_ylabel("Total Service Cost", fontsize=12)
ax.set_title("Pareto Curve: Service Cost vs Number of Open DCs\n(Peak / Off-Peak / Blended Comparison)", fontsize=13)
ax.legend(fontsize=8, loc="upper right")
ax.grid(alpha=0.2)

plt.tight_layout()
pareto_path = os.path.join(FIGS_DIR, "pareto_curve.png")
fig.savefig(pareto_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"   Saved → {pareto_path}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. ENHANCED 4-PANEL RESULTS SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print("\n═══ 4. GENERATING RESULTS SUMMARY ═══")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor("#1a1a2e")

# ── Panel 1: CBC vs Greedy Runtime ──
ax1 = axes[0, 0]
cbc_csv = os.path.join(RESULTS_DIR, "cbc_vs_greedy.csv")
if os.path.exists(cbc_csv):
    df_cbc = pd.read_csv(cbc_csv)
    # Pivot for grouped bar chart
    methods = df_cbc["method"].unique()
    sizes = sorted(df_cbc["instance_size"].unique())
    x = np.arange(len(sizes))
    width = 0.35
    
    for idx, method in enumerate(["greedy", "cbc"]):
        subset = df_cbc[df_cbc["method"] == method]
        runtimes = []
        valid_x = []
        for i, s in enumerate(sizes):
            row = subset[subset["instance_size"] == s]
            if not row.empty:
                runtimes.append(row["runtime_s"].values[0])
                valid_x.append(i)
        color = "#3498db" if method == "greedy" else "#2ecc71"
        ax1.bar(np.array(valid_x) + idx * width - width/2, runtimes,
                width, label=method, color=color, alpha=0.8)
    
    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels(sizes)
    ax1.set_xlabel("instance_size")
    ax1.set_ylabel("runtime_s")
    ax1.set_title("CBC vs Greedy Runtime (s)", fontsize=12, color="white")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.2)
    ax1.set_facecolor("#16213e")

# ── Panel 2: Robustness Boxplot ──
ax2 = axes[0, 1]
rob_csv = os.path.join(RESULTS_DIR, "robustness.csv")
if os.path.exists(rob_csv):
    df_rob = pd.read_csv(rob_csv)
    deltas = sorted(df_rob["delta"].unique())
    box_data = [df_rob[df_rob["delta"] == d]["jaccard"].values for d in deltas]
    
    bp = ax2.boxplot(box_data, positions=range(len(deltas)),
                     patch_artist=True, widths=0.6)
    colors_box = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71"]
    for patch, color in zip(bp["boxes"], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for element in ["whiskers", "caps", "medians"]:
        for item in bp[element]:
            item.set_color("white")
    for flier in bp["fliers"]:
        flier.set(markerfacecolor="white", markersize=4, alpha=0.5)
    
    ax2.set_xticks(range(len(deltas)))
    ax2.set_xticklabels([str(d) for d in deltas])
    ax2.set_xlabel("delta")
    ax2.set_ylabel("jaccard")
    ax2.set_title("Robustness: Jaccard Similarity under Demand Uncertainty", fontsize=12, color="white")
    ax2.grid(alpha=0.2)
    ax2.set_facecolor("#16213e")

# ── Panel 3: 1D Search Convergence ──
ax3 = axes[1, 0]
if res_golden and res_fib:
    golden_hist = res_golden["history"]
    fib_hist = res_fib["history"]
    
    # The two methods track each other almost exactly, so Fibonacci is drawn as
    # a thin dashed overlay — otherwise it hides the Golden Section curve entirely.
    ax3.plot([h["iter"] for h in golden_hist], [h["L"] for h in golden_hist],
             "o-", color="#f1c40f", markersize=6, linewidth=2,
             label="Golden Section", alpha=0.9, zorder=2)
    ax3.plot([h["iter"] for h in fib_hist], [h["L"] for h in fib_hist],
             "s--", color="#2ecc71", markersize=4, linewidth=1.2,
             label="Fibonacci", alpha=0.95, zorder=3)

    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("L(α)")
    ax3.set_title("1D Search Convergence L(α)", fontsize=12, color="white")
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.2)
    ax3.set_facecolor("#16213e")

# ── Panel 4: Nelder-Mead 2D Search Path ──
ax4 = axes[1, 1]
if res_nm:
    nm_hist = res_nm["history"]
    alphas_nm = [h["alpha"] for h in nm_hist]
    Qs_nm = [h["Q"] for h in nm_hist]
    
    ax4.plot(alphas_nm, Qs_nm, "o-", color="#2ecc71", markersize=4, alpha=0.6, linewidth=1)
    
    # Mark optimum
    ax4.scatter([res_nm["alpha_opt"]], [res_nm["Q_opt"]],
                c="red", s=150, marker="o", zorder=5, label="Optimum", edgecolors="white", linewidths=1.5)
    
    ax4.set_xlabel("Alpha")
    ax4.set_ylabel("Capacity (Q)")
    ax4.set_title("Nelder-Mead 2D Search Path", fontsize=12, color="white")
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.2)
    ax4.set_facecolor("#16213e")

for ax in axes.flat:
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for spine in ax.spines.values():
        spine.set_color("#333333")

plt.tight_layout()
summary_path = os.path.join(FIGS_DIR, "results_summary.png")
fig.savefig(summary_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"   Saved → {summary_path}")

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 60)
print("ALL OUTPUTS GENERATED SUCCESSFULLY")
print("═" * 60)

print("\nMaps:")
for f in sorted(os.listdir(MAPS_DIR)):
    if f.endswith(".html"):
        size = os.path.getsize(os.path.join(MAPS_DIR, f))
        print(f"   {f}  ({size/1024:.1f} KB)")

print("\nFigures:")
for f in sorted(os.listdir(FIGS_DIR)):
    if f.endswith(".png"):
        size = os.path.getsize(os.path.join(FIGS_DIR, f))
        print(f"   {f}  ({size/1024:.1f} KB)")

print("\nResults:")
for f in sorted(os.listdir(RESULTS_DIR)):
    if not f.startswith("."):
        size = os.path.getsize(os.path.join(RESULTS_DIR, f))
        print(f"   {f}  ({size/1024:.1f} KB)")
