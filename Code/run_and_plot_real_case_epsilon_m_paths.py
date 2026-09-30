from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from run_exact_dro_pcvrp_experiments import POMOWarmStart
from run_temporal_holdout_and_m_sensitivity import (
    load_instance,
    prize_coverage,
    solve_for_objective,
)


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "outputs" / "hourly_holdout_250"
OUTPUT_DIR = ROOT / "outputs" / "real_case_epsilon_m_paths"

EPSILON_LEVELS = (0.000, 0.025, 0.050, 0.100, 0.150, 0.200)
BASELINE_EPSILON = 0.10
BASELINE_M = 4000.0
RESOURCE_RATIO = 0.55
VEHICLES = 4
TIME_LIMIT = 10.0
SEEDS = (20260807, 20260808, 20260809)


mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 8,
        "axes.linewidth": 0.55,
    }
)


def precompute_plans(
    pomo: POMOWarmStart,
    max_candidates: int = 4,
) -> dict[int, tuple[list, float]]:
    instance, _, _ = load_instance(
        BASELINE_EPSILON,
        BASELINE_M,
        RESOURCE_RATIO,
        VEHICLES,
    )
    cache: dict[int, tuple[list, float]] = {}
    for seed in SEEDS:
        print(f"[POMO] candidate plans seed={seed}", flush=True)
        started = time.perf_counter()
        cache[seed] = (
            pomo.candidate_plans(instance, seed, max_candidates=max_candidates),
            time.perf_counter() - started,
        )
    return cache


def solve_grid(
    parameter: str,
    levels: tuple[float, ...],
    pomo: POMOWarmStart,
    plan_cache: dict[int, tuple[list, float]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for level in levels:
        epsilon = level if parameter == "epsilon" else BASELINE_EPSILON
        penalty_scale = level if parameter == "M" else BASELINE_M
        instance, _, holdout = load_instance(
            epsilon,
            penalty_scale,
            RESOURCE_RATIO,
            VEHICLES,
        )
        for seed in SEEDS:
            print(f"[{parameter}] {level:g}, seed={seed}", flush=True)
            plans, warm_runtime = plan_cache[seed]
            solution, runtime = solve_for_objective(
                "Exact DRO",
                instance,
                pomo,
                seed,
                TIME_LIMIT,
                plans,
                warm_runtime,
            )
            training_coverage = 100.0 * prize_coverage(
                solution.visited, instance.samples
            ).mean()
            holdout_coverage = 100.0 * prize_coverage(
                solution.visited, holdout
            ).mean()
            rows.append(
                {
                    "parameter": parameter,
                    "level": level,
                    "epsilon": epsilon,
                    "M": penalty_scale,
                    "seed": seed,
                    "objective": solution.objective,
                    "route_cost": solution.travel,
                    "robust_penalty": solution.robust_penalty,
                    "visited_nodes": len(solution.visited),
                    "used_vehicles": sum(bool(route) for route in solution.routes),
                    "training_prize_coverage_pct": training_coverage,
                    "holdout_prize_coverage_pct": holdout_coverage,
                    "runtime": runtime,
                    "routes": json.dumps(solution.routes),
                }
            )
            pd.DataFrame(rows).to_csv(
                OUTPUT_DIR / f"{parameter}_route_runs.csv", index=False
            )
    records = pd.DataFrame(rows)
    best = (
        records.sort_values(["level", "objective", "seed"])
        .groupby("level", as_index=False)
        .first()
        .sort_values("level")
    )
    best.to_csv(OUTPUT_DIR / f"{parameter}_selected_routes.csv", index=False)
    return best


def density_background(
    centers: pd.DataFrame,
    extent: tuple[float, float, float, float],
    bins: int = 360,
) -> np.ndarray:
    counts, _, _ = np.histogram2d(
        centers["latitude"].to_numpy(float),
        centers["longitude"].to_numpy(float),
        bins=bins,
        range=[[extent[2], extent[3]], [extent[0], extent[1]]],
        weights=centers["sampled_count"].to_numpy(float),
    )
    smoothed = gaussian_filter(counts, sigma=4.2)
    if smoothed.max() > 0:
        smoothed /= smoothed.max()
    return smoothed


def draw_grid(best: pd.DataFrame, parameter: str, filename: str) -> None:
    candidates = pd.read_csv(DATA_DIR / "candidate_nodes.csv")
    centers = pd.read_csv(DATA_DIR / "hourly_cluster_centers.csv")
    depot = np.array([104.0665, 30.6570])

    lon = candidates["longitude"].to_numpy(float)
    lat = candidates["latitude"].to_numpy(float)
    x_pad = 0.035 * (lon.max() - lon.min())
    y_pad = 0.035 * (lat.max() - lat.min())
    extent = (
        lon.min() - x_pad,
        lon.max() + x_pad,
        lat.min() - y_pad,
        lat.max() + y_pad,
    )
    visible_centers = centers[
        centers["longitude"].between(extent[0], extent[1])
        & centers["latitude"].between(extent[2], extent[3])
    ].copy()
    density = density_background(visible_centers, extent)

    fig, axes = plt.subplots(2, 3, figsize=(10.0, 7.15), constrained_layout=False)
    plt.subplots_adjust(left=0.035, right=0.985, top=0.965, bottom=0.105, wspace=0.035, hspace=0.090)
    letters = "abcdef"

    for panel, (ax, (_, row)) in enumerate(zip(axes.flat, best.iterrows())):
        ax.scatter(
            visible_centers["longitude"],
            visible_centers["latitude"],
            s=1.0,
            c="#a8aaad",
            alpha=0.34,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )
        masked = np.ma.masked_less_equal(density, 0.015)
        ax.imshow(
            masked,
            origin="lower",
            extent=extent,
            cmap="YlOrRd",
            norm=Normalize(0.015, 0.82),
            alpha=np.clip(masked * 1.25, 0, 0.88),
            interpolation="bilinear",
            aspect="auto",
            zorder=2,
        )

        routes = json.loads(row["routes"])
        for route in routes:
            if not route:
                continue
            indices = np.asarray(route, dtype=int)
            route_lon = np.r_[depot[0], lon[indices], depot[0]]
            route_lat = np.r_[depot[1], lat[indices], depot[1]]
            ax.plot(
                route_lon,
                route_lat,
                color="#1565c0",
                linewidth=0.75,
                alpha=0.88,
                zorder=4,
            )
            ax.scatter(
                lon[indices],
                lat[indices],
                s=6.5,
                color="#1565c0",
                edgecolors="none",
                zorder=5,
            )
        ax.scatter(
            depot[0],
            depot[1],
            marker="*",
            s=52,
            facecolor="white",
            edgecolor="black",
            linewidth=0.8,
            zorder=7,
        )

        label = (
            rf"$\epsilon={row['level']:.3f}$"
            if parameter == "epsilon"
            else rf"$M={row['level']:.0f}$"
        )
        route_cost = float(row["route_cost"])
        title = (
            f"{label} ({int(row['used_vehicles'])} vehicles, "
            f"{int(row['visited_nodes'])} nodes; route cost: {route_cost:,.0f})"
        )
        ax.set_title(title, fontsize=8.3, fontweight="bold", pad=4)
        ax.text(
            0.018,
            0.975,
            f"({letters[panel]})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            fontweight="bold",
            zorder=8,
        )
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#6b6b6b")
            spine.set_linewidth(0.55)

    legend_handles = [
        Line2D([], [], marker="o", linestyle="None", markersize=3.2, color="#a8aaad", label="Period-level cluster centers"),
        Line2D([], [], color="#1565c0", marker="o", markersize=3.2, linewidth=0.9, label="Optimized routes"),
        Line2D([], [], marker="*", linestyle="None", markersize=7, markerfacecolor="white", markeredgecolor="black", label="Depot"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.035),
        columnspacing=2.0,
        handletextpad=0.6,
    )
    color_ax = fig.add_axes([0.395, 0.018, 0.21, 0.012])
    gradient = np.linspace(0.015, 0.82, 256).reshape(1, -1)
    color_ax.imshow(gradient, aspect="auto", cmap="YlOrRd", origin="lower")
    color_ax.set_xticks([0, 128, 255], ["Low", "Medium", "High"], fontsize=6)
    color_ax.set_yticks([])
    color_ax.set_xlabel("Observed activity intensity", fontsize=7, labelpad=1)
    for spine in color_ax.spines.values():
        spine.set_linewidth(0.4)

    for suffix, kwargs in {
        "png": {"dpi": 500},
        "pdf": {},
        "svg": {},
    }.items():
        fig.savefig(
            OUTPUT_DIR / f"{filename}.{suffix}",
            bbox_inches="tight",
            facecolor="white",
            **kwargs,
        )
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[POMO] loading checkpoint", flush=True)
    pomo = POMOWarmStart()
    plan_cache = precompute_plans(pomo)
    epsilon_best = solve_grid("epsilon", EPSILON_LEVELS, pomo, plan_cache)
    draw_grid(epsilon_best, "epsilon", "chengdu_routes_by_epsilon")
    metadata = {
        "training_hours": 168,
        "candidate_nodes": 250,
        "epsilon_levels": EPSILON_LEVELS,
        "baseline_epsilon": BASELINE_EPSILON,
        "baseline_M": BASELINE_M,
        "resource_ratio": RESOURCE_RATIO,
        "vehicles": VEHICLES,
        "time_limit_seconds_per_run": TIME_LIMIT,
        "seeds": SEEDS,
        "selection_rule": "minimum exact DRO objective among three seeds",
    }
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("\nSelected epsilon routes", flush=True)
    print(epsilon_best.drop(columns="routes").to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
