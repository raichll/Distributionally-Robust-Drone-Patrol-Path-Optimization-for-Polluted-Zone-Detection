from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

import calibrate_epsilon_diverse_pomo as experiment
from calibrate_temporal_dro_parameters import BASE_SEED, load_arrays, make_instance
from run_exact_dro_pcvrp_experiments import POMOWarmStart


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "outputs" / "hourly_holdout_250_train120"
OUTPUT_DIR = ROOT / "outputs" / "epsilon_strict_kuhn_holdout_132_36_M4000"
EPSILON_LEVELS = (0.05, 0.10, 0.20, 0.25, 0.35, 0.50)

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.linewidth": 0.55,
    }
)


def density_background(
    centers: pd.DataFrame,
    extent: tuple[float, float, float, float],
) -> np.ndarray:
    counts, _, _ = np.histogram2d(
        centers["latitude"].to_numpy(float),
        centers["longitude"].to_numpy(float),
        bins=360,
        range=[[extent[2], extent[3]], [extent[0], extent[1]]],
        weights=centers["sampled_count"].to_numpy(float),
    )
    smoothed = gaussian_filter(counts, sigma=4.2)
    if smoothed.max() > 0:
        smoothed /= smoothed.max()
    return smoothed


def draw(results: pd.DataFrame, candidates: pd.DataFrame) -> None:
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
    visible = centers[
        centers["longitude"].between(extent[0], extent[1])
        & centers["latitude"].between(extent[2], extent[3])
    ]
    density = density_background(visible, extent)
    masked = np.ma.masked_less_equal(density, 0.015)

    fig, axes = plt.subplots(2, 3, figsize=(10.0, 7.15))
    plt.subplots_adjust(
        left=0.035, right=0.985, top=0.965, bottom=0.105,
        wspace=0.035, hspace=0.090,
    )
    for panel, (ax, row) in enumerate(zip(axes.flat, results.itertuples())):
        ax.scatter(
            visible["longitude"], visible["latitude"], s=1.0,
            c="#a8aaad", alpha=0.34, linewidths=0, rasterized=True, zorder=1,
        )
        ax.imshow(
            masked, origin="lower", extent=extent, cmap="YlOrRd",
            norm=Normalize(0.015, 0.82), alpha=np.clip(masked * 1.25, 0, 0.88),
            interpolation="bilinear", aspect="auto", zorder=2,
        )
        routes = json.loads(row.routes)
        for route in routes:
            if not route:
                continue
            indices = np.asarray(route, dtype=int)
            ax.plot(
                np.r_[depot[0], lon[indices], depot[0]],
                np.r_[depot[1], lat[indices], depot[1]],
                color="#1565c0", linewidth=0.75, alpha=0.88, zorder=4,
            )
            ax.scatter(lon[indices], lat[indices], s=6.5, color="#1565c0", zorder=5)
        ax.scatter(
            depot[0], depot[1], marker="*", s=52, facecolor="white",
            edgecolor="black", linewidth=0.8, zorder=7,
        )
        suffix = " (holdout-selected)" if np.isclose(row.epsilon, 0.25) else ""
        ax.set_title(
            rf"$\epsilon={row.epsilon:.2f}$" + suffix,
            fontsize=8.3, fontweight="bold", pad=4,
        )
        ax.text(
            0.018, 0.975, f"({'abcdef'[panel]})", transform=ax.transAxes,
            ha="left", va="top", fontsize=8.5, fontweight="bold", zorder=8,
        )
        ax.set(xlim=extent[:2], ylim=extent[2:])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#6b6b6b")
            spine.set_linewidth(0.55)

    fig.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="None", markersize=3.2,
                   color="#a8aaad", label="Period-level cluster centers"),
            Line2D([], [], color="#1565c0", marker="o", markersize=3.2,
                   linewidth=0.9, label="Optimized routes"),
            Line2D([], [], marker="*", linestyle="None", markersize=7,
                   markerfacecolor="white", markeredgecolor="black", label="Depot"),
        ],
        loc="lower center", ncol=3, frameon=False,
        bbox_to_anchor=(0.5, 0.035), columnspacing=2.0, handletextpad=0.6,
    )
    fig.savefig(ROOT / "chengdu_routes_by_epsilon.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "chengdu_routes_by_epsilon.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    candidates, _, _ = load_arrays(DATA_DIR)
    validation = pd.read_csv(OUTPUT_DIR / "fixed_holdout_validation_runs.csv")
    results = validation[validation["epsilon"].isin(EPSILON_LEVELS)].copy()
    results = results.sort_values("epsilon")
    results["used_vehicles"] = results["routes"].map(
        lambda value: sum(bool(route) for route in json.loads(value))
    )
    results = results.rename(columns={"training_objective": "objective"})
    results.to_csv(OUTPUT_DIR / "epsilon_route_figure_results.csv", index=False)
    draw(results, candidates)


if __name__ == "__main__":
    main()
