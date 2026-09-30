from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_and_plot_real_case_epsilon_m_paths import (
    BASELINE_EPSILON,
    OUTPUT_DIR,
    RESOURCE_RATIO,
    SEEDS,
    VEHICLES,
    precompute_plans,
)
from run_exact_dro_pcvrp_experiments import (
    ExactDROEvaluator,
    POMOWarmStart,
    Solution,
    construct_solution,
    evaluate_solution,
    mns_from_start,
)
from run_temporal_holdout_and_m_sensitivity import load_instance, prize_coverage


ROOT = Path(__file__).resolve().parent
M_LEVELS = np.unique(
    np.concatenate(
        [
            np.asarray(
                [0, 1000, 2000, 2500, 2750, 2900, 2925, 2950, 2975],
                dtype=float,
            ),
            np.arange(3000.0, 4000.0 + 25.0, 25.0),
            np.asarray(
                [
                    4250,
                    4500,
                    5000,
                    6000,
                    6250,
                    6500,
                    6750,
                    7000,
                    7500,
                    10000,
                    12500,
                    15000,
                    16000,
                    17000,
                    18000,
                    20000,
                    30000,
                    50000,
                    75000,
                    100000,
                ],
                dtype=float,
            ),
        ]
    )
)
MNS_SECONDS_PER_RESTART = 10.0
POMO_CANDIDATES_PER_SEED = 16
SELECTED_M = 4000.0
RESTARTS_PATH = OUTPUT_DIR / "M_mns_dense_extended_restarts.csv"
BEST_PATH = OUTPUT_DIR / "M_mns_dense_extended_best.csv"
FRONTIER_PATH = OUTPUT_DIR / "M_mns_dense_extended_frontier.csv"
METADATA_PATH = OUTPUT_DIR / "M_mns_dense_extended_metadata.json"


mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 9,
        "axes.linewidth": 0.7,
    }
)


def experiment_config(common_pool_size: int | None = None) -> dict[str, object]:
    return {
        "experiment": "dense_and_extended_M_optimization_with_common_warm_start_and_full_MNS_runs",
        "M_levels": [float(value) for value in M_LEVELS],
        "M_grid": "step 25 from 3000 to 4000, with coarse levels below and extended levels through 100000",
        "epsilon": BASELINE_EPSILON,
        "resource_ratio": RESOURCE_RATIO,
        "vehicles": VEHICLES,
        "restart_seeds": list(SEEDS),
        "combined_unique_POMO_plan_count": common_pool_size,
        "POMO_candidates_per_seed": POMO_CANDIDATES_PER_SEED,
        "MNS_seconds_per_restart": MNS_SECONDS_PER_RESTART,
        "selection_rule": (
            "At each M, select the best member of the common expanded POMO candidate "
            "pool and apply deterministic insertion completion once. Give the resulting "
            "identical warm start to every seed, run a full equal-time MNS refinement "
            "with a separate exact evaluator, "
            "compare the result with the empty solution, and retain the lowest "
            "exact-DRO objective across the three independent full-pipeline runs."
        ),
    }


def common_plan_pool(
    plan_cache: dict[int, tuple[list[list[list[int]]], float]],
) -> list[list[list[int]]]:
    plans: list[list[list[int]]] = []
    seen: set[str] = set()
    for seed in SEEDS:
        for plan in plan_cache[seed][0]:
            signature = json.dumps(plan, separators=(",", ":"))
            if signature not in seen:
                seen.add(signature)
                plans.append(plan)
    return plans


def load_completed_restarts() -> pd.DataFrame:
    if not RESTARTS_PATH.exists():
        return pd.DataFrame()
    records = pd.read_csv(RESTARTS_PATH)
    required = {"M", "seed", "objective", "routes"}
    if not required.issubset(records.columns):
        raise ValueError(f"Unexpected columns in {RESTARTS_PATH}")
    return records[
        records["M"].isin(M_LEVELS) & records["seed"].isin(SEEDS)
    ].copy()


def save_metadata(
    started_at: float,
    records: pd.DataFrame,
    status: str,
    common_pool_size: int | None,
) -> None:
    metadata = experiment_config(common_pool_size)
    metadata.update(
        {
            "status": status,
            "completed_restarts": int(len(records)),
            "completed_M_levels": int(records.groupby("M").size().eq(len(SEEDS)).sum())
            if not records.empty
            else 0,
            "elapsed_wall_seconds_current_run": time.perf_counter() - started_at,
            "POMO_plans_precomputed_once": True,
            "warm_pool_shared_within_each_M": True,
            "same_initial_solution_for_all_restarts": True,
            "solutions_shared_across_M": False,
            "holdout_used_for_calibration": False,
        }
    )
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def prepare_warm_start_templates(
    instance,
    pomo: POMOWarmStart,
    plans: list[list[list[int]]],
    evaluator: ExactDROEvaluator,
) -> list[Solution]:
    ranked: dict[str, Solution] = {}
    for plan in plans:
        candidate = pomo._prune_to_budget(plan, instance, evaluator)
        signature = json.dumps(candidate.routes, separators=(",", ":"))
        previous = ranked.get(signature)
        if previous is None or candidate.objective < previous.objective:
            ranked[signature] = candidate
    return list(ranked.values())


def rank_warm_start_templates(
    templates: list[Solution], penalty_scale: float
) -> list[Solution]:
    return sorted(
        templates,
        key=lambda solution: solution.travel + penalty_scale * solution.robust_penalty,
    )


def solve_restart(
    instance,
    evaluator: ExactDROEvaluator,
    empty: Solution,
    initial: Solution,
    seed: int,
    time_limit: float,
) -> tuple[Solution, float, dict[str, float | int | bool]]:
    if instance.penalty_scale == 0.0:
        return empty.clone(), 0.0, {
            "entered": False,
            "iterations": 0,
            "improvements": 0,
        }

    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    deadline = started + time_limit
    stats: dict[str, float | int | bool] = {}
    candidate = initial.clone()
    evaluate_solution(candidate, instance, evaluator)
    candidate = mns_from_start(
        candidate,
        instance,
        evaluator,
        rng,
        deadline,
        stats=stats,
    )
    if empty.objective < candidate.objective:
        candidate = empty.clone()
    return candidate, time.perf_counter() - started, stats


def run_independent_levels() -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started_at = time.perf_counter()
    records = load_completed_restarts()
    completed = (
        {
            (float(row.M), int(row.seed))
            for row in records[["M", "seed"]].itertuples(index=False)
        }
        if not records.empty
        else set()
    )
    missing = [
        (float(penalty), int(seed))
        for penalty in M_LEVELS
        for seed in SEEDS
        if (float(penalty), int(seed)) not in completed
    ]
    if not missing:
        save_metadata(started_at, records, "complete", None)
        return records.sort_values(["M", "objective", "seed"]).reset_index(drop=True)

    print(
        f"Running {len(M_LEVELS)} independent M levels with a separate "
        f"{MNS_SECONDS_PER_RESTART:g}-s budget for each of {len(SEEDS)} full POMO-NS runs per level",
        flush=True,
    )
    pomo = POMOWarmStart()
    plan_cache = precompute_plans(
        pomo, max_candidates=POMO_CANDIDATES_PER_SEED
    )
    plans = common_plan_pool(plan_cache)
    print(f"Combined POMO warm-start pool: {len(plans)} unique plans", flush=True)
    template_instance, _, _ = load_instance(
        BASELINE_EPSILON,
        1.0,
        RESOURCE_RATIO,
        VEHICLES,
    )
    template_evaluator = ExactDROEvaluator(
        template_instance.samples,
        epsilon=template_instance.epsilon,
        penalty_scale=1.0,
    )
    templates = prepare_warm_start_templates(
        template_instance,
        pomo,
        plans,
        template_evaluator,
    )
    print(f"Budget-feasible common POMO templates: {len(templates)}", flush=True)
    rows = records.to_dict("records")

    for level_index, penalty in enumerate(M_LEVELS, start=1):
        penalty = float(penalty)
        needed_seeds = [
            int(seed) for seed in SEEDS if (penalty, int(seed)) not in completed
        ]
        if not needed_seeds:
            continue
        level_started = time.perf_counter()
        instance, _, holdout = load_instance(
            BASELINE_EPSILON,
            penalty,
            RESOURCE_RATIO,
            VEHICLES,
        )
        preprocessing_runtime = time.perf_counter() - level_started
        print(
            f"[M {level_index:03d}/{len(M_LEVELS)}] M={penalty:g}; "
            f"base prep={preprocessing_runtime:.2f}s",
            flush=True,
        )

        initialization_started = time.perf_counter()
        setup_evaluator = ExactDROEvaluator(
            instance.samples,
            epsilon=instance.epsilon,
            penalty_scale=instance.penalty_scale,
        )
        setup_empty = Solution(routes=[[] for _ in range(instance.vehicles)])
        evaluate_solution(setup_empty, instance, setup_evaluator)
        ranked_templates = rank_warm_start_templates(templates, penalty)
        if ranked_templates:
            initial = evaluate_solution(
                ranked_templates[0].clone(), instance, setup_evaluator
            )
            if setup_empty.objective < initial.objective:
                initial = setup_empty.clone()
            else:
                initial = construct_solution(
                    instance,
                    setup_evaluator,
                    np.random.default_rng(0),
                    randomized=False,
                    initial=initial,
                )
        else:
            initial = setup_empty.clone()
        initialization_runtime = time.perf_counter() - initialization_started

        for seed in needed_seeds:
            restart_evaluator = ExactDROEvaluator(
                instance.samples,
                epsilon=instance.epsilon,
                penalty_scale=instance.penalty_scale,
            )
            restart_empty = Solution(routes=[[] for _ in range(instance.vehicles)])
            evaluate_solution(restart_empty, instance, restart_evaluator)
            solution, search_runtime, search_stats = solve_restart(
                instance,
                restart_evaluator,
                restart_empty,
                initial,
                seed,
                MNS_SECONDS_PER_RESTART,
            )
            rows.append(
                {
                    "M": penalty,
                    "epsilon": BASELINE_EPSILON,
                    "seed": seed,
                    "initial_rank": 1,
                    "common_pool_size": len(plans),
                    "pomo_candidate_pool_size": len(plans),
                    "feasible_template_count": len(templates),
                    "initial_objective": initial.objective,
                    "initial_route_cost": initial.travel,
                    "initial_robust_penalty": initial.robust_penalty,
                    "initial_visited_nodes": len(initial.visited),
                    "objective": solution.objective,
                    "route_cost": solution.travel,
                    "robust_penalty": solution.robust_penalty,
                    "visited_nodes": len(solution.visited),
                    "used_vehicles": sum(bool(route) for route in solution.routes),
                    "training_prize_coverage_pct": 100.0
                    * prize_coverage(solution.visited, instance.samples).mean(),
                    "holdout_prize_coverage_pct": 100.0
                    * prize_coverage(solution.visited, holdout).mean(),
                    "level_preprocessing_runtime": preprocessing_runtime,
                    "seed_initialization_runtime": initialization_runtime,
                    "allocated_search_seconds": MNS_SECONDS_PER_RESTART,
                    "search_runtime": search_runtime,
                    "mns_entered": search_stats.get("entered", False),
                    "mns_iterations": search_stats.get("iterations", 0),
                    "mns_improvements": search_stats.get("improvements", 0),
                    "mns_initial_vnd_seconds": search_stats.get(
                        "initial_vnd_seconds", 0.0
                    ),
                    "level_elapsed_after_restart": time.perf_counter() - level_started,
                    "routes": json.dumps(solution.routes),
                }
            )
            completed.add((penalty, seed))
            records = pd.DataFrame(rows).sort_values(["M", "objective", "seed"])
            records.to_csv(RESTARTS_PATH, index=False)
        save_metadata(started_at, records, "running", len(plans))

    records = pd.DataFrame(rows).sort_values(["M", "objective", "seed"])
    records.to_csv(RESTARTS_PATH, index=False)
    save_metadata(started_at, records, "complete", len(plans))
    return records.reset_index(drop=True)


def select_best_by_M(records: pd.DataFrame) -> pd.DataFrame:
    counts = records.groupby("M").size()
    incomplete = counts[counts < len(SEEDS)]
    if not incomplete.empty:
        raise RuntimeError(f"Incomplete M levels: {incomplete.to_dict()}")
    best = (
        records.sort_values(["M", "objective", "seed"])
        .groupby("M", as_index=False)
        .first()
        .sort_values("M")
        .reset_index(drop=True)
    )
    best["search_seconds_per_restart"] = MNS_SECONDS_PER_RESTART
    best.to_csv(BEST_PATH, index=False)
    return best


def pareto_frontier(records: pd.DataFrame) -> pd.DataFrame:
    candidates = records[
        (records["route_cost"] > 0)
        & (records["training_prize_coverage_pct"] >= 50.0)
    ].copy()
    candidates = candidates.drop_duplicates(
        subset=["route_cost", "training_prize_coverage_pct"]
    ).sort_values(
        ["route_cost", "training_prize_coverage_pct"], ascending=[True, False]
    )
    rows: list[dict[str, object]] = []
    highest_coverage = -np.inf
    for row in candidates.itertuples(index=False):
        if row.training_prize_coverage_pct > highest_coverage + 1e-9:
            rows.append(row._asdict())
            highest_coverage = row.training_prize_coverage_pct
    frontier = pd.DataFrame(rows)
    frontier.to_csv(FRONTIER_PATH, index=False)
    return frontier


def draw_results(
    records: pd.DataFrame,
    best: pd.DataFrame,
    frontier: pd.DataFrame,
) -> None:
    ranges = (
        records.groupby("M", as_index=False)
        .agg(
            route_cost_min=("route_cost", "min"),
            route_cost_max=("route_cost", "max"),
        )
        .sort_values("M")
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.05))
    plt.subplots_adjust(left=0.075, right=0.985, bottom=0.19, top=0.94, wspace=0.24)

    cost_ax = axes[0]
    cost_ax.fill_between(
        ranges["M"].to_numpy(float),
        ranges["route_cost_min"].to_numpy(float) / 1000.0,
        ranges["route_cost_max"].to_numpy(float) / 1000.0,
        color="#90caf9",
        alpha=0.28,
        linewidth=0,
        label="Routing-cost range across multistarts",
        zorder=1,
    )
    cost_ax.plot(
        best["M"],
        best["objective"] / 1000.0,
        color="#202124",
        linewidth=1.65,
        label="Best-found total objective",
        zorder=5,
    )
    cost_ax.plot(
        best["M"],
        best["route_cost"] / 1000.0,
        color="#1565c0",
        linewidth=1.35,
        label="Best-found routing cost",
        zorder=4,
    )
    cost_ax.plot(
        best["M"],
        best["robust_penalty"] / 1000.0,
        color="#d95f02",
        linewidth=1.35,
        label="Worst-case missed-prize penalty",
        zorder=3,
    )
    selected = best[np.isclose(best["M"], SELECTED_M)].iloc[0]
    cost_ax.axvline(SELECTED_M, color="#777777", linewidth=0.8, linestyle="--")
    cost_ax.scatter(
        [SELECTED_M],
        [selected["objective"] / 1000.0],
        marker="*",
        s=105,
        color="#2e7d32",
        edgecolor="white",
        linewidth=0.6,
        zorder=7,
    )
    cost_ax.annotate(
        r"Selected $M=4000$",
        xy=(SELECTED_M, selected["objective"] / 1000.0),
        xytext=(9, 10),
        textcoords="offset points",
        fontsize=7.8,
        color="#245c28",
    )
    cost_ax.set_xscale("symlog", linthresh=5000, linscale=1.0)
    cost_ax.set_xlim(0, float(M_LEVELS.max()) * 1.05)
    cost_ax.set_xticks([0, 3000, 4000, 10000, 30000, 100000])
    cost_ax.set_xticklabels(["0", "3000", "4000", "10000", "30000", "100000"])
    cost_ax.set_ylim(bottom=0)
    cost_ax.set_xlabel(r"Missed-prize coefficient $M$")
    cost_ax.set_ylabel(r"Cost ($\times 10^3$)")
    cost_ax.set_title("Parametric cost decomposition", fontsize=9.5, pad=5)
    cost_ax.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=6.7)

    pareto_ax = axes[1]
    service = records[
        (records["route_cost"] > 0)
        & (records["training_prize_coverage_pct"] >= 50.0)
    ]
    pareto_ax.scatter(
        service["route_cost"] / 1000.0,
        service["training_prize_coverage_pct"],
        color="#8e8e93",
        s=22,
        alpha=0.35,
        edgecolors="white",
        linewidths=0.3,
        label="Within-$M$ multistart solutions",
        zorder=2,
    )
    if not frontier.empty:
        pareto_ax.plot(
            frontier["route_cost"] / 1000.0,
            frontier["training_prize_coverage_pct"],
            color="#c62828",
            linewidth=1.55,
            marker="o",
            markersize=4.0,
            markerfacecolor="white",
            markeredgewidth=1.0,
            label="Observed nondominated frontier",
            zorder=4,
        )
    pareto_ax.scatter(
        selected["route_cost"] / 1000.0,
        selected["training_prize_coverage_pct"],
        marker="*",
        s=120,
        color="#1565c0",
        edgecolor="white",
        linewidth=0.7,
        label=r"Selected setting ($M=4000$)",
        zorder=6,
    )
    pareto_ax.set_xlabel(r"Routing cost ($\times 10^3$)")
    pareto_ax.set_ylabel("Training-period prize coverage (%)")
    pareto_ax.set_title("Observed cost--coverage frontier", fontsize=9.5, pad=5)
    pareto_ax.annotate(
        r"$M=4000$",
        xy=(selected["route_cost"] / 1000.0, selected["training_prize_coverage_pct"]),
        xytext=(8, -15),
        textcoords="offset points",
        fontsize=8,
        color="#0d47a1",
    )
    pareto_ax.legend(loc="lower right", frameon=True, framealpha=0.95, fontsize=7.0)

    for panel, ax in enumerate(axes):
        ax.grid(True, color="#d7d9dc", linewidth=0.55, alpha=0.85)
        ax.set_axisbelow(True)
        panel_x = 0.98 if panel == 0 else 0.02
        ax.text(
            panel_x,
            0.97,
            f"({'ab'[panel]})",
            transform=ax.transAxes,
            ha="right" if panel == 0 else "left",
            va="top",
            fontsize=9,
            fontweight="bold",
        )

    for directory in (OUTPUT_DIR, ROOT):
        for suffix, kwargs in {
            "png": {"dpi": 500},
            "pdf": {},
            "svg": {},
        }.items():
            fig.savefig(
                directory / f"M_cost_coverage_pareto_mns_dense_extended.{suffix}",
                bbox_inches="tight",
                facecolor="white",
                **kwargs,
            )
    plt.close(fig)


def main() -> None:
    records = run_independent_levels()
    best = select_best_by_M(records)
    frontier = pareto_frontier(records)
    draw_results(records, best, frontier)
    service = best[best["used_vehicles"] > 0]
    first_service_M = float(service["M"].min()) if not service.empty else np.nan
    selected = best[np.isclose(best["M"], SELECTED_M)].iloc[0]
    print(
        json.dumps(
            {
                "M_levels": len(best),
                "restart_runs": len(records),
                "first_service_M": first_service_M,
                "selected_M": SELECTED_M,
                "selected_objective": float(selected["objective"]),
                "selected_route_cost": float(selected["route_cost"]),
                "selected_training_coverage_pct": float(
                    selected["training_prize_coverage_pct"]
                ),
                "frontier_points": len(frontier),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
