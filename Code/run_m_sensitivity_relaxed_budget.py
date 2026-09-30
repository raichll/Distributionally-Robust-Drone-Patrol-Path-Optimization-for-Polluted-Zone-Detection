from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_and_plot_m_pareto import (
    common_plan_pool,
    prepare_warm_start_templates,
    solve_restart,
)
from run_and_plot_real_case_epsilon_m_paths import (
    BASELINE_EPSILON,
    VEHICLES,
    precompute_plans,
)
from run_exact_dro_pcvrp_experiments import (
    ExactDROEvaluator,
    POMOWarmStart,
    Solution,
    balanced_full_routes,
    evaluate_solution,
)
from run_temporal_holdout_and_m_sensitivity import load_instance, prize_coverage


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "real_case_m_relaxed_budget_R1"
RESTARTS_PATH = OUTPUT_DIR / "M_R1_restarts_30s.csv"
BEST_PATH = OUTPUT_DIR / "M_R1_best_30s.csv"
METADATA_PATH = OUTPUT_DIR / "M_R1_metadata.json"
FIGURE_PATH = ROOT / "M_sensitivity_relaxed_budget_R1_30s"

RESOURCE_RATIO = 1.0
SEARCH_SECONDS = 30.0
SEEDS = (20260831, 20260832, 20260833)
M_LEVELS = np.asarray(
    [
        0,
        1000,
        2000,
        2500,
        3000,
        3500,
        4000,
        5000,
        6000,
        7000,
        8000,
        9000,
        10000,
        10500,
        11000,
        11500,
        12000,
        12500,
        13000,
        13500,
        14000,
        15000,
        17500,
        20000,
        21000,
        22000,
        22500,
        23000,
        24000,
        25000,
        30000,
        50000,
        100000,
    ],
    dtype=float,
)


def route_signature(solution: Solution) -> str:
    return json.dumps(solution.routes, separators=(",", ":"))


def solution_from_json(routes: str) -> Solution:
    return Solution(routes=[[int(node) for node in route] for route in json.loads(routes)])


def distinct_ranked_starts(
    templates: list[Solution], full_solution: Solution, penalty: float
) -> list[Solution]:
    candidates = templates + [full_solution]
    ranked = sorted(
        candidates,
        key=lambda solution: solution.travel + penalty * solution.robust_penalty,
    )
    starts: list[Solution] = []
    seen: set[str] = set()
    for solution in ranked:
        signature = route_signature(solution)
        if signature not in seen:
            seen.add(signature)
            starts.append(solution)
        if len(starts) == len(SEEDS):
            break
    return starts


def run_restarts() -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    completed = pd.read_csv(RESTARTS_PATH) if RESTARTS_PATH.exists() else pd.DataFrame()
    done = (
        {(float(row.M), int(row.seed)) for row in completed.itertuples()}
        if not completed.empty
        else set()
    )
    rows = completed.to_dict("records") if not completed.empty else []

    unit_instance, _, _ = load_instance(
        BASELINE_EPSILON, 1.0, RESOURCE_RATIO, VEHICLES
    )
    unit_evaluator = ExactDROEvaluator(
        unit_instance.samples, unit_instance.epsilon, 1.0
    )
    full_solution = Solution(routes=balanced_full_routes(unit_instance))
    evaluate_solution(full_solution, unit_instance, unit_evaluator)
    if full_solution.robust_penalty > 1e-8:
        raise RuntimeError("The reference full-coverage solution must have zero penalty")

    pomo = POMOWarmStart()
    plans = common_plan_pool(precompute_plans(pomo, max_candidates=16))
    templates = prepare_warm_start_templates(
        unit_instance, pomo, plans, unit_evaluator
    )
    templates.append(full_solution)

    for level_index, penalty in enumerate(M_LEVELS, start=1):
        needed = [seed for seed in SEEDS if (float(penalty), seed) not in done]
        if not needed:
            continue
        instance, _, holdout = load_instance(
            BASELINE_EPSILON, float(penalty), RESOURCE_RATIO, VEHICLES
        )
        setup_evaluator = ExactDROEvaluator(
            instance.samples, instance.epsilon, instance.penalty_scale
        )
        empty = Solution(routes=[[] for _ in range(instance.vehicles)])
        evaluate_solution(empty, instance, setup_evaluator)
        starts = distinct_ranked_starts(templates, full_solution, float(penalty))
        print(
            f"[R=1.0 {level_index:02d}/{len(M_LEVELS)}] M={penalty:g}; "
            f"missing restarts={len(needed)}",
            flush=True,
        )

        for seed, unit_start in zip(SEEDS, starts):
            if seed not in needed:
                continue
            evaluator = ExactDROEvaluator(
                instance.samples, instance.epsilon, instance.penalty_scale
            )
            restart_empty = Solution(routes=[[] for _ in range(instance.vehicles)])
            evaluate_solution(restart_empty, instance, evaluator)
            initial = solution_from_json(route_signature(unit_start))
            evaluate_solution(initial, instance, evaluator)
            if restart_empty.objective < initial.objective:
                initial = restart_empty.clone()
            solution, runtime, stats = solve_restart(
                instance,
                evaluator,
                restart_empty,
                initial,
                seed,
                SEARCH_SECONDS,
            )
            rows.append(
                {
                    "M": float(penalty),
                    "seed": seed,
                    "objective": solution.objective,
                    "route_cost": solution.travel,
                    "robust_penalty": solution.robust_penalty,
                    "visited_nodes": len(solution.visited),
                    "used_vehicles": sum(bool(route) for route in solution.routes),
                    "training_prize_coverage_pct": 100.0
                    * prize_coverage(solution.visited, instance.samples).mean(),
                    "holdout_prize_coverage_pct": 100.0
                    * prize_coverage(solution.visited, holdout).mean(),
                    "search_runtime": runtime,
                    "mns_iterations": stats.get("iterations", 0),
                    "mns_improvements": stats.get("improvements", 0),
                    "routes": route_signature(solution),
                }
            )
            done.add((float(penalty), seed))
            pd.DataFrame(rows).to_csv(RESTARTS_PATH, index=False)
    return pd.DataFrame(rows)


def build_common_envelope(records: pd.DataFrame) -> pd.DataFrame:
    unit_instance, _, holdout = load_instance(
        BASELINE_EPSILON, 1.0, RESOURCE_RATIO, VEHICLES
    )
    unit_evaluator = ExactDROEvaluator(
        unit_instance.samples, unit_instance.epsilon, 1.0
    )
    pool: dict[str, Solution] = {}
    for route_json in records["routes"].drop_duplicates():
        solution = solution_from_json(route_json)
        evaluate_solution(solution, unit_instance, unit_evaluator)
        pool[route_json] = solution

    full_solution = Solution(routes=balanced_full_routes(unit_instance))
    evaluate_solution(full_solution, unit_instance, unit_evaluator)
    pool[route_signature(full_solution)] = full_solution
    empty = Solution(routes=[[] for _ in range(unit_instance.vehicles)])
    evaluate_solution(empty, unit_instance, unit_evaluator)
    pool[route_signature(empty)] = empty

    rows: list[dict[str, object]] = []
    for penalty in M_LEVELS:
        best = min(
            pool.values(),
            key=lambda solution: solution.travel + penalty * solution.robust_penalty,
        )
        rows.append(
            {
                "M": penalty,
                "objective": best.travel + penalty * best.robust_penalty,
                "route_cost": best.travel,
                "weighted_penalty": penalty * best.robust_penalty,
                "unit_robust_penalty": best.robust_penalty,
                "visited_nodes": len(best.visited),
                "used_vehicles": sum(bool(route) for route in best.routes),
                "training_prize_coverage_pct": 100.0
                * prize_coverage(best.visited, unit_instance.samples).mean(),
                "holdout_prize_coverage_pct": 100.0
                * prize_coverage(best.visited, holdout).mean(),
                "routes": route_signature(best),
            }
        )
    best = pd.DataFrame(rows)
    best.to_csv(BEST_PATH, index=False)
    return best


def draw_figure(best: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.05))
    plt.subplots_adjust(left=0.075, right=0.92, bottom=0.18, top=0.92, wspace=0.30)

    axes[0].plot(best["M"], best["objective"] / 1000, color="#202124", label="Total objective")
    axes[0].plot(best["M"], best["route_cost"] / 1000, color="#1565c0", label="Routing cost")
    axes[0].plot(best["M"], best["weighted_penalty"] / 1000, color="#d95f02", label="Weighted penalty")
    axes[0].set_xscale("symlog", linthresh=5000)
    axes[0].set_xlabel(r"Missed-prize coefficient $M$")
    axes[0].set_ylabel(r"Cost ($\times10^3$)")
    axes[0].set_title("Cost decomposition with relaxed route budgets")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    unit_ax = axes[1]
    coverage_ax = unit_ax.twinx()
    unit_ax.plot(best["M"], best["unit_robust_penalty"], color="#c62828", marker="o", markersize=3.5, label="Unit robust missed prize")
    coverage_ax.plot(best["M"], best["training_prize_coverage_pct"], color="#2e7d32", marker="s", markersize=3.0, label="Training prize coverage")
    unit_ax.set_xscale("symlog", linthresh=5000)
    unit_ax.set_xlabel(r"Missed-prize coefficient $M$")
    unit_ax.set_ylabel("Unit robust missed prize", color="#c62828")
    coverage_ax.set_ylabel("Training prize coverage (%)", color="#2e7d32")
    unit_ax.set_title("Missed-prize reduction and coverage")
    unit_ax.grid(alpha=0.25)
    lines = unit_ax.lines + coverage_ax.lines
    unit_ax.legend(lines, [line.get_label() for line in lines], fontsize=8, loc="center right")

    for suffix in ("png", "pdf", "svg"):
        fig.savefig(FIGURE_PATH.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    started = time.perf_counter()
    records = run_restarts()
    best = build_common_envelope(records)
    draw_figure(best)
    METADATA_PATH.write_text(
        json.dumps(
            {
                "resource_ratio": RESOURCE_RATIO,
                "route_budget_is_relaxed": True,
                "search_seconds_per_restart": SEARCH_SECONDS,
                "restart_seeds": list(SEEDS),
                "M_levels": [float(value) for value in M_LEVELS],
                "network_access": False,
                "elapsed_seconds_current_run": time.perf_counter() - started,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved relaxed-budget results to {BEST_PATH}", flush=True)


if __name__ == "__main__":
    main()
