from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from calibrate_temporal_dro_parameters import BASE_SEED, load_arrays, make_instance
from run_exact_dro_pcvrp_experiments import (
    ExactDROEvaluator,
    POMOWarmStart,
    Solution,
    construct_solution,
    evaluate_solution,
    selection_descent,
)
from run_temporal_holdout_and_m_sensitivity import (
    BoxRobustEvaluator,
    moving_block_bootstrap_ci,
    realized_metrics,
)


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "outputs" / "hourly_holdout_250_train120"
OUTPUT_DIR = ROOT / "outputs" / "rolling_exact_objective_search_m15000"
WARM_POOL_FILE = (
    ROOT
    / "outputs"
    / "pomo_pool_objective_evaluation"
    / "fixed_unique_pomo_candidates.csv"
)

M = 15000.0
EPSILON_LEVELS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.10)
MODELS = ("SP", "RO", "Exact DRO")
STARTS = 6
SECONDS_PER_START = 3.0
POMO_ORDER_STARTS = 3
FOLDS = (
    ("fold1", 120, 24, 24),
    ("fold2", 144, 24, 24),
    ("fold3", 168, 24, 24),
)


def evaluator_for(
    model: str,
    samples: np.ndarray,
    epsilon: float,
    penalty_scale: float = M,
):
    if model == "SP":
        return ExactDROEvaluator(samples, 0.0, penalty_scale)
    if model == "RO":
        return BoxRobustEvaluator(samples.shape[1], penalty_scale)
    if model == "Exact DRO":
        return ExactDROEvaluator(samples, epsilon, penalty_scale)
    raise ValueError(model)


def flatten_plan(plan: list[list[int]]) -> list[int]:
    return [node for route in plan for node in route]


def rank_feasible_starts(
    warm_pool: pd.DataFrame,
    model: str,
    samples: np.ndarray,
    epsilon: float,
) -> list[list[list[int]]]:
    evaluator = evaluator_for(model, samples, epsilon, M)
    ranked: list[tuple[float, list[list[int]]]] = []
    for row in warm_pool.itertuples(index=False):
        routes = json.loads(row.routes)
        visited = frozenset(node for route in routes for node in route)
        penalty, _ = evaluator.evaluate(visited)
        ranked.append((float(row.travel_cost) + penalty, routes))
    ranked.sort(key=lambda item: item[0])
    return [routes for _, routes in ranked[:POMO_ORDER_STARTS]]


def exact_multistart_search(
    model: str,
    samples: np.ndarray,
    candidates: pd.DataFrame,
    epsilon: float,
    plans: list[list[list[int]]],
    seed: int,
    penalty_scale: float = M,
    feasible_starts: list[list[list[int]]] | None = None,
) -> tuple[Solution, pd.DataFrame]:
    instance = make_instance(candidates, samples, epsilon, penalty_scale)
    evaluator = evaluator_for(model, samples, epsilon, penalty_scale)
    best = Solution(routes=[[] for _ in range(instance.vehicles)])
    evaluate_solution(best, instance, evaluator)
    rows: list[dict[str, object]] = []

    for start_index in range(STARTS):
        rng_seed = seed + 1000 * (MODELS.index(model) + 1) + start_index
        rng = np.random.default_rng(rng_seed)
        started = time.perf_counter()
        deadline = started + SECONDS_PER_START
        if start_index < POMO_ORDER_STARTS:
            if feasible_starts is None:
                candidate = POMOWarmStart._prune_to_budget(
                    plans[start_index], instance, evaluator
                )
            else:
                candidate = Solution(
                    routes=[route.copy() for route in feasible_starts[start_index]]
                )
                evaluate_solution(candidate, instance, evaluator)
            start_type = f"POMO-exact-ranked-{start_index + 1}"
        else:
            candidate = construct_solution(
                instance,
                evaluator,
                rng,
                randomized=True,
                deadline=deadline,
            )
            start_type = f"random-exact-{start_index - POMO_ORDER_STARTS + 1}"
        while time.perf_counter() < deadline:
            changed = selection_descent(
                candidate,
                instance,
                evaluator,
                deadline,
                allow_exchange=True,
            )
            if not changed:
                break
        elapsed = time.perf_counter() - started
        if candidate.objective < best.objective:
            best = candidate.clone()
        rows.append(
            {
                "model": model,
                "epsilon": epsilon,
                "start_index": start_index + 1,
                "start_type": start_type,
                "rng_seed": rng_seed,
                "objective": candidate.objective,
                "travel_cost": candidate.travel,
                "penalty": candidate.robust_penalty,
                "visited_nodes": len(candidate.visited),
                "runtime": elapsed,
                "routes": json.dumps(candidate.routes),
            }
        )
    return best, pd.DataFrame(rows)


def summary_metrics(solution: Solution, samples: np.ndarray) -> dict[str, float]:
    frame = realized_metrics(solution, samples, M)
    costs = frame["realized_cost"].to_numpy(float)
    coverage = frame["prize_coverage_pct"].to_numpy(float)
    tail_count = max(1, math.ceil(0.10 * len(costs)))
    return {
        "mean_realized_cost": float(costs.mean()),
        "cvar90_realized_cost": float(np.sort(costs)[-tail_count:].mean()),
        "worst_cost": float(costs.max()),
        "mean_coverage_pct": float(coverage.mean()),
    }


def calibrate_epsilon(
    fold: str,
    training: np.ndarray,
    validation: np.ndarray,
    candidates: pd.DataFrame,
    plans: list[list[list[int]]],
    warm_pool: pd.DataFrame,
    seed: int,
) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    start_frames: list[pd.DataFrame] = []
    for epsilon in EPSILON_LEVELS:
        print(f"[{fold}] calibrating epsilon={epsilon:g}", flush=True)
        feasible_starts = rank_feasible_starts(
            warm_pool, "Exact DRO", training, epsilon
        )
        solution, starts = exact_multistart_search(
            "Exact DRO",
            training,
            candidates,
            epsilon,
            plans,
            seed,
            feasible_starts=feasible_starts,
        )
        starts.insert(0, "fold", fold)
        starts.insert(1, "phase", "epsilon_calibration")
        start_frames.append(starts)
        metrics = summary_metrics(solution, validation)
        rows.append(
            {
                "fold": fold,
                "epsilon": epsilon,
                "training_objective": solution.objective,
                "travel_cost": solution.travel,
                "robust_penalty": solution.robust_penalty,
                "visited_nodes": len(solution.visited),
                "validation_mean_cost": metrics["mean_realized_cost"],
                "validation_cvar90": metrics["cvar90_realized_cost"],
                "validation_worst_cost": metrics["worst_cost"],
                "validation_coverage_pct": metrics["mean_coverage_pct"],
                "routes": json.dumps(solution.routes),
            }
        )
    frame = pd.DataFrame(rows)
    chosen = frame.sort_values(
        ["validation_cvar90", "validation_mean_cost", "epsilon", "travel_cost"]
    ).iloc[0]
    return float(chosen["epsilon"]), frame, pd.concat(start_frames, ignore_index=True)


def run_fold(
    fold: str,
    train_hours: int,
    validation_hours: int,
    test_hours_count: int,
    samples: np.ndarray,
    hours: pd.DatetimeIndex,
    candidates: pd.DataFrame,
    plans: list[list[list[int]]],
    warm_pool: pd.DataFrame,
    fold_index: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validation_end = train_hours + validation_hours
    test_end = validation_end + test_hours_count
    training = samples[:train_hours]
    validation = samples[train_hours:validation_end]
    refit = samples[:validation_end]
    test = samples[validation_end:test_end]
    test_hours = hours[validation_end:test_end]
    seed = BASE_SEED + 100000 * fold_index

    epsilon, calibration, calibration_starts = calibrate_epsilon(
        fold, training, validation, candidates, plans, warm_pool, seed
    )
    print(f"[{fold}] locked epsilon={epsilon:g}; refitting models", flush=True)

    route_rows: list[dict[str, object]] = []
    test_parts: list[pd.DataFrame] = []
    test_summary_rows: list[dict[str, object]] = []
    refit_start_frames: list[pd.DataFrame] = []
    for model in MODELS:
        feasible_starts = rank_feasible_starts(
            warm_pool, model, refit, epsilon
        )
        solution, starts = exact_multistart_search(
            model,
            refit,
            candidates,
            epsilon,
            plans,
            seed + 50000,
            feasible_starts=feasible_starts,
        )
        starts.insert(0, "fold", fold)
        starts.insert(1, "phase", "refit")
        refit_start_frames.append(starts)
        route_rows.append(
            {
                "fold": fold,
                "model": model,
                "epsilon": epsilon if model == "Exact DRO" else (0.0 if model == "SP" else math.nan),
                "refit_objective_native": solution.objective,
                "travel_cost": solution.travel,
                "penalty_native": solution.robust_penalty,
                "visited_nodes": len(solution.visited),
                "routes": json.dumps(solution.routes),
            }
        )
        hourly = realized_metrics(solution, test, M)
        hourly.insert(0, "hour", test_hours)
        hourly.insert(1, "fold", fold)
        hourly.insert(2, "model", model)
        test_parts.append(hourly)
        metrics = summary_metrics(solution, test)
        test_summary_rows.append(
            {
                "fold": fold,
                "model": model,
                "selected_epsilon": epsilon,
                "visited_nodes": len(solution.visited),
                "travel_cost": solution.travel,
                **metrics,
            }
        )
    starts = pd.concat([calibration_starts, *refit_start_frames], ignore_index=True)
    return (
        calibration,
        pd.DataFrame(route_rows),
        pd.concat(test_parts, ignore_index=True),
        pd.DataFrame(test_summary_rows),
        starts,
    )


def aggregate_results(hourly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for model_index, model in enumerate(MODELS):
        group = hourly[hourly["model"] == model]
        costs = group["realized_cost"].to_numpy(float)
        coverage = group["prize_coverage_pct"].to_numpy(float)
        cost_low, cost_high = moving_block_bootstrap_ci(
            costs, 4, 10000, BASE_SEED + model_index
        )
        cov_low, cov_high = moving_block_bootstrap_ci(
            coverage, 4, 10000, BASE_SEED + 50 + model_index
        )
        tail_count = max(1, math.ceil(0.10 * len(costs)))
        rows.append(
            {
                "model": model,
                "hours": len(costs),
                "mean_realized_cost": float(costs.mean()),
                "mean_cost_ci_low": cost_low,
                "mean_cost_ci_high": cost_high,
                "cvar90_realized_cost": float(np.sort(costs)[-tail_count:].mean()),
                "worst_cost": float(costs.max()),
                "mean_coverage_pct": float(coverage.mean()),
                "mean_coverage_ci_low": cov_low,
                "mean_coverage_ci_high": cov_high,
            }
        )

    pivot = hourly.pivot_table(
        index=["fold", "hour"], columns="model", values="realized_cost"
    )
    paired_rows: list[dict[str, object]] = []
    for benchmark in ("SP", "RO"):
        difference = (pivot["Exact DRO"] - pivot[benchmark]).to_numpy(float)
        low, high = moving_block_bootstrap_ci(
            difference, 4, 10000, BASE_SEED + 100 + len(paired_rows)
        )
        paired_rows.append(
            {
                "comparison": f"Exact DRO minus {benchmark}",
                "mean_cost_difference": float(difference.mean()),
                "difference_ci_low": low,
                "difference_ci_high": high,
                "dro_lower_cost_hours": int(np.sum(difference < -1e-8)),
                "ties": int(np.sum(np.abs(difference) <= 1e-8)),
                "dro_higher_cost_hours": int(np.sum(difference > 1e-8)),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(paired_rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates, hours, samples = load_arrays(DATA_DIR)
    warm_pool = pd.read_csv(WARM_POOL_FILE)
    print("[POMO] loading checkpoint and generating route-order starts", flush=True)
    pomo = POMOWarmStart()
    reference = make_instance(candidates, samples[:120], 0.0, M)
    plans = pomo.candidate_plans(reference, BASE_SEED, max_candidates=POMO_ORDER_STARTS)

    calibration_frames: list[pd.DataFrame] = []
    route_frames: list[pd.DataFrame] = []
    hourly_frames: list[pd.DataFrame] = []
    fold_summary_frames: list[pd.DataFrame] = []
    start_frames: list[pd.DataFrame] = []
    for fold_index, (fold, train_hours, validation_hours, test_hours) in enumerate(FOLDS, start=1):
        result = run_fold(
            fold,
            train_hours,
            validation_hours,
            test_hours,
            samples,
            hours,
            candidates,
            plans,
            warm_pool,
            fold_index,
        )
        calibration, routes, hourly, fold_summary, starts = result
        calibration_frames.append(calibration)
        route_frames.append(routes)
        hourly_frames.append(hourly)
        fold_summary_frames.append(fold_summary)
        start_frames.append(starts)
        pd.concat(calibration_frames, ignore_index=True).to_csv(
            OUTPUT_DIR / "epsilon_calibration.csv", index=False
        )
        pd.concat(route_frames, ignore_index=True).to_csv(
            OUTPUT_DIR / "refit_selected_routes.csv", index=False
        )
        pd.concat(hourly_frames, ignore_index=True).to_csv(
            OUTPUT_DIR / "rolling_test_hourly.csv", index=False
        )
        pd.concat(fold_summary_frames, ignore_index=True).to_csv(
            OUTPUT_DIR / "rolling_test_by_fold.csv", index=False
        )
        pd.concat(start_frames, ignore_index=True).to_csv(
            OUTPUT_DIR / "all_multistart_runs.csv", index=False
        )

    hourly = pd.concat(hourly_frames, ignore_index=True)
    aggregate, paired = aggregate_results(hourly)
    aggregate.to_csv(OUTPUT_DIR / "aggregate_72h_summary.csv", index=False)
    paired.to_csv(OUTPUT_DIR / "paired_cost_differences.csv", index=False)
    metadata = {
        "M_fixed": M,
        "M_interpretation": "one full missed prize equals 150 km at 100 cost units/km",
        "epsilon_grid": list(EPSILON_LEVELS),
        "folds": [
            {
                "name": fold,
                "training_hours": train,
                "validation_hours": validation,
                "test_hours": test,
            }
            for fold, train, validation, test in FOLDS
        ],
        "starts_per_solve": STARTS,
        "seconds_per_start": SECONDS_PER_START,
        "seconds_per_model_solve": STARTS * SECONDS_PER_START,
        "search": "exact-objective insertion, deletion, exchange, and route-order VND",
        "all_folds_reported": True,
        "test_periods_previously_examined": True,
        "analysis_status": "diagnostic, not pristine confirmatory holdout",
    }
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("\nPer-fold results", flush=True)
    print(pd.concat(fold_summary_frames, ignore_index=True).round(4).to_string(index=False))
    print("\nAggregate 72-hour results", flush=True)
    print(aggregate.round(4).to_string(index=False))
    print("\nPaired cost differences", flush=True)
    print(paired.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
