from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from run_exact_dro_pcvrp_experiments import (
    ExactDROEvaluator,
    Instance,
    POMOWarmStart,
    Solution,
    balanced_full_routes,
    make_distance_matrix,
    route_cost,
)
from run_temporal_holdout_and_m_sensitivity import (
    BoxRobustEvaluator,
    COST_PER_KM,
    DATA_DIR,
    moving_block_bootstrap_ci,
    prize_coverage,
    realized_metrics,
    solve_for_objective,
)


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "temporal_parameter_calibration"

TRAIN_HOURS = 144
VALIDATION_HOURS = 24
TEST_HOURS = 48
TARGET_COVERAGE_PCT = 85.0
RESOURCE_RATIO = 0.55
VEHICLES = 4
BASE_SEED = 20260810

PILOT_M = (3000.0, 4000.0, 5000.0, 6000.0, 8000.0, 10000.0, 15000.0, 23000.0)
PILOT_EPSILON = (0.0, 0.025, 0.05, 0.10, 0.15, 0.20)
PILOT_SECONDS = 5.0
REFINEMENT_SECONDS = 20.0
FINAL_SECONDS = 30.0
REFINEMENT_SEEDS = 3
FINAL_SEEDS = 3


def load_arrays(
    data_dir: Path = DATA_DIR,
) -> tuple[pd.DataFrame, pd.DatetimeIndex, np.ndarray]:
    candidates = pd.read_csv(data_dir / "candidate_nodes.csv")
    samples = np.load(data_dir / "all_prizes_216xn.npy")
    frame = pd.read_csv(data_dir / "hourly_prizes_216.csv", index_col=0)
    hours = pd.DatetimeIndex(pd.to_datetime(frame.index))
    if samples.shape != (TRAIN_HOURS + VALIDATION_HOURS + TEST_HOURS, len(candidates)):
        raise ValueError(f"Unexpected sample shape: {samples.shape}")
    return candidates, hours, samples


def make_instance(
    candidates: pd.DataFrame,
    samples: np.ndarray,
    epsilon: float,
    penalty_scale: float,
) -> Instance:
    depot_lon, depot_lat = 104.0665, 30.6570
    longitude = candidates["longitude"].to_numpy(float)
    latitude = candidates["latitude"].to_numpy(float)
    coordinates = np.column_stack(
        [
            (longitude - depot_lon) * 111.32 * math.cos(math.radians(depot_lat)),
            (latitude - depot_lat) * 110.57,
        ]
    )
    depot = np.array([0.0, 0.0])
    distance = make_distance_matrix(coordinates, depot, cost_per_km=COST_PER_KM)
    instance = Instance(
        name=f"chengdu_calibration_{samples.shape[0]}h",
        global_ids=candidates["node_id"].to_numpy(int),
        coords=coordinates,
        samples=samples,
        depot=depot,
        distance=distance,
        vehicles=VEHICLES,
        route_budget=math.inf,
        epsilon=float(epsilon),
        penalty_scale=float(penalty_scale),
    )
    full_routes = balanced_full_routes(instance)
    instance.route_budget = RESOURCE_RATIO * max(
        route_cost(route, distance) for route in full_routes
    )
    return instance


def warm_start_cache(
    pomo: POMOWarmStart,
    reference: Instance,
    seeds: list[int],
) -> dict[int, tuple[list[list[list[int]]], float]]:
    cache: dict[int, tuple[list[list[list[int]]], float]] = {}
    for seed in seeds:
        started = time.perf_counter()
        plans = pomo.candidate_plans(reference, seed)
        cache[seed] = (plans, time.perf_counter() - started)
    return cache


def tail_metrics(solution: Solution, samples: np.ndarray, penalty_scale: float) -> dict[str, float]:
    realized = realized_metrics(solution, samples, penalty_scale)
    costs = realized["realized_cost"].to_numpy(float)
    tail_count = max(1, math.ceil(0.10 * len(costs)))
    return {
        "mean_realized_cost": float(costs.mean()),
        "cvar90_realized_cost": float(np.sort(costs)[-tail_count:].mean()),
        "worst_realized_cost": float(costs.max()),
        "mean_coverage_pct": float(realized["prize_coverage_pct"].mean()),
    }


def solve_grid_row(
    candidates: pd.DataFrame,
    training: np.ndarray,
    validation: np.ndarray,
    pomo: POMOWarmStart,
    cache: dict[int, tuple[list[list[list[int]]], float]],
    penalty_scale: float,
    epsilon: float,
    seed: int,
    seconds: float,
) -> dict[str, object]:
    instance = make_instance(candidates, training, epsilon, penalty_scale)
    plans, warm_runtime = cache[seed]
    solution, runtime = solve_for_objective(
        "Exact DRO", instance, pomo, seed, seconds, plans, warm_runtime
    )
    metrics = tail_metrics(solution, validation, penalty_scale)
    return {
        "M": penalty_scale,
        "epsilon": epsilon,
        "seed": seed,
        "training_objective": solution.objective,
        "travel_cost": solution.travel,
        "robust_penalty": solution.robust_penalty,
        "visited_nodes": len(solution.visited),
        "visit_rate_pct": 100.0 * len(solution.visited) / instance.n,
        "validation_mean_cost": metrics["mean_realized_cost"],
        "validation_cvar90": metrics["cvar90_realized_cost"],
        "validation_worst_cost": metrics["worst_realized_cost"],
        "validation_coverage_pct": metrics["mean_coverage_pct"],
        "runtime": runtime,
        "routes": json.dumps(solution.routes),
    }


def select_refinement_pairs(pilot: pd.DataFrame) -> list[tuple[float, float]]:
    feasible = pilot[pilot["validation_coverage_pct"] >= TARGET_COVERAGE_PCT].copy()
    if feasible.empty:
        ranked = pilot.sort_values(
            ["validation_coverage_pct", "travel_cost"], ascending=[False, True]
        )
        return [(float(row.M), float(row.epsilon)) for row in ranked.head(6).itertuples()]

    first_m = float(feasible["M"].min())
    nearby_m = sorted(
        set(value for value in PILOT_M if first_m / 1.6 <= value <= first_m * 1.6)
    )
    ranked = pilot[pilot["M"].isin(nearby_m)].sort_values(
        ["validation_coverage_pct", "validation_cvar90", "travel_cost"],
        ascending=[False, True, True],
    )
    pairs = [(float(row.M), float(row.epsilon)) for row in ranked.head(8).itertuples()]
    if not pairs:
        pairs = [(float(row.M), float(row.epsilon)) for row in feasible.head(6).itertuples()]
    return list(dict.fromkeys(pairs))


def select_parameters(refined: pd.DataFrame) -> tuple[float, float, pd.DataFrame]:
    summary = (
        refined.groupby(["M", "epsilon"], as_index=False)
        .agg(
            validation_coverage_pct=("validation_coverage_pct", "mean"),
            validation_cvar90=("validation_cvar90", "mean"),
            validation_mean_cost=("validation_mean_cost", "mean"),
            travel_cost=("travel_cost", "mean"),
            visited_nodes=("visited_nodes", "mean"),
        )
    )
    feasible = summary[summary["validation_coverage_pct"] >= TARGET_COVERAGE_PCT]
    if feasible.empty:
        chosen = summary.sort_values(
            ["validation_coverage_pct", "validation_cvar90", "travel_cost"],
            ascending=[False, True, True],
        ).iloc[0]
    else:
        minimum_m = float(feasible["M"].min())
        chosen = feasible[feasible["M"] == minimum_m].sort_values(
            ["validation_cvar90", "validation_mean_cost", "travel_cost"]
        ).iloc[0]
    return float(chosen["M"]), float(chosen["epsilon"]), summary


def final_test(
    candidates: pd.DataFrame,
    training: np.ndarray,
    test: np.ndarray,
    test_hours: pd.DatetimeIndex,
    pomo: POMOWarmStart,
    cache: dict[int, tuple[list[list[list[int]]], float]],
    penalty_scale: float,
    epsilon: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = make_instance(candidates, training, epsilon, penalty_scale)
    rows: list[dict[str, object]] = []
    selected: dict[str, tuple[Solution, int, float]] = {}
    for model in ("SP", "RO", "Exact DRO"):
        for offset in range(FINAL_SEEDS):
            seed = BASE_SEED + offset
            instance = make_instance(candidates, training, epsilon, penalty_scale)
            plans, warm_runtime = cache[seed]
            solution, runtime = solve_for_objective(
                model, instance, pomo, seed, FINAL_SECONDS, plans, warm_runtime
            )
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "training_objective": solution.objective,
                    "travel_cost": solution.travel,
                    "visited_nodes": len(solution.visited),
                    "runtime": runtime,
                    "routes": json.dumps(solution.routes),
                }
            )
            incumbent = selected.get(model)
            if incumbent is None or solution.objective < incumbent[0].objective:
                selected[model] = (solution, seed, runtime)

    hourly_parts: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for model in ("SP", "RO", "Exact DRO"):
        solution, seed, runtime = selected[model]
        realized = realized_metrics(solution, test, penalty_scale)
        realized.insert(0, "hour", test_hours)
        realized.insert(1, "model", model)
        hourly_parts.append(realized)
        costs = realized["realized_cost"].to_numpy(float)
        coverage = realized["prize_coverage_pct"].to_numpy(float)
        cost_low, cost_high = moving_block_bootstrap_ci(
            costs, 4, 10000, BASE_SEED + len(summaries)
        )
        cov_low, cov_high = moving_block_bootstrap_ci(
            coverage, 4, 10000, BASE_SEED + 50 + len(summaries)
        )
        tail_count = max(1, math.ceil(0.10 * len(costs)))
        summaries.append(
            {
                "model": model,
                "selected_seed": seed,
                "visited_nodes": len(solution.visited),
                "visit_rate_pct": 100.0 * len(solution.visited) / base.n,
                "travel_cost": solution.travel,
                "mean_realized_cost": float(costs.mean()),
                "mean_cost_ci_low": cost_low,
                "mean_cost_ci_high": cost_high,
                "cvar90_realized_cost": float(np.sort(costs)[-tail_count:].mean()),
                "worst_hour_cost": float(costs.max()),
                "mean_prize_coverage_pct": float(coverage.mean()),
                "coverage_ci_low_pct": cov_low,
                "coverage_ci_high_pct": cov_high,
                "runtime": runtime,
            }
        )
    return pd.DataFrame(rows), pd.concat(hourly_parts), pd.DataFrame(summaries)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates, hours, samples = load_arrays()
    training_144 = samples[:TRAIN_HOURS]
    validation_24 = samples[TRAIN_HOURS : TRAIN_HOURS + VALIDATION_HOURS]
    training_168 = samples[: TRAIN_HOURS + VALIDATION_HOURS]
    test_48 = samples[-TEST_HOURS:]
    test_hours = hours[-TEST_HOURS:]

    print("Loading POMO checkpoint", flush=True)
    pomo = POMOWarmStart()
    seeds = [BASE_SEED + offset for offset in range(max(REFINEMENT_SEEDS, FINAL_SEEDS))]
    reference = make_instance(candidates, training_144, 0.10, 4000.0)
    cache = warm_start_cache(pomo, reference, seeds)

    pilot_rows: list[dict[str, object]] = []
    for penalty_scale in PILOT_M:
        for epsilon in PILOT_EPSILON:
            print(f"[pilot] M={penalty_scale:g}, epsilon={epsilon:g}", flush=True)
            pilot_rows.append(
                solve_grid_row(
                    candidates,
                    training_144,
                    validation_24,
                    pomo,
                    cache,
                    penalty_scale,
                    epsilon,
                    BASE_SEED,
                    PILOT_SECONDS,
                )
            )
            pd.DataFrame(pilot_rows).to_csv(OUTPUT_DIR / "pilot_grid.csv", index=False)
    pilot = pd.DataFrame(pilot_rows)
    pairs = select_refinement_pairs(pilot)
    print(f"Refining {len(pairs)} pairs: {pairs}", flush=True)

    refined_rows: list[dict[str, object]] = []
    for penalty_scale, epsilon in pairs:
        for offset in range(REFINEMENT_SEEDS):
            seed = BASE_SEED + offset
            print(
                f"[refine] M={penalty_scale:g}, epsilon={epsilon:g}, seed={seed}",
                flush=True,
            )
            refined_rows.append(
                solve_grid_row(
                    candidates,
                    training_144,
                    validation_24,
                    pomo,
                    cache,
                    penalty_scale,
                    epsilon,
                    seed,
                    REFINEMENT_SECONDS,
                )
            )
            pd.DataFrame(refined_rows).to_csv(
                OUTPUT_DIR / "refinement_runs.csv", index=False
            )
    refined = pd.DataFrame(refined_rows)
    selected_m, selected_epsilon, refinement_summary = select_parameters(refined)
    refinement_summary.to_csv(OUTPUT_DIR / "refinement_summary.csv", index=False)
    print(
        f"Locked parameters: M={selected_m:g}, epsilon={selected_epsilon:g}", flush=True
    )

    final_runs, final_hourly, final_summary = final_test(
        candidates,
        training_168,
        test_48,
        test_hours,
        pomo,
        cache,
        selected_m,
        selected_epsilon,
    )
    final_runs.to_csv(OUTPUT_DIR / "final_training_runs.csv", index=False)
    final_hourly.to_csv(OUTPUT_DIR / "final_test_hourly.csv", index=False)
    final_summary.to_csv(OUTPUT_DIR / "final_test_summary.csv", index=False)

    metadata = {
        "training_hours_for_calibration": TRAIN_HOURS,
        "validation_hours_for_calibration": VALIDATION_HOURS,
        "locked_training_hours": TRAIN_HOURS + VALIDATION_HOURS,
        "untouched_test_hours": TEST_HOURS,
        "coverage_target_pct": TARGET_COVERAGE_PCT,
        "selection_rule": (
            "smallest M attaining mean validation prize coverage >= target; "
            "within that M, minimum mean validation CVaR90, then mean cost and travel cost"
        ),
        "selected_M": selected_m,
        "selected_epsilon": selected_epsilon,
        "resource_ratio": RESOURCE_RATIO,
        "vehicles": VEHICLES,
        "pilot_seconds": PILOT_SECONDS,
        "refinement_seconds": REFINEMENT_SECONDS,
        "final_seconds": FINAL_SECONDS,
        "test_opened_only_after_parameter_lock": True,
    }
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("\nValidation refinement summary")
    print(refinement_summary.round(4).to_string(index=False))
    print("\nFinal untouched 48-hour test summary")
    print(final_summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
