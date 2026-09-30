from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from run_exact_dro_pcvrp_experiments import (
    ExactDROEvaluator,
    Instance,
    POMOWarmStart,
    Solution,
    balanced_full_routes,
    construct_solution,
    evaluate_solution,
    make_distance_matrix,
    mns_from_start,
    route_cost,
)


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "outputs" / "hourly_holdout_250"
OUTPUT_DIR = ROOT / "outputs" / "temporal_holdout_exact_dro"
COST_PER_KM = 100.0


class BoxRobustEvaluator:
    """Exact support-wide robust evaluator for prizes in [0,1]^n."""

    def __init__(self, n: int, penalty_scale: float):
        self.n = int(n)
        self.penalty_scale = float(penalty_scale)
        self.cache: dict[int, tuple[float, float]] = {}

    @staticmethod
    def _mask_key(visited: Iterable[int]) -> int:
        key = 0
        for node in visited:
            key |= 1 << int(node)
        return key

    def evaluate(self, visited: Iterable[int]) -> tuple[float, float]:
        key = self._mask_key(visited)
        if key not in self.cache:
            visited_count = key.bit_count()
            self.cache[key] = (
                self.penalty_scale * (self.n - visited_count),
                math.nan,
            )
        return self.cache[key]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilon", type=float, default=0.10)
    parser.add_argument("--baseline-m", type=float, default=4000.0)
    parser.add_argument(
        "--m-levels",
        type=str,
        default="0,1000,2000,3000,3500,3750,4000,5000,10000",
    )
    parser.add_argument("--resource-ratio", type=float, default=0.55)
    parser.add_argument("--vehicles", type=int, default=4)
    parser.add_argument("--time-limit", type=float, default=10.0)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--block-length", type=int, default=4)
    parser.add_argument("--sensitivity-only", action="store_true")
    parser.add_argument("--validation-only", action="store_true")
    return parser.parse_args()


def load_instance(
    epsilon: float,
    penalty_scale: float,
    resource_ratio: float,
    vehicles: int,
) -> tuple[Instance, pd.DatetimeIndex, np.ndarray]:
    candidates = pd.read_csv(DATA_DIR / "candidate_nodes.csv")
    training = np.load(DATA_DIR / "training_prizes_168xn.npy")
    holdout = np.load(DATA_DIR / "holdout_prizes_48xn.npy")
    prize_frame = pd.read_csv(DATA_DIR / "hourly_prizes_216.csv", index_col=0)
    hours = pd.to_datetime(prize_frame.index)
    holdout_hours = pd.DatetimeIndex(hours[168:])
    if training.shape != (168, len(candidates)):
        raise ValueError(f"Unexpected training shape: {training.shape}")
    if holdout.shape != (48, len(candidates)):
        raise ValueError(f"Unexpected holdout shape: {holdout.shape}")

    depot_lon = 104.0665
    depot_lat = 30.6570
    longitude = candidates["longitude"].to_numpy(float)
    latitude = candidates["latitude"].to_numpy(float)
    coordinates = np.column_stack(
        [
            (longitude - depot_lon)
            * 111.32
            * math.cos(math.radians(depot_lat)),
            (latitude - depot_lat) * 110.57,
        ]
    )
    depot = np.array([0.0, 0.0])
    distance = make_distance_matrix(coordinates, depot, cost_per_km=COST_PER_KM)
    provisional = Instance(
        name="chengdu_hourly_n250",
        global_ids=candidates["node_id"].to_numpy(int),
        coords=coordinates,
        samples=training,
        depot=depot,
        distance=distance,
        vehicles=vehicles,
        route_budget=math.inf,
        epsilon=epsilon,
        penalty_scale=penalty_scale,
    )
    full_routes = balanced_full_routes(provisional)
    full_route_costs = [route_cost(route, distance) for route in full_routes]
    provisional.route_budget = resource_ratio * max(full_route_costs)
    return provisional, holdout_hours, holdout


def solve_for_objective(
    model: str,
    instance: Instance,
    pomo: POMOWarmStart,
    seed: int,
    time_limit: float,
    cached_plans: list[list[list[int]]],
    warm_start_runtime: float,
) -> tuple[Solution, float]:
    if model == "SP":
        evaluator = ExactDROEvaluator(
            instance.samples, epsilon=0.0, penalty_scale=instance.penalty_scale
        )
    elif model == "Exact DRO":
        evaluator = ExactDROEvaluator(
            instance.samples,
            epsilon=instance.epsilon,
            penalty_scale=instance.penalty_scale,
        )
    elif model == "RO":
        evaluator = BoxRobustEvaluator(instance.n, instance.penalty_scale)
    else:
        raise ValueError(model)

    if instance.penalty_scale == 0.0:
        empty = Solution(routes=[[] for _ in range(instance.vehicles)])
        evaluate_solution(empty, instance, evaluator)
        return empty, warm_start_runtime

    started = time.perf_counter()
    deadline = started + time_limit
    rng = np.random.default_rng(seed)
    best = Solution(routes=[[] for _ in range(instance.vehicles)])
    evaluate_solution(best, instance, evaluator)
    for plan in cached_plans:
        pruned = pomo._prune_to_budget(plan, instance, evaluator)
        candidate = construct_solution(
            instance,
            evaluator,
            rng,
            initial=pruned,
            deadline=deadline,
        )
        if candidate.objective < best.objective:
            best = candidate
        if time.perf_counter() >= deadline:
            break
    if time.perf_counter() < deadline:
        best = mns_from_start(best, instance, evaluator, rng, deadline)
    search_runtime = time.perf_counter() - started
    return best, warm_start_runtime + search_runtime


def precompute_warm_starts(
    args: argparse.Namespace,
    pomo: POMOWarmStart,
) -> dict[int, tuple[list[list[list[int]]], float]]:
    instance, _, _ = load_instance(
        args.epsilon,
        args.baseline_m,
        args.resource_ratio,
        args.vehicles,
    )
    cache: dict[int, tuple[list[list[list[int]]], float]] = {}
    for offset in range(args.seeds):
        seed = args.seed + offset
        print(f"[POMO] precomputing warm starts for seed={seed}", flush=True)
        started = time.perf_counter()
        plans = pomo.candidate_plans(instance, seed)
        cache[seed] = (plans, time.perf_counter() - started)
    return cache


def prize_coverage(visited: frozenset[int], samples: np.ndarray) -> np.ndarray:
    totals = samples.sum(axis=1)
    if visited:
        collected = samples[:, sorted(visited)].sum(axis=1)
    else:
        collected = np.zeros(samples.shape[0])
    return np.divide(
        collected,
        totals,
        out=np.ones_like(collected, dtype=float),
        where=totals > 0,
    )


def realized_metrics(
    solution: Solution,
    holdout: np.ndarray,
    penalty_scale: float,
) -> pd.DataFrame:
    visited = solution.visited
    unvisited = np.array(
        [node for node in range(holdout.shape[1]) if node not in visited],
        dtype=int,
    )
    missed = (
        holdout[:, unvisited].sum(axis=1)
        if len(unvisited)
        else np.zeros(holdout.shape[0])
    )
    coverage = prize_coverage(visited, holdout)
    return pd.DataFrame(
        {
            "realized_cost": solution.travel + penalty_scale * missed,
            "travel_cost": solution.travel,
            "uncollected_prize": missed,
            "prize_coverage_pct": 100.0 * coverage,
        }
    )


def moving_block_bootstrap_ci(
    values: np.ndarray,
    block_length: int,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    n = len(values)
    rng = np.random.default_rng(seed)
    block_starts = np.arange(n)
    blocks_needed = math.ceil(n / block_length)
    means = np.empty(replicates)
    offsets = np.arange(block_length)
    for replicate in range(replicates):
        starts = rng.choice(block_starts, size=blocks_needed, replace=True)
        indices = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        means[replicate] = values[indices].mean()
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def summarize_holdout(
    hourly: pd.DataFrame,
    block_length: int,
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    for model, group in hourly.groupby("model", sort=False):
        costs = group["realized_cost"].to_numpy(float)
        low, high = moving_block_bootstrap_ci(
            costs, block_length, replicates, seed + len(summary_rows)
        )
        coverage = group["prize_coverage_pct"].to_numpy(float)
        coverage_low, coverage_high = moving_block_bootstrap_ci(
            coverage,
            block_length,
            replicates,
            seed + 50 + len(summary_rows),
        )
        tail_count = max(1, math.ceil(0.10 * len(costs)))
        summary_rows.append(
            {
                "model": model,
                "visited_nodes": int(group["visited_nodes"].iloc[0]),
                "visit_rate_pct": float(group["visit_rate_pct"].iloc[0]),
                "travel_cost": float(group["travel_cost"].iloc[0]),
                "mean_realized_cost": float(costs.mean()),
                "mean_cost_ci_low": low,
                "mean_cost_ci_high": high,
                "cvar90_realized_cost": float(np.sort(costs)[-tail_count:].mean()),
                "worst_hour_cost": float(costs.max()),
                "mean_uncollected_prize": float(
                    group["uncollected_prize"].mean()
                ),
                "mean_prize_coverage_pct": float(
                    group["prize_coverage_pct"].mean()
                ),
                "coverage_ci_low_pct": coverage_low,
                "coverage_ci_high_pct": coverage_high,
                "day_1_mean_cost": float(
                    group.iloc[:24]["realized_cost"].mean()
                ),
                "day_2_mean_cost": float(
                    group.iloc[24:]["realized_cost"].mean()
                ),
            }
        )

    pivot = hourly.pivot(index="hour", columns="model", values="realized_cost")
    test_rows: list[dict[str, object]] = []
    if "Exact DRO" in pivot:
        for benchmark in ["SP", "RO"]:
            if benchmark not in pivot:
                continue
            differences = (pivot[benchmark] - pivot["Exact DRO"]).to_numpy(float)
            low, high = moving_block_bootstrap_ci(
                differences, block_length, replicates, seed + 100 + len(test_rows)
            )
            if np.allclose(differences, 0.0):
                p_value = 1.0
            else:
                p_value = float(
                    wilcoxon(
                        differences,
                        alternative="two-sided",
                        zero_method="wilcox",
                    ).pvalue
                )
            test_rows.append(
                {
                    "benchmark": benchmark,
                    "reference": "Exact DRO",
                    "paired_hours": len(differences),
                    "mean_cost_reduction": float(differences.mean()),
                    "reduction_ci_low": low,
                    "reduction_ci_high": high,
                    "wins": int(np.sum(differences > 1e-8)),
                    "ties": int(np.sum(np.abs(differences) <= 1e-8)),
                    "losses": int(np.sum(differences < -1e-8)),
                    "wilcoxon_two_sided_p": p_value,
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(test_rows)


def run_m_sensitivity(
    args: argparse.Namespace,
    pomo: POMOWarmStart,
    levels: list[float],
    plan_cache: dict[int, tuple[list[list[list[int]]], float]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    seeds = [args.seed + offset for offset in range(args.seeds)]
    for penalty_scale in levels:
        instance, _, holdout = load_instance(
            args.epsilon,
            penalty_scale,
            args.resource_ratio,
            args.vehicles,
        )
        for seed in seeds:
            print(f"[M sensitivity] M={penalty_scale:g}, seed={seed}", flush=True)
            plans, warm_runtime = plan_cache[seed]
            solution, runtime = solve_for_objective(
                "Exact DRO",
                instance,
                pomo,
                seed,
                args.time_limit,
                plans,
                warm_runtime,
            )
            training_coverage = prize_coverage(
                solution.visited, instance.samples
            ).mean()
            holdout_frame = realized_metrics(solution, holdout, penalty_scale)
            rows.append(
                {
                    "M": penalty_scale,
                    "equivalent_km_per_full_prize": penalty_scale / COST_PER_KM,
                    "seed": seed,
                    "objective": solution.objective,
                    "travel_cost": solution.travel,
                    "robust_penalty": solution.robust_penalty,
                    "visited_nodes": len(solution.visited),
                    "visit_rate_pct": 100.0 * len(solution.visited) / instance.n,
                    "used_vehicles": int(sum(bool(route) for route in solution.routes)),
                    "training_prize_coverage_pct": 100.0 * training_coverage,
                    "holdout_prize_coverage_pct": float(
                        holdout_frame["prize_coverage_pct"].mean()
                    ),
                    "holdout_mean_realized_cost": float(
                        holdout_frame["realized_cost"].mean()
                    ),
                    "runtime": runtime,
                    "routes": json.dumps(solution.routes),
                }
            )
            pd.DataFrame(rows).to_csv(
                OUTPUT_DIR / "m_sensitivity_records.csv", index=False
            )
    records = pd.DataFrame(rows)
    summary = (
        records.groupby("M", as_index=False)
        .agg(
            equivalent_km_per_full_prize=(
                "equivalent_km_per_full_prize",
                "first",
            ),
            objective_mean=("objective", "mean"),
            objective_sd=("objective", "std"),
            travel_cost_mean=("travel_cost", "mean"),
            robust_penalty_mean=("robust_penalty", "mean"),
            visited_nodes_mean=("visited_nodes", "mean"),
            visit_rate_mean_pct=("visit_rate_pct", "mean"),
            used_vehicles_mean=("used_vehicles", "mean"),
            training_prize_coverage_mean_pct=(
                "training_prize_coverage_pct",
                "mean",
            ),
            holdout_prize_coverage_mean_pct=(
                "holdout_prize_coverage_pct",
                "mean",
            ),
            holdout_mean_realized_cost=(
                "holdout_mean_realized_cost",
                "mean",
            ),
            runtime_mean=("runtime", "mean"),
        )
        .sort_values("M")
    )
    summary.to_csv(OUTPUT_DIR / "m_sensitivity_summary.csv", index=False)
    return records, summary


def run_holdout_validation(
    args: argparse.Namespace,
    pomo: POMOWarmStart,
    plan_cache: dict[int, tuple[list[list[list[int]]], float]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    instance, holdout_hours, holdout = load_instance(
        args.epsilon,
        args.baseline_m,
        args.resource_ratio,
        args.vehicles,
    )
    seeds = [args.seed + offset for offset in range(args.seeds)]
    candidate_rows: list[dict[str, object]] = []
    best: dict[str, tuple[Solution, float, int]] = {}
    for model in ["SP", "RO", "Exact DRO"]:
        for seed in seeds:
            print(f"[holdout] {model}, seed={seed}", flush=True)
            plans, warm_runtime = plan_cache[seed]
            solution, runtime = solve_for_objective(
                model,
                instance,
                pomo,
                seed,
                args.time_limit,
                plans,
                warm_runtime,
            )
            candidate_rows.append(
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
            current = best.get(model)
            if current is None or solution.objective < current[0].objective:
                best[model] = (solution, runtime, seed)
        pd.DataFrame(candidate_rows).to_csv(
            OUTPUT_DIR / "holdout_training_runs.csv", index=False
        )

    hourly_frames: list[pd.DataFrame] = []
    selected_rows: list[dict[str, object]] = []
    for model in ["SP", "RO", "Exact DRO"]:
        solution, runtime, seed = best[model]
        frame = realized_metrics(solution, holdout, args.baseline_m)
        frame.insert(0, "hour", holdout_hours)
        frame.insert(1, "model", model)
        frame["visited_nodes"] = len(solution.visited)
        frame["visit_rate_pct"] = 100.0 * len(solution.visited) / instance.n
        hourly_frames.append(frame)
        selected_rows.append(
            {
                "model": model,
                "selected_seed": seed,
                "training_objective": solution.objective,
                "travel_cost": solution.travel,
                "visited_nodes": len(solution.visited),
                "visit_rate_pct": 100.0 * len(solution.visited) / instance.n,
                "runtime": runtime,
                "routes": json.dumps(solution.routes),
            }
        )
    hourly = pd.concat(hourly_frames, ignore_index=True)
    summary, tests = summarize_holdout(
        hourly,
        args.block_length,
        args.bootstrap_replicates,
        args.seed,
    )
    selected = pd.DataFrame(selected_rows)
    hourly.to_csv(OUTPUT_DIR / "holdout_hourly_results.csv", index=False)
    selected.to_csv(OUTPUT_DIR / "holdout_selected_routes.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "holdout_summary.csv", index=False)
    tests.to_csv(OUTPUT_DIR / "holdout_paired_tests.csv", index=False)
    return selected, hourly, summary, tests


def main() -> None:
    args = parse_args()
    if args.sensitivity_only and args.validation_only:
        raise ValueError("Choose at most one of --sensitivity-only and --validation-only")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    levels = [float(value) for value in args.m_levels.split(",")]
    print("[POMO] loading checkpoint", flush=True)
    pomo = POMOWarmStart()
    plan_cache = precompute_warm_starts(args, pomo)

    if not args.validation_only:
        _, sensitivity_summary = run_m_sensitivity(
            args, pomo, levels, plan_cache
        )
        print("\nM sensitivity summary", flush=True)
        print(sensitivity_summary.to_string(index=False), flush=True)
    if not args.sensitivity_only:
        _, _, holdout_summary, holdout_tests = run_holdout_validation(
            args, pomo, plan_cache
        )
        print("\nHoldout summary", flush=True)
        print(holdout_summary.to_string(index=False), flush=True)
        print("\nPaired holdout tests", flush=True)
        print(holdout_tests.to_string(index=False), flush=True)

    metadata = {
        "training_period": "2024-04-10 00:00 to 2024-04-16 23:00",
        "training_hours": 168,
        "holdout_period": "2024-04-17 00:00 to 2024-04-18 23:00",
        "holdout_hours": 48,
        "candidate_nodes": 250,
        "test_data_used_for_candidate_discovery": False,
        "epsilon": args.epsilon,
        "baseline_M": args.baseline_m,
        "M_levels": levels,
        "resource_ratio": args.resource_ratio,
        "vehicles": args.vehicles,
        "time_limit_seconds_per_run": args.time_limit,
        "seeds": [args.seed + offset for offset in range(args.seeds)],
        "block_bootstrap_replicates": args.bootstrap_replicates,
        "block_length_hours": args.block_length,
        "cost_per_km": COST_PER_KM,
        "hardware": "AMD Ryzen 5 3500X, 16 GB RAM, one CPU process",
    }
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
