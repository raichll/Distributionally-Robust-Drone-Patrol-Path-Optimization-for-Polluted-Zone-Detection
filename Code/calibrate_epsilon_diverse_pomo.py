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
    mns_from_start,
)
from run_rolling_exact_objective_search import evaluator_for
from run_temporal_holdout_and_m_sensitivity import realized_metrics


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "outputs" / "hourly_holdout_250_train120"
OUTPUT_DIR = ROOT / "outputs" / "epsilon_strict_kuhn_holdout_132_36_M4000"

M = 4000.0
EPSILON_LEVELS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
POMO_CANDIDATES = 64
DIVERSE_STARTS = 6
MNS_SECONDS = 3.0
TRAIN_HOURS = 132
VALIDATION_HOURS = 36
TEST_START = 168
TEST_HOURS = 48


def validation_metrics(solution: Solution, samples: np.ndarray) -> dict[str, float]:
    frame = realized_metrics(solution, samples, M)
    costs = frame["realized_cost"].to_numpy(float)
    tail_count = max(1, math.ceil(0.10 * len(costs)))
    return {
        "mean_cost": float(costs.mean()),
        "cvar90": float(np.sort(costs)[-tail_count:].mean()),
        "worst_cost": float(costs.max()),
        "coverage_pct": float(frame["prize_coverage_pct"].mean()),
    }


def diverse_ranked_starts(
    plans: list[list[list[int]]],
    samples: np.ndarray,
    epsilon: float,
    model: str,
) -> tuple[list[Solution], pd.DataFrame]:
    instance = make_instance(candidates_global, samples, epsilon, M)
    evaluator = evaluator_for(model, samples, epsilon, M)
    unique: dict[frozenset[int], Solution] = {}
    for plan in plans:
        candidate = POMOWarmStart._prune_to_budget(plan, instance, evaluator)
        key = candidate.visited
        incumbent = unique.get(key)
        if incumbent is None or candidate.objective < incumbent.objective:
            unique[key] = candidate.clone()
    ranked = sorted(unique.values(), key=lambda solution: solution.objective)
    selected = [solution.clone() for solution in ranked[:DIVERSE_STARTS]]
    rows = [
        {
            "rank": rank,
            "objective": solution.objective,
            "travel_cost": solution.travel,
            "penalty": solution.robust_penalty,
            "visited_nodes": len(solution.visited),
        }
        for rank, solution in enumerate(ranked, start=1)
    ]
    return selected, pd.DataFrame(rows)


def optimize(
    plans: list[list[list[int]]],
    samples: np.ndarray,
    epsilon: float,
    model: str,
    seed: int,
) -> tuple[Solution, pd.DataFrame, pd.DataFrame]:
    instance = make_instance(candidates_global, samples, epsilon, M)
    evaluator = evaluator_for(model, samples, epsilon, M)
    starts, ranking = diverse_ranked_starts(plans, samples, epsilon, model)
    best = Solution(routes=[[] for _ in range(instance.vehicles)])
    best = POMOWarmStart._prune_to_budget(best.routes, instance, evaluator)
    run_rows: list[dict[str, object]] = []
    for index, initial in enumerate(starts):
        rng = np.random.default_rng(seed + index)
        started = time.perf_counter()
        stats: dict[str, float | int | bool] = {}
        solution = mns_from_start(
            initial.clone(),
            instance,
            evaluator,
            rng,
            started + MNS_SECONDS,
            stats=stats,
        )
        if solution.objective < best.objective:
            best = solution.clone()
        run_rows.append(
            {
                "model": model,
                "epsilon": epsilon,
                "start": index + 1,
                "initial_objective": initial.objective,
                "final_objective": solution.objective,
                "travel_cost": solution.travel,
                "penalty": solution.robust_penalty,
                "visited_nodes": len(solution.visited),
                "runtime": time.perf_counter() - started,
                "routes": json.dumps(solution.routes),
            }
        )
    return best, pd.DataFrame(run_rows), ranking


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    global candidates_global
    candidates_global, hours, samples = load_arrays(DATA_DIR)

    print("[POMO] generating 64 geometry-based candidate routes", flush=True)
    reference = make_instance(candidates_global, samples[:120], 0.10, M)
    pomo = POMOWarmStart()
    plans = pomo.candidate_plans(
        reference, BASE_SEED, max_candidates=POMO_CANDIDATES
    )
    print(f"[POMO] generated {len(plans)} candidates", flush=True)

    calibration_rows: list[dict[str, object]] = []
    run_frames: list[pd.DataFrame] = []
    ranking_frames: list[pd.DataFrame] = []
    training = samples[:TRAIN_HOURS]
    validation = samples[TRAIN_HOURS : TRAIN_HOURS + VALIDATION_HOURS]
    dro_solutions: dict[float, Solution] = {}
    for epsilon in EPSILON_LEVELS:
        print(f"[train132_validate36] epsilon={epsilon:.2f}", flush=True)
        solution, runs, ranking = optimize(
            plans,
            training,
            epsilon,
            "Exact DRO",
            BASE_SEED + int(epsilon * 1000),
        )
        dro_solutions[epsilon] = solution.clone()
        metrics = validation_metrics(solution, validation)
        calibration_rows.append(
            {
                    "scheme": "train132_validate36",
                "train_hours": TRAIN_HOURS,
                "validation_hours": VALIDATION_HOURS,
                "epsilon": epsilon,
                "training_objective": solution.objective,
                "travel_cost": solution.travel,
                "robust_penalty": solution.robust_penalty,
                "visited_nodes": len(solution.visited),
                "validation_mean_cost": metrics["mean_cost"],
                "validation_cvar90": metrics["cvar90"],
                "validation_worst_cost": metrics["worst_cost"],
                "validation_coverage_pct": metrics["coverage_pct"],
                "routes": json.dumps(solution.routes),
            }
        )
        runs.insert(0, "scheme", "train120_validate48")
        ranking.insert(0, "scheme", "train120_validate48")
        ranking.insert(1, "epsilon", epsilon)
        run_frames.append(runs)
        ranking_frames.append(ranking)
        pd.DataFrame(calibration_rows).to_csv(
            OUTPUT_DIR / "fixed_holdout_validation_runs.csv", index=False
        )
        print(
            f"  nodes={len(solution.visited)}, "
            f"mean={metrics['mean_cost']:.2f}, CVaR90={metrics['cvar90']:.2f}",
            flush=True,
        )

    calibration = pd.DataFrame(calibration_rows)
    epsilon_selection = calibration.sort_values(
        ["validation_mean_cost", "epsilon"], ascending=[True, True]
    )
    selected_epsilon = float(epsilon_selection.iloc[0]["epsilon"])
    calibration.to_csv(OUTPUT_DIR / "epsilon_selection_all.csv", index=False)
    epsilon_selection.head(1).to_csv(
        OUTPUT_DIR / "selected_epsilon.csv", index=False
    )
    pd.concat(run_frames, ignore_index=True).to_csv(
        OUTPUT_DIR / "all_mns_runs.csv", index=False
    )
    pd.concat(ranking_frames, ignore_index=True).to_csv(
        OUTPUT_DIR / "all_pomo_candidate_rankings.csv", index=False
    )
    print(f"[locked] epsilon={selected_epsilon:.2f}; opening final 48h only now", flush=True)

    test = samples[TEST_START : TEST_START + TEST_HOURS]
    test_hours = hours[TEST_START : TEST_START + TEST_HOURS]
    test_rows: list[dict[str, object]] = []
    hourly_frames: list[pd.DataFrame] = []
    final_run_frames: list[pd.DataFrame] = []
    for model in ("SP", "RO", "Exact DRO"):
        model_epsilon = 0.0 if model == "SP" else selected_epsilon
        if model == "Exact DRO":
            # Strict holdout: test the exact training route whose radius won validation.
            solution = dro_solutions[selected_epsilon].clone()
            runs = pd.DataFrame()
        else:
            solution, runs, _ = optimize(
                plans,
                training,
                model_epsilon,
                model,
                BASE_SEED + 90000 + 1000 * ("SP", "RO").index(model),
            )
        metrics = validation_metrics(solution, test)
        test_rows.append(
            {
                "model": model,
                "M": M,
                "epsilon": model_epsilon,
                "visited_nodes": len(solution.visited),
                "travel_cost": solution.travel,
                "training_objective": solution.objective,
                "test_mean_cost": metrics["mean_cost"],
                "test_cvar90": metrics["cvar90"],
                "test_worst_cost": metrics["worst_cost"],
                "test_coverage_pct": metrics["coverage_pct"],
                "routes": json.dumps(solution.routes),
            }
        )
        hourly = realized_metrics(solution, test, M)
        hourly.insert(0, "hour", test_hours)
        hourly.insert(1, "model", model)
        hourly_frames.append(hourly)
        if not runs.empty:
            runs.insert(0, "phase", "strict_holdout_training")
            final_run_frames.append(runs)
        print(
            f"[test] {model}: mean={metrics['mean_cost']:.2f}, "
            f"CVaR90={metrics['cvar90']:.2f}, nodes={len(solution.visited)}",
            flush=True,
        )

    pd.DataFrame(test_rows).to_csv(OUTPUT_DIR / "final_48h_comparison.csv", index=False)
    pd.concat(hourly_frames, ignore_index=True).to_csv(
        OUTPUT_DIR / "final_48h_hourly.csv", index=False
    )
    pd.concat(final_run_frames, ignore_index=True).to_csv(
        OUTPUT_DIR / "baseline_training_mns_runs.csv", index=False
    )
    metadata = {
        "M": M,
        "epsilon_grid": list(EPSILON_LEVELS),
        "selected_epsilon": selected_epsilon,
        "holdout_scheme": "train132_validate36_test48",
        "selection_metric": "sample mean realized loss on the fixed validation set",
        "training_hours": TRAIN_HOURS,
        "validation_hours": VALIDATION_HOURS,
        "final_test_hours": TEST_HOURS,
        "refit_after_validation": False,
        "test_used_during_selection": False,
        "pomo_candidates": len(plans),
        "diverse_starts_per_model": DIVERSE_STARTS,
        "mns_seconds_per_start": MNS_SECONDS,
    }
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    candidates_global: pd.DataFrame
    main()
