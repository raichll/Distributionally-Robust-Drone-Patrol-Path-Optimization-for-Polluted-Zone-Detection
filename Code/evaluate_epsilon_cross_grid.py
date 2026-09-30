from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from run_exact_dro_pcvrp_experiments import (
    OUTPUT_DIR,
    ExactDROEvaluator,
    Solution,
    build_instances,
    evaluate_solution,
)


EPSILON_GRID = (0.00, 0.025, 0.05, 0.10, 0.15, 0.20)
REPORTING_EPSILON = 0.10
PENALTY_SCALE = 25000.0
RESOURCE_RATIO = 0.75


def main() -> None:
    records = pd.read_csv(OUTPUT_DIR / "epsilon_grid_records.csv")
    instances = build_instances(
        sizes=[40],
        replicates=10,
        epsilon=REPORTING_EPSILON,
        penalty_scale=PENALTY_SCALE,
        resource_ratio=RESOURCE_RATIO,
        seed=20260807 + 771,
    )
    instance_by_name = {}
    for instance in instances:
        instance.name = f"{instance.name}_rb"
        instance_by_name[instance.name] = instance

    evaluators = {
        (instance.name, epsilon): ExactDROEvaluator(
            instance.samples, epsilon, instance.penalty_scale
        )
        for instance in instances
        for epsilon in EPSILON_GRID
    }

    rows: list[dict[str, object]] = []
    for record in records.itertuples(index=False):
        instance = instance_by_name[record.instance]
        routes = json.loads(record.routes)
        for evaluation_epsilon in EPSILON_GRID:
            solution = Solution(routes=[list(route) for route in routes])
            evaluate_solution(
                solution,
                instance,
                evaluators[(instance.name, evaluation_epsilon)],
            )
            rows.append(
                {
                    "instance": record.instance,
                    "seed": int(record.seed),
                    "optimization_epsilon": float(record.optimization_epsilon),
                    "evaluation_epsilon": evaluation_epsilon,
                    "objective": solution.objective,
                    "travel": solution.travel,
                    "robust_penalty": solution.robust_penalty,
                    "visited": int(record.visited),
                    "visit_rate_pct": float(record.visit_rate_pct),
                    "runtime": float(record.runtime),
                }
            )

    cross_records = pd.DataFrame(rows)
    cross_records.to_csv(OUTPUT_DIR / "epsilon_cross_evaluation_records.csv", index=False)

    matrix = cross_records.pivot_table(
        index="optimization_epsilon",
        columns="evaluation_epsilon",
        values="objective",
        aggfunc="mean",
    ).sort_index().sort_index(axis=1)
    matrix.to_csv(OUTPUT_DIR / "epsilon_cross_evaluation_matrix.csv")

    summary = (
        records.groupby("optimization_epsilon", as_index=False)
        .agg(
            travel_mean=("travel", "mean"),
            visit_rate_mean_pct=("visit_rate_pct", "mean"),
            runtime_mean=("runtime", "mean"),
        )
        .sort_values("optimization_epsilon")
    )
    summary.to_csv(OUTPUT_DIR / "epsilon_cross_evaluation_route_summary.csv", index=False)

    column_winners = []
    for evaluation_epsilon in EPSILON_GRID:
        column = matrix[evaluation_epsilon]
        best_optimization_epsilon = float(column.idxmin())
        column_winners.append(
            {
                "evaluation_epsilon": evaluation_epsilon,
                "best_optimization_epsilon": best_optimization_epsilon,
                "best_mean_objective": float(column.min()),
                "matched_radius_mean_objective": float(
                    matrix.loc[evaluation_epsilon, evaluation_epsilon]
                ),
            }
        )
    pd.DataFrame(column_winners).to_csv(
        OUTPUT_DIR / "epsilon_cross_evaluation_winners.csv", index=False
    )

    paired = cross_records.pivot_table(
        index=["instance", "seed"],
        columns=["optimization_epsilon", "evaluation_epsilon"],
        values="objective",
        aggfunc="first",
    )
    tests = []
    for evaluation_epsilon in EPSILON_GRID[1:]:
        nominal_route = paired[(0.0, evaluation_epsilon)]
        matched_route = paired[(evaluation_epsilon, evaluation_epsilon)]
        reduction = nominal_route - matched_route
        try:
            p_value = float(
                wilcoxon(
                    reduction,
                    alternative="greater",
                    zero_method="zsplit",
                ).pvalue
            )
        except ValueError:
            p_value = 1.0
        tests.append(
            {
                "evaluation_epsilon": evaluation_epsilon,
                "matched_optimization_epsilon": evaluation_epsilon,
                "mean_reduction": float(reduction.mean()),
                "mean_reduction_pct": float(
                    100.0
                    * np.mean(
                        reduction / np.maximum(nominal_route.to_numpy(), 1e-12)
                    )
                ),
                "wins": int((reduction > 1e-8).sum()),
                "ties": int((reduction.abs() <= 1e-8).sum()),
                "losses": int((reduction < -1e-8).sum()),
                "wilcoxon_one_sided_p": p_value,
            }
        )
    tests_frame = pd.DataFrame(tests)
    tests_frame.to_csv(
        OUTPUT_DIR / "epsilon_cross_evaluation_matched_tests.csv", index=False
    )

    print("Mean exact objective matrix (rows: optimization epsilon; columns: evaluation epsilon)")
    print((matrix / 1000.0).round(3).to_string())
    print("\nRoute summary")
    print(summary.round(4).to_string(index=False))
    print("\nColumn winners")
    print(pd.DataFrame(column_winners).round(6).to_string(index=False))
    print("\nMatched-radius routes versus nominally optimized routes")
    print(tests_frame.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
