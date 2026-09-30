from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from run_exact_dro_pcvrp_experiments import (
    OUTPUT_DIR,
    POMOWarmStart,
    build_instances,
    run_heuristic,
)


EPSILON_GRID = (0.00, 0.025, 0.05, 0.10, 0.15, 0.20)
SEEDS = (20260807, 20260808, 20260809)
REPORTING_EPSILON = 0.10
TIME_LIMIT = 5.0


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    instances = build_instances(
        sizes=[40],
        replicates=10,
        epsilon=REPORTING_EPSILON,
        penalty_scale=25000.0,
        resource_ratio=0.75,
        seed=20260807 + 771,
    )
    pomo = POMOWarmStart()
    rows: list[dict[str, object]] = []
    for instance in instances:
        instance.name = f"{instance.name}_rb"
        for seed in SEEDS:
            for epsilon in EPSILON_GRID:
                print(
                    f"[epsilon-grid] {instance.name} epsilon={epsilon:.3f} seed={seed}",
                    flush=True,
                )
                row = run_heuristic(
                    "POMO-NS",
                    instance,
                    seed,
                    TIME_LIMIT,
                    pomo,
                    optimization_epsilon=epsilon,
                )
                row["method"] = f"POMO-NS-eps-{epsilon:.3f}"
                rows.append(row)
                pd.DataFrame(rows).to_csv(
                    OUTPUT_DIR / "epsilon_grid_records.csv", index=False
                )

    records = pd.DataFrame(rows)
    summary = (
        records.groupby("optimization_epsilon", as_index=False)
        .agg(
            nominal_objective_mean=("nominal_objective", "mean"),
            common_robust_objective_mean=("objective", "mean"),
            common_robust_objective_sd=("objective", "std"),
            travel_mean=("travel", "mean"),
            visit_rate_mean_pct=("visit_rate_pct", "mean"),
            runtime_mean=("runtime", "mean"),
            runs=("objective", "size"),
        )
        .sort_values("optimization_epsilon")
    )
    summary.to_csv(OUTPUT_DIR / "epsilon_grid_summary.csv", index=False)

    pivot = records.pivot_table(
        index=["instance", "seed"],
        columns="optimization_epsilon",
        values="objective",
        aggfunc="first",
    )
    baseline = pivot[0.0]
    tests = []
    for epsilon in EPSILON_GRID[1:]:
        delta = baseline - pivot[epsilon]
        try:
            p_value = float(
                wilcoxon(delta, alternative="greater", zero_method="zsplit").pvalue
            )
        except ValueError:
            p_value = 1.0
        tests.append(
            {
                "optimization_epsilon": epsilon,
                "paired_runs": int(delta.notna().sum()),
                "mean_common_robust_reduction": float(delta.mean()),
                "mean_reduction_pct": float(
                    100.0 * np.mean(delta / np.maximum(baseline, 1e-12))
                ),
                "wins": int((delta > 1e-8).sum()),
                "ties": int((delta.abs() <= 1e-8).sum()),
                "losses": int((delta < -1e-8).sum()),
                "wilcoxon_one_sided_p": p_value,
            }
        )
    pd.DataFrame(tests).to_csv(OUTPUT_DIR / "epsilon_grid_tests.csv", index=False)
    metadata = {
        "optimization_epsilon_grid": EPSILON_GRID,
        "common_reporting_epsilon": REPORTING_EPSILON,
        "instances": 10,
        "seeds": SEEDS,
        "time_limit_seconds": TIME_LIMIT,
        "penalty_scale": 25000.0,
        "resource_ratio": 0.75,
        "data_source": "training_prizes_168xn.npy",
    }
    (OUTPUT_DIR / "epsilon_grid_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)
    print(pd.DataFrame(tests).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
