# Final paper code

This directory contains the code retained for the accepted manuscript. The selection is limited to the final experiment chain and the local modules that it imports.

## Experiment map

| Manuscript component | Main entry point | Retained results |
|---|---|---|
| AD-DBSCAN comparison | `run_region_clustering_experiment.py` | `outputs/chengdu_region*_clustering_5alg_*.csv` |
| Exact DRO and POMO-NS benchmarks | `run_exact_dro_pcvrp_experiments.py` | `outputs/exact_dro_pcvrp_benchmarks_168h/` |
| Wasserstein-radius cross-grid ablation | `run_epsilon_grid_ablation.py`, `evaluate_epsilon_cross_grid.py` | same benchmark directory |
| Strict chronological epsilon selection | `calibrate_epsilon_diverse_pomo.py` | `outputs/epsilon_strict_kuhn_holdout_132_36_M4000/` |
| Route-sensitivity figure | `plot_route_epsilon_sensitivity.py` | manuscript figure plus retained result tables |
| Conditional sensitivity of M | `run_m_sensitivity_relaxed_budget.py` | `outputs/real_case_m_relaxed_budget_R1/` |
| Processed hourly prize construction | `build_hourly_holdout_prizes.py` | `outputs/hourly_holdout_250*/` |

Supporting scripts such as `calibrate_temporal_dro_parameters.py`, `run_temporal_holdout_and_m_sensitivity.py`, and `run_rolling_exact_objective_search.py` are retained because the final entry points import them.

## Included dependencies and data

- `ELG-master/CVRP/` contains the POMO inference modules, checkpoint, and CVRPLIB instances used by the retained workflow. Its upstream license is preserved.
- `dynamic_dbscan-main/` contains the AD-DBSCAN implementation used by the clustering experiment. Its upstream license is preserved.
- `clu_data/` contains only the compact processed regional clustering sample used by the retained benchmark.
- `outputs/` contains processed hourly prize arrays and final CSV/JSON result tables. Generated CPLEX LP/SOL/log files are excluded.

The large raw GPS exports, satellite/map tile caches, obsolete code variants, exploratory result folders, and manuscript working files are not part of this release.

## Solver configuration

The exact mixed-integer baseline invokes the CPLEX command-line executable. Configure it with either:

```powershell
$env:CPLEX_EXE = "C:\path\to\cplex.exe"
```

or make `cplex` available on `PATH`.
