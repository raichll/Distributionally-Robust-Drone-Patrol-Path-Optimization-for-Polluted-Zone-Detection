# Distributionally Robust Drone Patrol Path Optimization

This repository is the cleaned release corresponding to the accepted manuscript. It contains the accepted paper package and the final experimental code only; earlier manuscript revisions, obsolete experiment scripts, caches, map tiles, intermediate solver files, and large raw trajectory files have been removed.

## Repository layout

- [`Manuscript/`](Manuscript/) contains the accepted TeX source, bibliography, six referenced PDF figures, Elsevier style files, and the accepted compiled PDF.
- [`Code/`](Code/) contains the final AD-DBSCAN, exact Wasserstein DRO, POMO-NS, robustness-ablation, temporal-holdout, and sensitivity-analysis workflow.
- [`Code/outputs/`](Code/outputs/) contains the compact processed inputs and final numerical result tables needed by the retained scripts.

The accepted PDF is `Manuscript/Manuscript to part E-R2-clean.pdf` and should be treated as the release artifact.

## Python setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r Code/requirements.txt
```

The POMO checkpoint used by the retained workflow is included under `Code/ELG-master/CVRP/weights/`. Small processed Chengdu inputs are included, while the original multi-gigabyte GPS files are intentionally excluded.

## Main experiments

Run commands from `Code/`:

```bash
cd Code
python run_region_clustering_experiment.py
python run_exact_dro_pcvrp_experiments.py --smoke
python run_epsilon_grid_ablation.py
python evaluate_epsilon_cross_grid.py
python calibrate_epsilon_diverse_pomo.py
python plot_route_epsilon_sensitivity.py
python run_m_sensitivity_relaxed_budget.py
```

The exact mixed-integer baseline requires IBM ILOG CPLEX. Add `cplex` to `PATH` or set `CPLEX_EXE` to the executable. Other fixed-route Wasserstein evaluations use the analytic evaluator implemented in `run_exact_dro_pcvrp_experiments.py`.

See [`Code/README.md`](Code/README.md) for the experiment-to-file mapping and retained data description.

## Manuscript build

With a TeX distribution providing the `elsarticle` class:

```bash
cd Manuscript
latexmk -pdf "Manuscript to part E-R2-clean.tex"
```
