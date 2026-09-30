from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from sklearn.neighbors import BallTree

from build_hourly_holdout_prizes import (
    DEFAULT_BOUNDS,
    EARTH_RADIUS_M,
    discover_hour_files,
)
from run_exact_dro_pcvrp_experiments import DATA_DIR, TRAINING_PRIZE_FILE


ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs" / "prize_holdout_validation"
ASSIGNMENT_RADIUS_M = 600.0
BLOCK_LENGTH_HOURS = 4
BOOTSTRAP_REPLICATES = 10_000
SEED = 20260809


def parse_holdout_hour(path: Path) -> pd.DataFrame:
    table = pq.read_table(path, columns=["plate_num", "speed", "coord"], use_threads=True)
    frame = table.to_pandas()
    frame["speed"] = pd.to_numeric(frame["speed"], errors="coerce")
    coord_text = frame["coord"].astype(str)
    frame["longitude"] = pd.to_numeric(
        coord_text.str.extract(r'"lon"\s*:\s*(-?\d+(?:\.\d+)?)', expand=False),
        errors="coerce",
    )
    frame["latitude"] = pd.to_numeric(
        coord_text.str.extract(r'"lat"\s*:\s*(-?\d+(?:\.\d+)?)', expand=False),
        errors="coerce",
    )
    frame = frame.dropna(subset=["plate_num", "speed", "longitude", "latitude"])
    lon_min, lon_max, lat_min, lat_max = DEFAULT_BOUNDS
    return frame[
        frame["speed"].between(0.0, 5.0)
        & frame["longitude"].between(lon_min, lon_max)
        & frame["latitude"].between(lat_min, lat_max)
    ][["plate_num", "longitude", "latitude"]]


def circular_block_indices(rng: np.random.Generator, periods: int) -> np.ndarray:
    blocks = int(np.ceil(periods / BLOCK_LENGTH_HOURS))
    starts = rng.integers(0, periods, size=blocks)
    offsets = np.arange(BLOCK_LENGTH_HOURS)
    return ((starts[:, None] + offsets[None, :]) % periods).ravel()[:periods]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(DATA_DIR / "candidate_nodes.csv")
    training_prizes = np.load(TRAINING_PRIZE_FILE)
    if training_prizes.shape != (168, len(candidates)):
        raise ValueError(
            f"Unexpected training prize shape {training_prizes.shape}; "
            f"expected (168, {len(candidates)})"
        )

    candidate_radians = np.deg2rad(
        candidates[["latitude", "longitude"]].to_numpy(float)
    )
    tree = BallTree(candidate_radians, metric="haversine")
    hourly_files = discover_hour_files(RAW_DATA_DIR)
    holdout_files = hourly_files[168:]
    unique_vehicle_counts = np.zeros((len(holdout_files), len(candidates)), dtype=np.int64)
    processing_rows: list[dict[str, object]] = []

    for hour_index, (hour, path) in enumerate(holdout_files):
        slow = parse_holdout_hour(path)
        if slow.empty:
            processing_rows.append(
                {
                    "hour": hour,
                    "slow_or_idle_records": 0,
                    "assigned_records": 0,
                    "unique_vehicle_node_visits": 0,
                }
            )
            continue

        point_radians = np.deg2rad(slow[["latitude", "longitude"]].to_numpy(float))
        distances, indices = tree.query(point_radians, k=1)
        assigned = distances[:, 0] * EARTH_RADIUS_M <= ASSIGNMENT_RADIUS_M
        assigned_nodes = indices[assigned, 0]
        assigned_plates = slow.loc[assigned, "plate_num"].astype(str).to_numpy()
        pairs = pd.DataFrame({"node_index": assigned_nodes, "plate_num": assigned_plates})
        pairs = pairs.drop_duplicates(["node_index", "plate_num"])
        counts = pairs.groupby("node_index").size()
        unique_vehicle_counts[hour_index, counts.index.to_numpy(int)] = counts.to_numpy(int)
        processing_rows.append(
            {
                "hour": hour,
                "slow_or_idle_records": len(slow),
                "assigned_records": int(np.count_nonzero(assigned)),
                "unique_vehicle_node_visits": len(pairs),
            }
        )
        print(
            f"[{hour_index + 1:02d}/{len(holdout_files)}] {hour}: "
            f"slow={len(slow):,}, assigned={np.count_nonzero(assigned):,}, "
            f"unique vehicle-node visits={len(pairs):,}",
            flush=True,
        )

    maxima = unique_vehicle_counts.max(axis=1)
    holdout_normalized = np.zeros_like(unique_vehicle_counts, dtype=float)
    nonzero = maxima > 0
    holdout_normalized[nonzero] = (
        unique_vehicle_counts[nonzero] / maxima[nonzero, None]
    )

    training_mean = training_prizes.mean(axis=0)
    holdout_mean = holdout_normalized.mean(axis=0)
    holdout_total = unique_vehicle_counts.sum(axis=0)
    rho, p_value = spearmanr(training_mean, holdout_mean)

    slope, intercept = np.polyfit(training_mean, holdout_mean, 1)
    calibrated = intercept + slope * training_mean
    residual = holdout_mean - calibrated
    ss_total = float(np.sum((holdout_mean - holdout_mean.mean()) ** 2))
    r_squared = 1.0 - float(np.sum(residual**2)) / max(ss_total, 1e-12)
    mae = float(np.mean(np.abs(training_mean - holdout_mean)))
    rmse = float(np.sqrt(np.mean((training_mean - holdout_mean) ** 2)))

    rng = np.random.default_rng(SEED)
    bootstrap_rho = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for replicate in range(BOOTSTRAP_REPLICATES):
        indices = circular_block_indices(rng, len(holdout_files))
        realization = holdout_normalized[indices].mean(axis=0)
        bootstrap_rho[replicate] = spearmanr(training_mean, realization).statistic
    rho_ci = np.quantile(bootstrap_rho, [0.025, 0.975])

    ranked = pd.Series(training_mean).rank(method="first")
    quintile = pd.qcut(ranked, 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    node_results = candidates[["node_id", "longitude", "latitude"]].copy()
    node_results["training_mean_prize"] = training_mean
    node_results["holdout_mean_normalized_unique_visits"] = holdout_mean
    node_results["holdout_unique_vehicle_node_hours"] = holdout_total
    node_results["training_prize_quintile"] = quintile.astype(str)
    node_results.to_csv(OUTPUT_DIR / "node_level_prize_validation.csv", index=False)

    quintile_summary = (
        node_results.groupby("training_prize_quintile", observed=True)
        .agg(
            nodes=("node_id", "size"),
            training_mean_prize=("training_mean_prize", "mean"),
            holdout_mean_normalized_unique_visits=(
                "holdout_mean_normalized_unique_visits",
                "mean",
            ),
            holdout_unique_vehicle_node_hours=(
                "holdout_unique_vehicle_node_hours",
                "sum",
            ),
        )
        .reindex(["Q1", "Q2", "Q3", "Q4", "Q5"])
        .reset_index()
    )
    quintile_summary.to_csv(OUTPUT_DIR / "prize_quintile_calibration.csv", index=False)

    top_count = int(np.ceil(0.20 * len(candidates)))
    top_indices = np.argsort(training_mean)[-top_count:]
    top_share = float(holdout_total[top_indices].sum() / max(holdout_total.sum(), 1))
    top_lift = top_share / 0.20

    processing = pd.DataFrame(processing_rows)
    processing.to_csv(OUTPUT_DIR / "holdout_visit_processing.csv", index=False)
    pd.DataFrame(unique_vehicle_counts).to_csv(
        OUTPUT_DIR / "holdout_unique_vehicle_counts_48xn.csv", index=False
    )
    pd.DataFrame(holdout_normalized).to_csv(
        OUTPUT_DIR / "holdout_normalized_unique_visits_48xn.csv", index=False
    )

    summary = {
        "training_hours": 168,
        "holdout_hours": 48,
        "candidate_nodes": len(candidates),
        "assignment_radius_m": ASSIGNMENT_RADIUS_M,
        "outcome": "unique vehicle-node-hour visits among slow or idle observations",
        "spearman_rho": float(rho),
        "spearman_p_value": float(p_value),
        "spearman_block_bootstrap_ci_95": [float(rho_ci[0]), float(rho_ci[1])],
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "block_length_hours": BLOCK_LENGTH_HOURS,
        "calibration_intercept": float(intercept),
        "calibration_slope": float(slope),
        "calibration_r_squared": r_squared,
        "raw_scale_mae": mae,
        "raw_scale_rmse": rmse,
        "top_20_pct_holdout_visit_share": top_share,
        "top_20_pct_lift": top_lift,
        "total_holdout_unique_vehicle_node_hours": int(holdout_total.sum()),
    }
    (OUTPUT_DIR / "prize_validation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\nPrize validation summary")
    print(json.dumps(summary, indent=2), flush=True)
    print("\nQuintile calibration")
    print(quintile_summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
