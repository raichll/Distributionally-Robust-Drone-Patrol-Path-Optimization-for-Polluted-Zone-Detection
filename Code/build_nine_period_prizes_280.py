from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree, NearestNeighbors


EARTH_RADIUS_M = 6_371_008.8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build nine period-wise prize vectors for 280 fixed candidate nodes."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("trajectory_points_construction_ledger.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/nine_period_prizes_280"),
    )
    parser.add_argument("--sample-ratio", type=float, default=0.005)
    parser.add_argument("--selection-radius-m", type=float, default=750.0)
    parser.add_argument("--assignment-radius-m", type=float, default=500.0)
    parser.add_argument("--seed", type=int, default=20240410)
    return parser.parse_args()


def load_candidates(path: Path) -> pd.DataFrame:
    # The source CSV has trailing unnamed columns, so only the first four are needed.
    candidates = pd.read_csv(path, encoding="gb18030", usecols=range(4))
    candidates.columns = ["visit_order", "node_id", "longitude", "latitude"]
    candidates = candidates.apply(pd.to_numeric, errors="raise")
    if len(candidates) != 280:
        raise ValueError(f"Expected 280 fixed candidates, found {len(candidates)}")
    if candidates[["longitude", "latitude"]].duplicated().any():
        raise ValueError("Candidate coordinates must be unique")
    return candidates


def stable_seed(base_seed: int, date: str, file_name: str) -> int:
    key = f"{base_seed}|{date}|{file_name}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little")


def sample_parquet_file(
    path: Path, sample_ratio: float, base_seed: int, date: str
) -> tuple[pd.DataFrame, int]:
    table = pq.read_table(
        path,
        columns=["plate_num", "published_at", "speed", "coord"],
        use_threads=True,
    )
    total_rows = len(table)
    sample_size = max(1, int(round(total_rows * sample_ratio)))
    rng = np.random.default_rng(stable_seed(base_seed, date, path.name))
    indices = np.sort(rng.choice(total_rows, size=sample_size, replace=False))
    sampled = table.take(pa.array(indices, type=pa.int64())).to_pandas()
    return sampled, total_rows


def parse_and_clean_coordinates(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.dropna(subset=["coord", "plate_num", "published_at", "speed"]).copy()
    frame["speed"] = pd.to_numeric(frame["speed"], errors="coerce")
    frame["published_at"] = pd.to_datetime(frame["published_at"], errors="coerce")

    coord_text = frame["coord"].astype(str)
    frame["longitude"] = pd.to_numeric(
        coord_text.str.extract(r'"lon"\s*:\s*(-?\d+(?:\.\d+)?)', expand=False),
        errors="coerce",
    )
    frame["latitude"] = pd.to_numeric(
        coord_text.str.extract(r'"lat"\s*:\s*(-?\d+(?:\.\d+)?)', expand=False),
        errors="coerce",
    )

    frame = frame.dropna(subset=["published_at", "speed", "longitude", "latitude"])
    frame = frame[
        frame["longitude"].between(102.0, 105.0)
        & frame["latitude"].between(29.0, 32.0)
        & frame["speed"].between(0.0, 120.0)
    ]
    return frame.drop_duplicates().reset_index(drop=True)


def load_daily_sample(
    day_dir: Path, sample_ratio: float, base_seed: int
) -> tuple[pd.DataFrame, int, int]:
    date = day_dir.name.split("=", maxsplit=1)[-1]
    sampled_parts: list[pd.DataFrame] = []
    total_rows = 0
    files = sorted(day_dir.glob("*.c000"))
    if not files:
        raise FileNotFoundError(f"No .c000 files found in {day_dir}")

    for file_index, path in enumerate(files, start=1):
        sampled, file_rows = sample_parquet_file(path, sample_ratio, base_seed, date)
        sampled_parts.append(sampled)
        total_rows += file_rows
        print(
            f"  {date}: read file {file_index:02d}/{len(files)}, "
            f"rows={file_rows:,}, sampled={len(sampled):,}",
            flush=True,
        )

    combined = pd.concat(sampled_parts, ignore_index=True)
    cleaned = parse_and_clean_coordinates(combined)
    return cleaned, total_rows, len(combined)


def adaptive_geospatial_parameters(coordinates: np.ndarray) -> tuple[float, int, int]:
    """Apply the manuscript's scale- and density-adaptive parameter rules."""
    n = len(coordinates)
    if n < 5:
        raise ValueError("At least five nearby points are required for clustering")

    log_n = np.log(n)
    min_samples = int(
        min(max(5, 8 * 0.8 + 0.2 * (log_n - np.log(10_000))), 12)
    )
    hash_count = int(min(10 + (log_n - np.log(10_000)) / np.log(2), 20))
    hash_count = max(1, hash_count)

    rng = np.random.default_rng(42)
    if n > 50_000:
        parameter_sample = coordinates[
            rng.choice(n, size=50_000, replace=False)
        ]
    else:
        parameter_sample = coordinates

    nearest = NearestNeighbors(
        n_neighbors=min_samples,
        metric="euclidean",
        algorithm="ball_tree",
        n_jobs=1,
    ).fit(parameter_sample)
    distances, _ = nearest.kneighbors(parameter_sample)
    k_distances = distances[:, min_samples - 1]
    k_distances = k_distances[np.isfinite(k_distances)]
    if len(k_distances) < 10:
        raise ValueError("Insufficient finite neighbor distances for adaptive radius")

    nearest_one = NearestNeighbors(
        n_neighbors=min(5, len(parameter_sample) - 1),
        metric="euclidean",
        algorithm="ball_tree",
        n_jobs=1,
    ).fit(parameter_sample)
    nearest_distances, _ = nearest_one.kneighbors(parameter_sample)
    mean_nearest_distance = float(np.mean(nearest_distances[:, 1]))
    density = n / max(float(np.ptp(coordinates[:, 0]) * np.ptp(coordinates[:, 1])), 1e-12)
    density_factor = np.log10(max(1.0, density))
    size_factor = np.log10(n / 1_000) if n > 1_000 else 0.0
    base_eps = mean_nearest_distance * 1.5
    eps_min = max(0.001, base_eps * (0.5 - 0.1 * density_factor))
    eps_max = min(0.5, base_eps * (3.0 + 0.2 * size_factor))
    if eps_max <= eps_min:
        eps_max = eps_min * 2.0

    density_quantile = 0.7 if n > 50_000 else 0.6
    raw_eps = float(
        np.median(
            [
                np.quantile(k_distances, density_quantile),
                np.median(k_distances) * 1.2,
                np.percentile(k_distances, 75),
            ]
        )
    )
    data_min = float(k_distances.min())
    data_max = float(k_distances.max())
    if data_max - data_min > 1e-6:
        relative_position = (raw_eps - data_min) / (data_max - data_min)
        eps = eps_min + relative_position * (eps_max - eps_min) * 0.8
    else:
        eps = (eps_min + eps_max) / 2.0
    eps = float(np.clip(eps, eps_min, eps_max))
    return round(eps, 6), min_samples, hash_count


def cluster_and_count(
    points: pd.DataFrame,
    candidate_tree: BallTree,
    selection_radius_m: float,
    assignment_radius_m: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    coordinates = points[["longitude", "latitude"]].to_numpy(dtype=np.float64)
    coordinates_rad = np.deg2rad(coordinates[:, [1, 0]])
    distances_rad, nearest_indices = candidate_tree.query(coordinates_rad, k=1)
    nearest_distances_m = distances_rad[:, 0] * EARTH_RADIUS_M
    nearest_indices = nearest_indices[:, 0]

    selected_mask = nearest_distances_m <= selection_radius_m
    selected = coordinates[selected_mask]
    selected_nearest_indices = nearest_indices[selected_mask]
    selected_nearest_distances_m = nearest_distances_m[selected_mask]
    if len(selected) == 0:
        raise ValueError("No sampled points fall near the fixed candidate nodes")

    eps, min_samples, hash_count = adaptive_geospatial_parameters(selected)
    labels = DBSCAN(
        eps=float(eps),
        min_samples=int(min_samples),
        metric="euclidean",
        algorithm="ball_tree",
        n_jobs=1,
    ).fit_predict(selected)

    valid_mask = (labels != -1) & (
        selected_nearest_distances_m <= assignment_radius_m
    )
    counts = np.bincount(
        selected_nearest_indices[valid_mask], minlength=280
    ).astype(np.int64)
    unique_labels = np.unique(labels[labels != -1])
    summary = {
        "nearby_sampled_points": int(len(selected)),
        "clustered_nonnoise_points": int(np.count_nonzero(labels != -1)),
        "assigned_clustered_points": int(np.count_nonzero(valid_mask)),
        "clusters": int(len(unique_labels)),
        "noise_ratio": float(np.mean(labels == -1)),
        "eps_degrees": float(eps),
        "eps_approx_m": float(eps) * 111_320.0,
        "min_samples": int(min_samples),
        "adaptive_hash_count": int(hash_count),
        "nonzero_candidate_nodes": int(np.count_nonzero(counts)),
        "maximum_candidate_count": int(counts.max(initial=0)),
    }
    return counts, summary


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = load_candidates(args.candidates)
    candidate_coordinates_rad = np.deg2rad(
        candidates[["latitude", "longitude"]].to_numpy(dtype=np.float64)
    )
    candidate_tree = BallTree(candidate_coordinates_rad, metric="haversine")
    day_dirs = sorted(args.data_root.glob("publish_date=*"))
    if len(day_dirs) != 9:
        raise ValueError(f"Expected 9 daily partitions, found {len(day_dirs)}")

    dates: list[str] = []
    counts_by_day: list[np.ndarray] = []
    summaries: list[dict[str, object]] = []

    for day_index, day_dir in enumerate(day_dirs, start=1):
        date = day_dir.name.split("=", maxsplit=1)[-1]
        dates.append(date)
        print(f"\n[{day_index}/9] Processing {date}", flush=True)
        started = time.perf_counter()
        sample, total_rows, sampled_rows = load_daily_sample(
            day_dir, args.sample_ratio, args.seed
        )
        counts, clustering_summary = cluster_and_count(
            sample,
            candidate_tree,
            args.selection_radius_m,
            args.assignment_radius_m,
        )
        elapsed = time.perf_counter() - started
        counts_by_day.append(counts)
        summaries.append(
            {
                "date": date,
                "raw_rows": total_rows,
                "sampled_rows_before_cleaning": sampled_rows,
                "sampled_rows_after_cleaning": len(sample),
                "sample_ratio": args.sample_ratio,
                **clustering_summary,
                "runtime_seconds": elapsed,
            }
        )
        print(
            f"  completed {date}: clean sample={len(sample):,}, "
            f"near candidates={clustering_summary['nearby_sampled_points']:,}, "
            f"nonzero nodes={clustering_summary['nonzero_candidate_nodes']}, "
            f"runtime={elapsed:.1f}s",
            flush=True,
        )

    count_matrix = np.vstack(counts_by_day)
    maxima = count_matrix.max(axis=1)
    if np.any(maxima == 0):
        zero_dates = [dates[i] for i in np.flatnonzero(maxima == 0)]
        raise ValueError(f"Cannot normalize zero-count periods: {zero_dates}")
    prize_matrix = count_matrix / maxima[:, None]

    node_columns = [f"node_{int(node_id)}" for node_id in candidates["node_id"]]
    count_frame = pd.DataFrame(count_matrix, index=dates, columns=node_columns)
    count_frame.index.name = "date"
    prize_frame = pd.DataFrame(prize_matrix, index=dates, columns=node_columns)
    prize_frame.index.name = "date"

    long_frame = candidates.copy()
    for period_index, date in enumerate(dates):
        long_frame[f"count_{date}"] = count_matrix[period_index]
        long_frame[f"prize_{date}"] = prize_matrix[period_index]
    long_frame["empirical_mean_prize"] = prize_matrix.mean(axis=0)
    long_frame["nonzero_periods"] = np.count_nonzero(count_matrix, axis=0)

    count_frame.to_csv(output_dir / "counts_9x280.csv", encoding="utf-8-sig")
    prize_frame.to_csv(
        output_dir / "prize_matrix_9x280.csv",
        encoding="utf-8-sig",
        float_format="%.8f",
    )
    long_frame.to_csv(
        output_dir / "candidate_prizes_280x9.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.8f",
    )
    pd.DataFrame(summaries).to_csv(
        output_dir / "daily_processing_summary.csv", index=False, encoding="utf-8-sig"
    )
    np.save(output_dir / "empirical_prize_samples_9x280.npy", prize_matrix)

    metadata = {
        "dates": dates,
        "candidate_nodes": 280,
        "empirical_samples": 9,
        "sample_ratio": args.sample_ratio,
        "sampling_seed": args.seed,
        "coordinate_field": "coord (GCJ-02)",
        "selection_radius_m": args.selection_radius_m,
        "assignment_radius_m": args.assignment_radius_m,
        "normalization": "N_ti / max_j N_tj within each date",
        "empirical_probability_per_sample": 1.0 / 9.0,
        "files": {
            "count_matrix": "counts_9x280.csv",
            "prize_matrix": "prize_matrix_9x280.csv",
            "candidate_wide_table": "candidate_prizes_280x9.csv",
            "numpy_matrix": "empirical_prize_samples_9x280.npy",
            "processing_summary": "daily_processing_summary.csv",
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nOutputs written to", output_dir, flush=True)
    print(
        "Prize matrix checks:",
        f"shape={prize_matrix.shape},",
        f"min={prize_matrix.min():.6f},",
        f"max={prize_matrix.max():.6f},",
        f"finite={np.isfinite(prize_matrix).all()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
