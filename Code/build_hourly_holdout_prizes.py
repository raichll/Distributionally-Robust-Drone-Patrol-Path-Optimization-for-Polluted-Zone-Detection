from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree

from build_nine_period_prizes_280 import adaptive_geospatial_parameters


EARTH_RADIUS_M = 6_371_008.8
DEFAULT_BOUNDS = (103.40, 104.55, 30.10, 31.20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construct leakage-free hourly prize vectors: discover candidates from "
            "an initial training window and reserve all later hours for evaluation."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/hourly_holdout_250"),
    )
    parser.add_argument("--sample-ratio", type=float, default=0.01)
    parser.add_argument("--speed-threshold", type=float, default=5.0)
    parser.add_argument("--candidate-count", type=int, default=250)
    parser.add_argument("--training-hours", type=int, default=168)
    parser.add_argument("--merge-radius-m", type=float, default=400.0)
    parser.add_argument("--assignment-radius-m", type=float, default=600.0)
    parser.add_argument("--minimum-training-hours", type=int, default=3)
    parser.add_argument("--minimum-candidate-spacing-m", type=float, default=250.0)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--rebuild-hourly", action="store_true")
    return parser.parse_args()


def stable_seed(base_seed: int, key: str) -> int:
    payload = f"{base_seed}|{key}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def parquet_hour(path: Path) -> pd.Timestamp:
    parquet_file = pq.ParquetFile(path)
    schema_names = parquet_file.schema.names
    column_index = schema_names.index("published_at")
    statistics = parquet_file.metadata.row_group(0).column(column_index).statistics
    if statistics is None or not statistics.has_min_max:
        table = pq.read_table(path, columns=["published_at"], use_threads=True)
        timestamp = pd.to_datetime(table.slice(0, 1).to_pandas().iloc[0, 0])
    else:
        timestamp = pd.to_datetime(statistics.min)
    return timestamp.floor("h")


def discover_hour_files(data_root: Path) -> list[tuple[pd.Timestamp, Path]]:
    records: list[tuple[pd.Timestamp, Path]] = []
    for day_dir in sorted(data_root.glob("publish_date=*")):
        for path in day_dir.glob("*.c000"):
            records.append((parquet_hour(path), path))
    records.sort(key=lambda item: item[0])
    if len(records) != 216:
        raise ValueError(f"Expected 216 hourly files, found {len(records)}")
    expected = pd.date_range(records[0][0], periods=216, freq="h")
    actual = pd.DatetimeIndex([record[0] for record in records])
    if not actual.equals(expected):
        missing = expected.difference(actual)
        duplicates = actual[actual.duplicated()].unique()
        raise ValueError(
            f"Hourly sequence is incomplete: missing={missing.tolist()}, "
            f"duplicates={duplicates.tolist()}"
        )
    return records


def sample_hour(path: Path, ratio: float, seed: int) -> tuple[pd.DataFrame, int]:
    table = pq.read_table(
        path,
        columns=["published_at", "speed", "coord"],
        use_threads=True,
    )
    total_rows = len(table)
    sample_size = max(1, int(round(total_rows * ratio)))
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(total_rows, size=sample_size, replace=False))
    frame = table.take(pa.array(indices, type=pa.int64())).to_pandas()

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
    frame = frame.dropna(subset=["speed", "longitude", "latitude"])
    frame = frame.drop_duplicates(subset=["longitude", "latitude"])
    return frame, total_rows


def hourly_clusters(
    hour: pd.Timestamp,
    path: Path,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, object]]:
    started = time.perf_counter()
    sampled, raw_rows = sample_hour(
        path,
        args.sample_ratio,
        stable_seed(args.seed, str(hour)),
    )
    lon_min, lon_max, lat_min, lat_max = DEFAULT_BOUNDS
    selected = sampled[
        sampled["speed"].between(0.0, args.speed_threshold)
        & sampled["longitude"].between(lon_min, lon_max)
        & sampled["latitude"].between(lat_min, lat_max)
    ].copy()
    coordinates = selected[["longitude", "latitude"]].to_numpy(float)
    if len(coordinates) < 10:
        return pd.DataFrame(), {
            "hour": hour,
            "raw_rows": raw_rows,
            "sampled_rows": len(sampled),
            "selected_rows": len(selected),
            "clusters": 0,
            "noise_ratio": 1.0,
            "runtime_seconds": time.perf_counter() - started,
        }

    eps, min_samples, hash_count = adaptive_geospatial_parameters(coordinates)
    labels = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="euclidean",
        algorithm="ball_tree",
        n_jobs=1,
    ).fit_predict(coordinates)
    selected["label"] = labels
    clustered = selected[selected["label"] >= 0]
    if clustered.empty:
        centers = pd.DataFrame()
    else:
        centers = (
            clustered.groupby("label", as_index=False)
            .agg(
                longitude=("longitude", "mean"),
                latitude=("latitude", "mean"),
                sampled_count=("label", "size"),
            )
        )
        centers.insert(0, "hour", hour)
    summary = {
        "hour": hour,
        "raw_rows": raw_rows,
        "sampled_rows": len(sampled),
        "selected_rows": len(selected),
        "clusters": len(centers),
        "noise_ratio": float(np.mean(labels == -1)),
        "eps_degrees": eps,
        "eps_approx_m": eps * 111_320.0,
        "min_samples": min_samples,
        "adaptive_hash_count": hash_count,
        "runtime_seconds": time.perf_counter() - started,
    }
    return centers, summary


def local_km(coordinates: np.ndarray) -> np.ndarray:
    reference_latitude = float(np.mean(coordinates[:, 1]))
    return np.column_stack(
        [
            coordinates[:, 0] * 111.32 * math.cos(math.radians(reference_latitude)),
            coordinates[:, 1] * 110.57,
        ]
    )


def haversine_distances_to_selected(
    point: np.ndarray, selected: list[np.ndarray]
) -> np.ndarray:
    if not selected:
        return np.empty(0)
    point_rad = np.deg2rad(point[[1, 0]])
    selected_array = np.asarray(selected)
    selected_rad = np.deg2rad(selected_array[:, [1, 0]])
    delta = selected_rad - point_rad
    a = (
        np.sin(delta[:, 0] / 2.0) ** 2
        + np.cos(point_rad[0])
        * np.cos(selected_rad[:, 0])
        * np.sin(delta[:, 1] / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def build_candidates(
    all_centers: pd.DataFrame,
    training_end: pd.Timestamp,
    args: argparse.Namespace,
) -> pd.DataFrame:
    training = all_centers[all_centers["hour"] <= training_end].copy()
    coordinates = training[["longitude", "latitude"]].to_numpy(float)
    merged_labels = DBSCAN(
        eps=args.merge_radius_m / 1000.0,
        min_samples=args.minimum_training_hours,
        metric="euclidean",
        algorithm="ball_tree",
        n_jobs=1,
    ).fit_predict(local_km(coordinates))
    training["merged_label"] = merged_labels
    training = training[training["merged_label"] >= 0]
    if training.empty:
        raise RuntimeError("No persistent training clusters survived temporal merging")

    grouped_rows: list[dict[str, object]] = []
    for label, group in training.groupby("merged_label"):
        weights = group["sampled_count"].to_numpy(float)
        grouped_rows.append(
            {
                "merged_label": int(label),
                "longitude": float(np.average(group["longitude"], weights=weights)),
                "latitude": float(np.average(group["latitude"], weights=weights)),
                "training_sampled_count": int(weights.sum()),
                "active_training_hours": int(group["hour"].nunique()),
                "hourly_cluster_count": int(len(group)),
            }
        )
    candidates = pd.DataFrame(grouped_rows)
    candidates["persistence_score"] = (
        np.log1p(candidates["training_sampled_count"])
        * np.sqrt(candidates["active_training_hours"])
    )
    candidates = candidates.sort_values(
        ["persistence_score", "active_training_hours", "training_sampled_count"],
        ascending=False,
    )

    selected_indices: list[int] = []
    selected_coordinates: list[np.ndarray] = []
    for index, row in candidates.iterrows():
        point = np.array([row.longitude, row.latitude], dtype=float)
        distances = haversine_distances_to_selected(point, selected_coordinates)
        if len(distances) and float(distances.min()) < args.minimum_candidate_spacing_m:
            continue
        selected_indices.append(index)
        selected_coordinates.append(point)
        if len(selected_indices) == args.candidate_count:
            break
    if len(selected_indices) < 200:
        raise RuntimeError(
            f"Only {len(selected_indices)} spatially distinct candidates were found"
        )
    selected = candidates.loc[selected_indices].copy().reset_index(drop=True)
    selected.insert(0, "node_id", np.arange(1, len(selected) + 1))
    return selected


def assign_hourly_prizes(
    all_centers: pd.DataFrame,
    candidates: pd.DataFrame,
    hours: pd.DatetimeIndex,
    assignment_radius_m: float,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    candidate_coordinates_rad = np.deg2rad(
        candidates[["latitude", "longitude"]].to_numpy(float)
    )
    tree = BallTree(candidate_coordinates_rad, metric="haversine")
    counts = np.zeros((len(hours), len(candidates)), dtype=np.int64)
    assignment_rows: list[dict[str, object]] = []
    hour_to_index = {hour: index for index, hour in enumerate(hours)}

    for hour, group in all_centers.groupby("hour"):
        hour = pd.Timestamp(hour)
        hour_index = hour_to_index[hour]
        point_rad = np.deg2rad(
            group[["latitude", "longitude"]].to_numpy(float)
        )
        distances_rad, indices = tree.query(point_rad, k=1)
        distances_m = distances_rad[:, 0] * EARTH_RADIUS_M
        assigned = distances_m <= assignment_radius_m
        weights = group["sampled_count"].to_numpy(np.int64)
        np.add.at(counts[hour_index], indices[assigned, 0], weights[assigned])
        assignment_rows.append(
            {
                "hour": hour,
                "hourly_clusters": len(group),
                "assigned_clusters": int(np.count_nonzero(assigned)),
                "assignment_rate_pct": float(100.0 * np.mean(assigned)),
                "assigned_sampled_points": int(weights[assigned].sum()),
            }
        )

    maxima = counts.max(axis=1)
    prizes = np.zeros_like(counts, dtype=float)
    nonzero = maxima > 0
    prizes[nonzero] = counts[nonzero] / maxima[nonzero, None]
    return counts, prizes, pd.DataFrame(assignment_rows)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    hourly_cache = output_dir / "hourly_cluster_centers.csv"
    summary_cache = output_dir / "hourly_processing_summary.csv"

    records = discover_hour_files(args.data_root)
    hours = pd.DatetimeIndex([record[0] for record in records])
    if not 1 <= args.training_hours < len(hours):
        raise ValueError("training-hours must lie between 1 and 215")
    training_end = hours[args.training_hours - 1]
    test_start = hours[args.training_hours]

    if hourly_cache.exists() and summary_cache.exists() and not args.rebuild_hourly:
        all_centers = pd.read_csv(hourly_cache, parse_dates=["hour"])
        summaries = pd.read_csv(summary_cache, parse_dates=["hour"])
        print(f"Loaded cached hourly clusters: {len(all_centers):,}", flush=True)
    else:
        center_frames: list[pd.DataFrame] = []
        summary_rows: list[dict[str, object]] = []
        for index, (hour, path) in enumerate(records, start=1):
            centers, summary = hourly_clusters(hour, path, args)
            if not centers.empty:
                center_frames.append(centers)
            summary_rows.append(summary)
            print(
                f"[{index:03d}/216] {hour}: selected={summary['selected_rows']:,}, "
                f"clusters={summary['clusters']}, noise={summary['noise_ratio']:.3f}",
                flush=True,
            )
            if index % 12 == 0 or index == len(records):
                pd.concat(center_frames, ignore_index=True).to_csv(
                    hourly_cache, index=False
                )
                pd.DataFrame(summary_rows).to_csv(summary_cache, index=False)
        all_centers = pd.concat(center_frames, ignore_index=True)
        summaries = pd.DataFrame(summary_rows)

    candidates = build_candidates(all_centers, training_end, args)
    counts, prizes, assignment_summary = assign_hourly_prizes(
        all_centers, candidates, hours, args.assignment_radius_m
    )

    node_columns = [f"node_{node_id}" for node_id in candidates["node_id"]]
    counts_frame = pd.DataFrame(counts, index=hours, columns=node_columns)
    counts_frame.index.name = "hour"
    prizes_frame = pd.DataFrame(prizes, index=hours, columns=node_columns)
    prizes_frame.index.name = "hour"

    candidates.to_csv(output_dir / "candidate_nodes.csv", index=False)
    counts_frame.to_csv(output_dir / "hourly_counts_216.csv")
    prizes_frame.to_csv(output_dir / "hourly_prizes_216.csv", float_format="%.8f")
    assignment_summary.to_csv(
        output_dir / "hourly_assignment_summary.csv", index=False
    )
    holdout_hours = len(hours) - args.training_hours
    np.save(
        output_dir / f"training_prizes_{args.training_hours}xn.npy",
        prizes[: args.training_hours],
    )
    np.save(
        output_dir / f"holdout_prizes_{holdout_hours}xn.npy",
        prizes[args.training_hours :],
    )
    np.save(output_dir / "all_prizes_216xn.npy", prizes)

    metadata = {
        "data_start": str(hours[0]),
        "data_end": str(hours[-1]),
        "training_hours": args.training_hours,
        "training_end": str(training_end),
        "holdout_hours": holdout_hours,
        "holdout_start": str(test_start),
        "candidate_nodes": len(candidates),
        "candidate_discovery": "hourly clustering and temporal merging using training hours only",
        "sample_ratio_per_hour": args.sample_ratio,
        "speed_threshold": args.speed_threshold,
        "coordinate_field": "coord (GCJ-02)",
        "bounds": DEFAULT_BOUNDS,
        "merge_radius_m": args.merge_radius_m,
        "assignment_radius_m": args.assignment_radius_m,
        "minimum_training_hours": args.minimum_training_hours,
        "minimum_candidate_spacing_m": args.minimum_candidate_spacing_m,
        "normalization": "hourly assigned count divided by the maximum node count in that hour",
        "zero_count_hours": int(np.count_nonzero(counts.max(axis=1) == 0)),
        "mean_assignment_rate_pct": float(
            assignment_summary["assignment_rate_pct"].mean()
        ),
        "processing": {
            "total_raw_rows": int(summaries["raw_rows"].sum()),
            "total_sampled_rows": int(summaries["sampled_rows"].sum()),
            "total_selected_rows": int(summaries["selected_rows"].sum()),
            "total_hourly_clusters": int(summaries["clusters"].sum()),
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
