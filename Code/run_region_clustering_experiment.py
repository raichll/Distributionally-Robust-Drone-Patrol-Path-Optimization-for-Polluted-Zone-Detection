import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def install_stag_compat():
    stag = types.ModuleType("stag")
    graph = types.ModuleType("stag.graph")
    cluster = types.ModuleType("stag.cluster")
    random_mod = types.ModuleType("stag.random")

    class Graph:
        def __init__(self, adjacency=None):
            self.adjacency = adjacency

        def number_of_vertices(self):
            return 0 if self.adjacency is None else self.adjacency.shape[0]

    def connected_components(g):
        from scipy.sparse.csgraph import connected_components as scipy_cc

        n_components, labels = scipy_cc(g.adjacency, directed=False, return_labels=True)
        return [np.where(labels == i)[0].tolist() for i in range(n_components)]

    graph.Graph = Graph
    cluster.connected_components = connected_components
    random_mod.sbm = lambda *args, **kwargs: Graph()
    random_mod.sbm_gt_labels = lambda *args, **kwargs: []
    stag.graph = graph
    stag.cluster = cluster
    stag.random = random_mod
    sys.modules["stag"] = stag
    sys.modules["stag.graph"] = graph
    sys.modules["stag.cluster"] = cluster
    sys.modules["stag.random"] = random_mod


def main():
    install_stag_compat()
    repo = Path(__file__).resolve().parent
    dyn_dir = repo / "dynamic_dbscan-main"
    sys.path.insert(0, str(dyn_dir))

    from alglab.dataset import DynamicPointCloudDataset
    from algorithms import (
        Ada_dynamic_dbscan_alg,
        dynamic_dbscan_alg,
        fdbscan_dynamic_naive,
        fdbscan_dynamic_noncore,
        sklearn_dynamic_alg,
    )

    data_path = repo / "clu_data" / "clustering_dataset_20251215_120000_multicluster_region.csv"
    if len(sys.argv) > 2:
        data_path = Path(sys.argv[2]).resolve()
    df = pd.read_csv(data_path)
    sample_n = int(sys.argv[1]) if len(sys.argv) > 1 else len(df)
    if sample_n < len(df):
        df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)

    x = df[["lon", "lat"]].to_numpy(float)
    y = df["label"].to_numpy(int)
    dataset = DynamicPointCloudDataset(x, [(list(range(len(x))), [])], labels=y)

    # Fixed parameters follow the previous truck experiment scale; AD-DBSCAN uses
    # the adaptive parameter estimator from the existing code.
    fixed_eps = 0.001013
    fixed_min_samples = 5
    fixed_t = 12
    def adaptive_params(points):
        from sklearn.neighbors import NearestNeighbors

        n = len(points)
        k = int(min(max(5, 8 * 0.8 + 0.2 * (np.log(n) - np.log(10000))), 12))
        t = int(min(10 + (np.log(n) - np.log(10000)) / np.log(2), 20))
        t = max(6, t)
        sample = points
        if n > 50000:
            rng = np.random.default_rng(42)
            sample = points[rng.choice(n, 50000, replace=False)]
        nbrs = NearestNeighbors(n_neighbors=min(max(k, 2), len(sample) - 1), metric="euclidean").fit(sample)
        distances, _ = nbrs.kneighbors(sample)
        k_dist = np.sort(distances[:, min(k - 1, distances.shape[1] - 1)])
        eps = float(np.quantile(k_dist, 0.70))
        eps = float(np.clip(eps, 0.0002, 0.01))
        return eps, k, t

    ada_eps, ada_min_samples, ada_t = adaptive_params(x)

    algorithms = [
        ("Dynamic DBSCAN", dynamic_dbscan_alg, {"eps": fixed_eps, "min_samples": fixed_min_samples, "t": fixed_t, "d": 2}),
        ("Fdbscan Naive", fdbscan_dynamic_naive, {"eps": fixed_eps, "min_samples": fixed_min_samples, "t": fixed_t}),
        ("Fdbscan Noncore", fdbscan_dynamic_noncore, {"eps": fixed_eps, "min_samples": fixed_min_samples, "t": fixed_t, "burnin": 1}),
        ("Sklearn Dynamic", sklearn_dynamic_alg, {"eps": fixed_eps, "min_samples": fixed_min_samples}),
        ("AD-DBSCAN", Ada_dynamic_dbscan_alg, {"eps": ada_eps, "min_samples": ada_min_samples, "t": ada_t, "d": 2}),
    ]

    rows = []
    for name, alg, params in algorithms:
        t0 = time.perf_counter()
        labels = None
        error = None
        try:
            for labels in alg.run(dataset, params):
                pass
            labels = np.asarray(labels)
            elapsed = time.perf_counter() - t0
            ari = adjusted_rand_score(y, labels)
            nmi = normalized_mutual_info_score(y, labels)
            n_clusters = len(set(labels.tolist())) - (1 if -1 in set(labels.tolist()) else 0)
            n_noise = int(np.sum(labels == -1))
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            ari = np.nan
            nmi = np.nan
            n_clusters = np.nan
            n_noise = np.nan
            error = repr(exc)

        row = {
            "algorithm": name,
            "n": len(x),
            "time": elapsed,
            "ari": ari,
            "nmi": nmi,
            "clusters": n_clusters,
            "noise": n_noise,
            "params": params,
            "error": error,
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    out_dir = repo / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / f"chengdu_region_multicluster_clustering_5alg_{len(x)}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"saved={out_csv}")


if __name__ == "__main__":
    main()
