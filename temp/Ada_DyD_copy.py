# === 优化后的代码：加载真实轨迹数据并进行 AdaptiveDynamicDBSCAN 聚类 ===

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from collections import deque, defaultdict
from joblib import Parallel, delayed
import pyarrow.parquet as pq

# ------------------ 1. 数据读取与预处理 ------------------

base_folder = r"E:\dwd_truck"
date_range = pd.date_range(start="2024-04-10", end="2024-04-18")
date_folders = [f"publish_date={date.strftime('%Y-%m-%d')}" for date in date_range]

def load_day_data_fast(folder_path):
    file_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.c000')]
    tables = []
    for file in file_list:
        try:
            table = pq.read_table(file, columns=['plate_num', 'published_at', 'speed', 'coord'])
            tables.append(table)
        except Exception:
            continue
    if tables:
        return pd.concat([t.to_pandas() for t in tables], ignore_index=True)
    return pd.DataFrame()

def preprocess_fast(df):
    df = df.dropna(subset=['coord', 'plate_num', 'published_at'])
    df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
    df = df.dropna(subset=['published_at'])
    
    coords = df['coord'].apply(json.loads)
    df['lon'] = coords.map(lambda x: x.get('lon'))
    df['lat'] = coords.map(lambda x: x.get('lat'))

    df['speed'] = pd.to_numeric(df['speed'], errors='coerce')
    df = df.dropna(subset=['speed', 'lon', 'lat'])

    df = df[(df['lon'] >= 102) & (df['lon'] <= 105) & (df['lat'] >= 29) & (df['lat'] <= 32)]
    df = df[(df['speed'] >= 0) & (df['speed'] <= 120)]
    return df.drop_duplicates()

def load_and_sample(folder):
    full_path = os.path.join(base_folder, folder)
    if os.path.exists(full_path):
        df = load_day_data_fast(full_path)
        if not df.empty:
            #print(f"Loading data from {folder}, records: {len(df)}")
            df = df.sample(frac=0.001, random_state=42)
            df['date'] = folder.split('=')[1]
            return df
    return None

sampled_dfs = Parallel(n_jobs=4)(delayed(load_and_sample)(folder) for folder in date_folders)
full_data = pd.concat([df for df in sampled_dfs if df is not None], ignore_index=True)
full_data = preprocess_fast(full_data)
print(full_data)
# 最多抽取10000个轨迹点
full_data = full_data.sample(n=min(10000, len(full_data)), random_state=42)
trajectory_data = full_data[['lon', 'lat']].to_numpy()
print(f"Total trajectory points: {len(trajectory_data)}")

# ------------------ 2. AdaptiveDynamicDBSCAN 类定义 ------------------
'''
class AdaptiveDynamicDBSCAN:
    def __init__(self, d, window_size=1000, epsilon_0=0.001, k_0=10, alpha=0.3):
        self.d = d
        self.window_size = window_size
        self.epsilon_0 = epsilon_0
        self.k_t = k_0
        self.alpha = alpha
        self.window = deque(maxlen=window_size)
        self.window_array = None
        self.cluster_labels = {}
        self.core_points = set()
        self.adj = defaultdict(set)
        self.index = 0
        self.id_to_point = {}
        self.hash_cache = {}

    def compute_adaptive_epsilon(self, x):
        if len(self.window) < self.k_t:
            return self.epsilon_0
        nbrs = NearestNeighbors(n_neighbors=int(self.k_t)).fit(self.window_array)
        distances, _ = nbrs.kneighbors([x])
        return max(distances[0][-1], 1e-4)

    def estimate_density(self, epsilon):
        if not self.window_array.any():
            return self.k_t
        nbrs = NearestNeighbors(radius=epsilon).fit(self.window_array)
        densities = [len(nbrs.radius_neighbors([p], return_distance=False)[0]) for p in self.window_array]
        return np.mean(densities)

    def update_k_t(self, density):
        self.k_t = self.alpha * density + (1 - self.alpha) * self.k_t

    def hash_point(self, x, epsilon, t):
        hashes = []
        for _ in range(t):
            eta = np.random.uniform(0, 2 * epsilon, self.d)
            h = tuple(np.floor((x + eta) / (2 * epsilon)).astype(int))
            hashes.append(h)
        return hashes

    def insert_point(self, x):
        self.window.append(x)
        self.window_array = np.array(self.window)
        epsilon = self.compute_adaptive_epsilon(x)
        density = self.estimate_density(epsilon)
        self.update_k_t(density)
        t_t = max(1, int(np.ceil(np.log2(len(self.window_array) + 1))))
        hash_keys = self.hash_point(x, epsilon, t_t)

        point_id = self.index
        self.index += 1
        self.id_to_point[point_id] = x
        self.hash_cache[point_id] = hash_keys

        is_core = False
        hash_key_set = set(hash_keys)
        for i, p in enumerate(self.window_array):
            other_id = point_id - len(self.window_array) + i
            if other_id in self.hash_cache:
                other_hashes = self.hash_cache[other_id]
            else:
                other_hashes = self.hash_point(p, epsilon, t_t)
                self.hash_cache[other_id] = other_hashes
            if hash_key_set & set(other_hashes):
                is_core = True
                break

        if is_core:
            self.core_points.add(point_id)
            for other_id in self.core_points:
                if other_id == point_id:
                    continue
                dist = np.linalg.norm(x - self.id_to_point[other_id])
                if dist <= epsilon:
                    self.adj[point_id].add(other_id)
                    self.adj[other_id].add(point_id)
        else:
            for other_id in self.core_points:
                dist = np.linalg.norm(x - self.id_to_point[other_id])
                if dist <= epsilon:
                    self.adj[point_id].add(other_id)
                    break

        self.cluster_labels[point_id] = x
        return point_id

    def get_connected_component(self, start_id):
        visited = set()
        stack = [start_id]
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                stack.extend(self.adj[node] - visited)
        return visited
'''
#优化之后
class AdaptiveDynamicDBSCAN:
    def __init__(self, d, window_size=1000, epsilon_0=0.001, k_0=10, alpha=0.3, update_every=5):
        self.d = d
        self.window_size = window_size
        self.epsilon_0 = epsilon_0
        self.k_t = k_0
        self.alpha = alpha
        self.update_every = update_every

        self.window = deque(maxlen=window_size)
        self.window_array = None
        self.nbrs_model = None
        self.last_window_size = 0
        self.cached_epsilon = epsilon_0

        self.index = 0
        self.id_to_point = {}
        self.hash_cache = {}
        self.core_points = set()
        self.cluster_labels = {}
        self.adj = defaultdict(set)

    def compute_adaptive_epsilon(self, x):
        if len(self.window_array) < self.k_t:
            return self.epsilon_0
        distances, _ = self.nbrs_model.kneighbors([x])
        return max(distances[0][-1], 1e-4)

    def estimate_density(self, epsilon):
        if self.window_array.shape[0] == 0:
            return self.k_t
        nbrs = NearestNeighbors(radius=epsilon).fit(self.window_array)
        all_neighbors = nbrs.radius_neighbors(self.window_array, return_distance=False)
        return np.mean([len(nlist) for nlist in all_neighbors])

    def update_k_t(self, density):
        self.k_t = self.alpha * density + (1 - self.alpha) * self.k_t

    def hash_point(self, x, epsilon, t):
        hashes = []
        for _ in range(t):
            eta = np.random.uniform(0, 2 * epsilon, self.d)
            h = tuple(np.floor((x + eta) / (2 * epsilon)).astype(int))
            hashes.append(h)
        return hashes

    def insert_point(self, x):
        self.window.append(x)
        self.window_array = np.array(self.window)
        point_id = self.index
        self.index += 1
        self.id_to_point[point_id] = x

        # 每 update_every 个点才重新构建邻居模型、估计密度和更新 epsilon
        if point_id % self.update_every == 0 or self.nbrs_model is None:
            self.nbrs_model = NearestNeighbors(n_neighbors=int(self.k_t)).fit(self.window_array)
            epsilon = self.compute_adaptive_epsilon(x)
            density = self.estimate_density(epsilon)
            self.update_k_t(density)
            self.cached_epsilon = epsilon
        else:
            epsilon = self.cached_epsilon

        t_t = max(1, int(np.ceil(np.log2(len(self.window_array) + 1))))
        hash_keys = self.hash_point(x, epsilon, t_t)
        self.hash_cache[point_id] = hash_keys
        is_core = False

        hash_key_set = set(hash_keys)
        for i, p in enumerate(self.window_array):
            other_id = point_id - len(self.window_array) + i
            if other_id in self.hash_cache:
                other_hashes = self.hash_cache[other_id]
            else:
                other_hashes = self.hash_point(p, epsilon, t_t)
                self.hash_cache[other_id] = other_hashes
            if hash_key_set & set(other_hashes):
                is_core = True
                break

        if is_core:
            self.core_points.add(point_id)
            for other_id in self.core_points:
                if other_id == point_id:
                    continue
                dist = np.linalg.norm(x - self.id_to_point[other_id])
                if dist <= epsilon:
                    self.adj[point_id].add(other_id)
                    self.adj[other_id].add(point_id)
        else:
            for other_id in self.core_points:
                dist = np.linalg.norm(x - self.id_to_point[other_id])
                if dist <= epsilon:
                    self.adj[point_id].add(other_id)
                    break

        self.cluster_labels[point_id] = x
        return point_id

    def get_connected_component(self, start_id):
        visited = set()
        stack = [start_id]
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                stack.extend(self.adj[node] - visited)
        return visited

# ------------------ 3. 聚类并可视化 ------------------

model = AdaptiveDynamicDBSCAN(d=2, window_size=1000, epsilon_0=0.001, k_0=10, alpha=0.3)
id_to_component = {}

for point in trajectory_data:
    pid = model.insert_point(point)
    component = model.get_connected_component(pid)
    component_id = min(component)
    id_to_component[pid] = component_id

components = list(set(id_to_component.values()))
component_color_map = {cid: idx for idx, cid in enumerate(components)}
colors = [component_color_map[id_to_component[i]] for i in range(len(trajectory_data))]

plt.figure(figsize=(10, 8))
plt.scatter(trajectory_data[:, 0], trajectory_data[:, 1], c=colors, cmap='tab20', s=8, alpha=0.8)
plt.title("Adaptive DynamicDBSCAN on Sampled Trajectory Points")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True)
plt.tight_layout()
plt.show()
# ------------------ 4. 输出聚类结果 ------------------ 
output_dir = os.path.join(base_folder, "clustering_results")
os.makedirs(output_dir, exist_ok=True)
result_df = pd.DataFrame({
    'point_id': list(id_to_component.keys()),
    'lon': [model.id_to_point[i][0] for i in id_to_component.keys()],
    'lat': [model.id_to_point[i][1] for i in id_to_component.keys()],
    'component_id': [id_to_component[i] for i in id_to_component.keys()]
})
result_df.to_csv(os.path.join(output_dir, "adaptive_dynamic_dbscan_results.csv"), index=False)
print(f"Clustering results saved to {os.path.join(output_dir, 'adaptive_dynamic_dbscan_results.csv')}")
