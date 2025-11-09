import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.cluster import DBSCAN
import pyarrow.parquet as pq
from sklearn.neighbors import NearestNeighbors, KDTree # 引入KDTree
from collections import deque, defaultdict


# ========== 参数配置 ==========
base_folder = r"E:\dwd_truck"
cache_file = "trajectory_data.npy"
sample_number_port= 0.01  # 抽样比例
n_points = 100000  #最终个数
n_jobs = 4

# ========== 数据处理函数 ==========
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
            df = df.sample(frac=sample_number_port, random_state=42)
            df['date'] = folder.split('=')[1]
            return df
    return None

# ========== 数据加载模块 ==========
if os.path.exists(cache_file):
    print(f"✅ 已找到缓存数据 {cache_file}，正在加载...")
    trajectory_data = np.load(cache_file)
else:
    print("🚀 第一次运行，开始处理原始数据...")
    date_range = pd.date_range(start="2024-04-10", end="2024-04-18")
    date_folders = [f"publish_date={date.strftime('%Y-%m-%d')}" for date in date_range]
    sampled_dfs = Parallel(n_jobs=n_jobs)(delayed(load_and_sample)(folder) for folder in date_folders)
    full_data = pd.concat([df for df in sampled_dfs if df is not None], ignore_index=True)
    full_data = preprocess_fast(full_data)
    #full_data = full_data.sample(n=min(n_points, len(full_data)), random_state=42)
    trajectory_data = full_data[['lon', 'lat']].to_numpy()
    np.save(cache_file, trajectory_data)
    print(f"✅ 已保存缓存数据到 {cache_file}")

print(f"📌 轨迹数据加载完成，共 {trajectory_data.shape[0]} 个点")

# ========== 方法1：DBSCAN ==========
start_time = time.time()
db = DBSCAN(eps=0.001, min_samples=200, n_jobs=n_jobs).fit(trajectory_data)
dbscan_labels = db.labels_
dbscan_time = time.time() - start_time
print(f"[DBSCAN] 聚类耗时: {dbscan_time:.2f} 秒")

# ========== 方法2：AdaptiveDynamicDBSCAN + 并查集 (优化版) ==========
'''
class AdaptiveDynamicDBSCAN:
    def __init__(self, d, window_size=1000, epsilon_0=0.001, k_0=10, alpha=0.3, update_every=50):
        self.d = d
        self.window_size = window_size
        self.epsilon_0 = epsilon_0
        self.k_t = k_0
        self.alpha = alpha
        self.update_every = update_every

        self.window = deque(maxlen=window_size) # 存储 (point_id, coord)
        self.nbrs_model = None # 存储KDTree模型
        self.cached_epsilon = epsilon_0

        self.index = 0
        self.id_to_point = {}
        self.core_points = set()
        self.adj = defaultdict(set)

    def compute_adaptive_epsilon(self, x):
        if len(self.window) < self.k_t: # 使用 len(self.window) 判断窗口大小
            return self.epsilon_0
        # 使用 KDTree.query 方法进行 k 近邻查询
        # query 返回 (distances, indices)
        distances, _ = self.nbrs_model.query([x], k=int(self.k_t))
        return max(distances[0][-1], 1e-4)

    def estimate_density(self, epsilon):
        if len(self.window) == 0:
            return self.k_t
        
        current_window_coords = np.array([item[1] for item in self.window]) # 从deque中提取坐标
        if current_window_coords.shape[0] == 0:
             return self.k_t
            
        if self.nbrs_model is None or self.nbrs_model.data.shape[0] == 0:
            # 如果模型未初始化或数据为空，需要重新fit
            self.nbrs_model = KDTree(current_window_coords) # 确保模型是最新的
            
        all_neighbors_indices = self.nbrs_model.query_radius(current_window_coords, r=epsilon)
        return np.mean([len(nlist) for nlist in all_neighbors_indices])

    def update_k_t(self, density):
        self.k_t = self.alpha * density + (1 - self.alpha) * self.k_t
        self.k_t = max(1, self.k_t)

    def hash_point(self, x, epsilon, t):
        hashes = []
        for _ in range(t):
            eta = np.random.uniform(0, 2 * epsilon, self.d)
            h = tuple(np.floor((x + eta) / (2 * epsilon)).astype(int))
            hashes.append(h)
        return hashes

    def insert_point(self, x):
        point_id = self.index
        self.index += 1
        self.id_to_point[point_id] = x
        self.window.append((point_id, x)) # 存储 (point_id, coord)

        # 提取当前窗口的坐标，用于构建KDTree
        current_window_coords = np.array([item[1] for item in self.window])
        
        # 每 update_every 个点才重新构建邻居模型、估计密度和更新 epsilon
        # 或者在窗口大小变化时（即刚开始填满窗口时）
        if (point_id % self.update_every == 0) or (len(self.window) < self.window_size and len(self.window) % 10 == 0) :
            if len(self.window) > 0:
                self.nbrs_model = KDTree(current_window_coords)
                epsilon = self.compute_adaptive_epsilon(x)
                density = self.estimate_density(epsilon)
                self.update_k_t(density)
                self.cached_epsilon = epsilon
            else:
                epsilon = self.epsilon_0
        else:
            epsilon = self.cached_epsilon
        
        epsilon = max(epsilon, 1e-4)

        t_t = max(1, int(np.ceil(np.log2(len(self.window) + 1))))

        is_core = False
        
        if self.nbrs_model and len(self.window) > 1:
            epsilon_neighbors_indices = self.nbrs_model.query_radius([x], r=epsilon)[0]
            
            num_epsilon_neighbors = 0
            for idx in epsilon_neighbors_indices:
                neighbor_info = self.window[idx] # 获取deque中对应索引的(id, coord)
                neighbor_original_pid = neighbor_info[0]
                if neighbor_original_pid != point_id:
                    num_epsilon_neighbors += 1
            
            if num_epsilon_neighbors >= self.k_t:
                is_core = True
                self.core_points.add(point_id)
            
            # 连接逻辑
            for idx in epsilon_neighbors_indices:
                neighbor_info = self.window[idx]
                neighbor_original_pid = neighbor_info[0]
                
                if neighbor_original_pid == point_id:
                    continue
                
                # 如果x是核心点，或者邻居是核心点，则建立连接
                if is_core or (neighbor_original_pid in self.core_points):
                    self.adj[point_id].add(neighbor_original_pid)
                    self.adj[neighbor_original_pid].add(point_id)
        
        return point_id

class UnionFind:
    def __init__(self):
        self.parent = dict()

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            self.parent[rootX] = rootY

model = AdaptiveDynamicDBSCAN(d=2, window_size=1000, epsilon_0=0.001, k_0=10, alpha=0.3, update_every=50)
uf = UnionFind()
point_ids = []

start_time = time.time()
for i, point in enumerate(trajectory_data):
    pid = model.insert_point(point)
    point_ids.append(pid)

for pid, neighbors in model.adj.items():
    for nid in neighbors:
        uf.union(pid, nid)

adaptive_labels = [uf.find(pid) for pid in point_ids]
adaptive_time = time.time() - start_time
print(f"[AdaptiveDynamicDBSCAN] 聚类耗时: {adaptive_time:.2f} 秒")

# ========== 保存聚类时间 ==========
time_df = pd.DataFrame({
    'Algorithm': ['DBSCAN', 'AdaptiveDynamicDBSCAN'],
    'Time(s)': [dbscan_time, adaptive_time]
})
time_df.to_csv("clustering_time_compare.csv", index=False)'''

# ========== 可视化 ==========
plt.figure(figsize=(12, 5))

# DBSCAN 可视化
plt.subplot(1, 2, 1)
dbscan_labels = np.array(dbscan_labels)
noise_mask = dbscan_labels == -1
cluster_mask = dbscan_labels != -1

'''plt.scatter(trajectory_data[noise_mask, 0], trajectory_data[noise_mask, 1],
            c='blue', s=8, label='Noise')'''

plt.scatter(trajectory_data[cluster_mask, 0], trajectory_data[cluster_mask, 1],
            c='red', s=20, label='Clustered Points')

plt.title(f"DBSCAN (Time: {dbscan_time:.2f}s)")
plt.legend()

'''# AdaptiveDynamicDBSCAN 可视化
plt.subplot(1, 2, 2)
adaptive_labels = np.array(adaptive_labels)
noise_mask = adaptive_labels == -1
cluster_mask = adaptive_labels != -1'''

'''plt.scatter(trajectory_data[noise_mask, 0], trajectory_data[noise_mask, 1],
            c='blue', s=8, label='Noise')'''

'''plt.scatter(trajectory_data[cluster_mask, 0], trajectory_data[cluster_mask, 1],
            c='red', s=20, label='Clustered Points')'''

'''plt.title(f"AdaptiveDynamicDBSCAN (Time: {adaptive_time:.2f}s)")
plt.legend()'''

plt.suptitle("Clustering Comparison", fontsize=14)
plt.tight_layout()
plt.savefig("clustering_comparison.png", dpi=300)
plt.show()


