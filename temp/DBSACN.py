import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from collections import defaultdict
import networkx as nx

# ------------------------------------------
# 动态 DBSCAN 模拟器类
# ------------------------------------------
class DynamicDBSCAN:
    def __init__(self, eps=0.75, min_samples=5, t=5):
        self.eps = eps
        self.k = min_samples
        self.t = t
        self.buckets = [defaultdict(list) for _ in range(t)]
        self.graph = nx.Graph()
        self.points = []
        self.labels = []
        self.core_set = set()
        self.next_id = 0

    def _hash(self, x, t_idx):
        """哈希函数：将点划分到桶中"""
        shift = np.random.uniform(0, 2 * self.eps, size=x.shape)
        return tuple(np.floor((x + shift) / (2 * self.eps)).astype(int))

    def add_point(self, x):
        x = np.array(x)
        idx = self.next_id
        self.next_id += 1
        self.points.append(x)
        is_core = False

        collisions = []
        for i in range(self.t):
            h = self._hash(x, i)
            self.buckets[i][h].append(idx)
            collisions.extend(self.buckets[i][h])

        for i in range(self.t):
            h = self._hash(x, i)
            if len(self.buckets[i][h]) >= self.k:
                is_core = True
                break

        self.graph.add_node(idx)
        if is_core:
            self.core_set.add(idx)
            for i in range(self.t):
                h = self._hash(x, i)
                for j in self.buckets[i][h]:
                    if j != idx and j in self.core_set:
                        self.graph.add_edge(idx, j)
        else:
            for j in set(collisions):
                if j in self.core_set:
                    self.graph.add_edge(idx, j)
                    break

        self._update_labels()

    def _update_labels(self):
        clusters = list(nx.connected_components(self.graph))
        label_dict = {}
        for label, component in enumerate(clusters):
            for node in component:
                label_dict[node] = label
        self.labels = [label_dict.get(i, -1) for i in range(len(self.points))]

    def get_labels(self):
        return self.labels

    def get_points(self):
        return np.array(self.points)


# ------------------------------------------
# 生成虚拟数据（两个簇 + 噪声）
# ------------------------------------------
np.random.seed(0)
cluster1 = np.random.randn(50, 2) * 0.4 + np.array([2, 2])
cluster2 = np.random.randn(50, 2) * 0.4 + np.array([-2, -2])
noise = np.random.uniform(low=-5, high=5, size=(20, 2))
X_total = np.vstack((cluster1, cluster2, noise))

# ------------------------------------------
# 静态 DBSCAN
# ------------------------------------------
dbscan_static = DBSCAN(eps=0.75, min_samples=5)
labels_static = dbscan_static.fit_predict(X_total)

# ------------------------------------------
# 动态 DBSCAN 模拟器
# ------------------------------------------
dynamic_model = DynamicDBSCAN(eps=0.75, min_samples=5, t=5)
for point in X_total:
    dynamic_model.add_point(point)

labels_dynamic = dynamic_model.get_labels()

# ------------------------------------------
# 可视化对比
# ------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].scatter(X_total[:, 0], X_total[:, 1], c=labels_static, cmap='tab10', s=20)
axes[0].set_title("静态 DBSCAN 聚类结果")
axes[0].grid(True)

axes[1].scatter(X_total[:, 0], X_total[:, 1], c=labels_dynamic, cmap='tab10', s=20)
axes[1].set_title("动态 DBSCAN（模拟器）聚类结果")
axes[1].grid(True)

plt.tight_layout()
plt.show()
