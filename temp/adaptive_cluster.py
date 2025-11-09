import numpy as np
from sklearn.neighbors import KDTree

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
        self.parent[self.find(x)] = self.find(y)

class FastAdaptiveDBSCAN:
    def __init__(self, epsilon=0.003, min_pts=10):
        self.epsilon = epsilon
        self.min_pts = min_pts
        self.labels_ = None

    def fit(self, data):
        n = data.shape[0]
        tree = KDTree(data)
        neighbors = tree.query_radius(data, r=self.epsilon)

        uf = UnionFind()
        for i in range(n):
            if len(neighbors[i]) >= self.min_pts:
                for j in neighbors[i]:
                    if i != j:
                        uf.union(i, j)

        # 分配簇编号
        cluster_map = {}
        labels = np.full(n, -1, dtype=int)
        cluster_id = 0
        for i in range(n):
            root = uf.find(i)
            if root not in cluster_map:
                cluster_map[root] = cluster_id
                cluster_id += 1
            labels[i] = cluster_map[root]
        self.labels_ = labels
        return self
