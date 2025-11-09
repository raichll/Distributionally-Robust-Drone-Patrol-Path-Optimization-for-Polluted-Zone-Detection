import numpy as np
from scipy.stats import beta
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import json
import pandas as pd

# 假设已读取轨迹数据
df = pd.read_csv('output.csv')
df['lon'] = df['coord'].apply(lambda x: json.loads(x)['lon'])
df['lat'] = df['coord'].apply(lambda x: json.loads(x)['lat'])
coords = df[['lon', 'lat']].sample(n=10000, random_state=42).to_numpy()

# DBSCAN聚类
db = DBSCAN(eps=0.0015, min_samples=10).fit(coords)
labels = db.labels_

# 逐簇分析
unique_clusters = set(labels)
unique_clusters.discard(-1)  # 去除噪声点

for cluster_id in unique_clusters:
    cluster_mask = (labels == cluster_id)
    cluster_points = coords[cluster_mask]

    # 计算簇中心
    center = cluster_points.mean(axis=0)

    epsilon = 1e-4  # 防止边界问题

    # 计算归一化距离
    distances = np.linalg.norm(cluster_points - center, axis=1)
    norm_distances = (distances - distances.min()) / (distances.max() - distances.min() + 1e-6)
    norm_distances = norm_distances * (1 - 2 * epsilon) + epsilon
    a, b, loc, scale = beta.fit(norm_distances, floc=0, fscale=1)

    # 可视化
    x = np.linspace(0, 1, 100)
    y = beta.pdf(x, a, b, loc, scale)

    plt.figure(figsize=(6, 4))
    plt.hist(norm_distances, bins=15, density=True, alpha=0.6, label='Normalized Distances')
    plt.plot(x, y, 'r-', lw=2, label=f'Beta PDF (α={a:.2f}, β={b:.2f})')
    plt.title(f'Cluster {cluster_id} - Beta Fit of Normalized Distances')
    plt.xlabel('Normalized Distance to Cluster Center')
    plt.ylabel('Density')
    plt.legend()
    plt.show()
# 结果写入DataFrame