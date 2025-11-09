import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, MeanShift, estimate_bandwidth, KMeans, AgglomerativeClustering, SpectralClustering, OPTICS

# 读取轨迹数据
df = pd.read_csv('output.csv')
df['lon'] = df['coord'].apply(lambda x: json.loads(x)['lon'])
df['lat'] = df['coord'].apply(lambda x: json.loads(x)['lat'])

# 抽样数据
sample_df = df.sample(n=10000, random_state=42)
coords = sample_df[['lon', 'lat']].to_numpy()

# 1. DBSCAN
db = DBSCAN(eps=0.0015, min_samples=10).fit(coords)
labels_dbscan = db.labels_

# 2. Mean-Shift
bandwidth = estimate_bandwidth(coords, quantile=0.2, n_samples=500)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(coords)
labels_meanshift = ms.labels_

# 3. K-Means
kmeans = KMeans(n_clusters=5, random_state=42).fit(coords)
labels_kmeans = kmeans.labels_

# 4. Agglomerative Clustering (层次聚类)
agg = AgglomerativeClustering(n_clusters=5, linkage='ward').fit(coords)
labels_agg = agg.labels_

# 5. Spectral Clustering (谱聚类)
spectral = SpectralClustering(n_clusters=5, affinity='nearest_neighbors', random_state=42).fit(coords)
labels_spectral = spectral.labels_

# 6. OPTICS
optics = OPTICS(min_samples=10, xi=0.05, min_cluster_size=0.05).fit(coords)
labels_optics = optics.labels_

# 统一可视化
fig, axs = plt.subplots(2, 3, figsize=(18, 12))
axs = axs.flatten()

# 显示函数
def plot_clusters(ax, coords, labels, title):
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        class_mask = (labels == k)
        xy = coords[class_mask]
        if k == -1:
            col = [0, 0, 0, 0.3]
        ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=4)
    ax.set_title(title)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(True)

# 分别绘制六种算法结果
plot_clusters(axs[0], coords, labels_dbscan, 'DBSCAN')
plot_clusters(axs[1], coords, labels_meanshift, 'Mean-Shift')
plot_clusters(axs[2], coords, labels_kmeans, 'K-Means (K=5)')
plot_clusters(axs[3], coords, labels_agg, 'Agglomerative Clustering')
plot_clusters(axs[4], coords, labels_spectral, 'Spectral Clustering')
plot_clusters(axs[5], coords, labels_optics, 'OPTICS')

plt.suptitle('Trajectory Clustering Comparison: Multiple Algorithms', fontsize=16)
plt.tight_layout()
plt.show()
