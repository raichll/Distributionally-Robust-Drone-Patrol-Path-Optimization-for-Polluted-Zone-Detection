import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
import torch
from mamba_ssm import Mamba

# -------------------------------
# 1. 数据读取与停留点提取
# -------------------------------

# 读取轨迹数据，确保包含以下列：['vehicle_id', 'timestamp', 'longitude', 'latitude', 'speed']
df = pd.read_csv("vehicle_trajectory.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 停留点判定：速度小于阈值
SPEED_THRESHOLD = 3  # km/h
df['stopped'] = df['speed'] < SPEED_THRESHOLD

# 聚合停留点，按车辆每分钟统计
stop_points = df[df['stopped']].groupby('vehicle_id').resample('1min', on='timestamp').mean().dropna()
stop_locations = stop_points[['longitude', 'latitude']].reset_index()

# -------------------------------
# 2. 特征标准化
# -------------------------------

scaler = StandardScaler()
coords_scaled = scaler.fit_transform(stop_locations[['longitude', 'latitude']])

# -------------------------------
# 3. Mamba特征提取
# -------------------------------

sequence = torch.tensor(coords_scaled, dtype=torch.float32).unsqueeze(0)  # [1, N, 2]

model = Mamba(d_model=64, n_layers=2, vocab_size=None)

with torch.no_grad():
    mamba_output = model(sequence)

mamba_features = mamba_output.squeeze(0).numpy()  # [N, 64]

# -------------------------------
# 4. 聚类分析
# -------------------------------

# ① KMeans 基于原始坐标
kmeans_coords = KMeans(n_clusters=5, random_state=42)
labels_kmeans_coords = kmeans_coords.fit_predict(coords_scaled)

# ② Mamba特征 + KMeans
kmeans_mamba = KMeans(n_clusters=5, random_state=42)
labels_mamba_kmeans = kmeans_mamba.fit_predict(mamba_features)

# ③ Mamba特征降维 + DBSCAN
pca = PCA(n_components=2)
features_2d = pca.fit_transform(mamba_features)

dbscan = DBSCAN(eps=0.5, min_samples=5)
labels_dbscan = dbscan.fit_predict(features_2d)

# 结果写入DataFrame
stop_locations['cluster_kmeans_coords'] = labels_kmeans_coords
stop_locations['cluster_mamba_kmeans'] = labels_mamba_kmeans
stop_locations['cluster_dbscan'] = labels_dbscan

# -------------------------------
# 5. 聚类效果可视化
# -------------------------------

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# KMeans (原始坐标)
axes[0].scatter(stop_locations['longitude'], stop_locations['latitude'], c=labels_kmeans_coords, cmap='tab10', s=10)
axes[0].set_title("KMeans (原始坐标聚类)")
axes[0].set_xlabel("经度")
axes[0].set_ylabel("纬度")

# Mamba特征 + KMeans
axes[1].scatter(stop_locations['longitude'], stop_locations['latitude'], c=labels_mamba_kmeans, cmap='tab10', s=10)
axes[1].set_title("Mamba特征 + KMeans聚类")
axes[1].set_xlabel("经度")
axes[1].set_ylabel("纬度")

# Mamba特征降维 + DBSCAN
axes[2].scatter(stop_locations['longitude'], stop_locations['latitude'], c=labels_dbscan, cmap='tab10', s=10)
axes[2].set_title("Mamba特征 + DBSCAN聚类")
axes[2].set_xlabel("经度")
axes[2].set_ylabel("纬度")

plt.tight_layout()
plt.show()
