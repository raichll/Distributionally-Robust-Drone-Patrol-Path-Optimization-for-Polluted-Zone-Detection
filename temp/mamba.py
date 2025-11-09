import pandas as pd
import numpy as np
import json
from sklearn.cluster import DBSCAN
from scipy.stats import beta
import matplotlib.pyplot as plt

# 1. 数据准备
df = pd.read_csv('output.csv')

# 提取经纬度
df['lon'] = df['coord'].apply(lambda x: json.loads(x)['lon'])
df['lat'] = df['coord'].apply(lambda x: json.loads(x)['lat'])
df['timestamp'] = pd.to_datetime(df['published_at'])

# 2. 抽样加速
df = df.sample(n=30000, random_state=42)

# 3. DBSCAN聚类
coords = df[['lon', 'lat']].to_numpy()
db = DBSCAN(eps=0.0015, min_samples=10).fit(coords)
df['cluster'] = db.labels_

# 4. 逐簇统计停留时间并拟合贝塔分布
epsilon = 1e-4
results = []

for cluster_id in set(df['cluster']):
    if cluster_id == -1:
        continue  # 忽略噪声簇

    cluster_df = df[df['cluster'] == cluster_id]
    stay_times = []

    # 分车辆统计停留时间
    for plate, group in cluster_df.groupby('plate_num'):
        group_sorted = group.sort_values('timestamp')
        time_diffs = group_sorted['timestamp'].diff().dt.total_seconds().fillna(0)

        threshold = 300  # 5分钟以内连续轨迹判为停留
        stay_time = time_diffs[time_diffs <= threshold].sum()

        if stay_time > 0:
            stay_times.append(stay_time)

    if len(stay_times) < 5:
        print(f"Cluster {cluster_id} skipped: insufficient stay time samples.")
        continue

    # 归一化 + 微扰避免边界问题
    stay_times = np.array(stay_times)
    norm_stay_times = (stay_times - stay_times.min()) / (stay_times.max() - stay_times.min() + 1e-6)
    norm_stay_times = norm_stay_times * (1 - 2 * epsilon) + epsilon
    norm_stay_times += np.random.uniform(-1e-6, 1e-6, size=norm_stay_times.shape)
    norm_stay_times = np.clip(norm_stay_times, epsilon, 1 - epsilon)

    # 样本质量检查
    if len(norm_stay_times) < 5 or np.std(norm_stay_times) < 1e-5:
        print(f"Cluster {cluster_id} skipped: degenerate or low variance data.")
        continue

    # 拟合贝塔分布
    try:
        a, b, loc, scale = beta.fit(norm_stay_times, floc=0, fscale=1)
        results.append({'cluster_id': cluster_id, 'alpha': a, 'beta': b, 'n_points': len(stay_times)})

        # 可选绘图，控制绘图簇数
        if len(results) <= 3:
            x = np.linspace(0, 1, 100)
            y = beta.pdf(x, a, b, loc, scale)

            plt.figure(figsize=(6, 4))
            plt.hist(norm_stay_times, bins=15, density=True, alpha=0.6, label='Normalized Stay Time')
            plt.plot(x, y, 'r-', lw=2, label=f'Beta PDF (α={a:.2f}, β={b:.2f})')
            plt.title(f'Cluster {cluster_id} - Beta Fit of Stay Times')
            plt.xlabel('Normalized Stay Time')
            plt.ylabel('Density')
            plt.legend()
            plt.show()

    except Exception as e:
        print(f"Cluster {cluster_id} Beta fit failed: {e}")
        continue

# 5. 汇总结果
result_df = pd.DataFrame(results)


# 可选保存结果
result_df.to_csv('cluster_beta_params.csv', index=False)
