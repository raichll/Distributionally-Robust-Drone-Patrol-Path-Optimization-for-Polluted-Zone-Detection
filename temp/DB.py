
import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.cluster import DBSCAN
import pyarrow.parquet as pq # 确保已安装：pip install pyarrow

# ========== 参数配置 ==========
base_folder = r"E:\dwd_truck" # 请将此路径更改为您的实际数据文件夹路径
# 缓存完整的DataFrame，避免每次运行时都重新处理原始数据
full_data_cache_file = "full_trajectory_dataframe.pkl"
n_jobs = 4 # 用于并行处理的CPU核心数量
c=0.05 # 抽样比例，表示从每天的数据中抽取1/20的运渣车数据
# ========== 数据处理函数 ==========
def load_day_data_fast(folder_path):
    """
    快速加载指定文件夹中所有 .c000 文件的Parquet数据。
    """
    file_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.c000')]
    tables = []
    for file in file_list:
        try:
            # 只读取需要的列以节省内存和时间
            table = pq.read_table(file, columns=['plate_num', 'published_at', 'speed', 'coord'])
            tables.append(table)
        except Exception:
            # 可以在此处添加日志记录，以了解哪些文件加载失败
            continue
    if tables:
        return pd.concat([t.to_pandas() for t in tables], ignore_index=True)
    return pd.DataFrame()

def preprocess_fast(df):
    """
    对加载的DataFrame进行预处理，包括缺失值处理、坐标解析、数据类型转换和范围过滤。
    """
    df = df.dropna(subset=['coord', 'plate_num', 'published_at'])
    df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
    df = df.dropna(subset=['published_at']) # 再次删除转换失败的行

    # 解析 coord 字段中的经纬度
    coords = df['coord'].apply(json.loads)
    df['lon'] = coords.map(lambda x: x.get('lon'))
    df['lat'] = coords.map(lambda x: x.get('lat'))

    # 转换速度为数值类型
    df['speed'] = pd.to_numeric(df['speed'], errors='coerce')
    df = df.dropna(subset=['speed', 'lon', 'lat']) # 删除速度或经纬度缺失的行

    # 根据地理范围和速度范围过滤数据
    # 请根据您的实际数据范围调整这些值
    df = df[(df['lon'] >= 102) & (df['lon'] <= 105) & (df['lat'] >= 29) & (df['lat'] <= 32)]
    df = df[(df['speed'] >= 0) & (df['speed'] <= 120)]
    return df.drop_duplicates() # 删除重复的轨迹点

'''def load_and_sample(folder):
    """
    加载指定日期文件夹的数据并进行小比例采样。
    """
    full_path = os.path.join(base_folder, folder)
    if os.path.exists(full_path):
        df = load_day_data_fast(full_path)
        if not df.empty:
            # 从每天的数据中采样一小部分，以减少总数据量
            # 如果您的数据量适中，可以移除此行或调整采样比例 (frac)
            df = df.sample(frac=1, random_state=42) 
            df['date'] = folder.split('=')[1]
            return df
    return None'''
#加载指定日期文件夹的数据，并抽取1/4数量的运渣车数据。
def load_and_sample(folder):
    """
    加载指定日期文件夹的数据，并抽取1/4数量的运渣车数据。
    """
    full_path = os.path.join(base_folder, folder)
    if os.path.exists(full_path):
        df = load_day_data_fast(full_path)
        if not df.empty:
            # 获取当天所有的唯一车牌号
            unique_plates = df['plate_num'].unique()
            
            # 计算要抽取的车牌号数量（至少一个）
            num_plates_to_sample = max(1, int(len(unique_plates) * c)) 
            
            # 随机抽取1/4的唯一车牌号
            sampled_plates = np.random.choice(unique_plates, size=num_plates_to_sample, replace=False)
            
            # 过滤DataFrame，只保留这些被抽取的车牌号的轨迹数据
            df = df[df['plate_num'].isin(sampled_plates)]
            
            df['date'] = folder.split('=')[1]
            return df
    return None


# ========== 数据加载模块 ==========
if os.path.exists(full_data_cache_file):
    print(f"✅ 已找到缓存数据 {full_data_cache_file}，正在加载完整DataFrame...")
    full_data = pd.read_pickle(full_data_cache_file)
    trajectory_data = full_data[['lon', 'lat']].to_numpy()
else:
    print("🚀 第一次运行，开始处理原始数据并缓存完整DataFrame...")
    # 定义要加载数据的日期范围
    date_range = pd.date_range(start="2024-04-10", end="2024-04-10")
    date_folders = [f"publish_date={date.strftime('%Y-%m-%d')}" for date in date_range]
    
    # 并行加载和初步采样数据
    sampled_dfs = Parallel(n_jobs=n_jobs)(delayed(load_and_sample)(folder) for folder in date_folders)
    
    # 合并所有采样后的DataFrame并执行最终预处理
    full_data = pd.concat([df for df in sampled_dfs if df is not None], ignore_index=True)
    full_data = preprocess_fast(full_data)
    
    # 将预处理后的DataFrame保存为pickle文件，以便下次更快加载
    full_data.to_pickle(full_data_cache_file)
    
    # 提取经度和纬度用于聚类
    trajectory_data = full_data[['lon', 'lat']].to_numpy()
    print(f"✅ 已保存完整 DataFrame 到 {full_data_cache_file}")

print(f"📌 轨迹数据加载完成，共 {trajectory_data.shape[0]} 个点")

# ========== DBSCAN 聚类 ==========
print("\n⚙️ 开始执行 DBSCAN 聚类...")
start_time = time.time()
# DBSCAN 参数说明：
# eps (epsilon): 同一邻域内的最大距离。DBSCAN会在此距离内寻找邻居。
#                单位与您的数据（经纬度，通常是度）相同。
# min_samples: 形成核心点所需的最小样本数（包括点本身）。
#              值越大，对密度要求越高，聚类越稀疏，噪声点越多。
# n_jobs: 用于并行处理的CPU核心数量。-1 表示使用所有可用核心。
db = DBSCAN(eps=0.003, min_samples=30, n_jobs=n_jobs).fit(trajectory_data)
dbscan_labels = db.labels_ # 获取聚类标签，-1 表示噪声点
dbscan_time = time.time() - start_time
print(f"✅ [DBSCAN] 聚类耗时: {dbscan_time:.2f} 秒")

# ====== 筛选逻辑：剔除休息区和工地，保留倾倒区 ======
print("\n⚙️ 开始筛选聚类结果，识别潜在倾倒区...")

# 将聚类标签链接回原始 full_data DataFrame
full_data_with_labels = full_data.copy()
full_data_with_labels['cluster_label'] = dbscan_labels

# 过滤掉噪声点以便进行聚类特征分析 (噪声点标签为-1)
clustered_points_df = full_data_with_labels[full_data_with_labels['cluster_label'] != -1]

# 提取每个聚类的特征
cluster_features = {}
for cluster_id in clustered_points_df['cluster_label'].unique():
    cluster_df = clustered_points_df[clustered_points_df['cluster_label'] == cluster_id]

    avg_speed = cluster_df['speed'].mean()
    unique_plate_nums = cluster_df['plate_num'].nunique()
    
    # 聚类的时间跨度（从该聚类中最早的点到最晚的点）
    time_span_cluster = (cluster_df['published_at'].max() - cluster_df['published_at'].min()).total_seconds() / 3600 # 小时

    # 粗略估计个体（每辆车）在聚类区域内的平均停留时长
    individual_stay_durations = []
    for plate in cluster_df['plate_num'].unique():
        truck_points = cluster_df[cluster_df['plate_num'] == plate].sort_values(by='published_at')
        
        # 简化停留时长计算：同一辆车在集群内的最早和最晚时间点之间的差异
        # 这种方式会高估实际停留时间，但可以区分短暂停留和长时间停留
        if len(truck_points) > 1:
            duration = (truck_points['published_at'].max() - truck_points['published_at'].min()).total_seconds() / 60 # 分钟
            individual_stay_durations.append(duration)
    
    avg_individual_stay_duration_minutes = np.mean(individual_stay_durations) if individual_stay_durations else 0

    cluster_features[cluster_id] = {
        'avg_speed': avg_speed,
        'unique_plate_nums': unique_plate_nums,
        'time_span_cluster_hours': time_span_cluster,
        'avg_individual_stay_duration_minutes': avg_individual_stay_duration_minutes,
        'num_points_in_cluster': len(cluster_df)
    }

print("\n--- 聚类特征概览 (用于筛选) ---")
for cluster_id, features in cluster_features.items():
    print(f"聚类 {cluster_id}: {features}")

# 定义筛选规则
# 这些阈值是示例，需要根据您的数据和业务理解进行调整
filtered_dumping_zones = [] # 潜在倾倒区
filtered_rest_areas = []    # 潜在休息区
filtered_other_clusters = [] # 其他聚类 (可能是一般停留点、普通工地等)

for cluster_id, features in cluster_features.items():
    avg_speed = features['avg_speed']
    unique_plate_nums = features['unique_plate_nums']
    time_span_cluster_hours = features['time_span_cluster_hours']
    avg_individual_stay_duration_minutes = features['avg_individual_stay_duration_minutes']
    num_points_in_cluster = features['num_points_in_cluster']

    # 规则1: 识别潜在休息区
    # 特征：平均速度低，个体车辆长时间停留（表示休息），车辆多样性可能较低或集中在少数车辆
    # 阈值示例：
    # - 平均速度 < 5 km/h (表示停车或极慢移动)
    # - 个体平均停留时长 > 240 分钟 (4小时，可能表示长时间休息甚至过夜)
    if avg_speed < 5 and avg_individual_stay_duration_minutes > 240:
        filtered_rest_areas.append(cluster_id)
    # 规则2: 识别潜在倾倒区/大型工地
    # 特征：平均速度低，个体车辆停留时间适中（符合装卸时间），独特车辆数量多（表示高流量），活动时间跨度长（持续性操作）
    # 阈值示例：
    # - 平均速度 < 15 km/h (表示缓慢移动或停止装卸)
    # - 个体平均停留时长在 5 到 120 分钟之间 (5分钟到2小时，典型的装卸作业时间)
    # - 独特车辆数量 > 20 (表示来自不同来源的车辆高频进出)
    # - 聚类时间跨度 > 24 小时 (表示这不是一个短期事件，而是持续的作业)
    # 注意：此规则很难完全区分倾倒区和大型建筑工地。倾倒区可能在更偏远或非官方的地点。
    elif (avg_speed < 30 and
          avg_individual_stay_duration_minutes >= 0 and
          avg_individual_stay_duration_minutes <= 120 and
          unique_plate_nums > 5 and
          '''time_span_cluster_hours > 24'''):
        filtered_dumping_zones.append(cluster_id)
    else:
        filtered_other_clusters.append(cluster_id)

print(f"\n--- 筛选结果 ---")
print(f"潜在倾倒区 (聚类ID): {filtered_dumping_zones}")
print(f"潜在休息区 (聚类ID): {filtered_rest_areas}")
print(f"其他聚类 (聚类ID): {filtered_other_clusters}")

# ====== 可视化 (更新以显示筛选结果，不绘制噪声点) ======
print("\n📊 生成最终筛选结果可视化图表 (不含噪声点)...")
final_labels_for_plot = np.copy(dbscan_labels)
# 为不同类型的集群分配独特的负标签，以便可视化时区分
rest_area_label = -2
other_cluster_label = -3

# 将筛选后的聚类ID映射到新的标签
for cluster_id in filtered_rest_areas:
    final_labels_for_plot[final_labels_for_plot == cluster_id] = rest_area_label
for cluster_id in filtered_other_clusters:
    final_labels_for_plot[final_labels_for_plot == cluster_id] = other_cluster_label

plt.figure(figsize=(12, 10))

# 创建一个布尔掩码，用于排除噪声点 (-1)
non_noise_mask = (final_labels_for_plot != -1)

# 仅处理非噪声点
filtered_trajectory_data = trajectory_data[non_noise_mask]
filtered_labels = final_labels_for_plot[non_noise_mask]

# 获取需要着色的聚类ID（不包括特殊负标签）
unique_clustered_ids = np.unique(filtered_labels[filtered_labels >= 0])

if len(unique_clustered_ids) > 0:
    # 映射原始聚类ID到连续的颜色索引
    cluster_id_to_color_idx = {cid: i for i, cid in enumerate(unique_clustered_ids)}
    colors_for_clusters = [plt.cm.viridis(cluster_id_to_color_idx.get(label, 0) / len(unique_clustered_ids)) 
                           for label in filtered_labels if label >= 0] 

    # 绘制普通的聚类点（非噪声，非特定筛选区域）
    general_clustered_mask = (filtered_labels >= 0)
    plt.scatter(filtered_trajectory_data[general_clustered_mask, 0], 
                filtered_trajectory_data[general_clustered_mask, 1], 
                c=colors_for_clusters, 
                s=5, alpha=0.7, label='General Clustered Points', zorder=0) # 较低的zorder确保在其他高亮层之下

# 独立绘制特殊类别（休息区、其他聚类、潜在倾倒区）
# 这些点的绘制将覆盖或叠加在通用聚类点之上，以高亮显示

# 绘制潜在倾倒区 (高亮显示)
dumping_zone_mask = np.isin(filtered_labels, filtered_dumping_zones)
plt.scatter(filtered_trajectory_data[dumping_zone_mask, 0],
            filtered_trajectory_data[dumping_zone_mask, 1],
            color='lime', s=25, alpha=0.9, label='Potential Dumping Zone', edgecolor='black', linewidth=0.8, zorder=3)

# 绘制潜在休息区
rest_area_mask = (filtered_labels == rest_area_label)
plt.scatter(filtered_trajectory_data[rest_area_mask, 0],
            filtered_trajectory_data[rest_area_mask, 1],
            color='red', s=12, alpha=0.8, label='Potential Rest Area', edgecolor='black', linewidth=0.5, zorder=2)

# 绘制其他聚类
other_cluster_mask = (filtered_labels == other_cluster_label)
plt.scatter(filtered_trajectory_data[other_cluster_mask, 0],
            filtered_trajectory_data[other_cluster_mask, 1],
            color='orange', s=8, alpha=0.6, label='Other Clusters', edgecolor='black', linewidth=0.2, zorder=1)

# 由于不绘制噪声点，所以这里不再包含噪声点的 plt.scatter 调用

plt.title(f"DBSCAN Clustering with Filtered Zones (Time: {dbscan_time:.2f}s) - Noise Excluded")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

# 创建自定义图例
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lime', markersize=10, label='Potential Dumping Zone'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=7, label='Potential Rest Area'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=7, label='Other Clusters'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=5, label='General Clustered Points')
]
plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0.)

plt.tight_layout(rect=[0, 0, 0.85, 1]) # 调整布局，为图例留出空间
plt.savefig("dbscan_filtered_zones_no_noise.png", dpi=300) # 保存图表为PNG文件
plt.show()

print("\n--- 脚本执行完毕 ---")
print(f"DBSCAN 聚类并筛选结果（不含噪声点）已保存到 dbscan_filtered_zones_no_noise.png")
