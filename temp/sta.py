import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import pyarrow.parquet as pq # 确保已安装：pip install pyarrow

# ========== 参数配置 ==========
base_folder = r"E:\dwd_truck" # 请将此路径更改为您的实际数据文件夹路径
# 缓存完整的DataFrame，避免每次运行时都重新处理原始数据
full_data_cache_file = "full_trajectory_dataframe.pkl"
n_jobs = 4 # 用于并行处理的CPU核心数量

# 新增参数：DBSCAN聚类允许的最大点数，根据您的内存调整
# 如果数据量仍然过大，可能需要进一步减小这个值
max_points_for_processing = 1000000 

# 新增参数：网格单元的尺寸 (以经纬度为单位)
# 类似于DBSCAN的eps，决定了区域的“粒度”
grid_cell_size_lon = 0.003 
grid_cell_size_lat = 0.003

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
            # print(f"Warning: Could not read {file}")
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

def load_and_sample(folder):
    """
    加载指定日期文件夹的数据，并抽取1/10数量的运渣车数据。
    """
    full_path = os.path.join(base_folder, folder)
    if os.path.exists(full_path):
        df = load_day_data_fast(full_path)
        if not df.empty:
            # 获取当天所有的唯一车牌号
            unique_plates = df['plate_num'].unique()
            
            # 计算要抽取的车牌号数量（至少一个）
            num_plates_to_sample = max(1, int(len(unique_plates) * 0.05)) 
            
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
else:
    print("🚀 第一次运行，开始处理原始数据并缓存完整DataFrame...")
    # 定义要加载数据的日期范围
    date_range = pd.date_range(start="2024-04-11", end="2024-04-11")
    date_folders = [f"publish_date={date.strftime('%Y-%m-%d')}" for date in date_range]
    
    # 并行加载和初步采样数据
    sampled_dfs = Parallel(n_jobs=n_jobs)(delayed(load_and_sample)(folder) for folder in date_folders)
    
    # 合并所有采样后的DataFrame并执行最终预处理
    full_data = pd.concat([df for df in sampled_dfs if df is not None], ignore_index=True)
    full_data = preprocess_fast(full_data)
    
    # 将预处理后的DataFrame保存为pickle文件，以便下次更快加载
    full_data.to_pickle(full_data_cache_file)
    
    print(f"✅ 已保存完整 DataFrame 到 {full_data_cache_file}")

# ====== 如果数据量过大，进行二次采样 ======
if len(full_data) > max_points_for_processing:
    print(f"⚠️ 数据点数 ({len(full_data)}) 超过最大限制 ({max_points_for_processing})，进行随机抽样...")
    full_data = full_data.sample(n=max_points_for_processing, random_state=42)
    print(f"✅ 抽样后数据点数: {len(full_data)}")
    
print(f"📌 轨迹数据加载完成，共 {len(full_data)} 个点")

# ========== 网格划分和特征统计 ==========
print("\n⚙️ 开始网格划分并统计区域特征...")
start_time_grid = time.time()

# 为每个点计算其所属网格单元的 ID
full_data['grid_lon_id'] = np.floor(full_data['lon'] / grid_cell_size_lon).astype(int)
full_data['grid_lat_id'] = np.floor(full_data['lat'] / grid_cell_size_lat).astype(int)

# 按网格单元分组并计算特征
# 这里我们将直接使用 pandas 的 groupby 和 agg 方法来高效计算
grid_features_df = full_data.groupby(['grid_lon_id', 'grid_lat_id']).agg(
    avg_speed=('speed', 'mean'),
    unique_plate_nums=('plate_num', 'nunique'),
    min_published_at=('published_at', 'min'),
    max_published_at=('published_at', 'max'),
    num_points_in_cell=('lon', 'count') # 可以用任意列进行计数
).reset_index()

# 计算网格单元的时间跨度
grid_features_df['time_span_cell_hours'] = (grid_features_df['max_published_at'] - 
                                            grid_features_df['min_published_at']).dt.total_seconds() / 3600

# 计算每个网格内平均单车停留时长 (此处的计算方式与聚类保持一致，但应用于网格单元内)
# 这部分需要额外的迭代或更复杂的 groupby 操作，为了简化，我们可以先计算每个车牌在每个网格中的停留时间，再取平均
# 这是一个更耗时的步骤，但对于识别倾倒区很重要
print("  - 计算网格内平均单车停留时长...")
def calculate_avg_individual_stay(group):
    individual_stay_durations = []
    for plate in group['plate_num'].unique():
        truck_points = group[group['plate_num'] == plate].sort_values(by='published_at')
        if len(truck_points) > 1:
            duration = (truck_points['published_at'].max() - truck_points['published_at'].min()).total_seconds() / 60 # 分钟
            individual_stay_durations.append(duration)
    return np.mean(individual_stay_durations) if individual_stay_durations else 0

# 为每个网格单元应用此函数
# 注意：这可能会非常慢，因为它在每个网格组上都进行迭代
# 如果性能有问题，可能需要更优化的方法，或者简化停留时间特征
avg_individual_stay_series = full_data.groupby(['grid_lon_id', 'grid_lat_id']).apply(calculate_avg_individual_stay)
avg_individual_stay_series.name = 'avg_individual_stay_duration_minutes'
grid_features_df = grid_features_df.set_index(['grid_lon_id', 'grid_lat_id']).join(avg_individual_stay_series).reset_index()

# 清理不再需要的中间时间列
grid_features_df = grid_features_df.drop(columns=['min_published_at', 'max_published_at'])

grid_processing_time = time.time() - start_time_grid
print(f"✅ 网格划分和特征统计耗时: {grid_processing_time:.2f} 秒")
print(f"📌 共识别出 {len(grid_features_df)} 个活跃网格单元")

# ========== 基于统计特征的区域分类 ==========
print("\n⚙️ 开始基于统计特征分类网格区域...")

classified_regions = [] # 存储分类结果
# 倾倒区: 1, 休息区: 2, 普通区: 3

for index, row in grid_features_df.iterrows():
    cluster_id = (row['grid_lon_id'], row['grid_lat_id']) # 使用网格ID作为“聚类ID”
    avg_speed = row['avg_speed']
    unique_plate_nums = row['unique_plate_nums']
    time_span_cell_hours = row['time_span_cell_hours']
    avg_individual_stay_duration_minutes = row['avg_individual_stay_duration_minutes']
    num_points_in_cell = row['num_points_in_cell']

    # 规则1: 识别潜在休息区
    if avg_speed < 5 and avg_individual_stay_duration_minutes > 240:
        classified_regions.append({'grid_id': cluster_id, 'type': '休息区', 'lon_center': (row['grid_lon_id'] + 0.5) * grid_cell_size_lon, 'lat_center': (row['grid_lat_id'] + 0.5) * grid_cell_size_lat, 'num_points': num_points_in_cell})
    # 规则2: 识别潜在倾倒区/大型工地
    elif (avg_speed < 30 and
          avg_individual_stay_duration_minutes >= 0 and
          avg_individual_stay_duration_minutes <= 60 and
          unique_plate_nums > 5 
          #and time_span_cell_hours > 24
          ):
        classified_regions.append({'grid_id': cluster_id, 'type': '倾倒区', 'lon_center': (row['grid_lon_id'] + 0.5) * grid_cell_size_lon, 'lat_center': (row['grid_lat_id'] + 0.5) * grid_cell_size_lat, 'num_points': num_points_in_cell})
    else:
        classified_regions.append({'grid_id': cluster_id, 'type': '普通区', 'lon_center': (row['grid_lon_id'] + 0.5) * grid_cell_size_lon, 'lat_center': (row['grid_lat_id'] + 0.5) * grid_cell_size_lat, 'num_points': num_points_in_cell})

classified_df = pd.DataFrame(classified_regions)

print(f"\n--- 分类结果概览 ---")
print(classified_df['type'].value_counts())

# 如果需要详细调试，可以打印出各类区域的特征：
# print("\n--- '倾倒区' 特征 ---")
# print(grid_features_df[grid_features_df.index.isin([r['grid_id'] for r in classified_regions if r['type'] == '倾倒区'])])
# print("\n--- '休息区' 特征 ---")
# print(grid_features_df[grid_features_df.index.isin([r['grid_id'] for r in classified_regions if r['type'] == '休息区'])])
# print("\n--- '普通区' 特征 ---")
# print(grid_features_df[grid_features_df.index.isin([r['grid_id'] for r in classified_regions if r['type'] == '普通区'])])


# ========== 可视化分类结果 ==========
print("\n📊 生成分类区域可视化图表...")
plt.figure(figsize=(12, 10))

# 绘制不同类型的区域
if '倾倒区' in classified_df['type'].unique():
    dumping_zones = classified_df[classified_df['type'] == '倾倒区']
    plt.scatter(dumping_zones['lon_center'], dumping_zones['lat_center'],
                color='lime', s=dumping_zones['num_points'].apply(lambda x: min(max(x/100, 10), 1000)), # 标记大小与点数成正比，并设置 min/max
                alpha=0.9, label='Potential Dumping Zone', edgecolor='black', linewidth=0.8, zorder=3)

if '休息区' in classified_df['type'].unique():
    rest_areas = classified_df[classified_df['type'] == '休息区']
    plt.scatter(rest_areas['lon_center'], rest_areas['lat_center'],
                color='red', s=rest_areas['num_points'].apply(lambda x: min(max(x/100, 5), 500)), 
                alpha=0.8, label='Potential Rest Area', edgecolor='black', linewidth=0.5, zorder=2)

'''if '普通区' in classified_df['type'].unique():
    normal_areas = classified_df[classified_df['type'] == '普通区']
    plt.scatter(normal_areas['lon_center'], normal_areas['lat_center'],
                color='orange', s=normal_areas['num_points'].apply(lambda x: min(max(x/100, 2), 200)), 
                alpha=0.6, label='Normal Area', edgecolor='black', linewidth=0.2, zorder=1)
'''
plt.title(f"Area Classification by Statistical Features (Grid Size: {grid_cell_size_lon}x{grid_cell_size_lat})")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

# 创建自定义图例
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lime', markersize=10, label='Potential Dumping Zone'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=7, label='Potential Rest Area'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=5, label='Normal Area')
]
plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0.)

plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig("area_classification_grid.png", dpi=300)
plt.show()

print("\n--- 脚本执行完毕 ---")
print(f"区域分类结果已保存到 area_classification_grid.png")