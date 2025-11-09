import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.cluster import DBSCAN
import pyarrow.parquet as pq

# ========== 参数配置 ==========
base_folder = r"E:\dwd_truck"  # 请将此路径更改为您的实际数据文件夹路径
# 缓存完整的DataFrame，避免每次运行时都重新处理原始数据
# full_data_cache_file = "full_trajectory_dataframe.pkl" # This cache will be removed as we process day by day
n_jobs = 4  # 用于并行处理的CPU核心数量
c = 0.05  # 抽样比例，表示从每天的数据中抽取1/20的运渣车数据

# Output directories
output_plots_folder = "dbscan_plots"
output_csv_folder = "dbscan_results"
os.makedirs(output_plots_folder, exist_ok=True)
os.makedirs(output_csv_folder, exist_ok=True)

# CSV file to store dumping zone coordinates
dumping_zones_csv_file = os.path.join(output_csv_folder, "potential_dumping_zones_all_days.csv")


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
    df = df.dropna(subset=['published_at'])  # 再次删除转换失败的行

    # 解析 coord 字段中的经纬度
    coords = df['coord'].apply(json.loads)
    df['lon'] = coords.map(lambda x: x.get('lon'))
    df['lat'] = coords.map(lambda x: x.get('lat'))

    # 转换速度为数值类型
    df['speed'] = pd.to_numeric(df['speed'], errors='coerce')
    df = df.dropna(subset=['speed', 'lon', 'lat'])  # 删除速度或经纬度缺失的行

    # 根据地理范围和速度范围过滤数据
    # 请根据您的实际数据范围调整这些值
    df = df[(df['lon'] >= 102) & (df['lon'] <= 105) & (df['lat'] >= 29) & (df['lat'] <= 32)]
    df = df[(df['speed'] >= 0) & (df['speed'] <= 120)]
    return df.drop_duplicates()  # 删除重复的轨迹点

def load_and_sample(folder, c_sampling_ratio): # Added c_sampling_ratio as parameter
    """
    加载指定日期文件夹的数据，并抽取c_sampling_ratio比例的运渣车数据。
    """
    full_path = os.path.join(base_folder, folder)
    if os.path.exists(full_path):
        df = load_day_data_fast(full_path)
        if not df.empty:
            # 获取当天所有的唯一车牌号
            unique_plates = df['plate_num'].unique()
            
            # 计算要抽取的车牌号数量（至少一个）
            num_plates_to_sample = max(1, int(len(unique_plates) * c_sampling_ratio))
            
            # 随机抽取指定比例的唯一车牌号
            sampled_plates = np.random.choice(unique_plates, size=num_plates_to_sample, replace=False)
            
            # 过滤DataFrame，只保留这些被抽取的车牌号的轨迹数据
            df = df[df['plate_num'].isin(sampled_plates)]
            
            df['date'] = folder.split('=')[1]
            return df
    return None

# Function to process data for a single day
def process_single_day(date_str, c_sampling_ratio, n_jobs_dbscan):
    print(f"\n--- Processing data for {date_str} ---")
    folder = f"publish_date={date_str}"
    
    # Load and preprocess data for the specific day
    daily_data = load_and_sample(folder, c_sampling_ratio) # pass c_sampling_ratio
    
    if daily_data is None or daily_data.empty:
        print(f"No data found or empty for {date_str}. Skipping.")
        return None, None # Return None for both dumping zones and plot
    
    daily_data_preprocessed = preprocess_fast(daily_data)
    if daily_data_preprocessed.empty:
        print(f"No preprocessed data available for {date_str}. Skipping.")
        return None, None
        
    trajectory_data = daily_data_preprocessed[['lon', 'lat']].to_numpy()
    print(f"📌 Trajectory data loaded for {date_str}, total {trajectory_data.shape[0]} points")

    # ========== DBSCAN 聚类 ==========
    print(f"⚙️ Starting DBSCAN clustering for {date_str}...")
    start_time = time.time()
    db = DBSCAN(eps=0.003, min_samples=30, n_jobs=n_jobs_dbscan).fit(trajectory_data)
    dbscan_labels = db.labels_  # 获取聚类标签，-1 表示噪声点
    dbscan_time = time.time() - start_time
    print(f"✅ [DBSCAN] Clustering for {date_str} took: {dbscan_time:.2f} seconds")

    # ====== 筛选逻辑：剔除休息区和工地，保留倾倒区 ======
    print(f"⚙️ Starting to filter clustering results for {date_str}, identifying potential dumping zones...")
    
    full_data_with_labels = daily_data_preprocessed.copy()
    full_data_with_labels['cluster_label'] = dbscan_labels

    clustered_points_df = full_data_with_labels[full_data_with_labels['cluster_label'] != -1]

    cluster_features = {}
    for cluster_id in clustered_points_df['cluster_label'].unique():
        cluster_df = clustered_points_df[clustered_points_df['cluster_label'] == cluster_id]
        avg_speed = cluster_df['speed'].mean()
        unique_plate_nums = cluster_df['plate_num'].nunique()
        time_span_cluster = (cluster_df['published_at'].max() - cluster_df['published_at'].min()).total_seconds() / 3600 # hours

        # 获取聚类质心 (平均经纬度)
        centroid_lon = cluster_df['lon'].mean()
        centroid_lat = cluster_df['lat'].mean()

        cluster_features[cluster_id] = {
            'avg_speed': avg_speed,
            'unique_plate_nums': unique_plate_nums,
            'time_span_cluster': time_span_cluster,
            'centroid_lon': centroid_lon,
            'centroid_lat': centroid_lat,
            'size': len(cluster_df)
        }

    # 根据特征筛选聚类，识别潜在倾倒区和休息区
    dumping_zones_labels = []
    rest_area_labels = []
    other_cluster_labels = []

    # 阈值（可以根据实际情况调整）
    DUMPING_SPEED_THRESHOLD = 5 # 倾倒区通常速度很低
    DUMPING_UNIQUE_PLATES_THRESHOLD = 5 # 倾倒区应有多辆车
    DUMPING_TIME_SPAN_THRESHOLD = 0.5 # 倾倒区可能持续较长时间 (0.5小时以上)
    REST_AREA_SPEED_THRESHOLD = 5 # 休息区通常速度也很低
    REST_AREA_TIME_SPAN_THRESHOLD = 2 # 休息区停留时间可能更长 (2小时以上)
    REST_AREA_UNIQUE_PLATES_THRESHOLD = 2 # 休息区也应有多辆车

    for cluster_id, features in cluster_features.items():
        if (features['avg_speed'] <= DUMPING_SPEED_THRESHOLD and
            features['unique_plate_nums'] >= DUMPING_UNIQUE_PLATES_THRESHOLD and
            features['time_span_cluster'] >= DUMPING_TIME_SPAN_THRESHOLD):
            dumping_zones_labels.append(cluster_id)
        elif (features['avg_speed'] <= REST_AREA_SPEED_THRESHOLD and
              features['unique_plate_nums'] >= REST_AREA_UNIQUE_PLATES_THRESHOLD and
              features['time_span_cluster'] >= REST_AREA_TIME_SPAN_THRESHOLD):
            rest_area_labels.append(cluster_id)
        else:
            other_cluster_labels.append(cluster_id)

    print(f"✅ Filtered results for {date_str}:")
    print(f"   Potential Dumping Zones identified: {len(dumping_zones_labels)} clusters")
    print(f"   Potential Rest Areas identified: {len(rest_area_labels)} clusters")
    print(f"   Other Clusters: {len(other_cluster_labels)} clusters")

    # Prepare dumping zone coordinates for CSV
    dumping_zone_coords = []
    for cluster_id in dumping_zones_labels:
        features = cluster_features[cluster_id]
        dumping_zone_coords.append({
            'date': date_str,
            'cluster_id': cluster_id,
            'centroid_lon': features['centroid_lon'],
            'centroid_lat': features['centroid_lat'],
            'avg_speed': features['avg_speed'],
            'unique_plate_nums': features['unique_plate_nums'],
            'time_span_hours': features['time_span_cluster'],
            'cluster_size': features['size']
        })
    daily_dumping_zones_df = pd.DataFrame(dumping_zone_coords)

    # ========== 可视化聚类结果 ==========
    plt.figure(figsize=(12, 10))

    # 绘制潜在倾倒区
    for dumping_zone_label in dumping_zones_labels:
        dumping_zone_mask = (full_data_with_labels['cluster_label'] == dumping_zone_label)
        plt.scatter(full_data_with_labels[dumping_zone_mask]['lon'],
                    full_data_with_labels[dumping_zone_mask]['lat'],
                    color='lime', s=20, alpha=0.9, label='Potential Dumping Zone' if dumping_zone_label == dumping_zones_labels[0] else "", # Only label once
                    edgecolor='black', linewidth=0.7, zorder=3)

    # 绘制潜在休息区
    for rest_area_label in rest_area_labels:
        rest_area_mask = (full_data_with_labels['cluster_label'] == rest_area_label)
        plt.scatter(full_data_with_labels[rest_area_mask]['lon'],
                    full_data_with_labels[rest_area_mask]['lat'],
                    color='red', s=12, alpha=0.8, label='Potential Rest Area' if rest_area_label == rest_area_labels[0] else "", # Only label once
                    edgecolor='black', linewidth=0.5, zorder=2)

    # 绘制其他聚类
    for other_cluster_label in other_cluster_labels:
        other_cluster_mask = (full_data_with_labels['cluster_label'] == other_cluster_label)
        plt.scatter(full_data_with_labels[other_cluster_mask]['lon'],
                    full_data_with_labels[other_cluster_mask]['lat'],
                    color='orange', s=8, alpha=0.6, label='Other Clusters' if other_cluster_label == other_cluster_labels[0] else "", # Only label once
                    edgecolor='black', linewidth=0.2, zorder=1)

    # 由于不绘制噪声点，所以这里不再包含噪声点的 plt.scatter 调用

    plt.title(f"DBSCAN Clustering with Filtered Zones ({date_str}) - Noise Excluded (Time: {dbscan_time:.2f}s)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    # 创建自定义图例
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lime', markersize=10, label='Potential Dumping Zone'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=7, label='Potential Rest Area'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=7, label='Other Clusters'),
    ]

    plt.legend(handles=legend_elements, loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Save the plot for the current day
    plot_filename = os.path.join(output_plots_folder, f"dbscan_clustering_{date_str}.png")
    plt.savefig(plot_filename)
    plt.close() # Close the plot to free memory
    print(f"✅ Plot saved to {plot_filename}")

    return daily_dumping_zones_df, plot_filename


# ========== 主执行流程 ==========
if __name__ == "__main__":
    # Define the 9-day date range
    start_date = "2024-04-10"
    end_date = "2024-04-18"
    date_range = pd.date_range(start=start_date, end=end_date)
    c=0.05
    all_dumping_zones = pd.DataFrame() 
    
    # Clear the dumping zones CSV file if it exists, to ensure a fresh start
    if os.path.exists(dumping_zones_csv_file):
        os.remove(dumping_zones_csv_file)
        print(f"Cleared existing {dumping_zones_csv_file}")

    for single_date in date_range:
        date_str = single_date.strftime('%Y-%m-%d')
        
        # Process data for the current day
        daily_dumping_zones, _ = process_single_day(date_str, c, n_jobs)
        
        if daily_dumping_zones is not None and not daily_dumping_zones.empty:
            # Append daily dumping zones to the overall DataFrame
            all_dumping_zones = pd.concat([all_dumping_zones, daily_dumping_zones], ignore_index=True)
            
            # Save daily dumping zones to the CSV file
            # If the file doesn't exist, write header. Otherwise, append without header.
            header_needed = not os.path.exists(dumping_zones_csv_file)
            daily_dumping_zones.to_csv(dumping_zones_csv_file, mode='a', index=False, header=header_needed)
            print(f"✅ Dumping zones for {date_str} appended to {dumping_zones_csv_file}")

    print(f"\n--- All 9 days processed ---")
    print(f"Final combined dumping zones saved to: {dumping_zones_csv_file}")
    print(f"All daily plots saved in the '{output_plots_folder}' directory.")