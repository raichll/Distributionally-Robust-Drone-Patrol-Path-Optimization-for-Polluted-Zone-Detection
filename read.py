import os
import pyarrow.parquet as pq
import pandas as pd
import folium
from datetime import datetime
from geopy.distance import geodesic
import ast
# 定义文件夹路径
folder_path = r"E:\dwd_truck"
# 定义临时文件路径
temp_file_path = r"E:\dwd_truck\final_df.parquet"

# 如果临时文件已存在，则直接读取
if os.path.exists(temp_file_path):
    print("临时文件已存在，直接读取数据...")
    final_df = pd.read_parquet(temp_file_path)
else:
    # 用于存储所有数据的列表
    all_data = []

    # 遍历目录中的所有子文件夹
    for root, dirs, files in os.walk(folder_path):
        # 只处理包含 c000 文件的子文件夹
        c000_files = [file for file in files if file.endswith("c000")]
           
        if c000_files:
            print(f"正在读取文件夹: {root}")  # 打印当前文件夹路径
            
            for file_name in c000_files:
                file_path = os.path.join(root, file_name)
                print(f"  正在读取文件: {file_path}")  # 打印当前读取的文件路径
                
                # 读取 Parquet 文件并转换为 pandas DataFrame
                table = pq.read_table(file_path)
                df = table.to_pandas()
                # 1. 数据预处理
                # 将'published_at'转换为datetime对象
                df['published_at'] = pd.to_datetime(df['published_at'])

                # **关键修复：将 'speed' 列转换为数值类型**
                df['speed'] = pd.to_numeric(df['speed'], errors='coerce')
                df['speed'].fillna(0, inplace=True) # 用0填充转换失败的值（例如，如果有非数字字符）
                import json
                # 从'raw_coord'中提取经纬度
                df['longitude'] = df['raw_coord'].apply(lambda x: json.loads(x)['raw_lon'])
                df['latitude'] = df['raw_coord'].apply(lambda x: json.loads(x)['raw_lat'])

                # 由于数据量大，需要对坐标进行一定程度的聚类或离散化，以便将点归类到“区域”
                # 这里我们使用简单的四舍五入作为区域划分，实际中可能需要更复杂的聚类算法（如DBSCAN）
                df['area_lat'] = df['latitude'].round(2)
                df['area_lon'] = df['longitude'].round(2)
                df['area'] = df['area_lat'].astype(str) + '_' + df['area_lon'].astype(str)
                # 将数据添加到 all_data 列表
                all_data.append(df)
            break
        

   
    # 合并所有数据
    if all_data:
        # 将 DataFrame 分批保存，避免内存溢出
        chunk_size = 5000  # 根据内存大小调整每个批次的大小
        for i in range(0, len(all_data), chunk_size):
            chunk_df = pd.concat(all_data[i:i + chunk_size], ignore_index=True)
            # 将合并后的 DataFrame 保存为临时文件，模式为 append
            if i == 0:
                chunk_df.to_parquet(temp_file_path, index=False)
            else:
                chunk_df.to_parquet(temp_file_path, index=False, append=True)
        print(f"数据已保存到临时文件: {temp_file_path}")
    else:
        print("没有数据被加载，请检查文件夹路径和内容")


'''df= final_df.copy()
# 将 'published_at' 转换为日期格式
df['published_at'] = pd.to_datetime(df['published_at'])

# 计算车辆停留时长：找出同一车辆在相同坐标的多条记录，停留时间计算
# 假设坐标相同表示停留在同一位置

# Step 1: 计算车辆停留时间
df['next_published_at'] = df.groupby('plate_num')['published_at'].shift(-1)
df['stay_duration'] = (df['next_published_at'] - df['published_at']).dt.total_seconds() / 60  # 以分钟为单位

# Step 2: 过滤满足条件的数据
# 过滤速度小于 15 km/h 的记录
df_filtered = df[(df['speed'] == 0) & (df['stay_duration'] >= 5) & (df['stay_duration'] <= 120)]

# 统计独特车辆数量 > 20
vehicle_count = df_filtered['plate_num'].nunique()
if vehicle_count > 20:
    print("High traffic area (potential dumping zone)")

# Step 3: 可视化潜在倾倒区

# 创建一个地图（使用 Folium）
map_center = [30.527741, 104.196103]  # 设置地图的中心点
m = folium.Map(location=map_center, zoom_start=12)

# 在地图上标记所有停留记录
for _, row in df_filtered.iterrows():
    folium.Marker(
        location=[row['coord']['lat'], row['coord']['lon']],
        popup=f"Plate: {row['plate_num']}\nTime: {row['published_at']}\nStay Duration: {row['stay_duration']} min",
        icon=folium.Icon(color='blue')
    ).add_to(m)

# 显示地图
m.save("potential_dumping_zones_en.html")'''


df= final_df.copy()


# 2. 特征计算
area_stats = df.groupby('area').agg(
    avg_speed=('speed', 'mean'),
    unique_vehicles=('plate_num', lambda x: x.nunique()),
    min_time=('published_at', 'min'),
    max_time=('published_at', 'max'),
    area_center_lat=('latitude', 'mean'),
    area_center_lon=('longitude', 'mean')
).reset_index()

# 计算活动时间跨度
area_stats['activity_duration'] = (area_stats['max_time'] - area_stats['min_time']).dt.total_seconds() / 3600 # 小时

# 计算个体车辆停留时间 (这部分需要更精细的逻辑，因为需要跟踪每辆车在每个区域的进出时间)
# 这里我们简化处理，假设在一个区域内，同一辆车如果有多条记录，其时间差可以近似为停留时间的一部分
# 对于准确的停留时间，需要按车辆和区域进行分组，计算时间差。
# 由于数据量大，直接计算所有个体车辆在所有区域的停留时间会非常耗时。
# 我们可以先筛选出低速点，然后在这些区域内计算车辆停留时间。
low_speed_points = df[df['speed'] < 15]

# 计算每辆车在每个区域的停留时间
# 这一步是计算复杂度的关键，对于1000万条数据，需要优化
# 思路：按 plate_num 和 area 分组，然后对 published_at 排序，计算相邻时间戳的差值
# 然后对这些时间差进行聚合（求和或平均）作为停留时间
# 假设我们只关心在低速区域的停留时间
vehicle_stay_times = low_speed_points.groupby(['plate_num', 'area']).agg(
    min_entry_time=('published_at', 'min'),
    max_exit_time=('published_at', 'max')
).reset_index()

vehicle_stay_times['stay_duration'] = (vehicle_stay_times['max_exit_time'] - vehicle_stay_times['min_entry_time']).dt.total_seconds() / 60 # 分钟

# 计算每个区域的平均个体车辆停留时间
area_avg_stay_time = vehicle_stay_times.groupby('area')['stay_duration'].mean().reset_index()
area_stats = pd.merge(area_stats, area_avg_stay_time, on='area', how='left')
area_stats['stay_duration'].fillna(0, inplace=True) # 如果某个区域没有低速点，停留时间为0

# 3. 应用阈值判断倾倒区
# 阈值示例：
AVG_SPEED_THRESHOLD = 15  # km/h
MIN_STAY_TIME_MIN = 5     # 分钟
MAX_STAY_TIME_MIN = 120   # 分钟
UNIQUE_VEHICLES_THRESHOLD = 20 # 假设我们没有足够的数据来模拟20个独特车辆，这里设小一点以便演示
# 为了演示目的，将UNIQUE_VEHICLES_THRESHOLD 设为2，因为模拟数据中车辆种类较少
# 在实际数据中，请使用您定义的 20
UNIQUE_VEHICLES_THRESHOLD_DEMO = 30

# 由于模拟数据量小，可能无法满足所有阈值，这里我们为了演示，调整一些阈值
# 真实的阈值应按照您的要求来
dumping_areas = area_stats[
    (area_stats['avg_speed'] < AVG_SPEED_THRESHOLD) &
    (area_stats['stay_duration'] >= MIN_STAY_TIME_MIN) &
    (area_stats['stay_duration'] <= MAX_STAY_TIME_MIN) &
    (area_stats['unique_vehicles'] >= UNIQUE_VEHICLES_THRESHOLD_DEMO) # 演示阈值
]

print("识别出的倾倒区：")
print(dumping_areas[['area', 'avg_speed', 'stay_duration', 'unique_vehicles', 'activity_duration']])

# 4. 在地图上描绘
# 创建一个地图对象，中心点设置为所有点的平均经纬度
map_center_lat = df['latitude'].mean()
map_center_lon = df['longitude'].mean()
m = folium.Map(location=[map_center_lat, map_center_lon], zoom_start=12)

# 将识别出的倾倒区在地图上标记出来
for index, row in dumping_areas.iterrows():
    folium.Marker(
        location=[row['area_center_lat'], row['area_center_lon']],
        popup=f"倾倒区: {row['area']}<br>"
              f"平均速度: {row['avg_speed']:.2f} km/h<br>"
              f"平均停留时长: {row['stay_duration']:.2f} 分钟<br>"
              f"独特车辆数: {int(row['unique_vehicles'])}<br>"
              f"活动持续时间: {row['activity_duration']:.2f} 小时",
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)

# 保存地图为HTML文件
m.save('dumping_areas_map.html')
print("\n地图已保存至 dumping_areas_map.html 文件，请用浏览器打开查看。")