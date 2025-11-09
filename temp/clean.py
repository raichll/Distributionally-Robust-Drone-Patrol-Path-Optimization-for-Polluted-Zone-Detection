# 引入库
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
import json
from datetime import datetime
from matplotlib.gridspec import GridSpec

# 配置路径
base_folder = r"E:\dwd_truck"
date_range = pd.date_range(start="2024-04-10", end="2024-04-18")
date_folders = [f"publish_date={date.strftime('%Y-%m-%d')}" for date in date_range]

# 读取单日所有 .c000 文件
def load_day_data(folder_path):
    file_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.c000')]
    df_list = []
    for file in file_list:
        try:
            df = pd.read_parquet(file)
            df_list.append(df)
        except Exception:
            continue
    if df_list:
        return pd.concat(df_list, ignore_index=True)
    return pd.DataFrame()

# 数据预处理
def preprocess(df):
    df = df.dropna(subset=['coord', 'plate_num', 'published_at'])
    df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
    df = df.dropna(subset=['published_at'])
    df['lon'] = df['coord'].apply(lambda x: json.loads(x)['lon'] if isinstance(x, str) else None)
    df['lat'] = df['coord'].apply(lambda x: json.loads(x)['lat'] if isinstance(x, str) else None)
    df['speed'] = pd.to_numeric(df['speed'], errors='coerce')
    df = df.dropna(subset=['speed', 'lon', 'lat'])
    df = df[(df['lon'] >= 102) & (df['lon'] <= 105) & (df['lat'] >= 29) & (df['lat'] <= 32)]
    df = df[(df['speed'] >= 0) & (df['speed'] <= 120)]
    return df.drop_duplicates()

# 每日数据读取和处理
day_data = {}
for folder in date_folders:
    full_path = os.path.join(base_folder, folder)
    if os.path.exists(full_path):
        df = load_day_data(full_path)
        df = df.sample(frac=0.001, random_state=42)  # 抽样0.1%
        df = preprocess(df)
        df['date'] = folder.split('=')[1]
        day_data[folder] = df
#print(day_data)
# 统计信息计算
basic_stats, upload_freq_stats, interval_stats = [], [], []

for key, df in day_data.items():
    day = df['date'].iloc[0]
    basic_stats.append({
        'Date': day,
        'Total Records': len(df),
        'Unique Vehicles': df['plate_num'].nunique(),
        'Unique SIMs': df['sim_card'].nunique(),
        'Time Start': df['published_at'].min(),
        'Time End': df['published_at'].max()
    })

    upload_counts = df.groupby('plate_num')['published_at'].count()
    upload_freq_stats.append({
        'Date': day,
        'Min Uploads': upload_counts.min(),
        'Max Uploads': upload_counts.max(),
        'Mean Uploads': upload_counts.mean(),
        'Median Uploads': upload_counts.median()
    })

    df = df.sort_values(['plate_num', 'published_at'])
    df['time_diff'] = df.groupby('plate_num')['published_at'].diff().dt.total_seconds()
    time_diff = df['time_diff'].dropna()
    day_data[key] = df  # 保证后续画图能访问 time_diff
    interval_stats.append({
        'Date': day,
        'Min Interval': time_diff.min(),
        'Max Interval': time_diff.max(),
        'Mean Interval': time_diff.mean(),
        'Median Interval': time_diff.median()
    })

# 输出表格
output_dir = os.path.join(base_folder, "analysis_results")
os.makedirs(output_dir, exist_ok=True)

pd.DataFrame(basic_stats).to_csv(os.path.join(output_dir, "basic_statistics.csv"), index=False)
pd.DataFrame(upload_freq_stats).to_csv(os.path.join(output_dir, "upload_frequency_statistics.csv"), index=False)
pd.DataFrame(interval_stats).to_csv(os.path.join(output_dir, "time_interval_statistics.csv"), index=False)

# 绘图函数（9个子图）
def plot_distribution(day_data, column, title, filename, bins=30, clip_upper=None):
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig)
    for i, (key, df) in enumerate(day_data.items()):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        data = df[column].dropna()
        if clip_upper:
            data = data.clip(upper=clip_upper)
        sns.histplot(data, bins=bins, kde=True, ax=ax)
        ax.set_title(df['date'].iloc[0])
        ax.set_xlabel(column)
        ax.set_ylabel("Count")
    fig.suptitle(title, fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(output_dir, filename))
    plt.close()

plot_distribution(day_data, 'speed', "Speed Distribution (Daily)", "speed_distribution_all.png")
plot_distribution(day_data, 'time_diff', "Upload Time Interval Distribution (Daily)", "time_interval_distribution_all.png")

# 地图和热力图输出
for key, df in day_data.items():
    date_str = df['date'].iloc[0]
    center = [df['lat'].mean(), df['lon'].mean()]

    # 点图
    m = folium.Map(location=center, zoom_start=12)
    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=2,
            color='blue',
            fill=True,
            fill_opacity=0.5
        ).add_to(m)
    m.save(os.path.join(output_dir, f"map_{date_str}.html"))

    # 热力图
    heat_data = df[['lat', 'lon']].dropna().values.tolist()
    hmap = folium.Map(location=center, zoom_start=12)
    HeatMap(heat_data, radius=8, blur=15).add_to(hmap)
    hmap.save(os.path.join(output_dir, f"heatmap_{date_str}.html"))
