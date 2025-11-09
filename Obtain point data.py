import pandas as pd
import os
import numpy as np
from geopy.distance import geodesic

def calculate_distance(lat1, lon1, lat2, lon2):
    """
    计算两个地理坐标点之间的距离（米）
    使用geodesic计算更准确的地球表面距离
    """
    return geodesic((lat1, lon1), (lat2, lon2)).meters

def merge_nearby_clusters(centroids_df, distance_threshold=500):
    """
    合并距离在阈值内的聚类簇
    
    Args:
        centroids_df: 包含聚类中心点信息的DataFrame
        distance_threshold: 距离阈值（米），默认500米
    
    Returns:
        合并后的聚类簇DataFrame
    """
    # 创建副本避免修改原数据
    df = centroids_df.copy()
    df['merged'] = False  # 标记是否已被合并
    
    merged_clusters = []
    
    for i in range(len(df)):
        if df.iloc[i]['merged']:
            continue
            
        # 当前聚类簇信息
        current_cluster = df.iloc[i].copy()
        current_lat = current_cluster['latitude']
        current_lon = current_cluster['longitude']
        current_count = current_cluster['count']
        
        # 寻找需要合并的聚类簇
        clusters_to_merge = [i]
        total_count = current_count
        weighted_lat = current_lat * current_count
        weighted_lon = current_lon * current_count
        
        for j in range(i + 1, len(df)):
            if df.iloc[j]['merged']:
                continue
                
            target_lat = df.iloc[j]['latitude']
            target_lon = df.iloc[j]['longitude']
            
            # 计算距离
            distance = calculate_distance(current_lat, current_lon, target_lat, target_lon)
            
            if distance <= distance_threshold:
                clusters_to_merge.append(j)
                target_count = df.iloc[j]['count']
                total_count += target_count
                weighted_lat += target_lat * target_count
                weighted_lon += target_lon * target_count
                df.iloc[j, df.columns.get_loc('merged')] = True
        
        # 计算加权平均中心点
        merged_lat = weighted_lat / total_count
        merged_lon = weighted_lon / total_count
        
        # 标记当前聚类簇为已合并
        df.iloc[i, df.columns.get_loc('merged')] = True
        
        # 添加合并后的聚类簇信息
        merged_clusters.append({
            'cluster': len(merged_clusters),  # 新的聚类编号
            'longitude': merged_lon,
            'latitude': merged_lat,
            'count': total_count,
            'original_clusters': [df.iloc[idx]['cluster'] for idx in clusters_to_merge],
            'original_sources': [f"{df.iloc[idx]['source_file']}:{df.iloc[idx]['cluster']}" for idx in clusters_to_merge],
            'merged_from_count': len(clusters_to_merge)
        })
    
    return pd.DataFrame(merged_clusters)

def collect_all_centroids_and_merge(base_folder="cluster_res", distance_threshold=500):
    """
    收集所有CSV文件中的聚类中心点，然后合并相近的聚类簇
    
    Args:
        base_folder (str): 包含子文件夹的基础文件夹名称
        distance_threshold (int): 合并距离阈值（米）
    """
    if not os.path.isdir(base_folder):
        print(f"错误: 文件夹 '{base_folder}' 不存在。")
        return

    subfolders = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]
    
    if not subfolders:
        print(f"在 '{base_folder}' 中没有找到子文件夹。")
        return

    # 收集所有聚类中心点
    all_centroids = []
    
    print("=" * 80)
    print("第一步：收集所有聚类中心点")
    print("=" * 80)

    for subfolder in subfolders:
        subfolder_path = os.path.join(base_folder, subfolder)
        csv_files = [f for f in os.listdir(subfolder_path) if f.endswith(".csv")]

        if not csv_files:
            print(f"在子文件夹 '{subfolder}' 中没有找到 .csv 文件。")
            continue

        for csv_file in csv_files:
            file_path = os.path.join(subfolder_path, csv_file)
            print(f"\n--- 正在处理文件: {file_path} ---")
            try:
                df = pd.read_csv(file_path)

                # 确保必需的列存在
                required_columns = ['longitude', 'latitude', 'cluster', 'point_index']
                if not all(col in df.columns for col in required_columns):
                    print(f"错误: 文件 '{csv_file}' 缺少必要的列。需要: {required_columns}")
                    continue

                # 筛选掉 cluster = -1 的噪声点
                df_filtered = df[df['cluster'] != -1]

                if df_filtered.empty:
                    print(f"文件中没有有效的聚类（所有点都是噪声或文件为空）。")
                    continue

                # 计算每个 cluster 的中心点和点数
                centroids_and_counts = df_filtered.groupby('cluster').agg(
                    longitude=('longitude', 'mean'),
                    latitude=('latitude', 'mean'),
                    count=('cluster', 'size')
                ).reset_index()

                print(f"文件 '{csv_file}' 中的有效聚类簇个数: {len(centroids_and_counts)}")
                
                # 为每个中心点添加来源信息
                for index, row in centroids_and_counts.iterrows():
                    centroid_info = {
                        'cluster': int(row['cluster']),
                        'longitude': row['longitude'],
                        'latitude': row['latitude'],
                        'count': int(row['count']),
                        'source_file': csv_file,
                        'source_path': file_path
                    }
                    all_centroids.append(centroid_info)
                    
                    print(f"  聚类编号: {int(row['cluster'])}, "
                          f"中心点经度: {row['longitude']:.6f}, "
                          f"中心点纬度: {row['latitude']:.6f}, "
                          f"点数: {int(row['count'])}")

            except Exception as e:
                print(f"处理文件 '{file_path}' 时出错: {e}")

    if not all_centroids:
        print("没有找到任何有效的聚类中心点。")
        return

    # 将所有中心点转换为DataFrame
    all_centroids_df = pd.DataFrame(all_centroids)
    
    print("\n" + "=" * 80)
    print(f"第二步：收集完成，共找到 {len(all_centroids_df)} 个聚类中心点")
    print("=" * 80)
    
    print("所有聚类中心点汇总:")
    for index, row in all_centroids_df.iterrows():
        print(f"  {index+1}. 来源: {row['source_file']}, "
              f"聚类编号: {row['cluster']}, "
              f"经度: {row['longitude']:.6f}, "
              f"纬度: {row['latitude']:.6f}, "
              f"点数: {row['count']}")

    # 合并相近的聚类簇
    print("\n" + "=" * 80)
    print(f"第三步：合并距离 {distance_threshold} 米以内的聚类中心点")
    print("=" * 80)
    
    merged_clusters = merge_nearby_clusters(all_centroids_df, distance_threshold)
    
    print(f"合并前聚类中心点数量: {len(all_centroids_df)}")
    print(f"合并后聚类簇数量: {len(merged_clusters)}")
    print(f"合并阈值: {distance_threshold}米")
    print("\n合并后的聚类簇信息:")
    
    for index, row in merged_clusters.iterrows():
        print(f"  新聚类编号: {int(row['cluster'])}")
        print(f"    中心点经度: {row['longitude']:.6f}")
        print(f"    中心点纬度: {row['latitude']:.6f}")
        print(f"    总点数: {int(row['count'])}")
        print(f"    合并自 {row['merged_from_count']} 个原始聚类簇: {row['original_sources']}")
        print()
    
    # 保存合并结果
    output_file = os.path.join(base_folder, "all_merged_clusters.csv")
    merged_clusters[['cluster', 'longitude', 'latitude', 'count', 'merged_from_count']].to_csv(
        output_file, index=False
    )
    
    # 保存详细合并信息
    detailed_output_file = os.path.join(base_folder, "all_merged_clusters_detailed.csv")
    merged_clusters.to_csv(detailed_output_file, index=False)
    
    print(f"合并结果已保存到: {output_file}")
    print(f"详细合并信息已保存到: {detailed_output_file}")
    
    return merged_clusters

def process_csv_files_and_find_centroids_with_counts(base_folder="cluster_res"):
    """
    原始函数：读取CSV文件，计算聚类中心点和点数，并报告聚类数量
    """
    if not os.path.isdir(base_folder):
        print(f"错误: 文件夹 '{base_folder}' 不存在。")
        return

    subfolders = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]
    
    if not subfolders:
        print(f"在 '{base_folder}' 中没有找到子文件夹。")
        return

    for subfolder in subfolders:
        subfolder_path = os.path.join(base_folder, subfolder)
        csv_files = [f for f in os.listdir(subfolder_path) if f.endswith(".csv")]

        if not csv_files:
            print(f"在子文件夹 '{subfolder}' 中没有找到 .csv 文件。")
            continue

        for csv_file in csv_files:
            file_path = os.path.join(subfolder_path, csv_file)
            print(f"\n--- 正在处理文件: {file_path} ---")
            try:
                df = pd.read_csv(file_path)

                # 确保必需的列存在
                required_columns = ['longitude', 'latitude', 'cluster', 'point_index']
                if not all(col in df.columns for col in required_columns):
                    print(f"错误: 文件 '{csv_file}' 缺少必要的列。需要: {required_columns}")
                    continue

                # 筛选掉 cluster = -1 的噪声点
                df_filtered = df[df['cluster'] != -1]

                if df_filtered.empty:
                    print(f"文件中没有有效的聚类（所有点都是噪声或文件为空）。")
                    print("-" * 50)
                    continue

                # 计算每个 cluster 的中心点 (经度和纬度的平均值)
                # 同时计算每个 cluster 的点数 (size)
                centroids_and_counts = df_filtered.groupby('cluster').agg(
                    longitude=('longitude', 'mean'),  # 计算经度平均值
                    latitude=('latitude', 'mean'),    # 计算纬度平均值
                    count=('cluster', 'size')          # 计算每个簇的大小（点数）
                ).reset_index()

                # 获取当前文件中聚类簇的个数
                num_clusters = len(centroids_and_counts)

                print(f"文件 '{csv_file}' 中的有效聚类簇个数: {num_clusters}")
                print("每个聚类簇的中心点和点数信息:")
                
                # 打印每个聚类簇的中心点和点数信息
                for index, row in centroids_and_counts.iterrows():
                    print(f"  聚类编号: {int(row['cluster'])}, "
                          f"中心点经度: {row['longitude']:.6f}, "
                          f"中心点纬度: {row['latitude']:.6f}, "
                          f"点数: {int(row['count'])}")
                
                print("-" * 50)

            except Exception as e:
                print(f"处理文件 '{file_path}' 时出错: {e}")

# 运行脚本
if __name__ == "__main__":
    # 安装依赖: pip install geopy pandas
    
    print("选择运行模式:")
    print("1. 原始模式：分别处理每个CSV文件并显示聚类中心点")
    print("2. 合并模式：收集所有聚类中心点并合并相近的聚类簇")
    
    choice = input("请输入选择 (1 或 2，直接回车默认选择模式2): ").strip()
    
    if choice == "1":
        print("\n运行原始模式...")
        process_csv_files_and_find_centroids_with_counts()
    else:
        print("\n运行合并模式...")
        collect_all_centroids_and_merge()