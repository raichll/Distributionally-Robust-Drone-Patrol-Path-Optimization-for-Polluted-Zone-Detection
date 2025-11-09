"""
Demonstrate a simple usage of the dynamic DBSCAN clustering algorithm on real-world coordinates 
and visualize its results with proper geographic context.
"""
import matplotlib.pyplot as plt
import numpy as np
import dbscan.dynamic_fdbscan
import alglab.dataset_base
from matplotlib.patches import Circle
import seaborn as sns

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    计算两个地理坐标点之间的距离（单位：米）
    使用 Haversine 公式
    """
    # 转换为弧度
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine 公式
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # 地球半径（米）
    R = 6371000
    return R * c

def calculate_optimal_eps(data, k=4):
    """
    计算最优的 eps 参数，基于 k-distance 图
    """
    from sklearn.neighbors import NearestNeighbors
    
    # 使用欧几里得距离计算 k-nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k).fit(data)
    distances, indices = nbrs.kneighbors(data)
    
    # 对距离进行排序
    distances = np.sort(distances[:, k-1], axis=0)
    
    # 寻找拐点（可以使用更复杂的方法，这里简化处理）
    # 取距离数组的 95% 分位数作为建议的 eps 值
    optimal_eps = np.percentile(distances, 95)
    
    return optimal_eps, distances

def plot_k_distance_graph(distances, optimal_eps):
    """
    绘制 k-distance 图用于确定最优 eps
    """
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(distances)), distances, 'b-')
    plt.axhline(y=optimal_eps, color='r', linestyle='--', 
                label=f'Suggested eps = {optimal_eps:.6f}')
    plt.xlabel('Points sorted by distance')
    plt.ylabel('4th nearest neighbor distance')
    plt.title('K-Distance Graph for Optimal Eps Selection')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    # 加载轨迹数据集
    print("Loading trajectory dataset...")
    base_path = "E:/dwd_truck"
    dates = ["2024-04-14"]
    truck_dataset = alglab.dataset_base.LocalTrajectoryDataset(base_folder=base_path, date_range=dates)
    data = truck_dataset.data
    # 检查数据是否已经缩放，如果需要的话应用缩放
    '''if hasattr(truck_dataset, 'apply_scaling'):
        truck_dataset.apply_scaling()'''
    
    data = truck_dataset.data
    print(f"Data shape: {data.shape}")
    print(f"Data range - X: [{data[:, 0].min():.6f}, {data[:, 0].max():.6f}]")
    print(f"Data range - Y: [{data[:, 1].min():.6f}, {data[:, 1].max():.6f}]")
    
    # 如果数据看起来像是经纬度坐标（通常很小的数值）
    is_geographic = (abs(data[:, 0]).max() < 180 and abs(data[:, 1]).max() < 180 and 
                    abs(data[:, 0]).max() > 0.001 and abs(data[:, 1]).max() > 0.001)
    
    if is_geographic:
        print("检测到地理坐标数据 (经纬度)")
        coordinate_type = "Geographic (Lat/Lon)"
        x_label = "Longitude (°)"
        y_label = "Latitude (°)"
    else:
        print("检测到投影坐标数据 (米制)")
        coordinate_type = "Projected (meters)"
        x_label = "X (m)"
        y_label = "Y (m)"
    
    # 计算最优的 eps 参数
    print("Calculating optimal eps parameter...")
    optimal_eps, distances = calculate_optimal_eps(data)
    print(f"Suggested eps parameter: {optimal_eps:.8f}")
    
    # 显示 k-distance 图
    plot_k_distance_graph(distances, optimal_eps)
    
    # 询问用户是否使用建议的 eps 值
    use_suggested = input(f"Use suggested eps value ({optimal_eps:.8f})? (y/n): ").lower().strip()
    
    if use_suggested == 'y':
        eps_value = optimal_eps
    else:
        eps_input = input("Enter custom eps value: ").strip()
        try:
            eps_value = float(eps_input)
        except ValueError:
            print("Invalid input, using suggested value")
            eps_value = optimal_eps
    
    # 根据数据大小调整参数
    data_size = data.shape[0]
    if data_size < 500:
        k_param = max(5, data_size // 100)
        t_param = max(3, k_param // 2)
        min_samples = max(3, data_size // 200)
    else:
        k_param = max(10, data_size // 100)
        t_param = max(5, k_param // 2)
        min_samples = max(5, data_size // 200)
    
    initial_points = min(100, data_size // 10)
    
    print(f"DBSCAN Parameters:")
    print(f"  k (neighborhood size): {k_param}")
    print(f"  t (time threshold): {t_param}")
    print(f"  eps (distance threshold): {eps_value:.8f}")
    print(f"  min_samples (dimension): 2")
    print(f"  initial_points: {initial_points}")
    
    # 初始化动态 DBSCAN
    print("Initializing Dynamic DBSCAN...")
    dbscan_alg = dbscan.dynamic_fdbscan.DynamicDBSCAN(
        k=k_param, 
        t=t_param, 
        eps=eps_value, 
        d=2,  # 2D数据
        initial_data=data[:initial_points, :]
    )
    
    # 添加剩余数据点
    print(f"Adding remaining {data_size - initial_points} points...")
    for i in range(initial_points, data_size):
        dbscan_alg.add_point(data[i, :])
        if (i - initial_points) % 1000 == 0:
            print(f"  Added {i - initial_points} points...")
    
    # 获取所有数据点的聚类标签
    print("Getting cluster labels...")
    predicted_labels = np.array([dbscan_alg.get_cluster(i) for i in range(data_size)])
    
    # 分析聚类结果
    unique_labels = set(predicted_labels)
    n_clusters = len(unique_labels) - (1 if -1 in predicted_labels else 0)
    n_noise = list(predicted_labels).count(-1)
    
    print(f"\nClustering Results:")
    print(f"  Number of clusters: {n_clusters}")
    print(f"  Number of noise points: {n_noise}")
    print(f"  Percentage of noise: {n_noise/data_size*100:.1f}%")
    
    # 显示每个聚类的统计信息
    for label in sorted(unique_labels):
        if label != -1:
            cluster_points = np.sum(predicted_labels == label)
            print(f"  Cluster {label}: {cluster_points} points")
    
    # 创建可视化图表
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Dynamic DBSCAN Clustering Results\n{coordinate_type} Data', 
                fontsize=16, fontweight='bold')
    
    # 子图1: 原始数据分布
    ax1 = axes[0, 0]
    ax1.scatter(data[:, 0], data[:, 1], c='lightblue', alpha=0.6, s=20, edgecolors='navy', linewidth=0.5)
    ax1.set_title('Original Data Distribution', fontweight='bold')
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 聚类结果（彩色）
    ax2 = axes[0, 1]
    
    # 生成颜色映射
    if n_clusters > 0:
        colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))
        color_map = {}
        cluster_labels = [label for label in unique_labels if label != -1]
        for i, label in enumerate(sorted(cluster_labels)):
            color_map[label] = colors[i % len(colors)]
    
    # 绘制聚类点
    for label in unique_labels:
        if label == -1:  # 噪声点
            mask = predicted_labels == label
            ax2.scatter(data[mask, 0], data[mask, 1], c='black', marker='x', 
                       s=30, alpha=0.7, label='Noise')
        else:  # 聚类点
            mask = predicted_labels == label
            ax2.scatter(data[mask, 0], data[mask, 1], c=[color_map[label]], 
                       s=40, alpha=0.8, label=f'Cluster {label}', edgecolors='black', linewidth=0.5)
    
    ax2.set_title(f'Clustering Results ({n_clusters} clusters)', fontweight='bold')
    ax2.set_xlabel(x_label)
    ax2.set_ylabel(y_label)
    ax2.grid(True, alpha=0.3)
    
    # 只显示前10个聚类的图例，避免图例过于拥挤
    handles, labels = ax2.get_legend_handles_labels()
    if len(handles) > 11:  # 10个聚类 + 噪声
        ax2.legend(handles[:11], labels[:11], bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 子图3: 密度热图
    ax3 = axes[1, 0]
    # 创建2D直方图显示点密度
    hist, xedges, yedges = np.histogram2d(data[:, 0], data[:, 1], bins=50)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax3.imshow(hist.T, extent=extent, origin='lower', cmap='YlOrRd', alpha=0.8)
    ax3.set_title('Data Density Heatmap', fontweight='bold')
    ax3.set_xlabel(x_label)
    ax3.set_ylabel(y_label)
    plt.colorbar(im, ax=ax3, label='Point Density')
    
    # 子图4: 聚类大小分布
    ax4 = axes[1, 1]
    cluster_sizes = []
    cluster_ids = []
    for label in sorted(unique_labels):
        if label != -1:
            size = np.sum(predicted_labels == label)
            cluster_sizes.append(size)
            cluster_ids.append(f'C{label}')
    
    if cluster_sizes:
        bars = ax4.bar(cluster_ids, cluster_sizes, color='steelblue', alpha=0.7, edgecolor='navy')
        ax4.set_title('Cluster Size Distribution', fontweight='bold')
        ax4.set_xlabel('Cluster ID')
        ax4.set_ylabel('Number of Points')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 在柱状图上显示数值
        for bar, size in zip(bars, cluster_sizes):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + max(cluster_sizes)*0.01,
                    f'{size}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # 保存聚类结果
    save_results = input("Save clustering results to file? (y/n): ").lower().strip()
    if save_results == 'y':
        # 创建结果数组
        results = np.column_stack((data, predicted_labels))
        
        # 保存为CSV文件
        header = f"{x_label.replace(' ', '_')},{y_label.replace(' ', '_')},Cluster_ID"
        filename = f"clustering_results_{coordinate_type.replace(' ', '_').replace('(', '').replace(')', '')}.csv"
        np.savetxt(filename, results, delimiter=',', header=header, comments='', fmt='%.8f,%.8f,%d')
        print(f"Results saved to: {filename}")
        
        # 保存图表
        fig.savefig(filename.replace('.csv', '.png'), dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {filename.replace('.csv', '.png')}")

if __name__ == "__main__":
    main()