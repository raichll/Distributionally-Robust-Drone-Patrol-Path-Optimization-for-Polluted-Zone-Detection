"""In this file, we specify the experiments we will run."""
import alglab.algorithm
import alglab.dataset
import alglab.dataset_base
import alglab.experiment
import alglab.evaluation
import alglab.results
import os
import numpy as np
import math
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from algorithms import *
from sklearn.neighbors import BallTree
from alglab.dataset_base import LocalTrajectoryDataset
from Visualization import generate_2d_clustering_results
import folium

def generate_geo_clustering_results(static_dataset, results, name, dir_path, dynamic_dataset=None):
    """
    Generate map-based clustering result visualization for 2D longitude-latitude data
    Optimized version: Smart sampling, only renders cluster points and up to 1000 noise points
    
    Args:
        static_dataset: Static dataset (contains longitude-latitude coordinates)
        results: Experiment results
        name: Dataset name
        dir_path: Output directory path
        dynamic_dataset: Dynamic dataset (if provided, will be used directly; otherwise recreated)
    """
    print(f"Generating geographic clustering results for {name} data...")
    
    # Create output directory
    output_dir = os.path.join(dir_path, f"clustering_results/{name}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Read result data
    results_df = results.results_df
    
    # Get final clustering results (last iteration results)
    final_results = results_df.groupby('algorithm').last().reset_index()
    
    # Save detailed clustering results
    clustering_results = {}
    
    # If dynamic_dataset is not provided, recreate it
    if dynamic_dataset is None:
        try:
            n_init_points = 1000
            if static_dataset.n < n_init_points:
                n_init_points = static_dataset.n // 2 if static_dataset.n > 1 else 1
            
            dynamic_dataset_for_rerun = alglab.dataset.DynamicPointCloudDataset.from_pointcloud(
                static_dataset, n_init_points, stream_by_cluster=False
            )
        except Exception as e:
            print(f"Error creating DynamicPointCloudDataset: {e}. Unable to rerun algorithms to get cluster labels.")
            return
    else:
        dynamic_dataset_for_rerun = dynamic_dataset
        
    algorithms_to_run = [
        ('dynamic_dbscan', dynamic_dbscan_alg),
        ('fdbscan_dynamic_naive', fdbscan_dynamic_naive),
        ('fdbscan_dynamic_noncore', fdbscan_dynamic_noncore),
        ('Ada_dynamic_dbscan', Ada_dynamic_dbscan_alg)
    ]
    
    # ✅ Prioritize using unscaled original data
    if hasattr(static_dataset, 'original_data'):
        coordinates = static_dataset.original_data
        print("Using original unscaled data for visualization")
    else:
        coordinates = static_dataset.data  # fallback
        print("Warning: original_data not found, using scaled data for visualization")
    
    # Check data range to determine if it's longitude-latitude data
    coord_min = np.min(coordinates, axis=0)
    coord_max = np.max(coordinates, axis=0)
    
    print(f"Coordinate range: X({coord_min[0]:.6f}, {coord_max[0]:.6f}), Y({coord_min[1]:.6f}, {coord_max[1]:.6f})")
    
    # Determine coordinate format (longitude range typically -180 to 180, latitude range -90 to 90)
    if abs(coord_max[0]) <= 180 and abs(coord_min[0]) <= 180 and abs(coord_max[1]) <= 90 and abs(coord_min[1]) <= 90:
        # Format: [longitude, latitude]
        lons = coordinates[:, 0]
        lats = coordinates[:, 1]
        print("Detected coordinate format: [longitude, latitude]")
    elif abs(coord_max[1]) <= 180 and abs(coord_min[1]) <= 180 and abs(coord_max[0]) <= 90 and abs(coord_min[0]) <= 90:
        # Format: [latitude, longitude]
        lats = coordinates[:, 0]
        lons = coordinates[:, 1]
        print("Detected coordinate format: [latitude, longitude]")
    else:
        # May not be standard longitude-latitude format, use planar coordinates
        print("Warning: Coordinates may not be in standard longitude-latitude format, will use planar coordinates")
        lons = coordinates[:, 0]
        lats = coordinates[:, 1]
    
    # Run algorithms to get cluster labels
    for alg_name, alg in algorithms_to_run:
        try:
            # Get parameters for this algorithm
            alg_params = {}
            alg_result = results_df[results_df['algorithm'] == alg_name]
            
            if alg_result.empty:
                print(f"No results found for algorithm {alg_name}, skipping...")
                continue
            
            # Get parameters, ensure correct type
            param_mapping = {
                'eps': float,
                'min_samples': int,
                't': int,
                'd': int,
                'burnin': int
            }
            
            for param_col, param_type in param_mapping.items():
                if param_col in alg_result.columns:
                    param_value = alg_result[param_col].iloc[0]
                    if pd.notna(param_value):
                        try:
                            alg_params[param_col] = param_type(param_value)
                        except (ValueError, TypeError):
                            print(f"Parameter {param_col} type conversion failed, using default value")
                            continue
            
            print(f"Running algorithm {alg_name} to get cluster labels, parameters: {alg_params}")
            
            # Run algorithm to get cluster labels
            alg_generator = alg.run(dynamic_dataset_for_rerun, alg_params)
            final_labels = None
            
            # Get final clustering results
            for labels in alg_generator:
                final_labels = labels
            
            if final_labels is not None:
                clustering_results[alg_name] = final_labels
                
                # Save clustering results to CSV
                result_df = pd.DataFrame({
                    'longitude': lons,
                    'latitude': lats,
                    'cluster_label': final_labels
                })
                result_df.to_csv(os.path.join(output_dir, f"{alg_name}_clustering_results.csv"), index=False)
                
                # Statistics on clustering information
                unique_labels = np.unique(final_labels)
                n_clusters = len(unique_labels[unique_labels >= 0])
                n_noise = np.sum(final_labels == -1)
                
                print(f"Algorithm {alg_name}:")
                print(f"  - Number of clusters: {n_clusters}")
                print(f"  - Number of noise points: {n_noise}")
                print(f"  - Cluster label range: {unique_labels}")
                
        except Exception as e:
            print(f"Error running algorithm {alg_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Generate geographic clustering result visualization
    if clustering_results:
        generate_optimized_geo_clustering_visualization(lons, lats, clustering_results, output_dir)
        generate_optimized_matplotlib_clustering_visualization(lons, lats, clustering_results, output_dir)
    
    # Generate clustering result statistics report
    generate_clustering_report(clustering_results, results_df, output_dir)
    
    print(f"Clustering results saved to: {output_dir}")


def smart_sampling(indices, max_points, coordinates=None, method='random'):
    """
    Smart sampling function for selecting representative points from a large number of points
    
    Args:
        indices: Indices of points to sample
        max_points: Maximum number of sample points
        coordinates: Coordinate data, used for spatial sampling
        method: Sampling method ('random', 'spatial', 'uniform')
    
    Returns:
        Sampled indices
    """
    if len(indices) <= max_points:
        return indices
    
    if method == 'random':
        return np.random.choice(indices, max_points, replace=False)
    
    elif method == 'spatial' and coordinates is not None:
        # Use K-means for spatial sampling, select points near cluster centers
        try:
            sample_coords = coordinates[indices]
            n_clusters = min(max_points, len(indices))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(sample_coords)
            
            # Select the point closest to the center from each cluster
            selected_indices = []
            for i in range(n_clusters):
                cluster_mask = cluster_labels == i
                if np.any(cluster_mask):
                    cluster_indices = indices[cluster_mask]
                    cluster_coords = sample_coords[cluster_mask]
                    center = kmeans.cluster_centers_[i]
                    
                    # Calculate distance to cluster center
                    distances = np.sum((cluster_coords - center) ** 2, axis=1)
                    closest_idx = np.argmin(distances)
                    selected_indices.append(cluster_indices[closest_idx])
            
            return np.array(selected_indices)
        except:
            # If spatial sampling fails, fallback to random sampling
            return np.random.choice(indices, max_points, replace=False)
    
    elif method == 'uniform':
        # Uniform sampling
        step = len(indices) // max_points
        return indices[::step][:max_points]
    
    else:
        # Default random sampling
        return np.random.choice(indices, max_points, replace=False)


def generate_optimized_geo_clustering_visualization(lons, lats, clustering_results, output_dir):
    """
    Optimized Folium map visualization: Smart sampling and performance optimization
    """
    print("Generating optimized map-based clustering visualization...")
    
    # Calculate map center point
    center_lat = np.mean(lats)
    center_lon = np.mean(lons)
    
    # Calculate appropriate zoom level
    lat_range = np.max(lats) - np.min(lats)
    lon_range = np.max(lons) - np.min(lons)
    max_range = max(lat_range, lon_range)
    
    if max_range > 10:
        zoom_level = 6
    elif max_range > 1:
        zoom_level = 9
    elif max_range > 0.1:
        zoom_level = 12
    else:
        zoom_level = 15
    
    print(f"Map center: ({center_lat:.6f}, {center_lon:.6f}), zoom level: {zoom_level}")
    
    # Define color list
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 
              'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white', 
              'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray']
    
    coordinates = np.column_stack([lons, lats])
    
    # Generate map for each algorithm
    for alg_name, labels in clustering_results.items():
        # Create map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom_level,
            tiles='CartoDB positron'
        )
        
        # Get unique cluster labels
        unique_labels = np.unique(labels)
        
        # Assign colors to each cluster
        color_map = {}
        for i, label in enumerate(unique_labels):
            if label == -1:
                color_map[label] = 'black'  # Noise points in black
            else:
                color_map[label] = colors[i % len(colors)]
        
        # Separate cluster points and non-cluster points
        cluster_mask = labels != -1
        noise_mask = labels == -1
        
        cluster_indices = np.where(cluster_mask)[0]
        noise_indices = np.where(noise_mask)[0]
        
        # 🎯 Smart sampling of noise points
        max_noise_points = 1000
        if len(noise_indices) > max_noise_points:
            selected_noise_indices = smart_sampling(
                noise_indices, max_noise_points, coordinates, method='spatial'
            )
            print(f"Noise points smart sampling: {len(noise_indices)} -> {len(selected_noise_indices)}")
        else:
            selected_noise_indices = noise_indices
        
        # 🎯 Optional: If too many cluster points, also sample them
        max_cluster_points = 5000  # Set maximum display count for cluster points
        if len(cluster_indices) > max_cluster_points:
            selected_cluster_indices = smart_sampling(
                cluster_indices, max_cluster_points, coordinates, method='spatial'
            )
            print(f"Cluster points smart sampling: {len(cluster_indices)} -> {len(selected_cluster_indices)}")
        else:
            selected_cluster_indices = cluster_indices
        
        # Add cluster points to map (use batch addition for better performance)
        cluster_points = []
        for i in selected_cluster_indices:
            lon, lat, label = lons[i], lats[i], labels[i]
            cluster_points.append({
                'location': [lat, lon],
                'radius': 4,
                'popup': f'Cluster {label}, Point {i}',
                'color': color_map[label],
                'fillColor': color_map[label],
                'fillOpacity': 0.8,
                'weight': 2
            })
        
        # Batch add cluster points
        for point in cluster_points:
            folium.CircleMarker(**point).add_to(m)
        
        # Add noise points to map
        noise_points = []
        for i in selected_noise_indices:
            lon, lat = lons[i], lats[i]
            noise_points.append({
                'location': [lat, lon],
                'radius': 3,
                'popup': f'Noise Point {i}',
                'color': 'black',
                'fillColor': 'black',
                'fillOpacity': 0.7,
                'weight': 1
            })
        
        # Batch add noise points
        for point in noise_points:
            folium.CircleMarker(**point).add_to(m)
        
        # Add enhanced legend
        total_clusters = len(unique_labels[unique_labels >= 0])
        total_noise = len(noise_indices)
        displayed_noise = len(selected_noise_indices)
        total_cluster_points = len(cluster_indices)
        displayed_cluster_points = len(selected_cluster_indices)
        
        legend_html = f'''
        <div style="position: fixed; 
                    top: 10px; right: 10px; width: 250px; height: auto; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:12px; padding: 10px; border-radius: 5px;">
        <h4 style="margin-top:0; color: #333;">{alg_name}</h4>
        <p><strong>Total Points:</strong> {len(labels)}</p>
        <p><strong>Clusters:</strong> {total_clusters}</p>
        <p><strong>Cluster Points:</strong> {displayed_cluster_points}/{total_cluster_points}</p>
        <p><strong>Noise Points:</strong> {displayed_noise}/{total_noise}</p>
        <hr style="margin: 5px 0;">
        '''
        
        # Show detailed information for the first 10 clusters
        cluster_info = []
        for label in unique_labels:
            if label == -1:
                cluster_info.append(f'<p style="margin:2px 0;"><span style="color:black; font-size:16px;">●</span> Noise ({displayed_noise}/{total_noise})</p>')
            else:
                cluster_size = np.sum(labels == label)
                cluster_info.append(f'<p style="margin:2px 0;"><span style="color:{color_map[label]}; font-size:16px;">●</span> Cluster {label} ({cluster_size} points)</p>')
        
        # Only show first 10 clusters to avoid overly long legend
        for info in cluster_info[:10]:
            legend_html += info
        
        if len(cluster_info) > 10:
            legend_html += f'<p style="margin:2px 0; font-style:italic;">... and {len(cluster_info) - 10} more clusters</p>'
        
        legend_html += '</div>'
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Save map
        map_filename = os.path.join(output_dir, f"{alg_name}_optimized_geo_clustering_map.html")
        m.save(map_filename)
        print(f"Optimized map saved: {map_filename}")
        print(f"  - Displayed cluster points: {len(selected_cluster_indices)}/{len(cluster_indices)}")
        print(f"  - Displayed noise points: {len(selected_noise_indices)}/{len(noise_indices)}")
    
    print("Optimized map-based clustering visualization completed")


def generate_optimized_matplotlib_clustering_visualization(lons, lats, clustering_results, output_dir):
    """
    Optimized Matplotlib clustering visualization: Smart sampling and beautification
    """
    print("Generating optimized matplotlib clustering visualization...")
    
    # Generate scatter plot for each algorithm
    n_algorithms = len(clustering_results)
    
    if n_algorithms == 1:
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        axes = [ax]
    elif n_algorithms == 2:
        fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    elif n_algorithms <= 4:
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))
        axes = axes.flatten()
    
    # Use better color mapping
    colors = plt.cm.Set3(np.linspace(0, 1, 20))
    coordinates = np.column_stack([lons, lats])
    
    for idx, (alg_name, labels) in enumerate(clustering_results.items()):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        # Get unique cluster labels
        unique_labels = np.unique(labels)
        
        # Separate cluster points and non-cluster points
        cluster_mask = labels != -1
        noise_mask = labels == -1
        
        cluster_indices = np.where(cluster_mask)[0]
        noise_indices = np.where(noise_mask)[0]
        
        # 🎯 Smart sampling of noise points
        max_noise_points = 1000
        if len(noise_indices) > max_noise_points:
            selected_noise_indices = smart_sampling(
                noise_indices, max_noise_points, coordinates, method='spatial'
            )
        else:
            selected_noise_indices = noise_indices
        
        # 🎯 Optional: If too many cluster points, also sample them
        max_cluster_points = 5000
        if len(cluster_indices) > max_cluster_points:
            selected_cluster_indices = smart_sampling(
                cluster_indices, max_cluster_points, coordinates, method='spatial'
            )
        else:
            selected_cluster_indices = cluster_indices
        
        # Plot each cluster
        for label in unique_labels:
            if label == -1:
                # Only plot selected noise points
                if len(selected_noise_indices) > 0:
                    selected_mask = np.isin(np.arange(len(labels)), selected_noise_indices)
                    final_noise_mask = (labels == -1) & selected_mask
                    ax.scatter(lons[final_noise_mask], lats[final_noise_mask], 
                              c='black', marker='x', s=25, alpha=0.6, 
                              label=f'Noise ({len(selected_noise_indices)}/{len(noise_indices)})')
            else:
                # Plot selected cluster points
                cluster_points_mask = (labels == label)
                if len(selected_cluster_indices) < len(cluster_indices):
                    # If sampling was performed, only show sampled points
                    selected_mask = np.isin(np.arange(len(labels)), selected_cluster_indices)
                    cluster_points_mask = cluster_points_mask & selected_mask
                
                if np.any(cluster_points_mask):
                    cluster_size = np.sum(labels == label)  # Show real cluster size
                    displayed_points = np.sum(cluster_points_mask)
                    ax.scatter(lons[cluster_points_mask], lats[cluster_points_mask], 
                              c=[colors[label % len(colors)]], s=35, alpha=0.8, 
                              label=f'Cluster {label} ({displayed_points}/{cluster_size})')
        
        # Calculate display statistics
        total_displayed = len(selected_cluster_indices) + len(selected_noise_indices)
        total_points = len(labels)
        
        ax.set_title(f'{alg_name} Clustering Results\nShowing {total_displayed}/{total_points} points '
                    f'(clusters:{len(selected_cluster_indices)}/{len(cluster_indices)}, '
                    f'noise:{len(selected_noise_indices)}/{len(noise_indices)})', 
                    fontsize=12, fontweight='bold', pad=20)
        
        ax.set_xlabel('Longitude', fontsize=11)
        ax.set_ylabel('Latitude', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Beautify coordinate axes
        ax.ticklabel_format(style='plain', axis='both')
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        # Smart legend display
        handles, labels_legend = ax.get_legend_handles_labels()
        if len(handles) > 12:
            # If too many clusters, show first 10 clusters + noise points
            noise_handles = [h for h, l in zip(handles, labels_legend) if 'Noise' in l]
            cluster_handles = [h for h, l in zip(handles, labels_legend) if 'Cluster' in l][:10]
            final_handles = cluster_handles + noise_handles
            final_labels = [l for h, l in zip(handles, labels_legend) if h in final_handles]
            ax.legend(final_handles, final_labels, loc='upper right', fontsize=8, 
                     framealpha=0.9, fancybox=True, shadow=True)
        else:
            ax.legend(loc='upper right', fontsize=8, framealpha=0.9, fancybox=True, shadow=True)
        
        # Set same coordinate axis range for comparison
        margin = 0.02
        lon_range = np.max(lons) - np.min(lons)
        lat_range = np.max(lats) - np.min(lats)
        ax.set_xlim(np.min(lons) - margin * lon_range, np.max(lons) + margin * lon_range)
        ax.set_ylim(np.min(lats) - margin * lat_range, np.max(lats) + margin * lat_range)
        
        # Add border
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        
        print(f"  {alg_name}: Showing {len(selected_cluster_indices)}/{len(cluster_indices)} cluster points, "
              f"{len(selected_noise_indices)}/{len(noise_indices)} noise points")
    
    # Hide extra subplots
    for idx in range(len(clustering_results), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout(pad=2.0)
    
    # Save high-quality images
    plt.savefig(os.path.join(output_dir, 'optimized_clustering_visualization.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(output_dir, 'optimized_clustering_visualization.pdf'), 
                bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("Optimized matplotlib clustering visualization generated")

def generate_clustering_report(clustering_results, results_df, output_dir):
    """
    生成详细的聚类结果统计报告
    """
    print("正在生成聚类统计报告...")
    
    report_lines = []
    report_lines.append("# 🎯 轨迹数据聚类结果报告")
    report_lines.append("=" * 60)
    report_lines.append("")
    
    # 总体统计
    if clustering_results:
        total_trajectories = len(list(clustering_results.values())[0])
        report_lines.append(f"📊 总轨迹数量: {total_trajectories:,}")
        report_lines.append(f"🔍 算法数量: {len(clustering_results)}")
        report_lines.append("")
    
    # 各算法统计
    for alg_name, labels in clustering_results.items():
        report_lines.append(f"## 🚀 {alg_name} 算法结果")
        report_lines.append("-" * 40)
        
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels[unique_labels >= 0])
        n_noise = np.sum(labels == -1)
        
        report_lines.append(f"📈 聚类数量: {n_clusters}")
        report_lines.append(f"🔸 噪声点数量: {n_noise:,}")
        report_lines.append(f"📊 噪声点比例: {n_noise/len(labels)*100:.2f}%")
        report_lines.append(f"✅ 有效聚类点: {len(labels) - n_noise:,}")
        
        # 渲染优化信息
        max_noise_display = 1000
        max_cluster_display = 5000
        displayed_noise = min(n_noise, max_noise_display)
        displayed_cluster = min(len(labels) - n_noise, max_cluster_display)
        
        report_lines.append(f"🎨 渲染优化:")
        report_lines.append(f"   - 聚类点显示: {displayed_cluster:,}/{len(labels) - n_noise:,}")
        report_lines.append(f"   - 噪声点显示: {displayed_noise:,}/{n_noise:,}")
        
        # 聚类大小分布
        if n_clusters > 0:
            report_lines.append(f"\n📋 聚类大小分布:")
            cluster_sizes = []
            for label in unique_labels:
                if label >= 0:
                    cluster_size = np.sum(labels == label)
                    cluster_sizes.append(cluster_size)
                    report_lines.append(f"   聚类 {label:2d}: {cluster_size:4d} 个轨迹")
            
            if cluster_sizes:
                report_lines.append(f"\n📊 聚类统计:")
                report_lines.append(f"   - 平均聚类大小: {np.mean(cluster_sizes):.1f}")
                report_lines.append(f"   - 最大聚类大小: {np.max(cluster_sizes)}")
                report_lines.append(f"   - 最小聚类大小: {np.min(cluster_sizes)}")
        
        report_lines.append("")
    
    # 运行时间统计
    if not results_df.empty:
        report_lines.append("## ⏱️ 运行时间统计")
        report_lines.append("-" * 40)
        
        final_times = results_df.groupby('algorithm')['total_running_time_s'].last()
        times_sorted = final_times.sort_values()
        
        for alg_name, time_taken in times_sorted.items():
            if time_taken < 1:
                time_str = f"{time_taken*1000:.1f} ms"
            elif time_taken < 60:
                time_str = f"{time_taken:.3f} s"
            else:
                time_str = f"{time_taken/60:.1f} min"
            report_lines.append(f"🕒 {alg_name}: {time_str}")
        
        # 性能对比
        if len(times_sorted) > 1:
            fastest = times_sorted.iloc[0]
            slowest = times_sorted.iloc[-1]
            speedup = slowest / fastest
            report_lines.append(f"\n📈 性能对比:")
            report_lines.append(f"   - 最快算法: {times_sorted.index[0]}")
            report_lines.append(f"   - 最慢算法: {times_sorted.index[-1]}")
            report_lines.append(f"   - 速度差异: {speedup:.2f}x")
        
        report_lines.append("")
    
    # 算法比较摘要
    if len(clustering_results) > 1:
        report_lines.append("## 🔍 算法比较摘要")
        report_lines.append("-" * 40)
        
        comparison_data = []
        for alg_name, labels in clustering_results.items():
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels[unique_labels >= 0])
            n_noise = np.sum(labels == -1)
            noise_ratio = n_noise / len(labels) * 100
            
            # 获取运行时间
            runtime = 0
            if not results_df.empty:
                alg_time = results_df[results_df['algorithm'] == alg_name]['total_running_time_s'].last()
                if pd.notna(alg_time):
                    runtime = alg_time
            
            comparison_data.append({
                'algorithm': alg_name,
                'clusters': n_clusters,
                'noise_ratio': noise_ratio,
                'runtime': runtime
            })
        
        # 排序显示
        comparison_df = pd.DataFrame(comparison_data)
        
        report_lines.append("📊 按聚类数量排序:")
        sorted_by_clusters = comparison_df.sort_values('clusters', ascending=False)
        for _, row in sorted_by_clusters.iterrows():
            report_lines.append(f"   {row['algorithm']}: {row['clusters']} 个聚类")
        
        report_lines.append(f"\n🔸 按噪声比例排序:")
        sorted_by_noise = comparison_df.sort_values('noise_ratio')
        for _, row in sorted_by_noise.iterrows():
            report_lines.append(f"   {row['algorithm']}: {row['noise_ratio']:.2f}% 噪声")
        
        if not results_df.empty:
            report_lines.append(f"\n⚡ 按运行时间排序:")
            sorted_by_time = comparison_df.sort_values('runtime')
            for _, row in sorted_by_time.iterrows():
                if row['runtime'] < 1:
                    time_str = f"{row['runtime']*1000:.1f}ms"
                else:
                    time_str = f"{row['runtime']:.3f}s"
                report_lines.append(f"   {row['algorithm']}: {time_str}")
        
        report_lines.append("")
    
    # 可视化文件列表
    report_lines.append("## 📁 生成的可视化文件")
    report_lines.append("-" * 40)
    report_lines.append("🗺️  交互式地图:")
    for alg_name in clustering_results.keys():
        report_lines.append(f"   - {alg_name}_optimized_geo_clustering_map.html")
    
    report_lines.append(f"\n📊 静态图表:")
    report_lines.append(f"   - optimized_clustering_visualization.png (高分辨率)")
    report_lines.append(f"   - optimized_clustering_visualization.pdf (矢量图)")
    
    report_lines.append(f"\n📋 数据文件:")
    for alg_name in clustering_results.keys():
        report_lines.append(f"   - {alg_name}_clustering_results.csv")
    
    report_lines.append(f"\n📄 报告文件:")
    report_lines.append(f"   - clustering_report.txt (本报告)")
    
    # 优化建议
    report_lines.append(f"\n## 💡 优化建议")
    report_lines.append("-" * 40)
    
    if clustering_results:
        total_points = len(list(clustering_results.values())[0])
        if total_points > 10000:
            report_lines.append("🚀 大数据集优化:")
            report_lines.append("   - 已启用智能采样，提高渲染性能")
            report_lines.append("   - 聚类点采样上限: 5,000个")
            report_lines.append("   - 噪声点采样上限: 1,000个")
            report_lines.append("   - 采样方法: 空间聚类采样")
        
        # 分析噪声比例
        noise_ratios = []
        for labels in clustering_results.values():
            noise_ratio = np.sum(labels == -1) / len(labels) * 100
            noise_ratios.append(noise_ratio)
        
        avg_noise_ratio = np.mean(noise_ratios)
        if avg_noise_ratio > 50:
            report_lines.append(f"\n⚠️  高噪声比例警告:")
            report_lines.append(f"   - 平均噪声比例: {avg_noise_ratio:.1f}%")
            report_lines.append(f"   - 建议调整参数以减少噪声点")
            report_lines.append(f"   - 考虑增加 min_samples 或减少 eps")
        
        # 分析聚类数量
        cluster_counts = []
        for labels in clustering_results.values():
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels[unique_labels >= 0])
            cluster_counts.append(n_clusters)
        
        if max(cluster_counts) > 50:
            report_lines.append(f"\n📊 聚类数量分析:")
            report_lines.append(f"   - 最大聚类数: {max(cluster_counts)}")
            report_lines.append(f"   - 图例显示限制: 前10个聚类")
            report_lines.append(f"   - 建议考虑增加 eps 以合并相近聚类")
    
    report_lines.append(f"\n## 🎯 使用说明")
    report_lines.append("-" * 40)
    report_lines.append("1. 🗺️  打开 HTML 文件查看交互式地图")
    report_lines.append("2. 📊 查看 PNG/PDF 文件进行算法对比")
    report_lines.append("3. 📋 使用 CSV 文件进行进一步分析")
    report_lines.append("4. 🔍 点击地图标记查看详细信息")
    report_lines.append("5. 📈 使用图例识别不同聚类")
    
    report_lines.append(f"\n" + "=" * 60)
    report_lines.append(f"📅 报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"🔧 优化版本: v2.0 (智能采样 + 性能优化)")
    report_lines.append("=" * 60)
    
    # 保存报告
    report_content = "\n".join(report_lines)
    with open(os.path.join(output_dir, 'clustering_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print("📊 聚类统计报告已生成: clustering_report.txt")
    print("\n" + "="*60)
    print("🎯 聚类结果摘要:")
    print("="*60)
    print(report_content)


def adaptive_dbscan_params(n, d,
                           X=None,
                           base_k=10, max_k=30,
                           base_t=10, max_t=20,
                           eps_quantile=0.6,
                           eps_range=None,  # 现在会根据维度自动设置
                           return_all=False,
                           verbose=False):
    """
    自适应调整 DBSCAN 的 (eps, k, t) 参数，适用于归一化或标准化数据。
    根据数据维度自动调整eps目标范围：高维→0.75，低维→0.2-0.3

    参数：
        n (int): 数据量
        d (int): 数据维度
        X (np.ndarray): 数据矩阵，建议已归一化
        base_k, max_k: min_samples 的估计范围
        base_t, max_t: 哈希函数数量 (DynamicDBSCAN 中使用)
        eps_quantile (float): quantile 比例（默认 0.6）
        eps_range (tuple or None): 若为 (min_eps, max_eps)，则使用指定范围；为 None 则根据维度自动设置
        return_all (bool): 是否返回中间计算值
        verbose (bool): 是否打印估计过程日志

    返回：
        eps, k, t，或 (eps, k, t, quantile_eps, elbow_eps)
    """

    logn = np.log(n)
    k = int(min(base_k + 0.6 * (logn - np.log(10000)), max_k))
    t = int(min(base_t + (logn - np.log(10000)) / np.log(2), max_t))

    quantile_eps, elbow_eps = None, None

    def get_dimensional_eps_range(dim):
        """根据维度平滑计算eps范围"""
        # 使用sigmoid函数实现平滑过渡
        # 低维起始值和高维目标值
        low_dim_min, low_dim_max = 0.2, 0.3
        high_dim_min, high_dim_max = 0.7, 0.75
        
        # 平滑过渡函数: 1 / (1 + exp(-k * (x - x0)))
        # 调整k控制过渡陡峭程度，x0控制过渡中心点
        transition_factor = 1 / (1 + np.exp(-0.15 * (dim - 15)))
        
        # 线性插值计算范围
        eps_min = low_dim_min + (high_dim_min - low_dim_min) * transition_factor
        eps_max = low_dim_max + (high_dim_max - low_dim_max) * transition_factor
        
        return (eps_min, eps_max)

    def fallback_eps(dim_range):
        """根据维度范围返回fallback eps"""
        return round((dim_range[0] + dim_range[1]) / 2, 4)

    def rescale(value, vmin, vmax, target_min, target_max):
        """将值从原始范围缩放到目标范围"""
        if vmax == vmin:
            return target_min
        scaled = (value - vmin) / (vmax - vmin)
        return target_min + scaled * (target_max - target_min)

    # 确定eps范围
    if eps_range is None:
        eps_range = get_dimensional_eps_range(d)
    
    eps = fallback_eps(eps_range)

    if X is not None and X.shape[0] >= k:
        try:
            nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(X)
            distances, _ = nbrs.kneighbors(X)
            k_dists = distances[:, -1]
            k_dists = k_dists[np.isfinite(k_dists)]
            k_dists_sorted = np.sort(k_dists)

            quantile_eps = np.quantile(k_dists_sorted, eps_quantile)
            diffs = np.gradient(k_dists_sorted)
            elbow_index = np.argmax(diffs)
            elbow_eps = k_dists_sorted[elbow_index]

            # 综合估计 raw_eps
            if abs(quantile_eps - elbow_eps) / max(quantile_eps, 1e-6) > 0.5:
                raw_eps = (quantile_eps + elbow_eps) / 2
            else:
                raw_eps = quantile_eps

            # 根据维度和数据分布智能调整eps
            data_range_min = np.min(k_dists_sorted)
            data_range_max = np.max(k_dists_sorted)
            
            # 如果数据的k距离范围很小，直接使用维度范围的中值
            data_range_span = data_range_max - data_range_min
            use_fallback = data_range_span < 1e-6
            
            eps = fallback_eps(eps_range) if use_fallback else rescale(
                raw_eps, data_range_min, data_range_max, eps_range[0], eps_range[1]
            )
            
            # 平滑的维度自适应调整
            # 使用tanh函数进行平滑调整，避免突变
            dim_adjustment_factor = np.tanh((d - 6.5) / 8)  # 在d=6.5附近平滑过渡
            
            # 计算当前eps在目标范围内的相对位置
            eps_relative_pos = (eps - eps_range[0]) / (eps_range[1] - eps_range[0])
            
            # 对于高维数据，倾向于更大的eps值
            # 对于低维数据，倾向于更小的eps值
            target_relative_pos = 0.5 + 0.3 * dim_adjustment_factor
            
            # 平滑调整
            adjustment_strength = 0.4 * abs(dim_adjustment_factor)
            adjusted_relative_pos = eps_relative_pos * (1 - adjustment_strength) + target_relative_pos * adjustment_strength
            
            # 转换回实际eps值
            eps = eps_range[0] + adjusted_relative_pos * (eps_range[1] - eps_range[0])

            if verbose:
                print(f"[维度={d}] eps范围={eps_range}")
                print(f"[估计] raw_eps={raw_eps:.4f}, quantile={quantile_eps:.4f}, elbow={elbow_eps:.4f}")
                print(f"[结果] → eps={eps:.4f}")

        except Exception as e:
            print(f"[警告] eps估算失败，使用维度默认值: {e}")
            eps = fallback_eps(eps_range)

    # 使用平滑函数确保eps在合理范围内
    eps = 0.01 + 0.99 / (1 + np.exp(-10 * (eps - 0.5)))  # 平滑限制在0.01-1.0范围

    if return_all:
        return round(eps, 4), k, t, round(quantile_eps or -1, 4), round(elbow_eps or -1, 4)
    else:
        return round(eps, 4), k, t

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 尝试导入kneed库用于拐点检测
try:
    from kneed import KneeLocator
    HAS_KNEED = True
except ImportError:
    HAS_KNEED = False
    print("Warning: kneed库未安装，使用备选拐点检测方法")

def simplified_adaptive_dbscan_params(n, d, 
                                    X=None,
                                    base_k=10, max_k=30,
                                    base_t=10, max_t=20,
                                    eps_quantile=0.6,
                                    eps_range=None,
                                    sample_ratio=0.3,  # 新增：采样比例
                                    standardize_data=True,
                                    return_all=False,
                                    verbose=False):
    """
   
    
    时间复杂度优化策略：
    1. 数据采样：仅使用部分数据计算k-距离
    2. 减少k值候选：仅测试2-3个关键k值
    3. 简化拐点检测：使用高效的梯度方法
    4. 跳过聚类评估：直接基于k-距离分布选择参数
    
    参数：
        n (int): 数据量
        d (int): 数据维度
        X (np.ndarray): 数据矩阵
        sample_ratio (float): 采样比例，减少计算量
        其他参数同原函数
    
    返回：
        eps, k, t，或详细信息字典
    """
    
    # 1. 快速计算基础参数
    logn = np.log(max(n, 10))
    
    # k值计算（min_samples）- 稳定在8-12范围内
    def calculate_stable_k(n, d, base_k=10):
        """计算稳定的k值，确保在8-12范围内"""
        # 基于数据量的调整（非常温和）
        if n < 100:
            size_factor = 0.8  # 小数据集稍微减小
        elif n < 1000:
            size_factor = 1.0  # 中等数据集保持基础值
        elif n < 10000:
            size_factor = 1.1  # 大数据集轻微增加
        else:
            size_factor = 1.2  # 超大数据集适度增加
        
        # 基于维度的调整（非常温和）
        if d <= 5:
            dim_factor = 0.9
        elif d <= 10:
            dim_factor = 1.0
        elif d <= 20:
            dim_factor = 1.05
        else:
            dim_factor = 1.1  # 高维数据轻微增加
        
        # 计算k值
        k = base_k * size_factor * dim_factor
        
        # 强制限制在8-12范围内
        k = max(8, min(12, int(round(k))))
        
        return k
    
    k_base = calculate_stable_k(n, d, base_k)
    
    # t值计算（哈希函数数量）
    t = int(min(base_t + (logn - np.log(10000)) / np.log(2), max_t))
    
    # 2. 高维数据eps范围优化（更倾向于0.75）
    def get_dimensional_eps_range(dim):
        """根据维度计算eps范围，高维数据更倾向于0.75"""
        if dim <= 5:
            return (0.2, 0.3)
        elif dim <= 10:
            return (0.4, 0.5)
        elif dim <= 20:
            return (0.65, 0.75)
        else:
            # 高维数据：eps范围更集中在0.75附近
            return (0.72, 0.78)
    
    # 3. 确定eps范围
    if eps_range is None:
        eps_range = get_dimensional_eps_range(d)
    
    # 4. 快速eps估计
    def fast_eps_estimation(X_data, k_candidates, eps_range, sample_ratio=0.3):
        """使用采样和简化方法快速估计eps"""
        
        # 数据采样以减少计算量
        n_samples = X_data.shape[0]
        if n_samples > 1000:
            sample_size = max(1000, int(n_samples * sample_ratio))
            indices = np.random.choice(n_samples, sample_size, replace=False)
            X_sampled = X_data[indices]
        else:
            X_sampled = X_data
        
        best_eps = None
        best_k = k_base
        
        # 只测试稳定的k值候选（都在8-12范围内）
        key_k_candidates = [
            max(8, min(12, int(1.2 * np.log(n)))),  # 基于数据量，限制在8-12
            max(8, min(12, int(d * 0.3 + 8))),      # 基于维度，限制在8-12
            k_base  # 基础计算（已经在8-12范围内）
        ]
        
        # 去重并限制候选数量
        k_candidates = sorted(set(key_k_candidates))[:3]
        
        for k in k_candidates:
            try:
                if k >= X_sampled.shape[0]:
                    k = max(8, min(12, X_sampled.shape[0] - 1))
                
                # 计算k-距离（仅在采样数据上）
                nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean')
                nbrs.fit(X_sampled)
                distances, _ = nbrs.kneighbors(X_sampled)
                k_dists = distances[:, k-1]
                k_dists = k_dists[np.isfinite(k_dists)]
                
                if len(k_dists) == 0:
                    continue
                
                k_dists_sorted = np.sort(k_dists)
                
                # 简化的拐点检测：仅使用梯度方法
                if len(k_dists_sorted) > 5:
                    # 计算梯度
                    gradients = np.gradient(k_dists_sorted)
                    elbow_index = np.argmax(gradients)
                    eps_gradient = k_dists_sorted[elbow_index]
                    
                    # 分位数方法
                    eps_quantile_val = np.quantile(k_dists_sorted, eps_quantile)
                    
                    # 简单平均
                    raw_eps = (eps_gradient + eps_quantile_val) / 2
                else:
                    raw_eps = np.median(k_dists_sorted)
                
                # 缩放到目标范围
                data_min, data_max = np.min(k_dists_sorted), np.max(k_dists_sorted)
                if data_max - data_min > 1e-6:
                    scaled_eps = eps_range[0] + (raw_eps - data_min) / (data_max - data_min) * (eps_range[1] - eps_range[0])
                else:
                    scaled_eps = (eps_range[0] + eps_range[1]) / 2
                
                # 高维数据特别处理：更倾向于0.75
                if d > 20:
                    # 对于高维数据，强制向0.75靠近
                    target_eps = 0.75
                    weight = min(0.8, (d - 20) / 50)  # 维度越高权重越大
                    scaled_eps = scaled_eps * (1 - weight) + target_eps * weight
                elif d > 10:
                    # 中高维数据适度调整
                    range_center = (eps_range[0] + eps_range[1]) / 2
                    if range_center > 0.6:  # 当范围中心已经较高时
                        bias_factor = 0.8 + 0.2 * (d - 10) / 10  # 向范围上限偏移
                        scaled_eps = eps_range[0] + bias_factor * (eps_range[1] - eps_range[0])
                
                # 确保在范围内
                scaled_eps = np.clip(scaled_eps, eps_range[0], eps_range[1])
                
                # 选择最合适的结果（优先选择接近k_base的值）
                if abs(k - k_base) <= abs(best_k - k_base) or best_eps is None:
                    best_eps = scaled_eps
                    best_k = k
                
                if verbose:
                    print(f"k={k}: eps={scaled_eps:.4f} (原始={raw_eps:.4f})")
                
            except Exception as e:
                if verbose:
                    print(f"k={k}: 计算失败 - {e}")
                continue
        
        return best_eps, best_k
    
    # 5. 主要优化流程
    eps, k = k_base, k_base
    
    if X is not None and X.shape[0] >= 10:
        try:
            # 数据预处理
            if standardize_data:
                scaler = StandardScaler()
                X_processed = scaler.fit_transform(X)
            else:
                X_processed = X
            
            if verbose:
                print(f"数据维度: {d}, 数据量: {n}")
                print(f"稳定k值范围: 10-20, 计算得到: {k_base}")
                print(f"eps目标范围: {eps_range}")
                print(f"采样比例: {sample_ratio}")
                print("-" * 40)
            
            # 快速估计最优参数
            estimated_eps, estimated_k = fast_eps_estimation(
                X_processed, [k_base], eps_range, sample_ratio
            )
            
            if estimated_eps is not None:
                eps = estimated_eps
                k = estimated_k
            else:
                # 回退策略
                eps = (eps_range[0] + eps_range[1]) / 2
                # 高维数据特别处理
                if d > 20:
                    eps = 0.75
                elif d > 10:
                    eps = min(0.7, eps_range[1])
                k = k_base  # 保持稳定的k值
                
        except Exception as e:
            if verbose:
                print(f"优化过程失败，使用默认参数: {e}")
            eps = (eps_range[0] + eps_range[1]) / 2
            # 高维数据回退策略
            if d > 20:
                eps = 0.75
            elif d > 10:
                eps = min(0.7, eps_range[1])
            k = k_base  # 保持稳定的k值
    
    else:
        # 无数据或数据量不足时的快速估计
        eps = (eps_range[0] + eps_range[1]) / 2
        # 高维数据处理
        if d > 20:
            eps = 0.75
        elif d > 10:
            eps = min(0.7, eps_range[1])
        k = k_base  # 保持稳定的k值
        
        if verbose:
            print("数据不足，使用维度估计参数")
    
    # 6. 最终k值验证（确保在8-12范围内）
    k = max(8, min(12, k))
    
    # 7. 最终结果处理
    eps = round(eps, 4)
    
    if verbose:
        print("=" * 40)
        print(f"最终参数: eps={eps}, k={k}, t={t}")
        print(f"k值稳定范围: [10, 20], 实际值: {k}")
        print(f"eps范围: {eps_range}")
        if d > 20:
            print(f"高维数据({d}维)，eps优化至0.75附近")
    
    if return_all:
        return {
            'eps': eps,
            'k': k,
            't': t,
            'eps_range': eps_range,
            'dimension': d,
            'high_dim_optimized': d > 20,
            'sample_ratio': sample_ratio if X is not None else None,
            'k_stable_range': [8, 12],
            'k_calculation_method': 'stable_bounded'
        }
    else:
        return eps, k, t    
'''
def enhanced_adaptive_dbscan_params(n, d, 
                                   X=None,
                                   base_k=10, max_k=30,
                                   base_t=10, max_t=20,
                                   eps_quantile=0.6,
                                   eps_range=None,
                                   use_optimization=True,
                                   standardize_data=True,
                                   return_all=False,
                                   verbose=False):
    """
    增强版自适应DBSCAN参数优化，结合多种方法确定最优eps、k、t值
    
    参数：
        n (int): 数据量
        d (int): 数据维度
        X (np.ndarray): 数据矩阵
        base_k, max_k: min_samples 的估计范围
        base_t, max_t: 哈希函数数量
        eps_quantile (float): quantile 比例
        eps_range (tuple or None): eps范围，None时根据维度自动设置
        use_optimization (bool): 是否使用优化方法
        standardize_data (bool): 是否标准化数据
        return_all (bool): 是否返回详细信息
        verbose (bool): 是否显示详细过程
    
    返回：
        eps, k, t，或详细信息字典
    """
    
    # 1. 计算基础参数
    logn = np.log(n)
    
    # k值计算（min_samples）
    k_base = int(min(base_k + 0.6 * (logn - np.log(10000)), max_k))
    
    # t值计算（哈希函数数量）
    t = int(min(base_t + (logn - np.log(10000)) / np.log(2), max_t))
    
    # 2. 平滑的维度自适应eps范围
    def get_dimensional_eps_range(dim):
        """根据维度平滑计算eps范围"""
        low_dim_min, low_dim_max = 0.2, 0.3
        high_dim_min, high_dim_max = 0.7, 0.75
        
        # 使用sigmoid函数实现平滑过渡
        transition_factor = 1 / (1 + np.exp(-0.15 * (dim - 15)))
        
        eps_min = low_dim_min + (high_dim_min - low_dim_min) * transition_factor
        eps_max = low_dim_max + (high_dim_max - low_dim_max) * transition_factor
        
        return (eps_min, eps_max)
    
    # 3. 确定eps范围
    if eps_range is None:
        eps_range = get_dimensional_eps_range(d)
    
    # 4. 多种minPts估计方法
    def estimate_minpts_methods(n_samples, n_features):
        """多种minPts估计方法"""
        methods = {
            'dimension_rule': max(4, 2 * n_features),
            'log_rule': max(4, int(np.log(n_samples))),
            'base_calculation': k_base,
            'adaptive': int((max(4, 2 * n_features) + max(4, int(np.log(n_samples)))) / 2)
        }
        return methods
    
    # 5. 高级拐点检测方法
    def find_optimal_eps_advanced(X_data, k_candidates, eps_range, verbose=False):
        """使用多种方法找到最优eps"""
        best_eps = None
        best_score = -1
        best_k = k_base
        
        results = {}
        
        for k in k_candidates:
            try:
                # 计算k-距离
                nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(X_data)
                distances, _ = nbrs.kneighbors(X_data)
                k_dists = distances[:, k-1]
                k_dists = k_dists[np.isfinite(k_dists)]
                k_dists_sorted = np.sort(k_dists)
                
                if len(k_dists_sorted) == 0:
                    continue
                
                # 方法1: Kneed库检测拐点
                eps_kneed = None
                if HAS_KNEED:
                    try:
                        kl = KneeLocator(range(len(k_dists_sorted)), k_dists_sorted,
                                       curve='convex', direction='increasing', S=1.0)
                        if kl.knee is not None:
                            eps_kneed = k_dists_sorted[kl.knee]
                    except:
                        pass
                
                # 方法2: 梯度最大值
                gradients = np.gradient(k_dists_sorted)
                elbow_index = np.argmax(gradients)
                eps_gradient = k_dists_sorted[elbow_index]
                
                # 方法3: 二阶导数
                if len(k_dists_sorted) > 3:
                    second_deriv = np.gradient(gradients)
                    curvature_index = np.argmax(second_deriv)
                    eps_curvature = k_dists_sorted[curvature_index]
                else:
                    eps_curvature = eps_gradient
                
                # 方法4: 分位数方法
                eps_quantile_val = np.quantile(k_dists_sorted, eps_quantile)                
                # 综合多种方法
                eps_candidates = [eps for eps in [eps_kneed, eps_gradient, eps_curvature, eps_quantile_val] if eps is not None]
                
                if not eps_candidates:
                    continue
                
                # 选择中位数作为原始eps
                raw_eps = np.median(eps_candidates)
                
                # 缩放到目标范围
                data_min, data_max = np.min(k_dists_sorted), np.max(k_dists_sorted)
                if data_max - data_min > 1e-6:
                    scaled_eps = eps_range[0] + (raw_eps - data_min) / (data_max - data_min) * (eps_range[1] - eps_range[0])
                else:
                    scaled_eps = (eps_range[0] + eps_range[1]) / 2
                
                # 维度自适应微调
                dim_factor = np.tanh((d - 6.5) / 8)
                target_pos = 0.5 + 0.3 * dim_factor
                current_pos = (scaled_eps - eps_range[0]) / (eps_range[1] - eps_range[0])
                adjustment = 0.4 * abs(dim_factor)
                final_pos = current_pos * (1 - adjustment) + target_pos * adjustment
                final_eps = eps_range[0] + final_pos * (eps_range[1] - eps_range[0])
                
                # 平滑边界限制
                final_eps = 0.01 + 0.99 / (1 + np.exp(-10 * (final_eps - 0.5)))
                
                # 评估聚类效果
                try:
                    dbscan = DBSCAN(eps=final_eps, min_samples=k)
                    labels = dbscan.fit_predict(X_data)
                    
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    n_noise = list(labels).count(-1)
                    noise_ratio = n_noise / len(labels)
                    
                    # 综合评分：平衡聚类数和噪声比例
                    if n_clusters > 0:
                        cluster_score = min(n_clusters / max(1, np.sqrt(d)), 1.0)  # 维度惩罚
                        noise_penalty = max(0, 1 - 2 * noise_ratio)  # 噪声惩罚
                        consistency_bonus = 1 / (1 + abs(k - k_base) / k_base)  # k值一致性奖励
                        
                        score = cluster_score * noise_penalty * consistency_bonus
                    else:
                        score = 0
                    
                    results[k] = {
                        'eps': final_eps,
                        'raw_eps': raw_eps,
                        'scaled_eps': scaled_eps,
                        'n_clusters': n_clusters,
                        'n_noise': n_noise,
                        'noise_ratio': noise_ratio,
                        'score': score,
                        'methods': {
                            'kneed': eps_kneed,
                            'gradient': eps_gradient,
                            'curvature': eps_curvature,
                            'quantile': eps_quantile_val
                        }
                    }
                    
                    if score > best_score:
                        best_score = score
                        best_eps = final_eps
                        best_k = k
                        
                    if verbose:
                        print(f"k={k}: eps={final_eps:.4f}, 聚类={n_clusters}, 噪声={n_noise}({noise_ratio:.1%}), 得分={score:.3f}")
                
                except Exception as e:
                    if verbose:
                        print(f"k={k}: 聚类评估失败 - {e}")
                    continue
                    
            except Exception as e:
                if verbose:
                    print(f"k={k}: k-距离计算失败 - {e}")
                continue
        
        return best_eps, best_k, results
    
    # 6. 主要优化流程
    eps, k = k_base, k_base
    optimization_results = {}
    
    if X is not None and X.shape[0] >= k_base and use_optimization:
        try:
            # 数据预处理
            if standardize_data:
                scaler = StandardScaler()
                X_processed = scaler.fit_transform(X)
            else:
                X_processed = X
            
            # 估计minPts候选值
            minpts_methods = estimate_minpts_methods(n, d)
            
            # 生成k值候选范围
            k_candidates = list(set([
                minpts_methods['dimension_rule'],
                minpts_methods['log_rule'],
                minpts_methods['base_calculation'],
                minpts_methods['adaptive']
            ]))
            
            # 扩展候选范围
            k_min = max(3, min(k_candidates) - 1)
            k_max = min(max_k, max(k_candidates) + 2)
            k_candidates = sorted(set(k_candidates + list(range(k_min, k_max + 1))))
            
            if verbose:
                print(f"数据维度: {d}, 数据量: {n}")
                print(f"eps目标范围: {eps_range}")
                print(f"minPts候选: {minpts_methods}")
                print(f"k值候选: {k_candidates}")
                print("-" * 50)
            
            # 寻找最优参数
            eps, k, optimization_results = find_optimal_eps_advanced(
                X_processed, k_candidates, eps_range, verbose
            )
            
            if eps is None:
                # 回退到基础方法
                eps = (eps_range[0] + eps_range[1]) / 2
                k = k_base
                if verbose:
                    print("优化失败，使用默认参数")
            
        except Exception as e:
            if verbose:
                print(f"优化过程失败，使用默认参数: {e}")
            eps = (eps_range[0] + eps_range[1]) / 2
            k = k_base
    
    elif X is None:
        # 无数据时使用估计方法
        eps = (eps_range[0] + eps_range[1]) / 2
        k = k_base
        if verbose:
            print("无数据，使用维度估计参数")
    
    else:
        # 数据量不足时的备选方法
        if X is not None and X.shape[0] < k_base:
            k = max(3, min(X.shape[0] // 2, k_base))
        eps = (eps_range[0] + eps_range[1]) / 2
        if verbose:
            print("数据量不足，使用简化参数")
    
    # 7. 最终结果
    eps = round(eps, 4)
    
    if verbose:
        print("=" * 50)
        print(f"最终参数: eps={eps}, k={k}, t={t}")
        print(f"eps范围: {eps_range}")
        if optimization_results:
            best_result = optimization_results.get(k, {})
            if best_result:
                print(f"预期聚类数: {best_result.get('n_clusters', 'N/A')}")
                print(f"预期噪声比例: {best_result.get('noise_ratio', 'N/A'):.1%}")
    
    if return_all:
        return {
            'eps': eps,
            'k': k,
            't': t,
            'eps_range': eps_range,
            'optimization_results': optimization_results,
            'minpts_methods': estimate_minpts_methods(n, d) if n and d else {},
            'data_processed': X is not None and use_optimization
        }
    else:
        return eps, k, t

'''
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# Placeholder for KneeLocator, assuming it might not always be available
try:
    from kneed import KneeLocator
    HAS_KNEED = True
except ImportError:
    HAS_KNEED = False

def enhanced_adaptive_dbscan_params(n, d,
                                   X=None,
                                   base_k=8, max_k=15,
                                   base_t=10, max_t=20,
                                   eps_quantile=0.6,
                                   eps_range=None,
                                   use_optimization=True,
                                   standardize_data=True,
                                   return_all=False,
                                   verbose=False,
                                   is_geospatial=None):
    """
    增强版自适应DBSCAN参数优化，特别优化了对地理坐标数据的处理

    参数：
        n (int): 数据量
        d (int): 数据维度
        X (np.ndarray): 数据矩阵
        base_k, max_k: min_samples 的估计范围 (修改为5-15)
        base_t, max_t: 哈希函数数量
        eps_quantile (float): quantile 比例
        eps_range (tuple or None): eps范围，None时根据维度自动设置
        use_optimization (bool): 是否使用优化方法
        standardize_data (bool): 是否标准化数据
        return_all (bool): 是否返回详细信息
        verbose (bool): 是否显示详细过程
        is_geospatial (bool or None): 是否为地理空间数据，None时自动检测

    返回：
        eps, k, t，或详细信息字典
    """

    # 1. 地理空间数据检测
    def detect_geospatial_data(X_data):
        """检测是否为地理空间数据"""
        if X_data is None or X_data.shape[1] != 2:
            return False

        # 检查数据范围是否像经纬度坐标（归一化后通常在0-1范围）
        min_vals = np.min(X_data, axis=0)
        max_vals = np.max(X_data, axis=0)

        # 检查是否在合理的归一化范围内
        if (np.all(min_vals >= -0.1) and np.all(max_vals <= 1.1) and
            np.all(max_vals - min_vals > 0.1)):
            return True

        # 检查数据分布是否相对均匀（地理数据特征）
        std_vals = np.std(X_data, axis=0)
        if np.all(std_vals > 0.05) and np.all(std_vals < 0.5):
            return True

        return False

    # 2. 自动检测地理空间数据
    if is_geospatial is None and X is not None:
        is_geospatial = detect_geospatial_data(X)
        if verbose and is_geospatial:
            print("检测到地理空间数据，启用专用优化")

    # 3. 计算基础参数（修改：限制k值范围5-15）
    logn = np.log(n)

    # 针对地理数据调整k值计算
    if is_geospatial:
        # 地理数据通常需要更小的min_samples，限制在5-12范围
        k_base = int(min(max(5, base_k * 0.8 + 0.2 * (logn - np.log(10000))), 12))
    else:
        # 通用数据k值计算，限制在5-15范围
        k_base = int(min(max(5, base_k + 0.4 * (logn - np.log(10000))), max_k))

    # t值计算（哈希函数数量）
    t = int(min(base_t + (logn - np.log(10000)) / np.log(2), max_t))

    # 4. 地理空间数据专用eps范围计算
    def get_geospatial_eps_range(X_data, n_samples):
        """为地理空间数据计算更精确的eps范围"""
        if X_data is None or X_data.shape[1] != 2:
            return (0.01, 0.1)

        # 计算平均最近邻距离
        try:
            nbrs = NearestNeighbors(n_neighbors=min(5, n_samples-1), metric='euclidean').fit(X_data)
            distances, _ = nbrs.kneighbors(X_data)
            mean_nn_dist = np.mean(distances[:, 1])  # 最近邻距离的平均值

            # 基于数据密度调整eps范围
            data_density = n_samples / (np.ptp(X_data[:, 0]) * np.ptp(X_data[:, 1]))
            density_factor = np.log10(max(1, data_density))

            # 根据数据量调整
            size_factor = np.log10(n_samples / 1000) if n_samples > 1000 else 0

            # 计算eps范围
            base_eps = mean_nn_dist * 1.5
            eps_min = max(0.001, base_eps * (0.5 - 0.1 * density_factor))
            eps_max = min(0.5, base_eps * (3.0 + 0.2 * size_factor))

            # 确保范围合理
            if eps_max <= eps_min:
                eps_max = eps_min * 2

            return (eps_min, eps_max)

        except Exception as e:
            if verbose:
                print(f"地理空间eps范围计算失败，使用默认值: {e}")
            return (0.01, 0.1)

    # 5. 平滑的维度自适应eps范围（非地理数据）
    def get_dimensional_eps_range(dim):
        """根据维度平滑计算eps范围"""
        # Modified: For d >= 5, make eps stable around 0.75
        if dim >= 5:
            return (0.7, 0.8) # Keep eps in a tight range around 0.75
        else:
            low_dim_min, low_dim_max = 0.2, 0.3
            high_dim_min, high_dim_max = 0.7, 0.75

            # 使用sigmoid函数实现平滑过渡
            transition_factor = 1 / (1 + np.exp(-0.15 * (dim - 15)))

            eps_min = low_dim_min + (high_dim_min - low_dim_min) * transition_factor
            eps_max = low_dim_max + (high_dim_max - low_dim_max) * transition_factor

            return (eps_min, eps_max)

    # 6. 确定eps范围
    if eps_range is None:
        if is_geospatial and X is not None:
            eps_range = get_geospatial_eps_range(X, n)
        else:
            eps_range = get_dimensional_eps_range(d)

    # 7. 地理空间数据专用距离采样
    def sample_for_geospatial(X_data, max_samples=5000):
        """为地理空间数据进行智能采样"""
        if len(X_data) <= max_samples:
            return X_data

        # 快速空间均匀采样
        try:
            # 使用更简单的采样策略
            np.random.seed(42)  # 确保可重复性

            # 方法1: 网格采样（快速版）
            # 修改网格大小计算，适应更大的采样数
            grid_size = min(int(np.sqrt(max_samples * 0.8)), 100)  # 增加网格大小上限
            x_min, x_max = X_data[:, 0].min(), X_data[:, 0].max()
            y_min, y_max = X_data[:, 1].min(), X_data[:, 1].max()

            # 防止除零
            if x_max - x_min < 1e-10:
                x_max = x_min + 1e-10
            if y_max - y_min < 1e-10:
                y_max = y_min + 1e-10

            x_bins = np.linspace(x_min, x_max, grid_size + 1)
            y_bins = np.linspace(y_min, y_max, grid_size + 1)

            sampled_indices = []
            target_per_cell = max(1, max_samples // (grid_size * grid_size))

            for i in range(grid_size):
                for j in range(grid_size):
                    # 在每个网格中随机选择点
                    mask = ((X_data[:, 0] >= x_bins[i]) & (X_data[:, 0] < x_bins[i+1]) &
                        (X_data[:, 1] >= y_bins[j]) & (X_data[:, 1] < y_bins[j+1]))
                    indices = np.where(mask)[0]

                    if len(indices) > 0:
                        n_select = min(target_per_cell, len(indices))
                        selected = np.random.choice(indices, n_select, replace=False)
                        sampled_indices.extend(selected)

                        # 如果已经采样足够多的点，提前退出
                        if len(sampled_indices) >= max_samples:
                            break
                if len(sampled_indices) >= max_samples:
                    break

            if len(sampled_indices) > 0:
                # 如果采样过多，随机选择
                if len(sampled_indices) > max_samples:
                    sampled_indices = np.random.choice(sampled_indices, max_samples, replace=False)
                return X_data[sampled_indices]
            else:
                # 回退到纯随机采样
                return X_data[np.random.choice(len(X_data), max_samples, replace=False)]

        except Exception as e:
            # 如果出错，使用简单的随机采样
            return X_data[np.random.choice(len(X_data), max_samples, replace=False)]

    # 8. 优化的拐点检测方法
    def find_optimal_eps_advanced(X_data, k_candidates, eps_range, verbose=False):
        """使用多种方法找到最优eps，特别优化地理空间数据"""
        best_eps = None
        best_score = -1
        best_k = k_base
        
        # Store results for all k candidates to re-evaluate if all scores are 0
        all_results = {}

        results = {}

        # 修改这里: 对大数据集进行更积极的采样
        max_samples = 50000
        if len(X_data) > max_samples:
            if is_geospatial:
                X_sample = sample_for_geospatial(X_data, max_samples)
            else:
                # 随机采样
                indices = np.random.choice(len(X_data), max_samples, replace=False)
                X_sample = X_data[indices]
            if verbose:
                print(f"数据采样: {len(X_data)} -> {len(X_sample)}")
        else:
            X_sample = X_data

        # 限制k值候选数量，避免过度计算
        k_candidates = k_candidates[:10]

        for idx, k in enumerate(k_candidates):
            try:
                if verbose:
                    print(f"处理k={k} ({idx+1}/{len(k_candidates)})")

                # 计算k-距离
                k_actual = min(k, len(X_sample) - 1)
                if k_actual < 5:
                    continue

                # 使用更高效的算法
                nbrs = NearestNeighbors(
                    n_neighbors=k_actual,
                    metric='euclidean',
                    algorithm='ball_tree' if is_geospatial else 'auto',
                    n_jobs=1
                ).fit(X_sample)

                distances, _ = nbrs.kneighbors(X_sample)
                k_dists = distances[:, k_actual-1]
                k_dists = k_dists[np.isfinite(k_dists)]

                if len(k_dists) < 10:
                    continue

                k_dists_sorted = np.sort(k_dists)

                if len(k_dists_sorted) == 0:
                    continue

                # 快速eps估算，避免复杂计算
                if is_geospatial:
                    # 地理空间数据专用的快速eps计算
                    eps_candidates = []

                    # 方法1: 简单分位数
                    density_quantile = 0.7 if n > 50000 else 0.6
                    eps_density = np.quantile(k_dists_sorted, density_quantile)
                    eps_candidates.append(eps_density)

                    # 方法2: 基于中位数的调整
                    median_dist = np.median(k_dists_sorted)
                    eps_median = median_dist * 1.2
                    eps_candidates.append(eps_median)

                    # 方法3: 基于四分位数
                    q75 = np.percentile(k_dists_sorted, 75)
                    eps_candidates.append(q75)

                else:
                    # 原有方法的简化版本
                    eps_candidates = []

                    # 使用kneed库（如果可用）
                    if HAS_KNEED and len(k_dists_sorted) > 20:
                        try:
                            # 对长距离列表进行采样以加速kneed计算
                            sample_size = min(1000, len(k_dists_sorted))
                            step = len(k_dists_sorted) // sample_size
                            sampled_dists = k_dists_sorted[::step]
                            sampled_indices = list(range(0, len(k_dists_sorted), step))

                            kl = KneeLocator(sampled_indices, sampled_dists,
                                           curve='convex', direction='increasing', S=1.0)
                            if kl.knee is not None:
                                eps_kneed = sampled_dists[kl.knee // step] if kl.knee // step < len(sampled_dists) else sampled_dists[-1]
                                eps_candidates.append(eps_kneed)
                        except:
                            pass

                    # 简化的梯度方法
                    if len(k_dists_sorted) > 10:
                        # 使用子采样计算梯度
                        step = max(1, len(k_dists_sorted) // 200)
                        sub_dists = k_dists_sorted[::step]
                        gradients = np.gradient(sub_dists)
                        elbow_index = np.argmax(gradients)
                        eps_gradient = sub_dists[elbow_index]
                        eps_candidates.append(eps_gradient)

                    # 分位数方法
                    eps_quantile_val = np.quantile(k_dists_sorted, eps_quantile)
                    eps_candidates.append(eps_quantile_val)

                if not eps_candidates:
                    continue

                # 快速eps计算
                raw_eps = np.median(eps_candidates)

                # 简化的缩放
                data_min, data_max = np.min(k_dists_sorted), np.max(k_dists_sorted)
                if data_max - data_min > 1e-6:
                    if is_geospatial:
                        # 地理空间数据的保守缩放
                        relative_pos = (raw_eps - data_min) / (data_max - data_min)
                        final_eps = eps_range[0] + relative_pos * (eps_range[1] - eps_range[0]) * 0.8
                    else:
                        # 通用数据的标准缩放
                        final_eps = eps_range[0] + (raw_eps - data_min) / (data_max - data_min) * (eps_range[1] - eps_range[0])
                else:
                    final_eps = (eps_range[0] + eps_range[1]) / 2

                # 确保eps在合理范围内
                final_eps = np.clip(final_eps, eps_range[0], eps_range[1])

                # 快速聚类评估（使用更小的样本）
                try:
                    # 对于评估，使用更小的样本
                    eval_sample_size = min(2000, len(X_sample))
                    if len(X_sample) > eval_sample_size:
                        eval_indices = np.random.choice(len(X_sample), eval_sample_size, replace=False)
                        X_eval = X_sample[eval_indices]
                    else:
                        X_eval = X_sample

                    dbscan = DBSCAN(eps=final_eps, min_samples=k, n_jobs=1)
                    labels = dbscan.fit_predict(X_eval)

                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    n_noise = list(labels).count(-1)
                    noise_ratio = n_noise / len(labels) if len(labels) > 0 else 1.0

                    # 快速评分
                    if n_clusters > 0:
                        if is_geospatial:
                            # 地理数据评分
                            expected_clusters = max(2, int(np.sqrt(n / 2000)))
                            cluster_score = 1 / (1 + abs(n_clusters - expected_clusters) / max(1, expected_clusters))
                            noise_penalty = max(0.1, 1 - 1.5 * noise_ratio)
                            score = cluster_score * noise_penalty
                        else:
                            # 通用数据评分
                            cluster_score = min(n_clusters / max(1, np.sqrt(d)), 1.0)
                            noise_penalty = max(0, 1 - 2 * noise_ratio)
                            score = cluster_score * noise_penalty
                    else:
                        score = 0

                    all_results[k] = {  # Store results for all k candidates
                        'eps': final_eps,
                        'raw_eps': raw_eps,
                        'n_clusters': n_clusters,
                        'n_noise': n_noise,
                        'noise_ratio': noise_ratio,
                        'score': score
                    }

                    if score > best_score:
                        best_score = score
                        best_eps = final_eps
                        best_k = k

                    if verbose:
                        print(f"k={k}: eps={final_eps:.4f}, 聚类={n_clusters}, 噪声={n_noise}({noise_ratio:.1%}), 得分={score:.3f}")

                except Exception as e:
                    if verbose:
                        print(f"k={k}: 聚类评估失败 - {e}")
                    continue

            except Exception as e:
                if verbose:
                    print(f"k={k}: k-距离计算失败 - {e}")
                continue
        
        # New logic: If all scores are 0, select k closest to 10
        if best_score == 0 and all_results:
            min_diff = float('inf')
            for k_val, res in all_results.items():
                if res['score'] == 0:
                    diff = abs(k_val - 10)
                    if diff < min_diff:
                        min_diff = diff
                        best_k = k_val
                        best_eps = res['eps']
                        if verbose:
                            print(f"所有得分均为0，选择k={best_k} (距离10最近)")

        return best_eps, best_k, all_results # Return all_results instead of results

    # 9. 多种minPts估计方法（修改：限制k值范围5-15）
    def estimate_minpts_methods(n_samples, n_features):
        """多种minPts估计方法"""
        if is_geospatial:
            # 地理空间数据的专用估计，限制在5-12范围
            methods = {
                'geospatial_small': max(5, min(12, int(np.log10(n_samples) * 0.8))),
                'geospatial_density': max(5, min(12, int(np.log10(n_samples * 0.1) * 0.9))),
                'base_calculation': max(5, min(12, k_base)),
                'adaptive': max(5, min(12, int((max(5, int(np.log10(n_samples) * 0.8)) + k_base) / 2)))
            }
        else:
            # 通用数据的估计，限制在5-15范围
            methods = {
                'dimension_rule': max(5, min(15, 2 * n_features)),
                'log_rule': max(5, min(15, int(np.log(n_samples)))),
                'base_calculation': max(5, min(15, k_base)),
                'adaptive': max(5, min(15, int((max(5, min(15, 2 * n_features)) + max(5, min(15, int(np.log(n_samples))))) / 2)))
            }
        return methods

    # 10. 主要优化流程
    eps, k = k_base, k_base
    optimization_results = {}

    if X is not None and X.shape[0] >= k_base and use_optimization:
        try:
            # 数据预处理
            if standardize_data and not is_geospatial:
                # 地理空间数据通常已经归一化，不需要再次标准化
                scaler = StandardScaler()
                X_processed = scaler.fit_transform(X)
            else:
                X_processed = X

            # 估计minPts候选值
            minpts_methods = estimate_minpts_methods(n, d)

            # 生成k值候选范围（修改：限制在5-15范围）
            k_candidates = list(set([
                minpts_methods.get('dimension_rule', k_base),
                minpts_methods.get('log_rule', k_base),
                minpts_methods.get('geospatial_small', k_base),
                minpts_methods.get('geospatial_density', k_base),
                minpts_methods.get('base_calculation', k_base),
                minpts_methods.get('adaptive', k_base)
            ]))

            # 扩展候选范围，确保在5-15范围内
            k_min = max(5, min(k_candidates) - 1)
            k_max = min(15, max(k_candidates) + 2)
            k_candidates = sorted(set(k_candidates + list(range(k_min, k_max + 1))))

            # 确保所有k值都在5-15范围内
            k_candidates = [k_val for k_val in k_candidates if 5 <= k_val <= 15]

            if verbose:
                print(f"数据类型: {'地理空间数据' if is_geospatial else '通用数据'}")
                print(f"数据维度: {d}, 数据量: {n}")
                print(f"eps目标范围: {eps_range}")
                print(f"minPts候选: {minpts_methods}")
                print(f"k值候选: {k_candidates}")
                print("-" * 50)

            # 寻找最优参数
            eps, k, optimization_results = find_optimal_eps_advanced(
                X_processed, k_candidates, eps_range, verbose
            )

            if eps is None:
                # 回退到基础方法
                eps = (eps_range[0] + eps_range[1]) / 2
                k = max(5, min(15, k_base)) # 确保k在5-15范围内
                if verbose:
                    print("优化失败，使用默认参数")

        except Exception as e:
            if verbose:
                print(f"优化过程失败，使用默认参数: {e}")
            eps = (eps_range[0] + eps_range[1]) / 2
            k = max(5, min(15, k_base)) # 确保k在5-15范围内

    elif X is None:
        # 无数据时使用估计方法
        eps = (eps_range[0] + eps_range[1]) / 2
        k = max(5, min(15, k_base)) # 确保k在5-15范围内
        if verbose:
            print("无数据，使用维度估计参数")

    else:
        # 数据量不足时的备选方法
        if X is not None and X.shape[0] < k_base:
            k = max(5, min(X.shape[0] // 2, 15)) # 确保k在5-15范围内
        else:
            k = max(5, min(15, k_base)) # 确保k在5-15范围内
        eps = (eps_range[0] + eps_range[1]) / 2
        if verbose:
            print("数据量不足，使用简化参数")

    # 11. 最终结果（再次确保k值在范围内）
    k = max(5, min(15, k))
    eps = round(eps, 6) # 地理数据需要更高精度

    if verbose:
        print("=" * 50)
        print(f"最终参数: eps={eps}, k={k}, t={t}")
        print(f"eps范围: {eps_range}")
        print(f"k值范围: [5, 15]")
        if optimization_results:
            best_result = optimization_results.get(k, {})
            if best_result:
                print(f"预期聚类数: {best_result.get('n_clusters', 'N/A')}")
                print(f"预期噪声比例: {best_result.get('noise_ratio', 'N/A'):.1%}")

    if return_all:
        return {
            'eps': eps,
            'k': k,
            't': t,
            'eps_range': eps_range,
            'optimization_results': optimization_results,
            'minpts_methods': estimate_minpts_methods(n, d) if n and d else {},
            'data_processed': X is not None and use_optimization,
            'is_geospatial': is_geospatial,
            'k_range': [5, 15] # 添加k值范围信息
        }
    else:
        return eps, k, t
    
def dynamic_experiment(name: str, static_dataset, eps, min_samples_new, min_samples_sklearn, t, d, burnin,
                       cluster_by_cluster=False,
                       skip_sklearn=False,
                       skip_evaluation=False):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if cluster_by_cluster:
        name = name + "_by_cluster"
    results_filename = os.path.join(dir_path, f"results/{name}.csv")
    dynamic_dataset = alglab.dataset.DynamicPointCloudDataset.from_pointcloud(static_dataset, 1000,
                                                                              stream_by_cluster=cluster_by_cluster)
    #dynamic_dataset=static_dataset
    Ada_eps, Ada_min_samples_sklearn, Ada_t = enhanced_adaptive_dbscan_params(static_dataset.n, static_dataset.d,static_dataset.data, use_optimization=True, verbose=True)
    
    #Ada_eps, Ada_min_samples_sklearn, Ada_t = simplified_adaptive_dbscan_params(static_dataset.n, static_dataset.d,static_dataset.data,  verbose=True)
    


    #algorithms = [dynamic_dbscan_alg, fdbscan_dynamic_noncore, sklearn_dynamic_alg, Ada_dynamic_dbscan_alg]
    #algorithms = [dynamic_dbscan_alg, fdbscan_dynamic_naive, fdbscan_dynamic_noncore, sklearn_dynamic_alg, Ada_dynamic_dbscan_alg]
    algorithms = [Ada_dynamic_dbscan_alg]


    '''if not skip_sklearn:
        algorithms.append(sklearn_dynamic_alg)'''
    
    # 根据数据集是否有验证标签来决定使用哪些评估器
    evaluators = [alglab.evaluation.dataset_size]  # 总是包含数据集大小评估器
    
    

    if not skip_evaluation:
        # 检查数据集是否有验证标签
        if hasattr(static_dataset, 'gt_labels') and static_dataset.gt_labels is not None:
            evaluators.extend([alglab.evaluation.adjusted_rand_index, alglab.evaluation.normalised_mutual_info])
            print(f"数据集 {name} 有验证标签，将进行聚类质量评估")
        else:
            print(f"数据集 {name} 没有验证标签，跳过聚类质量评估")
    else:
        print(f"数据集 {name} 手动跳过聚类质量评估")
    
    experiments = alglab.experiment.ExperimentalSuite(
        algorithms,
        dynamic_dataset,
        results_filename,
        alg_fixed_params={
            'dynamic_dbscan': {'eps': eps, 'min_samples': min_samples_new, 't': t, 'd': d},
            'fdbscan_dynamic_naive': {'eps': eps, 'min_samples': min_samples_new, 't': t},
            'fdbscan_dynamic_noncore': {'eps': eps, 'min_samples': min_samples_new, 't': t, 'burnin': burnin},
            'sklearn_dynamic': {'eps': eps, 'min_samples': min_samples_sklearn},
            'Ada_dynamic_dbscan': {'eps': Ada_eps, 'min_samples': Ada_min_samples_sklearn, 't': Ada_t, 'd': d},
        },
        
        evaluators=evaluators,
        num_runs=1
    )
    
    results = experiments.run_all()
    
    
    # 确保figures目录存在
    figures_dir = os.path.join(dir_path, "figures")
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    
    # 生成时间对比图
    try:
        results.line_plot("n", "total_running_time_s", filename=f"figures/{name}_time_comparison.pdf")
        print(f"时间对比图已生成: figures/{name}_time_comparison.pdf")
    except Exception as e:
        print(f"生成时间对比图时出错: {e}")
    
    # 如果有验证标签才生成质量对比图
    if not skip_evaluation and hasattr(static_dataset, 'gt_labels') and static_dataset.gt_labels is not None:
        try:
            results.line_plot("n", "adjusted_rand_index", filename=f"figures/{name}_ari_comparison.pdf")
            results.line_plot("n", "normalised_mutual_info", filename=f"figures/{name}_nmi_comparison.pdf")
            print(f"质量对比图已生成: figures/{name}_ari_comparison.pdf 和 figures/{name}_nmi_comparison.pdf")
        except Exception as e:
            print(f"生成质量对比图时出错: {e}")
    
    # 为truck数据生成地理聚类结果可视化
    if name == 'truck':
        generate_geo_clustering_results(static_dataset, results, name, dir_path, dynamic_dataset)

    return results

def blobs_experiment():
    random.seed(2024)
    eps = 0.75
    t = 10
    k=10
    blobs_dataset = alglab.dataset.BlobsDataset(n=50000, d=5, k=10)
    dynamic_experiment('blobs', blobs_dataset, eps, k, 10, t, 5, 20,
                       cluster_by_cluster=False)
    ''' dynamic_experiment('blobs', blobs_dataset, 1, 50, 10, 5, 5, 20,
                       cluster_by_cluster=False)'''

def mnist_experiment():
    mnist_dataset = alglab.dataset.OpenMLDataset(name='mnist_784')
    mnist_dataset.apply_pca(20)
    mnist_dataset.apply_scaling()
    eps = 0.75
    t = 10
    dynamic_experiment('mnist', mnist_dataset, eps, 10, 10, t, mnist_dataset.d, 20,
                       cluster_by_cluster=False)

def fashion_mnist_experiment():
    mnist_dataset = alglab.dataset.OpenMLDataset(name='Fashion-MNIST')
    mnist_dataset.apply_pca(20)
    mnist_dataset.apply_scaling()
    eps = 0.75
    t = 10
    dynamic_experiment('fashion', mnist_dataset, eps, 10, 10, t, mnist_dataset.d, 20,
                       cluster_by_cluster=False)

def kdd_experiment():
    dataset = alglab.dataset.OpenMLDataset(name='KDDCup99')
    dataset.apply_pca(20)
    dataset.apply_scaling()
    eps = 0.75
    t = 10
    dynamic_experiment('KDDCup99', dataset, eps, 10, 10, t, dataset.d, 20,
                       cluster_by_cluster=False,
                       skip_sklearn=True)

def covertype_experiment():
    dataset = alglab.dataset.OpenMLDataset(name='covertype')
    dataset.apply_scaling()
    eps = 0.75
    t = 10
    dynamic_experiment('covertype', dataset, eps, 10, 10, t, dataset.d, 20,
                       cluster_by_cluster=False,
                       skip_sklearn=True)

def letters_experiment():
    letters_dataset = alglab.dataset.OpenMLDataset(name='letter')
    letters_dataset.apply_scaling()
    t = 10
    eps = 0.75
    dynamic_experiment('letter', letters_dataset, eps, 10, 10, t, letters_dataset.d, 1,
                       cluster_by_cluster=False)
    
def truck_experiment():
    base_path = "E:/dwd_truck"
    dates = ["2024-04-14"]
    truck_dataset = alglab.dataset_base.LocalTrajectoryDataset(base_folder=base_path, date_range=dates)
    
    print(f"加载的轨迹数据信息:")
    print(f"  - 数据点数量: {truck_dataset.n}")
    print(f"  - 数据维度: {truck_dataset.d}")
    print(f"  - 数据形状: {truck_dataset.data.shape}")
    truck_dataset.original_data = truck_dataset.data.copy()
    print(truck_dataset.data[:20])  # 打印前5行数据以检查加载情况
    
    t = 10
    eps = 0.75
    # 为truck实验跳过评估，因为没有验证标签
    result = dynamic_experiment('truck', truck_dataset, eps, 10, 10, t, truck_dataset.d, 1,
                               cluster_by_cluster=False,
                               skip_evaluation=True)  # 明确跳过评估
    
    return result

def plot_results():
    algorithm_names = {'dynamic_dbscan': r'\textsc{DyDBSCAN}',
                       'sklearn_dynamic': r'\textsc{Sklearn}',
                       'fdbscan_dynamic_naive': r'\textsc{EMZ}',
                       'fdbscan_dynamic_noncore': r'\textsc{EMZFixedCore}',
                       'Ada_dynamic_dbscan': r'\textsc{AdaDyDBSCAN}'}
    # Blobs experiment
    dir_path = os.path.dirname(os.path.realpath(__file__))
    results_filename = os.path.join(dir_path, "results/blobs.csv")
    results = alglab.results.Results(results_filename)
    results.line_plot("n", "total_running_time_s", filename="figures/blobs_time_comparison.pdf",
                      x_label="n",
                      y_label="Time (s)",
                      algorithm_names=algorithm_names)
    results.line_plot("n", "adjusted_rand_index", filename="figures/blobs_ari_comparison.pdf",
                      x_range=[0, 205000],
                      x_label="n",
                      y_label="ARI",
                      algorithm_names=algorithm_names,
                      show_legend=False)
    results_filename = os.path.join(dir_path, "results/blobs_by_cluster.csv")
    results = alglab.results.Results(results_filename)
    results.line_plot("n", "adjusted_rand_index", filename="figures/blobs_by_cluster_ari_comparison.pdf",
                      x_range=[18000, 205000],
                      x_label="n",
                      y_label="ARI",
                      algorithm_names=algorithm_names,
                      show_legend=False)

def format_table_number(stats, alg, best_val, lower_is_better=False):
    if alg not in stats:
        return '-'
    min_val = stats[alg][0] - stats[alg][1]
    max_val = stats[alg][0] + stats[alg][1]
    better = False
    if lower_is_better and min_val <= best_val:
        better = True
    elif not lower_is_better and max_val >= best_val:
        better = True

    if f"{stats[alg][0]: 0.2f}" == f"{best_val: 0.2f}" :
        better = True

    maybe_bold_start = "\\mathbf{" if better else ""
    maybe_bold_end = "}" if better else ""
    return "$" + maybe_bold_start + f"{stats[alg][0]: 0.2f}" + maybe_bold_end + "{\\scriptstyle \pm " + f"{stats[alg][1]: 0.3f}" + "}$"

def get_best_stat_from_stats(stats, algorithms, lower_is_better=False):
    best_val = None
    best_alg = None
    for alg in algorithms:
        if alg not in stats:
            continue
        stat = stats[alg]
        if best_val is None:
            if lower_is_better:
                best_val = stat[0] + stat[1]
            else:
                best_val = stat[0] - stat[1]
            best_alg = alg
        else:
            this_val = None
            if lower_is_better:
                this_val = stat[0] + stat[1]
            else:
                this_val = stat[0] - stat[1]
            if this_val < best_val and lower_is_better:
                best_val = this_val
                best_alg = alg
            elif this_val > best_val and not lower_is_better:
                best_val = this_val
                best_alg = alg
    return best_val

def create_numerical_results_table():
    algorithms = ['dynamic_dbscan', 'fdbscan_dynamic_naive', 'sklearn_dynamic', 'Ada_dynamic_dbscan']
    table = """\\begin{table}[]
    \\resizebox{\\columnwidth}{!}{
\\begin{tabular}{cccccc}
\\toprule
 & & \multicolumn{4}{c}{Algorithm} \\\\
 \\cmidrule{3-6}
 Dataset & Metric & \\textsc{DyDBSCAN} & \\textsc{EMZ} & \\textsc{Sklearn} & \\textsc{AdaDyDBSCAN} \\\\
 \\midrule
 """

    dir_path = os.path.dirname(os.path.realpath(__file__))
    experiments = ['letter', 'mnist', 'fashion', 'blobs', 'KDDCup99', 'covertype']
    for experiment_name in experiments:
        results_filename = os.path.join(dir_path, f"results/{experiment_name}.csv")
        results = alglab.results.Results(results_filename)
        table += """\\multirow{3}{*}{""" + experiment_name + """} & Time & """

        time_stats = results.get_final_stat("total_running_time_s")
        best_val = get_best_stat_from_stats(time_stats, algorithms, lower_is_better=True)
        table += " & ".join([format_table_number(time_stats, alg, best_val, lower_is_better=True) for alg in algorithms])

        final_stats = results.get_final_stat("adjusted_rand_index")
        best_val = get_best_stat_from_stats(final_stats, algorithms, lower_is_better=False)
        table += " \\\\\n & ARI & "
        table += " & ".join([format_table_number(final_stats, alg, best_val, lower_is_better=False) for alg in algorithms])

        final_stats = results.get_final_stat("normalised_mutual_info")
        best_val = get_best_stat_from_stats(final_stats, algorithms, lower_is_better=False)
        table += " \\\\\n & NMI & "
        table += " & ".join([format_table_number(final_stats, alg, best_val, lower_is_better=False) for alg in algorithms])

        table += "\\\\\n"

        if experiment_name != experiments[-1]:
            table += " \\midrule \n"

    table += "\\bottomrule \n \\end{tabular} \n } \n \caption{Experimental Results \\label{tab:results}} \n \\end{table}"
    return table

if __name__ == "__main__":
    table = create_numerical_results_table()
    print(table)