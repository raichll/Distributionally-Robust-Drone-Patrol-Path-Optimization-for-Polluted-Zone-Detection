"""2维数据聚类结果可视化函数"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')

def generate_2d_clustering_results(static_dataset, results, name, dir_path):
    """
    专门为2维轨迹数据生成聚类结果可视化和详细结果
    """
    print(f"正在生成 {name} 数据的2维聚类结果...")
    
    # 创建输出目录
    output_dir = os.path.join(dir_path, f"clustering_results/{name}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 读取结果数据
    results_df = results.results_df
    
    # 保存详细的聚类结果
    clustering_results = {}
    from algorithms import dynamic_dbscan_alg, fdbscan_dynamic_naive, fdbscan_dynamic_noncore, Ada_dynamic_dbscan_alg

    # 重新运行算法来获取聚类标签
    algorithms_to_run = [
        ('dynamic_dbscan', dynamic_dbscan_alg),
        ('fdbscan_dynamic_naive', fdbscan_dynamic_naive),
        ('fdbscan_dynamic_noncore', fdbscan_dynamic_noncore),
        ('Ada_dynamic_dbscan', Ada_dynamic_dbscan_alg)
    ]
    
    for alg_name, alg in algorithms_to_run:
        try:
            # 获取该算法的参数
            alg_params = {}
            for param_col in results_df.columns:
                if param_col in ['eps', 'min_samples', 't', 'd', 'burnin']:
                    alg_result = results_df[results_df['algorithm'] == alg_name]
                    if not alg_result.empty and param_col in alg_result.columns:
                        param_value = alg_result[param_col].iloc[0]
                        if pd.notna(param_value):
                            alg_params[param_col] = param_value
            
            print(f"运行算法 {alg_name} 获取聚类标签，参数: {alg_params}")
            
            # 运行算法获取聚类标签
            alg_generator = alg.run(static_dataset, alg_params)
            final_labels = None
            
            # 获取最终的聚类结果
            for labels in alg_generator:
                final_labels = labels
            
            if final_labels is not None:
                clustering_results[alg_name] = final_labels
                
                # 保存聚类结果到CSV
                result_df = pd.DataFrame({
                    'trajectory_id': range(len(final_labels)),
                    'cluster_label': final_labels,
                    'x_coord': static_dataset.data[:, 0],
                    'y_coord': static_dataset.data[:, 1]
                })
                result_df.to_csv(os.path.join(output_dir, f"{alg_name}_clustering_results.csv"), index=False)
                
                # 统计聚类信息
                unique_labels = np.unique(final_labels)
                n_clusters = len(unique_labels[unique_labels >= 0])
                n_noise = np.sum(final_labels == -1)
                
                print(f"算法 {alg_name}:")
                print(f"  - 聚类数量: {n_clusters}")
                print(f"  - 噪声点数量: {n_noise}")
                print(f"  - 聚类标签范围: {unique_labels}")
                
        except Exception as e:
            print(f"运行算法 {alg_name} 时出错: {e}")
            continue
    
    # 生成2维聚类结果可视化
    if clustering_results:
        generate_2d_clustering_plots(static_dataset.data, clustering_results, output_dir, name)
    
    # 生成聚类结果统计报告
    generate_clustering_report(clustering_results, results_df, output_dir)
    
    print(f"2维聚类结果已保存到: {output_dir}")


def generate_2d_clustering_plots(data_points, clustering_results, output_dir, dataset_name):
    """
    生成2维数据的聚类结果可视化图
    """
    print("正在生成2维聚类可视化图...")
    
    # 提取x和y坐标
    x_coords = data_points[:, 0]
    y_coords = data_points[:, 1]
    
    # 1. 生成主要的对比图（2x2子图）
    generate_main_comparison_plot(x_coords, y_coords, clustering_results, output_dir)
    
    # 2. 生成每个算法的单独详细图
    generate_individual_algorithm_plots(x_coords, y_coords, clustering_results, output_dir)
    
    # 3. 生成聚类统计图
    generate_clustering_statistics_plots(clustering_results, output_dir)
    
    # 4. 生成聚类质量评估图
    generate_clustering_quality_plots(data_points, clustering_results, output_dir)
    
    print("所有2维聚类可视化图已生成完成！")


def generate_main_comparison_plot(x_coords, y_coords, clustering_results, output_dir):
    """
    生成主要的算法对比图
    """
    # 设置图形样式
    plt.style.use('seaborn-v0_8')
    
    # 创建2x2子图
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    
    # 使用更美观的颜色调色板
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    
    algorithm_names = {
        'dynamic_dbscan': 'Dynamic DBSCAN',
        'fdbscan_dynamic_naive': 'FDBSCAN Naive',
        'fdbscan_dynamic_noncore': 'FDBSCAN NonCore',
        'Ada_dynamic_dbscan': 'Adaptive Dynamic DBSCAN'
    }
    
    for idx, (alg_name, labels) in enumerate(clustering_results.items()):
        if idx >= 4:
            break
            
        ax = axes[idx]
        
        # 获取唯一的聚类标签
        unique_labels = np.unique(labels)
        
        # 绘制每个聚类
        for label in unique_labels:
            if label == -1:
                # 噪声点用黑色X标记
                mask = labels == label
                ax.scatter(x_coords[mask], y_coords[mask], 
                          c='black', marker='x', s=80, alpha=0.8, 
                          label='Noise', linewidths=2)
            else:
                # 正常聚类点
                mask = labels == label
                ax.scatter(x_coords[mask], y_coords[mask], 
                          c=[colors[label % len(colors)]], s=60, alpha=0.7, 
                          label=f'Cluster {label}', edgecolors='black', linewidths=0.5)
        
        # 设置标题和标签
        display_name = algorithm_names.get(alg_name, alg_name)
        ax.set_title(f'{display_name}', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('X Coordinate', fontsize=14)
        ax.set_ylabel('Y Coordinate', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息框
        n_clusters = len(unique_labels[unique_labels >= 0])
        n_noise = np.sum(labels == -1)
        n_total = len(labels)
        noise_ratio = n_noise / n_total * 100
        
        stats_text = f'Clusters: {n_clusters}\nNoise: {n_noise}\nNoise Rate: {noise_ratio:.1f}%'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=12, 
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', 
                facecolor='white', alpha=0.9, edgecolor='gray'))
        
        # 只显示前8个聚类的图例
        handles, labels_legend = ax.get_legend_handles_labels()
        if len(handles) > 8:
            ax.legend(handles[:8], labels_legend[:8], loc='upper right', 
                     fontsize=10, framealpha=0.9)
        else:
            ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
        
        # 设置坐标轴范围，使图形更美观
        ax.set_xlim(x_coords.min() - 0.1 * (x_coords.max() - x_coords.min()),
                   x_coords.max() + 0.1 * (x_coords.max() - x_coords.min()))
        ax.set_ylim(y_coords.min() - 0.1 * (y_coords.max() - y_coords.min()),
                   y_coords.max() + 0.1 * (y_coords.max() - y_coords.min()))
    
    # 隐藏多余的子图
    for idx in range(len(clustering_results), 4):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'clustering_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'clustering_comparison.pdf'), bbox_inches='tight')
    plt.close()
    
    print("主要对比图已生成: clustering_comparison.png")


def generate_individual_algorithm_plots(x_coords, y_coords, clustering_results, output_dir):
    """
    为每个算法生成单独的高质量详细图
    """
    # 创建单独算法图目录
    individual_dir = os.path.join(output_dir, 'individual_algorithms')
    if not os.path.exists(individual_dir):
        os.makedirs(individual_dir)
    
    algorithm_names = {
        'dynamic_dbscan': 'Dynamic DBSCAN',
        'fdbscan_dynamic_naive': 'FDBSCAN Naive',
        'fdbscan_dynamic_noncore': 'FDBSCAN NonCore',
        'Ada_dynamic_dbscan': 'Adaptive Dynamic DBSCAN'
    }
    
    for alg_name, labels in clustering_results.items():
        # 创建大尺寸图形
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        # 获取唯一的聚类标签
        unique_labels = np.unique(labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        # 绘制每个聚类
        for i, label in enumerate(unique_labels):
            if label == -1:
                # 噪声点
                mask = labels == label
                ax.scatter(x_coords[mask], y_coords[mask], 
                          c='black', marker='x', s=100, alpha=0.8, 
                          label='Noise Points', linewidths=3)
            else:
                # 正常聚类点
                mask = labels == label
                ax.scatter(x_coords[mask], y_coords[mask], 
                          c=[colors[i]], s=80, alpha=0.7, 
                          label=f'Cluster {label}', edgecolors='black', linewidths=0.5)
        
        # 设置标题和标签
        display_name = algorithm_names.get(alg_name, alg_name)
        ax.set_title(f'{display_name} - Detailed Clustering Results', 
                    fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel('X Coordinate', fontsize=16)
        ax.set_ylabel('Y Coordinate', fontsize=16)
        ax.grid(True, alpha=0.3)
        
        # 添加详细统计信息
        n_clusters = len(unique_labels[unique_labels >= 0])
        n_noise = np.sum(labels == -1)
        n_total = len(labels)
        noise_ratio = n_noise / n_total * 100
        
        # 计算每个聚类的大小
        cluster_sizes = []
        for label in unique_labels:
            if label >= 0:
                cluster_size = np.sum(labels == label)
                cluster_sizes.append(f'Cluster {label}: {cluster_size}')
        
        stats_text = f'Total Points: {n_total}\n'
        stats_text += f'Number of Clusters: {n_clusters}\n'
        stats_text += f'Noise Points: {n_noise}\n'
        stats_text += f'Noise Ratio: {noise_ratio:.1f}%\n'
        stats_text += f'Avg Cluster Size: {(n_total-n_noise)/max(n_clusters,1):.1f}'
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=14, 
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.8', 
                facecolor='lightblue', alpha=0.9, edgecolor='navy'))
        
        # 图例
        ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
        
        # 设置坐标轴范围
        ax.set_xlim(x_coords.min() - 0.1 * (x_coords.max() - x_coords.min()),
                   x_coords.max() + 0.1 * (x_coords.max() - x_coords.min()))
        ax.set_ylim(y_coords.min() - 0.1 * (y_coords.max() - y_coords.min()),
                   y_coords.max() + 0.1 * (y_coords.max() - y_coords.min()))
        
        plt.tight_layout()
        plt.savefig(os.path.join(individual_dir, f'{alg_name}_detailed.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(individual_dir, f'{alg_name}_detailed.pdf'), bbox_inches='tight')
        plt.close()
    
    print(f"单独算法详细图已保存到: {individual_dir}")


def generate_clustering_statistics_plots(clustering_results, output_dir):
    """
    生成聚类统计条形图
    """
    algorithms = list(clustering_results.keys())
    algorithm_names = {
        'dynamic_dbscan': 'Dynamic\nDBSCAN',
        'fdbscan_dynamic_naive': 'FDBSCAN\nNaive',
        'fdbscan_dynamic_noncore': 'FDBSCAN\nNonCore',
        'Ada_dynamic_dbscan': 'Adaptive\nDBSCAN'
    }
    
    display_names = [algorithm_names.get(alg, alg) for alg in algorithms]
    
    # 准备数据
    n_clusters_list = []
    n_noise_list = []
    noise_ratios = []
    
    for alg_name, labels in clustering_results.items():
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels[unique_labels >= 0])
        n_noise = np.sum(labels == -1)
        n_total = len(labels)
        noise_ratio = n_noise / n_total * 100
        
        n_clusters_list.append(n_clusters)
        n_noise_list.append(n_noise)
        noise_ratios.append(noise_ratio)
    
    # 创建统计图
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # 聚类数量条形图
    bars1 = ax1.bar(display_names, n_clusters_list, color='steelblue', alpha=0.8, edgecolor='black')
    ax1.set_title('Number of Clusters', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Clusters', fontsize=12)
    ax1.set_xlabel('Algorithm', fontsize=12)
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 噪声点数量条形图
    bars2 = ax2.bar(display_names, n_noise_list, color='orange', alpha=0.8, edgecolor='black')
    ax2.set_title('Number of Noise Points', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Noise Points', fontsize=12)
    ax2.set_xlabel('Algorithm', fontsize=12)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 噪声点比例条形图
    bars3 = ax3.bar(display_names, noise_ratios, color='red', alpha=0.8, edgecolor='black')
    ax3.set_title('Noise Ratio (%)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Noise Ratio (%)', fontsize=12)
    ax3.set_xlabel('Algorithm', fontsize=12)
    
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'clustering_statistics.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'clustering_statistics.pdf'), bbox_inches='tight')
    plt.close()
    
    print("聚类统计图已生成: clustering_statistics.png")


def generate_clustering_quality_plots(data_points, clustering_results, output_dir):
    """
    生成聚类质量评估图
    """
    algorithms = list(clustering_results.keys())
    algorithm_names = {
        'dynamic_dbscan': 'Dynamic\nDBSCAN',
        'fdbscan_dynamic_naive': 'FDBSCAN\nNaive',
        'fdbscan_dynamic_noncore': 'FDBSCAN\nNonCore',
        'Ada_dynamic_dbscan': 'Adaptive\nDBSCAN'
    }
    
    display_names = [algorithm_names.get(alg, alg) for alg in algorithms]
    
    silhouette_scores = []
    calinski_scores = []
    davies_bouldin_scores = []
    
    for alg_name, labels in clustering_results.items():
        try:
            # 只计算有效聚类点的指标（排除噪声点）
            valid_mask = labels >= 0
            if np.sum(valid_mask) < 2:
                silhouette_scores.append(0)
                calinski_scores.append(0)
                davies_bouldin_scores.append(float('inf'))
                continue
            
            valid_data = data_points[valid_mask]
            valid_labels = labels[valid_mask]
            
            # 计算各种内部评估指标
            if len(np.unique(valid_labels)) > 1:
                silhouette_avg = silhouette_score(valid_data, valid_labels)
                calinski_avg = calinski_harabasz_score(valid_data, valid_labels)
                davies_bouldin_avg = davies_bouldin_score(valid_data, valid_labels)
            else:
                silhouette_avg = 0
                calinski_avg = 0
                davies_bouldin_avg = float('inf')
            
            silhouette_scores.append(silhouette_avg)
            calinski_scores.append(calinski_avg)
            davies_bouldin_scores.append(davies_bouldin_avg)
            
        except Exception as e:
            print(f"计算 {alg_name} 的质量指标时出错: {e}")
            silhouette_scores.append(0)
            calinski_scores.append(0)
            davies_bouldin_scores.append(float('inf'))
    
    # 创建质量评估图
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Silhouette Score (higher is better)
    bars1 = ax1.bar(display_names, silhouette_scores, color='green', alpha=0.8, edgecolor='black')
    ax1.set_title('Silhouette Score\n(Higher is Better)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Silhouette Score', fontsize=12)
    ax1.set_xlabel('Algorithm', fontsize=12)
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Calinski-Harabasz Score (higher is better)
    bars2 = ax2.bar(display_names, calinski_scores, color='blue', alpha=0.8, edgecolor='black')
    ax2.set_title('Calinski-Harabasz Score\n(Higher is Better)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Calinski-Harabasz Score', fontsize=12)
    ax2.set_xlabel('Algorithm', fontsize=12)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Davies-Bouldin Score (lower is better)
    # 处理无穷大值
    davies_bouldin_scores_display = [score if score != float('inf') else 0 for score in davies_bouldin_scores]
    bars3 = ax3.bar(display_names, davies_bouldin_scores_display, color='red', alpha=0.8, edgecolor='black')
    ax3.set_title('Davies-Bouldin Score\n(Lower is Better)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Davies-Bouldin Score', fontsize=12)
    ax3.set_xlabel('Algorithm', fontsize=12)
    
    for i, bar in enumerate(bars3):
        height = bar.get_height()
        if davies_bouldin_scores[i] == float('inf'):
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    'N/A', ha='center', va='bottom', fontsize=11, fontweight='bold')
        else:
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'clustering_quality.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'clustering_quality.pdf'), bbox_inches='tight')
    plt.close()
    
    print("聚类质量评估图已生成: clustering_quality.png")


def generate_clustering_report(clustering_results, results_df, output_dir):
    """
    生成聚类结果的统计报告
    """
    print("正在生成聚类统计报告...")
    
    report_lines = []
    report_lines.append("# 2维轨迹数据聚类结果报告")
    report_lines.append("=" * 50)
    report_lines.append("")
    
    # 总体统计
    if clustering_results:
        total_trajectories = len(list(clustering_results.values())[0])
        report_lines.append(f"总轨迹数量: {total_trajectories}")
        report_lines.append("")
    
    # 各算法统计
    for alg_name, labels in clustering_results.items():
        report_lines.append(f"## {alg_name} 算法结果")
        report_lines.append("-" * 30)
        
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels[unique_labels >= 0])
        n_noise = np.sum(labels == -1)
        n_total = len(labels)
        noise_ratio = n_noise / n_total * 100
        
        report_lines.append(f"聚类数量: {n_clusters}")
        report_lines.append(f"噪声点数量: {n_noise}")
        report_lines.append(f"噪声点比例: {noise_ratio:.2f}%")
        
        # 每个聚类的大小
        if n_clusters > 0:
            report_lines.append("\n各聚类大小:")
            for label in unique_labels:
                if label >= 0:
                    cluster_size = np.sum(labels == label)
                    cluster_ratio = cluster_size / n_total * 100
                    report_lines.append(f"  聚类 {label}: {cluster_size} 个轨迹 ({cluster_ratio:.1f}%)")
        
        report_lines.append("")
    
    # 运行时间统计
    report_lines.append("## 运行时间统计")
    report_lines.append("-" * 30)
    
    if 'total_running_time_s' in results_df.columns:
        final_times = results_df.groupby('algorithm')['total_running_time_s'].last()
        for alg_name, time_taken in final_times.items():
            report_lines.append(f"{alg_name}: {time_taken:.4f} 秒")
    else:
        report_lines.append("运行时间数据不可用")
    
    report_lines.append("")
    
    # 保存报告
    report_content = "\n".join(report_lines)
    with open(os.path.join(output_dir, 'clustering_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print("聚类统计报告已生成: clustering_report.txt")
    print("\n" + "="*50)
    print("聚类结果摘要:")
    print("="*50)
    print(report_content)


# 修改原始函数以使用新的2维可视化函数
def generate_trajectory_clustering_results(static_dataset, results, name, dir_path):
    """
    为轨迹数据生成聚类结果可视化和详细结果（调用2维专用函数）
    """
    if static_dataset.data.shape[1] == 2:
        # 使用专门的2维可视化函数
        generate_2d_clustering_results(static_dataset, results, name, dir_path)
    else:
        print(f"数据维度为{static_dataset.data.shape[1]}，不是2维数据，跳过可视化")