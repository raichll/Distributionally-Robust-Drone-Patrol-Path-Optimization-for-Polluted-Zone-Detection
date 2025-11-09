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

def generate_trajectory_clustering_results(static_dataset, results, name, dir_path):
    """
    为轨迹数据生成聚类结果可视化和详细结果
    """
    print(f"正在生成 {name} 数据的聚类结果...")
    
    # 创建输出目录
    output_dir = os.path.join(dir_path, f"clustering_results/{name}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 读取结果数据
    results_df = results.results_df
    
    # 获取最终的聚类结果（最后一次迭代的结果）
    final_results = results_df.groupby('algorithm').last().reset_index()
    
    # 保存详细的聚类结果
    clustering_results = {}
    
    # 从结果中提取聚类标签（这里需要根据您的具体算法实现来调整）
    # 假设算法返回的是聚类标签，我们需要重新运行算法来获取具体的聚类标签
    
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
                    'cluster_label': final_labels
                })
                result_df.to_csv(os.path.join(output_dir, f"{alg_name}_clustering_results.csv"), index=False)
                
                # 统计聚类信息
                unique_labels = np.unique(final_labels)
                n_clusters = len(unique_labels[unique_labels >= 0])  # 排除噪声点(-1)
                n_noise = np.sum(final_labels == -1)
                
                print(f"算法 {alg_name}:")
                print(f"  - 聚类数量: {n_clusters}")
                print(f"  - 噪声点数量: {n_noise}")
                print(f"  - 聚类标签范围: {unique_labels}")
                
        except Exception as e:
            print(f"运行算法 {alg_name} 时出错: {e}")
            continue
    
    # 生成聚类结果可视化
    if clustering_results:
        generate_clustering_visualization(static_dataset, clustering_results, output_dir)
    
    # 生成聚类结果统计报告
    generate_clustering_report(clustering_results, results_df, output_dir)
    
    print(f"聚类结果已保存到: {output_dir}")


def generate_clustering_visualization(static_dataset, clustering_results, output_dir):
    """
    生成聚类结果的可视化图
    """
    print("正在生成聚类可视化图...")
    
    # 获取数据点
    data_points = static_dataset.data
    
    # 如果数据维度大于2，使用前两个维度或进行降维
    if data_points.shape[1] >= 2:
        x_coords = data_points[:, 0]
        y_coords = data_points[:, 1]
        coord_labels = ['Feature 1', 'Feature 2']
    else:
        # 如果只有一个维度，创建一个人工的第二维度
        x_coords = data_points[:, 0]
        y_coords = np.zeros_like(x_coords)
        coord_labels = ['Feature 1', 'Artificial Y']
    
    # 为每个算法生成散点图
    n_algorithms = len(clustering_results)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    colors = plt.cm.Set3(np.linspace(0, 1, 20))  # 使用更多颜色
    
    for idx, (alg_name, labels) in enumerate(clustering_results.items()):
        if idx >= 4:  # 最多显示4个算法
            break
            
        ax = axes[idx]
        
        # 获取唯一的聚类标签
        unique_labels = np.unique(labels)
        
        # 绘制每个聚类
        for label in unique_labels:
            if label == -1:
                # 噪声点用黑色标记
                mask = labels == label
                ax.scatter(x_coords[mask], y_coords[mask], 
                          c='black', marker='x', s=50, alpha=0.6, label='Noise')
            else:
                # 正常聚类点
                mask = labels == label
                ax.scatter(x_coords[mask], y_coords[mask], 
                          c=[colors[label % len(colors)]], s=50, alpha=0.7, 
                          label=f'Cluster {label}')
        
        ax.set_title(f'{alg_name} Clustering Results')
        ax.set_xlabel(coord_labels[0])
        ax.set_ylabel(coord_labels[1])
        ax.grid(True, alpha=0.3)
        
        # 只显示前10个聚类的图例，避免图例过于拥挤
        handles, labels_legend = ax.get_legend_handles_labels()
        if len(handles) > 10:
            ax.legend(handles[:10], labels_legend[:10], loc='upper right', fontsize=8)
        else:
            ax.legend(loc='upper right', fontsize=8)
    
    # 隐藏多余的子图
    for idx in range(len(clustering_results), 4):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'clustering_visualization.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'clustering_visualization.pdf'), bbox_inches='tight')
    plt.close()
    
    print("聚类可视化图已生成: clustering_visualization.png 和 clustering_visualization.pdf")


def generate_clustering_report(clustering_results, results_df, output_dir):
    """
    生成聚类结果的统计报告
    """
    print("正在生成聚类统计报告...")
    
    report_lines = []
    report_lines.append("# 轨迹数据聚类结果报告")
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
        
        report_lines.append(f"聚类数量: {n_clusters}")
        report_lines.append(f"噪声点数量: {n_noise}")
        report_lines.append(f"噪声点比例: {n_noise/len(labels)*100:.2f}%")
        
        # 每个聚类的大小
        report_lines.append("\n各聚类大小:")
        for label in unique_labels:
            if label >= 0:
                cluster_size = np.sum(labels == label)
                report_lines.append(f"  聚类 {label}: {cluster_size} 个轨迹")
        
        report_lines.append("")
    
    # 运行时间统计
    report_lines.append("## 运行时间统计")
    report_lines.append("-" * 30)
    
    final_times = results_df.groupby('algorithm')['total_running_time_s'].last()
    for alg_name, time_taken in final_times.items():
        report_lines.append(f"{alg_name}: {time_taken:.4f} 秒")
    
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


def adaptive_dbscan_params(n, d, 
                                 base_eps=0.75, min_eps=0.4,
                                 base_k=10, max_k=30,
                                 base_t=10, max_t=20):
    """
    温和自适应调整 DBSCAN 的 (eps, k, t)，保持与初始值相近。
    
    参数：
        n (int): 数据规模
        d (int): 数据维度（用于理论扩展）
        
    返回：
        eps (float): 邻域半径
        k (int): 核心点阈值
        t (int): 哈希函数数量
    """
    logn = np.log(n)
    
    # ε：随 log(n) 缓慢衰减，控制在 [min_eps, base_eps]
    eps_decay = 1 / (1 + 0.03 * (logn - np.log(10000)))  # 平滑衰减因子
    eps = max(min_eps, base_eps * eps_decay)

    # k：随 log(n) 缓慢增加，最多不超过 max_k
    k = int(min(base_k + 0.6 * (logn - np.log(10000)), max_k))

    # t：对数增长，限制最大值
    t = int(min(base_t + (logn - np.log(10000)) / np.log(2), max_t))

    return round(eps, 4), k, t

def dynamic_experiment(name: str, static_dataset, eps, min_samples_new, min_samples_sklearn, t, d, burnin,
                       cluster_by_cluster=False,
                       skip_sklearn=False,
                       skip_evaluation=False):  # 添加参数来控制是否跳过评估
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if cluster_by_cluster:
        name = name + "_by_cluster"
    results_filename = os.path.join(dir_path, f"results/{name}.csv")
    dynamic_dataset = alglab.dataset.DynamicPointCloudDataset.from_pointcloud(static_dataset, 1000,
                                                                              stream_by_cluster=cluster_by_cluster)
    
    Ada_eps,Ada_min_samples_sklearn,Ada_t = adaptive_dbscan_params(static_dataset.n, static_dataset.d)
    
    algorithms = [dynamic_dbscan_alg, fdbscan_dynamic_naive, fdbscan_dynamic_noncore, Ada_dynamic_dbscan_alg]
    if not skip_sklearn:
        algorithms.append(sklearn_dynamic_alg)
    
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
    
    # 为轨迹数据生成聚类结果可视化和详细结果
    if name == 'truck':
        generate_trajectory_clustering_results(static_dataset, results, name, dir_path)
    
    return results

def blobs_experiment():
    random.seed(2024)
    blobs_dataset = alglab.dataset.BlobsDataset(n=50000, d=5, k=10)
    dynamic_experiment('blobs', blobs_dataset, 1, 50, 10, 5, 5, 20,
                       cluster_by_cluster=False)

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
    dates = ["2024-04-10"]
    truck_dataset = alglab.dataset_base.LocalTrajectoryDataset(base_folder=base_path, date_range=dates)
    
    print(f"加载的轨迹数据信息:")
    print(f"  - 数据点数量: {truck_dataset.n}")
    print(f"  - 数据维度: {truck_dataset.d}")
    print(f"  - 数据形状: {truck_dataset.data.shape}")
    
    truck_dataset.apply_scaling()
    
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