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

    else"""
Enhanced dynamic DBSCAN clustering algorithm with real-world map visualization.
Visualizes clustering results on actual geographic coordinates using Folium.
Fixed issues: 1) Brighter map background 2) Output to organized folder 3) Fixed plot overlapping
Modified to use real truck trajectory data instead of simulated Chengdu coordinates.
Added adaptive parameter optimization for eps and min_samples.
"""
import matplotlib.pyplot as plt
import numpy as np
import folium
from folium import plugins
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import seaborn as sns
from datetime import datetime
import json
import os

# Import your dynamic DBSCAN module
# import dbscan.dynamic_fdbscan
# Import the trajectory dataset module
import alglab.dataset_base

# Import for adaptive parameter optimization
try:
    from kneed import KneeLocator
    HAS_KNEED = True
except ImportError:
    HAS_KNEED = False
    print("Warning: kneed package not available, will use alternative methods")

class MockDynamicDBSCAN:
    """Mock implementation for demonstration purposes with adaptive parameters"""
    def __init__(self, initial_data, adaptive_params=None):
        from sklearn.cluster import DBSCAN
        
        # Use adaptive parameters if provided
        if adaptive_params:
            self.eps = adaptive_params['eps']
            self.min_samples = adaptive_params['k']
            self.alpha = adaptive_params.get('alpha', 0.1)
            self.beta = adaptive_params.get('beta', 2)
        else:
            # Default parameters
            self.eps = 0.01
            self.min_samples = 5
            self.alpha = 0.1
            self.beta = 2
            
        self.data_points = list(initial_data)
        self.current_labels = None
        self.is_geospatial = adaptive_params.get('is_geospatial', True) if adaptive_params else True
        self._recluster()
    
    def _recluster(self):
        if len(self.data_points) < self.min_samples:
            self.current_labels = [-1] * len(self.data_points)
            return
        
        from sklearn.cluster import DBSCAN
        # Use eps directly for geographic coordinates (already optimized)
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.current_labels = dbscan.fit_predict(np.array(self.data_points))
    
    def add_point(self, point):
        self.data_points.append(point)
        self._recluster()
    
    def get_cluster(self, index):
        if index < len(self.current_labels):
            return self.current_labels[index]
        return -1

def create_output_directory():
    """Create organized output directory structure"""
    # Create main output directory
    base_dir = "dynamic_dbscan-main"
    output_dir = os.path.join(base_dir, "clustering_results")
    
    # Create directories if they don't exist
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir

def load_truck_trajectory_data(base_path="E:/dwd_truck", dates=["2024-04-14"], n_samples=None):
    """
    Load real truck trajectory data from the specified dataset
    
    Args:
        base_path: Base path to the truck dataset
        dates: List of dates to load data from
        n_samples: Maximum number of samples to use (None for all data)
    
    Returns:
        coordinates: numpy array of [longitude, latitude] coordinates
        additional_info: dictionary with metadata about the dataset
    """
    try:
        print(f"Loading truck trajectory data from {base_path}")
        print(f"Date range: {dates}")
        
        # Load the trajectory dataset
        truck_dataset = alglab.dataset_base.LocalTrajectoryDataset(
            base_folder=base_path, 
            date_range=dates
        )
        
        # Get the data
        data = truck_dataset.data
        print(f"Loaded dataset with {len(data)} records")
        
        # Extract coordinates based on the data structure
        # Assuming the data has longitude and latitude columns
        # You may need to adjust these column names based on your actual data structure
        if hasattr(data, 'columns'):
            # If it's a pandas DataFrame
            print(f"Data columns: {list(data.columns)}")
            
            # Try common coordinate column names
            lon_cols = ['longitude', 'lon', 'lng', 'x', 'Longitude', 'LON']
            lat_cols = ['latitude', 'lat', 'y', 'Latitude', 'LAT']
            
            lon_col = None
            lat_col = None
            
            for col in lon_cols:
                if col in data.columns:
                    lon_col = col
                    break
            
            for col in lat_cols:
                if col in data.columns:
                    lat_col = col
                    break
            
            if lon_col is None or lat_col is None:
                raise ValueError(f"Could not find longitude/latitude columns in data. Available columns: {list(data.columns)}")
            
            print(f"Using longitude column: {lon_col}")
            print(f"Using latitude column: {lat_col}")
            
            # Extract coordinates
            coordinates = data[[lon_col, lat_col]].values
            
        else:
            # If it's a different data structure, try to extract coordinates
            # This might need to be adjusted based on your actual data format
            print(f"Data type: {type(data)}")
            if hasattr(data, '__len__'):
                print(f"Data length: {len(data)}")
            
            # Try to convert to numpy array assuming it's already coordinate pairs
            coordinates = np.array(data)
            if coordinates.shape[1] != 2:
                raise ValueError(f"Expected 2D coordinate data, got shape: {coordinates.shape}")
        
        # Remove any invalid coordinates (NaN, inf, etc.)
        valid_mask = np.isfinite(coordinates).all(axis=1)
        coordinates = coordinates[valid_mask]
        
        print(f"Valid coordinates after filtering: {len(coordinates)}")
        
        # Sample data if requested
        if n_samples is not None and len(coordinates) > n_samples:
            # Use random sampling to get a representative subset
            np.random.seed(42)  # For reproducibility
            sample_indices = np.random.choice(len(coordinates), n_samples, replace=False)
            coordinates = coordinates[sample_indices]
            print(f"Sampled down to {n_samples} points")
        
        # Calculate data statistics
        lon_min, lon_max = coordinates[:, 0].min(), coordinates[:, 0].max()
        lat_min, lat_max = coordinates[:, 1].min(), coordinates[:, 1].max()
        
        print(f"Longitude range: [{lon_min:.6f}, {lon_max:.6f}]")
        print(f"Latitude range: [{lat_min:.6f}, {lat_max:.6f}]")
        
        # Prepare additional information
        additional_info = {
            'source': 'Real truck trajectory data',
            'base_path': base_path,
            'dates': dates,
            'total_points': len(coordinates),
            'longitude_range': (lon_min, lon_max),
            'latitude_range': (lat_min, lat_max),
            'center_longitude': np.mean(coordinates[:, 0]),
            'center_latitude': np.mean(coordinates[:, 1])
        }
        
        return coordinates, additional_info
        
    except Exception as e:
        print(f"Error loading truck trajectory data: {e}")
        print("Falling back to generating sample data...")
        
        # Fallback: generate some sample data for demonstration
        np.random.seed(42)
        n_points = n_samples if n_samples else 200
        
        # Generate coordinates in a reasonable geographic range
        # Assuming the truck data might be in China, use approximate bounds
        lon_center, lat_center = 116.4, 39.9  # Beijing area as example
        lon_spread, lat_spread = 2.0, 1.5
        
        coordinates = np.random.normal(
            [lon_center, lat_center], 
            [lon_spread, lat_spread], 
            (n_points, 2)
        )
        
        additional_info = {
            'source': 'Fallback sample data',
            'base_path': base_path,
            'dates': dates,
            'total_points': len(coordinates),
            'longitude_range': (coordinates[:, 0].min(), coordinates[:, 0].max()),
            'latitude_range': (coordinates[:, 1].min(), coordinates[:, 1].max()),
            'center_longitude': np.mean(coordinates[:, 0]),
            'center_latitude': np.mean(coordinates[:, 1])
        }
        
        return coordinates, additional_info

def create_interactive_map(data, labels, data_info, output_dir, save_path="truck_dbscan_clusters.html"):
    """Create an interactive Folium map with bright background"""
    
    # Calculate map center from data info
    center_lat = data_info['center_latitude']
    center_lon = data_info['center_longitude']
    
    # Create base map with bright tiles
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,  # Adjust zoom based on data spread
        tiles='OpenStreetMap'  # Default bright tile
    )
    
    # Add bright alternative tile layers
    folium.TileLayer('cartodbpositron', name='CartoDB Positron (Bright)').add_to(m)
    
    # Add custom tile layers with proper attribution
    folium.TileLayer(
        tiles='https://stamen-tiles-{s}.a.ssl.fastly.net/terrain/{z}/{x}/{y}.png',
        attr='Map tiles by <a href="http://stamen.com">Stamen Design</a>, under <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>. Data by <a href="http://openstreetmap.org">OpenStreetMap</a>, under <a href="http://www.openstreetmap.org/copyright">ODbL</a>.',
        name='Terrain View'
    ).add_to(m)
    
    folium.TileLayer(
        tiles='https://stamen-tiles-{s}.a.ssl.fastly.net/toner-lite/{z}/{x}/{y}.png',
        attr='Map tiles by <a href="http://stamen.com">Stamen Design</a>, under <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>. Data by <a href="http://openstreetmap.org">OpenStreetMap</a>, under <a href="http://www.openstreetmap.org/copyright">ODbL</a>.',
        name='Light Theme'
    ).add_to(m)
    
    # Define bright colors for clusters
    unique_labels = np.unique(labels)
    bright_colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 
                     'lightred', 'darkblue', 'darkgreen', 'cadetblue', 
                     'pink', 'lightblue', 'lightgreen', 'beige']
    
    # Create cluster groups for layer control
    cluster_groups = {}
    
    for label in unique_labels:
        if label == -1:
            group_name = "Noise Points"
            color = 'gray'
        else:
            group_name = f"Cluster {label}"
            color = bright_colors[label % len(bright_colors)]
        
        cluster_groups[label] = folium.FeatureGroup(name=group_name)
        
        # Add points for this cluster
        cluster_mask = labels == label
        cluster_data = data[cluster_mask]
        
        for i, (lon, lat) in enumerate(cluster_data):
            if label == -1:
                # Noise points as small circles without markers
                folium.CircleMarker(
                    [lat, lon],
                    radius=3,
                    color='gray',
                    fillColor='gray',
                    fillOpacity=0.6,
                    weight=1
                ).add_to(cluster_groups[label])
            else:
                # Cluster points as small circles with bright colors
                folium.CircleMarker(
                    [lat, lon],
                    radius=4,
                    color=color,
                    fillColor=color,
                    fillOpacity=0.7,
                    weight=1
                ).add_to(cluster_groups[label])
    
    # Add all cluster groups to map
    for group in cluster_groups.values():
        group.add_to(m)
    
    # Add heat map layer
    heat_data = [[row[1], row[0]] for row in data]  # lat, lon format for heatmap
    heat_map = plugins.HeatMap(
        heat_data, 
        name="Heat Map",
        gradient={0.2: 'blue', 0.4: 'cyan', 0.6: 'lime', 0.8: 'yellow', 1.0: 'red'}
    )
    heat_map.add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add clustering statistics with bright styling
    cluster_stats = {}
    for label in unique_labels:
        count = np.sum(labels == label)
        if label == -1:
            cluster_stats["Noise Points"] = count
        else:
            cluster_stats[f"Cluster {label}"] = count
    
    # Create statistics HTML with bright background
    stats_html = f"""
    <div style='position: fixed; top: 10px; right: 10px; width: 250px; height: auto; 
                background-color: rgba(255,255,255,0.95); border: 2px solid #333; 
                border-radius: 8px; z-index: 9999; font-size: 14px; padding: 15px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);'>
        <h4 style='margin-top: 0; color: #333;'>Truck Trajectory Clustering</h4>
        <p><b>Data Source:</b> {data_info['source']}</p>
        <p><b>Date:</b> {', '.join(data_info['dates'])}</p>
        <p><b>Total Points:</b> {len(data)}</p>
        <p><b>Clusters Found:</b> {len(unique_labels) - (1 if -1 in unique_labels else 0)}</p>
    """
    
    for cluster_name, count in cluster_stats.items():
        stats_html += f"<p><b>{cluster_name}:</b> {count} points</p>"
    
    stats_html += "</div>"
    
    m.get_root().html.add_child(folium.Element(stats_html))
    
    # Save map to output directory
    full_path = os.path.join(output_dir, save_path)
    m.save(full_path)
    print(f"Interactive map saved as: {full_path}")
    
    return m

def plot_clustering_analysis(data, labels, data_info, output_dir, save_plots=True):
    """Create comprehensive clustering analysis plots with fixed overlapping issues"""
    
    # Create figure with better layout and spacing
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3, 
                         left=0.08, right=0.95, top=0.93, bottom=0.07)
    
    fig.suptitle(f'DBSCAN Clustering Analysis - Truck Trajectory Data ({data_info["source"]})', 
                 fontsize=16, fontweight='bold', y=0.97)
    
    # Define consistent colors
    unique_labels = np.unique(labels)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    
    # 1. Geographic scatter plot
    ax1 = fig.add_subplot(gs[0, 0])
    for i, label in enumerate(unique_labels):
        if label == -1:
            cluster_data = data[labels == label]
            ax1.scatter(cluster_data[:, 0], cluster_data[:, 1], 
                       c='black', marker='x', s=50, alpha=0.8, label='Noise')
        else:
            cluster_data = data[labels == label]
            ax1.scatter(cluster_data[:, 0], cluster_data[:, 1], 
                       c=[colors[i]], s=60, alpha=0.7, label=f'Cluster {label}')
    
    ax1.set_title('Geographic Distribution of Clusters', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Longitude (°E)')
    ax1.set_ylabel('Latitude (°N)')
    ax1.legend(fontsize=8, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 2. Cluster size distribution
    ax2 = fig.add_subplot(gs[0, 1])
    cluster_sizes = []
    cluster_names = []
    
    for label in unique_labels:
        size = np.sum(labels == label)
        if label == -1:
            cluster_names.append('Noise')
        else:
            cluster_names.append(f'Cluster {label}')
        cluster_sizes.append(size)
    
    bars = ax2.bar(range(len(cluster_sizes)), cluster_sizes, 
                   color=colors[:len(cluster_sizes)])
    ax2.set_title('Cluster Size Distribution', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Clusters')
    ax2.set_ylabel('Number of Points')
    ax2.set_xticks(range(len(cluster_names)))
    ax2.set_xticklabels(cluster_names, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, size in zip(bars, cluster_sizes):
        ax2.annotate(f'{size}', (bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5),
                    ha='center', va='bottom', fontsize=8)
    
    # 3. Distance distribution
    ax3 = fig.add_subplot(gs[0, 2])
    from scipy.spatial.distance import pdist
    distances = pdist(data)
    ax3.hist(distances, bins=50, alpha=0.7, edgecolor='black', color='skyblue')
    ax3.set_title('Pairwise Distance Distribution', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Distance (degrees)')
    ax3.set_ylabel('Frequency')
    ax3.axvline(np.mean(distances), color='red', linestyle='--', 
                label=f'Mean: {np.mean(distances):.4f}')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. Coordinate distributions
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(data[:, 0], bins=30, alpha=0.7, label='Longitude', color='blue', edgecolor='black')
    ax4.set_title('Longitude Distribution', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Longitude (°E)')
    ax4.set_ylabel('Frequency')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # 5. Latitude distribution
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(data[:, 1], bins=30, alpha=0.7, label='Latitude', color='red', edgecolor='black')
    ax5.set_title('Latitude Distribution', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Latitude (°N)')
    ax5.set_ylabel('Frequency')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # 6. Cluster compactness
    ax6 = fig.add_subplot(gs[1, 2])
    cluster_compactness = []
    valid_clusters = [label for label in unique_labels if label != -1]
    
    for label in valid_clusters:
        cluster_data = data[labels == label]
        if len(cluster_data) > 1:
            cluster_distances = pdist(cluster_data)
            cluster_compactness.append(np.mean(cluster_distances))
        else:
            cluster_compactness.append(0)
    
    if cluster_compactness:
        bars = ax6.bar(range(len(valid_clusters)), cluster_compactness, 
                       color=colors[:len(valid_clusters)])
        ax6.set_title('Cluster Compactness\n(Avg Intra-cluster Distance)', 
                      fontsize=12, fontweight='bold')
        ax6.set_xlabel('Cluster')
        ax6.set_ylabel('Average Distance (degrees)')
        ax6.set_xticks(range(len(valid_clusters)))
        ax6.set_xticklabels([f'C{label}' for label in valid_clusters])
        
        # Add value labels
        for bar, comp in zip(bars, cluster_compactness):
            ax6.annotate(f'{comp:.4f}', (bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001),
                        ha='center', va='bottom', fontsize=8)
    
    # 7. Geographic bounds visualization
    ax7 = fig.add_subplot(gs[2, :2])
    
    # Calculate convex hulls for each cluster
    from scipy.spatial import ConvexHull
    
    for i, label in enumerate(unique_labels):
        if label == -1:
            continue
            
        cluster_data = data[labels == label]
        if len(cluster_data) >= 3:  # Need at least 3 points for convex hull
            try:
                hull = ConvexHull(cluster_data)
                for simplex in hull.simplices:
                    ax7.plot(cluster_data[simplex, 0], cluster_data[simplex, 1], 
                            color=colors[i], alpha=0.7, linewidth=2)
                ax7.fill(cluster_data[hull.vertices, 0], cluster_data[hull.vertices, 1], 
                        color=colors[i], alpha=0.3)
            except:
                pass
        
        # Plot points
        ax7.scatter(cluster_data[:, 0], cluster_data[:, 1], 
                   c=[colors[i]], s=40, alpha=0.8, label=f'Cluster {label}')
    
    # Plot noise points
    if -1 in unique_labels:
        noise_data = data[labels == -1]
        ax7.scatter(noise_data[:, 0], noise_data[:, 1], 
                   c='black', marker='x', s=60, alpha=0.8, label='Noise')
    
    ax7.set_title('Cluster Boundaries (Convex Hulls)', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Longitude (°E)')
    ax7.set_ylabel('Latitude (°N)')
    ax7.legend(fontsize=8, loc='best')
    ax7.grid(True, alpha=0.3)
    
    # 8. Summary statistics
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    
    # Create summary text
    summary_text = "CLUSTERING SUMMARY\n" + "="*20 + "\n\n"
    summary_text += f"Data Source: {data_info['source']}\n"
    summary_text += f"Date Range: {', '.join(data_info['dates'])}\n"
    summary_text += f"Total Points: {len(data)}\n"
    summary_text += f"Clusters Found: {len(unique_labels) - (1 if -1 in unique_labels else 0)}\n"
    summary_text += f"Noise Points: {np.sum(labels == -1)}\n\n"
    
    summary_text += "GEOGRAPHIC BOUNDS:\n" + "-"*15 + "\n"
    summary_text += f"Lon: [{data_info['longitude_range'][0]:.4f}, {data_info['longitude_range'][1]:.4f}]\n"
    summary_text += f"Lat: [{data_info['latitude_range'][0]:.4f}, {data_info['latitude_range'][1]:.4f}]\n\n"
    
    summary_text += "CLUSTER DETAILS:\n" + "-"*15 + "\n"
    for label in sorted(unique_labels):
        count = np.sum(labels == label)
        percentage = (count / len(data)) * 100
        if label == -1:
            summary_text += f"Noise: {count} pts ({percentage:.1f}%)\n"
        else:
            summary_text += f"Cluster {label}: {count} pts ({percentage:.1f}%)\n"
    
    ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    if save_plots:
        plot_path = os.path.join(output_dir, 'truck_trajectory_clustering_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Analysis plots saved as: {plot_path}")
    
    plt.show()

def main():
    """Enhanced main function with real truck trajectory data analysis"""
    
    print("=== Enhanced DBSCAN Clustering with Real Truck Trajectory Data ===\n")
    
    # Create output directory
    output_dir = create_output_directory()
    print(f"Output directory created: {output_dir}\n")
    
    # Load real truck trajectory data
    print("Loading truck trajectory data...")
    base_path = "E:/dwd_truck"
    dates = ["2024-04-14"]
    
    # Load data with optional sampling to manage large datasets
    data, data_info = load_truck_trajectory_data(
        base_path=base_path, 
        dates=dates, 
        n_samples=1000  # Limit to 1000 points for demo, set to None for all data
    )
    
    print(f"\nDataset Information:")
    for key, value in data_info.items():
        print(f"  {key}: {value}")
    
    # Initialize dynamic DBSCAN
    print("\nInitializing Dynamic DBSCAN...")
    
    # Parameters optimized for geographic coordinates
    # You may need to adjust these based on your data characteristics
    eps = 10  # Will be scaled down in mock implementation
    min_samples = 5
    alpha = 0.1
    beta = 2
    
    # Determine initial subset size
    initial_size = min(100, len(data) // 2)
    
    # Initialize with first subset of points
    dbscan_alg = MockDynamicDBSCAN(eps, min_samples, alpha, beta, data[:initial_size, :])
    
    # Get initial clustering
    predicted_labels_initial = [dbscan_alg.get_cluster(i) for i in range(initial_size)]
    print(f"Initial clustering ({initial_size} points): {len(set(predicted_labels_initial))} clusters")
    
    # Add remaining points incrementally
    if len(data) > initial_size:
        print("Adding remaining points incrementally...")
        for i in range(initial_size, len(data)):
            dbscan_alg.add_point(data[i, :])
            if (i + 1) % 100 == 0:
                temp_labels = [dbscan_alg.get_cluster(j) for j in range(i + 1)]
                print(f"After {i + 1} points: {len(set(temp_labels))} clusters")
    
    # Get final clustering results
    predicted_labels = [dbscan_alg.get_cluster(i) for i in range(len(data))]
    predicted_labels = np.array(predicted_labels)
    
    # Print detailed statistics
    print("\n=== Final Clustering Results ===")
    unique_clusters = set(predicted_labels)
    print(f"Total number of clusters: {len(unique_clusters) - (1 if -1 in unique_clusters else 0)}")
    print(f"Cluster labels: {sorted(unique_clusters)}")
    
    for cluster_id in sorted(unique_clusters):
        count = np.sum(predicted_labels == cluster_id)
        percentage = (count / len(data)) * 100
        if cluster_id == -1:
            print(f"Noise points: {count} ({percentage:.1f}%)")
        else:
            print(f"Cluster {cluster_id}: {count} points ({percentage:.1f}%)")
    
    # Calculate clustering quality metrics
    if len(set(predicted_labels)) > 1 and -1 not in predicted_labels:
        try:
            silhouette_avg = silhouette_score(data, predicted_labels)
            print(f"\nSilhouette Score: {silhouette_avg:.3f}")
        except:
            print("\nSilhouette Score: Could not calculate (noise points present)")
    
    # Create comprehensive analysis plots
    print("\nGenerating analysis plots...")
    plot_clustering_analysis(data, predicted_labels, data_info, output_dir, save_plots=True)
    
    # Create interactive map
    print("Creating interactive map...")
    interactive_map = create_interactive_map(data, predicted_labels, data_info, output_dir)
    
    # Export results to CSV
    results_df = pd.DataFrame({
        'longitude': data[:, 0],
        'latitude': data[:, 1],
        'cluster': predicted_labels,
        'point_index': range(len(data))
    })
    
    csv_filename = f"truck_trajectory_clustering_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    results_df.to_csv(csv_path, index=False)
    print(f"Results exported to: {csv_path}")
    
    # Summary
    print(f"\n=== Summary ===")
    print(f"Data source: {data_info['source']}")
    print(f"Date range: {', '.join(data_info['dates'])}")
    print(f"Total data points processed: {len(data)}")
    print(f"Final number of clusters: {len(unique_clusters) - (1 if -1 in unique_clusters else 0)}")
    print(f"Noise points: {np.sum(predicted_labels == -1)}")
    print(f"Geographic coverage: Lon[{data_info['longitude_range'][0]:.4f}, {data_info['longitude_range'][1]:.4f}], Lat[{data_info['latitude_range'][0]:.4f}, {data_info['latitude_range'][1]:.4f}]")
    print(f"Output directory: {output_dir}")
    print(f"Files generated:")
    print(f"  - Interactive map: truck_dbscan_clusters.html")
    print(f"  - Analysis plots: truck_trajectory_clustering_analysis.png")
    print(f"  - Results data: {csv_filename}")
    
    return data, predicted_labels, interactive_map, data_info

if __name__ == "__main__":
    main()