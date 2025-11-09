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
