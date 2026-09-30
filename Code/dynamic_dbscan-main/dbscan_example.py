"""
Demonstrate a simple usage of the dynamic DBSCAN clustering algorithm and visualize its results.
"""
# from sklearn.metrics import adjusted_rand_score # 移除此行，因为不再需要验证
from sklearn.datasets import make_moons
import dbscan.dynamic_fdbscan
import matplotlib.pyplot as plt
import numpy as np
import alglab.dataset_base
from experiments import enhanced_adaptive_dbscan_params
def main():
    # Get the two moons dataset
    #data, labels = make_moons(n_samples=1000, random_state=0)

    base_path = "E:/dwd_truck"
    dates = ["2024-04-14"]
    truck_dataset = alglab.dataset_base.LocalTrajectoryDataset(base_folder=base_path, date_range=dates)
    data=truck_dataset.data

    # Initialise the dynamic DBSCAN data structure with the first 100 data points
    # 调整 DBSCAN 参数以适应 make_moons 数据集，您可能需要根据实际数据调整这些值。
    # 原始的 (10, 10) 对于 make_moons 数据集通常过大。
    #def __init__(self, k: int, t: int, eps: float, d: int, initial_data: Optional[np.ndarray] = None):

    Ada_eps, Ada_min, Ada_t = enhanced_adaptive_dbscan_params(truck_dataset.n, truck_dataset.d,data, use_optimization=True, verbose=True)
    print(f"Adaptive parameters: eps={Ada_eps}, min_samples={Ada_min}, t={Ada_t}")
    
    dbscan_alg = dbscan.dynamic_fdbscan.DynamicDBSCAN(Ada_min, Ada_t, Ada_eps, 2, initial_data=data[:100, :])

    # Add the remaining data points
    for i in range(100, data.shape[0]): # 使用 data.shape[0] 确保添加所有点
        dbscan_alg.add_point(data[i, :])

    # Get the final predicted labels for all data points
    predicted_labels = [dbscan_alg.get_cluster(i) for i in range(data.shape[0])]
    predicted_labels = np.array(predicted_labels) # 转换为 numpy 数组以便于索引

    print("Predicted Labels (first 20):", predicted_labels[:20]) # 仍然可以打印一些标签作为参考

    # --- 绘制聚类结果 ---
    plt.figure(figsize=(10, 7))
    unique_labels = set(predicted_labels)
    # 为每个簇（包括噪声点）生成一个独特的颜色
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    # 绘制所有点
    for k, col in zip(unique_labels, colors):
        if k == -1:  # 噪声点（标签为 -1）用黑色表示
            col = [0, 0, 0, 1]

        # 创建一个掩码，用于选择属于当前簇 k 的点
        class_member_mask = (predicted_labels == k)

        # 绘制数据点
        xy = data[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6, label=f'Cluster {k}' if k != -1 else 'Noise')

    plt.title('Dynamic DBSCAN Clustering Results on Make Moons Dataset')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()