"""Implementation of the Dataset object for use with algpy."""
from sklearn.datasets import make_moons, fetch_openml, make_blobs
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
import numpy as np
from abc import ABC, abstractmethod
import stag.graph
import stag.random
import matplotlib.pyplot as plt
from typing import Type, Dict, Tuple, List
import pandas as pd
import os
import json
import pandas as pd
import pyarrow.parquet as pq
from typing import List, Optional


class Dataset(ABC):

    @abstractmethod
    def __init__(self, **kwargs):
        """Construct the dataset."""
        pass


class NoDataset(Dataset):
    """Use this when no dataset is needed to compare algorithms."""

    def __init__(self):
        pass

    def __str__(self):
        return "NoDataset"


class ClusterableDataset(Dataset):
    """
    A dataset which may have ground truth clusters.
    """

    def __init__(self, labels):
        self.gt_labels = labels

    def cluster_ids(self):
        if self.gt_labels is None:
            return None
        else:
            return np.unique(self.gt_labels).tolist()

    def get_cluster(self, cluster_id: int):
        cluster = []
        for id, lab in enumerate(self.gt_labels):
            if lab == cluster_id:
                cluster.append(id)
        return cluster


class GraphDataset(ClusterableDataset):
    """
    A dataset whose central data is a graph.
    """

    def __init__(self, graph: stag.graph.Graph = None, labels=None):
        """Initialise the dataset with a stag Graph. Optionally, provide ground truth
        labels for classification."""
        self.graph = graph
        self.n = 0 if graph is None else graph.number_of_vertices()
        ClusterableDataset.__init__(self, labels)


class SBMDataset(GraphDataset):
    """
    Create a graph dataset from a stochastic block model.
    """

    def __init__(self, n: int = 1000, k: int = 10, p: float = 0.5, q: float = 0.1):
        self.n = int(n)
        self.k = int(k)
        self.p = p
        self.q = q
        g = stag.random.sbm(self.n, self.k, p, q)
        labels = stag.random.sbm_gt_labels(self.n, self.k)
        GraphDataset.__init__(self, graph=g, labels=labels)


    def __repr__(self):
        return f"SBMDataset({self.n}, {self.k}, {self.p}, {self.q})"


class PointCloudDataset(ClusterableDataset):
    """
    The simplest form of dataset: the data consists of a point cloud in Euclidean space.
    This is represented internally by a numpy array.
    """

    def __init__(self, data: np.array = None, labels=None):
        """Initialise the dataset with a numpy array. Optionally, provide labels for classification."""
        self.data = np.array(data)
        self.n, self.d = data.shape
        self.k = -1 if labels is None else len(np.unique(np.asarray(labels)))
        ClusterableDataset.__init__(self, labels)

    def apply_pca(self, new_dimension: int):
        pca = PCA(n_components=new_dimension)
        self.data = pca.fit_transform(self.data)
        assert self.data.shape[1] == new_dimension
        self.d = new_dimension

    def apply_scaling(self):
        scaler = StandardScaler().fit(self.data)
        self.data = scaler.transform(self.data)

    def plot_clusters(self, labels):
        """
        If the data is two-dimensional, plot the data, colored according to the labels.
        """
        if self.d != 2:
            raise ValueError("Dataset is not two-dimensional.")

        if len(labels) != self.n:
            raise ValueError("Labels length must match number of data points.")

        labels = np.array(labels)

        # Plot the data points, colored by their cluster labels
        plt.figure(figsize=(10, 7))
        unique_labels = np.unique(labels)

        for label in unique_labels:
            cluster_data = self.data[labels == label]
            plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {label}')

        plt.grid(True)
        plt.show()


class TwoMoonsDataset(PointCloudDataset):
    """The toy two moons dataset from sklearn."""

    def __init__(self, n=1000, noise=0.07):
        """Initialise the two moons dataset. Optionally, provide the number of points, n, and the noise parameter."""
        x, y = make_moons(n_samples=n, noise=noise)
        PointCloudDataset.__init__(self, data=x, labels=y)

    def __str__(self):
        return f"TwoMoonsDataset({self.n})"


class BlobsDataset(PointCloudDataset):
    """The toy blobs dataset from sklearn."""

    def __init__(self, n=1000, d=2, k=3):
        x, y = make_blobs(n_samples=n, n_features=d, centers=k)
        PointCloudDataset.__init__(self, data=x, labels=y)

    def __str__(self):
        return f"BlobsDataset({self.n}, {self.d}, {self.k})"


class OpenMLDataset(PointCloudDataset):
    """Load pointcloud data from OpenML."""

    def __init__(self, **kwargs):
        """Initialise the dataset by downloading from openML. Accepts the same arguments as the
        sklearn fetch_openml method."""
        kwargs["as_frame"] = False
        kwargs["parser"] = "liac-arff"
        data_info = fetch_openml(**kwargs)
        if isinstance(data_info.data, pd.DataFrame):
            data_info.data = data_info.data.to_numpy()

        target = data_info.target
        if isinstance(target, pd.Series) or isinstance(target, pd.DataFrame):
            if isinstance(target.dtype, pd.CategoricalDtype):
                target = target.cat.codes.to_numpy()
            else:
                target = data_info.target.to_numpy()

        PointCloudDataset.__init__(self, data=data_info.data, labels=target)


class KnnGraphDataset(GraphDataset, PointCloudDataset):
    """A k-nearest neighbour graph dataset is both a point cloud and a graph dataset."""

    def __init__(self,
                 k: int = 10,
                 pointcloud_class: Type[PointCloudDataset] = PointCloudDataset,
                 **pointcloud_parameters):
        # Initialise this as a pointcloud dataset
        pointcloud_class.__init__(self, **pointcloud_parameters)

        # Create the k nearest neighbours graph and initialise as a graph dataset
        adj_non_symmetric = kneighbors_graph(self.data, k)
        g = stag.graph.Graph(adj_non_symmetric + adj_non_symmetric.transpose())
        GraphDataset.__init__(self, graph=g, labels=self.gt_labels)


class DynamicDataset(Dataset):

    def __init__(self, num_updates: int):
        self.num_updates: int = num_updates

    @abstractmethod
    def set_iteration(self, t):
        """
        Configure the dataset to its state after the tth iteration.
        This is used to help evaluation algorithms work for both dynamic and static datasets.
        """
        pass

    @abstractmethod
    def get_update(self, t):
        pass


class DynamicPointCloudDataset(PointCloudDataset, DynamicDataset):

    def __init__(self, data: np.array, update_schedule: List[Tuple[List[int], List[int]]], labels: np.array = None):
        # The update schedule is a list of the order in which the data points arrive to the dataset.
        # Each element in the update schedule is a tuple of insertions and deletions.
        #
        # We assume that once a point is deleted, it is never re-inserted. If you'd like to model this, you'll need to
        # include the point multiple times in the data matrix.
        self.update_schedule = update_schedule

        # If labels are provided, we compute the ground truth labels for each update to the dataset.
        self.schedule_labels = None if labels is None else []
        if labels is not None:
            current_data = set()
            for i in range(len(update_schedule)):
                deletions = self.update_schedule[i][1]
                additions = self.update_schedule[i][0]
                current_data.update(additions)
                current_data.difference_update(set(deletions))

                current_labels = [labels[j] for j in sorted(current_data)]
                self.schedule_labels.append(current_labels)

        PointCloudDataset.__init__(self, data=data, labels=labels)
        DynamicDataset.__init__(self, len(update_schedule))

    @classmethod
    def from_pointcloud(cls, pointcloud_dataset: PointCloudDataset, batch_size: int,
                        stream_by_cluster=False):
        """
        Create a dynamic point cloud dataset using a normal pointcloud dataset and streaming the data in according to
        batch size.
        """
        # 检查数据集是否为空
        if pointcloud_dataset.n == 0:
            raise ValueError("Cannot create dynamic dataset from empty pointcloud dataset")
        
        if stream_by_cluster:
            # 如果要按聚类流式处理，但没有标签，则按普通方式处理
            if pointcloud_dataset.gt_labels is None:
                print("Warning: stream_by_cluster=True but no labels available, using random order instead")
                stream_by_cluster = False
            else:
                # Stream in the data one cluster at a time
                ordered_indexes = []
                gt_labels = []
                for cluster_id in pointcloud_dataset.cluster_ids():
                    for id in pointcloud_dataset.get_cluster(cluster_id):
                        ordered_indexes.append(id)
                        gt_labels.append(cluster_id)

                reordered_data = pointcloud_dataset.data[ordered_indexes, :]
        
        if not stream_by_cluster:
            random_order = np.random.permutation(pointcloud_dataset.data.shape[0])
            reordered_data = pointcloud_dataset.data[random_order, :]
            # 修复：检查 gt_labels 是否为 None
            if pointcloud_dataset.gt_labels is not None:
                gt_labels = pointcloud_dataset.gt_labels[random_order]
            else:
                gt_labels = None

        update_schedule = []
        current_n = 0
        while current_n < pointcloud_dataset.n:
            update_schedule.append((list(range(current_n, min(current_n + batch_size, pointcloud_dataset.n))),
                                    []))
            current_n += batch_size

        return cls(reordered_data, update_schedule, labels=gt_labels)

    def set_iteration(self, t):
        self.n = self.get_n(t)
        self.gt_labels = self.get_labels(t)
        self.k = -1 if self.gt_labels is None else len(np.unique(np.asarray(self.gt_labels)))

    def get_update(self, t):
        """Get the additions and deletions corresponding to the tth update."""
        return self.update_schedule[t]

    def get_labels(self, t):
        """Get the labels corresponding to the tth update."""
        if self.schedule_labels is None:
            return None
        return self.schedule_labels[t]

    def get_n(self, t):
        """Get the number of data points after the tth update."""
        if self.schedule_labels is None:
            # 如果没有标签，通过计算累积的插入和删除来获取数据点数量
            current_data = set()
            for i in range(t + 1):
                deletions = self.update_schedule[i][1]
                additions = self.update_schedule[i][0]
                current_data.update(additions)
                current_data.difference_update(set(deletions))
            return len(current_data)
        else:
            return len(self.get_labels(t))
#本地类
class LocalTrajectoryDataset(PointCloudDataset):
    def __init__(self,
                 base_folder: str,
                 date_range: List[str],
                 sample_ratio: float = 0.005,
                 cache_file: Optional[str] = None):
        """
        构建本地轨迹数据集（继承自 PointCloudDataset）

        :param base_folder: 本地基础路径，如 'E:/dwd_truck'
        :param date_range: 日期列表，例如 ['2024-04-10', ..., '2024-04-18']
        :param sample_ratio: 抽样比例，默认1%
        :param cache_file: 可选缓存路径（.npy 文件）
        """
        self.base_folder = base_folder
        self.date_range = date_range
        self.sample_ratio = sample_ratio
        self.cache_file = cache_file or "trajectory_data.npy"

        if os.path.exists(self.cache_file):
            print(f"✅ 加载缓存数据: {self.cache_file}")
            data = np.load(self.cache_file)
            labels = None
        else:
            print("🚀 第一次运行，加载并处理本地数据")
            data = self._load_all_days()
            np.save(self.cache_file, data)
            print(f"✅ 已保存缓存数据到 {self.cache_file}")
            labels = None

        # 调用父类构造函数
        super().__init__(data=data, labels=labels)

    def _load_all_days(self):
        dfs = []
        for date in self.date_range:
            folder_name = f"publish_date={date}"
            folder_path = os.path.join(self.base_folder, folder_name)
            if os.path.exists(folder_path):
                df = self._load_day_data(folder_path)
                if not df.empty:
                    df = df.sample(frac=self.sample_ratio, random_state=42)
                    dfs.append(df)
        full_df = pd.concat(dfs, ignore_index=True)
        full_df = self._preprocess(full_df)
        return full_df[['lon', 'lat']].to_numpy()

    def _load_day_data(self, folder_path):
        file_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.c000')]
        tables = []
        for file in file_list:
            try:
                table = pq.read_table(file, columns=['plate_num', 'published_at', 'speed', 'coord'])
                tables.append(table)
            except Exception:
                continue
        if tables:
            return pd.concat([t.to_pandas() for t in tables], ignore_index=True)
        return pd.DataFrame()

    def _preprocess(self, df):
        df = df.dropna(subset=['coord', 'plate_num', 'published_at'])
        df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
        df = df.dropna(subset=['published_at'])

        coords = df['coord'].apply(json.loads)
        df['lon'] = coords.map(lambda x: x.get('lon'))
        df['lat'] = coords.map(lambda x: x.get('lat'))

        df['speed'] = pd.to_numeric(df['speed'], errors='coerce')
        df = df.dropna(subset=['speed', 'lon', 'lat'])

        # 过滤经纬度和速度范围
        df = df[(df['lon'] >= 102) & (df['lon'] <= 105) & (df['lat'] >= 29) & (df['lat'] <= 32)]
        df = df[(df['speed'] >= 0) & (df['speed'] <= 120)]
        return df.drop_duplicates()
