"""
In this file, we specify the algorithms which will be compared in our experiments.
"""
import numpy as np
import sklearn.cluster

import alglab.algorithm
import alglab.dataset

from dbscan.dynamic_fdbscan import DynamicDBSCAN
from dbscan.fastdbscan import fastDBSCAN


# ----------------------------------------------------------------------------------------------------------------------
# Variations on Esfandiari et al. (2021)
# ----------------------------------------------------------------------------------------------------------------------
def __fdbscan_dynamic_naive_impl(data: alglab.dataset.DynamicPointCloudDataset, eps=0.07, min_samples=10, t=10):
    """
    This is the naive 'dynamic' version of the fast dbscan algorithm, in which we recompute the full
    graph at every iteration.
    """
    dbscan_alg = fastDBSCAN(eps=eps, min_samples=min_samples, t=t)

    for iteration in range(data.num_updates):
        data.set_iteration(iteration)
        insertions, deletions = data.get_update(iteration)

        for new_point in insertions:
            dbscan_alg.add_point(data.data[new_point,:])
        for old_point in deletions:
            dbscan_alg.remove_point(old_point)

        yield np.asarray(dbscan_alg.get_labels())


fdbscan_dynamic_naive = alglab.algorithm.Algorithm("fdbscan_dynamic_naive",
                                                   __fdbscan_dynamic_naive_impl,
                                                   return_type=np.ndarray,
                                                   parameter_names=["eps", "min_samples", "t"],
                                                   dataset_class=alglab.dataset.DynamicPointCloudDataset)


def __fdbscan_dynamic_noncore_impl(data: alglab.dataset.DynamicPointCloudDataset, eps=0.07, min_samples=10, t=10,
                                   burnin=5):
    dbscan_alg = fastDBSCAN(eps=eps, min_samples=min_samples, t=t)

    for iteration in range(data.num_updates):
        data.set_iteration(iteration)
        insertions, deletions = data.get_update(iteration)

        for new_point in insertions:
            dbscan_alg.add_point(data.data[new_point,:])
        for old_point in deletions:
            dbscan_alg.remove_point(old_point)

        if iteration <= burnin:
            yield np.asarray(dbscan_alg.get_labels(reconstruct_graph=True))
        else:
            yield np.asarray(dbscan_alg.get_labels(reconstruct_graph=False))


fdbscan_dynamic_noncore = alglab.algorithm.Algorithm("fdbscan_dynamic_noncore",
                                                     __fdbscan_dynamic_noncore_impl,
                                                     return_type=np.ndarray,
                                                     parameter_names=["eps", "min_samples", "t", "burnin"],
                                                     dataset_class=alglab.dataset.DynamicPointCloudDataset)


# ----------------------------------------------------------------------------------------------------------------------
# Our new Dynamic DBSCAN algorithm
# ----------------------------------------------------------------------------------------------------------------------
def __dynamic_dbscan_init_impl(data: alglab.dataset.PointCloudDataset, eps=0.07, min_samples=10, t=10):
    dynamic_dbscan = DynamicDBSCAN(min_samples, t, eps, initial_data=data.data)
    yield np.asarray([dynamic_dbscan.get_cluster(i) for i in range(data.n)])


dynamic_dbscan_init_alg = alglab.algorithm.Algorithm("dynamic_dbscan_static",
                                                     __dynamic_dbscan_init_impl,
                                                     return_type=np.ndarray,
                                                     parameter_names=["eps", "min_samples", "t"],
                                                     dataset_class=alglab.dataset.PointCloudDataset)


def __dynamic_dbscan_impl(data: alglab.dataset.DynamicPointCloudDataset,eps=0.07, min_samples=10, t=10, d=2):
    dynamic_dbscan = DynamicDBSCAN(min_samples, t, eps, d)

    for iteration in range(data.num_updates):
        data.set_iteration(iteration)
        insertions, deletions = data.get_update(iteration)

        for new_point in insertions:
            dynamic_dbscan.add_point(data.data[new_point,:])
        for old_point in deletions:
            dynamic_dbscan.remove_point(old_point)

        yield np.asarray([dynamic_dbscan.get_cluster(i) for i in dynamic_dbscan.data.keys()])


dynamic_dbscan_alg = alglab.algorithm.Algorithm("dynamic_dbscan",
                                                __dynamic_dbscan_impl,
                                                return_type=np.ndarray,
                                                parameter_names=["eps", "min_samples", "t", 'd'],
                                                dataset_class=alglab.dataset.DynamicPointCloudDataset)

# ----------------------------------------------------------------------------------------------------------------------
# Our new Ada Dynamic DBSCAN algorithm
# ----------------------------------------------------------------------------------------------------------------------


def Ada__dynamic_dbscan_impl(data: alglab.dataset.DynamicPointCloudDataset, eps=0.07, min_samples=10, t=10, d=2):
    """
    This is an adaptive variant copy of the dynamic_dbscan algorithm, duplicated for further customization.
    """
    dynamic_dbscan = DynamicDBSCAN(min_samples, t, eps, d)

    for iteration in range(data.num_updates):
        data.set_iteration(iteration)
        insertions, deletions = data.get_update(iteration)

        for new_point in insertions:
            dynamic_dbscan.add_point(data.data[new_point, :])
        for old_point in deletions:
            dynamic_dbscan.remove_point(old_point)

        yield np.asarray([dynamic_dbscan.get_cluster(i) for i in dynamic_dbscan.data.keys()])


Ada_dynamic_dbscan_alg = alglab.algorithm.Algorithm("Ada_dynamic_dbscan",
                                                    Ada__dynamic_dbscan_impl,
                                                    return_type=np.ndarray,
                                                    parameter_names=["eps", "min_samples", "t", "d"],
                                                    dataset_class=alglab.dataset.DynamicPointCloudDataset)

# ----------------------------------------------------------------------------------------------------------------------
# Variations on sklearn DBSCAN
# ----------------------------------------------------------------------------------------------------------------------
def __dbscan_impl(data: alglab.dataset.PointCloudDataset, eps=0.1, min_samples=10):
    dbscan_alg = sklearn.cluster.DBSCAN(eps=eps, min_samples=min_samples)
    dbscan_alg.fit(data.data)
    yield np.asarray([lab + 1 for lab in dbscan_alg.labels_])


sklearn_static_alg = alglab.algorithm.Algorithm("sklearn_static",
                                                __dbscan_impl,
                                                return_type=np.ndarray,
                                                parameter_names=["eps", "min_samples"],
                                                dataset_class=alglab.dataset.PointCloudDataset)


def __sklearn_dynamic_impl(data: alglab.dataset.DynamicPointCloudDataset, eps=0.1, min_samples=10):
    """
    This is the naive 'dynamic' verswion of the fast dbscan algorithm, in which we recompute the full
    algorithm at every iteration.
    """
    current_dataset = []
    for iteration in range(data.num_updates):
        data.set_iteration(iteration)
        insertions, deletions = data.get_update(iteration)
        current_dataset.extend(data.data[i, :] for i in insertions)

        dbscan_alg = sklearn.cluster.DBSCAN(eps=eps, min_samples=min_samples)
        dbscan_alg.fit(current_dataset)
        yield np.asarray([lab + 1 for lab in dbscan_alg.labels_])


sklearn_dynamic_alg = alglab.algorithm.Algorithm("sklearn_dynamic",
                                                 __sklearn_dynamic_impl,
                                                 return_type=np.ndarray,
                                                 parameter_names=["eps", "min_samples"],
                                                 dataset_class=alglab.dataset.DynamicPointCloudDataset)
