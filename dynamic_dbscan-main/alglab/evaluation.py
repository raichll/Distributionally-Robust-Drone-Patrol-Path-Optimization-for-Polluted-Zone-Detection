"""
Methods for evaluating the performance of an algorithm.
"""
from typing import Callable, Type
import numpy as np
import alglab.algorithm
import alglab.dataset
import stag.cluster
import stag.graph
import scipy.sparse.linalg
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, accuracy_score


class Evaluator(object):

    def __init__(self,
                 name: str,
                 implementation: Callable,
                 alg_result_type: Type = None,
                 dataset_class: Type[alglab.dataset.Dataset] = alglab.dataset.NoDataset):
        """Define a method of evaluating an algorithm. Specify the evaluator implementation as
        well as the expected result type of the algorithm to be evaluated and the type of the dataset
        the algorithm should be applied to."""
        self.implementation = implementation
        self.alg_result_type = alg_result_type
        self.dataset_class = dataset_class
        self.name = name

    def apply(self, dataset: alglab.dataset.Dataset, alg_result):
        if not isinstance(dataset, self.dataset_class):
            raise TypeError(f"Expected dataset type to be {self.dataset_class}, got {type(dataset)}.")

        if not isinstance(alg_result, self.alg_result_type):
            raise TypeError(f"Expected alg_result type to be {self.alg_result_type} but got {type(alg_result)}.")

        if self.dataset_class is not alglab.dataset.NoDataset:
            result = self.implementation(dataset, alg_result)
        else:
            result = self.implementation(alg_result)

        return result

    def __str__(self):
        return self.name

# -----------------------------------------------------------------------------
# Clustering Evaluation
# -----------------------------------------------------------------------------


def __ari_impl(data: alglab.dataset.ClusterableDataset, labels):
    if data.gt_labels is not None:
        return adjusted_rand_score(data.gt_labels, labels)
    else:
        raise ValueError('No ground truth labels provided.')


adjusted_rand_index = Evaluator('adjusted_rand_index',
                                __ari_impl,
                                alg_result_type=np.ndarray,
                                dataset_class=alglab.dataset.ClusterableDataset)


def __nmi_impl(data: alglab.dataset.ClusterableDataset, labels):
    if data.gt_labels is not None:
        return normalized_mutual_info_score(data.gt_labels, labels)
    else:
        raise ValueError('No ground truth labels provided.')


normalised_mutual_info = Evaluator('normalised_mutual_info',
                                   __nmi_impl,
                                   alg_result_type=np.ndarray,
                                   dataset_class=alglab.dataset.ClusterableDataset)


def __acc_impl(data: alglab.dataset.ClusterableDataset, labels):
    if data.gt_labels is not None:
        return accuracy_score(data.gt_labels, labels)
    else:
        raise ValueError('No ground truth labels provided.')


accuracy = Evaluator('accuracy',
                     __acc_impl,
                     alg_result_type=np.ndarray,
                     dataset_class=alglab.dataset.ClusterableDataset)


# -----------------------------------------------------------------------------
# Helpful hacks
# -----------------------------------------------------------------------------
def __n_impl(data: alglab.dataset.PointCloudDataset, _):
    return data.n


dataset_size = Evaluator('n',
                         __n_impl,
                         alg_result_type=object,
                         dataset_class=alglab.dataset.PointCloudDataset)


# -----------------------------------------------------------------------------
# Graph Evaluation
# -----------------------------------------------------------------------------

def __num_vertices_impl(_: alglab.dataset.Dataset, graph: stag.graph.Graph):
    return graph.number_of_vertices()


num_vertices = Evaluator('number_of_vertices',
                         __num_vertices_impl,
                         alg_result_type=stag.graph.Graph,
                         dataset_class=alglab.dataset.Dataset)


def __avg_degree_impl(_: alglab.dataset.Dataset, graph: stag.graph.Graph):
    return graph.average_degree()


avg_degree = Evaluator('average_degree',
                       __avg_degree_impl,
                       alg_result_type=stag.graph.Graph,
                       dataset_class=alglab.dataset.Dataset)


def __normalised_laplacian_second_eigenvalue_impl(_: alglab.dataset.Dataset, graph: stag.graph.Graph):
    lap = graph.normalised_laplacian().to_scipy()
    eigs, _ = scipy.sparse.linalg.eigsh(lap, which='SM', k=2)
    return eigs[1]


normalised_laplacian_second_eigenvalue = Evaluator('lap_second_eigenvalue',
                                                   __normalised_laplacian_second_eigenvalue_impl,
                                                   alg_result_type=stag.graph.Graph,
                                                   dataset_class=alglab.dataset.Dataset)
