"""
Implementation of the fast dbscan algorithm in the following paper.

https://ojs.aaai.org/index.php/AAAI/article/view/16902
"""
from typing import Set, List
import scipy as sp
import numpy as np
import random
import stag.graph
import stag.cluster
from .hash import LSHash, RoundHash

class RoundHash(object):
    """
    Create a 'hash' function based on rounding vectors.
    """

    def __init__(self, bucket_size):
        self.bucket_size = bucket_size

    def apply(self, x):
        x = np.array(x) / self.bucket_size
        x = np.floor(x)
        return hash(x.data.tobytes())



class LSHash(object):
    """Create a single lsh function as described in Lemma 2 of the paper."""

    def __init__(self, bucket_size):
        self.round_hash = RoundHash(bucket_size)
        self.eta = random.uniform(0, bucket_size)

    def apply(self, x):
        x = np.array(x) + self.eta * np.ones(x.shape)
        return self.round_hash.apply(x)


class fastDBSCAN(object):

    def __init__(self, eps=0.1, t=10, min_samples=10):
        """Initialise an instance of the fastDBSCAN data structure."""
        self.eps = eps
        self.k = min_samples
        self.t = t
        self.labels_ = []

        # Store the dataset as a dictionary from indexes to vectors
        self.data = {}

        # Keep track of the number of data points
        self.n = 0

        self.core_hash = RoundHash(2 * eps)
        self.core_buckets = {}
        self.hashes = [LSHash(2 * eps) for _ in range(self.t)]
        self.hash_buckets = [{} for _ in range(self.t)]

        self.core_points = set()
        self.g = None
        self.next_label = 0

    def hash_data(self, index):
        h_x = self.core_hash.apply(self.data[index])
        if h_x not in self.core_buckets:
            self.core_buckets[h_x] = [index]
        else:
            self.core_buckets[h_x].append(index)

        for j in range(self.t):
            hash_value = self.hashes[j].apply(self.data[index])
            if hash_value not in self.hash_buckets[j]:
                self.hash_buckets[j][hash_value] = [index]
            else:
                self.hash_buckets[j][hash_value].append(index)

    def update_core_points(self):
        """Algorithm 2 in the fast DBSCAN paper."""
        core_points = set()

        # Find the core points
        for point_array in self.core_buckets.values():
            if len(point_array) >= self.k:
                core_points.update(point_array)

        self.core_points = core_points

    def construct_graph(self):
        # Keep track of the non-core points which have already been connected
        connected_not_c = set()

        # Construct the adjacency matrix
        a = sp.sparse.lil_matrix((self.n, self.n))
        for j in range(self.t):
            for hash_value, hash_bucket in self.hash_buckets[j].items():
                core_points = list(set(hash_bucket).intersection(self.core_points))
                non_core_points = list(set(hash_bucket).difference(self.core_points))

                if len(core_points) > 0:
                    first_point = core_points[0]
                    for second_point in core_points[1:]:
                        a[first_point, second_point] = 1
                        a[second_point, first_point] = 1

                    # Handle non-core points in the bucket
                    for p in non_core_points:
                        if p not in connected_not_c:
                            a[p, first_point] = 1
                            a[first_point, p] = 1
                            connected_not_c.add(p)

        # Return the connected components of the graph
        self.g = stag.graph.Graph(a)

    def get_label_for_new_point(self, index):
        found_core_point = False
        for j in range(self.t):
            this_hash_value = self.hashes[j].apply(self.data[index])
            bucket = self.hash_buckets[j][this_hash_value]
            core_points = list(set(bucket).intersection(self.core_points))
            if len(core_points) > 0:
                first_point = core_points[0]

                # Assign the new point to the same cluster as the colliding core point.
                self.labels_.append(self.labels_[first_point])
                found_core_point = True
                break

        if not found_core_point:
            self.labels_.append(0)

    def add_point(self, point: np.ndarray):
        self.n += 1
        self.data[self.n - 1] = point
        self.hash_data(self.n - 1)
        self.get_label_for_new_point(self.n - 1)

    def remove_point(self, i: int):
        # todo
        pass

    def get_labels(self, reconstruct_graph=True):
        if reconstruct_graph:
            # Construct core point set
            self.update_core_points()

            # Construct graph
            self.construct_graph()

            # Return the clusters
            connected_components = stag.cluster.connected_components(self.g)
            self.labels_ = [0] * self.n
            next_label = 1
            for cc in connected_components:
                if len(cc) > 1:
                    for i in cc:
                        self.labels_[i] = next_label
                    next_label += 1
            self.next_label = next_label
            return self.labels_
        else:
            return self.labels_



