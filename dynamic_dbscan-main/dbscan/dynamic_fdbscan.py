import numpy as np
import random
from collections import defaultdict
from typing import Optional, List

from .hash import LSHash
from .dynamic_forest import EulerTourDynamicForest


class DynamicDBSCAN(object):

    def __init__(self, k: int, t: int, eps: float, d: int, initial_data: Optional[np.ndarray] = None):
        self.k = k
        self.eps = eps
        self.t = t

        # We will store the dataset as a dictionary from indices to vectors
        self.data = {}

        # Sets store indices of data points
        self.core_points = set()
        self.non_core_points = set()

        # Initialise the dynamic forest data structure
        self.forest = EulerTourDynamicForest()

        # Initialise the hash functions. For each has function, we maintain a dictionary of hash buckets.
        # Each dictionary in the hash buckets array corresponds to a hash function and uses integers as keys and
        # lists of point indices as values.
        self.hash_functions: List[LSHash] = [LSHash(2 * self.eps) for _ in range(self.t)]
        self.hash_buckets = [{} for _ in range(self.t)]

        # Hash the initial data into the hash buckets
        if initial_data is not None:
            for i in range(initial_data.shape[0]):
                point = initial_data[i, :]

                # Add this point as a disconnected point in the dynamic forest
                # We let the dynamic forest specify what index we should use for this node.
                index = self.forest.add_node()

                self.data[index] = point
                self.non_core_points.add(index)
                for j in range(self.t):
                    this_hash_value = self.hash_functions[j].apply(point)
                    if this_hash_value not in self.hash_buckets[j]:
                        self.hash_buckets[j][this_hash_value] = [index]
                    else:
                        self.hash_buckets[j][this_hash_value].append(index)

        # Find the core points
        for j in range(self.t):
            for hash_value, bucket in self.hash_buckets[j].items():
                if len(bucket) >= self.k:
                    self.core_points.update(bucket)
                    self.non_core_points.difference_update(set(bucket))

        for i in self.core_points:
            self.link_core_point(i)

        for i in self.non_core_points:
            self.link_non_core_point(i)

    def link_core_point(self, point_index: int, new_point=False):
        non_core_points_to_relink = set()
        for j in range(self.t):
            this_hash_value = self.hash_functions[j].apply(self.data[point_index])
            bucket = self.hash_buckets[j][this_hash_value]

            if new_point:
                # If this is a new point, we know that we just need to link it to the
                # core point with largest index
                for i in reversed(bucket):
                    if i in self.core_points and i != point_index:
                        self.forest.link(point_index, i)
                        break
                    else:
                        non_core_points_to_relink.add(i)
            else:
                # We will link point_index to the point with the largest index not more than its own
                # In this way, the graph on the core points in the bucket will always be a path
                previous_point = None
                next_point = None
                for i in bucket:
                    if i not in self.core_points:
                        non_core_points_to_relink.add(i)
                        continue
                    if i < point_index:
                        if previous_point is None or previous_point < i:
                            previous_point = i
                    if i > point_index:
                        if next_point is None or next_point > i:
                            next_point = i

                # If the previous point and next point are connected in the forest, disconnect them
                must_link_both = False
                if (next_point is not None and previous_point is not None and
                        next_point in self.forest.nodes[previous_point].neighbours):
                    self.forest.cut(previous_point, next_point)
                    must_link_both = True

                # Add an edge between this_point and its neighbours only if they are not already in the same
                # connected component.
                if must_link_both or previous_point is not None:
                    self.forest.link(point_index, previous_point)
                if must_link_both:
                    self.forest.link(point_index, next_point)

        # Update non-core points
        for i in non_core_points_to_relink:
            self.link_non_core_point(i)

    def link_non_core_point(self, point_index: int):
        if len(self.forest.nodes[point_index].neighbours) > 0:
            # If this point is already linked, there is nothing to do
            return

        added_edge = False
        for j in range(self.t):
            hash_value = self.hash_functions[j].apply(self.data[point_index])
            hash_bucket = self.hash_buckets[j][hash_value]

            # Check for core points in this hash bucket
            for k in hash_bucket:
                if k in self.core_points:
                    self.forest.link(point_index, k)
                    added_edge = True
                    break

            if added_edge:
                break

        # Check that we have only added one edge for this non-core point
        assert len(self.forest.neighbours(point_index)) <= 1

    def unlink_non_core_point(self, point_index: int):
        if len(self.forest.nodes[point_index].neighbours) > 0:
            ns = list(self.forest.nodes[point_index].neighbours)
            for other_point in ns:
                self.forest.cut(other_point, point_index)

    def add_point(self, point: np.ndarray):
        # Add this point as a disconnected point in the dynamic forest
        # We let the dynamic forest specify what index we should use for this node.
        index = self.forest.add_node()
        self.data[index] = point

        # Get the set of new core points
        c_prime = set()
        for j in range(self.t):
            this_hash_value = self.hash_functions[j].apply(point)
            if this_hash_value not in self.hash_buckets[j]:
                self.hash_buckets[j][this_hash_value] = [index]
            else:
                self.hash_buckets[j][this_hash_value].append(index)

            # Check whether the new point is a core point
            if len(self.hash_buckets[j][this_hash_value]) > self.k:
                c_prime.add(index)
            if len(self.hash_buckets[j][this_hash_value]) == self.k:
                c_prime.update(self.hash_buckets[j][this_hash_value])

        # Remove elements of c_prime that are already in C
        c_prime = c_prime.difference(self.core_points)
        self.core_points.update(c_prime)
        self.non_core_points.difference_update(c_prime)

        # If the new point is not a core point, we just process it like any non-core point
        if len(c_prime) == 0:
            self.non_core_points.add(index)
            self.link_non_core_point(index)

        # If there are new core points, process them. We first disconnect them, and then add them as
        # core points.
        for new_core_point in c_prime:
            self.unlink_non_core_point(new_core_point)
            self.link_core_point(new_core_point, new_point=(len(c_prime) == 1))

        return index

    def remove_point(self, index: int):
        # todo
        pass

    def get_cluster(self, index: int):
        if len(self.forest.nodes[index].neighbours) == 0:
            return 0
        else:
            return self.forest.root(index)