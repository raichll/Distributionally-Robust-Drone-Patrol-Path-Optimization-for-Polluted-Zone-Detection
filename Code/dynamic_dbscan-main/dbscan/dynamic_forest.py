"""
This file provides an interface and implementation of the dynamic tree data structure.
"""
from typing import List
from abc import abstractmethod, ABC
import random

random.seed(0)

class DynamicForest(ABC):

    def __init__(self):
        """Initialise the tree with no nodes."""
        pass

    @abstractmethod
    def add_node(self) -> int:
        """
        Add a node to the forest, unconnected to any other tree.

        Returns an index which should be used to refer to the node in subsequent method calls.
        """
        pass

    @abstractmethod
    def link(self, i: int, j: int):
        """
        Link the nodes i and j, making i the parent of j.

        Throws an exception if linking i and j would introduce a cycle into the forest.
        """
        pass

    @abstractmethod
    def cut(self, i: int, j: int):
        """
        Remove the edge between nodes i and j.

        Throws an exception if there is no edge between i and j.
        """
        pass

    @abstractmethod
    def root(self, i: int) -> int:
        """
        Return the index of the node at the root of the tree containing i.
        """
        pass

    @abstractmethod
    def remove_node(self, i: int):
        """
        Remove the node i from the forest.

        Throws an exception if the degree of node i is not 0.
        """
        pass

    @abstractmethod
    def neighbours(self, i: int) -> List[int]:
        """
        Return the neighbours of node i.
        """
        pass


class NaiveDynamicForest(DynamicForest):
    """A naive implementation of the dynamic forest data structure."""

    class Node(object):
        """A node in the naive forest."""

        def __init__(self, index):
            self.index = index
            self.parent = None
            self.neighbours = set()

    def __init__(self):
        """
        Initialise the tree with no nodes.
        """
        super().__init__()
        self.next_index = 0
        self.nodes = {}

    @staticmethod
    def make_root(n: Node):
        """Make n the root of its tree."""
        current_node = n
        next_node = n.parent
        while next_node is not None:
            new_next_node = next_node.parent
            next_node.parent = current_node
            current_node = next_node
            next_node = new_next_node
        n.parent = None

    def add_node(self):
        """
        Add a node to the forest, unconnected to any other tree.

        Returns an index which should be used to refer to the node in subsequent method calls.
        """
        new_node_index = self.next_index
        self.next_index += 1
        self.nodes[new_node_index] = self.Node(new_node_index)
        return new_node_index

    def link(self, i: int, j: int):
        if self.root(i) == self.root(j):
            raise ValueError(f"Nodes {i} and {j} are already in the same tree.")

        node_i = self.nodes[i]
        node_j = self.nodes[j]

        self.make_root(node_j)
        assert node_j.parent is None
        node_j.parent = node_i
        node_j.neighbours.add(i)
        node_i.neighbours.add(j)

    def cut(self, i: int, j: int):
        node_i = self.nodes[i]
        node_j = self.nodes[j]

        if i not in node_j.neighbours:
            raise ValueError(f"There is no edge between nodes {i} and {j}.")

        if node_i.parent == node_j:
            node_i.parent = None
        else:
            assert node_j.parent == i
            node_j.parent = None

        node_i.neighbours.remove(j)
        node_j.neighbours.remove(i)

    def root(self, i: int) -> int:
        if i not in self.nodes:
            raise ValueError(f"There is no node {i}.")
        current_node = self.nodes[i]
        while current_node.parent is not None:
            current_node = current_node.parent
        return current_node.index

    def remove_node(self, i: int):
        if len(self.nodes[i].neighbours) != 0:
            raise ValueError(f"The degree of node {i} is not 0.")
        del self.nodes[i]

    def neighbours(self, i: int) -> List[int]:
        if i not in self.nodes:
            raise ValueError(f"There is no node {i}.")
        return self.nodes[i].neighbours


class EulerTourDynamicForest(DynamicForest):
    """An implementation of the dynamic forest data structure, using Euler tour skip lists."""

    class ListElement(object):
        """An item in the skip list."""

        def __init__(self, index: int):
            # Sample the skip list height for this element
            self.height = 0
            while random.random() > 0.5:
                self.height += 1

            # Initialise the forward and backward list pointers
            self.next_pointers = {i: None for i in range(self.height + 1)}
            self.prev_pointers = {i: None for i in range(self.height + 1)}

            self.index = index

        def next(self):
            """Get the next element in this list"""
            return self.next_pointers[0]

        def prev(self):
            """Get the previous element in this list"""
            return self.prev_pointers[0]

        def get_first_element(self):
            """
            Get the first element in this list.

            O(log(n)) time complexity.
            """
            current_node = self
            while True:
                found = False
                for h in range(current_node.height, -1, -1):
                    if current_node.prev_pointers[h] is not None:
                        current_node = current_node.prev_pointers[h]
                        found = True
                        break

                if not found:
                    return current_node

        def get_last_element(self):
            """
            Get the last element in this list.

            O(log(n)) tiem complexity.
            """
            latest_node = self
            current_node = self
            while current_node is not None:
                latest_node = current_node

                found = False
                for h in range(current_node.height, -1, -1):
                    if current_node.next_pointers[h] is not None:
                        current_node = current_node.next_pointers[h]
                        found = True
                        break

                if not found:
                    current_node = None
            return latest_node

        def get_forward_height(self):
            """
            Returns the maximum height of the list traversing in the forward direction.

            Running time O(log(n))
            """
            current_node = self
            max_height = 0
            while current_node is not None:
                if current_node.height > max_height:
                    max_height = current_node.height
                next_node = current_node.next_pointers[current_node.height]
                current_node = next_node
            return max_height

        def get_backward_height(self):
            """
            Returns the maximum height of the list traversing in the backward direction.

            Running time O(log(n))
            """
            current_node = self
            max_height = 0
            while current_node is not None:
                if current_node.height > max_height:
                    max_height = current_node.height
                next_node = current_node.prev_pointers[current_node.height]
                current_node = next_node
            return max_height

        def link_backwards_at_height(self, h, other_list_start):
            """Link this list to the start of another list at height h."""
            this_list_node = self
            while this_list_node.height < h:
                this_list_node = this_list_node.prev_pointers[this_list_node.height]

            other_list_node = other_list_start
            while other_list_node.height < h:
                other_list_node = other_list_node.next_pointers[other_list_node.height]

            this_list_node.next_pointers[h] = other_list_node
            other_list_node.prev_pointers[h] = this_list_node

        def link_forwards_at_height(self, h, other_list_end):
            """Link the end of another list to the next node in this list, at height h."""
            this_list_node = self.next_pointers[0]
            while this_list_node is not None and this_list_node.height < h:
                this_list_node = this_list_node.next_pointers[this_list_node.height]

            other_list_node = other_list_end
            while other_list_node is not None and other_list_node.height < h:
                other_list_node = other_list_node.prev_pointers[other_list_node.height]

            if this_list_node is not None:
                this_list_node.prev_pointers[h] = other_list_node
            if other_list_node is not None:
                other_list_node.next_pointers[h] = this_list_node

        def insert_after(self, other_list_start, other_list_end):
            max_height = other_list_start.get_forward_height()

            self_forward_height = self.get_forward_height()
            self_backward_height = self.get_backward_height()

            # Link the maximum height links first
            for h in range(max_height, -1, -1):
                # Link at height h
                if h > 0:
                    if self_backward_height >= h:
                        self.link_backwards_at_height(h, other_list_start)

                    if self_forward_height >= h:
                        self.link_forwards_at_height(h, other_list_end)
                else:
                    # To link at height 0, we just update forward and backward pointers directly
                    other_list_end.next_pointers[0] = self.next_pointers[0]
                    if self.next_pointers[0] is not None:
                        self.next_pointers[0].prev_pointers[0] = other_list_end
                    self.next_pointers[0] = other_list_start
                    other_list_start.prev_pointers[0] = self


        def split_after(self):
            """Split the Euler tour list after this element"""
            next_node = self.next()

            if next_node is None:
                return

            # Set the forward pointers to None
            current_node = self
            max_height_cut = -1
            while current_node is not None:
                if current_node.height > max_height_cut:
                    for h in range(max_height_cut + 1, current_node.height + 1):
                        current_node.next_pointers[h] = None
                    max_height_cut = current_node.height
                current_node = current_node.prev_pointers[current_node.height]

            # Set the backward pointers of the next node to None
            current_node = next_node
            max_height_cut = -1
            while current_node is not None:
                if current_node.height > max_height_cut:
                    for h in range(max_height_cut + 1, current_node.height + 1):
                        current_node.prev_pointers[h] = None
                    max_height_cut = current_node.height
                current_node = current_node.next_pointers[current_node.height]

    class Node(object):
        """A node in the forest."""

        def __init__(self, index, euler_tour_elem):
            self.index = index
            self.neighbours = set()
            self.euler_tour_elem = euler_tour_elem

    def __init__(self):
        """
        Initialise the tree with no nodes.
        """
        super().__init__()
        self.next_index = 0
        self.nodes = {}

        # Keep track of all of the edges in the forest.
        # Each edge stores two pointers to the Euler tour list elements which represent this edge
        self.edges = {}

    def print_forest(self):
        self.drawn = set()
        for n in self.nodes.values():
            if n.index not in self.drawn:
                current_elem = n.euler_tour_elem.get_first_element()
                current_height = current_elem.height + 1

                while current_elem is not None:
                    this_height = current_elem.height + 1
                    current_height = max(current_height, this_height)
                    print(f"[{current_elem.index}] {''.join(['*' for _ in range(this_height)])}{''.join(['|' for _ in range(current_height - this_height)])}")
                    print(f"    {''.join(['|' for _ in range(current_height)])}")
                    self.drawn.add(current_elem.index)
                    current_elem = current_elem.next()

                print(f"[*] {''.join(['*' for _ in range(current_height)])}")
                print()

    def make_root(self, i: int):
        """Make node i the root of its tree. This makes its elements the start and end of the euler tour"""
        n = self.nodes[i]
        ets_elem = n.euler_tour_elem

        # We will need the previous element, the root element, and the last element
        prev_elem = ets_elem.prev()
        root_elem = ets_elem.get_first_element()
        last_elem = ets_elem.get_last_element()

        if prev_elem is None:
            # If this element is already the root of the tree, do nothing
            return
        else:
            # Split the euler tour tree before the current node element
            prev_elem.split_after()

            # Add the original root element after the original last element
            last_elem.insert_after(root_elem, prev_elem)

    def add_node(self):
        """
        Add a node to the forest, unconnected to any other tree.

        Returns an index which should be used to refer to the node in subsequent method calls.
        """
        new_node_index = self.next_index
        self.next_index += 1

        # Create a little Euler tour list for the new node
        euler_tour_elem = self.ListElement(new_node_index)

        self.nodes[new_node_index] = self.Node(new_node_index, euler_tour_elem)
        return new_node_index

    def link(self, i: int, j: int):
        if self.root(i) == self.root(j):
            return

        node_i_elem = self.nodes[i].euler_tour_elem
        node_j_elem = self.nodes[j].euler_tour_elem

        i_singleton = node_i_elem.next() is None and node_i_elem.prev() is None
        j_singleton = node_j_elem.next() is None and node_j_elem.prev() is None

        self.make_root(j)
        node_j_last_elem = node_j_elem.get_last_element()

        # Get the edge currently represented by node_i
        old_node_i_edge = None
        if not i_singleton:
            if node_i_elem.next():
                old_node_i_edge = (i, node_i_elem.next().index)
            else:
                old_node_i_edge = (i, node_i_elem.get_first_element().index)

        # Need to add two new nodes to the ETS starting at node j:
        #  an element for node j and an element for node i
        new_j_elem = node_j_elem
        new_i_elem = node_i_elem
        if not j_singleton:
            new_j_elem = self.ListElement(j)
            node_j_last_elem.insert_after(new_j_elem, new_j_elem)
        if not i_singleton:
            new_i_elem = self.ListElement(i)
            new_j_elem.insert_after(new_i_elem, new_i_elem)

        # Now, insert the sequence starting at node j into the sequence
        # containing node i
        new_last_j_elem = node_j_elem.get_last_element()
        node_i_elem.insert_after(node_j_elem, new_last_j_elem)

        # Update the adjacency lists
        self.nodes[j].neighbours.add(i)
        self.nodes[i].neighbours.add(j)

        # Update the edge lists
        self.edges[(i, j)] = node_i_elem
        self.edges[(j, i)] = new_j_elem

        if not i_singleton:
            self.edges[old_node_i_edge] = new_i_elem

    def cut(self, i: int, j: int):
        if i not in self.nodes[j].neighbours:
            raise ValueError(f"There is no edge between nodes {i} and {j}.")

        i_elem = self.edges[(i, j)]
        j_elem = self.edges[(j, i)]

        other_i = j_elem.next()
        j_is_end = False
        if other_i is None:
            j_is_end = True
            other_i = j_elem.get_first_element()
        other_j = i_elem.next()
        i_is_end = False
        if other_j is None:
            i_is_end = True
            other_j = i_elem.get_first_element()

        elem_after_i = i_elem.next()
        elem_after_j = j_elem.next()

        # Cut the sequence in the two given places
        i_elem.split_after()
        j_elem.split_after()

        # Join the sequence together again in the correct way
        if elem_after_i is not None and elem_after_i.get_last_element() == j_elem:
            # i_elem came first in the list
            # split i_elem off the end of its list
            elem_before_i = i_elem.prev()
            if elem_before_i:
                elem_before_i.split_after()
                if not j_is_end:
                    elem_before_i.insert_after(other_i, other_i.get_last_element())

            # split j_elem off the end of its list
            elem_before_j = j_elem.prev()
            if elem_before_j:
                elem_before_j.split_after()
        else:
            # j_elem came first in the list
            # split j_elem off the end of its list
            elem_before_j = j_elem.prev()
            if elem_before_j:
                elem_before_j.split_after()
                if not i_is_end:
                    elem_before_j.insert_after(other_j, other_j.get_last_element())

            # split i_elem off the end of its list
            elem_before_i = i_elem.prev()
            if elem_before_i:
                elem_before_i.split_after()

        # Possibly delete ETS elements unless they are singletons
        if i_elem != other_i:
            self.nodes[i].euler_tour_elem = other_i
            assert other_i.index == i
            del i_elem
        if j_elem != other_j:
            self.nodes[j].euler_tour_elem = other_j
            assert other_j.index == j
            del j_elem

        # Update the adjacency lists
        self.nodes[i].neighbours.remove(j)
        self.nodes[j].neighbours.remove(i)

        # Update the edge data structure
        del self.edges[(i, j)]
        del self.edges[(j, i)]

    def root(self, i: int) -> int:
        if i not in self.nodes:
            raise ValueError(f"There is no node {i}.")

        return self.nodes[i].euler_tour_elem.get_first_element().index

    def remove_node(self, i: int):
        if len(self.nodes[i].neighbours) != 0:
            raise ValueError(f"The degree of node {i} is not 0.")
        del self.nodes[i]

    def neighbours(self, i: int) -> List[int]:
        if i not in self.nodes:
            raise ValueError(f"There is no node {i}.")
        return self.nodes[i].neighbours

