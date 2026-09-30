"""
An example script demonstrating the use of the dynamic forest data structure.
"""
import dbscan.dynamic_forest


def main():
    # Create a new dynamic forest data structure
    forest = dbscan.dynamic_forest.EulerTourDynamicForest()

    # Add 6 nodes to the forest
    node_ids = [forest.add_node() for _ in range(6)]

    # Connect nodes 0, 2, and 4
    forest.link(0, 2)
    forest.link(2, 4)
    forest.link(4, 0) # this will not add an edge as it would introduce a cycle

    # Connect nodes 1, 3, and 5
    forest.link(1, 3)
    forest.link(3, 5)
    forest.link(5, 1) # this will not add an edge as it would introduce a cycle

    # Identify the connected component containing each node
    ccs = [forest.root(i) for i in node_ids]
    print(ccs)

    # Now, link nodes 2 and 5 - this connects the whole graph
    forest.link(2, 5)

    # Printing the connected components for each node demonstrates
    # that the forest is connected
    ccs = [forest.root(i) for i in node_ids]
    print(ccs)

    # Unlink nodes 0 and 2 - this disconnects the graph again.
    forest.cut(0, 2)

    # Showing the connected components demonstrates the new structure
    ccs = [forest.root(i) for i in node_ids]
    print(ccs)


if __name__ == '__main__':
    main()
