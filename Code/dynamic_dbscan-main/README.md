# Dynamic DBSCAN with Euler Tour Sequences

This directory contains the code required to reproduce the experiments in the paper
"Dynamic DBSCAN with Euler Tour Sequences", published in AISTATS'25.

The repository contains three elements which could be of interest:

* an implementation of the dynamic forest data structure, using Euler tour sequences based on the work of [1] and [2].
* an implementation of the dynamic DBSCAN data structure, as described in our paper [3].
* an implementation of the experiments which are presented in [3].

## Installing Dependencies

The code is written in Python, and it is recommended to run the code inside a virtual environment.
To install the project dependencies, run

```bash
python -m pip install -r requirements.txt
```

## Dynamic Forest Data Structure

The dynamic forest data structure is implemented in the `dbscan/dynamic_forest.py` file.
The module provides a data structure, `EulerTourDynamicForest`, which provides the following interface:

* `add_node()`: adds a new unconnected node to the forest
* `link(i, j)`: add an edge between nodes i and j if doing so would not introduce a cycle
* `cut(i, j)`: remove the edge between nodes i and j, if it exists
* `root(i)`: return the index of the node at the root of the tree containing node i
* `remove_node(i)`: remove the node i from the forest, if its degree is 0.
* `neighbors(i)`: return the neighbors of node i

The methods `link`, `cut`, and `root` all have time complexity O(log(n)), where
n is the number of nodes in the forest.

The methods `add_node`, `remove_node`, and `neighbors` have complexity O(1).

The `forest_example.py` script demonstrates a simple usage of the dynamic forest data structure.

## Implementation of Dynamic DBSCAN

The dynamic DBSCAN algorithm is implemented in the `dbscan/dynamic_fdbscan.py` file.
The module provides a data structure, `DynamicDBSCAN` with the following methods.

* `init(k, t, eps, d, initial_data)`: the initialisation method takes parameters
* * `k`: the minimum number of points required for a point to be a core point (`min_samples` in the original DBSCAN algorithm)
* * `t`: the number of hash functions to use - setting this to be a small constant (say, 10) is reasonable
* * `eps`: the standard epsilon parameter of the DBSCAN algorithm
* * `d`: the dimension of the data points
* * `initial_data`: (optional) a numpy array of data points to add to the data structure
* `add_point(point)`: add a data point (numpy array object) to the data structure
* `get_cluster(index)`: get the ID of the cluster containing the point with the given index

The `dbscan_example.py` script demonstrates a simple usage of the dynamic DBSCAN data structure.

## Running the Experiments

To perform an experiment from the paper, run

```bash
python main.py run {experiment_name}
```

where `{experiment_name}` is one of `blobs`, `mnist`, `letters`, `fashion`, or `kdd`.
Once the experiment is complete, the recorded metrics will be saved in a CSV file in the results folder.

### Plotting the Figure

To display the plots shown in Figure 2 in the paper, run

```bash
python main.py plot fig2
```

## Contact

If you have any questions about the paper or the code, please feel free to get in touch with

* Seiyun Shin (seiyuns2@illinois.edu) or
* Peter Macgregor (contact details at https://pmacg.io).

## References

[1] Henzinger, M. R. and King, V. (1995). Randomized dynamic graph algorithms with polylogarithmic time per operation. In Proceedings of the twenty-seventh annual ACM symposium on Theory of computing, pages 519–527.

[2] Tseng, T., Dhulipala, L., and Blelloch, G. (2019). Batch-parallel euler tour trees. In 2019 Proceedings of the Twenty-First Workshop on Algorithm Engineering and Experiments (ALENEX), pages 92–106. SIAM.

[3] Shin, S., Shomorony, I., and Macgregor, P. (2025). Dynamic DBSCAN with Euler Tour Sequences. In 28th International Conference on Artificial Intelligence and Statistics (AISTATS 2025). PMLR.
