# PySimplicial

PySimplicial is a Python library for the study and computation of finite simplicial complexes, strong collapses, cores, simplicial geometric category (`gscat`), and computational experiments in simplicial topology.

The project is designed both as an educational resource and as a research-oriented computational framework for investigating conjectures in combinatorial and algebraic topology.

---

## Main Features

### Simplicial Complexes

* Construction of finite simplicial complexes
* Automatic generation of all faces
* Computation of dimension
* Detection of maximal simplices
* Extraction of the 1-skeleton
* Join construction
* Graph join construction

### Strong Collapse Theory

* Detection of dominated vertices
* Implementation of strong collapse procedures
* Core computation
* Testing for strong collapsibility

### Simplicial Geometric Category

* Computation of `gscat`
* Construction of categorical covers
* Specialized graph algorithm based on arboricity
* Graph covers by strongly collapsible subcomplexes

### Visualization

* Visualization of simplicial complexes
* Visualization of categorical covers
* Highlighting shared edges across cover elements
* Rendering of 1-dimensional and 2-dimensional complexes

### Experiments

* Random graph generation
* Random 2-dimensional complex generation
* Search for counterexamples
* Core statistics
* Graph family tables
* Computational verification of conjectures

---

## Performance

PySimplicial combines a mathematically intuitive interface with optimized internal representations for efficient computation.

### Internal Representation

The public API represents simplices as Python `frozenset` objects, ensuring clarity and mathematical readability.

For performance-critical operations, simplices are internally encoded as compact bitmasks. This enables efficient execution of:

* simplex inclusion tests
* maximal simplex detection
* face comparisons
* graph-theoretic computations

This optimization layer is fully transparent to the user.

### NumPy Acceleration

Several algorithms rely on NumPy-based implementations, including:

* computation of arboricity via the Nash–Williams formula
* subset and bitmask transformations
* induced subgraph statistics

These optimizations significantly improve performance on medium-sized instances while preserving usability.

---

## Project Structure

```text
PySimplicial/
│
├── simplicial_topology/
│   ├── Simplicial_Complex.py
│   └── visualization.py
│
├── generators/
│   ├── complexes.py
│   └── graphs.py
│
├── experiments/
│   ├── check_gscat_vs_skeleton.py
│   ├── core_statistics.py
│   ├── counterexample_search.py
│   ├── graph_family_table.py
│   └── random_complexes.py
│
├── tests/
│   ├── test_complete_graphs.py
│   ├── test_core.py
│   ├── test_cover.py
│   ├── test_cover_strongly_collapsible.py
│   ├── test_cycle.py
│   ├── test_dimension.py
│   ├── test_graph_gscat.py
│   ├── test_join.py
│   ├── test_one_skeleton.py
│   └── test_strongly_collapsible.py
│
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/zgenng/PySimplicial.git
cd PySimplicial
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Main dependencies:

```bash
pip install numpy matplotlib networkx more-itertools pytest
```

---

## Quick Example

```python
from simplicial_topology.Simplicial_Complex import SimplicialComplex

K = SimplicialComplex()

K.add_simplex([1, 2, 3])
K.add_simplex([3, 4])

print("Dimension:", K.dimension())

value, cover = K.gscat()

print("gscat(K) =", value)
print("Cover:", cover)
```

---

## Example: 1-Skeleton

```python
from simplicial_topology.Simplicial_Complex import SimplicialComplex

K = SimplicialComplex()

K.add_simplex([0, 1, 2])

G = K.one_skeleton()

print(G.simplicial_complex)
```

The 1-skeleton of a filled triangle contains its vertices and edges, but not the 2-simplex.

---

## Example: Strong Collapse and Core

```python
from simplicial_topology.Simplicial_Complex import SimplicialComplex

K = SimplicialComplex()

K.add_simplex([0, 1, 2])

core = K.core()

print("Core:", core.simplicial_complex)
print("Strongly collapsible:", K.is_strongly_collapsible())
```

A filled simplex is strongly collapsible, and therefore its core consists of a single vertex.

---

## Example: Graphs

```python
from generators.graphs import cycle_graph

K = cycle_graph(3)

value, cover = K.gscat(for_graph=True)

print("gscat(C3) =", value)
```

For graphs, PySimplicial uses the relationship between the simplicial geometric category and graph arboricity.

For a connected graph $$G$$:

$$
gscat(G) = a(G) - 1
$$

where $$a(G)$$ denotes the arboricity of $$G$$.

---

## Example: Standard Complexes

```python
from generators.complexes import tetrahedron_boundary, torus_minimal

S2 = tetrahedron_boundary()
T2 = torus_minimal()

print("dim(S2) =", S2.dimension())
print("dim(T2) =", T2.dimension())
```

The `generators/` module provides standard examples of simplicial complexes and graphs for experimentation and testing.

---

## Visualization

```python
from generators.complexes import tetrahedron_boundary
from simplicial_topology.visualization import draw_complex

K = tetrahedron_boundary()

draw_complex(K)
```

To visualize a categorical cover:

```python
from generators.graphs import complete_graph
from simplicial_topology.visualization import draw_cover_on_complex

K = complete_graph(5)

value, cover = K.gscat(for_graph=True)

draw_cover_on_complex(K, cover)
```

If an edge belongs to multiple cover elements, the visualization displays it with slight offsets to preserve clarity.

---

## Experiments

The `experiments/` directory contains scripts for computational investigations.

### Check the skeleton inequality

```bash
python experiments/check_gscat_vs_skeleton.py
```

This script tests the inequality

$$
gscat(K) \leq gscat(K^{(1)})
$$

on randomly generated 2-dimensional simplicial complexes.

This result is currently supported by computational evidence but is not proven in general.

### Search for counterexamples

```bash
python experiments/counterexample_search.py
```

This script performs an extended parameter search and terminates upon detecting a potential counterexample.

### Core statistics

```bash
python experiments/core_statistics.py
```

This script analyzes the frequency of strong collapsibility in random complexes and computes statistics of their cores.

### Graph family table

```bash
python experiments/graph_family_table.py
```

This script computes $$gscat$$ values for standard graph families such as paths, cycles, stars, wheels, and complete graphs.

---

## Tests

Run all tests using:

```bash
pytest
```

The test suite verifies:

* dimension computation
* 1-skeleton construction
* strong collapsibility
* core computation
* graph `gscat`
* correctness of categorical covers
* strong collapsibility of cover elements
* join construction

---

## Research Motivation

This project was developed as an experimental tool for studying the simplicial geometric category and strong collapse theory.

One of the motivating research questions is the inequality

$$
gscat(K) \leq gscat(K^{(1)}),
$$

where $$K^{(1)}$$ denotes the 1-skeleton of K.

At the moment this inequality is supported by computational experiments, but no general proof is currently included in the project.

PySimplicial allows one to generate examples, compute categorical covers, test special cases, and search for possible counterexamples.

---

## Current Limitations

* General computation of `gscat` is exponential and feasible only for small complexes
* The graph-based algorithm is more efficient, but explicit cover construction remains costly for dense graphs
* Join decomposition (`is_join`) relies on exhaustive search and becomes expensive for complexes with many vertices
* Some generated complexes are intended for experimentation and are not minimal triangulations
* The project is under active development

---

## References

* J. A. Barmak, *Algebraic Topology of Finite Topological Spaces and Applications*
* D. Kozlov, *Combinatorial Algebraic Topology*
* D. Fernández-Ternero, E. Macías-Virgós, J. A. Vilches, works on simplicial LS-category
* C. St. J. A. Nash-Williams, work on graph arboricity

---

## Author

Islam Yeginbay

B.Sc. Student in Mathematical Computer Modeling
SDU University 

Email: [islam.yeginbay@gmail.com](mailto:islam.yeginbay@gmail.com)

Research interests:

* Simplicial topology
* Strong collapse theory
* Simplicial LS-category
* Computational topology
