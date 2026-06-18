# PySimplicial

PySimplicial is a small Python library for working with finite simplicial complexes, strong collapses, cores, simplicial geometric category (`gscat`), and computational experiments in simplicial topology.

The project is designed both as a learning tool and as an experimental research environment for testing conjectures about simplicial complexes.

---

## Main Features

### Simplicial Complexes

- Construction of finite simplicial complexes
- Automatic generation of all faces
- Computation of dimension
- Detection of maximal simplices
- Extraction of the 1-skeleton
- Join construction
- Graph join construction

### Strong Collapse Theory

- Dominated vertex detection
- Strong collapse procedure
- Core computation
- Strong collapsibility testing

### Simplicial Geometric Category

- Computation of `gscat`
- Construction of categorical covers
- Special graph algorithm based on arboricity
- Graph covers by strongly collapsible subcomplexes

### Visualization

- Drawing simplicial complexes
- Drawing categorical covers
- Highlighting shared edges in different cover elements
- Visualization of 1-dimensional and 2-dimensional complexes

### Experiments

- Random graph generation
- Random 2-dimensional complex generation
- Search for counterexamples
- Core statistics
- Graph family tables
- Computational checks of conjectures

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

A filled simplex is strongly collapsible, so its core consists of a single vertex.

---

## Example: Graphs

```python
from generators.graphs import cycle_graph

K = cycle_graph(3)

value, cover = K.gscat(for_graph=True)

print("gscat(C3) =", value)
```

For graphs, PySimplicial uses the relationship between `gscat` and graph arboricity.

For a connected graph G:

$$
gscat(G) = a(G) - 1
$$

where a(G) is the arboricity of G.


---

## Example: Standard Complexes

```python
from generators.complexes import tetrahedron_boundary, torus_minimal

S2 = tetrahedron_boundary()
T2 = torus_minimal()

print("dim(S2) =", S2.dimension())
print("dim(T2) =", T2.dimension())
```

The `generators/` folder contains ready-made examples of graphs and simplicial complexes for tests and experiments.

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

If the same edge belongs to several cover elements, the visualization can display shared edges using small offsets so that all memberships remain visible.

---

## Experiments

The `experiments/` folder contains scripts for computational exploration.

### Check the skeleton inequality

```bash
python experiments/check_gscat_vs_skeleton.py
```

This script tests the experimental inequality
$$
gscat(K) \leq gscat(K^(1))
$$
on randomly generated 2-dimensional simplicial complexes.

This is not a proof. It is a computational experiment.

### Search for counterexamples

```bash
python experiments/counterexample_search.py
```

This script runs a larger parameter sweep and stops if a potential counterexample is found.

### Core statistics

```bash
python experiments/core_statistics.py
```

This script studies how often random complexes are strongly collapsible and computes basic statistics about their cores.

### Graph family table

```bash
python experiments/graph_family_table.py
```

This script computes `gscat` for standard graph families such as paths, cycles, stars, wheels, and complete graphs.

---

## Tests

Run all tests using:

```bash
pytest
```

The tests check:

- dimension computation
- 1-skeleton construction
- strong collapsibility
- core computation
- graph `gscat`
- categorical cover correctness
- strong collapsibility of cover elements
- join construction

---

## Research Motivation

This project was developed as an experimental tool for studying the simplicial geometric category and strong collapse theory.

One of the motivating questions is the inequality

$$
gscat(K) \leq gscat(K^{(1)})
$$


where $$K^{(1)}$$ is the 1-skeleton of the simplicial complex K.

PySimplicial allows one to generate examples, compute categorical covers, test special cases, and search for possible counterexamples.

---

## Current Limitations

- General `gscat` computation is exponential and intended only for small complexes.
- The graph algorithm is faster, but constructing explicit covers can still be expensive for large dense graphs.
- Some generated complexes are intended mainly for experiments and are not canonical minimal triangulations.
- The project is currently under active development.

---


## References

- J. A. Barmak, *Algebraic Topology of Finite Topological Spaces and Applications*
- D. Kozlov, *Combinatorial Algebraic Topology*
- D. Fernández-Ternero, E. Macías-Virgós, J. A. Vilches, works on simplicial LS-category
- C. St. J. A. Nash-Williams, work on graph arboricity

---

## Author

Islam Yeginbay

Mathematical Computer Modeling, SDU University
