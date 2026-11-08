# PySimplicial

**A Python library for computational simplicial topology, simplicial maps, contiguity, and simplicial Lusternik–Schnirelmann category.**

PySimplicial provides tools for constructing and analyzing finite abstract simplicial complexes, studying strong homotopy properties, and computing simplicial topological invariants. It includes exact algorithms for simplicial category and contiguity distance, along with CPU and optional CUDA-assisted methods for the simplicial geometric category.

## Features

- **Simplicial complexes:** Immutable finite simplicial complexes, faces, facets, dimensions, and skeletons.
- **Operations:** Joins, wedges, disjoint unions, links, categorical products, and connected components.
- **Strong homotopy:** Dominated vertices, strong collapsibility, and strong cores.
- **Simplicial maps:** Construction, restriction, composition, and validation.
- **Contiguity:** Direct contiguity, contiguity classes, chains, and distance.
- **Simplicial LS category:** Computation of `scat` and `gscat`.
- **Exact computation:** Categorical covers, certificates, and rigorous bounds.
- **Optional CUDA acceleration:** GPU-assisted geometric category computations.
- **Datasets and visualization:** Graph families, triangulated manifolds, random and Vietoris–Rips complexes, 2D and interactive 3D plots.

## Installation

Requires **Python 3.10+** and **NumPy 1.24+**.

Install directly from GitHub:

```bash
python -m pip install git+https://github.com/zgenng/PySimplicial.git
```

Or clone the repository and install in editable mode:

```bash
git clone https://github.com/zgenng/PySimplicial.git
cd PySimplicial
python -m pip install -e .
```

Optional visualization and CUDA dependencies (run from the project root):

```bash
python -m pip install -e ".[viz]"       # Matplotlib + Plotly
python -m pip install -e ".[plot]"      # Matplotlib only
python -m pip install -e ".[plot3d]"    # Plotly only
python -m pip install -e ".[cuda12]"    # CUDA 12
python -m pip install -e ".[cuda13]"    # CUDA 13
```

CUDA requires a compatible NVIDIA GPU and driver. Once the package is published to PyPI, the intended command is:

```bash
pip install pysimplicial
```

## Quick start

### Finite simplicial complexes

```python
from pysimplicial import SimplicialComplex

# Boundary of a triangle, a simplicial model of S¹
K = SimplicialComplex([(0, 1), (1, 2), (2, 0)])

print(K.dimension())  # 1
print(K.vertices)
print(K.facets)
print(K.faces())
```

`SimplicialComplex` is immutable. Its facets are the inclusion-maximal simplices; nonempty faces are generated on demand. Use `K.faces(d)`, `K.skeleton(d)`, or `K.induced_subcomplex(vertices)` for common operations.

### Strong collapsibility

```python
from pysimplicial import is_strongly_collapsible, strong_core
from pysimplicial.datasets import path_graph

K = path_graph(5)
print(is_strongly_collapsible(K))  # True
result = strong_core(K)
print(result.core)
print(result.steps)
```

A finite complex is *strongly collapsible* when dominated vertices can be removed iteratively until one vertex remains.

### Simplicial maps and contiguity

Two simplicial maps `f, g: K → L` are contiguous if `f(σ) ∪ g(σ)` is a simplex of `L` for every simplex `σ` of `K`.

```python
from pysimplicial import (
    SimplicialComplex, SimplicialMap,
    is_contiguous, same_contiguity_class, contiguity_chain,
)

K = SimplicialComplex([(0, 1)])
f = SimplicialMap.identity(K)
g = SimplicialMap.constant(K, K, vertex=0)

print(is_contiguous(f, g))
print(same_contiguity_class(f, g))
print(contiguity_chain(f, g))
```

## Simplicial Lusternik–Schnirelmann category

PySimplicial computes two related invariants:

- `scat(K)`: reduced simplicial LS category, using covers by categorical subcomplexes.
- `gscat(K)`: reduced simplicial geometric category, using covers by strongly collapsible subcomplexes.

A cover by `r` subcomplexes yields the **reduced value `r - 1`**.

### Simplicial category

```python
from pysimplicial import scat
from pysimplicial.datasets import cycle_graph

K = cycle_graph(3)
result = scat(K)
print(result.value)  # 1
print(result.cover)
print(result.chains)
```

### Geometric category

```python
from pysimplicial import gscat, verify_gscat_cover
from pysimplicial.datasets import projective_plane

K = projective_plane()
result = gscat(K)

print(result.value)        # 2
print(result.exact)
print(result.lower_bound, result.upper_bound)
assert verify_gscat_cover(K, result.cover)
```

Mathematically, in the reduced convention,

\[
\operatorname{gscat}(K) = \min\{r \geq 0 : K = K_0 \cup \cdots \cup K_r,\; K_i \text{ strongly collapsible}\}.
\]

When the result's lower and upper bounds agree, the answer is exact. Otherwise, `value` is `None` and the verified cover supplies an upper bound. Search limits can be configured with `max_candidate_states` and `max_search_states`.

## Optional CUDA acceleration

```python
from pysimplicial import cuda_status, gscat
from pysimplicial.datasets import projective_plane

print(cuda_status())
K = projective_plane()
result = gscat(K, backend="cuda", restarts=1000, seed=42)
print(result.value, result.gpu_used, result.exact)
```

Set `backend="auto"` to use CUDA when available, or `backend="cpu"` for CPU-only computation. Set `restarts=0` to skip randomized restarts and proceed to the exact search. CPU verification is used to check returned covers. The randomized CUDA restart kernel supports up to 64 vertices and 512 facets; larger inputs use the exact-search path.

## Datasets

### Standard graph families

```python
from pysimplicial.datasets import (
    complete_graph, complete_bipartite_graph, cycle_graph, path_graph,
)

K1 = complete_graph(5)
K2 = complete_bipartite_graph(3, 4)
K3 = cycle_graph(6)
K4 = path_graph(5)
```

### Named triangulations

```python
from pysimplicial.datasets import (
    torus, projective_plane, tetrahedron_boundary,
    triangulation, triangulation_names,
)

T = torus()
RP2 = projective_plane()
S2 = tetrahedron_boundary()
CP2 = triangulation("CP2")
print(CP2.dimension())  # 4
print(triangulation_names())
```

### Random and Vietoris–Rips complexes

```python
from pysimplicial.datasets import random_d_complex, generate_rips_data

K = random_d_complex(n=10, dim=2, seed=42)
points, R, epsilon = generate_rips_data(n_points=20, shape="circle", seed=42)
print(R.dimension(), epsilon)
```

## Visualization

Install `.[viz]` first.

### 2D

```python
from pysimplicial import gscat
from pysimplicial.datasets import complete_graph
from pysimplicial.visualization import draw_complex_and_cover

K = complete_graph(5)
result = gscat(K)
draw_complex_and_cover(K, result, legend=True)
```

### Interactive 3D

```python
from pysimplicial import gscat
from pysimplicial.datasets import projective_plane
from pysimplicial.visualization import draw_surface_3d

K = projective_plane()
result = gscat(K)
draw_surface_3d(K, cover=result.cover, show=False, save_html="rp2-cover.html")
```

## Tests and development

After installing the package:

```bash
python -m unittest discover -s tests -v
```

Build and validate distributions:

```bash
python -m pip install build twine
python -m build
python -m twine check dist/*
```

The test suite covers topological operations, contiguity, geometric category, datasets, and visualization. CUDA-specific tests require compatible hardware.

## Mathematical background and references

This project is motivated by computational problems in finite simplicial complexes, strong homotopy theory, contiguity, and simplicial Lusternik–Schnirelmann category.

- J. A. Barmak and E. G. Minian, *Strong homotopy types, nerves and collapses*, Discrete & Computational Geometry (2012).
- D. Fernández-Ternero, E. Macías-Virgós, E. Minuz, and J. A. Vilches, *Simplicial Lusternik–Schnirelmann category*, Publicacions Matemàtiques (2019).
- [Frank Lutz's triangulated surface catalog](https://www3.math.tu-berlin.de/IfM/Nachrufe/Frank_Lutz/stellar/surfaces.html).

## Contributing

Bug reports, suggestions, and contributions are welcome. Please open an [issue](https://github.com/zgenng/PySimplicial/issues) to discuss problems or improvements.

## Project status

PySimplicial is an evolving research-oriented project. Public APIs and algorithms may change as development continues.

**Current source version:** `0.2.0`

## Author

**Islam Yeginbay** — [@zgenng](https://github.com/zgenng/PySimplicial)
