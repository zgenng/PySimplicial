# PySimplicial

PySimplicial is a Python library for finite abstract simplicial complexes, simplicial maps, contiguity, and simplicial Lusternik–Schnirelmann category. It computes exact values and returns the subcomplexes or contiguity chains that witness them.

## Installation

PySimplicial requires Python 3.10 or newer. Run one of these commands from the project directory.

| What you need | Command |
| --- | --- |
| Core library and CPU solver | `python -m pip install -e .` |
| Matplotlib and Plotly visualization | `python -m pip install -e ".[viz]"` |
| Visualization and CUDA 12 | `python -m pip install -e ".[viz,cuda12]"` |
| Visualization and CUDA 13 | `python -m pip install -e ".[viz,cuda13]"` |

Install `.[plot]` for Matplotlib only or `.[plot3d]` for Plotly only. Use `cuda_status()` to check whether the CUDA backend can access your NVIDIA GPU.

## Quick start

The three edges below form the boundary of a triangle.

```python
from pysimplicial import (
    SimplicialComplex,
    SimplicialMap,
    contiguity_distance,
    gscat,
    scat,
    same_contiguity_class,
)

K = SimplicialComplex([(0, 1), (1, 2), (2, 0)])
identity = SimplicialMap.identity(K)
constant = SimplicialMap.constant(K, K, vertex=0)

print(K.dimension())                                # 1
print(same_contiguity_class(identity, constant))     # False
print(contiguity_distance(identity, constant).value) # 1
print(scat(K).value)                                 # 1
print(gscat(K).value)                                # 1
```

### Complexes and maps

A `SimplicialComplex` is immutable. Its constructor takes simplices, keeps the inclusion-maximal ones as `K.facets`, and generates faces on demand. Use `K.faces(d)` for the faces of dimension `d`, `K.skeleton(d)` for the `d`-skeleton, and `K.induced_subcomplex(vertices)` for a full subcomplex. By convention, `faces()` lists nonempty faces, and the empty complex has dimension `-1`.

The operations module includes `join`, `graph_join`, `wedge`, `disjoint_union`, `link`, `relabel`, `connected_components`, and `categorical_product`. Joins require disjoint vertex labels. For a wedge, pass the two vertices to identify. `strong_core(K)` returns the core together with the sequence of dominated vertex deletions.

A `SimplicialMap(K, L, vertex_map)` validates the images of the domain facets. Maps support restriction to a subcomplex, composition, and evaluation on vertices and simplices.

### Contiguity and category

| Function | What it computes |
| --- | --- |
| `is_contiguous(f, g)` | Whether the union of the two images of each domain simplex is a simplex of the codomain |
| `contiguity_chain(f, g)` | A chain of contiguous maps between `f` and `g`, or `None` if they lie in different classes |
| `contiguity_class(f)` | An iterator over all maps in the class of `f` |
| `contiguity_distance(f, g)` | A minimum domain cover on whose parts the restricted maps are in the same contiguity class |
| `scat(K)` | A minimum cover whose inclusions into `K` are in the contiguity class of a constant map |
| `gscat(K)` | A minimum cover by subcomplexes that are strongly collapsible on their own |

All three numerical invariants use the reduced convention. A minimum cover with `r` members gives the value `r - 1`. For an empty domain or complex the value is `0`. For a disconnected complex, `scat` can use a different constant in each component. Contiguity distance returns `math.inf` when no qualifying cover exists.

The contiguity distance and `scat` results provide `value`, `cover`, and `chains`. Each chain corresponds to one cover member. Use `max_states` to bound the search for a contiguity chain, and `max_subcomplexes` or `max_search_states` to bound a cover search. These searches raise `SearchLimitExceeded` if a limit interrupts them.

## Exact geometric category on CPU or CUDA

`gscat(K)` uses the CPU by default. `backend="cuda"` requires an available CUDA device for the general search, and `backend="auto"` uses CUDA when available. Simple cases and graphs have exact CPU shortcuts.

### Deterministic CUDA search without restarts

```python
from pysimplicial import cuda_status, gscat, verify_gscat_cover
from pysimplicial.datasets import projective_plane

K = projective_plane()
print(cuda_status())
result = gscat(K, backend="cuda", restarts=0)

print(result.value)  # 2
print(result.gpu_used)
assert verify_gscat_cover(K, result.cover)
```

With `restarts=0`, the solver goes straight to exact search. It constructs a sufficient family of strongly collapsible candidates by reversing strong collapses. CUDA accelerates candidate construction and strong-collapse checks. The CPU finds the minimum cover.

Set `restarts` to a positive number to try randomized GPU searches first. A verified cover gives an upper bound. If it matches the proven lower bound, the solver returns immediately. Otherwise, the exact phase tries to prove a smaller cover exists or certify the current one. The restart kernel supports up to 64 vertices and 512 facets. For larger inputs the solver skips that kernel and uses the exact path. With more than 64 vertices, the exact path can still construct candidates on CUDA and uses the CPU for strong-collapse checks.

For long runs, set `max_candidate_states` and `max_search_states` to limit the exact search. If a limit is reached before minimality is proved, the result has `exact=False`, `value=None`, a verified `cover`, and `lower_bound` and `upper_bound`. Use `progress=print` to see search progress.

### Exhaustive enumeration for small complexes

The internal solver also provides an independent cross-check that enumerates subcomplexes. The CPU performs enumeration and solves the minimum-cover problem. When CUDA is available, it batches the strong-collapse checks.

```python
from pysimplicial._solvers.exact import exact_gscat
from pysimplicial.datasets import tetrahedron_boundary

K = tetrahedron_boundary()
check = exact_gscat(
    K.facets,
    use_gpu=True,
    algorithm="all_subcomplexes",
    max_subcomplexes=100_000,
)
print(check.gscat)  # 1
print(check.gpu_used)
```

This mode can be considerably more expensive. It is an internal API and can fall back to the CPU, so check `gpu_used` if you need to verify device usage.

## Visualization

```python
from pysimplicial import gscat
from pysimplicial.datasets import projective_plane
from pysimplicial.visualization import draw_cover, draw_surface_3d

K = projective_plane()
result = gscat(K)

draw_cover(K, result, legend=True)
draw_surface_3d(K, cover=result.cover, show=False, save_html="rp2-cover.html")
```

`draw_complex`, `draw_cover`, and `draw_complex_and_cover` use Matplotlib and show the 2-skeleton of higher-dimensional complexes. `draw_surface_3d` renders triangulated surfaces with Plotly. The minimal triangulation of the projective plane can be displayed on its orientable double cover. Use `draw_complex_3d` to plot a 2-skeleton at supplied vertex coordinates.

## Datasets, examples, and tests

`pysimplicial.datasets` includes standard graph families, named triangulations such as `torus()` and `triangulation("CP2")`, 865 genus-two triangulations, random complexes, and Vietoris–Rips constructions.

```bash
python examples/basic.py
python experiments/triangulations.py --name RP2 --backend cpu
python -m unittest discover -s tests -v
```

## Migrating from the original layout

Import `SimplicialComplex` and `SimplicialMap` from `pysimplicial`. Collect simplices in a list before constructing an immutable complex. Replace `K.gscat()` with `gscat(K)`. Cover members are now `SimplicialComplex` objects, so use `part.faces()` if you need their full lists of faces.

The triangulation data comes from the original project. Its references include [Frank Lutz’s surface catalog](https://www3.math.tu-berlin.de/IfM/Nachrufe/Frank_Lutz/stellar/surfaces.html), [a Klein bottle model](https://www.math.union.edu/~dpvc/papers/1993-03.KB/vmkb.pdf), and [arXiv:1306.3550](https://arxiv.org/abs/1306.3550).
