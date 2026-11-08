"""Standard graphs as one-dimensional simplicial complexes."""

from itertools import combinations, product

from ..complex import SimplicialComplex


def _graph(vertices, edges):
    return SimplicialComplex([*((v,) for v in vertices), *edges])


def complete_graph(n: int) -> SimplicialComplex:
    vertices = range(1, n + 1)
    return _graph(vertices, combinations(vertices, 2))


def cycle_graph(n: int) -> SimplicialComplex:
    return _graph(range(n), ((v, (v + 1) % n) for v in range(n)))


def path_graph(n: int) -> SimplicialComplex:
    return _graph(range(n), ((v, v + 1) for v in range(n - 1)))


def star_graph(n: int) -> SimplicialComplex:
    return _graph(range(n), ((0, v) for v in range(1, n)))


def wheel_graph(n: int) -> SimplicialComplex:
    if n < 4:
        raise ValueError("A wheel graph needs at least four vertices.")
    cycle = cycle_graph(n - 1)
    return _graph(range(n), [*cycle.facets, *((n - 1, v) for v in range(n - 1))])


def complete_bipartite_graph(m: int, n: int) -> SimplicialComplex:
    return complete_multipartite_graph(m, n)


def complete_multipartite_graph(*parts: int) -> SimplicialComplex:
    if len(parts) < 2 or any(size < 1 for size in parts):
        raise ValueError("Specify at least two parts with positive sizes.")
    groups = []
    start = 0
    for size in parts:
        groups.append(range(start, start + size))
        start += size
    return _graph(range(start), (edge for a, b in combinations(groups, 2) for edge in product(a, b)))
