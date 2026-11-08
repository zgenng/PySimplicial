"""Combinatorial constructions on finite simplicial complexes."""

from collections.abc import Iterable, Mapping
from itertools import product

from .complex import SimplicialComplex, Vertex


def relabel(complex_: SimplicialComplex, mapping: Mapping[Vertex, Vertex]) -> SimplicialComplex:
    """Rename vertices injectively, leaving omitted labels unchanged."""
    images = {v: mapping.get(v, v) for v in complex_.vertices}
    if len(set(images.values())) != len(images):
        raise ValueError("Relabelling must be injective on the vertices.")
    return _image(complex_, images)


def _image(complex_: SimplicialComplex, mapping: Mapping[Vertex, Vertex]) -> SimplicialComplex:
    return SimplicialComplex(tuple(mapping[v] for v in facet) for facet in complex_.facets)


def _require_disjoint(first: SimplicialComplex, second: SimplicialComplex) -> None:
    if first.vertices & second.vertices:
        raise ValueError("The factors must have disjoint vertex labels. Use relabel or disjoint_union.")


def disjoint_union(first: SimplicialComplex, second: SimplicialComplex) -> SimplicialComplex:
    """Take tagged copies whose vertex labels are (0, v) and (1, w)."""
    return SimplicialComplex(
        tuple((tag, vertex) for vertex in facet)
        for tag, factor in enumerate((first, second))
        for facet in factor.facets
    )


def join(first: SimplicialComplex, second: SimplicialComplex) -> SimplicialComplex:
    """Simplicial join of two complexes on disjoint vertex sets."""
    _require_disjoint(first, second)
    empty_face = (frozenset(),)
    return SimplicialComplex(a | b for a, b in product(first.facets or empty_face, second.facets or empty_face))


def graph_join(first: SimplicialComplex, second: SimplicialComplex) -> SimplicialComplex:
    """Join two graphs by adding all cross edges."""
    if max(first.dimension(), second.dimension()) > 1:
        raise ValueError("Graph join requires complexes of dimension at most one.")
    _require_disjoint(first, second)
    return SimplicialComplex([*first.facets, *second.facets, *product(first.vertices, second.vertices)])


def wedge(
    first: SimplicialComplex,
    second: SimplicialComplex,
    first_vertex: Vertex,
    second_vertex: Vertex,
) -> SimplicialComplex:
    """Identify the specified vertices. Preserve other labels when possible."""
    if first_vertex not in first.vertices or second_vertex not in second.vertices:
        raise ValueError("Wedge points must be vertices of their respective factors.")
    other_vertices = second.vertices - {second_vertex}
    used = set(first.vertices | other_vertices)
    mapping = {second_vertex: first_vertex}
    for vertex in other_vertices:
        label = vertex
        if label in first.vertices:
            index = 0
            label = (vertex, index)
            while label in used:
                index += 1
                label = (vertex, index)
            used.add(label)
        mapping[vertex] = label
    return SimplicialComplex(first.facets | _image(second, mapping).facets)


def link(complex_: SimplicialComplex, simplex: Iterable[Vertex]) -> SimplicialComplex:
    """The link of a face. The empty face has link equal to the whole complex."""
    face = frozenset(simplex)
    if face and face not in complex_:
        raise ValueError("Link requires a simplex of the complex.")
    return SimplicialComplex(facet - face for facet in complex_.facets if face <= facet)


def categorical_product(first: SimplicialComplex, second: SimplicialComplex) -> SimplicialComplex:
    """The product whose simplices project to simplices in both factors."""
    return SimplicialComplex(product(a, b) for a, b in product(first.facets, second.facets))


def connected_components(complex_: SimplicialComplex) -> tuple[SimplicialComplex, ...]:
    """Return components of the 1-skeleton as induced subcomplexes."""
    parent = {vertex: vertex for vertex in complex_.vertices}

    def root(vertex):
        while parent[vertex] != vertex:
            parent[vertex] = parent[parent[vertex]]
            vertex = parent[vertex]
        return vertex

    for facet in complex_.facets:
        anchor, *others = facet
        for vertex in others:
            parent[root(vertex)] = root(anchor)
    groups = {}
    for facet in complex_.facets:
        groups.setdefault(root(next(iter(facet))), []).append(facet)
    return tuple(SimplicialComplex(facets) for facets in groups.values())
