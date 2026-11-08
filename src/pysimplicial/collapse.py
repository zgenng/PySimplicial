"""Strong collapses by deleting dominated vertices."""

from dataclasses import dataclass

from .complex import SimplicialComplex, Vertex


@dataclass(frozen=True, slots=True)
class CollapseStep:
    vertex: Vertex
    dominator: Vertex


@dataclass(frozen=True, slots=True)
class StrongCoreResult:
    core: SimplicialComplex
    steps: tuple[CollapseStep, ...]


def _vertex_key(vertex: Vertex):
    cls = type(vertex)
    return cls.__module__, cls.__qualname__, repr(vertex)


def dominators(complex_: SimplicialComplex, vertex: Vertex) -> frozenset[Vertex]:
    if vertex not in complex_.vertices:
        raise ValueError("The given vertex does not belong to the complex.")
    incident = iter(facet for facet in complex_.facets if vertex in facet)
    return next(incident).intersection(*incident) - {vertex}


def strong_core(complex_: SimplicialComplex) -> StrongCoreResult:
    remaining = complex_
    steps = []
    while remaining.vertices:
        for vertex in sorted(remaining.vertices, key=_vertex_key):
            choices = dominators(remaining, vertex)
            if choices:
                steps.append(CollapseStep(vertex, min(choices, key=_vertex_key)))
                remaining = SimplicialComplex(facet - {vertex} for facet in remaining.facets)
                break
        else:
            break
    return StrongCoreResult(remaining, tuple(steps))


def core(complex_: SimplicialComplex) -> SimplicialComplex:
    return strong_core(complex_).core


def is_strongly_collapsible(complex_: SimplicialComplex) -> bool:
    return len(core(complex_).vertices) == 1
