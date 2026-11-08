"""Finite abstract simplicial complexes specified by their facets."""

from collections.abc import Hashable, Iterable, Iterator
from dataclasses import dataclass, field
from itertools import combinations
from typing import TypeAlias

Vertex: TypeAlias = Hashable
Simplex: TypeAlias = frozenset[Vertex]


@dataclass(frozen=True, slots=True, init=False)
class SimplicialComplex:
    """An immutable complex. Input simplices generate all their nonempty faces."""

    facets: frozenset[Simplex]
    vertices: frozenset[Vertex] = field(compare=False, repr=False)

    def __init__(self, simplices: Iterable[Iterable[Vertex]] = ()):
        faces = {frozenset(simplex) for simplex in simplices}
        maximal: list[Simplex] = []
        for face in sorted(faces - {frozenset()}, key=len, reverse=True):
            if not any(face < facet for facet in maximal):
                maximal.append(face)
        object.__setattr__(self, "facets", frozenset(maximal))
        object.__setattr__(self, "vertices", frozenset().union(*maximal))

    def dimension(self) -> int:
        return max(map(len, self.facets), default=0) - 1

    def maximal_simplices(self) -> frozenset[Simplex]:
        return self.facets

    def iter_faces(self, dim: int | None = None) -> Iterator[Simplex]:
        """Enumerate faces once, optionally at a single dimension."""
        if dim is not None and dim < 0:
            return
        seen: set[Simplex] = set()
        for facet in self.facets:
            sizes = range(1, len(facet) + 1) if dim is None else (dim + 1,)
            for size in sizes:
                for vertices in combinations(facet, size):
                    face = frozenset(vertices)
                    if face not in seen:
                        seen.add(face)
                        yield face

    def faces(self, dim: int | None = None) -> frozenset[Simplex]:
        return frozenset(self.iter_faces(dim))

    @property
    def simplicial_complex(self) -> frozenset[Simplex]:
        """Read-only compatibility view of all nonempty simplices."""
        return self.faces()

    def __iter__(self) -> Iterator[Simplex]:
        return self.iter_faces()

    def __contains__(self, simplex: Iterable[Vertex]) -> bool:
        face = frozenset(simplex)
        return bool(face) and any(face <= facet for facet in self.facets)

    def is_subcomplex_of(self, other: "SimplicialComplex") -> bool:
        return all(face in other for face in self.facets)

    def induced_subcomplex(self, vertices: Iterable[Vertex]) -> "SimplicialComplex":
        selected = frozenset(vertices)
        if not selected <= self.vertices:
            raise ValueError("Induced subcomplex requires vertices of this complex.")
        return SimplicialComplex(facet & selected for facet in self.facets)

    def skeleton(self, dim: int) -> "SimplicialComplex":
        return SimplicialComplex(
            face
            for facet in self.facets
            for face in combinations(facet, min(max(dim + 1, 0), len(facet)))
        )

    def one_skeleton(self) -> "SimplicialComplex":
        return self.skeleton(1)
