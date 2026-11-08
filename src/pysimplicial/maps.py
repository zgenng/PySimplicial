"""Vertex maps between simplicial complexes."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .complex import Simplex, SimplicialComplex, Vertex


@dataclass(frozen=True, slots=True, init=False)
class SimplicialMap:
    domain: SimplicialComplex
    codomain: SimplicialComplex
    vertex_map: Mapping[Vertex, Vertex]

    def __init__(
        self,
        domain: SimplicialComplex,
        codomain: SimplicialComplex,
        vertex_map: Mapping[Vertex, Vertex],
    ):
        images = dict(vertex_map)
        if images.keys() != domain.vertices:
            raise ValueError("Give exactly one image for each domain vertex.")
        if not set(images.values()) <= codomain.vertices:
            raise ValueError("Vertex images must belong to the codomain.")
        for facet in domain.facets:
            image = frozenset(images[v] for v in facet)
            if image not in codomain:
                raise ValueError(f"The image {image} of {facet} is not a simplex.")
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "codomain", codomain)
        object.__setattr__(self, "vertex_map", MappingProxyType(images))

    def __hash__(self) -> int:
        return hash((self.domain, self.codomain, frozenset(self.vertex_map.items())))

    def __call__(self, vertex: Vertex) -> Vertex:
        return self.vertex_map[vertex]

    def image_simplex(self, simplex: Iterable[Vertex]) -> Simplex:
        face = frozenset(simplex)
        if face and face not in self.domain:
            raise ValueError("The input is not a simplex of the domain.")
        return frozenset(self(v) for v in face)

    def restrict(self, subcomplex: SimplicialComplex) -> "SimplicialMap":
        if not subcomplex.is_subcomplex_of(self.domain):
            raise ValueError("Restriction requires a subcomplex of the domain.")
        return SimplicialMap(subcomplex, self.codomain, {v: self(v) for v in subcomplex.vertices})

    def compose(self, other: "SimplicialMap") -> "SimplicialMap":
        """Return self ∘ other."""
        if other.codomain != self.domain:
            raise ValueError("The intermediate complexes must agree.")
        return SimplicialMap(other.domain, self.codomain, {v: self(other(v)) for v in other.domain.vertices})

    def _check_parallel(self, other: "SimplicialMap") -> None:
        if self.domain != other.domain or self.codomain != other.codomain:
            raise ValueError("The maps must have the same domain and codomain.")

    def is_contiguous(self, other: "SimplicialMap") -> bool:
        from .contiguity import is_contiguous

        return is_contiguous(self, other)

    @classmethod
    def identity(cls, complex_: SimplicialComplex) -> "SimplicialMap":
        return cls(complex_, complex_, {v: v for v in complex_.vertices})

    @classmethod
    def constant(cls, domain: SimplicialComplex, codomain: SimplicialComplex, vertex: Vertex) -> "SimplicialMap":
        if vertex not in codomain.vertices:
            raise ValueError("The constant vertex must belong to the codomain.")
        return cls(domain, codomain, dict.fromkeys(domain.vertices, vertex))

    @classmethod
    def inclusion(cls, subcomplex: SimplicialComplex, ambient: SimplicialComplex) -> "SimplicialMap":
        if not subcomplex.is_subcomplex_of(ambient):
            raise ValueError("The domain must be a subcomplex of the codomain.")
        return cls(subcomplex, ambient, {v: v for v in subcomplex.vertices})
