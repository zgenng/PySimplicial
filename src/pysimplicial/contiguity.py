"""Direct contiguity and chains in the graph of simplicial maps."""

from collections import deque
from collections.abc import Iterator

from ._errors import SearchLimitExceeded
from .maps import SimplicialMap


def is_contiguous(first: SimplicialMap, second: SimplicialMap) -> bool:
    first._check_parallel(second)
    return all(
        frozenset(first(v) for v in facet) | frozenset(second(v) for v in facet)
        in first.codomain
        for facet in first.domain.facets
    )


class _MapSearch:
    def __init__(self, source: SimplicialMap, max_states: int | None):
        if max_states is not None and max_states < 1:
            raise ValueError("max_states must be positive.")
        self.source = source
        self.max_states = max_states
        self.vertices = tuple(source.domain.vertices)
        self.targets = tuple(source.codomain.vertices)
        indices = {vertex: i for i, vertex in enumerate(self.vertices)}
        facets = [tuple(indices[v] for v in facet) for facet in source.domain.facets]
        self.incident = [tuple(face for face in facets if i in face) for i in range(len(indices))]
        self.start = self.encode(source)
        self.parents = {self.start: None}

    def encode(self, map_: SimplicialMap) -> tuple:
        return tuple(map_(vertex) for vertex in self.vertices)

    def decode(self, state: tuple) -> SimplicialMap:
        return SimplicialMap(self.source.domain, self.source.codomain, dict(zip(self.vertices, state)))

    def visit(self) -> Iterator[tuple]:
        queue = deque([self.start])
        yield self.start
        while queue:
            state = queue.popleft()
            for i, facets in enumerate(self.incident):
                images = tuple(frozenset(state[j] for j in face) for face in facets)
                for target in self.targets:
                    if target == state[i]:
                        continue
                    candidate = (*state[:i], target, *state[i + 1:])
                    if candidate in self.parents:
                        continue
                    if all(image | {target} in self.source.codomain for image in images):
                        if self.max_states is not None and len(self.parents) >= self.max_states:
                            raise SearchLimitExceeded("Contiguity search exceeded max_states.")
                        self.parents[candidate] = state
                        queue.append(candidate)
                        yield candidate


def contiguity_class(source: SimplicialMap, *, max_states: int | None = None) -> Iterator[SimplicialMap]:
    """Yield the maps in the contiguity class of source."""
    search = _MapSearch(source, max_states)
    for state in search.visit():
        yield search.decode(state)


def contiguity_chain(
    first: SimplicialMap,
    second: SimplicialMap,
    *,
    max_states: int | None = None,
) -> tuple[SimplicialMap, ...] | None:
    """Return a chain if one exists, with no minimum-length guarantee.

    Directly contiguous maps can be connected by changing their vertex images
    one at a time. Each intermediate simplex image is contained in the union
    of the two endpoint images. Thus single-vertex moves suffice for reachability.
    """
    first._check_parallel(second)
    search = _MapSearch(first, max_states)
    if first == second:
        return (first,)
    if is_contiguous(first, second):
        return first, second
    goal = search.encode(second)
    for state in search.visit():
        if state == goal:
            chain = []
            while state is not None:
                chain.append(search.decode(state))
                state = search.parents[state]
            return tuple(reversed(chain))
    return None


def same_contiguity_class(first: SimplicialMap, second: SimplicialMap, *, max_states: int | None = None) -> bool:
    return contiguity_chain(first, second, max_states=max_states) is not None
