"""Direct contiguity and chains in the graph of simplicial maps."""

from collections import deque
from collections.abc import Iterator

from ._errors import SearchLimitExceeded
from ._solvers.set_cover import _iter_bits
from .maps import SimplicialMap


def is_contiguous(first: SimplicialMap, second: SimplicialMap) -> bool:
    first._check_parallel(second)
    return all(
        frozenset(first(v) for v in facet) | frozenset(second(v) for v in facet)
        in first.codomain
        for facet in first.domain.facets
    )


class _MoveTable:
    """Encode a target complex once for all domains searched against it."""

    def __init__(self, target):
        self.vertices = tuple(target.vertices)
        self.indices = {vertex: i for i, vertex in enumerate(self.vertices)}
        self.bits = tuple(1 << i for i in range(len(self.vertices)))
        self.full = (1 << len(self.vertices)) - 1
        self.facets = tuple(sum(self.bits[self.indices[v]] for v in facet) for facet in target.facets)
        self._extensions: dict[int, int] = {}

    def extensions(self, image: int) -> int:
        """Return all w such that image union {w} is a simplex."""
        if image not in self._extensions:
            allowed = 0
            for facet in self.facets:
                if image & facet == image:
                    allowed |= facet
            self._extensions[image] = allowed
        return self._extensions[image]


class _MapSearch:
    def __init__(self, source: SimplicialMap, max_states: int | None, moves=None, cuda_moves=None):
        if max_states is not None and max_states < 1:
            raise ValueError("max_states must be positive.")
        self.source = source
        self.max_states = max_states
        self.vertices = tuple(source.domain.vertices)
        self.moves = _MoveTable(source.codomain) if moves is None else moves
        indices = {vertex: i for i, vertex in enumerate(self.vertices)}
        self.facets = tuple(tuple(indices[v] for v in facet) for facet in source.domain.facets)
        incident = [[] for _ in self.vertices]
        for face_index, face in enumerate(self.facets):
            for i in face:
                incident[i].append(face_index)
        self.incident = tuple(tuple(row) for row in incident)
        self.cuda_moves = cuda_moves.for_domain(self.facets, self.incident) if cuda_moves else None
        self.batch_size = cuda_moves.batch_size if cuda_moves else 1
        self.start = self.encode(source)
        self.parents = {self.start: None}

    def encode(self, map_: SimplicialMap) -> tuple[int, ...]:
        return tuple(self.moves.indices[map_(vertex)] for vertex in self.vertices)

    def decode(self, state: tuple[int, ...]) -> SimplicialMap:
        images = {vertex: self.moves.vertices[target] for vertex, target in zip(self.vertices, state)}
        return SimplicialMap(self.source.domain, self.source.codomain, images)

    def _cpu_moves(self, state: tuple[int, ...]) -> tuple[int, ...]:
        images = []
        for face in self.facets:
            image = 0
            for i in face:
                image |= self.moves.bits[state[i]]
            images.append(self.moves.extensions(image))
        result = []
        for i, faces in enumerate(self.incident):
            allowed = self.moves.full
            for face_index in faces:
                allowed &= images[face_index]
            result.append(allowed & ~self.moves.bits[state[i]])
        return tuple(result)

    def visit(self) -> Iterator[tuple[int, ...]]:
        queue = deque([self.start])
        yield self.start
        while queue:
            states = [queue.popleft() for _ in range(min(len(queue), self.batch_size))]
            rows = self.cuda_moves(states) if self.cuda_moves else None
            if rows is None:
                rows = map(self._cpu_moves, states)
            for state, row in zip(states, rows):
                for i, allowed in enumerate(row):
                    for target in _iter_bits(int(allowed)):
                        candidate = (*state[:i], target, *state[i + 1:])
                        if candidate in self.parents:
                            continue
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
    return _find_chain(first, second, max_states)


def _find_chain(first, second, max_states, *, moves=None, cuda_moves=None):
    first._check_parallel(second)
    if max_states is not None and max_states < 1:
        raise ValueError("max_states must be positive.")
    if first == second:
        return (first,)
    if is_contiguous(first, second):
        return first, second
    search = _MapSearch(first, max_states, moves, cuda_moves)
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
