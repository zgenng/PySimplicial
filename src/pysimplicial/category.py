"""Contiguity distance and reduced simplicial LS categories."""

from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from math import inf

from ._errors import CudaUnavailableError, SearchLimitExceeded
from ._solvers.set_cover import _iter_bits, minimum_set_cover
from .collapse import is_strongly_collapsible
from .complex import SimplicialComplex
from .contiguity import _find_chain, _MoveTable, contiguity_chain, is_contiguous, same_contiguity_class
from .maps import SimplicialMap
from .operations import connected_components

Cover = tuple[SimplicialComplex, ...]
Chain = tuple[SimplicialMap, ...]


@dataclass(frozen=True, slots=True)
class CategoryResult:
    value: int | float
    cover: Cover
    chains: tuple[Chain, ...]


@dataclass(frozen=True, slots=True)
class GSCatResult:
    """A checked cover and rigorous bounds for the reduced gscat."""

    lower_bound: int
    upper_bound: int
    cover: Cover
    gpu_used: bool = False
    restarts_completed: int = 0
    method: str = ""
    statistics: Mapping[str, object] = field(default_factory=dict, hash=False)

    @property
    def exact(self) -> bool:
        return self.lower_bound == self.upper_bound

    @property
    def value(self) -> int | None:
        return self.upper_bound if self.exact else None


def is_categorical(
    subcomplex: SimplicialComplex,
    ambient: SimplicialComplex,
    *,
    max_states: int | None = None,
) -> bool:
    """Check if the inclusion belongs to the contiguity class of a constant."""
    inclusion = SimplicialMap.inclusion(subcomplex, ambient)
    if not subcomplex.vertices:
        return True
    base = next(iter(subcomplex.vertices))
    constant = SimplicialMap.constant(subcomplex, ambient, base)
    return same_contiguity_class(inclusion, constant, max_states=max_states)


def _inclusion_move_test(facets):
    incident = {}
    neighbors = {}
    for i, facet in enumerate(facets):
        for vertex in facet:
            incident[vertex] = incident.get(vertex, 0) | (1 << i)
            neighbors.setdefault(vertex, set()).update(facet)
    blockers = []
    for vertex, row in incident.items():
        bad_rows = {row & ~incident[w] for w in neighbors[vertex] - {vertex}}
        blockers.append((row, bad_rows))

    def has_move(mask):
        # With ambient facets as the domain facets, a new image w must
        # already lie in every selected facet incident to the moved vertex.
        for row, bad_rows in blockers:
            selected = mask & row
            if selected and any(not (selected & bad) for bad in bad_rows):
                return True
        return False

    return has_move


def _local_cover(
    domain: SimplicialComplex,
    find_chain: Callable[[SimplicialComplex], Chain | None],
    max_subcomplexes: int | None,
    max_search_states: int | None,
    *,
    inclusion: bool = False,
) -> CategoryResult:
    """Only closures of selected ambient facets are needed for a hereditary test."""
    facets = tuple(domain.facets)
    has_move = _inclusion_move_test(facets) if inclusion and len(facets) > 1 else None
    full = (1 << len(facets)) - 1
    frontier = deque([full])
    seen = {full}
    witnesses: dict[int, tuple[SimplicialComplex, Chain]] = {}
    tested = 0

    def result_for(masks):
        return CategoryResult(
            len(masks) - 1,
            tuple(witnesses[mask][0] for mask in masks),
            tuple(witnesses[mask][1] for mask in masks),
        )

    while frontier:
        mask = frontier.popleft()
        if any(mask & accepted == mask for accepted in witnesses):
            continue
        tested += 1
        if max_subcomplexes is not None and tested > max_subcomplexes:
            raise SearchLimitExceeded("Cover candidate generation exceeded max_subcomplexes.")
        chain = None
        # Singleton facets can be categorical even if no move is necessary.
        if has_move is None or mask.bit_count() == 1 or has_move(mask):
            part = domain if mask == full else SimplicialComplex(facets[i] for i in _iter_bits(mask))
            chain = find_chain(part)
        if chain is not None:
            witnesses[mask] = part, chain
            if mask == full:
                return result_for((mask,))
            for other in witnesses:
                if mask | other == full:
                    return result_for((mask, other))
            continue

        # Expand only rejected masks. Any subset of an accepted mask is
        # already covered by that accepted candidate.
        for index in _iter_bits(mask):
            child = mask ^ (1 << index)
            if child and child not in seen:
                seen.add(child)
                frontier.append(child)

    union = 0
    for mask in witnesses:
        union |= mask
    if union != full:
        return CategoryResult(inf, (), ())
    selected, _ = minimum_set_cover(len(facets), tuple(witnesses), max_search_states=max_search_states)
    return result_for(selected)


def _validate_limit(name: str, limit: int | None) -> None:
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError(f"{name} must be a positive integer.")


def contiguity_distance(
    first: SimplicialMap,
    second: SimplicialMap,
    *,
    max_states: int | None = None,
    max_subcomplexes: int | None = None,
    max_search_states: int | None = None,
) -> CategoryResult:
    """Find a minimum cover on which restrictions are in the same class."""
    first._check_parallel(second)
    _validate_limit("max_states", max_states)
    _validate_limit("max_subcomplexes", max_subcomplexes)
    _validate_limit("max_search_states", max_search_states)
    if not first.domain.vertices:
        return CategoryResult(0, (), ())

    target_components = {
        vertex: index
        for index, component in enumerate(connected_components(first.codomain))
        for vertex in component.vertices
    }
    if any(target_components[first(v)] != target_components[second(v)] for v in first.domain.vertices):
        return CategoryResult(inf, (), ())

    def find_chain(part):
        return contiguity_chain(first.restrict(part), second.restrict(part), max_states=max_states)

    return _local_cover(first.domain, find_chain, max_subcomplexes, max_search_states)


def scat(
    complex_: SimplicialComplex,
    *,
    backend: str = "cpu",
    device_id: int = 0,
    batch_size: int = 1024,
    max_states: int | None = None,
    max_subcomplexes: int | None = None,
    max_search_states: int | None = None,
) -> CategoryResult:
    """Find an exact categorical cover, optionally batching map moves on CUDA.

    Explicit CUDA requires a device and supports up to 64 vertices per
    connected component. Auto uses CPU for small batches or unavailable CUDA.
    """
    if backend not in {"cpu", "cuda", "auto"}:
        raise ValueError("backend must be 'cpu', 'cuda', or 'auto'.")
    if type(device_id) is not int or device_id < 0:
        raise ValueError("device_id must be a nonnegative integer.")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    _validate_limit("max_states", max_states)
    _validate_limit("max_subcomplexes", max_subcomplexes)
    _validate_limit("max_search_states", max_search_states)
    components = tuple(connected_components(complex_))
    use_cuda = False
    if components and backend != "cpu":
        from ._solvers.cuda_contiguity import MAX_VERTICES, _CudaMoves
        from ._solvers.cuda_restarts import _require_cuda

        if backend == "cuda" and any(len(part.vertices) > MAX_VERTICES for part in components):
            raise ValueError(f"scat CUDA supports up to {MAX_VERTICES} vertices per connected component.")
        try:
            _require_cuda(device_id)
        except CudaUnavailableError:
            if backend == "cuda":
                raise
        else:
            use_cuda = True

    cover = []
    chains = []
    for component in components:
        vertex = next(iter(component.vertices))
        moves = _MoveTable(component)
        cuda_moves = (
            _CudaMoves(moves, batch_size=batch_size, auto=backend == "auto")
            if use_cuda and len(component.vertices) <= MAX_VERTICES else None
        )

        def find_chain(part):
            return _find_chain(
                SimplicialMap.inclusion(part, complex_), SimplicialMap.constant(part, complex_, vertex),
                max_states, moves=moves, cuda_moves=cuda_moves,
            )

        result = _local_cover(component, find_chain, max_subcomplexes, max_search_states, inclusion=True)
        if cuda_moves and not all(
            is_contiguous(a, b) for chain in result.chains for a, b in zip(chain, chain[1:])
        ):
            raise RuntimeError("The CPU verifier rejected a CUDA contiguity chain.")
        cover.extend(result.cover)
        chains.extend(result.chains)
    return CategoryResult(max(0, len(cover) - 1), tuple(cover), tuple(chains))


def verify_gscat_cover(complex_: SimplicialComplex, cover: Iterable[SimplicialComplex]) -> bool:
    """Check inclusion, strong collapsibility and coverage of the ambient facets."""
    parts = tuple(cover)
    return all(part.is_subcomplex_of(complex_) and is_strongly_collapsible(part) for part in parts) and all(
        any(facet in part for part in parts) for facet in complex_.facets
    )


def gscat(
    complex_: SimplicialComplex,
    *,
    backend: str = "cpu",
    restarts: int = 256,
    seed: int = 42,
    device_id: int = 0,
    batch_size: int = 8192,
    restart_batch_size: int = 4096,
    max_candidate_states: int | None = None,
    max_search_states: int | None = None,
    progress: Callable[[dict], None] | None = None,
) -> GSCatResult:
    """Compute reduced gscat, optionally using CUDA restarts and exact kernels.

    The CPU also handles graph and strong-collapse shortcuts. Backend "auto"
    uses CUDA when available. Explicit "cuda" requires a device for inputs
    that reach the general search. A budget interruption returns rigorous
    bounds and a verified cover with value=None unless the bounds coincide.
    """
    if backend not in {"cpu", "cuda", "auto"}:
        raise ValueError("backend must be 'cpu', 'cuda', or 'auto'.")
    for name, value in (("restarts", restarts), ("device_id", device_id), ("seed", seed)):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer.")
    if seed >= 1 << 64:
        raise ValueError("seed must fit in an unsigned 64-bit integer.")
    _validate_limit("batch_size", batch_size)
    _validate_limit("restart_batch_size", restart_batch_size)
    _validate_limit("max_candidate_states", max_candidate_states)
    _validate_limit("max_search_states", max_search_states)
    if not complex_.vertices:
        return GSCatResult(0, 0, (), method="empty complex")
    if is_strongly_collapsible(complex_):
        return GSCatResult(0, 0, (complex_,), method="strong collapse")

    from ._solvers.exact import exact_gscat

    if complex_.dimension() <= 1:
        result = exact_gscat(complex_.facets, use_gpu=False)
        cover = tuple(SimplicialComplex(part) for part in result.cover_maximal_simplices)
        return GSCatResult(result.gscat, result.gscat, cover, method=result.method, statistics=result.statistics)

    lower = max(1, len(connected_components(complex_)) - 1)
    incumbent = tuple(SimplicialComplex((facet,)) for facet in complex_.facets)
    use_gpu = False
    completed = 0
    if backend != "cpu":
        from ._solvers import cuda_restarts

        try:
            cuda_restarts._require_cuda(device_id)
        except CudaUnavailableError:
            if backend == "cuda":
                raise
        else:
            use_gpu = True
            if (
                restarts
                and len(complex_.vertices) <= cuda_restarts.MAX_VERTICES
                and len(complex_.facets) <= cuda_restarts.MAX_FACETS
            ):
                incumbent, completed = cuda_restarts.randomized_cover(
                    complex_, restarts=restarts, seed=seed, lower_bound=lower,
                    batch_size=restart_batch_size, progress=progress,
                )
                if not verify_gscat_cover(complex_, incumbent):
                    raise RuntimeError("The CPU verifier rejected the CUDA cover.")

    upper = len(incumbent) - 1
    if upper < lower:
        raise RuntimeError("The computed cover contradicts the lower bound.")
    if upper == lower:
        return GSCatResult(lower, upper, incumbent, bool(completed), completed, "matching bounds")
    try:
        result = exact_gscat(
            complex_.facets, use_gpu=use_gpu, batch_size=batch_size,
            max_candidate_states=max_candidate_states, max_search_states=max_search_states,
            incumbent=incumbent, lower_bound=lower, progress=progress,
        )
    except SearchLimitExceeded as error:
        if error.cover_lower_bound is not None:
            lower = max(lower, error.cover_lower_bound - 1)
        if error.cover_witnesses and len(error.cover_witnesses) < len(incumbent):
            incumbent = tuple(SimplicialComplex(part) for part in error.cover_witnesses)
            upper = len(incumbent) - 1
        if progress:
            progress({"phase": "search_limit", "reason": str(error)})
        return GSCatResult(lower, upper, incumbent, bool(completed) or error.gpu_used, completed, str(error))

    cover = tuple(SimplicialComplex(part) for part in result.cover_maximal_simplices)
    return GSCatResult(
        result.gscat, result.gscat, cover, bool(completed) or result.gpu_used, completed, result.method,
        result.statistics,
    )


def cuda_status() -> dict:
    """Report accelerator availability without launching a kernel."""
    from ._solvers.cuda_restarts import cuda_status as status

    return status()
