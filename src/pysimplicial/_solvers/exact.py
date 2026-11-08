"""Exact cover of a complex by strongly collapsible subcomplexes."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Hashable, Iterable, Iterator, Sequence
from itertools import groupby

import numpy as np


try:  # CUDA is optional; the CPU path remains fully exact.
    from numba import cuda

    _NUMBA_CUDA_IMPORTED = True
except Exception:  # pragma: no cover - depends on the local CUDA installation.
    cuda = None
    _NUMBA_CUDA_IMPORTED = False


from .._errors import SearchLimitExceeded
from .set_cover import _iter_bits, minimum_set_cover


@dataclass(frozen=True)
class ExactGSCATResult:
    """Result and independently checkable cover returned by :func:`exact_gscat`.

    ``gscat`` uses the reduced convention: a cover by ``r`` strongly
    collapsible subcomplexes has value ``r - 1``.
    """

    gscat: int
    cover: tuple[tuple[tuple[Hashable, ...], ...], ...]
    cover_maximal_simplices: tuple[tuple[tuple[Hashable, ...], ...], ...]
    method: str
    exact: bool = True
    gpu_used: bool = False
    statistics: dict[str, Any] = field(default_factory=dict)

    @property
    def cover_count(self) -> int:
        return len(self.cover)


@dataclass(frozen=True)
class _Encoding:
    labels: tuple[Hashable, ...]
    label_to_vertex: dict[Hashable, int]
    facets: tuple[int, ...]
    simplices: tuple[int, ...]
    simplex_to_index: dict[int, int]
    facet_simplex_indices: tuple[int, ...]

    @property
    def dimension(self) -> int:
        return max((x.bit_count() for x in self.facets), default=0) - 1

    @property
    def vertex_count(self) -> int:
        return len(self.labels)


def _maximal_masks(masks: Iterable[int]) -> tuple[int, ...]:
    """Return the inclusion-maximal non-empty masks, deterministically."""
    unique = sorted(set(int(x) for x in masks if x), key=lambda x: (-x.bit_count(), x))
    maximal: list[int] = []
    for simplex in unique:
        if not any(simplex != other and (simplex & other) == simplex for other in maximal):
            maximal.append(simplex)
    return tuple(sorted(maximal, key=lambda x: (x.bit_count(), x)))


def _encode(maximal_simplices: Iterable[Iterable[Hashable]]) -> _Encoding:
    raw: list[tuple[Hashable, ...]] = []
    label_to_vertex: dict[Hashable, int] = {}
    labels: list[Hashable] = []

    for simplex in maximal_simplices:
        values = tuple(dict.fromkeys(simplex))
        if not values:
            continue
        raw.append(values)
        for vertex in values:
            if vertex not in label_to_vertex:
                label_to_vertex[vertex] = len(labels)
                labels.append(vertex)

    if not raw:
        return _Encoding(tuple(), {}, tuple(), tuple(), {}, tuple())

    masks = []
    for simplex in raw:
        mask = 0
        for vertex in simplex:
            mask |= 1 << label_to_vertex[vertex]
        masks.append(mask)

    facets = _maximal_masks(masks)

    closure: set[int] = set()
    for facet in facets:
        face = facet
        while face:
            closure.add(face)
            face = (face - 1) & facet

    simplices = tuple(sorted(closure, key=lambda x: (x.bit_count(), x)))
    simplex_to_index = {simplex: i for i, simplex in enumerate(simplices)}
    facet_simplex_indices = tuple(simplex_to_index[facet] for facet in facets)

    return _Encoding(
        labels=tuple(labels),
        label_to_vertex=label_to_vertex,
        facets=facets,
        simplices=simplices,
        simplex_to_index=simplex_to_index,
        facet_simplex_indices=facet_simplex_indices,
    )


def _decode_simplex(mask: int, labels: Sequence[Hashable]) -> tuple[Hashable, ...]:
    return tuple(labels[i] for i in _iter_bits(mask))


def _normalize_after_vertex_deletions(
    original_maximal_simplices: Sequence[int], active_vertices: int
) -> tuple[int, ...]:
    return _maximal_masks(simplex & active_vertices for simplex in original_maximal_simplices)


def is_strongly_collapsible(maximal_simplex_masks: Iterable[int]) -> bool:
    """Exact CPU test for strong collapsibility from maximal simplex bitmasks."""
    original = _maximal_masks(maximal_simplex_masks)
    if not original:
        return False

    active = 0
    for simplex in original:
        active |= simplex

    while active.bit_count() > 1:
        maximals = _normalize_after_vertex_deletions(original, active)
        deleted = False

        for v in _iter_bits(active):
            v_bit = 1 << v
            # A dominator belongs to every maximal simplex containing v.
            # Their intersection tests all possible dominators at once.
            common = active
            found = False
            for simplex in maximals:
                if simplex & v_bit:
                    found = True
                    common &= simplex
                    if common == v_bit:
                        break
            if found and (common & ~v_bit):
                active ^= v_bit
                deleted = True
                break

        if not deleted:
            return False

    return active != 0


if _NUMBA_CUDA_IMPORTED:  # Definitions are conditional so NumPy-only use works.

    @cuda.jit(device=True)
    def _gpu_popcount(value):
        count = 0
        one = np.uint64(1)
        while value != 0:
            value = value & (value - one)
            count += 1
        return count


    @cuda.jit
    def _strong_collapse_batch_kernel(maxima, counts, vertex_count, output):
        """One exact strong-collapse test per CUDA thread."""
        row = cuda.grid(1)
        if row >= counts.shape[0]:
            return

        face_count = counts[row]
        active = np.uint64(0)
        for j in range(face_count):
            active |= maxima[row, j]

        one = np.uint64(1)

        while True:
            if _gpu_popcount(active) <= 1:
                output[row] = 1 if active != 0 else 0
                return

            deleted = 0

            for v in range(vertex_count):
                v_bit = one << np.uint64(v)
                if (active & v_bit) == 0:
                    continue

                common = active
                has_maximal_with_v = 0
                for j in range(face_count):
                    face_j = maxima[row, j] & active
                    if (face_j & v_bit) == 0:
                        continue

                    is_maximal = 1
                    for k in range(face_count):
                        face_k = maxima[row, k] & active
                        if face_j != face_k and (face_j & face_k) == face_j:
                            is_maximal = 0
                            break

                    if is_maximal == 0:
                        continue
                    has_maximal_with_v = 1
                    common &= face_j
                    if common == v_bit:
                        break

                if has_maximal_with_v == 1 and (common & (active ^ v_bit)) != 0:
                    active ^= v_bit
                    deleted = 1
                    break

            if deleted == 0:
                output[row] = 0
                return


    @cuda.jit
    def _maximal_extension_batch_kernel(
        state_words,
        task_state,
        task_pair,
        rule_offsets,
        rule_output_simplex,
        rule_required_simplex,
        output_words,
    ):
        """Construct one maximal dominated-vertex extension per CUDA thread."""
        task = cuda.grid(1)
        if task >= task_state.shape[0]:
            return

        state_index = task_state[task]
        for word in range(state_words.shape[1]):
            output_words[task, word] = state_words[state_index, word]

        pair = task_pair[task]
        first_rule = rule_offsets[pair]
        last_rule = rule_offsets[pair + 1]
        one = np.uint64(1)

        for rule in range(first_rule, last_rule):
            required = rule_required_simplex[rule]
            required_word = required // 64
            required_bit = required - 64 * required_word

            if (state_words[state_index, required_word] & (one << np.uint64(required_bit))) == 0:
                continue

            output_simplex = rule_output_simplex[rule]
            output_word = output_simplex // 64
            output_bit = output_simplex - 64 * output_word
            output_words[task, output_word] |= one << np.uint64(output_bit)


def cuda_is_available() -> bool:
    if not _NUMBA_CUDA_IMPORTED:
        return False
    try:
        return bool(cuda.is_available())
    except Exception:
        return False


class _StrongCollapseTester:
    def __init__(self, use_gpu: bool, vertex_count: int, threads_per_block: int = 64):
        self.vertex_count = vertex_count
        self.threads_per_block = threads_per_block
        self._use_gpu = bool(use_gpu and vertex_count <= 64 and cuda_is_available())
        self.gpu_used = False

    def test(self, batch: Sequence[Sequence[int]]) -> np.ndarray:
        if not batch:
            return np.empty(0, dtype=np.uint8)

        if not self._use_gpu:
            return np.fromiter(
                (1 if is_strongly_collapsible(row) else 0 for row in batch),
                dtype=np.uint8,
                count=len(batch),
            )

        width = max(len(row) for row in batch)
        maxima = np.zeros((len(batch), width), dtype=np.uint64)
        counts = np.empty(len(batch), dtype=np.int32)

        for i, row in enumerate(batch):
            counts[i] = len(row)
            maxima[i, : len(row)] = np.asarray(row, dtype=np.uint64)

        d_maxima = cuda.to_device(maxima)
        d_counts = cuda.to_device(counts)
        d_output = cuda.device_array(len(batch), dtype=np.uint8)
        blocks = (len(batch) + self.threads_per_block - 1) // self.threads_per_block
        _strong_collapse_batch_kernel[blocks, self.threads_per_block](
            d_maxima, d_counts, self.vertex_count, d_output
        )
        result = d_output.copy_to_host()
        self.gpu_used = True
        return result


def _ideals_to_words(ideals: Sequence[int], word_count: int) -> np.ndarray:
    words = np.zeros((len(ideals), word_count), dtype=np.uint64)
    word_mask = (1 << 64) - 1
    for row, ideal in enumerate(ideals):
        for word in range(word_count):
            words[row, word] = (ideal >> (64 * word)) & word_mask
    return words


def _words_to_ideals(words: np.ndarray) -> list[int]:
    ideals = []
    for row in words:
        ideal = 0
        for word, value in enumerate(row):
            ideal |= int(value) << (64 * word)
        ideals.append(ideal)
    return ideals


@dataclass(frozen=True)
class _ExtensionRules:
    python_rules: tuple[tuple[tuple[int, int], ...], ...]
    offsets: np.ndarray
    output_simplex: np.ndarray
    required_simplex: np.ndarray


def _build_extension_rules(encoding: _Encoding) -> _ExtensionRules:
    """Precompute maximal extensions for every ordered vertex pair (v, u).

    A rule ``(s, r)`` means: when simplex ``r`` belongs to the current
    strongly collapsible subcomplex J, include simplex ``s`` in the maximal
    extension which adds v dominated by u.
    """
    n = encoding.vertex_count
    rules: list[tuple[tuple[int, int], ...]] = []
    flat_output: list[int] = []
    flat_required: list[int] = []
    offsets = [0]

    for v in range(n):
        v_bit = 1 << v
        for u in range(n):
            pair_rules: list[tuple[int, int]] = []
            if u != v:
                u_bit = 1 << u
                for simplex_index, simplex in enumerate(encoding.simplices):
                    if not (simplex & v_bit):
                        continue

                    extended = simplex | u_bit
                    if extended not in encoding.simplex_to_index:
                        continue

                    required_face = (simplex & ~v_bit) | u_bit
                    required_index = encoding.simplex_to_index.get(required_face)
                    if required_index is None:
                        continue

                    pair_rules.append((simplex_index, required_index))
                    flat_output.append(simplex_index)
                    flat_required.append(required_index)

            rules.append(tuple(pair_rules))
            offsets.append(len(flat_output))

    return _ExtensionRules(
        python_rules=tuple(rules),
        offsets=np.asarray(offsets, dtype=np.int64),
        output_simplex=np.asarray(flat_output, dtype=np.int32),
        required_simplex=np.asarray(flat_required, dtype=np.int32),
    )


class _MaximalExtensionEngine:
    """CPU/CUDA engine for exact dominated-vertex extensions."""

    def __init__(
        self,
        encoding: _Encoding,
        rules: _ExtensionRules,
        *,
        use_gpu: bool,
        batch_size: int,
        threads_per_block: int = 128,
    ):
        self.encoding = encoding
        self.rules = rules
        self.batch_size = batch_size
        self.threads_per_block = threads_per_block
        self.word_count = (len(encoding.simplices) + 63) // 64
        self._use_gpu = bool(use_gpu and cuda_is_available())
        self.gpu_used = False
        # Several output simplices can share the same prerequisite.  Group
        # them and compute the arbitrary-width Python bitmasks only once.
        cpu_rules = []
        for pair_rules in rules.python_rules:
            grouped: dict[int, int] = {}
            for output_index, required_index in pair_rules:
                required_bit = 1 << required_index
                grouped[required_bit] = grouped.get(required_bit, 0) | (1 << output_index)
            cpu_rules.append(tuple(grouped.items()))
        self.cpu_rules = tuple(cpu_rules)
        self._device_rules = None

    def _cpu_extend(self, state: int, pair: int) -> int:
        result = state
        for required_bit, output_bits in self.cpu_rules[pair]:
            if state & required_bit:
                result |= output_bits
        return result

    def extend_tasks(
        self,
        states: Sequence[int],
        tasks: Sequence[tuple[int, int]],
    ) -> list[int]:
        """Return maximal extensions for ``(state_index, pair_index)`` tasks."""
        if not tasks:
            return []
        if not self._use_gpu:
            return [self._cpu_extend(states[state], pair) for state, pair in tasks]

        state_words = _ideals_to_words(states, self.word_count)
        d_states = cuda.to_device(state_words)
        # These arrays are immutable for the lifetime of this engine.
        if self._device_rules is None:
            self._device_rules = (
                cuda.to_device(self.rules.offsets),
                cuda.to_device(self.rules.output_simplex),
                cuda.to_device(self.rules.required_simplex),
            )
        d_offsets, d_output_simplex, d_required_simplex = self._device_rules

        result: list[int] = []
        for start in range(0, len(tasks), self.batch_size):
            chunk = tasks[start : start + self.batch_size]
            task_state = np.fromiter((x[0] for x in chunk), dtype=np.int32, count=len(chunk))
            task_pair = np.fromiter((x[1] for x in chunk), dtype=np.int32, count=len(chunk))
            d_task_state = cuda.to_device(task_state)
            d_task_pair = cuda.to_device(task_pair)
            d_output = cuda.device_array((len(chunk), self.word_count), dtype=np.uint64)
            blocks = (len(chunk) + self.threads_per_block - 1) // self.threads_per_block

            _maximal_extension_batch_kernel[blocks, self.threads_per_block](
                d_states,
                d_task_state,
                d_task_pair,
                d_offsets,
                d_output_simplex,
                d_required_simplex,
                d_output,
            )
            result.extend(_words_to_ideals(d_output.copy_to_host()))
            self.gpu_used = True

        return result


def _face_poset_relations(encoding: _Encoding) -> tuple[tuple[int, ...], tuple[int, ...]]:
    simplices = encoding.simplices
    count = len(simplices)
    up = [0] * count
    down = [0] * count

    for i, small in enumerate(simplices):
        for j, large in enumerate(simplices):
            if (small & large) == small:
                up[i] |= 1 << j
                down[j] |= 1 << i

    return tuple(up), tuple(down)


def _choose_ideal_branch_element(available: int, up: Sequence[int], down: Sequence[int]) -> int:
    """Choose a deterministic, approximately balanced branch in the face poset."""
    best_index = -1
    best_score = -1
    for i in _iter_bits(available):
        above = (available & up[i]).bit_count()
        below = (available & down[i]).bit_count()
        score = min(above, below)
        if score > best_score:
            best_score = score
            best_index = i
    return best_index


def _iter_order_ideals(up: Sequence[int], down: Sequence[int]) -> Iterator[int]:
    """Enumerate every order ideal of a finite poset exactly once."""
    full = (1 << len(up)) - 1

    def visit(available: int, included: int) -> Iterator[int]:
        if available == 0:
            yield included
            return

        i = _choose_ideal_branch_element(available, up, down)

        # Excluding i forces exclusion of every element above i.
        yield from visit(available & ~up[i], included)

        # Including i forces inclusion of every still-available element below i.
        forced = available & down[i]
        yield from visit(available & ~forced, included | forced)

    yield from visit(full, 0)


def _maximal_faces_of_ideal(ideal: int, strict_up: Sequence[int], simplices: Sequence[int]) -> tuple[int, ...]:
    result = []
    for i in _iter_bits(ideal):
        if (ideal & strict_up[i]) == 0:
            result.append(simplices[i])
    return tuple(result)


def _facet_coverage(ideal: int, facet_simplex_indices: Sequence[int]) -> int:
    coverage = 0
    for j, simplex_index in enumerate(facet_simplex_indices):
        if ideal & (1 << simplex_index):
            coverage |= 1 << j
    return coverage


def _maximal_masks_indexed(ideals):
    """As above, omitting equal-size masks from superset queries."""
    ordered = sorted(set(int(x) for x in ideals if x), key=lambda x: (-x.bit_count(), x))
    return _maximal_ordered_grouped_bitsets(ordered)


def _maximal_ordered_grouped_bitsets(ordered):
    if not ordered:
        return ()
    minimum_size = ordered[-1].bit_count()
    if ordered[0].bit_count() == minimum_size:
        return tuple(ordered)
    postings = [0] * max(x.bit_length() for x in ordered)
    counts = [0] * len(postings)
    kept = []
    for size, group in groupby(ordered, int.bit_count):
        additions = []
        for candidate in group:
            bits = tuple(_iter_bits(candidate))
            rare = min(bits, key=counts.__getitem__)
            possible = postings[rare]
            if possible:
                for bit in bits:
                    possible &= postings[bit]
                    if not possible:
                        break
            if not possible:
                additions.append((candidate, bits))
        if size == minimum_size:
            kept.extend(candidate for candidate, _ in additions)
            break
        for candidate, bits in additions:
            flag = 1 << len(kept)
            kept.append(candidate)
            for bit in bits:
                postings[bit] |= flag
                counts[bit] += 1
    return tuple(kept)


def _maximal_level_ideals(ideals, vertex_count):
    """Filter one fixed-vertex-count level, grouping by vertex support.

    The first ``vertex_count`` simplex positions must be the singleton faces.
    All nonzero input ideals must use the same number of vertices. Otherwise
    this routine falls back to global filtering to preserve exactness.
    """
    mask = (1 << vertex_count) - 1
    groups = {}
    support_sizes = set()
    for ideal in ideals:
        ideal = int(ideal)
        if ideal:
            support = ideal & mask
            support_sizes.add(support.bit_count())
            groups.setdefault(support, set()).add(ideal)
    if len(support_sizes) > 1:
        return _maximal_masks_indexed(x for group in groups.values() for x in group)
    kept = []
    for support, group in groups.items():
        if len(group) == 1:
            kept.extend(group)
            continue
        # Every mask in this group has these singleton bits. Removing them
        # preserves inclusion, cardinality ordering and numeric tie-breaking.
        kept.extend(ideal | support for ideal in _maximal_masks_indexed(
            ideal ^ support for ideal in group
        ))
    return tuple(sorted(kept, key=lambda x: (-x.bit_count(), x)))


def _maximal_coverages(coverages: Iterable[int]) -> tuple[int, ...]:
    """Remove coverages contained in another mask, using inverted bitsets."""
    return _maximal_masks_indexed(coverages)


def _maximal_ideals(ideals: Iterable[int]) -> tuple[int, ...]:
    """Keep inclusion-maximal ideals; preserve the original deterministic order."""
    return _maximal_masks_indexed(ideals)


def _generate_candidates_by_collapse_certificates(
    encoding: _Encoding,
    *,
    use_gpu: bool,
    batch_size: int,
    max_candidate_states: int | None,
    progress: Callable[[dict], None] | None,
) -> tuple[dict[int, int], dict[str, Any], bool]:
    """Generate a complete dominating family of strongly collapsible subcomplexes.

    Let J be strongly collapsible, let v be absent from J, and let u be a
    vertex of J.  There is a unique largest subcomplex L of K such that

    * L without v equals J; and
    * v is dominated by u in L.

    It contains exactly those v-simplices sigma for which both sigma union u
    is a simplex of K and (sigma without v) union u belongs to J.

    Starting with one-point complexes and repeatedly taking every such maximal
    extension is complete for gscat: reverse any strong-collapse sequence of
    any candidate U.  At each reversed step the generated maximal extension
    contains the corresponding partial U.  Hence U is contained in a generated
    terminal candidate, which can replace U in a cover.  This proves that the
    family returned here is sufficient for an exact minimum-cover proof without
    enumerating every order ideal of the face poset.
    """
    started = perf_counter()
    n = encoding.vertex_count
    rules = _build_extension_rules(encoding)
    engine = _MaximalExtensionEngine(
        encoding,
        rules,
        use_gpu=use_gpu,
        batch_size=batch_size,
    )

    vertex_ideal_bits = tuple(
        1 << encoding.simplex_to_index[1 << vertex] for vertex in range(n)
    )
    current = tuple(vertex_ideal_bits)
    terminal: set[int] = set()
    generated_total = len(current)
    attempted_extensions = 0
    level_sizes: list[int] = []
    extension_seconds = 0.0
    dominance_seconds = 0.0

    for level in range(1, n + 1):
        if not current:
            break
        level_sizes.append(len(current))

        states = list(current)
        tasks: list[tuple[int, int]] = []

        for state_index, state in enumerate(states):
            vertices = 0
            for vertex, singleton_bit in enumerate(vertex_ideal_bits):
                if state & singleton_bit:
                    vertices |= 1 << vertex

            for v in range(n):
                if vertices & (1 << v):
                    continue
                for u in _iter_bits(vertices):
                    pair = v * n + u
                    if rules.python_rules[pair]:
                        tasks.append((state_index, pair))

        extension_started = perf_counter()
        outputs = engine.extend_tasks(states, tasks)
        extension_seconds += perf_counter() - extension_started
        attempted_extensions += len(tasks)
        expanded_states: set[int] = set()
        next_states: set[int] = set()

        for (state_index, pair), extension in zip(tasks, outputs):
            state = states[state_index]
            if extension == state:
                continue

            v = pair // n
            singleton_bit = vertex_ideal_bits[v]
            if not (extension & singleton_bit):
                raise RuntimeError(
                    "Internal extension error: a non-trivial extension did not add its vertex."
                )

            expanded_states.add(state_index)
            next_states.add(extension)

        for state_index, state in enumerate(states):
            if state_index not in expanded_states:
                terminal.add(state)

        dominance_started = perf_counter()
        current = _maximal_level_ideals(next_states, n)
        dominance_seconds += perf_counter() - dominance_started
        generated_total += len(current)
        if max_candidate_states is not None and generated_total > max_candidate_states:
            raise SearchLimitExceeded(
                "Collapse-certificate generation exceeded max_candidate_states="
                f"{max_candidate_states:,}.",
                gpu_used=engine.gpu_used,
            )

        if progress:
            elapsed = perf_counter() - started
            progress({
                "phase": "exact_candidates", "level": level,
                "states": len(states), "next_states": len(current),
                "terminal": len(terminal), "extensions": attempted_extensions,
                "seconds": elapsed,
            })

    terminal.update(current)
    dominance_started = perf_counter()
    maximal_terminal = _maximal_ideals(terminal)
    dominance_seconds += perf_counter() - dominance_started

    # No further extensions follow this stage.  It is now safe to identify
    # candidates by their facet coverage, which would NOT be safe inside the
    # dynamic program (lower faces can affect later extensions).
    witnesses: dict[int, int] = {}
    for ideal in maximal_terminal:
        coverage = _facet_coverage(ideal, encoding.facet_simplex_indices)
        if coverage:
            previous = witnesses.get(coverage)
            if previous is None or previous.bit_count() < ideal.bit_count():
                witnesses[coverage] = ideal
    coverage_count_before_dominance = len(witnesses)
    retained_coverages = _maximal_coverages(witnesses)
    witnesses = {coverage: witnesses[coverage] for coverage in retained_coverages}

    # Validate every retained witness independently before it enters the
    # minimum-cover proof.  Removed coverages are contained in retained ones
    # and cannot improve a cover.  Final chosen witnesses are checked on CPU
    # again by _verify_general_cover, including when generation uses CUDA.
    validation_started = perf_counter()
    up, _ = _face_poset_relations(encoding)
    strict_up = tuple(up[i] & ~(1 << i) for i in range(len(up)))
    retained_ideals = tuple(witnesses.values())
    validator = _StrongCollapseTester(
        use_gpu=use_gpu,
        vertex_count=encoding.vertex_count,
    )
    for start in range(0, len(retained_ideals), batch_size):
        maxima = [
            _maximal_faces_of_ideal(ideal, strict_up, encoding.simplices)
            for ideal in retained_ideals[start : start + batch_size]
        ]
        valid = validator.test(maxima)
        if not np.all(valid == 1):
            bad = start + int(np.flatnonzero(valid != 1)[0])
            raise RuntimeError(
                "A collapse-certificate candidate failed independent validation "
                f"at index {bad}. The exact computation was aborted."
            )

    stats = {
        "collapse_certificate_states": generated_total,
        "collapse_certificate_terminal_states": len(maximal_terminal),
        "collapse_certificate_extensions": attempted_extensions,
        "collapse_certificate_level_sizes": level_sizes,
        "candidate_coverages_before_dominance": coverage_count_before_dominance,
        "candidate_witnesses_validated": len(retained_ideals),
        "extension_seconds": extension_seconds,
        "dominance_seconds": dominance_seconds,
        "candidate_validation_seconds": perf_counter() - validation_started,
        "candidate_generation_seconds": perf_counter() - started,
    }
    return witnesses, stats, bool(engine.gpu_used or validator.gpu_used)


def _enumerate_strongly_collapsible_candidates(
    encoding: _Encoding,
    *,
    use_gpu: bool,
    batch_size: int,
    max_subcomplexes: int | None,
    progress: Callable[[dict], None] | None,
) -> tuple[dict[int, int], dict[str, Any], bool]:
    up, down = _face_poset_relations(encoding)
    strict_up = tuple(up[i] & ~(1 << i) for i in range(len(up)))
    tester = _StrongCollapseTester(use_gpu=use_gpu, vertex_count=encoding.vertex_count)

    witnesses: dict[int, int] = {}

    # Every simplex is strongly collapsible.  Seeding these witnesses both
    # supplies an immediate finite upper bound and avoids retesting them.
    for j, simplex_index in enumerate(encoding.facet_simplex_indices):
        witnesses[1 << j] = down[simplex_index]

    # Closed stars are cones and therefore strongly collapsible.
    for vertex in range(encoding.vertex_count):
        coverage = 0
        ideal = 0
        vertex_bit = 1 << vertex
        for j, facet in enumerate(encoding.facets):
            if facet & vertex_bit:
                coverage |= 1 << j
                ideal |= down[encoding.facet_simplex_indices[j]]
        if coverage:
            witnesses.setdefault(coverage, ideal)

    batch_maxima: list[tuple[int, ...]] = []
    batch_ideals: list[int] = []
    batch_coverages: list[int] = []

    enumerated = 0
    tested = 0
    gpu_positive = 0
    started = perf_counter()

    def flush() -> None:
        nonlocal tested, gpu_positive
        if not batch_maxima:
            return

        results = tester.test(batch_maxima)
        tested += len(batch_maxima)

        for valid, coverage, ideal, maxima in zip(
            results, batch_coverages, batch_ideals, batch_maxima
        ):
            if not valid:
                continue

            gpu_positive += 1
            # A positive CUDA result is independently verified on the CPU.
            if not is_strongly_collapsible(maxima):
                raise RuntimeError(
                    "CUDA/CPU disagreement in the strong-collapse test; "
                    "the exact computation was aborted."
                )
            witnesses.setdefault(coverage, ideal)

        batch_maxima.clear()
        batch_ideals.clear()
        batch_coverages.clear()

        if progress:
            elapsed = perf_counter() - started
            rate = enumerated / elapsed if elapsed else 0.0
            progress({
                "phase": "all_subcomplexes", "enumerated": enumerated,
                "tested": tested, "candidate_coverages": len(witnesses), "rate": rate,
            })

    for ideal in _iter_order_ideals(up, down):
        if ideal == 0:
            continue

        enumerated += 1
        if max_subcomplexes is not None and enumerated > max_subcomplexes:
            raise SearchLimitExceeded(
                "Subcomplex enumeration exceeded max_subcomplexes="
                f"{max_subcomplexes:,}.",
                gpu_used=tester.gpu_used,
            )

        coverage = _facet_coverage(ideal, encoding.facet_simplex_indices)
        if coverage == 0 or coverage in witnesses:
            continue

        maxima = _maximal_faces_of_ideal(ideal, strict_up, encoding.simplices)
        batch_maxima.append(maxima)
        batch_ideals.append(ideal)
        batch_coverages.append(coverage)

        if len(batch_maxima) >= batch_size:
            flush()

    flush()

    stats = {
        "subcomplexes_enumerated": enumerated,
        "subcomplexes_strong_tested": tested,
        "positive_tests": gpu_positive,
        "candidate_coverages_before_dominance": len(witnesses),
        "enumeration_seconds": perf_counter() - started,
    }
    return witnesses, stats, tester.gpu_used


def _ideal_to_simplices(ideal: int, encoding: _Encoding) -> tuple[tuple[Hashable, ...], ...]:
    return tuple(
        _decode_simplex(encoding.simplices[i], encoding.labels)
        for i in _iter_bits(ideal)
    )


def _ideal_to_maximal_simplices(ideal: int, encoding: _Encoding) -> tuple[tuple[Hashable, ...], ...]:
    up, _ = _face_poset_relations(encoding)
    strict_up = tuple(up[i] & ~(1 << i) for i in range(len(up)))
    maxima = _maximal_faces_of_ideal(ideal, strict_up, encoding.simplices)
    return tuple(_decode_simplex(simplex, encoding.labels) for simplex in maxima)


def _verify_general_cover(encoding: _Encoding, ideals: Sequence[int]) -> None:
    full = (1 << len(encoding.simplices)) - 1
    union = 0
    up, _ = _face_poset_relations(encoding)
    strict_up = tuple(up[i] & ~(1 << i) for i in range(len(up)))

    for ideal in ideals:
        union |= ideal
        maxima = _maximal_faces_of_ideal(ideal, strict_up, encoding.simplices)
        if not is_strongly_collapsible(maxima):
            raise RuntimeError("Final cover verification found a non-collapsible element.")

    if union != full:
        raise RuntimeError("Final cover verification found uncovered simplices.")


# ---------------------------------------------------------------------------
# Exact graph branch: graphic-matroid partition and a minimum tree cover.
# ---------------------------------------------------------------------------


def _fundamental_cycle_edges(forest: Sequence[tuple[int, int]], edge: tuple[int, int]):
    a, b = edge
    adjacency: dict[int, list[tuple[int, tuple[int, int]]]] = {}
    for u, v in forest:
        adjacency.setdefault(u, []).append((v, (u, v)))
        adjacency.setdefault(v, []).append((u, (u, v)))

    if a not in adjacency or b not in adjacency:
        return None

    stack = [a]
    seen = {a}
    parent: dict[int, tuple[int, tuple[int, int]]] = {}

    while stack:
        u = stack.pop()
        if u == b:
            break
        for v, stored_edge in adjacency.get(u, []):
            if v not in seen:
                seen.add(v)
                parent[v] = (u, stored_edge)
                stack.append(v)

    if b not in seen:
        return None

    path = []
    current = b
    while current != a:
        previous, stored_edge = parent[current]
        path.append(stored_edge)
        current = previous
    return path


def _matroid_augmenting_path(
    edge: tuple[int, int], forests: Sequence[Sequence[tuple[int, int]]]
):
    from collections import deque

    if not forests:
        return None

    predecessor: dict[tuple[int, int], tuple[tuple[int, int], int] | None] = {edge: None}
    queue = deque([edge])
    terminal_edge = None
    terminal_forest = None

    while queue:
        current = queue.popleft()
        for i, forest in enumerate(forests):
            cycle = _fundamental_cycle_edges(forest, current)
            if cycle is None:
                terminal_edge = current
                terminal_forest = i
                queue.clear()
                break
            for exchanged in cycle:
                if exchanged not in predecessor:
                    predecessor[exchanged] = (current, i)
                    queue.append(exchanged)
        if terminal_edge is not None:
            break

    if terminal_edge is None:
        return None

    path = [terminal_edge]
    labels = []
    current = terminal_edge
    while predecessor[current] is not None:
        previous, forest_index = predecessor[current]
        labels.append(forest_index)
        path.append(previous)
        current = previous
    path.reverse()
    labels.reverse()

    transitions = [(terminal_forest, terminal_edge, None)]
    for j in range(len(labels) - 1, -1, -1):
        transitions.append((labels[j], path[j], path[j + 1]))
    return transitions


def _is_forest(vertices: Sequence[int], edges: Iterable[tuple[int, int]]) -> bool:
    parent = {v: v for v in vertices}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        parent[ra] = rb
    return True


def _matroid_partition(
    vertices: Sequence[int], edges: Iterable[tuple[int, int]]
) -> tuple[int, list[list[tuple[int, int]]]]:
    all_edges = set(edges)
    forests: list[list[tuple[int, int]]] = []

    for edge in sorted(all_edges):
        transitions = _matroid_augmenting_path(edge, forests)
        if transitions is None:
            forests.append([edge])
            continue

        for i, add_edge, remove_edge in transitions:
            if remove_edge is not None:
                forests[i].remove(remove_edge)
            if add_edge not in forests[i]:
                forests[i].append(add_edge)

        if any(not _is_forest(vertices, forest) for forest in forests):
            raise RuntimeError("Internal graphic-matroid partition error.")

    partition_edges = set().union(*(set(forest) for forest in forests)) if forests else set()
    if partition_edges != all_edges:
        raise RuntimeError("Graphic-matroid partition did not preserve every edge.")
    return len(forests), forests


def _connected_components(vertices: Sequence[int], edges: set[tuple[int, int]]) -> list[set[int]]:
    adjacency = {v: set() for v in vertices}
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)

    components = []
    unseen = set(vertices)
    while unseen:
        start = min(unseen)
        stack = [start]
        unseen.remove(start)
        component = {start}
        while stack:
            v = stack.pop()
            for u in adjacency[v]:
                if u in unseen:
                    unseen.remove(u)
                    component.add(u)
                    stack.append(u)
        components.append(component)
    return components


def _extend_forest_to_spanning_tree(
    vertices: Sequence[int], component_edges: set[tuple[int, int]], forest: Sequence[tuple[int, int]]
) -> set[tuple[int, int]]:
    parent = {v: v for v in vertices}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        parent[ra] = rb
        return True

    tree = set(forest)
    for a, b in forest:
        if not union(a, b):
            raise RuntimeError("Matroid partition returned a cyclic part.")

    for edge in sorted(component_edges):
        if len(tree) == len(vertices) - 1:
            break
        if union(*edge):
            tree.add(edge)

    if len(vertices) > 1 and len(tree) != len(vertices) - 1:
        raise RuntimeError("Could not extend a forest to a spanning tree.")
    return tree


def _exact_graph_gscat(encoding: _Encoding) -> ExactGSCATResult:
    vertices = list(range(encoding.vertex_count))
    edges = {
        tuple(_iter_bits(simplex))
        for simplex in encoding.simplices
        if simplex.bit_count() == 2
    }
    components = _connected_components(vertices, edges)
    cover_masks: list[set[int]] = []

    for component in components:
        component_edges = {
            edge for edge in edges if edge[0] in component and edge[1] in component
        }

        if not component_edges:
            cover_masks.append({1 << next(iter(component))})
            continue

        raw_arboricity, forests = _matroid_partition(sorted(component), component_edges)
        if raw_arboricity == 0:
            raise RuntimeError("A non-empty connected graph received arboricity zero.")

        for forest in forests:
            tree = _extend_forest_to_spanning_tree(
                sorted(component), component_edges, forest
            )
            simplices = {1 << v for v in component}
            simplices.update((1 << a) | (1 << b) for a, b in tree)
            cover_masks.append(simplices)

    target = set(encoding.simplices)
    union = set().union(*cover_masks) if cover_masks else set()
    if union != target:
        raise RuntimeError("Final graph cover verification failed.")

    cover = tuple(
        tuple(
            _decode_simplex(simplex, encoding.labels)
            for simplex in sorted(part, key=lambda x: (x.bit_count(), x))
        )
        for part in cover_masks
    )
    cover_maximal = tuple(
        tuple(
            _decode_simplex(simplex, encoding.labels)
            for simplex in _maximal_masks(part)
        )
        for part in cover_masks
    )

    return ExactGSCATResult(
        gscat=max(0, len(cover) - 1),
        cover=cover,
        cover_maximal_simplices=cover_maximal,
        method="exact graphic-matroid partition",
        exact=True,
        gpu_used=False,
        statistics={
            "vertices": encoding.vertex_count,
            "edges": len(edges),
            "connected_components": len(components),
            "cover_elements": len(cover),
        },
    )


def exact_gscat(
    maximal_simplices: Iterable[Iterable[Hashable]],
    *,
    use_gpu: bool = True,
    algorithm: str = "collapse_certificates",
    batch_size: int = 8192,
    max_candidate_states: int | None = None,
    max_subcomplexes: int | None = None,
    max_search_states: int | None = None,
    progress: Callable[[dict], None] | None = None,
    incumbent=(),
    lower_bound: int = 0,
) -> ExactGSCATResult:
    """Compute the exact geometric simplicial category.

    Parameters
    ----------
    maximal_simplices:
        An iterable of maximal simplices.  Faces may also be present; redundant
        non-maximal simplices are removed automatically.
    use_gpu:
        Use CUDA for batched strong-collapse tests when available.  Falling
        back to CPU does not change the answer.
    algorithm:
        ``"collapse_certificates"`` (default) uses the exact maximal-extension
        dynamic program.  ``"all_subcomplexes"`` independently enumerates
        every order ideal of the face poset and is intended for small-instance
        cross-checks.
    batch_size:
        Maximum number of independent CUDA tasks per batch.
    max_candidate_states:
        Optional safety limit for collapse-certificate generation.
    max_subcomplexes:
        Optional safety limit for ``algorithm="all_subcomplexes"``.
    max_search_states:
        Optional safety limit for the exact set-cover proof.  Reaching it also
        raises SearchLimitExceeded without returning a value.
    progress:
        Optional callback receiving exact-search progress events.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if algorithm not in {"collapse_certificates", "all_subcomplexes"}:
        raise ValueError(
            "algorithm must be 'collapse_certificates' or 'all_subcomplexes'."
        )
    if max_candidate_states is not None and max_candidate_states <= 0:
        raise ValueError("max_candidate_states must be positive or None.")
    if max_subcomplexes is not None and max_subcomplexes <= 0:
        raise ValueError("max_subcomplexes must be positive or None.")
    if max_search_states is not None and max_search_states <= 0:
        raise ValueError("max_search_states must be positive or None.")

    started = perf_counter()
    encoding = _encode(maximal_simplices)

    if not encoding.facets:
        return ExactGSCATResult(
            gscat=0,
            cover=tuple(),
            cover_maximal_simplices=tuple(),
            method="empty complex",
            exact=True,
            gpu_used=False,
            statistics={"seconds": perf_counter() - started},
        )

    if encoding.dimension <= 1:
        result = _exact_graph_gscat(encoding)
        result.statistics["seconds"] = perf_counter() - started
        return result

    # The zero case can be decided immediately and exactly.
    if is_strongly_collapsible(encoding.facets):
        ideal = (1 << len(encoding.simplices)) - 1
        cover = (_ideal_to_simplices(ideal, encoding),)
        cover_maximal = (
            tuple(_decode_simplex(simplex, encoding.labels) for simplex in encoding.facets),
        )
        return ExactGSCATResult(
            gscat=0,
            cover=cover,
            cover_maximal_simplices=cover_maximal,
            method="exact strong-collapse test",
            exact=True,
            gpu_used=False,
            statistics={
                "vertices": encoding.vertex_count,
                "simplices": len(encoding.simplices),
                "maximal_simplices": len(encoding.facets),
                "seconds": perf_counter() - started,
            },
        )

    if algorithm == "collapse_certificates":
        witnesses, enumeration_stats, gpu_used = (
            _generate_candidates_by_collapse_certificates(
                encoding,
                use_gpu=use_gpu,
                batch_size=batch_size,
                max_candidate_states=max_candidate_states,
                progress=progress,
            )
        )
        method = "exact maximal strong-collapse certificates + integer set cover"
    else:
        witnesses, enumeration_stats, gpu_used = _enumerate_strongly_collapsible_candidates(
            encoding,
            use_gpu=use_gpu,
            batch_size=batch_size,
            max_subcomplexes=max_subcomplexes,
            progress=progress,
        )
        method = "all-subcomplex enumeration + exact integer set cover"

    maximal_coverages = _maximal_coverages(witnesses)
    incumbent_masks = tuple(
        sum(1 << index for index, mask in enumerate(encoding.facets)
            if _decode_simplex(mask, encoding.labels) in part)
        for part in incumbent
    )
    if progress:
        progress({"phase": "exact_set_cover", "candidates": len(maximal_coverages)})
    try:
        solution_coverages, set_cover_stats = minimum_set_cover(
            len(encoding.facets),
            maximal_coverages,
            max_search_states=max_search_states,
            incumbent=incumbent_masks,
            lower_bound=lower_bound + 1,
        )
    except SearchLimitExceeded as error:
        error.gpu_used = gpu_used
        best_ideals = tuple(witnesses[coverage] for coverage in error.cover_masks)
        _verify_general_cover(encoding, best_ideals)
        error.cover_witnesses = tuple(
            _ideal_to_maximal_simplices(ideal, encoding) for ideal in best_ideals
        )
        raise
    solution_ideals = tuple(witnesses[coverage] for coverage in solution_coverages)
    _verify_general_cover(encoding, solution_ideals)

    cover = tuple(_ideal_to_simplices(ideal, encoding) for ideal in solution_ideals)
    cover_maximal = tuple(
        _ideal_to_maximal_simplices(ideal, encoding) for ideal in solution_ideals
    )

    statistics = {
        "vertices": encoding.vertex_count,
        "simplices": len(encoding.simplices),
        "maximal_simplices": len(encoding.facets),
        **enumeration_stats,
        "candidate_coverages_after_dominance": len(maximal_coverages),
        "set_cover_states": set_cover_stats.states,
        "set_cover_lower_bound": set_cover_stats.lower_bound,
        "greedy_upper_bound": set_cover_stats.greedy_upper_bound,
        "cover_elements": len(solution_ideals),
        "seconds": perf_counter() - started,
    }

    return ExactGSCATResult(
        gscat=len(solution_ideals) - 1,
        cover=cover,
        cover_maximal_simplices=cover_maximal,
        method=method,
        exact=True,
        gpu_used=gpu_used,
        statistics=statistics,
    )


