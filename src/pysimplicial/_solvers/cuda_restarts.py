"""Randomized CUDA search for a small geometric cover."""

from time import monotonic

import numpy as np

from .._errors import CudaUnavailableError
from ..collapse import _vertex_key
from ..complex import SimplicialComplex

try:
    from numba import cuda, int16, int32, uint8, uint64
except Exception as error:  # An optional accelerator can fail to load.
    cuda = None
    _NUMBA_IMPORT_ERROR = error
else:
    _NUMBA_IMPORT_ERROR = None


MAX_VERTICES = 64
MAX_FACETS = 512

_NEIGHBOR_NONE = 0
_NEIGHBOR_PREFER = 1
_NEIGHBOR_STRICT = 2
_NEIGHBOR_CODES = {
    "none": _NEIGHBOR_NONE,
    "prefer": _NEIGHBOR_PREFER,
    "strict": _NEIGHBOR_STRICT,
}


def cuda_status() -> dict[str, bool | str | int]:
    """Return dependency/device status without launching a kernel."""

    if cuda is None:
        return {
            "numba_imported": False,
            "cuda_available": False,
            "reason": repr(_NUMBA_IMPORT_ERROR),
            "max_vertices": MAX_VERTICES,
            "max_facets": MAX_FACETS,
        }
    try:
        available = bool(cuda.is_available())
    except Exception as error:  # Driver initialization failures vary by host.
        return {
            "numba_imported": True,
            "cuda_available": False,
            "reason": repr(error),
            "max_vertices": MAX_VERTICES,
            "max_facets": MAX_FACETS,
        }
    return {
        "numba_imported": True,
        "cuda_available": available,
        "reason": "ok" if available else "CUDA driver/device is unavailable",
        "max_vertices": MAX_VERTICES,
        "max_facets": MAX_FACETS,
    }


if cuda is not None:

    @cuda.jit(device=True, inline=True)
    def _popcount64(value):
        count = 0
        while value != 0:
            value &= value - uint64(1)
            count += 1
        return count


    @cuda.jit(device=True, inline=True)
    def _xorshift64(state):
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
        return state


    @cuda.jit(device=True, inline=True)
    def _are_neighbors(first, second):
        first_size = _popcount64(first)
        second_size = _popcount64(second)
        required = first_size
        if second_size < required:
            required = second_size
        required -= 1
        if required < 1:
            return False
        return _popcount64(first & second) >= required


    @cuda.jit(device=True)
    def _normalize_maximal_faces(work_faces, live, facet_count):
        """Mark exactly one copy of every inclusion-maximal nonempty face."""

        for first in range(facet_count):
            live[first] = uint8(0)
            first_face = work_faces[first]
            if first_face == 0:
                continue
            keep = True
            for second in range(facet_count):
                if first == second:
                    continue
                second_face = work_faces[second]
                if second_face == 0:
                    continue
                if (first_face & second_face) == first_face:
                    if first_face != second_face or second < first:
                        keep = False
                        break
            if keep:
                live[first] = uint8(1)
        # Non-maximal generators must be discarded permanently.  Allowing a
        # discarded face to reappear after a later vertex deletion would no
        # longer represent the elementary strong collapse of the current
        # complex.
        for first in range(facet_count):
            if live[first] == 0:
                work_faces[first] = uint64(0)


    @cuda.jit(device=True)
    def _selection_is_strongly_collapsible(
        facet_masks,
        assignment,
        facet_count,
        vertex_count,
        first_part,
        second_part,
        added_facet,
        work_faces,
        live,
    ):
        """Exact dominated-vertex test for one facet-generated subcomplex."""

        for facet_index in range(facet_count):
            selected = assignment[facet_index] == first_part
            if second_part >= 0 and assignment[facet_index] == second_part:
                selected = True
            if facet_index == added_facet:
                selected = True
            if selected:
                work_faces[facet_index] = facet_masks[facet_index]
            else:
                work_faces[facet_index] = uint64(0)

        _normalize_maximal_faces(work_faces, live, facet_count)

        # At most vertex_count-1 elementary strong collapses are possible.
        for _ in range(vertex_count + 1):
            active_vertices = uint64(0)
            for facet_index in range(facet_count):
                if live[facet_index] != 0:
                    active_vertices |= work_faces[facet_index]

            if active_vertices == 0:
                return False
            if _popcount64(active_vertices) == 1:
                return True

            dominated_vertex = -1
            for vertex in range(vertex_count):
                vertex_bit = uint64(1) << vertex
                if (active_vertices & vertex_bit) == 0:
                    continue
                for apex in range(vertex_count):
                    if apex == vertex:
                        continue
                    apex_bit = uint64(1) << apex
                    if (active_vertices & apex_bit) == 0:
                        continue
                    dominated = True
                    for facet_index in range(facet_count):
                        if live[facet_index] == 0:
                            continue
                        face = work_faces[facet_index]
                        if (face & vertex_bit) != 0 and (face & apex_bit) == 0:
                            dominated = False
                            break
                    if dominated:
                        dominated_vertex = vertex
                        break
                if dominated_vertex >= 0:
                    break

            if dominated_vertex < 0:
                return False

            deleted_bit = uint64(1) << dominated_vertex
            for facet_index in range(facet_count):
                if live[facet_index] != 0:
                    work_faces[facet_index] &= ~deleted_bit
            _normalize_maximal_faces(work_faces, live, facet_count)

        return False


    @cuda.jit(device=True)
    def _parts_share_a_vertex(
        facet_masks,
        assignment,
        facet_count,
        first_part,
        second_part,
    ):
        first_vertices = uint64(0)
        second_vertices = uint64(0)
        for facet_index in range(facet_count):
            part = assignment[facet_index]
            if part == first_part:
                first_vertices |= facet_masks[facet_index]
            elif part == second_part:
                second_vertices |= facet_masks[facet_index]
        return (first_vertices & second_vertices) != 0


    @cuda.jit
    def _greedy_restarts_kernel(
        facet_masks,
        facet_count,
        vertex_count,
        restart_offset,
        restart_count,
        base_seed,
        neighbor_mode,
        do_merge,
        output_counts,
        output_assignment,
    ):
        local_restart = cuda.grid(1)
        if local_restart >= restart_count:
            return

        restart = restart_offset + local_restart
        state = base_seed ^ (
            uint64(restart + 1) * uint64(0x9E3779B97F4A7C15)
        )
        if state == 0:
            state = uint64(0xD1B54A32D192ED03)

        order = cuda.local.array(MAX_FACETS, dtype=int16)
        assignment = cuda.local.array(MAX_FACETS, dtype=int16)
        part_order = cuda.local.array(MAX_FACETS, dtype=int16)
        part_sizes = cuda.local.array(MAX_FACETS, dtype=int16)
        neighbor_scores = cuda.local.array(MAX_FACETS, dtype=int16)
        used = cuda.local.array(MAX_FACETS, dtype=uint8)
        frontier = cuda.local.array(MAX_FACETS, dtype=uint8)
        work_faces = cuda.local.array(MAX_FACETS, dtype=uint64)
        live = cuda.local.array(MAX_FACETS, dtype=uint8)

        for index in range(facet_count):
            order[index] = int16(index)
            assignment[index] = int16(-1)
            part_sizes[index] = int16(0)
            used[index] = uint8(0)
            frontier[index] = uint8(0)

        if restart == 1:
            for index in range(facet_count):
                order[index] = int16(facet_count - 1 - index)
        elif restart >= 2 and neighbor_mode != _NEIGHBOR_NONE and restart % 3 == 2:
            # Random walk over the codimension-one facet adjacency frontier.
            state = _xorshift64(state)
            current = int32(state % uint64(facet_count))
            for position in range(facet_count):
                if position > 0:
                    candidate_count = 0
                    for candidate in range(facet_count):
                        if used[candidate] == 0 and frontier[candidate] != 0:
                            candidate_count += 1
                    if candidate_count == 0:
                        for candidate in range(facet_count):
                            if used[candidate] == 0:
                                candidate_count += 1
                    state = _xorshift64(state)
                    selected_rank = int32(state % uint64(candidate_count))
                    rank = 0
                    for candidate in range(facet_count):
                        eligible = used[candidate] == 0
                        if candidate_count < facet_count - position:
                            eligible = eligible and frontier[candidate] != 0
                        if eligible:
                            if rank == selected_rank:
                                current = candidate
                                break
                            rank += 1

                order[position] = int16(current)
                used[current] = uint8(1)
                frontier[current] = uint8(0)
                current_mask = facet_masks[current]
                for candidate in range(facet_count):
                    if used[candidate] == 0 and _are_neighbors(
                        current_mask,
                        facet_masks[candidate],
                    ):
                        frontier[candidate] = uint8(1)
        elif restart >= 2:
            # Fisher-Yates shuffle.
            for index in range(facet_count - 1, 0, -1):
                state = _xorshift64(state)
                other = int32(state % uint64(index + 1))
                temporary = order[index]
                order[index] = order[other]
                order[other] = temporary

        part_count = 0
        for position in range(facet_count):
            facet_index = int32(order[position])

            for part in range(part_count):
                neighbor_scores[part] = int16(0)
                part_order[part] = int16(part)

            if neighbor_mode != _NEIGHBOR_NONE:
                facet_mask = facet_masks[facet_index]
                for previous_position in range(position):
                    previous_facet = int32(order[previous_position])
                    if _are_neighbors(facet_mask, facet_masks[previous_facet]):
                        previous_part = int32(assignment[previous_facet])
                        neighbor_scores[previous_part] += int16(1)

            # Shuffle before a stable insertion sort to randomize equal scores.
            for index in range(part_count - 1, 0, -1):
                state = _xorshift64(state)
                other = int32(state % uint64(index + 1))
                temporary = part_order[index]
                part_order[index] = part_order[other]
                part_order[other] = temporary

            for index in range(1, part_count):
                key = int32(part_order[index])
                cursor = index - 1
                while cursor >= 0:
                    previous = int32(part_order[cursor])
                    key_score = int32(neighbor_scores[key])
                    previous_score = int32(neighbor_scores[previous])
                    key_size = int32(part_sizes[key])
                    previous_size = int32(part_sizes[previous])
                    if neighbor_mode == _NEIGHBOR_NONE:
                        key_score = 0
                        previous_score = 0
                    if key_score < previous_score:
                        break
                    if key_score == previous_score and key_size <= previous_size:
                        break
                    part_order[cursor + 1] = part_order[cursor]
                    cursor -= 1
                part_order[cursor + 1] = int16(key)

            placed = False
            for target_position in range(part_count):
                target_part = int32(part_order[target_position])
                if (
                    neighbor_mode == _NEIGHBOR_STRICT
                    and neighbor_scores[target_part] == 0
                ):
                    continue
                if _selection_is_strongly_collapsible(
                    facet_masks,
                    assignment,
                    facet_count,
                    vertex_count,
                    target_part,
                    -1,
                    facet_index,
                    work_faces,
                    live,
                ):
                    assignment[facet_index] = int16(target_part)
                    part_sizes[target_part] += int16(1)
                    placed = True
                    break

            if not placed:
                assignment[facet_index] = int16(part_count)
                part_sizes[part_count] = int16(1)
                part_count += 1

        if do_merge != 0:
            merged = True
            while merged and part_count > 1:
                merged = False
                state = _xorshift64(state)
                first_start = int32(state % uint64(part_count))
                for first_offset in range(part_count):
                    first = (first_start + first_offset) % part_count
                    if merged:
                        break
                    for second_offset in range(1, part_count):
                        second = (first + second_offset) % part_count
                        if not _parts_share_a_vertex(
                            facet_masks,
                            assignment,
                            facet_count,
                            first,
                            second,
                        ):
                            continue
                        if not _selection_is_strongly_collapsible(
                            facet_masks,
                            assignment,
                            facet_count,
                            vertex_count,
                            first,
                            second,
                            -1,
                            work_faces,
                            live,
                        ):
                            continue

                        last = part_count - 1
                        for facet_index in range(facet_count):
                            if assignment[facet_index] == second:
                                assignment[facet_index] = int16(first)
                        if second != last:
                            for facet_index in range(facet_count):
                                if assignment[facet_index] == last:
                                    assignment[facet_index] = int16(second)
                        part_count -= 1
                        merged = True
                        break

        output_counts[local_restart] = int16(part_count)
        for facet_index in range(facet_count):
            output_assignment[local_restart, facet_index] = assignment[facet_index]


def _require_cuda(device_id: int) -> None:
    if cuda is None:
        raise CudaUnavailableError(
            "Numba CUDA is not installed. For CUDA 12 run: "
            "python -m pip install numpy 'numba-cuda[cu12]'"
        ) from _NUMBA_IMPORT_ERROR
    try:
        if not cuda.is_available():
            raise CudaUnavailableError(
                "Numba cannot access an NVIDIA CUDA device. Check the NVIDIA "
                "driver and run pysimplicial.cuda_status()."
            )
        cuda.select_device(device_id)
    except CudaUnavailableError:
        raise
    except Exception as error:
        raise CudaUnavailableError(
            f"Failed to initialize CUDA device {device_id}: {error}"
        ) from error



def randomized_cover(
    complex_, *, restarts, seed, lower_bound, batch_size, progress,
    neighbor_mode="prefer", merge=True,
):
    """Return the best facet partition and the number of executed restarts."""
    labels = sorted(complex_.vertices, key=_vertex_key)
    indices = {vertex: index for index, vertex in enumerate(labels)}
    encoded = sorted(
        (sum(1 << indices[v] for v in facet), facet)
        for facet in complex_.facets
    )
    masks, facets = zip(*encoded)
    facet_count = len(facets)
    device_facets = cuda.to_device(np.asarray(masks, dtype=np.uint64))
    neighbor_code = _NEIGHBOR_CODES[neighbor_mode]
    threads = 32
    best_count = facet_count
    best_assignment = np.arange(facet_count, dtype=np.int16)
    completed = 0
    started = monotonic()
    while completed < restarts and best_count - 1 > lower_bound:
        count = min(batch_size, restarts - completed)
        device_counts = cuda.device_array(count, dtype=np.int16)
        device_assignment = cuda.device_array((count, facet_count), dtype=np.int16)
        blocks = (count + threads - 1) // threads
        _greedy_restarts_kernel[blocks, threads](
            device_facets, facet_count, len(labels), completed, count,
            np.uint64(seed), neighbor_code, int(merge),
            device_counts, device_assignment,
        )
        counts = device_counts.copy_to_host()
        best = int(np.argmin(counts))
        local_count = int(counts[best])
        if local_count < best_count:
            best_count = local_count
            best_assignment = device_assignment[best].copy_to_host()
        completed += count
        if progress:
            progress({
                "phase": "cuda_restarts",
                "restarts": completed,
                "upper_bound": best_count - 1,
                "seconds": monotonic() - started,
            })
    groups = [[] for _ in range(best_count)]
    for facet, part in zip(facets, best_assignment):
        if not 0 <= part < best_count:
            raise RuntimeError("CUDA returned an invalid cover assignment.")
        groups[int(part)].append(facet)
    if any(not group for group in groups):
        raise RuntimeError("CUDA returned an empty cover member.")
    return tuple(SimplicialComplex(group) for group in groups), completed
