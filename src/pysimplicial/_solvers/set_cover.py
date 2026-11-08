"""Minimum set cover over integer masks."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from .._errors import SearchLimitExceeded


def _iter_bits(mask: int) -> Iterator[int]:
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


@dataclass
class _SetCoverStats:
    states: int = 0
    lower_bound: int = 0
    greedy_upper_bound: int = 0


def minimum_set_cover(
    universe_size: int,
    candidates: Sequence[int],
    *,
    max_search_states: int | None = None,
    incumbent: Sequence[int] = (),
    lower_bound: int = 1,
) -> tuple[tuple[int, ...], _SetCoverStats]:
    """Exact integer branch-and-bound set cover over a bit-mask universe."""
    if universe_size == 0:
        return tuple(), _SetCoverStats()

    full = (1 << universe_size) - 1
    candidates = tuple(sorted(set(candidates), key=lambda x: (-x.bit_count(), x)))
    if not candidates:
        raise RuntimeError("No strongly collapsible candidates were generated.")

    union = 0
    for candidate in candidates:
        union |= candidate
    if union != full:
        missing = [i for i in range(universe_size) if not (union & (1 << i))]
        raise RuntimeError(f"Candidates do not cover maximal simplices {missing}.")

    options: list[tuple[int, ...]] = []
    for bit in range(universe_size):
        containing = tuple(c for c in candidates if c & (1 << bit))
        options.append(containing)

    # A valid greedy cover supplies an upper bound, never the final proof.
    remaining = full
    greedy: list[int] = []
    while remaining:
        best = max(candidates, key=lambda c: (c & remaining).bit_count())
        if (best & remaining) == 0:
            raise RuntimeError("Internal set-cover error: greedy search got stuck.")
        greedy.append(best)
        remaining &= ~best

    if incumbent and len(incumbent) < len(greedy):
        # Lift each incumbent part to a candidate with at least its coverage.
        # A complete candidate family guarantees that a superset exists.
        lifted = [
            next((candidate for candidate in candidates if candidate & mask == mask), None)
            for mask in incumbent
        ]
        if all(mask is not None for mask in lifted):
            greedy = list(dict.fromkeys(lifted))

    max_size = max(c.bit_count() for c in candidates)
    lower = max(lower_bound, (universe_size + max_size - 1) // max_size)
    stats = _SetCoverStats(lower_bound=lower, greedy_upper_bound=len(greedy))

    def choose_pivot(remaining_mask: int) -> int:
        return min(_iter_bits(remaining_mask), key=lambda b: len(options[b]))

    for target_count in range(lower, len(greedy)):
        failed: set[tuple[int, int]] = set()

        def search(remaining_mask: int, slots: int) -> tuple[int, ...] | None:
            if remaining_mask == 0:
                return tuple()
            if slots == 0:
                return None

            key = (remaining_mask, slots)
            if key in failed:
                return None

            stats.states += 1
            if max_search_states is not None and stats.states > max_search_states:
                raise SearchLimitExceeded(
                    f"Set cover exceeded max_search_states={max_search_states:,}.",
                    cover_lower_bound=target_count,
                    cover_masks=tuple(greedy),
                )

            remaining_count = remaining_mask.bit_count()
            global_max_gain = max((c & remaining_mask).bit_count() for c in candidates)
            if global_max_gain == 0 or (remaining_count + global_max_gain - 1) // global_max_gain > slots:
                failed.add(key)
                return None

            pivot = choose_pivot(remaining_mask)
            branches = sorted(
                options[pivot],
                key=lambda c: (-(c & remaining_mask).bit_count(), -c.bit_count(), c),
            )

            if slots == 1:
                for candidate in branches:
                    if (remaining_mask & ~candidate) == 0:
                        return (candidate,)
                failed.add(key)
                return None

            for candidate in branches:
                new_remaining = remaining_mask & ~candidate
                if new_remaining == remaining_mask:
                    continue
                suffix = search(new_remaining, slots - 1)
                if suffix is not None:
                    return (candidate,) + suffix

            failed.add(key)
            return None

        solution = search(full, target_count)
        if solution is not None:
            return solution, stats

    if lower > len(greedy):
        raise RuntimeError("The proved lower bound exceeds the size of a valid cover.")
    return tuple(greedy), stats
