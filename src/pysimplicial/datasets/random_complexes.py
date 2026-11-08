"""Seeded random examples for experiments."""

from itertools import combinations
from random import Random

from ..complex import SimplicialComplex


def random_d_complex(n, dim=3, p_edge=0.5, p_simplex=0.3, seed=None):
    """Sample edges and fill higher-dimensional simplices with complete boundary."""
    if n < 0 or dim < 0 or not 0 <= p_edge <= 1 or not 0 <= p_simplex <= 1:
        raise ValueError("Use nonnegative sizes and probabilities in [0, 1].")
    rng = Random(seed)
    simplices = {frozenset((v,)) for v in range(n)}
    for size in range(2, min(dim + 1, n) + 1):
        probability = p_edge if size == 2 else p_simplex
        for simplex in combinations(range(n), size):
            if all(frozenset(face) in simplices for face in combinations(simplex, size - 1)):
                if rng.random() < probability:
                    simplices.add(frozenset(simplex))
    return SimplicialComplex(simplices)


def random_graph(n, p, seed=None):
    return random_d_complex(n, dim=1, p_edge=p, seed=seed)


def random_2_complex(n, p_edge=0.5, p_triangle=0.3, seed=None):
    return random_d_complex(n, dim=2, p_edge=p_edge, p_simplex=p_triangle, seed=seed)
