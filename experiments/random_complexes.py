import random
from itertools import combinations

from simplicial_topology.Simplicial_Complex import SimplicialComplex


def random_graph(n, p, seed=None):
    """
    Generate an Erdos-Renyi random graph G(n,p) as a 1-dimensional complex.

    Parameters
    ----------
    n : int
        Number of vertices.
    p : float
        Probability of adding each edge.
    seed : int, optional
        Random seed.

    Returns
    -------
    SimplicialComplex
        A random graph as a simplicial complex.
    """
    rng = random.Random(seed)
    K = SimplicialComplex()

    for v in range(n):
        K.add_simplex([v])

    for i, j in combinations(range(n), 2):
        if rng.random() < p:
            K.add_simplex([i, j])

    return K


def random_2_complex(n, p_edge=0.5, p_triangle=0.3, seed=None):
    """
    Generate a random 2-dimensional simplicial complex.

    First, a random graph is generated.
    Then a triangle [i,j,k] is added only if all three boundary edges
    already exist.

    Parameters
    ----------
    n : int
        Number of vertices.
    p_edge : float
        Probability of adding each edge.
    p_triangle : float
        Probability of filling each available triangle.
    seed : int, optional
        Random seed.

    Returns
    -------
    SimplicialComplex
        A random 2-dimensional simplicial complex.
    """
    rng = random.Random(seed)
    K = SimplicialComplex()

    for v in range(n):
        K.add_simplex([v])

    edges = set()

    for i, j in combinations(range(n), 2):
        if rng.random() < p_edge:
            K.add_simplex([i, j])
            edges.add(frozenset([i, j]))

    for i, j, k in combinations(range(n), 3):
        boundary = {
            frozenset([i, j]),
            frozenset([i, k]),
            frozenset([j, k]),
        }

        if boundary.issubset(edges) and rng.random() < p_triangle:
            K.add_simplex([i, j, k])

    return K
