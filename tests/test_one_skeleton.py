from simplicial_topology.Simplicial_Complex import SimplicialComplex


def test_one_skeleton():
    K = SimplicialComplex()

    K.add_simplex([1, 2, 3])

    G = K.one_skeleton()

    edges = {
        frozenset([1, 2]),
        frozenset([1, 3]),
        frozenset([2, 3]),
    }

    assert edges.issubset(G.simplicial_complex)