from simplicial_topology.Simplicial_Complex import SimplicialComplex


def test_dimension():
    K = SimplicialComplex()

    K.add_simplex([1, 2, 3])

    assert K.dimension() == 2