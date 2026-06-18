from simplicial_topology.Simplicial_Complex import SimplicialComplex


def test_triangle_core():
    K = SimplicialComplex()

    K.add_simplex([1, 2, 3])

    core = K.core()

    assert len(core.vertices) == 1