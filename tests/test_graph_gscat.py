from simplicial_topology.Simplicial_Complex import SimplicialComplex


def test_triangle_gscat():
    K = SimplicialComplex()

    K.add_simplex([1, 2])
    K.add_simplex([2, 3])
    K.add_simplex([3, 1])

    value, _ = K.gscat(for_graph=True)

    assert value == 1