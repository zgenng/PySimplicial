from simplicial_topology.Simplicial_Complex import SimplicialComplex


def test_triangle_is_strongly_collapsible():
    K = SimplicialComplex()

    K.add_simplex([1, 2, 3])

    assert K.is_strongly_collapsible()