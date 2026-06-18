from simplicial_topology.Simplicial_Complex import SimplicialComplex


def test_cycle_not_strongly_collapsible():
    K = SimplicialComplex()

    K.add_simplex([1, 2])
    K.add_simplex([2, 3])
    K.add_simplex([3, 1])

    assert not K.is_strongly_collapsible()