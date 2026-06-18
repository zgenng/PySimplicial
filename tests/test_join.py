from simplicial_topology.Simplicial_Complex import SimplicialComplex

def test_join_dimension():
    A = SimplicialComplex()
    A.add_simplex([1])

    B = SimplicialComplex()
    B.add_simplex([2])

    J = A.join(B)

    assert J.dimension() == 1