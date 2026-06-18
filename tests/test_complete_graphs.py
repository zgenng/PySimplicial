from simplicial_topology.Simplicial_Complex import SimplicialComplex
from generators.graphs import complete_graph

def test_K5():
    K = complete_graph(5)

    value, _ = K.gscat(for_graph=True)

    assert value == 2