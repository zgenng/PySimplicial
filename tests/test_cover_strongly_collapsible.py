from simplicial_topology.Simplicial_Complex import SimplicialComplex
from generators.graphs import complete_graph

def test_cover_elements_are_strongly_collapsible():
    K = complete_graph(5)

    value, cover = K.gscat(for_graph=True)

    for U in cover:
        T = SimplicialComplex(U)

        assert T.is_strongly_collapsible()