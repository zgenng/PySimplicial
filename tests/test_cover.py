from simplicial_topology.Simplicial_Complex import SimplicialComplex
from generators.graphs import complete_graph

def test_cover_is_cover():
    K = complete_graph(5)

    value, cover = K.gscat(for_graph=True)

    union = set()

    for U in cover:
        union |= set(U)

    assert union == K.simplicial_complex