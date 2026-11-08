import math
import unittest

from pysimplicial import (
    SearchLimitExceeded, SimplicialComplex, SimplicialMap, categorical_product,
    connected_components, contiguity_chain, contiguity_class, contiguity_distance,
    disjoint_union, dominators, graph_join, is_categorical, is_contiguous,
    is_strongly_collapsible, join, link, relabel, same_contiguity_class, scat,
    strong_core, wedge,
)
from pysimplicial.datasets import cycle_graph, path_graph


class ComplexTests(unittest.TestCase):
    def test_faces_and_skeleton(self):
        K = SimplicialComplex([(0, 1, 2), (0, 1), ("isolated",)])
        self.assertEqual(K.dimension(), 2)
        self.assertEqual(len(K.facets), 2)
        self.assertEqual(len(K.faces()), 8)
        self.assertEqual(K.one_skeleton().dimension(), 1)
        self.assertIn(("isolated",), K.one_skeleton())
        self.assertEqual(K.skeleton(0).faces(), K.faces(0))
        self.assertEqual(K.skeleton(-1), SimplicialComplex())
        self.assertEqual(K.induced_subcomplex([1, "isolated"]).dimension(), 0)

    def test_join_and_product(self):
        K = SimplicialComplex([(0, 1)])
        L = SimplicialComplex([("a", "b")])
        self.assertEqual(join(K, L).dimension(), 3)
        self.assertEqual(graph_join(K, L).dimension(), 1)
        self.assertEqual(categorical_product(K, L).facets, {frozenset({(0, "a"), (0, "b"), (1, "a"), (1, "b")})})
        self.assertEqual(join(K, SimplicialComplex()), K)
        with self.assertRaises(ValueError):
            join(K, K)

    def test_wedge_and_links(self):
        K = SimplicialComplex([(0, 1, 2)])
        W = wedge(K, K, 0, 0)
        self.assertEqual(len(W.vertices), 5)
        self.assertEqual(len(W.facets), 2)
        self.assertEqual(link(K, [0]), SimplicialComplex([(1, 2)]))
        self.assertEqual(link(K, []), K)
        self.assertEqual(len(connected_components(disjoint_union(K, K))), 2)
        with self.assertRaises(ValueError):
            relabel(K, {0: 1})

    def test_collapse_certificate(self):
        K = path_graph(5)
        result = strong_core(K)
        self.assertEqual(len(result.steps), 4)
        self.assertEqual(len(result.core.vertices), 1)
        current = K
        for step in result.steps:
            self.assertIn(step.dominator, dominators(current, step.vertex))
            current = current.induced_subcomplex(current.vertices - {step.vertex})
        self.assertEqual(current, result.core)
        self.assertFalse(is_strongly_collapsible(cycle_graph(3)))
        self.assertFalse(is_strongly_collapsible(SimplicialComplex()))


class MapTests(unittest.TestCase):
    def test_mapping_and_restriction(self):
        K = SimplicialComplex([(0, 1)])
        L = path_graph(3)
        mapping = {0: 0, 1: 1}
        f = SimplicialMap(K, L, mapping)
        mapping[1] = 2
        self.assertEqual(f.image_simplex((0, 1)), {0, 1})
        self.assertEqual(f.restrict(SimplicialComplex([(1,)])).vertex_map, {1: 1})
        self.assertEqual(SimplicialMap.identity(L).compose(f), f)
        self.assertEqual(len({f, SimplicialMap(K, L, {1: 1, 0: 0})}), 1)
        with self.assertRaises(ValueError):
            SimplicialMap(K, L, mapping)

    def test_chain_but_no_direct_contiguity(self):
        K = SimplicialComplex([(0, 1)])
        L = path_graph(3)
        f = SimplicialMap(K, L, {0: 0, 1: 1})
        g = SimplicialMap(K, L, {0: 1, 1: 2})
        self.assertFalse(is_contiguous(f, g))
        self.assertTrue(same_contiguity_class(f, g))
        chain = contiguity_chain(f, g)
        self.assertEqual((chain[0], chain[-1]), (f, g))
        self.assertTrue(all(is_contiguous(a, b) for a, b in zip(chain, chain[1:])))
        self.assertEqual(len(set(contiguity_class(f))), 7)
        self.assertEqual(contiguity_distance(f, g).value, 0)
        with self.assertRaises(SearchLimitExceeded):
            contiguity_chain(f, g, max_states=1)

    def test_cycle_and_minimal_cover(self):
        K = cycle_graph(3)
        f = SimplicialMap.identity(K)
        g = SimplicialMap.constant(K, K, 0)
        self.assertIsNone(contiguity_chain(f, g))
        self.assertFalse(is_categorical(K, K))
        self.assertTrue(is_categorical(path_graph(3), K))
        distance = contiguity_distance(f, g)
        self.assertEqual(distance.value, 1)
        self.assertEqual(scat(K).value, 1)
        self.assertEqual(len(distance.cover), 2)
        for part, chain in zip(distance.cover, distance.chains):
            self.assertEqual((chain[0], chain[-1]), (f.restrict(part), g.restrict(part)))
            self.assertTrue(all(is_contiguous(a, b) for a, b in zip(chain, chain[1:])))
        with self.assertRaises(SearchLimitExceeded):
            contiguity_distance(f, g, max_subcomplexes=1)

    def test_infinite_distance_and_disconnected_category(self):
        K = SimplicialComplex([(0,), (1,)])
        f = SimplicialMap.constant(K, K, 0)
        g = SimplicialMap.constant(K, K, 1)
        self.assertEqual(contiguity_distance(f, g).value, math.inf)
        self.assertEqual(scat(K).value, 1)
        self.assertFalse(is_categorical(K, K))

    def test_empty_domain(self):
        K = SimplicialComplex()
        f = SimplicialMap.identity(K)
        self.assertEqual(contiguity_chain(f, f), (f,))
        self.assertEqual(contiguity_distance(f, f).value, 0)
        self.assertEqual(scat(K).cover, ())
