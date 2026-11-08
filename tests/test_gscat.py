from itertools import combinations
from random import Random
import unittest
from unittest.mock import patch

from pysimplicial import CudaUnavailableError, SimplicialComplex, cuda_status, gscat, verify_gscat_cover
from pysimplicial._solvers.exact import exact_gscat
from pysimplicial.datasets import complete_graph, projective_plane, tetrahedron_boundary


def mixed_regression():
    return SimplicialComplex([*combinations(range(4), 2), (4, 5), (5, 6), (4, 6), (0, 1, 4)])


class GSCatTests(unittest.TestCase):
    def test_standard_examples_and_cover(self):
        for K, value in (
            (SimplicialComplex(), 0),
            (SimplicialComplex([(0, 1, 2)]), 0),
            (SimplicialComplex([(0,), (1,)]), 1),
            (complete_graph(5), 2),
            (tetrahedron_boundary(), 1),
            (projective_plane(), 2),
        ):
            with self.subTest(K=K):
                result = gscat(K)
                self.assertEqual(result.value, value)
                self.assertTrue(result.exact)
                self.assertTrue(verify_gscat_cover(K, result.cover))

    def test_optimal_cover_uses_a_nonfacet(self):
        K = mixed_regression()
        result = gscat(K)
        self.assertEqual(result.value, 1)
        self.assertTrue(verify_gscat_cover(K, result.cover))
        self.assertTrue(any(part.facets - K.facets for part in result.cover))

    def test_exact_algorithm_against_independent_oracle(self):
        rng = Random(42)
        choices = list(combinations(range(5), 2)) + list(combinations(range(5), 3))
        examples = [mixed_regression()]
        examples.extend(SimplicialComplex(rng.sample(choices, rng.randint(3, 8))) for _ in range(30))
        for K in examples:
            oracle = exact_gscat(K.facets, use_gpu=False, algorithm="all_subcomplexes")
            result = gscat(K)
            self.assertEqual(result.value, oracle.gscat, K.facets)
            self.assertTrue(verify_gscat_cover(K, result.cover))

    def test_budget_returns_honest_bounds(self):
        K = projective_plane()
        result = gscat(K, max_candidate_states=1)
        self.assertIsNone(result.value)
        self.assertFalse(result.exact)
        self.assertLessEqual(result.lower_bound, 2)
        self.assertGreaterEqual(result.upper_bound, 2)
        self.assertTrue(verify_gscat_cover(K, result.cover))

    def test_set_cover_budget_keeps_the_best_verified_cover(self):
        K = projective_plane()
        result = gscat(K, max_search_states=1)
        self.assertIsNone(result.value)
        self.assertEqual((result.lower_bound, result.upper_bound), (1, 2))
        self.assertTrue(verify_gscat_cover(K, result.cover))

    def test_explicit_cuda_vs_auto_fallback(self):
        K = projective_plane()
        with patch("pysimplicial._solvers.cuda_restarts._require_cuda", side_effect=CudaUnavailableError):
            with self.assertRaises(CudaUnavailableError):
                gscat(K, backend="cuda")
            result = gscat(K, backend="auto")
            self.assertEqual(result.value, 2)
            self.assertFalse(result.gpu_used)

    def test_cuda_matching_bounds_skip_exact_search(self):
        K = tetrahedron_boundary()
        cover = (SimplicialComplex([(0, 1, 2), (0, 1, 3), (0, 2, 3)]),
                 SimplicialComplex([(1, 2, 3)]))
        with (
            patch("pysimplicial._solvers.cuda_restarts._require_cuda"),
            patch("pysimplicial._solvers.cuda_restarts.randomized_cover", return_value=(cover, 17)),
            patch("pysimplicial._solvers.exact.exact_gscat", side_effect=AssertionError),
        ):
            result = gscat(K, backend="cuda", restarts=17)
        self.assertEqual(result.value, 1)
        self.assertEqual(result.restarts_completed, 17)
        self.assertTrue(result.gpu_used)

    def test_cuda_heuristic_can_be_improved_by_exact_search(self):
        K = mixed_regression()
        initial = tuple(SimplicialComplex([facet]) for facet in K.facets)
        with (
            patch("pysimplicial._solvers.cuda_restarts._require_cuda"),
            patch("pysimplicial._solvers.cuda_restarts.randomized_cover", return_value=(initial, 8)),
            patch("pysimplicial._solvers.exact.cuda_is_available", return_value=False),
        ):
            result = gscat(K, backend="cuda", restarts=8)
        self.assertEqual(result.value, 1)
        self.assertTrue(verify_gscat_cover(K, result.cover))
        self.assertEqual(result.restarts_completed, 8)

    def test_cuda_cover_is_verified_before_certifying(self):
        with (
            patch("pysimplicial._solvers.cuda_restarts._require_cuda"),
            patch("pysimplicial._solvers.cuda_restarts.randomized_cover", return_value=((SimplicialComplex([(0,)]),), 3)),
        ):
            with self.assertRaisesRegex(RuntimeError, "verifier rejected"):
                gscat(tetrahedron_boundary(), backend="cuda")

    def test_real_cuda_if_available(self):
        if not cuda_status()["cuda_available"]:
            self.skipTest("CUDA is unavailable.")
        for restarts in (0, 16):
            with self.subTest(restarts=restarts):
                K = projective_plane()
                result = gscat(K, backend="cuda", restarts=restarts)
                self.assertEqual(result.value, 2)
                self.assertTrue(result.gpu_used)
                self.assertEqual(result.restarts_completed, restarts)
                self.assertTrue(verify_gscat_cover(K, result.cover))
