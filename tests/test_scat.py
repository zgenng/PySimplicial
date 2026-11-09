import unittest
from unittest.mock import patch

from pysimplicial import (
    CudaUnavailableError, SearchLimitExceeded, SimplicialComplex, SimplicialMap,
    cuda_status, disjoint_union, is_contiguous, scat,
)
from pysimplicial.datasets import cycle_graph, path_graph, projective_plane


class ScatBackendTests(unittest.TestCase):
    def assert_certificate(self, complex_, result):
        self.assertEqual(len(result.cover), len(result.chains))
        self.assertEqual(result.value, len(result.cover) - 1)
        self.assertTrue(all(any(facet in part for part in result.cover) for facet in complex_.facets))
        for part, chain in zip(result.cover, result.chains):
            self.assertEqual(chain[0], SimplicialMap.inclusion(part, complex_))
            self.assertEqual(len(set(chain[-1].vertex_map.values())), 1)
            self.assertTrue(all(is_contiguous(a, b) for a, b in zip(chain, chain[1:])))

    def test_cpu_certificate_and_limits(self):
        complex_ = projective_plane()
        result = scat(complex_, backend="cpu")
        self.assertEqual(result.value, 2)
        self.assert_certificate(complex_, result)
        with self.assertRaisesRegex(SearchLimitExceeded, "max_subcomplexes"):
            scat(complex_, max_subcomplexes=1)
        with self.assertRaisesRegex(SearchLimitExceeded, "max_states"):
            scat(path_graph(4), max_states=1)

    def test_auto_falls_back_and_explicit_cuda_requires_device(self):
        complex_ = disjoint_union(cycle_graph(3), path_graph(4))
        with patch("pysimplicial._solvers.cuda_restarts._require_cuda", side_effect=CudaUnavailableError):
            result = scat(complex_, backend="auto")
            self.assertEqual(result.value, scat(complex_).value)
            self.assert_certificate(complex_, result)
            with self.assertRaises(CudaUnavailableError):
                scat(complex_, backend="cuda")

    def test_large_component_is_supported_by_cpu(self):
        complex_ = SimplicialComplex([range(65)])
        self.assertEqual(scat(complex_, backend="cpu").value, 0)
        with self.assertRaisesRegex(ValueError, "64 vertices"):
            scat(complex_, backend="cuda")

    def test_cuda_batches_if_available(self):
        if not cuda_status()["cuda_available"]:
            self.skipTest("CUDA is unavailable.")
        from pysimplicial._solvers.cuda_contiguity import _CudaMoves, cuda
        from pysimplicial.contiguity import _MapSearch

        complex_ = cycle_graph(4)
        with patch.object(cuda, "to_device", wraps=cuda.to_device) as to_device:
            result = scat(complex_, backend="cuda", batch_size=3)
            self.assertTrue(to_device.called)
        self.assertEqual(result, scat(complex_, backend="cpu"))
        self.assert_certificate(complex_, result)

        # Exercise facet scans and bit 63 on the largest supported target.
        source = SimplicialMap.constant(SimplicialComplex([("v",)]), path_graph(64), 0)
        search = _MapSearch(source, None)
        cuda_moves = _CudaMoves(search.moves, batch_size=64, auto=False)
        run_batch = cuda_moves.for_domain(search.facets, search.incident)
        states = [(i,) for i in range(64)]
        self.assertEqual(run_batch(states).tolist(), [list(search._cpu_moves(state)) for state in states])
