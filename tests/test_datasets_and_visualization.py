from importlib.util import find_spec
from tempfile import TemporaryDirectory
from pathlib import Path
import unittest

from pysimplicial import gscat
from pysimplicial.datasets import (
    complete_graph, generate_rips_data,projective_plane,
    rips_complex, torus, triangulation,
)
from pysimplicial.visualization import (
    draw_complex_3d, draw_complex_and_cover, draw_surface_3d, surface_positions_3d, surface_statistics,
)


class DatasetTests(unittest.TestCase):
    def test_catalog(self):
        self.assertEqual(triangulation("CP2").dimension(), 4)
        self.assertEqual(surface_statistics(projective_plane())["crosscap_number"], 1)
        self.assertEqual(surface_statistics(torus())["genus"], 1)
        self.assertEqual(set(surface_positions_3d(torus())), torus().vertices)

    def test_rips(self):
        K = rips_complex([(0, 0), (1, 0), (0, 1)], 2)
        self.assertEqual(K.dimension(), 2)
        points, generated, _ = generate_rips_data(8, seed=42)
        self.assertEqual(len(points), len(generated.vertices))


class VisualizationTests(unittest.TestCase):
    @unittest.skipUnless(find_spec("matplotlib"), "Matplotlib is unavailable.")
    def test_2d_cover_export(self):
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        K = complete_graph(5)
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "cover.png"
            figure, axes = draw_complex_and_cover(K, gscat(K), show=False, legend=True, save_path=output)
            self.assertEqual(len(axes), 2)
            self.assertGreater(output.stat().st_size, 1000)
            plt.close(figure)

    @unittest.skipUnless(find_spec("plotly"), "Plotly is unavailable.")
    def test_plotly_surface_and_general_complex(self):
        K = projective_plane()
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "cover.html"
            figure, positions = draw_surface_3d(K, gscat(K).cover, show=False, save_html=output)
            self.assertGreater(len(figure.data), 0)
            self.assertEqual(len(positions), 12)
            self.assertGreater(output.stat().st_size, 1000)
        K = complete_graph(3)
        positions = {1: (0, 0, 0), 2: (1, 0, 0), 3: (0, 1, 0)}
        self.assertGreater(len(draw_complex_3d(K, positions, cover=gscat(K), show=False).data), 0)
