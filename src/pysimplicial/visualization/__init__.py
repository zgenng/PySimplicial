"""Optional Matplotlib and Plotly views of complexes and covers."""

from ._surface import draw_surface_3d, surface_positions_3d, surface_statistics

__all__ = [
    "draw_complex", "draw_cover", "draw_complex_and_cover", "draw_complex_3d",
    "draw_surface_3d", "surface_positions_3d", "surface_statistics",
]


def draw_complex(K, *args, **kwargs):
    """Draw vertices, edges, and triangles with Matplotlib."""
    from ._two_d import draw_complex as draw

    return draw(K, *args, **kwargs)


def draw_cover(K, cover, *args, **kwargs):
    """Color the visible simplices according to cover membership."""
    from ._two_d import draw_cover as draw

    return draw(K, getattr(cover, "cover", cover), *args, **kwargs)


def draw_complex_and_cover(K, cover, *args, **kwargs):
    """Draw a complex and its cover side by side with the same layout."""
    from ._two_d import draw_complex_and_cover as draw

    return draw(K, getattr(cover, "cover", cover), *args, **kwargs)


def draw_complex_3d(K, positions, *, cover=None, title="Simplicial complex", show=True):
    """Draw the 2-skeleton at supplied 3D coordinates. Isolated vertices are kept."""
    from ._points import draw_complex_3d as draw

    return draw(K, positions, cover=getattr(cover, "cover", cover), title=title, show=show)
