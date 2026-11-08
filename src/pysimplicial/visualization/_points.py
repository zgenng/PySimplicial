"""Render a 2-skeleton at user-supplied coordinates."""

from ._layout import DEFAULT_PALETTE, _vertex_sort_key
from ._surface import _coerce_positions_3d, _line_coordinates, _plotly_graph_objects


def draw_complex_3d(K, positions, *, cover, title, show):
    go = _plotly_graph_objects()
    positions = _coerce_positions_3d(K, positions)
    vertices = sorted(K.vertices, key=_vertex_sort_key)
    coords = list(zip(*(positions[v] for v in vertices))) if vertices else [(), (), ()]
    indices = {v: i for i, v in enumerate(vertices)}
    figure = go.Figure()

    def add_complex(part, color, name):
        edges = [(*map(positions.__getitem__, edge), "") for edge in part.iter_faces(1)]
        x, y, z, _ = _line_coordinates(edges)
        figure.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode="lines", name=name, legendgroup=name,
            line=dict(color=color, width=4),
        ))
        triangles = [tuple(indices[v] for v in face) for face in part.iter_faces(2)]
        if triangles:
            i, j, k = zip(*triangles)
            figure.add_trace(go.Mesh3d(
                x=coords[0], y=coords[1], z=coords[2], i=i, j=j, k=k,
                color=color, opacity=0.24, name=name, legendgroup=name, showlegend=False,
            ))

    add_complex(K, "#475569", "K")
    if cover is not None:
        cover = tuple(cover)
        if any(not part.is_subcomplex_of(K) for part in cover) or any(
            not any(facet in part for part in cover) for facet in K.facets
        ):
            raise ValueError("Cover members must be subcomplexes whose union is K.")
        for index, part in enumerate(cover):
            add_complex(part, DEFAULT_PALETTE[index % len(DEFAULT_PALETTE)], f"U{index + 1}")
    figure.add_trace(go.Scatter3d(
        x=coords[0], y=coords[1], z=coords[2], mode="markers",
        text=list(map(str, vertices)), name="Vertices", marker=dict(size=4, color="#0F172A"),
    ))
    figure.update_layout(title=title, scene=dict(aspectmode="data"))
    if show:
        figure.show()
    return figure
