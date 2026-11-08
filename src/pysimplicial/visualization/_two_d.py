"""Matplotlib drawings of vertices, edges, triangles and covers."""

import math
from collections import defaultdict

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Polygon

from ._layout import (
    DEFAULT_PALETTE, _build_graph, _edge_key, _get_positions, _ordered_edge,
    _resolve_positions, _simplex_sort_key, _vertex_sort_key,
)

DEFAULT_THEME = {
    "background": "#FFFFFF",
    "panel_background": "#FFFFFF",
    "base_edge": "#0F172A",
    "base_triangle": "#0F172A",
    "base_triangle_edge": "#0F172A",
    "node_face": "#FFFFFF",
    "node_edge": "#0F172A",
    "label": "#0F172A",
    "title": "#0F172A",
    "subtitle": "#475569",
    "outline": "#FFFFFF",
}



def _offset_segment(p1, p2, offset):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)

    if length == 0:
        return p1, p2

    normal_x = -dy / length
    normal_y = dx / length
    return (
        (x1 + offset * normal_x, y1 + offset * normal_y),
        (x2 + offset * normal_x, y2 + offset * normal_y),
    )


def _node_radius_points(node_size, gap=0.0):
    return max(math.sqrt(max(float(node_size), 0.0) / math.pi) + gap, 0.0)


def _draw_trimmed_edge(
    ax,
    p1,
    p2,
    color,
    *,
    width,
    alpha,
    zorder,
    node_size,
    node_gap=2.0,
    outline=None,
    outline_width=0.0,
):
    patch = FancyArrowPatch(
        tuple(np.asarray(p1, dtype=float)),
        tuple(np.asarray(p2, dtype=float)),
        arrowstyle="-",
        shrinkA=_node_radius_points(node_size, node_gap),
        shrinkB=_node_radius_points(node_size, node_gap),
        linewidth=width,
        color=color,
        alpha=alpha,
        capstyle="round",
        joinstyle="round",
        mutation_scale=1.0,
        zorder=zorder,
    )
    if outline is not None and outline_width > 0.0:
        patch.set_path_effects(
            [
                path_effects.Stroke(
                    linewidth=width + outline_width,
                    foreground=outline,
                    alpha=0.96,
                ),
                path_effects.Normal(),
            ]
        )
    ax.add_patch(patch)
    return patch


def _draw_offset_edge(
    ax,
    pos,
    edge,
    color,
    *,
    offset=0.0,
    width=4.2,
    alpha=0.96,
    zorder=8,
    outline="#FFFFFF",
    node_size=600,
    node_gap=3.2,
):
    u, v = _ordered_edge(edge)
    p1, p2 = _offset_segment(pos[u], pos[v], offset)
    return _draw_trimmed_edge(
        ax,
        p1,
        p2,
        color=color,
        width=width,
        alpha=alpha,
        zorder=zorder,
        node_size=node_size,
        node_gap=node_gap,
        outline=outline,
        outline_width=2.8,
    )


def _ordered_triangle_points(simplex, pos, scale=1.0):
    points = np.asarray([pos[vertex] for vertex in simplex], dtype=float)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    points = points[np.argsort(angles)]
    return center + scale * (points - center)


def _draw_triangle(
    ax,
    simplex,
    pos,
    *,
    facecolor,
    edgecolor,
    alpha,
    linewidth,
    zorder,
    scale=1.0,
):
    points = _ordered_triangle_points(simplex, pos, scale=scale)
    patch = Polygon(
        points,
        closed=True,
        facecolor=facecolor,
        edgecolor=edgecolor,
        alpha=alpha,
        linewidth=linewidth,
        joinstyle="round",
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def _draw_top_nodes(ax, pos, theme, node_size=620):
    if not pos:
        return None

    ordered = sorted(pos, key=_vertex_sort_key)
    collection = ax.scatter(
        [pos[vertex][0] for vertex in ordered],
        [pos[vertex][1] for vertex in ordered],
        s=node_size,
        c=theme["node_face"],
        edgecolors=theme["node_edge"],
        linewidths=1.8,
        zorder=20,
    )
    collection.set_path_effects(
        [
            path_effects.SimplePatchShadow(
                offset=(1.2, -1.2),
                alpha=0.18,
                shadow_rgbFace="#0F172A",
            ),
            path_effects.Normal(),
        ]
    )
    return collection


def _draw_top_labels(ax, pos, theme, font_size=10):
    for vertex in sorted(pos, key=_vertex_sort_key):
        x, y = pos[vertex]
        ax.text(
            x,
            y,
            str(vertex),
            fontsize=font_size,
            fontweight="semibold",
            ha="center",
            va="center",
            color=theme["label"],
            zorder=21,
        )


def _draw_base_structure(
    ax,
    K,
    pos,
    theme,
    *,
    triangle_alpha=0.18,
    node_size=620,
):
    triangles = sorted(
        K.iter_faces(2),
        key=_simplex_sort_key,
    )
    for simplex in triangles:
        _draw_triangle(
            ax,
            simplex,
            pos,
            facecolor=theme["base_triangle"],
            edgecolor=theme["base_triangle_edge"],
            alpha=triangle_alpha,
            linewidth=1.3,
            zorder=1,
        )

    _, edges, _ = _build_graph(K)
    for edge in edges:
        u, v = _ordered_edge(edge)
        _draw_trimmed_edge(
            ax,
            pos[u],
            pos[v],
            color=theme["base_edge"],
            width=1.7,
            alpha=0.58,
            zorder=2,
            node_size=node_size,
            node_gap=1.5,
        )


def _style_axes(ax, pos, title, theme):
    ax.set_facecolor(theme["panel_background"])
    ax.set_title(
        title,
        fontsize=14,
        fontweight="bold",
        color=theme["title"],
        pad=16,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    if pos:
        xs = np.asarray([point[0] for point in pos.values()])
        ys = np.asarray([point[1] for point in pos.values()])
        x_span = max(float(np.ptp(xs)), 1.0)
        y_span = max(float(np.ptp(ys)), 1.0)
        ax.set_xlim(xs.min() - 0.18 * x_span, xs.max() + 0.18 * x_span)
        ax.set_ylim(ys.min() - 0.18 * y_span, ys.max() + 0.18 * y_span)


def _make_gscat_title(
    complex_name="K",
    gscat_value=None,
    arboricity_value=None,
    default_title="Categorical cover",
):
    pieces = []
    if gscat_value is not None:
        pieces.append(f"gscat({complex_name}) = {gscat_value}")
    if arboricity_value is not None:
        pieces.append(f"a({complex_name}) = {arboricity_value}")
    return "   ·   ".join(pieces) if pieces else default_title


def _cover_memberships(cover):
    edge_memberships = defaultdict(list)
    vertex_memberships = defaultdict(list)
    triangle_memberships = defaultdict(list)

    for cover_index, subcomplex in enumerate(cover):
        for dim in range(3):
            for simplex in subcomplex.iter_faces(dim):
                simplex = frozenset(simplex)
                if len(simplex) == 1:
                    vertex = next(iter(simplex))
                    vertex_memberships[vertex].append(cover_index)
                elif len(simplex) == 2:
                    edge_memberships[_edge_key(simplex)].append(cover_index)
                elif len(simplex) == 3:
                    triangle_memberships[simplex].append(cover_index)

    return vertex_memberships, edge_memberships, triangle_memberships


def _validate_cover(K, cover):
    if any(not part.is_subcomplex_of(K) for part in cover):
        raise ValueError("Cover members must be subcomplexes of K.")
    if any(not any(facet in part for part in cover) for facet in K.facets):
        raise ValueError("The cover must contain all facets of K.")


def _draw_vertex_membership_rings(
    ax,
    pos,
    vertex_memberships,
    colors,
    *,
    zorder=15,
):
    if not pos:
        return
    xs = np.asarray([point[0] for point in pos.values()])
    ys = np.asarray([point[1] for point in pos.values()])
    span = max(float(np.ptp(xs)), float(np.ptp(ys)), 1.0)
    base_radius = 0.044 * span
    ring_step = 0.016 * span

    for vertex, members in vertex_memberships.items():
        for local_index, cover_index in enumerate(members):
            ring = Circle(
                pos[vertex],
                radius=base_radius + local_index * ring_step,
                facecolor="none",
                edgecolor=colors[cover_index % len(colors)],
                linewidth=2.5,
                alpha=0.78,
                zorder=zorder - local_index * 0.01,
            )
            ax.add_patch(ring)


def _cover_legend(ax, cover, colors, labels=None):
    if not cover:
        return None
    if labels is not None and len(labels) != len(cover):
        raise ValueError("cover_labels must have the same length as cover.")

    handles = []
    for index, subcomplex in enumerate(cover):
        edge_count = len(subcomplex.faces(1))
        triangle_count = len(subcomplex.faces(2))
        label = labels[index] if labels is not None else f"$U_{{{index + 1}}}$"
        details = []
        if edge_count:
            noun = "edge" if edge_count == 1 else "edges"
            details.append(f"{edge_count} {noun}")
        if triangle_count:
            noun = "face" if triangle_count == 1 else "faces"
            details.append(f"{triangle_count} {noun}")
        if details:
            label = f"{label}  ·  {', '.join(details)}"

        handles.append(
            Line2D(
                [0],
                [0],
                color=colors[index % len(colors)],
                linewidth=5,
                solid_capstyle="round",
                label=label,
            )
        )

    legend = ax.legend(
        handles=handles,
        loc="best",
        frameon=True,
        fancybox=True,
        framealpha=0.96,
        facecolor="#FFFFFF",
        edgecolor="#E2E8F0",
        fontsize=9,
        borderpad=0.8,
        labelspacing=0.65,
        handlelength=2.4,
    )
    legend.set_zorder(30)
    return legend


def draw_complex(
    K,
    ax=None,
    title="Simplicial complex",
    seed=42,
    show=True,
    *,
    pos=None,
    layout="auto",
    node_size=620,
    theme=None,
):
    theme = {**DEFAULT_THEME, **(theme or {})}
    if ax is None:
        figure, ax = plt.subplots(figsize=(7.4, 5.6), facecolor=theme["background"])
        figure.patch.set_facecolor(theme["background"])

    pos = _resolve_positions(K, pos, seed, layout)
    _draw_base_structure(
        ax,
        K,
        pos,
        theme,
        triangle_alpha=0.22,
        node_size=node_size,
    )
    _draw_top_nodes(ax, pos, theme, node_size=node_size)
    _draw_top_labels(ax, pos, theme, font_size=10)
    _style_axes(ax, pos, title, theme)

    if show:
        plt.show()
    return ax, pos


def draw_cover(
    K,
    cover,
    ax=None,
    title=None,
    complex_name="K",
    gscat_value=None,
    arboricity_value=None,
    seed=42,
    show=True,
    offset_shared_edges=True,
    *,
    pos=None,
    layout="auto",
    legend=False,
    cover_labels=None,
    validate=True,
    node_size=600,
    palette=None,
    theme=None,
):
    cover = tuple(cover)
    if validate:
        _validate_cover(K, cover)

    theme = {**DEFAULT_THEME, **(theme or {})}
    colors = tuple(palette or DEFAULT_PALETTE)
    if not colors:
        raise ValueError("palette must contain at least one color.")

    if ax is None:
        figure, ax = plt.subplots(figsize=(7.4, 5.6), facecolor=theme["background"])
        figure.patch.set_facecolor(theme["background"])

    pos = _resolve_positions(K, pos, seed, layout)
    if title is None:
        title = _make_gscat_title(
            complex_name=complex_name,
            gscat_value=gscat_value,
            arboricity_value=arboricity_value,
        )

    vertex_memberships, edge_memberships, triangle_memberships = (
        _cover_memberships(cover)
    )

    _draw_base_structure(
        ax,
        K,
        pos,
        theme,
        triangle_alpha=0.09,
        node_size=node_size,
    )

    # Shared triangles are nested slightly, so every membership remains visible.
    for simplex in sorted(triangle_memberships, key=_simplex_sort_key):
        members = triangle_memberships[simplex]
        for local_index, cover_index in enumerate(members):
            color = colors[cover_index % len(colors)]
            _draw_triangle(
                ax,
                simplex,
                pos,
                facecolor=color,
                edgecolor=color,
                alpha=0.22 if len(members) == 1 else 0.18,
                linewidth=2.2,
                zorder=4 + local_index * 0.1,
                scale=max(0.68, 1.0 - 0.085 * local_index),
            )

    for edge in sorted(edge_memberships, key=lambda item: tuple(
        _vertex_sort_key(vertex) for vertex in _ordered_edge(item)
    )):
        members = edge_memberships[edge]
        count = len(members)
        for local_index, cover_index in enumerate(members):
            offset = 0.0
            if offset_shared_edges and count > 1:
                offset = (local_index - (count - 1) / 2.0) * 0.045
            _draw_offset_edge(
                ax,
                pos,
                edge,
                colors[cover_index % len(colors)],
                offset=offset,
                width=4.3,
                outline=theme["outline"],
                node_size=node_size,
            )

    _draw_vertex_membership_rings(
        ax,
        pos,
        vertex_memberships,
        colors,
    )
    _draw_top_nodes(ax, pos, theme, node_size=node_size)
    _draw_top_labels(ax, pos, theme, font_size=10)
    _style_axes(ax, pos, title, theme)

    if legend:
        _cover_legend(ax, cover, colors, labels=cover_labels)

    if show:
        plt.show()
    return ax, pos


def draw_complex_and_cover(
    K,
    cover,
    complex_name="K",
    gscat_value=None,
    arboricity_value=None,
    seed=42,
    *,
    title=None,
    layout="auto",
    show=True,
    save_path=None,
    dpi=180,
    legend=False,
    cover_labels=None,
    validate=True,
    palette=None,
    theme=None,
):
    theme = {**DEFAULT_THEME, **(theme or {})}
    positions = _get_positions(K, seed=seed, layout=layout)

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(14.2, 6.0),
        facecolor=theme["background"],
        constrained_layout=True,
    )
    figure.patch.set_facecolor(theme["background"])

    draw_complex(
        K,
        ax=axes[0],
        title=f"Original complex · {complex_name}",
        pos=positions,
        seed=seed,
        layout=layout,
        show=False,
        theme=theme,
    )
    draw_cover(
        K,
        cover,
        ax=axes[1],
        title=title,
        complex_name=complex_name,
        gscat_value=gscat_value,
        arboricity_value=arboricity_value,
        pos=positions,
        seed=seed,
        layout=layout,
        show=False,
        legend=legend,
        cover_labels=cover_labels,
        validate=validate,
        palette=palette,
        theme=theme,
    )

    if save_path is not None:
        figure.savefig(
            save_path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor=figure.get_facecolor(),
        )
    if show:
        plt.show()
    return figure, axes
