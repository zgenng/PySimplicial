import math
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Polygon


def _build_graph(K):
    """
    Build the 1-skeleton graph of a simplicial complex.
    """
    G = nx.Graph()

    for v in K.vertices:
        G.add_node(v)

    for simplex in K.simplicial_complex:
        if len(simplex) == 2:
            u, v = tuple(simplex)
            G.add_edge(u, v)

    return G


def _get_positions(K, seed=42):
    """
    Compute fixed positions for the vertices of the complex.
    """
    G = _build_graph(K)

    if len(G.nodes) == 0:
        return {}

    if len(G.edges) == 0:
        return nx.circular_layout(G)

    return nx.spring_layout(G, seed=seed)


def _edge_key(edge):
    """
    Return a canonical representation of an undirected edge.
    """
    u, v = edge
    return tuple(sorted((u, v)))


def _offset_segment(p1, p2, offset):
    """
    Shift a line segment by a small perpendicular offset.
    """
    x1, y1 = p1
    x2, y2 = p2

    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)

    if length == 0:
        return p1, p2

    nx_ = -dy / length
    ny_ = dx / length

    return (
        (x1 + offset * nx_, y1 + offset * ny_),
        (x2 + offset * nx_, y2 + offset * ny_),
    )


def _draw_offset_edge(
    ax,
    pos,
    edge,
    color,
    offset=0.0,
    width=4,
    alpha=0.9,
    zorder=5,
):
    """
    Draw an edge manually, optionally shifted by a perpendicular offset.
    """
    u, v = edge
    p1, p2 = _offset_segment(pos[u], pos[v], offset)

    ax.plot(
        [p1[0], p2[0]],
        [p1[1], p2[1]],
        color=color,
        linewidth=width,
        alpha=alpha,
        solid_capstyle="round",
        zorder=zorder,
    )


def _make_gscat_title(
    complex_name="K",
    gscat_value=None,
    arboricity_value=None,
    default_title="gscat cover",
):
    """
    Build a title for the cover visualization.
    """
    if gscat_value is None and arboricity_value is None:
        return default_title

    if gscat_value is not None and arboricity_value is None:
        return f"gscat({complex_name}) = {gscat_value}"

    if gscat_value is None and arboricity_value is not None:
        return f"a({complex_name}) = {arboricity_value}"

    return (
        f"gscat({complex_name}) = {gscat_value}, "
        f"a({complex_name}) = {arboricity_value}"
    )


def _draw_top_nodes(ax, pos, node_size=540):
    """
    Draw final visible vertices above everything else.
    """
    xs = [pos[v][0] for v in pos]
    ys = [pos[v][1] for v in pos]

    ax.scatter(
        xs,
        ys,
        s=node_size,
        c="white",
        edgecolors="black",
        linewidths=1.8,
        zorder=20,
    )


def _draw_top_labels(ax, pos, font_size=10):
    """
    Draw labels above everything else.
    """
    for v, (x, y) in pos.items():
        ax.text(
            x,
            y,
            str(v),
            fontsize=font_size,
            ha="center",
            va="center",
            color="black",
            zorder=21,
        )


def draw_complex(K, ax=None, title="", seed=42, show=True):
    """
    Draw a simplicial complex up to dimension 2.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    G = _build_graph(K)
    pos = _get_positions(K, seed=seed)

    for simplex in K.simplicial_complex:
        if len(simplex) == 3:
            points = [pos[v] for v in simplex]
            triangle = Polygon(
                points,
                closed=True,
                alpha=0.25,
                facecolor="lightgray",
                edgecolor="gray",
                linewidth=1.5,
                zorder=1,
            )
            ax.add_patch(triangle)

    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        edge_color="gray",
        width=2,
        alpha=0.8,
    )

    _draw_top_nodes(ax, pos, node_size=600)
    _draw_top_labels(ax, pos, font_size=10)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")

    if show:
        plt.show()

    return ax, pos


def draw_cover_on_complex(
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
):
    """
    Draw a simplicial complex together with a categorical cover.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    G = _build_graph(K)
    pos = _get_positions(K, seed=seed)

    if title is None:
        title = _make_gscat_title(
            complex_name=complex_name,
            gscat_value=gscat_value,
            arboricity_value=arboricity_value,
            default_title="gscat cover",
        )

    colors = [
        "red",
        "blue",
        "green",
        "orange",
        "purple",
        "cyan",
        "magenta",
        "brown",
        "pink",
        "olive",
    ]

    edge_memberships = defaultdict(list)
    vertex_memberships = defaultdict(list)
    triangle_memberships = defaultdict(list)

    for idx, subcomplex in enumerate(cover):
        for simplex in subcomplex:
            if len(simplex) == 1:
                v = next(iter(simplex))
                vertex_memberships[v].append(idx)

            elif len(simplex) == 2:
                edge_memberships[_edge_key(tuple(simplex))].append(idx)

            elif len(simplex) == 3:
                triangle_memberships[frozenset(simplex)].append(idx)

    # background triangles
    for simplex in K.simplicial_complex:
        if len(simplex) == 3:
            points = [pos[v] for v in simplex]
            triangle = Polygon(
                points,
                closed=True,
                alpha=0.08,
                facecolor="gray",
                edgecolor="gray",
                linewidth=1,
                zorder=1,
            )
            ax.add_patch(triangle)

    # background edges
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        edge_color="gray",
        width=1.5,
        alpha=0.35,
    )

    # colored triangles
    for simplex, members in triangle_memberships.items():
        points = [pos[v] for v in simplex]

        for cover_index in members:
            color = colors[cover_index % len(colors)]
            alpha = 0.14 if len(members) > 1 else 0.22

            triangle = Polygon(
                points,
                closed=True,
                alpha=alpha,
                facecolor=color,
                edgecolor=color,
                linewidth=2,
                zorder=3,
            )
            ax.add_patch(triangle)

    # colored edges
    for edge, members in edge_memberships.items():
        count = len(members)

        for local_index, cover_index in enumerate(members):
            color = colors[cover_index % len(colors)]

            if offset_shared_edges and count > 1:
                offset = (local_index - (count - 1) / 2) * 0.035
            else:
                offset = 0.0

            _draw_offset_edge(
                ax,
                pos,
                edge,
                color=color,
                offset=offset,
                width=4,
                alpha=0.90,
                zorder=5,
            )

    # colored halos for vertices
    for v, members in vertex_memberships.items():
        for local_index, cover_index in enumerate(members):
            color = colors[cover_index % len(colors)]

            ax.scatter(
                [pos[v][0]],
                [pos[v][1]],
                s=850 + 90 * local_index,
                color=color,
                alpha=0.12,
                edgecolors="none",
                zorder=6,
            )

    # FINAL top nodes and labels
    _draw_top_nodes(ax, pos, node_size=540)
    _draw_top_labels(ax, pos, font_size=10)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")

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
):
    """
    Draw the original complex and its cover side by side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    draw_complex(
        K,
        ax=axes[0],
        title=f"Original complex: {complex_name}",
        seed=seed,
        show=False,
    )

    draw_cover_on_complex(
        K,
        cover,
        ax=axes[1],
        complex_name=complex_name,
        gscat_value=gscat_value,
        arboricity_value=arboricity_value,
        seed=seed,
        show=False,
    )

    plt.tight_layout()
    plt.show()

    return fig, axes