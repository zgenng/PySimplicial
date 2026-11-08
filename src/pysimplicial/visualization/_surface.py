"""Plotly visualization of triangulated surfaces and their covers."""

import math
from collections import defaultdict
from collections.abc import Mapping
from itertools import combinations, permutations

import numpy as np

from ..complex import SimplicialComplex
from ._layout import DEFAULT_PALETTE, _get_positions, _ordered_edge, _simplex_sort_key, _vertex_sort_key

DEFAULT_SURFACE_THEME = {
    "background": "#FFFFFF",
    "base_face": "#CBD5E1",
    "base_edge": "#64748B",
    "node_face": "#FFFFFF",
    "node_edge": "#0F172A",
    "label": "#0F172A",
    "title": "#0F172A",
}


_CSASZAR_FACETS = frozenset(
    frozenset(face)
    for face in (
        (0, 1, 2),
        (0, 1, 3),
        (0, 2, 4),
        (0, 3, 5),
        (0, 4, 6),
        (0, 5, 6),
        (1, 2, 5),
        (1, 3, 6),
        (1, 4, 5),
        (1, 4, 6),
        (2, 3, 4),
        (2, 3, 6),
        (2, 5, 6),
        (3, 4, 5),
    )
)


def _surface_triangles(K):
    triangles = {
        frozenset(simplex)
        for simplex in K.iter_faces(2)
    }
    return [
        tuple(sorted(triangle, key=_vertex_sort_key))
        for triangle in sorted(triangles, key=_simplex_sort_key)
    ]


def _surface_edges(triangles):
    edges = {
        frozenset(edge)
        for triangle in triangles
        for edge in combinations(triangle, 2)
    }
    return sorted(
        edges,
        key=lambda edge: tuple(
            _vertex_sort_key(vertex) for vertex in _ordered_edge(edge)
        ),
    )


def _surface_edge_incidence(triangles):
    incidence = defaultdict(list)
    for triangle_index, triangle in enumerate(triangles):
        for edge in combinations(triangle, 2):
            incidence[frozenset(edge)].append(triangle_index)
    return incidence


def _icosahedron_geometry():
    """Return an oriented icosahedron and its six antipodal vertex pairs."""
    golden_ratio = (1.0 + math.sqrt(5.0)) / 2.0
    points = []
    for second in (-1.0, 1.0):
        for third in (-golden_ratio, golden_ratio):
            points.append((0.0, second, third))
    for first in (-1.0, 1.0):
        for second in (-golden_ratio, golden_ratio):
            points.append((first, second, 0.0))
    for first in (-golden_ratio, golden_ratio):
        for third in (-1.0, 1.0):
            points.append((first, 0.0, third))
    points = np.asarray(points, dtype=float)

    faces = []
    tolerance = 1e-10
    for face in combinations(range(len(points)), 3):
        a, b, c = points[list(face)]
        normal = np.cross(b - a, c - a)
        normal_length = float(np.linalg.norm(normal))
        if normal_length <= tolerance:
            continue
        signed_distances = (points - a) @ normal
        side_tolerance = tolerance * normal_length
        supports_hull = (
            np.all(signed_distances <= side_tolerance)
            or np.all(signed_distances >= -side_tolerance)
        )
        if not supports_hull:
            continue

        oriented_face = face
        face_centre = (a + b + c) / 3.0
        if float(np.dot(normal, face_centre)) < 0.0:
            oriented_face = (face[0], face[2], face[1])
        faces.append(oriented_face)

    pair_of = {}
    representative_of_pair = {}
    for vertex_index, point in enumerate(points):
        if vertex_index in pair_of:
            continue
        opposite_index = int(
            np.argmin(np.linalg.norm(points + point, axis=1))
        )
        pair_index = len(representative_of_pair)
        pair_of[vertex_index] = pair_index
        pair_of[opposite_index] = pair_index
        representative_of_pair[pair_index] = vertex_index

    if len(faces) != 20 or len(representative_of_pair) != 6:
        raise RuntimeError("Failed to construct the canonical icosahedron.")
    return points, faces, pair_of, representative_of_pair


def _find_quotient_isomorphism(vertices, triangles, quotient_facets):
    """Map the vertices of K to the vertices of a small canonical quotient."""
    if len(vertices) != len(set().union(*quotient_facets)):
        return None

    vertex_index = {vertex: index for index, vertex in enumerate(vertices)}
    indexed_facets = frozenset(
        frozenset(vertex_index[vertex] for vertex in triangle)
        for triangle in triangles
    )
    canonical_vertices = sorted(set().union(*quotient_facets))
    for candidate in permutations(canonical_vertices):
        relabelled = frozenset(
            frozenset(candidate[index] for index in face)
            for face in indexed_facets
        )
        if relabelled == quotient_facets:
            return {
                vertex: candidate[index]
                for index, vertex in enumerate(vertices)
            }
    return None


def _rp2_double_cover_model(K, triangles, cover=None):
    """Build the icosahedral orientable double cover of the minimal RP2."""
    vertices = sorted(K.vertices, key=_vertex_sort_key)
    if len(vertices) != 6 or len(triangles) != 10:
        return None
    if len(_surface_edges(triangles)) != 15:
        return None

    points, faces, pair_of, representatives = _icosahedron_geometry()
    quotient_facets = frozenset(
        frozenset(pair_of[vertex] for vertex in face)
        for face in faces
    )
    if len(quotient_facets) != 10:
        return None

    vertex_to_pair = _find_quotient_isomorphism(
        vertices,
        triangles,
        quotient_facets,
    )
    if vertex_to_pair is None:
        return None
    pair_to_vertex = {
        pair_index: vertex
        for vertex, pair_index in vertex_to_pair.items()
    }

    display_vertex = {}
    projection = {}
    vertex_labels = {}
    positions = {}
    for point_index, point in enumerate(points):
        pair_index = pair_of[point_index]
        original_vertex = pair_to_vertex[pair_index]
        sheet = 1 if representatives[pair_index] == point_index else -1
        lifted_vertex = (original_vertex, sheet)
        display_vertex[point_index] = lifted_vertex
        projection[lifted_vertex] = original_vertex
        sign = "+" if sheet == 1 else "−"
        vertex_labels[lifted_vertex] = f"{original_vertex}{sign}"
        positions[lifted_vertex] = point

    display_facets = [
        tuple(display_vertex[index] for index in face)
        for face in faces
    ]
    display_complex = SimplicialComplex(display_facets)

    lifted_cover = None
    if cover is not None:
        original_elements = [part.faces() for part in cover]
        lifted_cover = []
        for element in original_elements:
            lifted_element = []
            for simplex in display_complex.simplicial_complex:
                projected = frozenset(projection[vertex] for vertex in simplex)
                if projected in element:
                    lifted_element.append(simplex)
            lifted_cover.append(SimplicialComplex(lifted_element))

    simplex_labels = {}
    for simplex in display_complex.simplicial_complex:
        projected = sorted(
            {projection[vertex] for vertex in simplex},
            key=_vertex_sort_key,
        )
        simplex_labels[frozenset(simplex)] = (
            "{" + ", ".join(map(str, projected)) + "}"
        )

    return {
        "complex": display_complex,
        "cover": lifted_cover,
        "positions": positions,
        "projection": projection,
        "vertex_labels": vertex_labels,
        "simplex_labels": simplex_labels,
        "note": (
            "display: orientable double cover "
            "S<sup>2</sup> → RP<sup>2</sup>; every face is lifted twice"
        ),
    }


def _triangle_edge_direction(triangle, edge):
    u, v = _ordered_edge(edge)
    a, b, c = triangle
    directed_boundary = ((a, b), (b, c), (c, a))
    if (u, v) in directed_boundary:
        return 1
    if (v, u) in directed_boundary:
        return -1
    raise ValueError(f"The edge {edge} is not contained in {triangle}.")


def _orient_surface_triangles(triangles):
    """Orient adjacent triangles consistently whenever this is possible."""
    incidence = _surface_edge_incidence(triangles)
    adjacency = defaultdict(list)

    for edge, triangle_indices in incidence.items():
        if len(triangle_indices) != 2:
            continue
        left, right = triangle_indices
        left_direction = _triangle_edge_direction(triangles[left], edge)
        right_direction = _triangle_edge_direction(triangles[right], edge)
        relation = -left_direction * right_direction
        adjacency[left].append((right, relation))
        adjacency[right].append((left, relation))

    signs = {}
    orientable = True
    for start in range(len(triangles)):
        if start in signs:
            continue
        signs[start] = 1
        stack = [start]
        while stack:
            triangle_index = stack.pop()
            for neighbour, relation in adjacency[triangle_index]:
                expected = signs[triangle_index] * relation
                if neighbour not in signs:
                    signs[neighbour] = expected
                    stack.append(neighbour)
                elif signs[neighbour] != expected:
                    orientable = False

    oriented = []
    for index, triangle in enumerate(triangles):
        a, b, c = triangle
        oriented.append((a, b, c) if signs.get(index, 1) == 1 else (a, c, b))
    return oriented, orientable


def _surface_components(vertices, edges):
    adjacency = {vertex: set() for vertex in vertices}
    for edge in edges:
        u, v = tuple(edge)
        adjacency[u].add(v)
        adjacency[v].add(u)

    unseen = set(vertices)
    component_count = 0
    while unseen:
        component_count += 1
        start = min(unseen, key=_vertex_sort_key)
        unseen.remove(start)
        stack = [start]
        while stack:
            vertex = stack.pop()
            for neighbour in adjacency[vertex]:
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    stack.append(neighbour)
    return component_count


def surface_statistics(K):
    """Return basic combinatorial information about a 2-dimensional complex."""
    vertices = sorted(K.vertices, key=_vertex_sort_key)
    triangles = _surface_triangles(K)
    edges = _surface_edges(triangles)
    incidence = _surface_edge_incidence(triangles)
    boundary_edges = [edge for edge, uses in incidence.items() if len(uses) == 1]
    nonmanifold_edges = [edge for edge, uses in incidence.items() if len(uses) > 2]
    _, orientable = _orient_surface_triangles(triangles)
    components = _surface_components(vertices, edges) if vertices else 0
    euler_characteristic = len(vertices) - len(edges) + len(triangles)
    closed = bool(triangles) and not boundary_edges and not nonmanifold_edges
    connected = components == 1

    genus = None
    crosscap_number = None
    if closed and connected and orientable:
        numerator = 2 - euler_characteristic
        if numerator >= 0 and numerator % 2 == 0:
            genus = numerator // 2
    elif closed and connected and not orientable:
        candidate = 2 - euler_characteristic
        if candidate >= 1:
            crosscap_number = candidate

    return {
        "vertices": len(vertices),
        "edges": len(edges),
        "triangles": len(triangles),
        "euler_characteristic": euler_characteristic,
        "boundary_edges": len(boundary_edges),
        "components": components,
        "closed": closed,
        "connected": connected,
        "orientable": orientable,
        "genus": genus,
        "crosscap_number": crosscap_number,
        "nonmanifold_edges": len(nonmanifold_edges),
    }


def _validate_triangulated_surface(K, triangles):
    if not triangles:
        raise ValueError("A triangulated surface must contain 2-simplices.")

    higher_simplices = [
        simplex for simplex in K.facets if len(simplex) > 3
    ]
    if higher_simplices:
        raise ValueError(
            "draw_surface_3d expects a 2-dimensional complex; "
            f"found a simplex with {len(higher_simplices[0])} vertices."
        )

    triangle_vertices = set().union(*(set(triangle) for triangle in triangles))
    unused_vertices = set(K.vertices) - triangle_vertices
    if unused_vertices:
        raise ValueError(
            "The complex has vertices outside the triangulated surface: "
            f"{sorted(unused_vertices, key=_vertex_sort_key)}."
        )

    triangle_edges = set(_surface_edges(triangles))
    complex_edges = {
        frozenset(simplex)
        for simplex in K.iter_faces(1)
    }
    dangling_edges = complex_edges - triangle_edges
    if dangling_edges:
        raise ValueError(
            "The complex contains edges that belong to no triangle: "
            f"{sorted(dangling_edges, key=_simplex_sort_key)[:8]}."
        )

    incidence = _surface_edge_incidence(triangles)
    nonmanifold_edges = {
        edge: uses for edge, uses in incidence.items() if len(uses) > 2
    }
    if nonmanifold_edges:
        edge, uses = next(iter(nonmanifold_edges.items()))
        raise ValueError(
            f"The edge {_ordered_edge(edge)} belongs to {len(uses)} triangles; "
            "a surface edge may belong to at most two."
        )

    boundary_edges = {edge for edge, uses in incidence.items() if len(uses) == 1}
    for vertex in sorted(K.vertices, key=_vertex_sort_key):
        link_adjacency = defaultdict(set)
        for triangle in triangles:
            if vertex not in triangle:
                continue
            opposite = [item for item in triangle if item != vertex]
            left, right = opposite
            link_adjacency[left].add(right)
            link_adjacency[right].add(left)

        if not link_adjacency:
            raise ValueError(f"The link of vertex {vertex!r} is empty.")

        unseen = set(link_adjacency)
        link_components = 0
        while unseen:
            link_components += 1
            start = unseen.pop()
            stack = [start]
            while stack:
                item = stack.pop()
                for neighbour in link_adjacency[item]:
                    if neighbour in unseen:
                        unseen.remove(neighbour)
                        stack.append(neighbour)

        degrees = [len(link_adjacency[item]) for item in link_adjacency]
        vertex_on_boundary = any(vertex in edge for edge in boundary_edges)
        if vertex_on_boundary:
            valid_link = (
                link_components == 1
                and degrees.count(1) == 2
                and all(degree in (1, 2) for degree in degrees)
            )
        else:
            valid_link = link_components == 1 and all(
                degree == 2 for degree in degrees
            )
        if not valid_link:
            expected = "a path" if vertex_on_boundary else "a cycle"
            raise ValueError(
                f"The link of vertex {vertex!r} is not {expected}; "
                "the complex is not a triangulated 2-manifold."
            )


def _normalize_positions_3d(pos):
    if not pos:
        return {}
    vertices = list(pos)
    points = np.asarray([pos[vertex] for vertex in vertices], dtype=float)
    points -= points.mean(axis=0)
    scale = max(float(np.ptp(points, axis=0).max()), 1e-12)
    points = 2.0 * points / scale
    return {
        vertex: point
        for vertex, point in zip(vertices, points)
    }


def _validate_geometric_triangles(triangles, pos, tolerance=1e-8):
    """Reject coordinates that flatten one of the declared 2-simplices."""
    edges = _surface_edges(triangles)
    edge_lengths = [
        float(np.linalg.norm(pos[u] - pos[v]))
        for u, v in (_ordered_edge(edge) for edge in edges)
    ]
    scale = max(float(np.median(edge_lengths)) if edge_lengths else 1.0, 1e-12)
    minimum_double_area = tolerance * scale * scale
    degenerate = []
    for triangle in triangles:
        a, b, c = (np.asarray(pos[vertex], dtype=float) for vertex in triangle)
        double_area = float(np.linalg.norm(np.cross(b - a, c - a)))
        if double_area <= minimum_double_area:
            degenerate.append(tuple(triangle))

    if degenerate:
        raise ValueError(
            "The chosen coordinates flatten these triangles: "
            f"{degenerate[:8]}. Supply different pos coordinates or use "
            "another layout/seed."
        )


def _coerce_positions_3d(K, pos):
    vertices = sorted(K.vertices, key=_vertex_sort_key)
    if isinstance(pos, Mapping):
        missing = set(vertices) - set(pos)
        if missing:
            raise ValueError(
                "The supplied 3D positions omit vertices: "
                f"{sorted(missing, key=_vertex_sort_key)}."
            )
        raw = {vertex: np.asarray(pos[vertex], dtype=float) for vertex in vertices}
    else:
        array = np.asarray(pos, dtype=float)
        if array.ndim != 2 or array.shape[0] != len(vertices):
            raise ValueError(
                "pos must be a vertex-to-coordinate mapping or an array with "
                f"shape ({len(vertices)}, 2) / ({len(vertices)}, 3)."
            )
        raw = {vertex: array[index] for index, vertex in enumerate(vertices)}

    result = {}
    for vertex, point in raw.items():
        point = np.asarray(point, dtype=float).reshape(-1)
        if point.size == 2:
            point = np.append(point, 0.0)
        if point.size != 3:
            raise ValueError(
                f"The position of vertex {vertex!r} must have 2 or 3 coordinates."
            )
        if not np.isfinite(point).all():
            raise ValueError(f"The position of vertex {vertex!r} is not finite.")
        result[vertex] = point
    return result


def _spring_positions_3d(vertices, edges, seed=42, iterations=500):
    count = len(vertices)
    if count == 0:
        return {}
    if count == 1:
        return {vertices[0]: np.zeros(3)}

    index = {vertex: i for i, vertex in enumerate(vertices)}
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(count, 3))
    norms = np.linalg.norm(points, axis=1)
    points /= np.maximum(norms[:, None], 1e-12)
    points += rng.normal(scale=0.025, size=points.shape)

    edge_indices = [
        (index[u], index[v])
        for u, v in (_ordered_edge(edge) for edge in edges)
    ]
    ideal = 1.7 / max(count ** (1.0 / 3.0), 1.0)
    temperature = 0.25

    for step in range(max(int(iterations), 1)):
        delta = points[:, None, :] - points[None, :, :]
        distance = np.linalg.norm(delta, axis=2)
        np.fill_diagonal(distance, np.inf)
        distance = np.maximum(distance, 1e-6)

        displacement = (
            delta * (ideal * ideal / (distance ** 3))[:, :, None]
        ).sum(axis=1)

        for left, right in edge_indices:
            vector = points[left] - points[right]
            length = max(float(np.linalg.norm(vector)), 1e-6)
            force = vector * (length / ideal)
            displacement[left] -= force
            displacement[right] += force

        displacement -= 0.025 * points
        norms = np.linalg.norm(displacement, axis=1)
        safe_norms = np.maximum(norms, 1e-12)
        step_size = np.minimum(norms, temperature)
        points += displacement / safe_norms[:, None] * step_size[:, None]
        points -= points.mean(axis=0)
        temperature = 0.25 * (1.0 - (step + 1) / max(iterations, 1)) ** 1.35

    return {
        vertex: point
        for vertex, point in zip(vertices, points)
    }


def _csaszar_positions(K, triangles):
    """Return a non-self-intersecting Csaszar realization when K is the 7-torus."""
    vertices = sorted(K.vertices, key=_vertex_sort_key)
    if len(vertices) != 7 or len(triangles) != 14:
        return None
    if len(_surface_edges(triangles)) != 21:
        return None

    vertex_index = {vertex: index for index, vertex in enumerate(vertices)}
    indexed_facets = [
        frozenset(vertex_index[vertex] for vertex in triangle)
        for triangle in triangles
    ]

    isomorphism = None
    for candidate in permutations(range(7)):
        relabelled = frozenset(
            frozenset(candidate[index] for index in face)
            for face in indexed_facets
        )
        if relabelled == _CSASZAR_FACETS:
            isomorphism = candidate
            break
    if isomorphism is None:
        return None

    root_two = math.sqrt(2.0)
    geometric_vertices = np.asarray(
        [
            (12.0, 0.0, -6.0 * root_two),
            (-12.0, 0.0, -6.0 * root_two),
            (0.0, 12.0, 6.0 * root_two),
            (0.0, -12.0, 6.0 * root_two),
            (-4.0, -3.0, root_two / 2.0),
            (4.0, 3.0, root_two / 2.0),
            (0.0, 0.0, 8.0 * root_two / 3.0),
        ],
        dtype=float,
    )
    # This permutation realizes the canonical facet labelling above.
    canonical_to_geometric = (0, 1, 2, 3, 5, 6, 4)
    canonical_positions = geometric_vertices[list(canonical_to_geometric)]

    return {
        vertex: canonical_positions[isomorphism[index]]
        for index, vertex in enumerate(vertices)
    }


def surface_positions_3d(
    K,
    pos=None,
    *,
    layout="auto",
    seed=42,
    iterations=500,
    normalize=True,
):
    """Resolve 3D coordinates for a triangulated surface.

    ``pos`` may be a mapping ``vertex -> (x, y, z)`` or an array whose rows
    follow the sorted vertex order.  With ``layout='auto'`` the minimal
    7-vertex torus receives a true Csaszar-polyhedron realization, spheres
    receive a spherical layout, and other complexes use a deterministic
    three-dimensional spring layout.
    """
    triangles = _surface_triangles(K)
    vertices = sorted(K.vertices, key=_vertex_sort_key)
    edges = _surface_edges(triangles)

    supported = {"auto", "spring", "sphere", "flat", "csaszar"}
    if layout not in supported:
        expected = ", ".join(repr(item) for item in sorted(supported))
        raise ValueError(f"Unknown layout={layout!r}; expected one of {expected}.")

    if pos is not None:
        result = _coerce_positions_3d(K, pos)
    elif layout == "csaszar":
        result = _csaszar_positions(K, triangles)
        if result is None:
            raise ValueError(
                "layout='csaszar' is available only for a triangulation "
                "combinatorially isomorphic to the minimal 7-vertex torus."
            )
    elif layout == "auto":
        result = _csaszar_positions(K, triangles)
        if result is None:
            result = _spring_positions_3d(
                vertices,
                edges,
                seed=seed,
                iterations=iterations,
            )
            statistics = surface_statistics(K)
            if statistics["closed"] and statistics["genus"] == 0:
                centered = _normalize_positions_3d(result)
                result = {}
                for vertex, point in centered.items():
                    length = max(float(np.linalg.norm(point)), 1e-12)
                    result[vertex] = point / length
    elif layout == "flat":
        flat = _get_positions(K, seed=seed, layout="spring")
        result = {
            vertex: np.asarray((point[0], point[1], 0.0), dtype=float)
            for vertex, point in flat.items()
        }
    else:
        result = _spring_positions_3d(
            vertices,
            edges,
            seed=seed,
            iterations=iterations,
        )
        if layout == "sphere":
            centered = _normalize_positions_3d(result)
            result = {}
            for vertex, point in centered.items():
                length = max(float(np.linalg.norm(point)), 1e-12)
                result[vertex] = point / length

    return _normalize_positions_3d(result) if normalize else result


def _surface_geometry(triangles, pos):
    oriented, orientable = _orient_surface_triangles(triangles)

    if orientable and oriented:
        signed_volume = 0.0
        for triangle in oriented:
            a, b, c = (pos[vertex] for vertex in triangle)
            signed_volume += float(np.dot(a, np.cross(b, c))) / 6.0
        if signed_volume < 0.0:
            oriented = [(a, c, b) for a, b, c in oriented]

    face_normals = {}
    face_centres = {}
    vertex_normal_sums = defaultdict(lambda: np.zeros(3, dtype=float))
    edge_normal_sums = defaultdict(lambda: np.zeros(3, dtype=float))

    for triangle in oriented:
        points = np.asarray([pos[vertex] for vertex in triangle], dtype=float)
        normal = np.cross(points[1] - points[0], points[2] - points[0])
        length = float(np.linalg.norm(normal))
        if length <= 1e-12:
            normal = np.asarray((0.0, 0.0, 1.0))
        else:
            normal /= length

        key = frozenset(triangle)
        face_normals[key] = normal
        face_centres[key] = points.mean(axis=0)
        for vertex in triangle:
            vertex_normal_sums[vertex] += normal
        for edge in combinations(triangle, 2):
            edge_normal_sums[frozenset(edge)] += normal

    def normalized(vector, fallback):
        length = float(np.linalg.norm(vector))
        if length <= 1e-12:
            return np.asarray(fallback, dtype=float)
        return vector / length

    centre = np.asarray(list(pos.values()), dtype=float).mean(axis=0)
    vertex_normals = {
        vertex: normalized(normal, pos[vertex] - centre)
        for vertex, normal in vertex_normal_sums.items()
    }
    edge_normals = {
        edge: normalized(normal, (0.0, 0.0, 1.0))
        for edge, normal in edge_normal_sums.items()
    }

    edges = _surface_edges(triangles)
    edge_lengths = [
        float(np.linalg.norm(pos[u] - pos[v]))
        for u, v in (_ordered_edge(edge) for edge in edges)
    ]
    characteristic_length = (
        float(np.median(edge_lengths)) if edge_lengths else 1.0
    )

    return {
        "triangles": oriented,
        "face_normals": face_normals,
        "face_centres": face_centres,
        "vertex_normals": vertex_normals,
        "edge_normals": edge_normals,
        "characteristic_length": max(characteristic_length, 1e-6),
        "orientable": orientable,
    }


def _surface_cover_memberships(K, cover, validate=True):
    target = K.faces()
    elements = [part.faces() for part in cover]
    union = set().union(*elements) if elements else set()

    if validate:
        missing = target - union
        extra = union - target
        if missing or extra:
            missing_preview = sorted(missing, key=_simplex_sort_key)[:8]
            extra_preview = sorted(extra, key=_simplex_sort_key)[:8]
            raise ValueError(
                "cover does not coincide with the surface: "
                f"missing={missing_preview}, extra={extra_preview}."
            )

    memberships = defaultdict(list)
    for cover_index, element in enumerate(elements):
        for simplex in target.intersection(element):
            memberships[simplex].append(cover_index)
    return memberships, elements


def _mesh_indices(vertices, triangles):
    index = {vertex: position for position, vertex in enumerate(vertices)}
    return (
        [index[triangle[0]] for triangle in triangles],
        [index[triangle[1]] for triangle in triangles],
        [index[triangle[2]] for triangle in triangles],
    )


def _line_coordinates(segments):
    x_values = []
    y_values = []
    z_values = []
    hover_values = []
    for start, end, hover_text in segments:
        x_values.extend((start[0], end[0], None))
        y_values.extend((start[1], end[1], None))
        z_values.extend((start[2], end[2], None))
        hover_values.extend((hover_text, hover_text, None))
    return x_values, y_values, z_values, hover_values


def _cover_face_mesh(
    triangles,
    positions,
    geometry,
    memberships,
    cover_index,
    shared_face_step,
    simplex_labels=None,
):
    x_values = []
    y_values = []
    z_values = []
    i_values = []
    j_values = []
    k_values = []
    hover_values = []
    simplex_count = 0

    for triangle in triangles:
        key = frozenset(triangle)
        members = memberships.get(key, [])
        if cover_index not in members:
            continue

        local_index = members.index(cover_index)
        scale = 1.0
        if len(members) > 1:
            scale = max(0.58, 1.0 - shared_face_step * local_index)

        points = np.asarray([positions[vertex] for vertex in triangle], dtype=float)
        centre = geometry["face_centres"][key]
        normal = geometry["face_normals"][key]
        lift = geometry["characteristic_length"] * (
            0.006 + 0.010 * local_index
        )
        points = centre + scale * (points - centre) + lift * normal

        start_index = len(x_values)
        x_values.extend(points[:, 0])
        y_values.extend(points[:, 1])
        z_values.extend(points[:, 2])
        i_values.append(start_index)
        j_values.append(start_index + 1)
        k_values.append(start_index + 2)
        simplex_text = (
            simplex_labels.get(key)
            if simplex_labels is not None and key in simplex_labels
            else "{" + ", ".join(map(str, triangle)) + "}"
        )
        hover_values.extend((simplex_text, simplex_text, simplex_text))
        simplex_count += 1

    return {
        "x": x_values,
        "y": y_values,
        "z": z_values,
        "i": i_values,
        "j": j_values,
        "k": k_values,
        "hover": hover_values,
        "count": simplex_count,
    }


def _cover_edge_segments(
    edges,
    positions,
    geometry,
    memberships,
    cover_index,
    simplex_labels=None,
):
    segments = []
    spacing = 0.026 * geometry["characteristic_length"]
    lift = 0.015 * geometry["characteristic_length"]

    for edge in edges:
        members = memberships.get(edge, [])
        if cover_index not in members:
            continue

        local_index = members.index(cover_index)
        centred_index = local_index - (len(members) - 1) / 2.0
        u, v = _ordered_edge(edge)
        start = np.asarray(positions[u], dtype=float)
        end = np.asarray(positions[v], dtype=float)
        tangent = end - start
        tangent_length = max(float(np.linalg.norm(tangent)), 1e-12)
        tangent /= tangent_length

        normal = geometry["edge_normals"].get(
            edge,
            np.asarray((0.0, 0.0, 1.0)),
        )
        lateral = np.cross(normal, tangent)
        lateral_length = float(np.linalg.norm(lateral))
        if lateral_length <= 1e-12:
            fallback = np.asarray((1.0, 0.0, 0.0))
            if abs(float(np.dot(fallback, tangent))) > 0.9:
                fallback = np.asarray((0.0, 1.0, 0.0))
            lateral = np.cross(fallback, tangent)
            lateral_length = max(float(np.linalg.norm(lateral)), 1e-12)
        lateral /= lateral_length

        offset = centred_index * spacing * lateral + lift * normal
        edge_text = (
            simplex_labels.get(edge)
            if simplex_labels is not None and edge in simplex_labels
            else "{" + ", ".join(map(str, _ordered_edge(edge))) + "}"
        )
        segments.append((start + offset, end + offset, edge_text))
    return segments


def _plotly_graph_objects():
    try:
        import plotly.graph_objects as go
    except ImportError as error:
        raise ImportError(
            "Interactive surface visualization requires Plotly. "
            "Install it with: pip install \"pysimplicial[plot3d]\""
        ) from error
    return go


def _surface_title(
    K,
    complex_name,
    title,
    gscat_value,
    statistics,
    representation_note=None,
):
    if title is None:
        title = f"Triangulation of {complex_name}"
        if gscat_value is not None:
            title += f" &nbsp;·&nbsp; gscat({complex_name}) = {gscat_value}"

    details = (
        f"V = {statistics['vertices']} &nbsp;·&nbsp; "
        f"E = {statistics['edges']} &nbsp;·&nbsp; "
        f"F = {statistics['triangles']} &nbsp;·&nbsp; "
        f"χ = {statistics['euler_characteristic']}"
    )
    if statistics["genus"] is not None:
        details += f" &nbsp;·&nbsp; genus = {statistics['genus']}"
    elif statistics["crosscap_number"] is not None:
        details += f" &nbsp;·&nbsp; crosscaps = {statistics['crosscap_number']}"
    if statistics["boundary_edges"]:
        details += f" &nbsp;·&nbsp; boundary edges = {statistics['boundary_edges']}"
    if representation_note:
        details += f"<br>{representation_note}"
    return f"<b>{title}</b><br><sup>{details}</sup>"


def draw_surface_3d(
    K,
    cover=None,
    *,
    pos=None,
    layout="auto",
    representation="auto",
    seed=42,
    iterations=500,
    title=None,
    complex_name="K",
    gscat_value=None,
    show=True,
    renderer=None,
    save_html=None,
    validate=True,
    palette=None,
    cover_labels=None,
    show_edges=True,
    show_cover_edges=True,
    show_vertices=True,
    show_labels=True,
    controls=True,
    base_opacity=0.16,
    face_opacity=0.64,
    shared_face_step=0.11,
    edge_width=2.5,
    cover_edge_width=5.0,
    node_size=5.5,
    width=980,
    height=720,
    theme=None,
):
    """Draw a rotatable triangulated surface and its categorical cover.

    Parameters
    ----------
    K:
        A ``SimplicialComplex`` containing a triangulated 2-manifold.
    cover:
        The cover returned by the gscat computation.  Each cover element is a SimplicialComplex.
    pos:
        Optional ``vertex -> (x, y, z)`` mapping or an ``(n, 3)`` array.
        Supplying a geometric realization is the most faithful option.
    layout:
        ``'auto'``, ``'spring'``, ``'sphere'``, ``'flat'``, or ``'csaszar'``.
        ``'auto'`` recognizes every relabelling of the minimal 7-vertex torus.
    representation:
        ``'auto'`` uses the icosahedral orientable double cover for the
        minimal projective plane, so its triangles never self-intersect.
        Use ``'direct'`` to force the original immersed 3D picture, or
        ``'double_cover'`` to request the RP2 double cover explicitly.
    save_html:
        Optional path for a self-contained interactive HTML file.

    Returns
    -------
    (figure, positions)
        A Plotly figure and the resolved vertex-coordinate dictionary.
    """
    go = _plotly_graph_objects()
    surface_theme = {**DEFAULT_SURFACE_THEME, **(theme or {})}
    colors = tuple(palette or DEFAULT_PALETTE)
    if not colors:
        raise ValueError("palette must contain at least one color.")

    supported_representations = {"auto", "direct", "double_cover"}
    if representation not in supported_representations:
        expected = ", ".join(
            repr(item) for item in sorted(supported_representations)
        )
        raise ValueError(
            f"Unknown representation={representation!r}; "
            f"expected one of {expected}."
        )

    original_complex = K
    original_triangles = _surface_triangles(original_complex)
    if validate:
        _validate_triangulated_surface(original_complex, original_triangles)
    original_statistics = surface_statistics(original_complex)
    original_cover = None if cover is None else tuple(cover)
    if original_cover is not None and validate:
        _surface_cover_memberships(
            original_complex,
            original_cover,
            validate=True,
        )

    vertex_display_labels = None
    simplex_display_labels = None
    representation_note = None
    use_double_cover = representation == "double_cover"
    if representation == "auto":
        use_double_cover = pos is None and layout == "auto"

    display_model = None
    if use_double_cover:
        display_model = _rp2_double_cover_model(
            original_complex,
            original_triangles,
            cover=original_cover,
        )
        if representation == "double_cover" and display_model is None:
            raise ValueError(
                "representation='double_cover' currently supports the "
                "minimal 6-vertex triangulation of RP2."
            )

    if display_model is not None:
        K = display_model["complex"]
        cover = display_model["cover"]
        pos = display_model["positions"]
        vertex_display_labels = display_model["vertex_labels"]
        simplex_display_labels = display_model["simplex_labels"]
        representation_note = display_model["note"]
    else:
        K = original_complex
        cover = original_cover
        if (
            original_statistics["closed"]
            and not original_statistics["orientable"]
        ):
            representation_note = (
                "direct 3D immersion; self-intersections are unavoidable "
                "for a closed non-orientable surface"
            )

    triangles = _surface_triangles(K)
    if validate and display_model is not None:
        _validate_triangulated_surface(K, triangles)
    vertices = sorted(K.vertices, key=_vertex_sort_key)
    edges = _surface_edges(triangles)
    positions = surface_positions_3d(
        K,
        pos=pos,
        layout=layout,
        seed=seed,
        iterations=iterations,
        normalize=True,
    )
    if validate:
        _validate_geometric_triangles(triangles, positions)
    geometry = _surface_geometry(triangles, positions)
    oriented_triangles = geometry["triangles"]
    statistics = original_statistics

    if cover is None:
        cover = ()
        memberships = defaultdict(list)
        cover_elements = []
    else:
        cover = tuple(cover)
        memberships, cover_elements = _surface_cover_memberships(
            K,
            cover,
            validate=validate,
        )

    if cover_labels is not None and len(cover_labels) != len(cover):
        raise ValueError("cover_labels must have the same length as cover.")
    labels = [
        str(cover_labels[index])
        if cover_labels is not None
        else f"U_{index + 1}"
        for index in range(len(cover))
    ]

    figure = go.Figure()
    trace_owners = []

    def add_trace(trace, cover_owner=None):
        figure.add_trace(trace)
        trace_owners.append(cover_owner)

    x_values = [positions[vertex][0] for vertex in vertices]
    y_values = [positions[vertex][1] for vertex in vertices]
    z_values = [positions[vertex][2] for vertex in vertices]
    i_values, j_values, k_values = _mesh_indices(vertices, oriented_triangles)

    add_trace(
        go.Mesh3d(
            x=x_values,
            y=y_values,
            z=z_values,
            i=i_values,
            j=j_values,
            k=k_values,
            color=surface_theme["base_face"],
            opacity=base_opacity,
            flatshading=True,
            name="Triangulation",
            showlegend=False,
            hoverinfo="skip",
            lighting={
                "ambient": 0.78,
                "diffuse": 0.72,
                "specular": 0.18,
                "roughness": 0.72,
                "fresnel": 0.08,
            },
            lightposition={"x": 120, "y": 180, "z": 240},
        )
    )

    if show_edges:
        base_segments = []
        for edge in edges:
            u, v = _ordered_edge(edge)
            edge_text = (
                simplex_display_labels.get(edge)
                if simplex_display_labels is not None
                and edge in simplex_display_labels
                else "{" + ", ".join(map(str, (u, v))) + "}"
            )
            base_segments.append((positions[u], positions[v], edge_text))
        edge_x, edge_y, edge_z, edge_hover = _line_coordinates(base_segments)
        add_trace(
            go.Scatter3d(
                x=edge_x,
                y=edge_y,
                z=edge_z,
                mode="lines",
                line={"color": surface_theme["base_edge"], "width": edge_width},
                text=edge_hover,
                hovertemplate="edge %{text}<extra></extra>",
                name="Edges",
                showlegend=False,
            )
        )

    for cover_index, label in enumerate(labels):
        color = colors[cover_index % len(colors)]
        face_mesh = _cover_face_mesh(
            oriented_triangles,
            positions,
            geometry,
            memberships,
            cover_index,
            shared_face_step,
            simplex_labels=simplex_display_labels,
        )
        edge_count = sum(
            len(simplex) == 2 for simplex in cover_elements[cover_index]
        )
        legend_name = (
            f"{label} · {face_mesh['count']} faces · {edge_count} edges"
        )

        if face_mesh["count"]:
            add_trace(
                go.Mesh3d(
                    x=face_mesh["x"],
                    y=face_mesh["y"],
                    z=face_mesh["z"],
                    i=face_mesh["i"],
                    j=face_mesh["j"],
                    k=face_mesh["k"],
                    color=color,
                    opacity=face_opacity,
                    flatshading=True,
                    text=face_mesh["hover"],
                    hovertemplate=(
                        f"<b>{label}</b><br>triangle %{{text}}<extra></extra>"
                    ),
                    name=legend_name,
                    legendgroup=f"cover-{cover_index}",
                    showlegend=True,
                    lighting={
                        "ambient": 0.72,
                        "diffuse": 0.68,
                        "specular": 0.20,
                        "roughness": 0.74,
                        "fresnel": 0.06,
                    },
                ),
                cover_owner=cover_index,
            )

        if show_cover_edges:
            segments = _cover_edge_segments(
                edges,
                positions,
                geometry,
                memberships,
                cover_index,
                simplex_labels=simplex_display_labels,
            )
            if segments:
                edge_x, edge_y, edge_z, edge_hover = _line_coordinates(segments)
                add_trace(
                    go.Scatter3d(
                        x=edge_x,
                        y=edge_y,
                        z=edge_z,
                        mode="lines",
                        line={"color": color, "width": cover_edge_width},
                        text=edge_hover,
                        hovertemplate=(
                            f"<b>{label}</b><br>edge %{{text}}<extra></extra>"
                        ),
                        name=label,
                        legendgroup=f"cover-{cover_index}",
                        showlegend=not bool(face_mesh["count"]),
                    ),
                    cover_owner=cover_index,
                )

    triangle_hover = []
    triangle_centres = []
    for triangle in oriented_triangles:
        key = frozenset(triangle)
        triangle_centres.append(geometry["face_centres"][key])
        member_labels = [labels[index] for index in memberships.get(key, [])]
        membership_text = ", ".join(member_labels) if member_labels else "none"
        simplex_text = (
            simplex_display_labels.get(key)
            if simplex_display_labels is not None
            and key in simplex_display_labels
            else "{" + ", ".join(map(str, triangle)) + "}"
        )
        triangle_hover.append(
            f"<b>triangle {simplex_text}</b><br>cover: {membership_text}"
        )
    if triangle_centres:
        centres = np.asarray(triangle_centres)
        add_trace(
            go.Scatter3d(
                x=centres[:, 0],
                y=centres[:, 1],
                z=centres[:, 2],
                mode="markers",
                marker={"size": 7, "color": "rgba(15, 23, 42, 0.001)"},
                text=triangle_hover,
                hovertemplate="%{text}<extra></extra>",
                name="Triangles",
                showlegend=False,
            )
        )

    if show_vertices:
        vertex_hover = []
        for vertex in vertices:
            key = frozenset((vertex,))
            member_labels = [labels[index] for index in memberships.get(key, [])]
            membership_text = ", ".join(member_labels) if member_labels else "none"
            display_label = (
                vertex_display_labels.get(vertex, str(vertex))
                if vertex_display_labels is not None
                else str(vertex)
            )
            vertex_hover.append(
                f"<b>vertex {display_label}</b><br>cover: {membership_text}"
            )
        displayed_vertex_text = [
            vertex_display_labels.get(vertex, str(vertex))
            if vertex_display_labels is not None
            else str(vertex)
            for vertex in vertices
        ]
        add_trace(
            go.Scatter3d(
                x=x_values,
                y=y_values,
                z=z_values,
                mode="markers+text" if show_labels else "markers",
                marker={
                    "size": node_size,
                    "color": surface_theme["node_face"],
                    "line": {"color": surface_theme["node_edge"], "width": 2.0},
                },
                text=displayed_vertex_text if show_labels else None,
                textposition="top center",
                textfont={"color": surface_theme["label"], "size": 12},
                customdata=vertex_hover,
                hovertemplate="%{customdata}<extra></extra>",
                name="Vertices",
                showlegend=False,
            )
        )

    update_menus = []
    if controls and cover:
        base_visibility = [owner is None for owner in trace_owners]
        buttons = [
            {
                "label": "All covers",
                "method": "update",
                "args": [{"visible": [True] * len(trace_owners)}],
            },
            {
                "label": "Triangulation",
                "method": "update",
                "args": [{"visible": base_visibility}],
            },
        ]
        for cover_index, label in enumerate(labels):
            visibility = [
                owner is None or owner == cover_index for owner in trace_owners
            ]
            buttons.append(
                {
                    "label": label,
                    "method": "update",
                    "args": [{"visible": visibility}],
                }
            )
        update_menus.append(
            {
                "type": "dropdown",
                "direction": "down",
                "buttons": buttons,
                "x": 0.0,
                "xanchor": "left",
                "y": 0.99,
                "yanchor": "top",
                "showactive": True,
                "bgcolor": "#FFFFFF",
                "bordercolor": "#CBD5E1",
                "font": {"color": "#0F172A", "size": 12},
            }
        )

    figure.update_layout(
        title={
            "text": _surface_title(
                K,
                complex_name,
                title,
                gscat_value,
                statistics,
                representation_note=representation_note,
            ),
            "x": 0.5,
            "xanchor": "center",
            "font": {"color": surface_theme["title"], "size": 20},
        },
        width=width,
        height=height,
        paper_bgcolor=surface_theme["background"],
        plot_bgcolor=surface_theme["background"],
        margin={"l": 0, "r": 0, "b": 0, "t": 105},
        scene={
            "bgcolor": surface_theme["background"],
            "aspectmode": "data",
            "dragmode": "orbit",
            "camera": {
                "eye": {"x": 1.55, "y": -1.55, "z": 1.10},
                "projection": {"type": "perspective"},
            },
            "xaxis": {"visible": False, "showbackground": False},
            "yaxis": {"visible": False, "showbackground": False},
            "zaxis": {"visible": False, "showbackground": False},
        },
        legend={
            "orientation": "v",
            "x": 0.99,
            "xanchor": "right",
            "y": 0.99,
            "yanchor": "top",
            "bgcolor": "rgba(255,255,255,0.88)",
            "bordercolor": "#E2E8F0",
            "borderwidth": 1,
            "groupclick": "togglegroup",
        },
        updatemenus=update_menus,
        hoverlabel={
            "bgcolor": "#FFFFFF",
            "bordercolor": "#CBD5E1",
            "font": {"color": "#0F172A", "size": 12},
        },
        uirevision="pysimplicial-surface-camera",
    )

    config = {
        "displaylogo": False,
        "displayModeBar": True,
        "scrollZoom": True,
        "responsive": True,
        "toImageButtonOptions": {
            "format": "png",
            "filename": f"{complex_name}_triangulation",
            "scale": 2,
        },
    }
    if save_html is not None:
        figure.write_html(
            save_html,
            include_plotlyjs=True,
            full_html=True,
            config=config,
        )
    if show:
        figure.show(renderer=renderer, config=config)
    return figure, positions

