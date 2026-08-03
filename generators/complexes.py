"""Alternative triangulations for experiments with simplicial category.

The torus, real projective plane, Mobius band, and Klein bottle below are
genuinely different from the minimal examples used in
``verified_triangulations.py``: each is obtained by a stellar 1-to-3
subdivision of one triangular facet.  Thus one new vertex and two new
triangles are added; this is not merely a relabeling.

A stellar subdivision preserves the PL-homeomorphism type.  See
W. B. R. Lickorish, "Simplicial moves on complexes and manifolds",
Theorem 4.5:
https://homepages.math.uic.edu/~kauffman/Lickorish.pdf

The starting facet lists are taken from the following sources:

* torus: F. H. Lutz, "Triangulated Manifolds with Few Vertices";
  https://arxiv.org/abs/math/0506372
* RP^2: B. Bagchi, B. Datta, J. Spreer, Example 6.6;
  https://arxiv.org/abs/1412.0412
* Mobius band: the antistar of a vertex in RP^2_6, as described by
  S. Lawrencenko and A. Lao;
  https://arxiv.org/abs/2201.02146
* Klein bottle: Frank Lutz's small-manifold database, exposed by the
  Macaulay2 ``kleinBottleComplex`` routine;
  https://macaulay2.com/doc/Macaulay2/share/doc/Macaulay2/
  SimplicialComplexes/html/
  _klein__Bottle__Complex_lp__Polynomial__Ring_rp.html

The module also contains three unambiguous interpretations of the Russian
word "shar": a triangulated 2-sphere S^2, a 2-ball B^2 (a disk), and a
3-ball B^3.  The 2-sphere is the octahedron boundary, and the 3-ball is its
simplicial cone.
"""

from collections import defaultdict
from itertools import combinations
from simplicial_topology.Simplicial_Complex import SimplicialComplex

_TORUS_BASE_FACETS = (
    (0, 1, 2),
    (0, 3, 4),
    (0, 4, 5),
    (2, 3, 4),
    (0, 5, 6),
    (3, 5, 6),
    (1, 3, 6),
    (0, 1, 3),
    (1, 2, 5),
    (1, 4, 5),
    (2, 3, 5),
    (1, 4, 6),
    (2, 4, 6),
    (0, 2, 6),
)

_PROJECTIVE_PLANE_BASE_FACETS = (
    (0, 1, 2),
    (0, 2, 3),
    (0, 3, 4),
    (0, 4, 5),
    (0, 1, 5),
    (1, 2, 4),
    (1, 3, 4),
    (1, 3, 5),
    (2, 3, 5),
    (2, 4, 5),
)

_MOBIUS_BAND_BASE_FACETS = (
    (0, 1, 2),
    (0, 2, 3),
    (0, 3, 4),
    (1, 2, 4),
    (1, 3, 4),
)

_KLEIN_BOTTLE_BASE_FACETS = (
    (2, 6, 7),
    (0, 6, 7),
    (2, 5, 7),
    (0, 5, 7),
    (4, 5, 6),
    (3, 5, 6),
    (0, 4, 6),
    (2, 3, 6),
    (1, 4, 5),
    (0, 3, 5),
    (1, 2, 5),
    (2, 3, 4),
    (1, 3, 4),
    (0, 2, 4),
    (0, 1, 3),
    (0, 1, 2),
)


def stellar_subdivide_triangle_facets(facets, triangle, new_vertex):
    normalized_facets = tuple(tuple(facet) for facet in facets)
    target = frozenset(triangle)

    if len(target) != 3:
        raise ValueError("triangle must contain exactly three distinct vertices")

    used_vertices = {vertex for facet in normalized_facets for vertex in facet}
    if new_vertex in used_vertices:
        raise ValueError(f"new vertex {new_vertex!r} is already used")

    matches = [facet for facet in normalized_facets if frozenset(facet) == target]
    if len(matches) != 1:
        raise ValueError(
            "triangle must occur exactly once as a maximal facet; "
            f"found {len(matches)} matches"
        )

    a, b, c = matches[0]
    untouched = [
        facet for facet in normalized_facets if frozenset(facet) != target
    ]
    replacement = [
        (new_vertex, a, b),
        (new_vertex, b, c),
        (new_vertex, c, a),
    ]
    return tuple(untouched + replacement)

TORUS_ALTERNATIVE_FACETS = stellar_subdivide_triangle_facets(
    _TORUS_BASE_FACETS, (0, 1, 2), new_vertex=7
)

PROJECTIVE_PLANE_ALTERNATIVE_FACETS = stellar_subdivide_triangle_facets(
    _PROJECTIVE_PLANE_BASE_FACETS, (0, 1, 2), new_vertex=6
)

MOBIUS_BAND_ALTERNATIVE_FACETS = stellar_subdivide_triangle_facets(
    _MOBIUS_BAND_BASE_FACETS, (0, 1, 2), new_vertex=5
)

KLEIN_BOTTLE_ALTERNATIVE_FACETS = stellar_subdivide_triangle_facets(
    _KLEIN_BOTTLE_BASE_FACETS, (2, 6, 7), new_vertex=8
)


SPHERE_S2_FACETS = (
    (0, 1, 2),
    (0, 2, 3),
    (0, 3, 4),
    (0, 4, 1),
    (5, 1, 2),
    (5, 2, 3),
    (5, 3, 4),
    (5, 4, 1),
)


DISK_B2_FACETS = (
    (0, 1, 2),
    (0, 2, 3),
    (0, 3, 4),
    (0, 4, 5),
    (0, 5, 6),
    (0, 6, 1),
)


BALL_B3_FACETS = tuple((6,) + facet for facet in SPHERE_S2_FACETS)


def _complex_from_facets(facets):
    """Build a complex from maximal facets; ``add_simplex`` adds all faces."""
    complex_ = SimplicialComplex()
    for facet in facets:
        complex_.add_simplex(list(facet))
    return complex_


def torus_alternative():
    """Return an 8-vertex, 16-triangle triangulation of the torus."""
    return _complex_from_facets(TORUS_ALTERNATIVE_FACETS)


def projective_plane_alternative():
    """Return a 7-vertex, 12-triangle triangulation of RP^2."""
    return _complex_from_facets(PROJECTIVE_PLANE_ALTERNATIVE_FACETS)


def mobius_band_alternative():
    """Return a 6-vertex, 7-triangle triangulation of the Mobius band."""
    return _complex_from_facets(MOBIUS_BAND_ALTERNATIVE_FACETS)


def mobius_strip_alternative():
    """Alias for :func:`mobius_band_alternative`."""
    return mobius_band_alternative()


def klein_bottle_alternative():
    """Return a 9-vertex, 18-triangle triangulation of the Klein bottle."""
    return _complex_from_facets(KLEIN_BOTTLE_ALTERNATIVE_FACETS)


def sphere_s2():
    """Return the octahedral triangulation of the 2-sphere S^2."""
    return _complex_from_facets(SPHERE_S2_FACETS)


def sphere():
    """Alias for :func:`sphere_s2` (the surface of a ball)."""
    return sphere_s2()


def disk_b2():
    """Return a six-sector triangulation of the 2-ball B^2."""
    return _complex_from_facets(DISK_B2_FACETS)


def ball_b3():
    """Return the cone over an octahedral S^2, a triangulated 3-ball B^3."""
    return _complex_from_facets(BALL_B3_FACETS)


def three_ball():
    """Alias for :func:`ball_b3`."""
    return ball_b3()


def _component_count(adjacency):
    seen = set()
    count = 0
    for start in adjacency:
        if start in seen:
            continue
        count += 1
        stack = [start]
        seen.add(start)
        while stack:
            current = stack.pop()
            for neighbour in adjacency[current]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
    return count


def _edge_direction(oriented_triangle, canonical_edge):
    a, b, c = oriented_triangle
    positive = {(a, b), (b, c), (c, a)}
    return 1 if canonical_edge in positive else -1


def _is_orientable(triangles, edge_to_triangles):
    oriented = [tuple(sorted(triangle)) for triangle in triangles]
    constraints = defaultdict(list)

    for edge, incident in edge_to_triangles.items():
        if len(incident) != 2:
            continue
        first, second = incident
        canonical_edge = tuple(sorted(edge))
        first_direction = _edge_direction(oriented[first], canonical_edge)
        second_direction = _edge_direction(oriented[second], canonical_edge)
        factor = -first_direction * second_direction
        constraints[first].append((second, factor))
        constraints[second].append((first, factor))

    signs = {}
    for start in range(len(triangles)):
        if start in signs:
            continue
        signs[start] = 1
        stack = [start]
        while stack:
            current = stack.pop()
            for neighbour, factor in constraints[current]:
                required = signs[current] * factor
                if neighbour in signs:
                    if signs[neighbour] != required:
                        return False
                else:
                    signs[neighbour] = required
                    stack.append(neighbour)
    return True


def surface_report_from_facets(facets):
    triangles = [frozenset(facet) for facet in facets]
    if not triangles or any(len(triangle) != 3 for triangle in triangles):
        raise ValueError("a surface facet list must contain only triangles")
    if len(set(triangles)) != len(triangles):
        raise ValueError("the facet list contains duplicate triangles")

    vertices = set().union(*triangles)
    edges = {
        frozenset(edge)
        for triangle in triangles
        for edge in combinations(triangle, 2)
    }

    edge_to_triangles = defaultdict(list)
    for index, triangle in enumerate(triangles):
        for edge in combinations(triangle, 2):
            edge_to_triangles[frozenset(edge)].append(index)

    valid_edge_incidence = all(
        len(incident) in (1, 2) for incident in edge_to_triangles.values()
    )
    boundary_edges = {
        edge
        for edge, incident in edge_to_triangles.items()
        if len(incident) == 1
    }
    boundary_vertices = set().union(*boundary_edges) if boundary_edges else set()

    graph = {vertex: set() for vertex in vertices}
    for edge in edges:
        u, v = tuple(edge)
        graph[u].add(v)
        graph[v].add(u)
    connected = _component_count(graph) == 1

    links_valid = True
    for vertex in vertices:
        link = defaultdict(set)
        for triangle in triangles:
            if vertex not in triangle:
                continue
            u, v = tuple(triangle - {vertex})
            link[u].add(v)
            link[v].add(u)

        link_connected = bool(link) and _component_count(link) == 1
        degrees = [len(neighbours) for neighbours in link.values()]
        if vertex in boundary_vertices:
            correct_type = (
                link_connected
                and degrees.count(1) == 2
                and all(degree in (1, 2) for degree in degrees)
            )
        else:
            correct_type = link_connected and all(
                degree == 2 for degree in degrees
            )
        links_valid = links_valid and correct_type

    boundary_graph = defaultdict(set)
    for edge in boundary_edges:
        u, v = tuple(edge)
        boundary_graph[u].add(v)
        boundary_graph[v].add(u)
    boundary_is_cycles = all(
        len(neighbours) == 2 for neighbours in boundary_graph.values()
    )
    boundary_components = (
        _component_count(boundary_graph) if boundary_graph else 0
    )

    is_surface = (
        connected
        and valid_edge_incidence
        and links_valid
        and boundary_is_cycles
    )
    return {
        "vertices": len(vertices),
        "edges": len(edges),
        "triangles": len(triangles),
        "euler_characteristic": len(vertices) - len(edges) + len(triangles),
        "boundary_edges": len(boundary_edges),
        "boundary_components": boundary_components,
        "orientable": (
            _is_orientable(triangles, edge_to_triangles)
            if is_surface
            else None
        ),
        "is_surface": is_surface,
    }


EXPECTED_SURFACE_DATA = {
    "torus": (
        TORUS_ALTERNATIVE_FACETS,
        (8, 24, 16, 0, 0, True),
    ),
    "projective_plane": (
        PROJECTIVE_PLANE_ALTERNATIVE_FACETS,
        (7, 18, 12, 1, 0, False),
    ),
    "mobius_band": (
        MOBIUS_BAND_ALTERNATIVE_FACETS,
        (6, 13, 7, 0, 1, False),
    ),
    "klein_bottle": (
        KLEIN_BOTTLE_ALTERNATIVE_FACETS,
        (9, 27, 18, 0, 0, False),
    ),
    "sphere_s2": (
        SPHERE_S2_FACETS,
        (6, 12, 8, 2, 0, True),
    ),
    "disk_b2": (
        DISK_B2_FACETS,
        (7, 12, 6, 1, 1, True),
    ),
}


def ball_b3_report():
    tetrahedra = [frozenset(facet) for facet in BALL_B3_FACETS]
    faces_by_size = {
        size: {
            frozenset(face)
            for tetrahedron in tetrahedra
            for face in combinations(tetrahedron, size)
        }
        for size in range(1, 5)
    }

    triangle_incidence = defaultdict(int)
    for tetrahedron in tetrahedra:
        for triangle in combinations(tetrahedron, 3):
            triangle_incidence[frozenset(triangle)] += 1

    boundary = {
        triangle
        for triangle, incidence in triangle_incidence.items()
        if incidence == 1
    }
    valid_incidence = all(
        incidence in (1, 2) for incidence in triangle_incidence.values()
    )
    boundary_is_octahedral_sphere = boundary == {
        frozenset(facet) for facet in SPHERE_S2_FACETS
    }

    f_vector = tuple(len(faces_by_size[size]) for size in range(1, 5))
    euler_characteristic = sum(
        (-1) ** (dimension) * count
        for dimension, count in enumerate(f_vector)
    )
    return {
        "f_vector": f_vector,
        "tetrahedra": len(tetrahedra),
        "boundary_triangles": len(boundary),
        "valid_triangle_incidence": valid_incidence,
        "boundary_is_octahedral_sphere": boundary_is_octahedral_sphere,
        "euler_characteristic": euler_characteristic,
        "is_verified_ball": (
            valid_incidence
            and boundary_is_octahedral_sphere
            and euler_characteristic == 1
        ),
    }


def verify_all_triangulations():
    reports = {}
    keys = (
        "vertices",
        "edges",
        "triangles",
        "euler_characteristic",
        "boundary_components",
        "orientable",
    )

    for name, (facets, expected_values) in EXPECTED_SURFACE_DATA.items():
        report = surface_report_from_facets(facets)
        if not report["is_surface"]:
            raise AssertionError(f"{name} failed the 2-manifold checks: {report}")
        actual_values = tuple(report[key] for key in keys)
        if actual_values != expected_values:
            raise AssertionError(
                f"{name}: expected {expected_values}, got {actual_values}"
            )
        reports[name] = report

    ball_report = ball_b3_report()
    if not ball_report["is_verified_ball"]:
        raise AssertionError(f"ball_b3 failed verification: {ball_report}")
    reports["ball_b3"] = ball_report
    return reports


__all__ = [
    "TORUS_ALTERNATIVE_FACETS",
    "PROJECTIVE_PLANE_ALTERNATIVE_FACETS",
    "MOBIUS_BAND_ALTERNATIVE_FACETS",
    "KLEIN_BOTTLE_ALTERNATIVE_FACETS",
    "SPHERE_S2_FACETS",
    "DISK_B2_FACETS",
    "BALL_B3_FACETS",
    "stellar_subdivide_triangle_facets",
    "torus_alternative",
    "projective_plane_alternative",
    "mobius_band_alternative",
    "mobius_strip_alternative",
    "klein_bottle_alternative",
    "sphere_s2",
    "sphere",
    "disk_b2",
    "ball_b3",
    "three_ball",
    "surface_report_from_facets",
    "ball_b3_report",
    "verify_all_triangulations",
]


if __name__ == "__main__":
    from pprint import pprint

    pprint(verify_all_triangulations(), sort_dicts=False)