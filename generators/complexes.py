
from simplicial_topology.Simplicial_Complex import SimplicialComplex


def vertex():
    """
    Return a single vertex.
    """
    K = SimplicialComplex()
    K.add_simplex([0])
    return K


def edge():
    """
    Return a single edge.
    """
    K = SimplicialComplex()
    K.add_simplex([0, 1])
    return K


def filled_triangle():
    """
    Return a filled 2-simplex.
    """
    K = SimplicialComplex()
    K.add_simplex([0, 1, 2])
    return K


def triangle_boundary():
    """
    Return the boundary of a triangle, i.e. the cycle C3.
    """
    K = SimplicialComplex()
    K.add_simplex([0, 1])
    K.add_simplex([1, 2])
    K.add_simplex([2, 0])
    return K


def two_triangles_sharing_edge():
    """
    Return two filled triangles sharing one common edge.
    """
    K = SimplicialComplex()
    K.add_simplex([0, 1, 2])
    K.add_simplex([0, 1, 3])
    return K


def two_triangles_sharing_vertex():
    """
    Return two filled triangles sharing exactly one vertex.
    """
    K = SimplicialComplex()
    K.add_simplex([0, 1, 2])
    K.add_simplex([0, 3, 4])
    return K


def tetrahedron_boundary():
    """
    Return the boundary of a tetrahedron.

    This is a triangulation of the 2-sphere S^2.
    """
    K = SimplicialComplex()
    K.add_simplex([0, 1, 2])
    K.add_simplex([0, 1, 3])
    K.add_simplex([0, 2, 3])
    K.add_simplex([1, 2, 3])
    return K


def filled_tetrahedron():
    """
    Return the full 3-simplex.

    This is included for completeness, although most current algorithms
    in the project are focused on dimensions 1 and 2.
    """
    K = SimplicialComplex()
    K.add_simplex([0, 1, 2, 3])
    return K


def square_with_diagonal():
    """
    Return a triangulated square made of two filled triangles.
    """
    K = SimplicialComplex()
    K.add_simplex([0, 1, 2])
    K.add_simplex([0, 2, 3])
    return K


def triangulated_disk():
    """
    Return a small triangulated disk.

    The complex consists of four triangles around a central vertex.
    """
    K = SimplicialComplex()
    K.add_simplex([0, 1, 2])
    K.add_simplex([0, 2, 3])
    K.add_simplex([0, 3, 4])
    K.add_simplex([0, 4, 1])
    return K


def annulus():
    """
    Return a small triangulated annulus.

    Vertices 0,1,2 form the inner boundary.
    Vertices 3,4,5 form the outer boundary.
    """
    K = SimplicialComplex()
    K.add_simplex([0, 1, 3])
    K.add_simplex([1, 3, 4])
    K.add_simplex([1, 2, 4])
    K.add_simplex([2, 4, 5])
    K.add_simplex([2, 0, 5])
    K.add_simplex([0, 5, 3])
    return K


def dunce_hat_like():
    """
    Return a small dunce-hat-like 2-dimensional complex.

    This is not a canonical minimal triangulation.
    It is mainly useful as a nontrivial 2D test object.
    """
    K = SimplicialComplex()
    K.add_simplex([0, 1, 2])
    K.add_simplex([0, 2, 3])
    K.add_simplex([0, 3, 4])
    K.add_simplex([0, 4, 1])
    K.add_simplex([1, 2, 5])
    K.add_simplex([2, 3, 5])
    K.add_simplex([3, 4, 5])
    K.add_simplex([4, 1, 5])
    K.add_simplex([0, 5])
    return K


def projective_plane_minimal():
    """
    Return a 6-vertex triangulation of the real projective plane.

    This is a standard minimal triangulation with 10 triangular faces.
    """
    K = SimplicialComplex()
    faces = [
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 4],
        [0, 3, 5],
        [0, 4, 5],
        [1, 2, 5],
        [1, 3, 4],
        [1, 4, 5],
        [2, 3, 4],
        [2, 3, 5],
    ]
    for face in faces:
        K.add_simplex(face)
    return K


def torus_minimal():
    """
    Return a 7-vertex triangulation of the torus.

    This is a classical small triangulation of the 2-torus.
    """
    K = SimplicialComplex()
    faces = [
        [0, 1, 3],
        [0, 1, 4],
        [0, 2, 3],
        [0, 2, 6],
        [0, 4, 5],
        [0, 5, 6],
        [1, 2, 4],
        [1, 2, 5],
        [1, 3, 6],
        [1, 5, 6],
        [2, 3, 5],
        [2, 4, 6],
        [3, 4, 5],
        [3, 4, 6],
    ]
    for face in faces:
        K.add_simplex(face)
    return K


def wedge_of_two_circles():
    """
    Return a 1-dimensional complex homotopy equivalent to S^1 wedge S^1.
    """
    K = SimplicialComplex()
    K.add_simplex([0, 1])
    K.add_simplex([1, 2])
    K.add_simplex([2, 0])
    K.add_simplex([0, 3])
    K.add_simplex([3, 4])
    K.add_simplex([4, 0])
    return K


def cone_over_complex(L, apex=None):
    """
    Return the cone over a simplicial complex L.

    The cone over any complex is strongly collapsible.
    """
    if apex is None:
        apex = max(L.vertices) + 1 if L.vertices else 0

    K = SimplicialComplex(L.simplicial_complex)
    for simplex in L.simplicial_complex:
        K.add_simplex(list(simplex | frozenset([apex])))

    K.add_simplex([apex])
    return K


def suspension_over_complex(L, apex1=None, apex2=None):
    """
    Return the suspension over a simplicial complex L.
    """
    if apex1 is None:
        apex1 = max(L.vertices) + 1 if L.vertices else 0

    if apex2 is None:
        apex2 = apex1 + 1

    K = SimplicialComplex(L.simplicial_complex)
    for simplex in L.simplicial_complex:
        K.add_simplex(list(simplex | frozenset([apex1])))
        K.add_simplex(list(simplex | frozenset([apex2])))

    K.add_simplex([apex1])
    K.add_simplex([apex2])
    return K
