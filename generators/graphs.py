from simplicial_topology.Simplicial_Complex import SimplicialComplex


def complete_graph(n):
    """
    Return the complete graph K_n.

    Every pair of distinct vertices is connected by an edge.
    """
    K = SimplicialComplex()

    for i in range(n):
        for j in range(i + 1, n):
            K.add_simplex([i, j])

    return K


def cycle_graph(n):
    """
    Return the cycle graph C_n.

    Vertices are arranged in a cycle, and each vertex is connected
    to its two neighbors.
    """
    K = SimplicialComplex()

    for i in range(n):
        K.add_simplex([i, (i + 1) % n])

    return K


def path_graph(n):
    """
    Return the path graph P_n.

    Vertices are connected in a single chain.
    """
    K = SimplicialComplex()

    for i in range(n - 1):
        K.add_simplex([i, i + 1])

    return K


def star_graph(n):
    """
    Return the star graph S_n.

    Vertex 0 is the center and is connected to all other vertices.
    """
    K = SimplicialComplex()

    for i in range(1, n):
        K.add_simplex([0, i])

    return K


def wheel_graph(n):
    """
    Return the wheel graph W_n.

    A wheel graph consists of a cycle on n-1 vertices together
    with one central vertex connected to every vertex of the cycle.
    """
    K = cycle_graph(n - 1)

    center = n - 1

    for i in range(n - 1):
        K.add_simplex([center, i])

    return K