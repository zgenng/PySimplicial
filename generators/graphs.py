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

    Parameters
    ----------
    n : int
        Total number of vertices.
    """
    K = SimplicialComplex()

    for i in range(1, n):
        K.add_simplex([0, i])

    return K


def wheel_graph(n):
    """
    Return the wheel graph W_n.

    A wheel graph consists of a cycle on n - 1 vertices together
    with one central vertex connected to every vertex of the cycle.

    Parameters
    ----------
    n : int
        Total number of vertices. Must be at least 4.
    """
    if n < 4:
        raise ValueError("A wheel graph W_n must have n >= 4 vertices.")

    K = cycle_graph(n - 1)

    center = n - 1

    for i in range(n - 1):
        K.add_simplex([center, i])

    return K


def complete_bipartite_graph(m, n):
    """
    Return the complete bipartite graph K_{m,n}.

    The vertex set is divided into two parts:

        A = {0, 1, ..., m - 1}
        B = {m, m + 1, ..., m + n - 1}

    Every vertex from A is connected to every vertex from B.
    There are no edges inside A or inside B.

    Examples
    --------
    complete_bipartite_graph(3, 3) returns K_{3,3}.
    complete_bipartite_graph(4, 4) returns K_{4,4}.
    """
    if m <= 0 or n <= 0:
        raise ValueError("Both parts must have positive size.")

    K = SimplicialComplex()

    left_part = range(m)
    right_part = range(m, m + n)

    for u in left_part:
        for v in right_part:
            K.add_simplex([u, v])

    return K


def complete_multipartite_graph(*parts):
    """
    Return the complete multipartite graph K_{n1,n2,...,nk}.

    The vertex set is divided into several independent parts.
    Every vertex is connected to all vertices from different parts.
    No two vertices inside the same part are connected.

    Parameters
    ----------
    *parts : int
        Sizes of the parts.

    Examples
    --------
    complete_multipartite_graph(2, 3)
        returns K_{2,3}, which is a complete bipartite graph.

    complete_multipartite_graph(2, 2, 2)
        returns K_{2,2,2}.

    complete_multipartite_graph(1, 1, 1, 1)
        returns K_4.
    """
    if len(parts) < 2:
        raise ValueError("A multipartite graph must have at least two parts.")

    if any(size <= 0 for size in parts):
        raise ValueError("All parts must have positive size.")

    K = SimplicialComplex()

    vertex_parts = []
    current_vertex = 0

    for size in parts:
        part = list(range(current_vertex, current_vertex + size))
        vertex_parts.append(part)
        current_vertex += size

    for i in range(len(vertex_parts)):
        for j in range(i + 1, len(vertex_parts)):
            for u in vertex_parts[i]:
                for v in vertex_parts[j]:
                    K.add_simplex([u, v])

    return K