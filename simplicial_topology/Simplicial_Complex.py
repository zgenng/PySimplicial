from itertools import combinations
import copy
import math
import random
import numpy as np

class SimplicialComplex:
    def __init__(self, simplicial_complex=None):
        self.simplicial_complex = set()
        self.vertices = set()

        if simplicial_complex:
            for s in simplicial_complex:
                self.add_simplex(list(s))

    def get_simplicial_complex(self):
        return list(self.simplicial_complex)

    def update_vertices(self):
        self.vertices = set()
        for s in self.simplicial_complex:
            self.vertices.update(s)

    def add_simplex(self, simplex):
        s = frozenset(simplex)
        for r in range(1, len(s) + 1):
            for face in combinations(s, r):
                self.simplicial_complex.add(frozenset(face))
        self.vertices.update(s)

    def one_skeleton(self):
        result = SimplicialComplex()

        for simplex in self.simplicial_complex:
            if len(simplex) == 1:
                result.add_simplex(list(simplex))

            elif len(simplex) >= 2:
                for edge in combinations(simplex, 2):
                    result.add_simplex(list(edge))

        return result

    def maximal_simplices(self):
        simplices = list(self.simplicial_complex)
        maximal = set()
        for s in simplices:
            if not any(s < t for t in simplices if s != t):
                maximal.add(s)
        return maximal

    def dimension(self):
        if not self.simplicial_complex:
            return -1
        return max(len(s) for s in self.simplicial_complex) - 1

    def is_dominated(self, v, u, max_simplices_by_vertex):
        if v == u:
            return False
        v_maximals = max_simplices_by_vertex.get(v, set())
        if not v_maximals:
            return False
        return all(u in s for s in v_maximals)

    def strong_collapse(self):
        changed = True
        while changed:
            changed = False
            maximals = self.maximal_simplices()

            max_by_vertex = {}
            for s in maximals:
                for v in s:
                    max_by_vertex.setdefault(v, set()).add(s)

            for v in list(self.vertices):
                for u in list(self.vertices):
                    if self.is_dominated(v, u, max_by_vertex):
                        self.simplicial_complex = {
                            s for s in self.simplicial_complex if v not in s
                        }
                        self.vertices.discard(v)
                        changed = True
                        break
                if changed:
                    break

    def core(self):
        k = copy.deepcopy(self)
        k.strong_collapse()
        return k

    def is_strongly_collapsible(self):
        c = self.core()
        return len(c.vertices) <= 1

    def join(self, other):
        result = SimplicialComplex()
        simplices1 = list(self.simplicial_complex) + [frozenset()]
        simplices2 = list(other.simplicial_complex) + [frozenset()]
        for sigma in simplices1:
            for tau in simplices2:
                if sigma or tau:
                    result.add_simplex(list(sigma | tau))
        return result

    def is_join(self):
        self.update_vertices()
        V = list(self.vertices)
        n = len(V)
        if n < 2:
            return False, None, None, None

        A, B = set(), set()

        def valid_partition(A, B):
            if not A or not B:
                return False
            K1 = SimplicialComplex(
                [list(s) for s in self.simplicial_complex if s.issubset(A)]
            )
            K2 = SimplicialComplex(
                [list(s) for s in self.simplicial_complex if s.issubset(B)]
            )
            J = K1.join(K2)
            return J.simplicial_complex == self.simplicial_complex

        def dfs(i):
            if i == n:
                if valid_partition(A, B):
                    return True, set(A), set(B)
                return False, None, None
            v = V[i]
            A.add(v)
            ok, a, b = dfs(i + 1)
            if ok:
                return True, a, b
            A.remove(v)
            B.add(v)
            ok, a, b = dfs(i + 1)
            if ok:
                return True, a, b
            B.remove(v)
            return False, None, None

        ok, A, B = dfs(0)
        if not ok:
            return False, None, None, None

        K1 = SimplicialComplex(
            [list(s) for s in self.simplicial_complex if s.issubset(A)]
        )
        K2 = SimplicialComplex(
            [list(s) for s in self.simplicial_complex if s.issubset(B)]
        )
        return True, K1, K2, (A, B)

    def graph_join(self, other):
        if self.dimension() > 1 or other.dimension() > 1:
            raise ValueError(
                "graph_join requires both complexes to be graphs "
                f"(dimension <= 1), but received dim(self)={self.dimension()}, "
                f"dim(other)={other.dimension()}"
            )

        result = SimplicialComplex()

        # vertices and edges of self
        for v in self.vertices:
            result.add_simplex([v])
        for s in self.simplicial_complex:
            if len(s) == 2:
                result.add_simplex(list(s))

        # vertices and edges of other
        for v in other.vertices:
            result.add_simplex([v])
        for s in other.simplicial_complex:
            if len(s) == 2:
                result.add_simplex(list(s))

        # all edges between V(G) and V(H)
        for v in self.vertices:
            for w in other.vertices:
                result.add_simplex([v, w])

        return result

    def is_graph_join(self):
        """
        Determine whether a graph is a nontrivial join G = G1 + G2.

        If yes, return two largest/balanced induced graph factors.

        The method is only for graphs, i.e. simplicial complexes of dimension <= 1.
        If dimension > 1, it raises ValueError.

        Returns
        -------
        (bool, G1, G2, partition)
            If G is a join, returns:

                True, G1, G2, (A, B)

            where A and B are non-empty vertex sets and G = G[A] + G[B].

            The partition (A, B) is chosen to maximize min(|A|, |B|),
            so it avoids trivial decompositions like one vertex + everything else
            when a larger decomposition is possible.

            If G is not a join, returns:

                False, None, None, None
        """
        if self.dimension() > 1:
            raise ValueError(
                "is_graph_join_largest requires a graph, i.e. dimension <= 1, "
                f"but received dimension {self.dimension()}."
            )

        self.update_vertices()
        vertices = list(self.vertices)

        if len(vertices) < 2:
            return False, None, None, None

        edges = {frozenset(e) for e in _edges(self)}

        # Build complement graph.
        complement_adj = {v: set() for v in vertices}

        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                u = vertices[i]
                v = vertices[j]

                if frozenset([u, v]) not in edges:
                    complement_adj[u].add(v)
                    complement_adj[v].add(u)

        # Connected components of complement graph.
        complement_components = []
        seen = set()

        for start in vertices:
            if start in seen:
                continue

            component = set()
            stack = [start]
            seen.add(start)

            while stack:
                v = stack.pop()
                component.add(v)

                for u in complement_adj[v]:
                    if u not in seen:
                        seen.add(u)
                        stack.append(u)

            complement_components.append(component)

        # G is a join iff complement(G) is disconnected.
        if len(complement_components) < 2:
            return False, None, None, None

        # Choose a balanced split of complement components.
        # Any union of complement-components gives a valid join factor.
        n = len(vertices)
        target = n // 2

        # DP subset sum over component sizes.
        dp = {0: []}

        for idx, comp in enumerate(complement_components):
            size = len(comp)
            new_dp = dict(dp)

            for current_size, chosen_indices in dp.items():
                new_size = current_size + size

                if new_size <= target and new_size not in new_dp:
                    new_dp[new_size] = chosen_indices + [idx]

            dp = new_dp

        best_size = max(dp.keys())
        chosen = set(dp[best_size])

        A = set()
        B = set()

        for idx, comp in enumerate(complement_components):
            if idx in chosen:
                A.update(comp)
            else:
                B.update(comp)

        if not A or not B:
            return False, None, None, None

        G1 = self.induced_graph_subcomplex(A)
        G2 = self.induced_graph_subcomplex(B)

        return True, G1, G2, (A, B)

    def induced_graph_subcomplex(self, vertex_subset):
        """
        Return the induced graph subcomplex on vertex_subset.

        This keeps all vertices in vertex_subset and all edges of self whose
        endpoints both belong to vertex_subset.
        """
        if self.dimension() > 1:
            raise ValueError(
                "induced_graph_subcomplex requires dimension <= 1, "
                f"but received dimension {self.dimension()}."
            )

        vertex_subset = set(vertex_subset)

        result = SimplicialComplex()

        for v in vertex_subset:
            result.add_simplex([v])

        for a, b in _edges(self):
            if a in vertex_subset and b in vertex_subset:
                result.add_simplex([a, b])

        return result

    def relabeled(self, mapping):
        """
        Return a relabeled copy of the simplicial complex.

        Parameters
        ----------
        mapping : dict
            Dictionary v -> new_v. Vertices not contained in mapping remain unchanged.
        """
        result = SimplicialComplex()

        for simplex in self.simplicial_complex:
            new_simplex = [
                mapping.get(v, v)
                for v in simplex
            ]
            result.add_simplex(new_simplex)

        return result

    def wedge(
        self,
        other,
        self_vertex=None,
        other_vertex=None,
        relabel_other=True,
        return_mapping=False,
    ):
        """
        Return the wedge sum of two simplicial complexes.

        The wedge K ∨ L is obtained by identifying one vertex of K
        with one vertex of L.

        Unlike join(), this operation does not add mixed simplices.
        It only takes the union of the two complexes after gluing
        one chosen vertex.

        Parameters
        ----------
        other : SimplicialComplex
            The second simplicial complex.

        self_vertex : hashable or None
            Vertex of self used for gluing. If None, a default vertex is chosen.

        other_vertex : hashable or None
            Vertex of other used for gluing. If None, a default vertex is chosen.

        relabel_other : bool
            If True, vertices of other are automatically relabeled to avoid
            accidental collisions with vertices of self. Only other_vertex is
            intentionally identified with self_vertex.

            If False, an error is raised when other has vertex labels that
            collide with self, except for the glued vertex.

        return_mapping : bool
            If True, return (result, mapping), where mapping tells how vertices
            of other were relabeled.
        """
        self.update_vertices()
        other.update_vertices()

        if not self.vertices:
            raise ValueError("Cannot take wedge: self has no vertices.")

        if not other.vertices:
            raise ValueError("Cannot take wedge: other has no vertices.")

        if self_vertex is None:
            self_vertex = _choose_default_vertex(self.vertices)

        if other_vertex is None:
            other_vertex = _choose_default_vertex(other.vertices)

        if self_vertex not in self.vertices:
            raise ValueError(
                f"self_vertex={self_vertex} is not a vertex of self."
            )

        if other_vertex not in other.vertices:
            raise ValueError(
                f"other_vertex={other_vertex} is not a vertex of other."
            )

        result = SimplicialComplex()

        for simplex in self.simplicial_complex:
            result.add_simplex(list(simplex))

        used = set(result.vertices)

        mapping = {
            other_vertex: self_vertex
        }

        other_remaining = set(other.vertices) - {other_vertex}

        if not relabel_other:
            overlap = used & other_remaining

            if overlap:
                raise ValueError(
                    "Vertex labels collide between self and other. "
                    f"Overlapping vertices: {overlap}. "
                    "Use relabel_other=True to relabel other automatically."
                )

            for v in other_remaining:
                mapping[v] = v

        else:
            preserved = {
                v for v in other_remaining
                if v not in used
            }

            reserved = set(used) | set(preserved)

            for v in preserved:
                mapping[v] = v

            colliding = other_remaining - preserved

            for v in colliding:
                new_v = _fresh_vertex_label(reserved, v)
                mapping[v] = new_v
                reserved.add(new_v)

        for simplex in other.simplicial_complex:
            new_simplex = [
                mapping[v]
                for v in simplex
            ]
            result.add_simplex(new_simplex)

        if return_mapping:
            return result, mapping

        return result

    def gscat(
        self,
        for_graph=None,
        method="nash_williams",
        *,
        for_graphs=None,
        edge_disjoint=True,
    ):
        """
        Parameters
        ----------
        for_graph : bool or None
            See original behaviour: forces / autodetects the graph-specific
            branch for 1-dimensional complexes.

        for_graphs : bool or None
            Backward-compatible alias for ``for_graph``.  Thus both
            ``gscat(for_graph=True)`` and ``gscat(for_graphs=True)`` are
            accepted.  Supplying contradictory values raises ``ValueError``.

        method : {"nash_williams", "matroid_partition"}
            Only relevant when the graph branch (_gscat_graph) is used
            (for_graph=True, or for_graph=None with dimension <= 1).

            "nash_williams" (default): keeps the original approach —
            arboricity computed via the Nash-Williams subset formula
            (exponential in |V| per connected component, vectorized with
            NumPy), followed by a randomized greedy / backtracking forest
            cover.

            "matroid_partition": exact arboricity AND forest decomposition
            computed in one pass via Edmonds' matroid partitioning
            algorithm (augmenting-path exchange search), polynomial in
            the number of edges. Recommended for larger graphs where the
            Nash-Williams subset enumeration becomes impractical.

        edge_disjoint : bool
            Only relevant for graphs.  If True (default), every edge of the
            graph belongs to exactly one returned cover element.  Different
            elements may still meet at vertices, as is unavoidable in a
            connected graph.

            A minimum gscat-cover cannot always be chosen edge-disjoint.  In
            that case the first returned value is still the exact gscat, while
            the returned edge-disjoint cover contains more than gscat + 1
            strongly collapsible subcomplexes.  Pass ``edge_disjoint=False``
            to obtain a minimum cover; that cover may repeat connecting edges.
        """
        if for_graphs is not None:
            if for_graph is not None and for_graph != for_graphs:
                raise ValueError(
                    "for_graph and for_graphs were given contradictory values"
                )
            for_graph = for_graphs

        # For graphs, the cover must be constructed on the original graph, not on its core.
        # Otherwise a tree would have the correct value gscat=0, but an empty cover,
        # and a graph with pendant edges would lose those edges in the returned cover.
        if for_graph is True:
            if self.dimension() > 1:
                raise ValueError(
                    "for_graph=True requires a 1-dimensional complex (graph), "
                    f"but dimension is {self.dimension()}"
                )
            return _gscat_graph(
                self,
                method=method,
                edge_disjoint=edge_disjoint,
            )

        if for_graph is None and self.dimension() <= 1:
            return _gscat_graph(
                self,
                method=method,
                edge_disjoint=edge_disjoint,
            )

        # For general simplicial complexes we may work with the core since gscat
        # is preserved under strong collapse. However, the resulting cover belongs to the core.
        # If one needs a cover of the original 2-dimensional complex, a separate lifting procedure is required.
        if len(copy.deepcopy(self).vertices) <= 1:
            return 0, [self.simplicial_complex]

        if for_graph is False:
            return _gscat_general_naive(copy.deepcopy(self))

        return _gscat_general(copy.deepcopy(self))


def _gscat_general_naive(k0):
    from itertools import combinations as _comb
    from more_itertools import powerset as _powerset

    simplices = list(k0.simplicial_complex)
    all_subsets = list(_powerset(simplices))[1:]

    covers = []
    seen = set()

    for subset in all_subsets:
        t = SimplicialComplex()
        for s in subset:
            t.add_simplex(list(s))

        t_core = t.core()

        if len(t_core.simplicial_complex) == 1:
            s = next(iter(t_core.simplicial_complex))
            if len(s) == 1:
                key = tuple(sorted(tuple(sorted(x)) for x in t.simplicial_complex))
                if key not in seen:
                    seen.add(key)
                    covers.append(t.simplicial_complex)

    target = k0.simplicial_complex
    for i in range(1, len(covers) + 1):
        for u in _comb(covers, i):
            if set().union(*u) == target:
                return i - 1, u

    return None, []


def _edges(k):
    """Return all edges of the 1-skeleton of the complex."""
    return {tuple(sorted(s)) for s in k.simplicial_complex if len(s) == 2}


def _make_subcomplex_from_edges(edges, vertices=None):
    """
    Construct a 1-dimensional subcomplex from a collection of edges.

    The optional ``vertices`` argument is used to include isolated vertices
    that must also belong to the cover element.
    """
    sc = SimplicialComplex()
    if vertices is not None:
        for v in vertices:
            sc.add_simplex([v])
    for a, b in edges:
        sc.add_simplex([a, b])
    return sc


def _is_forest(vertices, edges):
    """Check whether the given edge set is acyclic, i.e. whether it is a forest."""
    parent = {v: v for v in vertices}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        if a == b:
            return False
        if a not in parent or b not in parent:
            return False
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        parent[ra] = rb
    return True


def _is_connected(vertices, edges):
    """Check whether the graph is connected on the given vertex set."""
    vertices = list(vertices)
    if len(vertices) <= 1:
        return True

    adj = {v: set() for v in vertices}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)

    seen = {vertices[0]}
    stack = [vertices[0]]
    while stack:
        v = stack.pop()
        for u in adj[v]:
            if u not in seen:
                seen.add(u)
                stack.append(u)
    return len(seen) == len(vertices)


def _connected_components(vertices, edges):
    """Return the connected components of the graph."""
    vertices = list(vertices)
    adj = {v: set() for v in vertices}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)

    components = []
    seen = set()
    for start in vertices:
        if start in seen:
            continue
        comp = set()
        stack = [start]
        seen.add(start)
        while stack:
            v = stack.pop()
            comp.add(v)
            for u in adj[v]:
                if u not in seen:
                    seen.add(u)
                    stack.append(u)
        components.append(comp)
    return components


def _arboricity_nash_williams(vertices, edges):
    """
    Compute graph arboricity using the Nash-Williams formula:

        a(G) = max ceil(|E(H)| / (|V(H)| - 1)), |V(H)| >= 2.

    NumPy optimized version.

    Idea:
    1. Encode every vertex subset by a bitmask.
    2. Put 1 at mask {u,v} for every edge uv.
    3. Use a vectorized subset zeta transform to compute, for every mask S,
       the number of edges completely contained in S.

    Still exponential in |V|, because Nash-Williams requires checking all
    vertex subsets in the exact brute-force form, but the inner counting is
    moved from Python loops to NumPy operations.
    """
    V = list(vertices)
    n = len(V)

    if n < 2 or not edges:
        return 0

    # 2^n memory is unavoidable for this exact subset version.
    # For n > 25 this can already become too large.
    total_masks = 1 << n

    index = {v: i for i, v in enumerate(V)}

    # edge_count[mask] initially stores only real edge masks.
    # After the zeta transform, edge_count[S] = number of induced edges in S.
    edge_count = np.zeros(total_masks, dtype=np.int32)

    for a, b in edges:
        emask = (1 << index[a]) | (1 << index[b])
        edge_count[emask] += 1

    # Subset zeta transform:
    # after processing bit i, every mask containing bit i also receives
    # the value from the same mask without bit i.
    for i in range(n):
        step = 1 << i
        blocks = edge_count.reshape(-1, step * 2)
        blocks[:, step:] += blocks[:, :step]

    # Size of every vertex subset. This part is cheap compared to edge counting.
    # np.fromiter avoids storing Python list of length 2^n first.
    sizes = np.fromiter(
        (mask.bit_count() for mask in range(total_masks)),
        dtype=np.int16,
        count=total_masks,
    )

    valid = sizes >= 2
    if not np.any(valid):
        return 0

    ec = edge_count[valid].astype(np.int64)
    denom = sizes[valid].astype(np.int64) - 1

    # ceil(ec / denom) using integer arithmetic.
    values = (ec + denom - 1) // denom
    return int(values.max(initial=0))

def _cover_by_forests_fast(vertices, edges, k, max_restarts=300, seed=0):
    """
    Decompose E(G) into k forests using a fast randomized greedy algorithm.

    The Nash-Williams formula gives the correct minimum value k = a(G).
    Constructing the actual decomposition is a matroid partition problem.
    This function uses a practical randomized greedy method instead of
    exponential backtracking.

    It is much faster on medium-size examples. If it fails, the caller may
    either retry with more restarts or fall back to the exact backtracking
    method for small graphs.
    """
    vertices = list(vertices)
    edge_list = list(edges)

    if not edge_list:
        return [[] for _ in range(k)]

    idx = {v: i for i, v in enumerate(vertices)}
    rng = random.Random(seed)

    def find(parent, x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def can_add(parent, a, b):
        ia, ib = idx[a], idx[b]
        return find(parent, ia) != find(parent, ib)

    def add_edge(parent, a, b):
        ia, ib = idx[a], idx[b]
        ra, rb = find(parent, ia), find(parent, ib)
        if ra == rb:
            return False
        parent[ra] = rb
        return True

    # Edges with high endpoint degrees are usually harder, so try them earlier.
    degree = {v: 0 for v in vertices}
    for a, b in edge_list:
        degree[a] += 1
        degree[b] += 1

    base_order = sorted(
        edge_list,
        key=lambda e: degree[e[0]] + degree[e[1]],
        reverse=True,
    )

    best_forests = None
    best_component_count = math.inf

    for attempt in range(max_restarts):
        if attempt == 0:
            order = base_order[:]
        else:
            order = base_order[:]
            rng.shuffle(order)

        parents = [list(range(len(vertices))) for _ in range(k)]
        forests = [[] for _ in range(k)]
        active_vertices = [set() for _ in range(k)]

        success = True

        for a, b in order:
            possible = [
                f for f in range(k)
                if can_add(parents[f], a, b)
            ]

            if not possible:
                success = False
                break

            # Prefer a forest already containing the endpoints.  Besides
            # keeping the decomposition valid, this tends to make every part
            # connected and therefore reduces the number of trees needed by
            # an edge-disjoint gscat cover.
            f = min(
                possible,
                key=lambda x: (
                    -int(a in active_vertices[x])
                    -int(b in active_vertices[x]),
                    -len(forests[x]),
                    x,
                ),
            )

            add_edge(parents[f], a, b)
            forests[f].append((a, b))
            active_vertices[f].update((a, b))

        if success:
            if any(not _is_forest(vertices, f) for f in forests):
                continue

            union_edges = set().union(*(set(f) for f in forests)) if forests else set()
            if union_edges == set(edge_list):
                component_count = sum(
                    len(_split_forest_into_trees(forest))
                    for forest in forests
                )

                if component_count < best_component_count:
                    best_forests = [forest[:] for forest in forests]
                    best_component_count = component_count

                # Every one of the k forests is already a tree, so this is an
                # edge-disjoint minimum gscat-cover and cannot be improved.
                if component_count == k and all(forests):
                    return best_forests

    return best_forests


def _cover_by_forests_backtrack(vertices, edges, k):
    """
    Decompose E(G) into k forests.

    This uses rollback via full DSU snapshots rather than partial
    reconstruction. This makes it less likely to keep an incorrect parent
    state after a failed search branch.
    """
    vertices = list(vertices)
    edge_list = sorted(edges)
    idx = {v: i for i, v in enumerate(vertices)}

    parents = [list(range(len(vertices))) for _ in range(k)]
    forests = [[] for _ in range(k)]

    def find(parent, x):
        # No path compression is used, so rollback remains exact and simple.
        while parent[x] != x:
            x = parent[x]
        return x

    def backtrack(i):
        if i == len(edge_list):
            return True

        a, b = edge_list[i]
        ia, ib = idx[a], idx[b]

        for f in range(k):
            ra, rb = find(parents[f], ia), find(parents[f], ib)
            if ra == rb:
                continue

            old_parent = parents[f][:]
            parents[f][ra] = rb
            forests[f].append((a, b))

            if backtrack(i + 1):
                return True

            forests[f].pop()
            parents[f] = old_parent

        return False

    if not backtrack(0):
        return None

    # Final consistency check: every element must be a forest.
    if any(not _is_forest(vertices, f) for f in forests):
        raise RuntimeError("Decomposition error: one of the cover elements is not a forest")

    # Final consistency check for edge coverage.
    union_edges = set().union(*(set(f) for f in forests)) if forests else set()
    if union_edges != set(edge_list):
        raise RuntimeError("Decomposition error: the forests do not cover all graph edges")

    return forests


# ---------------------------------------------------------------------------
# Edmonds' matroid partitioning algorithm (exact, polynomial in |E|),
# specialized for the graphic matroid.
#
# This computes arboricity AND a valid forest decomposition in a single
# incremental pass, as an alternative to the Nash-Williams subset formula
# (which is exponential in |V|). For each edge, it tries to place it into
# an existing forest directly; if that would create a cycle, it searches
# for an augmenting sequence of exchanges (matroid union augmenting path):
# some edge y on the cycle created by adding e may itself be relocated to
# another forest (possibly triggering further exchanges), which frees up
# room for e. Only if no such augmenting sequence exists is a new forest
# opened. This is the standard textbook algorithm for matroid union /
# partitioning (see e.g. Schrijver, "Combinatorial Optimization", chapter
# on matroid union), specialized here to graphic matroids via union-find
# based forest membership and BFS-based fundamental-cycle detection.
# ---------------------------------------------------------------------------

def _fundamental_cycle_edges(forest_edges_i, new_edge, vertices):
    """
    Return the list of edges of forest_edges_i lying on the unique path
    between the endpoints of new_edge (i.e. the fundamental cycle that
    would be created if new_edge were added to this forest).

    Returns None if the endpoints of new_edge lie in different components
    of forest_edges_i, meaning new_edge can be added directly with no
    cycle.
    """
    a, b = new_edge
    if a == b:
        return []

    adj = {}
    for (u, v) in forest_edges_i:
        adj.setdefault(u, []).append((v, (u, v)))
        adj.setdefault(v, []).append((u, (u, v)))

    if a not in adj and a != b:
        if b not in adj:
            return None if a != b else []
    if a not in adj or b not in adj:
        return None

    visited = {a}
    parent_edge = {}
    stack = [a]

    while stack:
        u = stack.pop()
        if u == b:
            break
        for (v, edge) in adj.get(u, []):
            if v not in visited:
                visited.add(v)
                parent_edge[v] = (u, edge)
                stack.append(v)

    if b not in visited:
        return None

    path_edges = []
    cur = b
    while cur != a:
        u, edge = parent_edge[cur]
        path_edges.append(edge)
        cur = u
    return path_edges


def _try_place_matroid(x, forest_edges, vertices, visited=None):
    """
    Try to place edge ``x`` into the existing list of forests by using the
    augmenting-path exchange graph for the graphic matroid partition problem.

    Important fix
    -------------
    The previous recursive implementation returned the first DFS exchange
    chain it found.  Such a chain can be non-shortest and can use the same
    forest several times in a way that makes the final simultaneous exchange
    cyclic.  In other words, every *single* swap looked legal, but the whole
    sequence could still leave a produced part that was not a forest.

    This version uses the standard BFS shortest augmenting path:

    * a node is an edge that currently has to be inserted;
    * from edge ``e`` we draw an exchange arc to every edge ``y`` on the
      fundamental cycle created by adding ``e`` to a forest ``F_i``;
    * if ``e`` can be added to some ``F_i`` without a cycle, we have reached
      the terminal of an augmenting path.

    Returns
    -------
    list[tuple[int, tuple, tuple | None]] or None
        A list of transitions ``(forest_index, edge_to_add, edge_to_remove)``
        in the exact order in which they must be applied by
        ``_apply_matroid_transitions``.  Returns ``None`` if no augmenting path
        exists for the current number of forests, so the caller must open a new
        forest.

    Notes
    -----
    The ``visited`` parameter is kept only for backward compatibility with the
    old call signature.  It is not used by the BFS implementation.
    """
    from collections import deque

    if not forest_edges:
        return None

    predecessor = {x: None}  # edge -> (previous_edge, forest_index)
    queue = deque([x])

    terminal_edge = None
    terminal_forest = None

    while queue:
        current = queue.popleft()

        for i, fe in enumerate(forest_edges):
            cycle = _fundamental_cycle_edges(fe, current, vertices)

            if cycle is None:
                terminal_edge = current
                terminal_forest = i
                queue.clear()
                break

            for y in cycle:
                if y not in predecessor:
                    predecessor[y] = (current, i)
                    queue.append(y)

        if terminal_edge is not None:
            break

    if terminal_edge is None:
        return None

    # Reconstruct the edge path:
    #     x = path[0] -> path[1] -> ... -> path[-1] = terminal_edge
    # labels[j] is the forest where path[j] replaces path[j + 1].
    path = [terminal_edge]
    labels = []
    current = terminal_edge

    while predecessor[current] is not None:
        previous, forest_index = predecessor[current]
        labels.append(forest_index)
        path.append(previous)
        current = previous

    path.reverse()
    labels.reverse()

    # Apply from the terminal backwards:
    # first insert the terminal edge into the forest that accepts it directly;
    # then each previous edge replaces the next edge on the augmenting path.
    transitions = [(terminal_forest, terminal_edge, None)]

    for j in range(len(labels) - 1, -1, -1):
        transitions.append((labels[j], path[j], path[j + 1]))

    return transitions


def _apply_matroid_transitions(forest_edges, transitions):
    """Apply a sequence of (forest_index, edge_to_add, edge_to_remove) swaps."""
    for (i, add_e, remove_e) in transitions:
        if remove_e is not None:
            try:
                forest_edges[i].remove(remove_e)
            except ValueError as exc:
                raise RuntimeError(
                    "Invalid matroid-partition transition: tried to remove "
                    f"edge {remove_e} from forest {i}, but it is not there."
                ) from exc

        if add_e not in forest_edges[i]:
            forest_edges[i].append(add_e)


def _arboricity_and_forests_matroid_partition(vertices, edges):
    """
    Compute the exact arboricity of a graph together with a valid forest
    decomposition, using Edmonds' matroid partitioning algorithm.

    Unlike _arboricity_nash_williams (which is O(V * 2^V) because it must
    enumerate vertex subsets to evaluate the Nash-Williams formula exactly),
    this algorithm is polynomial in the number of edges: each edge is
    either placed directly into an existing forest, or an augmenting
    exchange sequence is found (or, failing that, a new forest is opened).
    In the worst case this is roughly O(E^2 * V) (each of E edges may
    trigger a DFS over up to E prior edges, each fundamental-cycle lookup
    costing O(V + E)), which is still far better than the exponential
    Nash-Williams subset enumeration for large V.

    Returns (k, forests) where k = arboricity(G) - 1 is NOT applied here;
    this function returns k = arboricity(G) itself (the raw number of
    forests used), and forests is a list of k lists of edge tuples whose
    union is exactly `edges`.
    """
    edge_list = list(edges)
    forest_edges = []

    for e in edge_list:
        visited = set()
        result = _try_place_matroid(e, forest_edges, vertices, visited)

        if result is not None:
            _apply_matroid_transitions(forest_edges, result)

            # Catch implementation mistakes immediately.  The next augmenting
            # step assumes every current part is a forest.
            if any(not _is_forest(vertices, fe) for fe in forest_edges):
                raise RuntimeError(
                    "Internal error in matroid partitioning: an augmenting "
                    "exchange produced a cyclic part."
                )
        else:
            forest_edges.append([e])

    # Sanity checks (cheap relative to the algorithm itself, but useful
    # to catch any implementation bug rather than silently returning a
    # wrong decomposition).
    for fe in forest_edges:
        if not _is_forest(vertices, fe):
            raise RuntimeError(
                "Internal error in matroid partitioning: a produced "
                "component is not a forest."
            )

    union_edges = set().union(*(set(fe) for fe in forest_edges)) if forest_edges else set()
    if union_edges != set(edge_list):
        raise RuntimeError(
            "Internal error in matroid partitioning: the decomposition "
            "does not cover all edges."
        )

    return len(forest_edges), forest_edges


def _split_forest_into_trees(forest_edges):
    """
    Split a forest into its non-empty connected components.

    Every returned edge set is a tree.  No edge is added or copied, so a
    partition of the graph edges into forests becomes a partition into trees.
    """
    forest_edges = set(forest_edges)
    if not forest_edges:
        return []

    vertices = {v for edge in forest_edges for v in edge}
    if not _is_forest(vertices, forest_edges):
        raise RuntimeError("Cannot split a cyclic edge set into forest components")

    adjacency = {v: set() for v in vertices}
    for a, b in forest_edges:
        adjacency[a].add(b)
        adjacency[b].add(a)

    unseen = set(vertices)
    trees = []

    while unseen:
        start = min(unseen, key=repr)
        component_vertices = {start}
        stack = [start]
        unseen.remove(start)

        while stack:
            v = stack.pop()
            for u in adjacency[v]:
                if u in unseen:
                    unseen.remove(u)
                    component_vertices.add(u)
                    stack.append(u)

        component_edges = {
            (a, b)
            for a, b in forest_edges
            if a in component_vertices and b in component_vertices
        }

        if component_edges:
            trees.append(component_edges)

    return trees


def _merge_edge_disjoint_trees(trees):
    """
    Greedily merge edge-disjoint trees whenever their union is still a tree.

    Two non-empty edge-disjoint trees have a tree as their union exactly when
    their vertex sets meet in one vertex.  Merging such pairs keeps all edges
    pairwise disjoint and usually makes the returned cover substantially
    smaller than simply returning every component of every forest.
    """
    parts = []
    seen_edges = set()

    for tree in trees:
        tree_edges = set(tree)
        if not tree_edges:
            continue

        overlap = seen_edges & tree_edges
        if overlap:
            raise RuntimeError(
                "The forest decomposition repeats an edge: "
                f"{next(iter(overlap))}"
            )

        tree_vertices = {v for edge in tree_edges for v in edge}
        if not _is_forest(tree_vertices, tree_edges) or not _is_connected(
            tree_vertices,
            tree_edges,
        ):
            raise RuntimeError("An edge-disjoint cover part is not a tree")

        parts.append((tree_edges, tree_vertices))
        seen_edges.update(tree_edges)

    while True:
        best = None

        for i in range(len(parts)):
            for j in range(i + 1, len(parts)):
                edges_i, vertices_i = parts[i]
                edges_j, vertices_j = parts[j]

                if len(vertices_i & vertices_j) != 1:
                    continue

                score = len(edges_i) + len(edges_j)
                if best is None or score > best[0]:
                    best = (score, i, j)

        if best is None:
            break

        _, i, j = best
        edges_i, vertices_i = parts[i]
        edges_j, vertices_j = parts[j]
        merged = (edges_i | edges_j, vertices_i | vertices_j)

        parts[i] = merged
        parts.pop(j)

    return [edges for edges, _ in parts]


def _extend_forest_to_tree(component_vertices, component_edges, forest_edges):
    """
    Extend a forest to a spanning tree inside the same connected component.

    For gscat, we need strongly collapsible subcomplexes, not merely forests.
    In a connected graph, such subcomplexes are trees. Therefore, each forest
    obtained from the forest decomposition (Nash-Williams or matroid
    partition) is extended to a tree inside the same connected component.
    The added edges may appear in several cover elements; this is allowed.
    """
    component_vertices = list(component_vertices)
    tree_edges = set(forest_edges)

    parent = {v: v for v in component_vertices}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        parent[ra] = rb
        return True

    # First add the forest edges.
    for a, b in forest_edges:
        if not union(a, b):
            raise RuntimeError("The input edge set is not a forest: a cycle was detected during extension")

    # Then add edges from the original graph until a spanning tree of the component is obtained.
    for a, b in sorted(component_edges):
        if len(tree_edges) == len(component_vertices) - 1:
            break
        if union(a, b):
            tree_edges.add((a, b))

    if len(component_vertices) == 1:
        return set()

    if len(tree_edges) != len(component_vertices) - 1:
        raise RuntimeError("Could not extend the forest to a tree: the graph component is not connected")

    if not _is_forest(component_vertices, tree_edges):
        raise RuntimeError("After extension the result is not a forest, so there is an error in the algorithm")

    return tree_edges


def _gscat_graph(k0, method="nash_williams", edge_disjoint=True):
    """
    Compute gscat for a 1-dimensional simplicial complex using arboricity.

    For a connected graph G,

        gscat(G) = arboricity(G) - 1.

    However, the cover itself must consist of strongly collapsible
    subcomplexes. Therefore, the algorithm:

      1. computes arboricity and a forest decomposition of the edge set,
         using either the Nash-Williams subset formula (method="nash_williams",
         exponential in |V| per component, but exact via brute force) or
         Edmonds' matroid partitioning algorithm (method="matroid_partition",
         polynomial in |E|, also exact);
      2. if ``edge_disjoint=True``, splits the forests into trees and merges
         compatible trees without ever copying an edge;
      3. otherwise, extends every forest to a spanning tree, which gives a
         minimum gscat-cover but may place a connecting edge in several cover
         elements;
      4. returns the resulting trees as cover elements.

    For disconnected graphs, the components are treated independently and
    their covers are combined. Hence the number of cover elements is the sum
    of the numbers of trees/singletons used for the components.

    Parameters
    ----------
    method : {"nash_williams", "matroid_partition"}
        Selects the algorithm used to compute arboricity + forest cover.
        Both are exact; matroid_partition is recommended for larger graphs
        since it avoids the 2^V subset enumeration of Nash-Williams.

    edge_disjoint : bool
        If True, the returned trees have pairwise disjoint edge sets.  Their
        intersections may contain vertices.  Such a cover is not necessarily
        minimum, so its length may exceed the returned gscat value plus one.
    """
    if method not in ("nash_williams", "matroid_partition"):
        raise ValueError(
            f"Unknown method={method!r} for _gscat_graph; "
            "expected 'nash_williams' or 'matroid_partition'."
        )

    vertices = sorted(k0.vertices)
    edges = _edges(k0)

    if not vertices:
        return 0, []

    components = _connected_components(vertices, edges)
    cover = []
    minimum_cover_elements = 0

    for comp_vertices in components:
        comp_vertices = sorted(comp_vertices)
        comp_edges = {
            (a, b) for a, b in edges
            if a in comp_vertices and b in comp_vertices
        }

        # An isolated vertex is covered by a singleton complex.
        if not comp_edges:
            sc = SimplicialComplex()
            sc.add_simplex([comp_vertices[0]])
            cover.append(sc.simplicial_complex)
            minimum_cover_elements += 1
            continue

        if method == "matroid_partition":
            a, forests = _arboricity_and_forests_matroid_partition(comp_vertices, comp_edges)
        else:
            a = _arboricity_nash_williams(comp_vertices, comp_edges)
            forests = _cover_by_forests_fast(comp_vertices, comp_edges, a)

            # For small graphs we keep an exact fallback. For larger graphs,
            # exponential backtracking can be too slow, so the fast method is preferred.
            if forests is None and len(comp_edges) <= 18:
                forests = _cover_by_forests_backtrack(comp_vertices, comp_edges, a)

            if forests is None:
                raise RuntimeError(
                    "Could not construct a forest cover with the fast method. "
                    "Try increasing max_restarts in _cover_by_forests_fast, "
                    "use the exact backtracking method for a smaller graph, "
                    "or pass method='matroid_partition' to gscat() for an "
                    "exact polynomial-time alternative."
                )

        minimum_cover_elements += a

        if edge_disjoint:
            trees = []
            for forest in forests:
                trees.extend(_split_forest_into_trees(forest))
            trees = _merge_edge_disjoint_trees(trees)
        else:
            trees = [
                _extend_forest_to_tree(comp_vertices, comp_edges, forest)
                for forest in forests
            ]

        for tree_edges in trees:
            tree_vertices = None if edge_disjoint else comp_vertices
            sc = _make_subcomplex_from_edges(
                tree_edges,
                vertices=tree_vertices,
            )

            # Final mathematical check: each cover element must be strongly collapsible.
            if not sc.is_strongly_collapsible():
                raise RuntimeError("A cover element is not strongly collapsible")

            cover.append(sc.simplicial_complex)

    target = k0.simplicial_complex
    union_cover = set().union(*cover) if cover else set()
    if union_cover != target:
        raise RuntimeError("The constructed cover does not coincide with the original complex")

    if edge_disjoint:
        used_edges = set()
        for cover_element in cover:
            element_edges = {
                tuple(sorted(simplex))
                for simplex in cover_element
                if len(simplex) == 2
            }
            repeated = used_edges & element_edges
            if repeated:
                raise RuntimeError(
                    "The constructed cover is not edge-disjoint; repeated edge "
                    f"{next(iter(repeated))}"
                )
            used_edges.update(element_edges)

        if used_edges != edges:
            raise RuntimeError(
                "The edge-disjoint cover does not partition all graph edges"
            )

    return minimum_cover_elements - 1, tuple(cover)

def _is_categorical_candidate(simplices_subset, ambient):
    sc = SimplicialComplex()
    for s in simplices_subset:
        sc.add_simplex(list(s))
    return sc.is_strongly_collapsible()


def _candidate_categorical_subcomplexes(k0):
    maximals = list(k0.maximal_simplices())
    M = len(maximals)
    candidates = []

    # all non-empty subsets of maximal simplices
    for r in range(1, M + 1):
        for combo in combinations(range(M), r):
            subset = [maximals[i] for i in combo]
            if _is_categorical_candidate(subset, k0):
                full = set()
                sc = SimplicialComplex()
                for s in subset:
                    sc.add_simplex(list(s))
                candidates.append(frozenset(sc.simplicial_complex))
    return list(set(candidates))


def _greedy_cover_upper_bound(target, candidates):
    remaining = set(target)
    chosen = []
    cands = sorted(candidates, key=len, reverse=True)
    while remaining:
        best = max(cands, key=lambda c: len(remaining & c), default=None)
        if best is None or len((remaining & best)) == 0:
            return None
        chosen.append(best)
        remaining -= best
    return chosen


def _gscat_general(k0):
    target = k0.simplicial_complex
    candidates = _candidate_categorical_subcomplexes(k0)

    if not candidates:
        return None, []

    upper = _greedy_cover_upper_bound(target, candidates)
    upper_n = len(upper) if upper is not None else len(candidates)

    for i in range(1, upper_n + 1):
        for combo in combinations(candidates, i):
            if set().union(*combo) == target:
                return i - 1, combo

    return upper_n - 1, tuple(upper)

def _choose_default_vertex(vertices):
    """
    Choose a default vertex.

    We try to use min(vertices), because graph constructors usually use
    integer labels. If labels are not comparable, we just take an arbitrary one.
    """
    if not vertices:
        raise ValueError("Cannot choose a vertex from an empty vertex set.")

    try:
        return min(vertices)
    except TypeError:
        return next(iter(vertices))


def _fresh_vertex_label(used, preferred):
    """
    Create a fresh vertex label that does not belong to used.

    For integer-labelled complexes, this keeps labels integer.
    For other labels, it creates labels like 'v_copy'.
    """
    if isinstance(preferred, int) and not isinstance(preferred, bool):
        int_labels = [
            v for v in used
            if isinstance(v, int) and not isinstance(v, bool)
        ]

        candidate = max(int_labels, default=-1) + 1

        while candidate in used:
            candidate += 1

        return candidate

    base = f"{preferred}_copy"
    candidate = base
    index = 2

    while candidate in used:
        candidate = f"{base}_{index}"
        index += 1

    return candidate
