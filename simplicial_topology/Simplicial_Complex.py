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

    def gscat(self, for_graph=None):
        # For graphs, the cover must be constructed on the original graph, not on its core.
        # Otherwise a tree would have the correct value gscat=0, but an empty cover,
        # and a graph with pendant edges would lose those edges in the returned cover.
        if for_graph is True:
            if self.dimension() > 1:
                raise ValueError(
                    "for_graph=True requires a 1-dimensional complex (graph), "
                    f"but dimension is {self.dimension()}"
                )
            return _gscat_graph(self)

        if for_graph is None and self.dimension() <= 1:
            return _gscat_graph(self)

        # For general simplicial complexes we may work with the core since gscat
        # is preserved under strong collapse. However, the resulting cover belongs to the core.
        # If one needs a cover of the original 2-dimensional complex, a separate lifting procedure is required.
        k0 = self.core()

        if len(k0.vertices) <= 1:
            return 0, [self.simplicial_complex]

        if for_graph is False:
            return _gscat_general_naive(k0)

        return _gscat_general(k0)


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

    for attempt in range(max_restarts):
        if attempt == 0:
            order = base_order[:]
        else:
            order = base_order[:]
            rng.shuffle(order)

        parents = [list(range(len(vertices))) for _ in range(k)]
        forests = [[] for _ in range(k)]

        success = True

        for a, b in order:
            possible = [
                f for f in range(k)
                if can_add(parents[f], a, b)
            ]

            if not possible:
                success = False
                break

            # Prefer the currently smallest forest. This balances the cover.
            f = min(possible, key=lambda x: len(forests[x]))

            add_edge(parents[f], a, b)
            forests[f].append((a, b))

        if success:
            if any(not _is_forest(vertices, f) for f in forests):
                continue

            union_edges = set().union(*(set(f) for f in forests)) if forests else set()
            if union_edges == set(edge_list):
                return forests

    return None


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


def _extend_forest_to_tree(component_vertices, component_edges, forest_edges):
    """
    Extend a forest to a spanning tree inside the same connected component.

    For gscat, we need strongly collapsible subcomplexes, not merely forests.
    In a connected graph, such subcomplexes are trees. Therefore, each forest
    obtained from the Nash-Williams decomposition is extended to a tree inside
    the same connected component. The added edges may appear in several cover
    elements; this is allowed.
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


def _gscat_graph(k0):
    """
    Compute gscat for a 1-dimensional simplicial complex using arboricity.

    For a connected graph G,

        gscat(G) = arboricity(G) - 1.

    However, the cover itself must consist of strongly collapsible
    subcomplexes. Therefore, the algorithm:

      1. computes arboricity using the Nash-Williams formula;
      2. decomposes the edge set into the minimum number of forests;
      3. extends each forest to a tree inside the corresponding component;
      4. returns these trees as cover elements.

    For disconnected graphs, the components are treated independently and
    their covers are combined. Hence the number of cover elements is the sum
    of the numbers of trees/singletons used for the components.
    """
    vertices = sorted(k0.vertices)
    edges = _edges(k0)

    if not vertices:
        return 0, []

    components = _connected_components(vertices, edges)
    cover = []

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
            continue

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
                "or use the exact backtracking method for a smaller graph."
            )

        for forest in forests:
            tree_edges = _extend_forest_to_tree(comp_vertices, comp_edges, forest)
            sc = _make_subcomplex_from_edges(tree_edges, vertices=comp_vertices)

            # Final mathematical check: each cover element must be strongly collapsible.
            if not sc.is_strongly_collapsible():
                raise RuntimeError("A cover element is not strongly collapsible")

            cover.append(sc.simplicial_complex)

    target = k0.simplicial_complex
    union_cover = set().union(*cover) if cover else set()
    if union_cover != target:
        raise RuntimeError("The constructed cover does not coincide with the original complex")

    return len(cover) - 1, tuple(cover)

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