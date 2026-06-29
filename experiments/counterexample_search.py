# counterexample_search.py

import argparse
import json
import random
from datetime import datetime


from simplicial_topology.Simplicial_Complex import SimplicialComplex


def value_only(result):
    """
    Your gscat() returns:

        (value, cover)

    This function extracts only the value.
    """
    if isinstance(result, tuple):
        return result[0]
    return result


def sorted_simplex(simplex):
    return sorted(list(simplex))


def sorted_maximal_simplices(K):
    return sorted(
        [sorted_simplex(s) for s in K.maximal_simplices()],
        key=lambda x: (len(x), x),
    )


def random_facet_complex(
    n,
    dim,
    facets,
    seed=None,
    pure=False,
    min_facet_dim=1,
):
    """
    Generate a random simplicial complex of exact dimension dim.

    Idea:
        Instead of generating all lower-dimensional faces randomly,
        we generate random maximal simplices and use K.add_simplex().
        This is much better for dim > 2.

    Parameters
    ----------
    n : int
        Number of vertices.
    dim : int
        Target dimension of the complex.
    facets : int
        Number of random facets to add.
    seed : int, optional
        Random seed.
    pure : bool
        If True, all generated facets have dimension exactly dim.
        If False, facets may have dimensions between min_facet_dim and dim.
    min_facet_dim : int
        Minimal dimension of additional random facets.

    Returns
    -------
    SimplicialComplex
        A random simplicial complex with dimension exactly dim.
    """
    if dim < 1:
        raise ValueError("dim must be at least 1")

    if n < dim + 1:
        raise ValueError(
            f"Need at least dim + 1 vertices. Got n={n}, dim={dim}."
        )

    if facets < 1:
        raise ValueError("facets must be at least 1")

    rng = random.Random(seed)
    K = SimplicialComplex()

    # Add all vertices, including isolated ones.
    for v in range(n):
        K.add_simplex([v])

    vertices = list(range(n))

    # Force the complex to have exact dimension dim.
    first_facet = rng.sample(vertices, dim + 1)
    K.add_simplex(first_facet)

    # Add remaining random facets.
    for _ in range(facets - 1):
        if pure:
            facet_dim = dim
        else:
            facet_dim = rng.randint(min_facet_dim, dim)

        size = facet_dim + 1
        simplex = rng.sample(vertices, size)
        K.add_simplex(simplex)

    return K


def random_mixed_complex(
    n,
    dim,
    p_edge=0.5,
    p_simplex=0.3,
    seed=None,
):
    """
    Generate a random complex by increasing dimension step by step.

    This is closer to the random_2_complex style:
        1. Generate random edges.
        2. Add higher-dimensional simplices only if their boundary exists.

    Warning:
        For dim >= 4 this often produces too few high-dimensional simplices.
        For serious counterexample search, random_facet_complex is usually better.
    """
    from itertools import combinations

    if n < dim + 1:
        raise ValueError(
            f"Need at least dim + 1 vertices. Got n={n}, dim={dim}."
        )

    rng = random.Random(seed)
    K = SimplicialComplex()

    for v in range(n):
        K.add_simplex([v])

    # Edges
    for i, j in combinations(range(n), 2):
        if rng.random() < p_edge:
            K.add_simplex([i, j])

    # Higher-dimensional simplices
    for size in range(3, dim + 2):
        for simplex in combinations(range(n), size):
            boundary_exists = True

            for face in combinations(simplex, size - 1):
                if frozenset(face) not in K.simplicial_complex:
                    boundary_exists = False
                    break

            if boundary_exists and rng.random() < p_simplex:
                K.add_simplex(simplex)

    # If dimension is still too small, force one dim-simplex.
    if K.dimension() < dim:
        simplex = rng.sample(list(range(n)), dim + 1)
        K.add_simplex(simplex)

    return K


def compute_values(K, exact_general=False):
    """
    Compute

        lhs = gscat(K)
        rhs = gscat(K^(1))

    If exact_general=False, use your faster general gscat algorithm.
    If exact_general=True, use gscat(for_graph=False), which is slower
    but better for verifying a suspected counterexample.
    """
    if exact_general:
        lhs = value_only(K.gscat(for_graph=False))
    else:
        lhs = value_only(K.gscat())

    rhs = value_only(K.one_skeleton().gscat(for_graph=True))

    return lhs, rhs


def save_counterexample(K, lhs, rhs, filename="counterexample.json"):
    data = {
        "datetime": datetime.now().isoformat(),
        "dimension": K.dimension(),
        "gscat_K": lhs,
        "gscat_K_1_skeleton": rhs,
        "maximal_simplices": sorted_maximal_simplices(K),
        "all_simplices": sorted(
            [sorted_simplex(s) for s in K.simplicial_complex],
            key=lambda x: (len(x), x),
        ),
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"\nSaved to {filename}")


def check_gscat_vs_skeleton(
    trials=100,
    n=7,
    dim=3,
    facets=6,
    seed=0,
    pure=False,
    generator="facets",
    p_edge=0.55,
    p_simplex=0.35,
    exact_verify=True,
    verbose=True,
    save=True,
):
    """
    Search for a random counterexample to

        gscat(K) <= gscat(K^(1))

    for complexes of exact dimension dim.

    Returns
    -------
    bool
        True if no counterexample was found.
        False if a counterexample was found.
    """
    print("\nSearch parameters")
    print("-----------------")
    print(f"trials      = {trials}")
    print(f"n           = {n}")
    print(f"dim         = {dim}")
    print(f"facets      = {facets}")
    print(f"generator   = {generator}")
    print(f"pure        = {pure}")
    print(f"seed        = {seed}")
    print(f"verify exact= {exact_verify}")

    for t in range(trials):
        current_seed = seed + t

        if generator == "facets":
            K = random_facet_complex(
                n=n,
                dim=dim,
                facets=facets,
                seed=current_seed,
                pure=pure,
            )

        elif generator == "mixed":
            K = random_mixed_complex(
                n=n,
                dim=dim,
                p_edge=p_edge,
                p_simplex=p_simplex,
                seed=current_seed,
            )

        else:
            raise ValueError(
                "Unknown generator. Use 'facets' or 'mixed'."
            )

        if K.dimension() != dim:
            continue

        try:
            lhs, rhs = compute_values(K, exact_general=False)

        except RuntimeError as e:
            print(f"\ntrial={t:04d} skipped because of RuntimeError:")
            print(e)
            continue

        if verbose:
            print(
                f"trial={t:04d} | "
                f"seed={current_seed} | "
                f"dim={K.dimension()} | "
                f"gscat(K)={lhs} | "
                f"gscat(K^1)={rhs}"
            )

        if lhs is not None and rhs is not None and lhs > rhs:
            print("\nCandidate counterexample found by fast gscat!")
            print("Now verifying with exact general gscat...")

            if exact_verify:
                try:
                    lhs_exact, rhs_exact = compute_values(K, exact_general=True)
                except Exception as e:
                    print("\nExact verification failed:")
                    print(e)
                    print("\nThis is only a candidate, not confirmed.")
                    print("Maximal simplices:")
                    for s in sorted_maximal_simplices(K):
                        print(s)
                    return False

                lhs, rhs = lhs_exact, rhs_exact

                print(f"exact gscat(K)   = {lhs}")
                print(f"gscat(K^1)       = {rhs}")

            if lhs is not None and rhs is not None and lhs > rhs:
                print("\nCOUNTEREXAMPLE FOUND")
                print("--------------------")
                print(f"seed = {current_seed}")
                print(f"dim(K) = {K.dimension()}")
                print(f"gscat(K) = {lhs}")
                print(f"gscat(K^1) = {rhs}")

                print("\nMaximal simplices:")
                for s in sorted_maximal_simplices(K):
                    print(s)

                if save:
                    save_counterexample(K, lhs, rhs)

                return False

            else:
                print("\nAfter exact verification this was not a counterexample.")

    print("\nNo counterexample found.")
    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Search for counterexamples to gscat(K) <= gscat(K^(1))."
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=1000,
        help="Number of random complexes to test.",
    )

    parser.add_argument(
        "--n",
        type=int,
        default=11,
        help="Number of vertices.",
    )

    parser.add_argument(
        "--dim",
        type=int,
        default=3,
        help="Target dimension of generated complexes.",
    )

    parser.add_argument(
        "--facets",
        type=int,
        default=8,
        help="Number of random facets for the facets generator.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Initial random seed.",
    )

    parser.add_argument(
        "--pure",
        action="store_true",
        help="Generate pure dim-dimensional complexes.",
    )

    parser.add_argument(
        "--generator",
        choices=["facets", "mixed"],
        default="facets",
        help="Random generator type.",
    )

    parser.add_argument(
        "--p-edge",
        type=float,
        default=0.55,
        help="Edge probability for mixed generator.",
    )

    parser.add_argument(
        "--p-simplex",
        type=float,
        default=0.35,
        help="Higher simplex probability for mixed generator.",
    )

    parser.add_argument(
        "--no-exact-verify",
        action="store_true",
        help="Do not verify candidates using exact general gscat.",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print every trial.",
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save counterexample to JSON.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    check_gscat_vs_skeleton(
        trials=args.trials,
        n=args.n,
        dim=args.dim,
        facets=args.facets,
        seed=args.seed,
        pure=args.pure,
        generator=args.generator,
        p_edge=args.p_edge,
        p_simplex=args.p_simplex,
        exact_verify=not args.no_exact_verify,
        verbose=not args.quiet,
        save=not args.no_save,
    )