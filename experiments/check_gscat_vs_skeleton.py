from random_complexes import random_2_complex


def check_gscat_vs_skeleton(
    trials=100,
    n=6,
    p_edge=0.55,
    p_triangle=0.35,
    seed=0,
    verbose=True,
):
    """
    Search for a random counterexample to

        gscat(K) <= gscat(K^(1)).

    Returns
    -------
    bool
        True if no counterexample was found.
        False if a counterexample was found.
    """
    for t in range(trials):
        K = random_2_complex(
            n=n,
            p_edge=p_edge,
            p_triangle=p_triangle,
            seed=seed + t,
        )

        lhs, _ = K.gscat()
        rhs, _ = K.one_skeleton().gscat(for_graph=True)

        if verbose:
            print(
                f"trial={t:04d} | "
                f"dim={K.dimension()} | "
                f"gscat(K)={lhs} | "
                f"gscat(K^1)={rhs}"
            )

        if lhs is not None and rhs is not None and lhs > rhs:
            print("\nPotential counterexample found!")
            print("simplices =", K.simplicial_complex)
            print("gscat(K) =", lhs)
            print("gscat(K^1) =", rhs)
            return False

    print("\nNo counterexample found.")
    return True


if __name__ == "__main__":
    check_gscat_vs_skeleton(
        trials=500,
        n=7,
        p_edge=0.55,
        p_triangle=0.35,
        seed=42,
    )
