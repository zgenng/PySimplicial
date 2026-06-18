from check_gscat_vs_skeleton import check_gscat_vs_skeleton


def run_search():
    """
    Run a parameter sweep over random 2-dimensional complexes.
    """
    configs = [
        {"n": 5, "p_edge": 0.40, "p_triangle": 0.20},
        {"n": 5, "p_edge": 0.60, "p_triangle": 0.40},
        {"n": 6, "p_edge": 0.45, "p_triangle": 0.25},
        {"n": 6, "p_edge": 0.65, "p_triangle": 0.45},
        {"n": 7, "p_edge": 0.50, "p_triangle": 0.30},
    ]

    global_seed = 1000

    for idx, config in enumerate(configs):
        print()
        print("=" * 60)
        print("Configuration:", config)
        print("=" * 60)

        ok = check_gscat_vs_skeleton(
            trials=100,
            seed=global_seed + 10000 * idx,
            verbose=False,
            **config,
        )

        if not ok:
            print("Search stopped.")
            return

    print()
    print("Search finished. No counterexample found in these experiments.")


if __name__ == "__main__":
    run_search()
