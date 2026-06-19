from generators.graphs import (
    complete_graph,
    cycle_graph,
    path_graph,
    star_graph,
    wheel_graph,
)


def graph_family_table(max_n=10):
    """
    Print gscat values for several standard graph families.
    """
    print("Graph family table")
    print("------------------")
    print(f"{'Graph':<12} {'n':<4} {'gscat':<6} {'cover size':<10}")
    print("-" * 38)

    families = [
        ("Path", path_graph),
        ("Cycle", cycle_graph),
        ("Star", star_graph),
        ("Wheel", wheel_graph),
        ("Complete", complete_graph),
    ]

    for name, constructor in families:
        for n in range(3, max_n + 1):
            K = constructor(n)
            value, cover = K.gscat(for_graph=True)

            print(f"{name:<12} {n:<4} {value:<6} {len(cover):<10}")

        print()


if __name__ == "__main__":
    graph_family_table(max_n=8)
