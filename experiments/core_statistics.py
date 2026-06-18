from random_complexes import random_2_complex

def core_statistics(
    trials=100,
    n=7,
    p_edge=0.55,
    p_triangle=0.35,
    seed=0,
):
    """
    Print basic statistics about strong cores of random complexes.
    """
    strongly_collapsible_count = 0
    total_core_vertices = 0
    total_core_simplices = 0

    for t in range(trials):
        K = random_2_complex(
            n=n,
            p_edge=p_edge,
            p_triangle=p_triangle,
            seed=seed + t,
        )

        C = K.core()

        if len(C.vertices) <= 1:
            strongly_collapsible_count += 1

        total_core_vertices += len(C.vertices)
        total_core_simplices += len(C.simplicial_complex)

    print("Core statistics")
    print("---------------")
    print("trials:", trials)
    print("vertices:", n)
    print("p_edge:", p_edge)
    print("p_triangle:", p_triangle)
    print()
    print("strongly collapsible:", strongly_collapsible_count)
    print("average core vertices:", total_core_vertices / trials)
    print("average core simplices:", total_core_simplices / trials)


if __name__ == "__main__":
    core_statistics(
        trials=100,
        n=7,
        p_edge=0.55,
        p_triangle=0.35,
        seed=42,
    )
