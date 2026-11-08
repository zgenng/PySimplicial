from pysimplicial import SimplicialMap, contiguity_distance, gscat, scat
from pysimplicial.datasets import cycle_graph


def main():
    complex_ = cycle_graph(3)
    identity = SimplicialMap.identity(complex_)
    constant = SimplicialMap.constant(complex_, complex_, 0)
    print("D =", contiguity_distance(identity, constant).value)
    print("scat =", scat(complex_).value)
    print("gscat =", gscat(complex_).value)


if __name__ == "__main__":
    main()
