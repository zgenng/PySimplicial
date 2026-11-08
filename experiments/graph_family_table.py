"""Print exact gscat values for standard graph families."""

import argparse

from pysimplicial import gscat
from pysimplicial.datasets import complete_graph, cycle_graph, path_graph, star_graph, wheel_graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-n", type=int, default=10)
    args = parser.parse_args()
    graphs = (complete_graph, cycle_graph, path_graph, star_graph, wheel_graph)
    print("n", *(graph.__name__ for graph in graphs), sep="\t")
    for n in range(4, args.max_n + 1):
        print(n, *(gscat(graph(n)).value for graph in graphs), sep="\t")


if __name__ == "__main__":
    main()
