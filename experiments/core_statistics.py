"""Count embedded strong-core sizes in a random sample."""

import argparse
from collections import Counter
from random import Random

from pysimplicial import core
from pysimplicial.datasets import random_2_complex


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vertices", type=int, default=6)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = Random(args.seed)
    counts = Counter(
        len(core(random_2_complex(args.vertices, seed=rng.randrange(2**32))).vertices)
        for _ in range(args.samples)
    )
    for size, count in sorted(counts.items()):
        print(size, count, sep="\t")


if __name__ == "__main__":
    main()
