"""Compute gscat for a named triangulation or a numbered genus-two example."""

import argparse

from pysimplicial import gscat
from pysimplicial.datasets import genus_two_triangulation, triangulation, triangulation_names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--name", choices=triangulation_names())
    source.add_argument("--genus-two", type=int)
    parser.add_argument("--backend", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--restarts", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-states", type=int)
    args = parser.parse_args()
    complex_ = triangulation(args.name) if args.name else genus_two_triangulation(args.genus_two)
    result = gscat(complex_, backend=args.backend, restarts=args.restarts, seed=args.seed,
                   max_candidate_states=args.max_states, progress=print)
    print(f"gscat = {result.value}" if result.exact else f"gscat ∈ [{result.lower_bound}, {result.upper_bound}]")
    print("GPU used:", result.gpu_used)
    for index, part in enumerate(result.cover, 1):
        print(f"U{index}", sorted(tuple(sorted(facet)) for facet in part.facets))


if __name__ == "__main__":
    main()
