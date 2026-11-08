"""Search for counterexamples to gscat(K) ≤ gscat(K^(1))."""

import argparse
import json
from pathlib import Path
from random import Random

from pysimplicial import gscat
from pysimplicial.datasets import random_2_complex


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vertices", type=int, default=6)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--p-edge", type=float, default=0.65)
    parser.add_argument("--p-triangle", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-states", type=int, default=20_000)
    parser.add_argument("--backend", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rng = Random(args.seed)
    unresolved = 0
    for _ in range(args.samples):
        seed = rng.randrange(2**32)
        complex_ = random_2_complex(args.vertices, args.p_edge, args.p_triangle, seed)
        result = gscat(complex_, backend=args.backend, max_candidate_states=args.max_states)
        skeleton = gscat(complex_.one_skeleton()).value
        if result.lower_bound > skeleton:
            data = {
                "seed": seed, "lower_bound": result.lower_bound, "upper_bound": result.upper_bound,
                "skeleton_value": skeleton,
                "facets": sorted(sorted(facet) for facet in complex_.facets),
            }
            encoded = json.dumps(data, indent=2)
            print(encoded)
            if args.output:
                args.output.write_text(encoded + "\n", encoding="utf-8")
            return
        if result.upper_bound > skeleton:
            unresolved += 1
    print(f"Samples: {args.samples}; proved within bound: {args.samples - unresolved}; unresolved: {unresolved}")


if __name__ == "__main__":
    main()
