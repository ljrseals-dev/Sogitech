"""
Lightweight local planner that prints a Brute Genius cost estimate
and the expanded search space. Does not launch real remote nodes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .space import SOGITECH_CODEC_SPACE, build_space, estimate_cardinality


def main():
    parser = argparse.ArgumentParser(description="Sogitech Codec – Brute Genius plan")
    parser.add_argument("--space", type=str, default=None, help="Optional JSON space override")
    parser.add_argument("--max-cost-usd", type=float, default=5.0)
    args = parser.parse_args()

    if args.space and Path(args.space).exists():
        with open(args.space) as f:
            space = json.load(f)
    else:
        space = build_space()

    card = estimate_cardinality(space)
    # Very rough cost model (for illustration only)
    cost_per_combo = 0.00015   # USD
    estimated = card * cost_per_combo

    print("=== Sogitech Codec – Brute Genius Plan ===")
    print(f"Space name     : {space.get('name')}")
    print(f"Cardinality    : {card:,}")
    print(f"Est. cost (USD): {estimated:.2f}")
    print(f"Max allowed    : {args.max_cost_usd:.2f}")
    if estimated > args.max_cost_usd:
        print(">>> REFUSED by cost ceiling. Reduce dimensions or raise --max-cost-usd.")
    else:
        print(">>> Plan accepted (local estimate only).")
    print("\nDimensions:")
    for k, v in space["dimensions"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
