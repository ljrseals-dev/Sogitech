"""Local Brute Genius planner; estimates only and never launches remote jobs."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from .space import SOGITECH_CODEC_SPACE, build_space, estimate_cardinality

def main():
    parser=argparse.ArgumentParser(description="Sogitech Codec – Brute Genius plan")
    parser.add_argument("--space",type=str,default=None,help="Optional JSON space override")
    parser.add_argument("--max-cost-usd",type=float,default=5.0)
    parser.add_argument("--cost-per-combo-usd",type=float,default=0.00015)
    args=parser.parse_args()
    if (not math.isfinite(args.max_cost_usd) or not math.isfinite(args.cost_per_combo_usd)
            or args.max_cost_usd < 0 or args.cost_per_combo_usd < 0):
        parser.error("cost limits must be finite and cannot be negative")
    if args.space:
        path=Path(args.space)
        if not path.is_file(): parser.error(f"space file not found: {path}")
        try:
            space=json.loads(path.read_text(encoding="utf-8"))
        except (OSError,json.JSONDecodeError) as exc:
            parser.error(f"cannot read space file: {exc}")
    else:
        space=build_space()
    if not isinstance(space, dict):
        parser.error("search space must be a JSON object")
    try:
        cardinality=estimate_cardinality(space)
    except (KeyError,TypeError,ValueError) as exc:
        parser.error(f"invalid search space: {exc}")
    constraints = space.get("constraints", [])
    if not isinstance(constraints, list):
        parser.error("constraints must be a list")
    limits = [item["max_combinations"] for item in constraints
              if isinstance(item, dict) and "max_combinations" in item]
    if any(not isinstance(limit, int) or isinstance(limit, bool) or limit < 1
           for limit in limits):
        parser.error("max_combinations constraints must be positive integers")
    max_combinations = min([50000, *limits])
    try:
        estimated=cardinality*args.cost_per_combo_usd
    except OverflowError:
        parser.error("estimated cost overflowed; reduce the search space or cost assumption")
    if not math.isfinite(estimated):
        parser.error("estimated cost is not finite; reduce the search space or cost assumption")
    print("=== Sogitech Codec – Brute Genius Plan ===")
    print(f"Space           : {space.get('name','unnamed')}")
    print(f"Cartesian upper : {cardinality:,}")
    print(f"Cost estimate   : ${estimated:.2f} (illustrative only)")
    print(f"Cost ceiling    : ${args.max_cost_usd:.2f}")
    print(f"Combo ceiling   : {max_combinations:,}")
    if cardinality > max_combinations:
        print("REFUSED: Cartesian upper bound exceeds the configured combination ceiling.")
    elif estimated > args.max_cost_usd:
        print("REFUSED: estimated cost exceeds the configured cost ceiling.")
    else:
        print("PLAN ONLY: estimate is within limits; no Brute Genius jobs were launched.")
    print("\nDimensions:")
    for key,values in space["dimensions"].items():
        print(f"  {key}: {', '.join(map(str,values))}")
    print("\nConstraints are descriptive upper-bound guards; conditional pruning is not applied locally.")

if __name__=="__main__": main()
