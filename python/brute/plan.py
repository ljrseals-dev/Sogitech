"""Local Brute Genius planner; estimates only and never launches remote jobs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .space import SOGITECH_CODEC_SPACE, build_space, estimate_cardinality

def main():
    parser=argparse.ArgumentParser(description="Sogitech Codec – Brute Genius plan")
    parser.add_argument("--space",type=str,default=None,help="Optional JSON space override")
    parser.add_argument("--max-cost-usd",type=float,default=5.0)
    parser.add_argument("--cost-per-combo-usd",type=float,default=0.00015)
    args=parser.parse_args()
    if args.max_cost_usd < 0 or args.cost_per_combo_usd < 0:
        parser.error("cost limits cannot be negative")
    if args.space:
        path=Path(args.space)
        if not path.is_file(): parser.error(f"space file not found: {path}")
        try:
            space=json.loads(path.read_text(encoding="utf-8"))
        except (OSError,json.JSONDecodeError) as exc:
            parser.error(f"cannot read space file: {exc}")
    else:
        space=build_space()
    try:
        cardinality=estimate_cardinality(space)
    except (KeyError,TypeError,ValueError) as exc:
        parser.error(f"invalid search space: {exc}")
    limits = [item["max_combinations"] for item in space.get("constraints", [])
              if isinstance(item, dict) and "max_combinations" in item]
    max_combinations = min(limits, default=50000)
    estimated=cardinality*args.cost_per_combo_usd
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
