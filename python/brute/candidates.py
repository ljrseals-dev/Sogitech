"""Deterministic, locally constrained candidate enumeration for Brute Genius.

This module enumerates parameter combinations only. It does not run DSP,
training, remote workers, or benchmark jobs.
"""
from __future__ import annotations

from itertools import product
from math import prod
from typing import Any, Iterator


def _validated_parts(space: dict[str, Any] | None):
    selected = SOGITECH_CODEC_SPACE if space is None else space
    if not isinstance(selected, dict):
        raise ValueError("search space must be an object")
    dimensions = selected.get("dimensions")
    if not isinstance(dimensions, dict) or not dimensions:
        raise ValueError("dimensions must be a non-empty object")
    keys = list(dimensions)
    values = []
    for name in keys:
        axis = dimensions[name]
        if not isinstance(name, str) or not name:
            raise ValueError("dimension names must be non-empty strings")
        if not isinstance(axis, list) or not axis:
            raise ValueError(f"dimension {name} must contain a non-empty list")
        values.append(axis)

    constraints = selected.get("constraints", [])
    if not isinstance(constraints, list):
        raise ValueError("constraints must be a list")
    max_combinations = 50000
    for rule in constraints:
        if not isinstance(rule, dict):
            raise ValueError("each constraint must be an object")
        if "max_combinations" in rule:
            limit = rule["max_combinations"]
            if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
                raise ValueError("max_combinations constraints must be positive integers")
            max_combinations = min(max_combinations, limit)
            continue
        condition = rule.get("if")
        consequence = rule.get("then")
        if not isinstance(condition, dict) or not condition or not isinstance(consequence, dict):
            raise ValueError("conditional constraints require non-empty 'if' and object 'then'")
        if not consequence:
            raise ValueError("conditional constraint 'then' cannot be empty")
        unknown = (set(condition) | set(consequence)) - set(keys)
        if unknown:
            raise ValueError(f"constraint references unknown dimensions: {', '.join(sorted(unknown))}")
        for key, allowed in consequence.items():
            if not isinstance(allowed, list) or not allowed:
                raise ValueError(f"constraint values for {key} must be a non-empty list")
    cardinality = prod(len(axis) for axis in values)
    if cardinality > max_combinations:
        raise ValueError(
            f"Cartesian upper bound {cardinality} exceeds combination ceiling {max_combinations}"
        )
    return keys, values, constraints, cardinality


def iter_candidates(space: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Yield valid candidates in stable dimension/value declaration order."""
    keys, axes, constraints, _ = _validated_parts(space)
    for combination in product(*axes):
        candidate = dict(zip(keys, combination))
        accepted = True
        for rule in constraints:
            if "max_combinations" in rule:
                continue
            condition = rule["if"]
            if all(candidate[key] == expected for key, expected in condition.items()):
                if any(candidate[key] not in allowed for key, allowed in rule["then"].items()):
                    accepted = False
                    break
        if accepted:
            yield candidate


def count_candidates(space: dict[str, Any] | None = None) -> int:
    """Count valid candidates after applying conditional constraints."""
    return sum(1 for _ in iter_candidates(space))


# Imported last to avoid a circular dependency during module initialization.
from .space import SOGITECH_CODEC_SPACE  # noqa: E402
