"""Brute Genius search space for staged Sogitech codec optimisation.

The full Cartesian product is deliberately capped below 50k candidates.
Each axis contains concrete algorithm/configuration choices; unavailable
external models are not advertised as installed or runnable backends.
"""
from __future__ import annotations
from copy import deepcopy
from math import prod
from typing import Any

SOGITECH_CODEC_SPACE: dict[str, Any] = {
    "name": "sogitech_codec_v2",
    "description": "Bounded search across analysis, codec, quantisation, neural-controller and deployment trade-offs.",
    "dimensions": {
        "codec_mode": ["dsp", "parameter"],
        "pitch_tracker": ["autocorr", "librosa_pyin"],
        "envelope_method": ["rms", "peak", "hilbert"],
        "residual_strategy": ["none", "highpass_3k", "harmonic_subtract"],
        "filter_mapping": ["centroid_linear", "centroid_log", "spectral_rolloff"],
        "quantisation_bits": [4, 8, 12],
        "quantisation_layout": ["uniform", "perceptual"],
        "neural_controller": ["none", "mlp_64_128", "mlp_64_256"],
        "objective_primary": ["spectral_l1", "multi_res_stft", "perceptual_proxy"],
        "sample_rate": [48000, 96000],
        "latency_budget_ms": [5, 20],
    },
    "constraints": [
        {"if": {"neural_controller": "none"}, "then": {"residual_strategy": ["none", "highpass_3k", "harmonic_subtract"]}},
        {"if": {"codec_mode": "dsp"}, "then": {"neural_controller": ["none"]}},
        {"max_combinations": 50000},
    ],
    "default_objective": "multi_res_stft",
    "evaluation_protocol": {
        "split": "grouped_by_source_recording",
        "metrics": ["spectral_l1", "multi_res_stft", "param_l1", "residual_energy_error", "latency_ms", "cpu_percent", "peak_dbfs"],
        "repetitions": 3,
        "seed_policy": "fixed_per_candidate",
    },
}

def build_space(overrides: dict[str, list[Any]] | None = None) -> dict[str, Any]:
    """Return a validated deep copy with optional dimension overrides."""
    space = deepcopy(SOGITECH_CODEC_SPACE)
    if overrides:
        unknown = set(overrides) - set(space["dimensions"])
        if unknown:
            raise KeyError(f"unknown search dimensions: {', '.join(sorted(unknown))}")
        for key, values in overrides.items():
            if not isinstance(values, list) or not values:
                raise ValueError(f"override for {key} must be a non-empty list")
            space["dimensions"][key] = list(values)
    return space

def estimate_cardinality(space: dict[str, Any] | None = None) -> int:
    """Return the unconstrained Cartesian upper bound for the dimensions."""
    selected = SOGITECH_CODEC_SPACE if space is None else space
    if not isinstance(selected, dict):
        raise ValueError("search space must be an object")
    dimensions = selected.get("dimensions", {})
    if not isinstance(dimensions, dict) or not dimensions:
        raise ValueError("dimensions must be a non-empty object")
    for name, values in dimensions.items():
        if not isinstance(name, str) or not name:
            raise ValueError("dimension names must be non-empty strings")
        if not isinstance(values, list) or not values:
            raise ValueError(f"dimension {name} must contain a non-empty list")
    return prod(len(values) for values in dimensions.values())
