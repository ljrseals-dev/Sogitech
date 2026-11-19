"""
Brute Genius search space definition for the Sogitech Codec.

This is a pure data description of the discrete dimensions that Brute Genius
will expand into a full Cartesian product. It does not execute any search
itself – that is the job of the Brute Genius MCP / server.
"""

from __future__ import annotations

from typing import Any, Dict, List


SOGITECH_CODEC_SPACE: Dict[str, Any] = {
    "name": "sogitech_codec_v1",
    "description": (
        "Exhaustive search over analysis front-ends, residual strategies, "
        "quantisation, and optional neural controller variants for the "
        "Sogitech hybrid codec."
    ),
    "dimensions": {
        "pitch_tracker": [
            "librosa_pyin",
            "autocorr",
            "crepe_tiny",          # placeholder for future
            "rmvpe",              # if weights available
        ],
        "envelope_method": [
            "rms",
            "peak",
            "harmonic_32",
        ],
        "residual_strategy": [
            "none",
            "highpass_3k",
            "harmonic_subtract",
            "neural_residual",
        ],
        "filter_mapping": [
            "centroid_linear",
            "centroid_log",
            "formant_proxy",
        ],
        "quantisation_bits": [
            4, 6, 8, 12,          # per major parameter group
        ],
        "neural_controller": [
            "none",
            "mlp_64_128",
            "mlp_128_256",
            # future: "conformer_tiny", etc.
        ],
        "objective_primary": [
            "spectral_l1",
            "multi_res_stft",
            "param_l1",
            "perceptual_proxy",
        ],
        "sample_rate": [
            48000,
            96000,
        ],
        "latency_budget_ms": [
            5, 10, 20, 40,
        ],
    },
    "constraints": [
        # Example soft constraints that a planner can enforce
        {"if": {"neural_controller": "none"}, "then": {"residual_strategy": ["none", "highpass_3k", "harmonic_subtract"]}},
        {"max_combinations": 50_000},   # safety for local runs
    ],
    "default_objective": "multi_res_stft",
}


def build_space(overrides: Dict[str, List[Any]] | None = None) -> Dict[str, Any]:
    """Return a copy of the space with optional dimension overrides."""
    import copy
    space = copy.deepcopy(SOGITECH_CODEC_SPACE)
    if overrides:
        for k, v in overrides.items():
            if k in space["dimensions"]:
                space["dimensions"][k] = v
    return space


def estimate_cardinality(space: Dict[str, Any] | None = None) -> int:
    """Product of the sizes of all dimensions."""
    space = space or SOGITECH_CODEC_SPACE
    card = 1
    for vals in space["dimensions"].values():
        card *= len(vals)
    return card
