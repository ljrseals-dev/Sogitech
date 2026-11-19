"""
Named objectives for Brute Genius scoring of Sogitech Codec combinations.

Each objective is a pure function that receives a dict of results from a
single combination run and returns a scalar score (higher is better unless
noted).
"""

from __future__ import annotations

from typing import Any, Callable, Dict


def spectral_l1(results: Dict[str, Any]) -> float:
    """Negative mean absolute spectral error (so higher is better)."""
    return -float(results.get("spectral_l1", 1e6))


def multi_res_stft(results: Dict[str, Any]) -> float:
    return -float(results.get("multi_res_stft", 1e6))


def param_l1(results: Dict[str, Any]) -> float:
    return -float(results.get("param_l1", 1e6))


def perceptual_proxy(results: Dict[str, Any]) -> float:
    """Simple proxy: combination of spectral error + residual energy match."""
    spec = float(results.get("spectral_l1", 1.0))
    resid = float(results.get("residual_energy_error", 1.0))
    return -(spec + 0.3 * resid)


def latency_penalty(results: Dict[str, Any], budget_ms: float = 20.0) -> float:
    lat = float(results.get("latency_ms", 999.0))
    if lat <= budget_ms:
        return 0.0
    return -(lat - budget_ms) * 0.1


def composite_score(results: Dict[str, Any]) -> float:
    """Default ranking objective used by the search."""
    return (
        multi_res_stft(results)
        + 0.2 * param_l1(results)
        + latency_penalty(results)
    )


registry: Dict[str, Callable[[Dict[str, Any]], float]] = {
    "spectral_l1": spectral_l1,
    "multi_res_stft": multi_res_stft,
    "param_l1": param_l1,
    "perceptual_proxy": perceptual_proxy,
    "composite": composite_score,
}
