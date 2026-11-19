"""
Offline Python reference renderer for the Sogitech hybrid engine.

Implements the same signal path (32-harmonic additive + VA + PowerSag +
dual-pole SVF) so the analysis → parameter → audio chain can be validated
without the Cmajor runtime.

Supports both static parameter dicts and time-varying parameter streams
(dicts of arrays aligned to the output length, or shorter arrays that are
linearly interpolated).
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional, Union


def _va_osc(phase: np.ndarray, wave: int) -> np.ndarray:
    if wave == 0:
        return np.sin(phase)
    if wave == 1:
        return (phase / np.pi) - 1.0
    if wave == 2:
        return np.where(phase < np.pi * 0.5, 1.0, -1.0)
    if wave == 3:
        return np.abs(np.sin(phase))
    return np.sin(phase)


def _digital_osc(phase: np.ndarray, n_harm: int = 16) -> np.ndarray:
    """Vectorised additive partials (capped at 16 for offline speed)."""
    sig = np.zeros_like(phase)
    for k in range(1, n_harm + 1):
        sig += np.sin(phase * k) / k
    return sig * 0.14


def _interp_param(val: Union[float, np.ndarray], n: int) -> np.ndarray:
    if np.isscalar(val) or (isinstance(val, np.ndarray) and val.ndim == 0):
        return np.full(n, float(val), dtype=np.float64)
    arr = np.asarray(val, dtype=np.float64)
    if len(arr) == n:
        return arr
    # linear resample to n samples
    x_old = np.linspace(0.0, 1.0, len(arr))
    x_new = np.linspace(0.0, 1.0, n)
    return np.interp(x_new, x_old, arr)


def render(
    n_samples: int,
    sr: float = 48000.0,
    params: Optional[Dict[str, Union[float, np.ndarray]]] = None,
    residual: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Render n_samples of the hybrid engine.

    params may contain scalars or arrays. Arrays are interpolated to n_samples.
    residual (if supplied) is mixed according to residualGain / dryWet.
    """
    defaults = {
        "osc1Wave": 1.0, "osc2Wave": 2.0,
        "osc1Level": 0.80, "osc2Level": 0.80,
        "subLevel": 0.50, "noiseLevel": 0.12,
        "additiveBlend": 0.28, "powerSagDepth": 0.26,
        "filterLP": 0.72, "filterHP": 0.08, "filterRes": 0.22,
        "keyTrack": 0.55, "velMod": 0.62, "glideTime": 0.06,
        "midiNote": 69.0, "velocity": 0.82, "pressure": 0.0,
        "residualGain": 0.15, "dryWet": 1.0,
    }
    if params:
        defaults.update(params)
    p = {k: _interp_param(v, n_samples) for k, v in defaults.items()}

    if residual is None:
        residual = np.zeros(n_samples, dtype=np.float64)
    else:
        residual = np.asarray(residual[:n_samples], dtype=np.float64)
        if len(residual) < n_samples:
            residual = np.pad(residual, (0, n_samples - len(residual)))

    sr_inv = 1.0 / sr
    two_pi = 2.0 * np.pi

    # Glide state
    midi = p["midiNote"]
    target_freq = 440.0 * (2.0 ** ((midi - 69.0) / 12.0))
    current_freq = float(target_freq[0])
    glide = np.maximum(p["glideTime"], 0.001)
    glide_coeff = np.exp(-1.0 / (glide * sr))

    # Filter coefficients (slowly varying)
    base_lp = p["filterLP"] * 18000.0
    base_hp = p["filterHP"] * 120.0
    key_mod = ((midi - 60.0) / 48.0) * p["keyTrack"] * 0.6
    press_mod = (p["velocity"] * 0.8 + p["pressure"] * 0.2) * p["velMod"] * 2400.0
    lp_cut = np.clip(base_lp * (1.0 + key_mod) + press_mod * 0.7, 180.0, 22000.0)
    hp_cut = np.clip(base_hp * (1.0 + key_mod * 0.5) - press_mod * 0.3, 18.0, 7500.0)
    res = np.clip(p["filterRes"] * (1.0 + p["velocity"] * 0.35), 0.0, 0.95)

    def prewarp(f):
        omega = two_pi * f / sr
        return 2.0 * sr * np.tan(omega * 0.5)

    lp_k = prewarp(lp_cut) * sr_inv
    hp_k = prewarp(hp_cut) * sr_inv
    damp = 1.0 / (1.0 + 2.0 * res)

    # Oscillator state (scalar, updated sample-by-sample for phase continuity)
    osc1_ph = osc2_ph = sub_ph = 0.0
    sag = 1.0
    load = 0.0
    lp1 = bp1 = hp1 = 0.0
    lp2 = bp2 = hp2 = 0.0

    rng = np.random.default_rng(42)
    noise = rng.uniform(-1.0, 1.0, n_samples)
    out = np.zeros(n_samples, dtype=np.float64)

    # Pre-extract for speed
    o1w = p["osc1Wave"].astype(int)
    o2w = p["osc2Wave"].astype(int)
    o1l = p["osc1Level"]
    o2l = p["osc2Level"]
    subl = p["subLevel"]
    noisel = p["noiseLevel"]
    blend = p["additiveBlend"]
    sagd = p["powerSagDepth"]
    rg = p["residualGain"]
    dw = p["dryWet"]

    for i in range(n_samples):
        current_freq = current_freq * glide_coeff[i] + target_freq[i] * (1.0 - glide_coeff[i])
        phase_inc = current_freq * two_pi * sr_inv

        osc1_ph = (osc1_ph + phase_inc) % two_pi
        osc2_ph = (osc2_ph + phase_inc * 1.01) % two_pi
        sub_ph = (sub_ph + phase_inc * 0.5) % two_pi

        va = (
            _va_osc(np.array([osc1_ph]), o1w[i])[0] * o1l[i]
            + _va_osc(np.array([osc2_ph]), o2w[i])[0] * o2l[i]
            + np.sin(sub_ph) * subl[i]
            + noise[i] * noisel[i]
        )
        add = _digital_osc(np.array([osc1_ph]))[0]
        mixed = (1.0 - blend[i]) * va + blend[i] * add

        load = 0.992 * load + 0.008 * abs(mixed)
        sag *= np.exp(-0.12 * sagd[i] * load * sr_inv)
        sag = max(0.48, sag)
        mixed *= sag

        lk = lp_k[i]
        hk = hp_k[i]
        d = damp[i]
        hp1 = (mixed - lp1 - bp1 * d) / (1.0 + lk * lk)
        bp1 += lk * hp1
        lp1 += lk * bp1
        hp2 = (hp1 - lp2 - bp2 * d) / (1.0 + hk * hk)
        bp2 += hk * hp2
        lp2 += hk * bp2

        filtered = lp2 * 0.72 + hp2 * 0.28
        synth = filtered * 0.78
        res_sig = residual[i] * rg[i]
        out[i] = (1.0 - dw[i]) * res_sig + dw[i] * (synth + res_sig * 0.3)

    return np.tanh(out * 1.2).astype(np.float32)
