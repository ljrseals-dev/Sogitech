"""Residual / noise energy estimation for the Sogitech Codec."""

from __future__ import annotations

import numpy as np
from typing import Tuple


def residual_energy(
    audio: np.ndarray,
    sr: int,
    frame_length: int = 1024,
    hop_length: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate residual (noise / unvoiced) energy.
    Returns (times, residual_rms).
    """
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # High-pass the signal roughly above typical f0 range to isolate residual-ish energy
    try:
        from scipy.signal import butter, sosfilt
        sos = butter(4, 3000, btype="high", fs=sr, output="sos")
        residual = sosfilt(sos, audio)
    except Exception:
        residual = audio

    n_frames = 1 + max(0, (len(residual) - frame_length) // hop_length)
    energy = np.zeros(n_frames, dtype=np.float64)
    times = np.arange(n_frames) * hop_length / sr

    for i in range(n_frames):
        start = i * hop_length
        frame = residual[start : start + frame_length]
        if len(frame) == 0:
            break
        energy[i] = np.sqrt(np.mean(frame ** 2) + 1e-12)

    return times, energy


def extract_residual(
    audio: np.ndarray,
    sr: int,
    f0: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """
    Placeholder residual extraction.
    In a full system this would be the signal after subtracting a harmonic reconstruction.
    For the demo we return a high-passed version of the input.
    """
    try:
        from scipy.signal import butter, sosfilt
        sos = butter(4, 2500, btype="high", fs=sr, output="sos")
        return sosfilt(sos, audio).astype(np.float32)
    except Exception:
        return audio.astype(np.float32)
