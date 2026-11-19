"""Amplitude / harmonic envelope helpers for the Sogitech Codec."""

from __future__ import annotations

import numpy as np
from typing import Tuple


def rms_envelope(
    audio: np.ndarray,
    sr: int,
    frame_length: int = 1024,
    hop_length: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (times, rms)."""
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    n_frames = 1 + max(0, (len(audio) - frame_length) // hop_length)
    rms = np.zeros(n_frames, dtype=np.float64)
    times = np.arange(n_frames) * hop_length / sr

    for i in range(n_frames):
        start = i * hop_length
        frame = audio[start : start + frame_length]
        if len(frame) == 0:
            break
        rms[i] = np.sqrt(np.mean(frame ** 2) + 1e-12)

    return times, rms


def harmonic_envelope(
    audio: np.ndarray,
    sr: int,
    f0: np.ndarray,
    times: np.ndarray,
    n_harmonics: int = 32,
) -> np.ndarray:
    """
    Very lightweight harmonic amplitude estimate.
    Returns an array of shape (n_frames, n_harmonics) with relative amplitudes.
    Used to set additiveBlend and level cues.
    """
    n_frames = len(times)
    amps = np.zeros((n_frames, n_harmonics), dtype=np.float32)

    # Simple magnitude spectrum approach
    try:
        import librosa
        hop = int(round((times[1] - times[0]) * sr)) if n_frames > 1 else 512
        S = np.abs(librosa.stft(audio, n_fft=2048, hop_length=hop))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

        for i, (t, f0_hz) in enumerate(zip(times, f0)):
            if f0_hz <= 0:
                continue
            for k in range(1, n_harmonics + 1):
                target = f0_hz * k
                if target >= sr / 2:
                    break
                idx = np.argmin(np.abs(freqs - target))
                if idx < S.shape[0] and i < S.shape[1]:
                    amps[i, k - 1] = S[idx, min(i, S.shape[1] - 1)]
        # Normalise per frame
        norms = amps.sum(axis=1, keepdims=True) + 1e-8
        amps /= norms
    except Exception:
        # Fallback: decaying harmonic series
        for i, f0_hz in enumerate(f0):
            if f0_hz > 0:
                amps[i] = 1.0 / np.arange(1, n_harmonics + 1)

    return amps
