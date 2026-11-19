"""Spectral analysis → filter parameters and additive cues for Sogitech Codec."""

from __future__ import annotations

import numpy as np
from typing import Dict


def spectral_features(
    audio: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> Dict[str, np.ndarray]:
    """
    Compute basic spectral features used to drive SOL-DPKVA and additiveBlend.
    Returns dict with keys: times, centroid, flatness, flux, bandwidth.
    """
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr <= 0 or n_fft < 2 or hop_length < 1:
        raise ValueError("invalid sample rate or STFT frame geometry")
    if audio.size == 0 or not np.all(np.isfinite(audio)):
        raise ValueError("audio must contain finite samples")
    if audio.size < n_fft:
        audio = np.pad(audio, (0, n_fft - audio.size))

    try:
        import librosa
        S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
        times = librosa.times_like(S, sr=sr, hop_length=hop_length)
        centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
        flatness = librosa.feature.spectral_flatness(S=S)[0]
        flux = librosa.onset.onset_strength(S=S, sr=sr)
        bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
    except Exception:
        # Minimal fallback
        n_frames = 1 + max(0, (len(audio) - n_fft) // hop_length)
        times = np.arange(n_frames) * hop_length / sr
        centroid = np.full(n_frames, 2000.0)
        flatness = np.full(n_frames, 0.1)
        flux = np.zeros(n_frames)
        bandwidth = np.full(n_frames, 1500.0)

    return {
        "times": times,
        "centroid": centroid,
        "flatness": flatness,
        "flux": flux,
        "bandwidth": bandwidth,
    }


def estimate_filter_params(
    features: Dict[str, np.ndarray],
    sr: int = 96000,
) -> Dict[str, np.ndarray]:
    """
    Map spectral features onto the normalised filter controls of the hybrid engine.
    filterLP / filterHP are in [0, 1] as expected by the Cmajor processor.
    """
    centroid = features["centroid"]
    flatness = features["flatness"]

    # Map centroid (Hz) roughly onto LP cutoff control
    # 180 Hz → ~0.0, 18 kHz → ~1.0
    lp = np.clip((centroid - 180.0) / 18000.0, 0.05, 0.98)

    # High-pass rises a little when the spectrum is brighter / noisier
    hp = np.clip(0.02 + 0.25 * flatness + 0.1 * (centroid / 8000.0), 0.01, 0.4)

    # Resonance gently follows flatness (noisier → slightly less resonant)
    res = np.clip(0.35 - 0.25 * flatness, 0.05, 0.7)

    return {
        "filterLP": lp.astype(np.float32),
        "filterHP": hp.astype(np.float32),
        "filterRes": res.astype(np.float32),
        "times": features["times"],
    }
