"""Pitch / f0 estimation for the Sogitech Codec analysis front-end."""

from __future__ import annotations

import numpy as np
from typing import Tuple


def estimate_f0(
    audio: np.ndarray,
    sr: int,
    frame_length: int = 2048,
    hop_length: int = 512,
    fmin: float = 50.0,
    fmax: float = 2000.0,
    method: str = "auto",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple autocorrelation-based f0 estimator (robust enough for the codec demo).
    Returns (times, f0_hz). Unvoiced frames are set to 0.
    """
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr <= 0 or frame_length < 2 or hop_length < 1 or fmin <= 0 or fmax <= fmin:
        raise ValueError("invalid sample rate, frame geometry, or pitch range")
    if method not in ("auto", "autocorr", "librosa_pyin"):
        raise ValueError("method must be auto, autocorr, or librosa_pyin")
    if audio.size == 0 or not np.all(np.isfinite(audio)):
        raise ValueError("audio must contain finite samples")
    if audio.size < frame_length:
        audio = np.pad(audio, (0, frame_length - audio.size))

    # Prefer librosa if available for better quality
    if method in ("auto", "librosa_pyin"):
        try:
            import librosa
            f0, voiced_flag, _ = librosa.pyin(
                audio,
                fmin=fmin,
                fmax=fmax,
                sr=sr,
                frame_length=frame_length,
                hop_length=hop_length,
            )
            times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
            f0 = np.where(voiced_flag, f0, 0.0)
            return times, f0
        except Exception:
            if method == "librosa_pyin":
                raise RuntimeError("librosa pYIN was requested but is unavailable or failed")

    # Fallback pure-numpy autocorrelation
    n_frames = 1 + max(0, (len(audio) - frame_length) // hop_length)
    f0 = np.zeros(n_frames, dtype=np.float64)
    times = np.arange(n_frames) * hop_length / sr

    min_lag = int(sr / fmax)
    max_lag = int(sr / fmin)

    for i in range(n_frames):
        start = i * hop_length
        frame = audio[start : start + frame_length]
        if len(frame) < frame_length:
            break
        frame = frame - frame.mean()
        corr = np.correlate(frame, frame, mode="full")
        corr = corr[len(corr) // 2 :]
        if max_lag >= len(corr):
            continue
        segment = corr[min_lag:max_lag]
        if segment.size == 0:
            continue
        lag = min_lag + int(np.argmax(segment))
        if corr[lag] > 0.3 * corr[0]:
            f0[i] = sr / lag

    return times, f0


def midi_from_f0(f0_hz: np.ndarray, default_midi: float = 69.0) -> np.ndarray:
    """Convert f0 (Hz) to continuous MIDI note numbers. 0 Hz → default_midi."""
    midi = np.full_like(f0_hz, default_midi, dtype=np.float64)
    voiced = f0_hz > 0
    midi[voiced] = 69.0 + 12.0 * np.log2(f0_hz[voiced] / 440.0)
    return midi
