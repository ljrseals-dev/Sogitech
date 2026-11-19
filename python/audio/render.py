"""Deterministic offline renderer for Sogitech parameter streams.

This is a portable reference renderer for demos and dataset generation. The
Cmajor processors remain the real-time implementation; this renderer ensures
that parameter and neural demos produce audible, non-placeholder WAV files.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, lfilter, sosfilt

PARAM_DEFAULTS = {
    "osc1Wave": 1.0, "osc2Wave": 2.0, "osc1Level": 0.72,
    "osc2Level": 0.42, "subLevel": 0.25, "noiseLevel": 0.015,
    "additiveBlend": 0.25, "powerSagDepth": 0.2, "filterLP": 0.72,
    "filterHP": 0.04, "filterRes": 0.2, "keyTrack": 0.5,
    "velMod": 0.5, "glideTime": 0.03, "midiNote": 60.0,
    "velocity": 0.8, "pressure": 0.0, "residualGain": 0.0,
}

def _control(frames, times, key, sample_times):
    values = np.asarray([float(frame.get(key, PARAM_DEFAULTS[key])) for frame in frames])
    if not np.all(np.isfinite(values)):
        raise ValueError(f"non-finite parameter: {key}")
    return np.interp(sample_times, times, values)

def _oscillator(phase, shape):
    # Phase is accumulated across the entire render, so wrap it before
    # evaluating discontinuous waveforms. Without this, sawtooth amplitude
    # grows without bound after the first cycle.
    phase = np.remainder(phase, 2.0 * np.pi)
    shape = np.clip(np.rint(shape), 0, 3)
    sine = np.sin(phase)
    saw = (phase / np.pi) - 1.0
    square = np.where(sine >= 0.0, 1.0, -1.0)
    triangle = 1.0 - (2.0 / np.pi) * np.arccos(np.clip(sine, -1.0, 1.0))
    return np.select([shape == 0, shape == 1, shape == 2],
                     [sine, saw, square], default=triangle)

def render_parameter_stream(frames, times, sample_count, sample_rate=48000, seed=7,
                            residual=None):
    """Render parameter frames to mono float32 audio."""
    if not np.isfinite(sample_rate) or sample_rate < 8000 or sample_rate > 192000:
        raise ValueError("sample_rate must be finite and between 8000 and 192000 Hz")
    if not isinstance(sample_count, (int, np.integer)) or sample_count < 1:
        raise ValueError("sample_count must be positive")
    if not frames:
        raise ValueError("at least one parameter frame is required")
    frame_times = np.asarray(times, dtype=np.float64)
    if frame_times.ndim != 1 or len(frame_times) != len(frames):
        raise ValueError("times must contain one timestamp per frame")
    if not np.all(np.isfinite(frame_times)):
        raise ValueError("frame times must be finite")
    if len(frame_times) > 1 and np.any(np.diff(frame_times) <= 0):
        raise ValueError("frame times must be strictly increasing")
    duration = sample_count / float(sample_rate)
    sample_times = np.arange(sample_count, dtype=np.float64) / sample_rate
    if len(frame_times) == 1:
        frame_times = np.array([0.0, max(duration, 1.0 / sample_rate)])
        frames = [frames[0], frames[0]]

    p = {key: _control(frames, frame_times, key, sample_times)
         for key in PARAM_DEFAULTS}
    midi = np.clip(p["midiNote"], 0.0, 127.0)
    frequency = 440.0 * np.power(2.0, (midi - 69.0) / 12.0)
    frequency = np.clip(frequency, 1.0, sample_rate * 0.45)
    phase = np.cumsum((2.0 * np.pi * frequency) / sample_rate)
    phase2 = np.cumsum((2.0 * np.pi * frequency * (1.0 + 0.004)) / sample_rate)
    sub_phase = np.cumsum((np.pi * frequency) / sample_rate)

    va = (_oscillator(phase, p["osc1Wave"]) * np.clip(p["osc1Level"], 0, 1)
          + _oscillator(phase2, p["osc2Wave"]) * np.clip(p["osc2Level"], 0, 1)
          + np.sin(sub_phase) * np.clip(p["subLevel"], 0, 1))
    additive = np.zeros(sample_count, dtype=np.float64)
    for harmonic in range(1, 33):
        additive += np.sin(phase * harmonic) / harmonic
    additive *= 0.14
    blend = np.clip(p["additiveBlend"], 0, 1)
    signal = va * (1.0 - blend) + additive * blend
    rng = np.random.default_rng(seed)
    signal += rng.standard_normal(sample_count) * np.clip(p["noiseLevel"], 0, 0.5)

    # Power-sag envelope, followed by a stable fixed-cutoff Butterworth stage.
    load = lfilter([0.008], [1.0, -0.992], np.abs(signal))
    sag = np.clip(1.0 - np.clip(p["powerSagDepth"], 0, 1) * load * 0.8, 0.35, 1.0)
    signal *= sag * np.clip(p["velocity"], 0, 1) * (0.65 + 0.35 * np.clip(p["pressure"], 0, 1))

    lp = float(np.clip(np.median(p["filterLP"]) * 18000.0, 80.0, sample_rate * 0.45))
    hp = float(np.clip(np.median(p["filterHP"]) * 120.0, 10.0, min(7000.0, lp * 0.8)))
    if lp < sample_rate * 0.49:
        signal = sosfilt(butter(2, lp, fs=sample_rate, output="sos"), signal)
    if hp > 10.0 and hp < lp:
        signal = sosfilt(butter(1, hp, btype="highpass", fs=sample_rate, output="sos"), signal)

    if residual is not None:
        dry = np.asarray(residual, dtype=np.float64).reshape(-1)
        if len(dry) != sample_count:
            raise ValueError("residual length must match sample_count")
        if not np.all(np.isfinite(dry)):
            raise ValueError("residual must contain only finite samples")
        signal += dry * np.clip(p["residualGain"], 0, 1)

    signal = np.tanh(signal * 0.8)
    peak = float(np.max(np.abs(signal)))
    if peak > 0.98:
        signal *= 0.98 / peak
    return signal.astype(np.float32)
