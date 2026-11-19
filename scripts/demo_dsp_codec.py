#!/usr/bin/env python3
"""
Pure DSP analysis–resynthesis demo for the Sogitech Codec.

Reads a wav, runs the analysis front-end, writes a parameter stream
and a simple resynthesised approximation (Python-side oscillator bank
for demo purposes). The authoritative real-time decoder is the Cmajor
processor; this script shows the analysis → control flow.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from python.analysis import (
    estimate_f0,
    midi_from_f0,
    spectral_features,
    estimate_filter_params,
    rms_envelope,
    residual_energy,
)


def simple_resynth(
    times: np.ndarray,
    midi: np.ndarray,
    filter_params: dict,
    rms: np.ndarray,
    sr: int,
    duration: float,
) -> np.ndarray:
    """Very lightweight additive + filtered noise resynthesis for demo."""
    n = int(duration * sr)
    t = np.arange(n) / sr
    out = np.zeros(n, dtype=np.float64)

    # Interpolate controls
    midi_i = np.interp(t, times, midi)
    rms_i = np.interp(t, times, rms)
    lp_i = np.interp(t, filter_params["times"], filter_params["filterLP"])

    phase = 0.0
    for i in range(n):
        f0 = 440.0 * (2.0 ** ((midi_i[i] - 69.0) / 12.0))
        phase += 2.0 * np.pi * f0 / sr
        # 8-harmonic approximation of the 32-harmonic digitalOsc
        s = 0.0
        for k in range(1, 9):
            s += np.sin(phase * k) / k
        s *= 0.15
        # Simple one-pole low-pass approximation driven by filterLP
        # (not the full dual-pole SVF – that lives in Cmajor)
        out[i] = s * rms_i[i] * (0.4 + 0.6 * lp_i[i])

    # Soft clip
    out = np.tanh(out * 1.5)
    return out.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--out", type=str, default="out_dsp.wav")
    parser.add_argument("--sr", type=int, default=48000)
    args = parser.parse_args()

    audio, sr = sf.read(args.input, always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != args.sr:
        # naive resample for demo
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=args.sr)
            sr = args.sr
        except Exception:
            pass

    print("Estimating f0 …")
    times, f0 = estimate_f0(audio, sr)
    midi = midi_from_f0(f0)

    print("Spectral features …")
    feats = spectral_features(audio, sr)
    filt = estimate_filter_params(feats, sr)

    print("Envelopes …")
    t_rms, rms = rms_envelope(audio, sr)
    # Align rms to pitch times
    rms_aligned = np.interp(times, t_rms, rms)

    print("Resynthesising (demo oscillator bank) …")
    duration = len(audio) / sr
    synth = simple_resynth(times, midi, filt, rms_aligned, sr, duration)

    # Match length
    n = min(len(audio), len(synth))
    sf.write(args.out, synth[:n], sr)
    print(f"Wrote {args.out}")

    # Also dump a simple parameter CSV for inspection
    csv_path = Path(args.out).with_suffix(".params.csv")
    with open(csv_path, "w") as f:
        f.write("time,midi,f0,filterLP,filterHP,filterRes,rms\n")
        for i in range(len(times)):
            f.write(
                f"{times[i]:.5f},{midi[i]:.3f},{f0[i]:.2f},"
                f"{filt['filterLP'][i]:.4f},{filt['filterHP'][i]:.4f},"
                f"{filt['filterRes'][i]:.4f},{rms_aligned[i]:.5f}\n"
            )
    print(f"Wrote parameter stream {csv_path}")


if __name__ == "__main__":
    main()
