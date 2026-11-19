#!/usr/bin/env python3
"""
Pure DSP analysis–resynthesis demo for the Sogitech Codec.

Runs analysis → builds a time-varying parameter stream → renders through
the offline reference engine → writes real audio.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    import soundfile as sf
except ImportError:
    from scipy.io import wavfile
    def _read(path):
        sr, data = wavfile.read(path)
        if data.dtype.kind in "iu":
            data = data.astype(np.float64) / np.iinfo(data.dtype).max
        return data, sr
    def _write(path, data, sr):
        wavfile.write(path, sr, (np.clip(data, -1, 1) * 32767).astype(np.int16))
else:
    def _read(path):
        return sf.read(path, always_2d=False)
    def _write(path, data, sr):
        sf.write(path, data, sr)

from python.analysis import (
    estimate_f0, midi_from_f0, spectral_features,
    estimate_filter_params, rms_envelope, residual_energy,
)
from python.render.offline_engine import render


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="examples/demo.wav")
    parser.add_argument("--out", type=str, default="out_dsp.wav")
    parser.add_argument("--sr", type=int, default=48000)
    args = parser.parse_args()

    audio, sr = _read(args.input)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float64)

    print("Estimating f0 …")
    times, f0 = estimate_f0(audio, sr)
    midi = midi_from_f0(f0)

    print("Spectral + envelopes …")
    feats = spectral_features(audio, sr)
    filt = estimate_filter_params(feats, sr)
    t_rms, rms = rms_envelope(audio, sr)
    rms_a = np.interp(times, t_rms, rms)
    t_res, res_e = residual_energy(audio, sr)
    res_a = np.interp(times, t_res, res_e)

    n = len(audio)
    params = {
        "midiNote": midi,
        "velocity": np.clip(rms_a * 3.0, 0.1, 1.0),
        "filterLP": filt["filterLP"],
        "filterHP": filt["filterHP"],
        "filterRes": filt["filterRes"],
        "additiveBlend": np.clip(0.2 + 0.5 * (1.0 - filt["filterRes"]), 0.0, 1.0),
        "powerSagDepth": np.clip(rms_a * 1.5, 0.1, 0.6),
        "noiseLevel": np.clip(res_a * 2.0, 0.0, 0.35),
        "residualGain": np.full_like(rms_a, 0.08),
        "dryWet": np.full_like(rms_a, 0.92),
        "osc1Level": np.full_like(rms_a, 0.75),
        "osc2Level": np.full_like(rms_a, 0.65),
        "subLevel": np.full_like(rms_a, 0.4),
    }

    print("Rendering offline engine (time-varying params) …")
    synth = render(n, sr=float(sr), params=params, residual=audio * 0.04)

    _write(args.out, synth, sr)
    print(f"Wrote {args.out}  (RMS={float(np.sqrt(np.mean(synth**2))):.5f})")

    csv_path = Path(args.out).with_suffix(".params.csv")
    with open(csv_path, "w") as f:
        f.write("time,midi,f0,filterLP,filterHP,filterRes,rms\n")
        for i in range(len(times)):
            f.write(
                f"{times[i]:.5f},{midi[i]:.3f},{f0[i]:.2f},"
                f"{filt['filterLP'][i]:.4f},{filt['filterHP'][i]:.4f},"
                f"{filt['filterRes'][i]:.4f},{rms_a[i]:.5f}\n"
            )
    print(f"Wrote parameter stream {csv_path}")


if __name__ == "__main__":
    main()
