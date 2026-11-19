#!/usr/bin/env python3
"""
Offline validation of the Sogitech Codec signal path.

1. Runs analysis on the demo wav
2. Builds a parameter vector
3. Renders through the Python reference engine
4. Writes audio + a short report

This does not require the Cmajor toolchain and can be used to confirm
that analysis → parameters → audio is coherent before hardware testing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from python.analysis import (
    estimate_f0, midi_from_f0, spectral_features,
    estimate_filter_params, rms_envelope, residual_energy,
)
from python.render.offline_engine import render


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="examples/demo.wav")
    parser.add_argument("--out", default="out_offline_validate.wav")
    parser.add_argument("--sr", type=int, default=48000)
    args = parser.parse_args()

    audio, sr = sf.read(args.input, always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != args.sr:
        try:
            import librosa
            audio = librosa.resample(audio.astype(np.float64), orig_sr=sr, target_sr=args.sr)
            sr = args.sr
        except Exception:
            pass

    print("=== Sogitech Codec – Offline Validation ===")
    print(f"Input: {args.input}  ({len(audio)/sr:.2f}s @ {sr} Hz)")

    times, f0 = estimate_f0(audio, sr)
    midi = midi_from_f0(f0)
    feats = spectral_features(audio, sr)
    filt = estimate_filter_params(feats, sr)
    t_rms, rms = rms_envelope(audio, sr)
    rms_a = np.interp(times, t_rms, rms)
    t_res, res_e = residual_energy(audio, sr)
    res_a = np.interp(times, t_res, res_e)

    # Use median / mean of the analysis as a static parameter set for the offline render
    params = {
        "midiNote": float(np.median(midi[midi > 0])) if np.any(midi > 0) else 69.0,
        "velocity": float(np.clip(np.mean(rms_a) * 3.0, 0.2, 1.0)),
        "filterLP": float(np.median(filt["filterLP"])),
        "filterHP": float(np.median(filt["filterHP"])),
        "filterRes": float(np.median(filt["filterRes"])),
        "additiveBlend": 0.35,
        "powerSagDepth": float(np.clip(np.mean(rms_a) * 1.2, 0.1, 0.5)),
        "noiseLevel": float(np.clip(np.mean(res_a) * 1.5, 0.05, 0.3)),
        "residualGain": 0.1,
        "dryWet": 0.9,
    }

    print("\nDerived parameters:")
    for k, v in params.items():
        print(f"  {k:16s} = {v:.4f}")

    print("\nRendering offline engine …")
    synth = render(len(audio), sr=float(sr), params=params, residual=audio * 0.05)

    sf.write(args.out, synth, sr)
    print(f"Wrote {args.out}")

    # Simple metrics
    rms_in = np.sqrt(np.mean(audio ** 2) + 1e-12)
    rms_out = np.sqrt(np.mean(synth ** 2) + 1e-12)
    print(f"\nRMS in  : {rms_in:.5f}")
    print(f"RMS out : {rms_out:.5f}")
    print("Offline path completed successfully.")
    print("\nNext (on a machine with Cmajor toolchain):")
    print("  cmaj play cmajor/patches/SogitechParamDecoder/SogitechParamDecoder.cmajorpatch")
    print("  cmaj play cmajor/patches/SogitechCodecProcessor/SogitechCodecProcessor.cmajorpatch")


if __name__ == "__main__":
    main()
