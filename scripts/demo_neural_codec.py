#!/usr/bin/env python3
"""
Hybrid neural codec demo.

Uses the ParameterController (random or loaded weights) to predict
the parameter vector from simple features, then writes the predicted
stream. Real audio decoding remains the responsibility of the Cmajor
processor.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from python.analysis import estimate_f0, midi_from_f0, spectral_features, rms_envelope
from python.neural import load_controller, PARAM_NAMES


def make_feature_vector(audio: np.ndarray, sr: int) -> np.ndarray:
    """Build a fixed-size feature vector for the controller (demo)."""
    times, f0 = estimate_f0(audio, sr)
    midi = midi_from_f0(f0)
    feats = spectral_features(audio, sr)
    t_rms, rms = rms_envelope(audio, sr)

    # Aggregate statistics
    vec = np.array([
        np.mean(midi),
        np.std(midi),
        np.mean(f0[f0 > 0]) if np.any(f0 > 0) else 0.0,
        np.mean(feats["centroid"]),
        np.std(feats["centroid"]),
        np.mean(feats["flatness"]),
        np.mean(rms),
        np.std(rms),
    ], dtype=np.float32)

    # Pad / tile to expected controller input dim (64)
    if vec.size < 64:
        vec = np.pad(vec, (0, 64 - vec.size))
    return vec[:64]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--out", type=str, default="out_neural.wav")
    parser.add_argument("--weights", type=str, default=None)
    args = parser.parse_args()

    audio, sr = sf.read(args.input, always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    print("Building features …")
    feat = make_feature_vector(audio, sr)

    print("Loading controller …")
    ctrl = load_controller(args.weights)
    params = ctrl.predict_dict(feat)

    print("Predicted parameter vector:")
    for k, v in params.items():
        print(f"  {k:16s} = {v:.4f}")

    # Write a JSON sidecar that the Cmajor host can consume
    out_json = Path(args.out).with_suffix(".neural_params.json")
    import json
    with open(out_json, "w") as f:
        json.dump(params, f, indent=2)
    print(f"Wrote {out_json}")

    # Placeholder audio
    sf.write(args.out, np.zeros(len(audio), dtype=np.float32), sr)
    print(f"Wrote placeholder {args.out}")


if __name__ == "__main__":
    main()
