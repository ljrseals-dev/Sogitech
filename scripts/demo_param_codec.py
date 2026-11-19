#!/usr/bin/env python3
"""
Parameter / feature codec demo.

Extracts the low-bitrate parameter vector, quantises it, and writes
both the packed representation and a reconstructed control stream.
"""

from __future__ import annotations

import argparse
import json
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


PARAM_KEYS = [
    "osc1Wave", "osc2Wave", "osc1Level", "osc2Level",
    "subLevel", "noiseLevel", "additiveBlend", "powerSagDepth",
    "filterLP", "filterHP", "filterRes", "keyTrack", "velMod",
    "glideTime", "midiNote", "velocity", "pressure", "residualGain",
]


def quantise(value: float, bits: int, lo: float, hi: float) -> int:
    q = int(round((value - lo) / (hi - lo) * (2 ** bits - 1)))
    return max(0, min(2 ** bits - 1, q))


def dequantise(q: int, bits: int, lo: float, hi: float) -> float:
    return lo + (q / (2 ** bits - 1)) * (hi - lo)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--out", type=str, default="out_param.wav")
    parser.add_argument("--bits", type=int, default=8)
    args = parser.parse_args()

    audio, sr = sf.read(args.input, always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    times, f0 = estimate_f0(audio, sr)
    midi = midi_from_f0(f0)
    feats = spectral_features(audio, sr)
    filt = estimate_filter_params(feats, sr)
    t_rms, rms = rms_envelope(audio, sr)
    rms_a = np.interp(times, t_rms, rms)
    t_res, res = residual_energy(audio, sr)
    res_a = np.interp(times, t_res, res)

    # Build a time-varying parameter stream (simplified static defaults + driven fields)
    stream = []
    for i in range(len(times)):
        p = {
            "osc1Wave": 1.0,
            "osc2Wave": 2.0,
            "osc1Level": 0.75,
            "osc2Level": 0.65,
            "subLevel": 0.4,
            "noiseLevel": float(np.clip(res_a[i] * 2.0, 0.0, 0.4)),
            "additiveBlend": float(np.clip(0.2 + 0.5 * (1.0 - filt["filterRes"][i]), 0.0, 1.0)),
            "powerSagDepth": float(np.clip(rms_a[i] * 1.5, 0.1, 0.6)),
            "filterLP": float(filt["filterLP"][i]),
            "filterHP": float(filt["filterHP"][i]),
            "filterRes": float(filt["filterRes"][i]),
            "keyTrack": 0.55,
            "velMod": 0.6,
            "glideTime": 0.05,
            "midiNote": float(midi[i]),
            "velocity": float(np.clip(rms_a[i] * 3.0, 0.1, 1.0)),
            "pressure": 0.0,
            "residualGain": float(np.clip(res_a[i] * 1.5, 0.0, 0.5)),
        }
        stream.append(p)

    # Quantise and pack
    packed = []
    ranges = {
        "osc1Wave": (0, 3), "osc2Wave": (0, 3),
        "osc1Level": (0, 1), "osc2Level": (0, 1),
        "subLevel": (0, 1), "noiseLevel": (0, 1),
        "additiveBlend": (0, 1), "powerSagDepth": (0, 1),
        "filterLP": (0, 1), "filterHP": (0, 1), "filterRes": (0, 1),
        "keyTrack": (0, 1), "velMod": (0, 1), "glideTime": (0, 0.5),
        "midiNote": (24, 96), "velocity": (0, 1), "pressure": (0, 1),
        "residualGain": (0, 1),
    }
    for p in stream:
        frame = {}
        for k in PARAM_KEYS:
            lo, hi = ranges[k]
            frame[k] = quantise(p[k], args.bits, lo, hi)
        packed.append(frame)

    out_json = Path(args.out).with_suffix(".params.json")
    with open(out_json, "w") as f:
        json.dump({"sr": sr, "bits": args.bits, "times": times.tolist(), "frames": packed}, f)
    print(f"Wrote packed parameter codec stream → {out_json}")

    # For the demo we also write a silent placeholder wav so the pipeline is complete.
    # Real audio comes from the Cmajor decoder driven by these parameters.
    sf.write(args.out, np.zeros(len(audio), dtype=np.float32), sr)
    print(f"Wrote placeholder {args.out} (drive Cmajor decoder with the JSON for real audio)")


if __name__ == "__main__":
    main()
