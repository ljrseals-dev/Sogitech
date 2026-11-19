#!/usr/bin/env python3
"""Train the optional parameter controller from a directory of WAV files.

This is supervised pseudo-label training: DSP analysis produces the target
control vector, while the network learns a compact mapping from global audio
features to those controls. It does not claim to learn a lossless codec.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch import nn

from python.analysis import estimate_f0, midi_from_f0, spectral_features, rms_envelope
from python.neural.controller import PARAM_NAMES, PARAM_RANGES, ParameterController

def analyse(path: Path):
    audio, sr = sf.read(path, always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if audio.size < 2048 or not np.all(np.isfinite(audio)):
        raise ValueError(f"{path}: audio must contain at least 2048 finite samples")
    _, f0 = estimate_f0(audio, sr)
    midi = midi_from_f0(f0)
    spectral = spectral_features(audio, sr)
    _, rms = rms_envelope(audio, sr)
    pitch = midi[f0 > 0] if np.any(f0 > 0) else np.array([60.0])
    centroid = np.asarray(spectral["centroid"])
    flatness = np.asarray(spectral["flatness"])
    features = np.zeros(64, dtype=np.float32)
    features[:8] = [
        float(np.mean(pitch)), float(np.std(pitch)),
        float(np.mean(f0[f0 > 0])) if np.any(f0 > 0) else 0.0,
        float(np.mean(centroid)), float(np.std(centroid)),
        float(np.mean(flatness)), float(np.mean(rms)), float(np.std(rms)),
    ]
    rms_mean = float(np.clip(np.mean(rms), 0.0, 1.0))
    centroid_mean = float(np.clip(np.mean(centroid) / (sr * 0.5), 0.0, 1.0))
    target = {
        "osc1Wave": 1.0, "osc2Wave": 2.0,
        "osc1Level": 0.72, "osc2Level": 0.42, "subLevel": 0.25,
        "noiseLevel": float(np.clip(np.mean(flatness) * 0.08, 0.0, 0.2)),
        "additiveBlend": float(np.clip(0.15 + 0.6 * (1.0 - centroid_mean), 0, 1)),
        "powerSagDepth": float(np.clip(0.1 + rms_mean, 0, 1)),
        "filterLP": float(np.clip(0.25 + 0.7 * centroid_mean, 0, 1)),
        "filterHP": 0.04, "filterRes": 0.2, "keyTrack": 0.5, "velMod": 0.5,
        "glideTime": 0.03, "midiNote": float(np.clip(np.mean(pitch), 0, 127)),
        "velocity": float(np.clip(0.3 + rms_mean, 0, 1)), "pressure": 0.0,
        "residualGain": 0.0,
    }
    normalised = [(target[k] - PARAM_RANGES[k][0]) / (PARAM_RANGES[k][1] - PARAM_RANGES[k][0])
                  for k in PARAM_NAMES]
    return features, np.clip(normalised, 0.0, 1.0).astype(np.float32)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Directory containing WAV training files")
    parser.add_argument("--out", default="weights/controller.pt")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.epochs < 1 or args.lr <= 0:
        parser.error("--epochs and --lr must be positive")
    files = sorted(Path(args.data).glob("*.wav"))
    if len(files) < 2:
        parser.error("--data must contain at least two WAV files")
    pairs = [analyse(path) for path in files]
    x = torch.tensor(np.stack([p[0] for p in pairs]), dtype=torch.float32)
    y = torch.tensor(np.stack([p[1] for p in pairs]), dtype=torch.float32)
    torch.manual_seed(args.seed)
    model = ParameterController()
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    model.train()
    for epoch in range(args.epochs):
        optimiser.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        optimiser.step()
        if (epoch + 1) % max(1, args.epochs // 10) == 0:
            print(f"epoch {epoch + 1:4d}/{args.epochs} loss={loss.item():.6f}")
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "input_dim": 64,
                "parameter_names": PARAM_NAMES, "training_files": len(files),
                "label_method": "deterministic DSP pseudo-labels"}, output)
    print(f"Saved {output} (trained on {len(files)} WAV files)")

if __name__ == "__main__":
    main()
