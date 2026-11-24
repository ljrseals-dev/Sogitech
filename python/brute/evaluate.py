"""Reproducible local evaluation of the implemented parameter-codec path.

Only quantisation bit depth and output sample rate are varied. Other search
axes remain explicitly unsupported until their implementations are wired in.
This evaluates the Python reference renderer, not the Cmajor real-time patch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, stft

from python.analysis import (
    estimate_f0, midi_from_f0, spectral_features, estimate_filter_params,
    rms_envelope, residual_energy,
)
from python.audio import render_parameter_stream

PARAM_KEYS = [
    "osc1Wave", "osc2Wave", "osc1Level", "osc2Level", "subLevel", "noiseLevel",
    "additiveBlend", "powerSagDepth", "filterLP", "filterHP", "filterRes",
    "keyTrack", "velMod", "glideTime", "midiNote", "velocity", "pressure",
    "residualGain",
]
RANGES = {
    "osc1Wave": (0, 3), "osc2Wave": (0, 3), "osc1Level": (0, 1),
    "osc2Level": (0, 1), "subLevel": (0, 1), "noiseLevel": (0, 1),
    "additiveBlend": (0, 1), "powerSagDepth": (0, 1), "filterLP": (0, 1),
    "filterHP": (0, 1), "filterRes": (0, 1), "keyTrack": (0, 1),
    "velMod": (0, 1), "glideTime": (0, 0.5), "midiNote": (0, 127),
    "velocity": (0, 1), "pressure": (0, 1), "residualGain": (0, 1),
}
SUPPORTED_AXES = {"codec_mode", "pitch_tracker", "envelope_method",
                  "residual_strategy", "filter_mapping", "quantisation_bits",
                  "quantisation_layout", "neural_controller", "objective_primary",
                  "sample_rate", "latency_budget_ms"}
IMPLEMENTED = {
    "codec_mode": "parameter", "pitch_tracker": "autocorr",
    "envelope_method": "rms", "residual_strategy": "none",
    "filter_mapping": "centroid_linear", "quantisation_layout": "uniform",
    "neural_controller": "none", "objective_primary": "multi_res_stft",
    "latency_budget_ms": None,
}


def _quantise(value: float, bits: int, lo: float, hi: float) -> int:
    levels = (1 << bits) - 1
    return int(np.clip(np.rint((float(value) - lo) / (hi - lo) * levels), 0, levels))


def _dequantise(code: int, bits: int, lo: float, hi: float) -> float:
    return lo + (int(code) / ((1 << bits) - 1)) * (hi - lo)


def _mono(audio: np.ndarray) -> np.ndarray:
    data = np.asarray(audio, dtype=np.float64)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if data.ndim != 1 or data.size == 0 or not np.all(np.isfinite(data)):
        raise ValueError("fixture must be non-empty finite mono/stereo audio")
    return data


def _analyse(audio: np.ndarray, sr: int, bits: int, seed: int) -> tuple[np.ndarray, dict]:
    times, f0 = estimate_f0(audio, sr, method="autocorr")
    midi = midi_from_f0(f0)
    features = spectral_features(audio, sr)
    filt = estimate_filter_params(features, sr)
    rms_times, rms = rms_envelope(audio, sr)
    rms_aligned = np.interp(times, rms_times, rms)
    res_times, residual = residual_energy(audio, sr)
    residual_aligned = np.interp(times, res_times, residual)

    frames = []
    for i in range(len(times)):
        frame = {
            "osc1Wave": 1.0, "osc2Wave": 2.0, "osc1Level": 0.75,
            "osc2Level": 0.65, "subLevel": 0.4,
            "noiseLevel": float(np.clip(residual_aligned[i] * 2, 0, 0.4)),
            "additiveBlend": float(np.clip(0.2 + 0.5 * (1 - filt["filterRes"][i]), 0, 1)),
            "powerSagDepth": float(np.clip(rms_aligned[i] * 1.5, 0.1, 0.6)),
            "filterLP": float(filt["filterLP"][i]), "filterHP": float(filt["filterHP"][i]),
            "filterRes": float(filt["filterRes"][i]), "keyTrack": 0.55, "velMod": 0.6,
            "glideTime": 0.05, "midiNote": float(midi[i]),
            "velocity": float(np.clip(rms_aligned[i] * 3, 0.1, 1)),
            "pressure": 0.0, "residualGain": 0.0,
        }
        frames.append(frame)

    packed = [{key: _quantise(frame[key], bits, *RANGES[key]) for key in PARAM_KEYS}
              for frame in frames]
    decoded = [{key: _dequantise(frame[key], bits, *RANGES[key]) for key in PARAM_KEYS}
               for frame in packed]
    output = render_parameter_stream(decoded, times, len(audio), sr, seed, residual=None)
    param_error = float(np.mean([
        abs(frames[i][key] - decoded[i][key])
        for i in range(len(frames)) for key in PARAM_KEYS
    ]))
    return output, {"parameter_l1": param_error, "frame_count": len(frames)}


def _spectral_metrics(reference: np.ndarray, rendered: np.ndarray, sr: int) -> dict[str, float]:
    values = []
    for n_fft in (512, 1024, 2048):
        hop = n_fft // 4
        _, _, a = stft(reference, fs=sr, nperseg=n_fft, noverlap=n_fft-hop,
                       boundary=None, padded=False)
        _, _, b = stft(rendered, fs=sr, nperseg=n_fft, noverlap=n_fft-hop,
                       boundary=None, padded=False)
        width = min(a.shape[1], b.shape[1])
        if width == 0:
            raise ValueError("fixture is too short for spectral evaluation")
        ma, mb = np.abs(a[:, :width]), np.abs(b[:, :width])
        values.append(float(np.mean(np.abs(ma - mb))))
    return {"spectral_l1": values[1], "multi_res_stft": float(np.mean(values))}


def evaluate_candidate(audio: np.ndarray, source_sha256: str, source_group: str,
                       split: str, sr: int, bits: int, seed: int,
                       repetitions: int, latency_budget_ms: int) -> dict[str, Any]:
    runs, metrics = [], []
    for repetition in range(repetitions):
        started = time.perf_counter()
        rendered, detail = _analyse(audio, sr, bits, seed)
        elapsed = time.perf_counter() - started
        if rendered.shape != audio.shape or not np.all(np.isfinite(rendered)):
            raise RuntimeError("correctness gate failed: output shape or samples invalid")
        spectral = _spectral_metrics(audio, rendered, sr)
        residual_in = float(np.sqrt(np.mean(np.square(audio))))
        residual_out = float(np.sqrt(np.mean(np.square(audio - rendered))))
        peak = float(np.max(np.abs(rendered)))
        record = {
            **spectral, "parameter_l1": detail["parameter_l1"],
            "residual_energy_error": abs(residual_in - residual_out),
            "peak_dbfs": 20.0 * math.log10(max(peak, 1e-12)),
            "elapsed_ms": elapsed * 1000.0,
        }
        if not all(math.isfinite(v) for v in record.values()):
            raise RuntimeError("correctness gate failed: non-finite metric")
        runs.append(record)
        metrics.append(detail)

    names = list(runs[0])
    aggregate = {name: {"mean": float(np.mean([r[name] for r in runs])),
                        "stdev": float(np.std([r[name] for r in runs], ddof=0))}
                 for name in names}
    candidate = {"codec_mode": "parameter", "pitch_tracker": "autocorr",
                 "envelope_method": "rms", "residual_strategy": "none",
                 "filter_mapping": "centroid_linear", "quantisation_bits": bits,
                 "quantisation_layout": "uniform", "neural_controller": "none",
                 "objective_primary": "multi_res_stft", "sample_rate": sr,
                 "latency_budget_ms": latency_budget_ms}
    canonical = json.dumps(candidate, sort_keys=True, separators=(",", ":"))
    candidate_id = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    return {
        "schema_version": 1, "candidate_id": candidate_id, "candidate": candidate,
        "source_sha256": source_sha256, "source_group": source_group, "split": split,
        "sample_count": int(audio.size), "seed": seed, "repetitions": repetitions,
        "correctness": "PASS", "metrics": aggregate, "raw_runs": runs,
        "limitations": ["python_reference_renderer_only", "cpu_load_not_measured",
                        "cmajor_parity_not_checked", "single_fixture_run"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="JSONL result path")
    parser.add_argument("--source-group", required=True,
                        help="Stable performer/session/source recording group ID")
    parser.add_argument("--split", required=True, choices=("train", "validation", "test"))
    parser.add_argument("--bits", type=int, choices=(4, 8, 12), default=8)
    parser.add_argument("--sample-rate", type=int, choices=(48000, 96000), default=48000)
    parser.add_argument("--latency-budget-ms", type=int, choices=(5, 20), default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.repetitions < 1 or args.repetitions > 20:
        parser.error("--repetitions must be between 1 and 20")
    raw = args.input.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    audio, input_sr = sf.read(args.input, always_2d=False, dtype="float32")
    audio = _mono(audio)
    if input_sr != args.sample_rate:
        from math import gcd
        divisor = gcd(int(input_sr), int(args.sample_rate))
        audio = resample_poly(audio, args.sample_rate // divisor,
                              input_sr // divisor).astype(np.float64)
    result = evaluate_candidate(audio, digest, args.source_group, args.split,
                                args.sample_rate, args.bits, args.seed, args.repetitions,\n                                args.latency_budget_ms)
    result["input_sample_rate"] = int(input_sr)
    result["evaluation_sample_rate"] = args.sample_rate
    result["tool"] = "sogitech.python.brute.evaluate"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"candidate_id": result["candidate_id"],
                      "correctness": result["correctness"],
                      "result": str(args.output), "metrics": result["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
