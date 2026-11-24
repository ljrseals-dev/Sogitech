#!/usr/bin/env python3
"""Neural parameter-controller demo. Requires a trained checkpoint."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from python.analysis import estimate_f0,midi_from_f0,spectral_features,rms_envelope
from python.neural import load_controller
from python.audio import render_parameter_stream

def make_feature_vector(audio: np.ndarray, sr: int) -> np.ndarray:
    _,f0=estimate_f0(audio,sr); midi=midi_from_f0(f0)
    feats=spectral_features(audio,sr); _,rms=rms_envelope(audio,sr)
    voiced=f0>0
    vec=np.array([
        (float(np.mean(midi[voiced])) / 127.0) if np.any(voiced) else (60.0 / 127.0),
        (float(np.std(midi[voiced])) / 24.0) if np.any(voiced) else 0.0,
        (float(np.mean(f0[voiced])) / 2000.0) if np.any(voiced) else 0.0,
        float(np.mean(feats["centroid"])) / (sr * 0.5),
        float(np.std(feats["centroid"])) / (sr * 0.5),
        float(np.mean(feats["flatness"])),float(np.mean(rms)),float(np.std(rms)),
    ],dtype=np.float32)
    return np.pad(vec,(0,56))

def build_neural_frames(params: dict, f0: np.ndarray) -> list[dict]:
    """Build note frames and gate unvoiced regions instead of synthesising A4."""
    frequencies = np.asarray(f0, dtype=np.float64)
    if frequencies.ndim != 1 or not np.all(np.isfinite(frequencies)) or np.any(frequencies < 0):
        raise ValueError("f0 must be a finite non-negative one-dimensional array")
    base_note = float(params["midiNote"])
    base_velocity = float(params["velocity"])
    if not np.isfinite(base_note) or not 0.0 <= base_note <= 127.0:
        raise ValueError("predicted midiNote must be within [0, 127]")
    if not np.isfinite(base_velocity) or not 0.0 <= base_velocity <= 1.0:
        raise ValueError("predicted velocity must be within [0, 1]")
    notes = midi_from_f0(frequencies)
    frames = []
    for note, voiced in zip(notes, frequencies > 0.0):
        frame = dict(params)
        frame["midiNote"] = float(note) if voiced else base_note
        frame["velocity"] = base_velocity if voiced else 0.0
        # Residual extraction is not implemented in this path; do not leak
        # the original waveform through an unverified learned residual gain.
        frame["residualGain"] = 0.0
        frames.append(frame)
    return frames


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",required=True); parser.add_argument("--out",default="out_neural.wav")
    parser.add_argument("--weights",required=True,help="trained .pt/.pth checkpoint")
    parser.add_argument("--seed",type=int,default=7)
    args=parser.parse_args()
    audio,sr=sf.read(args.input,always_2d=False,dtype="float32")
    if audio.ndim>1: audio=audio.mean(axis=1)
    if audio.size<1 or not np.all(np.isfinite(audio)): parser.error("input audio is empty or non-finite")
    model=load_controller(args.weights)
    params=model.predict_dict(make_feature_vector(audio,sr))
    print("Predicted bounded parameter vector:")
    for key,value in params.items(): print(f"  {key:16s} = {value:.4f}")
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    sidecar=out.with_suffix(".neural_params.json")
    sidecar.write_text(json.dumps(params,indent=2),encoding="utf-8")
    # The network predicts a global timbre vector; retain note/envelope motion
    # from the source analysis so the rendered result follows the input phrase.
    frame_times,f0=estimate_f0(audio,sr)
    frames=build_neural_frames(params,f0)
    rendered=render_parameter_stream(frames,frame_times,len(audio),sr,args.seed,residual=audio)
    sf.write(out,rendered,sr,subtype="PCM_16")
    print(f"Wrote neural resynthesis: {out}")
    print(f"Wrote neural parameter sidecar: {sidecar}")
if __name__=="__main__": main()
