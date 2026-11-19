#!/usr/bin/env python3
"""Parameter codec demo: analyse, quantise, decode and render audible audio."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import soundfile as sf
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
def quantise(value, bits, lo, hi):
    if not 2 <= bits <= 16 or hi <= lo:
        raise ValueError("bits must be 2..16 and range must be increasing")
    return int(np.clip(round((float(value)-lo)/(hi-lo)*((1<<bits)-1)), 0, (1<<bits)-1))
def dequantise(value, bits, lo, hi):
    return lo + (int(value)/((1<<bits)-1))*(hi-lo)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", default="out_param.wav")
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    args=parser.parse_args()
    if not 2 <= args.bits <= 16: parser.error("--bits must be between 2 and 16")
    audio,sr=sf.read(args.input,always_2d=False,dtype="float32")
    if audio.ndim>1: audio=audio.mean(axis=1)
    if audio.size<1 or not np.all(np.isfinite(audio)): parser.error("input audio is empty or non-finite")
    times,f0=estimate_f0(audio,sr); midi=midi_from_f0(f0)
    feats=spectral_features(audio,sr); filt=estimate_filter_params(feats,sr)
    t_rms,rms=rms_envelope(audio,sr); rms_a=np.interp(times,t_rms,rms)
    t_res,res=residual_energy(audio,sr); res_a=np.interp(times,t_res,res)
    stream=[]
    for i in range(len(times)):
        stream.append({
            "osc1Wave":1.0,"osc2Wave":2.0,"osc1Level":0.75,"osc2Level":0.65,
            "subLevel":0.4,"noiseLevel":float(np.clip(res_a[i]*2,0,0.4)),
            "additiveBlend":float(np.clip(0.2+0.5*(1-filt["filterRes"][i]),0,1)),
            "powerSagDepth":float(np.clip(rms_a[i]*1.5,0.1,0.6)),
            "filterLP":float(filt["filterLP"][i]),"filterHP":float(filt["filterHP"][i]),
            "filterRes":float(filt["filterRes"][i]),"keyTrack":0.55,"velMod":0.6,
            "glideTime":0.05,"midiNote":float(midi[i]),
            "velocity":float(np.clip(rms_a[i]*3,0.1,1)),"pressure":0.0,
            "residualGain":float(np.clip(res_a[i]*1.5,0,0.5)),
        })
    packed=[{k:quantise(p[k],args.bits,*RANGES[k]) for k in PARAM_KEYS} for p in stream]
    decoded=[{k:dequantise(frame[k],args.bits,*RANGES[k]) for k in PARAM_KEYS}
             for frame in packed]
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    sidecar=out.with_suffix(".params.json")
    with sidecar.open("w",encoding="utf-8") as f:
        json.dump({"sample_rate":sr,"bits":args.bits,"times":times.tolist(),
                   "frames":packed,"ranges":RANGES},f,indent=2)
    rendered=render_parameter_stream(decoded,times,len(audio),sr,args.seed,residual=audio)
    sf.write(out,rendered,sr,subtype="PCM_16")
    print(f"Wrote audible parameter decode: {out}")
    print(f"Wrote quantised control stream: {sidecar}")
if __name__=="__main__": main()
