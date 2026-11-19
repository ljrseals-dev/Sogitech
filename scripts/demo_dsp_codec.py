#!/usr/bin/env python3
"""Pure DSP analysis–resynthesis demo for the Sogitech Codec."""
from __future__ import annotations
import argparse
from math import gcd
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from python.analysis import (
    estimate_f0, midi_from_f0, spectral_features, estimate_filter_params,
    rms_envelope,
)

def simple_resynth(times, midi, filter_params, rms, sr, duration):
    """Lightweight additive reference render (the Cmajor patch is authoritative)."""
    n=max(1,int(round(duration*sr)))
    t=np.arange(n,dtype=np.float64)/sr
    midi_i=np.interp(t,times,midi)
    rms_i=np.interp(t,times,rms)
    lp_i=np.interp(t,filter_params["times"],filter_params["filterLP"])
    f0=440.0*np.power(2.0,(np.clip(midi_i,0,127)-69.0)/12.0)
    phase=np.cumsum(2.0*np.pi*np.clip(f0,1.0,sr*0.45)/sr)
    signal=np.zeros(n,dtype=np.float64)
    for harmonic in range(1,9):
        signal += np.sin(phase*harmonic)/harmonic
    signal *= 0.15*np.clip(rms_i,0,1)*(0.4+0.6*np.clip(lp_i,0,1))
    return np.tanh(signal*1.5).astype(np.float32)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",required=True)
    parser.add_argument("--out",default="out_dsp.wav")
    parser.add_argument("--sr",type=int,default=48000)
    args=parser.parse_args()
    if not 8000 <= args.sr <= 192000:
        parser.error("--sr must be between 8000 and 192000 Hz")
    audio,sr=sf.read(args.input,always_2d=False,dtype="float32")
    if audio.ndim>1: audio=audio.mean(axis=1)
    if audio.size<1 or not np.all(np.isfinite(audio)):
        parser.error("input audio is empty or contains non-finite samples")
    if sr != args.sr:
        divisor=gcd(int(sr),int(args.sr))
        audio=resample_poly(audio,args.sr//divisor,sr//divisor).astype(np.float32)
        sr=args.sr
    print("Estimating f0 …")
    times,f0=estimate_f0(audio,sr)
    midi=midi_from_f0(f0)
    print("Extracting spectral controls …")
    features=spectral_features(audio,sr)
    filt=estimate_filter_params(features,sr)
    print("Estimating envelope …")
    rms_times,rms=rms_envelope(audio,sr)
    rms_aligned=np.interp(times,rms_times,rms)
    print("Rendering audio …")
    synth=simple_resynth(times,midi,filt,rms_aligned,sr,len(audio)/sr)
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    sf.write(out,synth,sr,subtype="PCM_16")
    csv_path=out.with_suffix(".params.csv")
    with csv_path.open("w",encoding="utf-8") as stream:
        stream.write("time,midi,f0,filterLP,filterHP,filterRes,rms\n")
        for i in range(len(times)):
            stream.write(
                f"{times[i]:.5f},{midi[i]:.3f},{f0[i]:.2f},"
                f"{filt['filterLP'][i]:.4f},{filt['filterHP'][i]:.4f},"
                f"{filt['filterRes'][i]:.4f},{rms_aligned[i]:.5f}\n"
            )
    print(f"Wrote audio: {out}")
    print(f"Wrote parameter stream: {csv_path}")
if __name__=="__main__": main()
