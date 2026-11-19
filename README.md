# Sogitech Codec

**Hybrid Analysis–Resynthesis / Parameter / Neural Codec built on the Sogitec OpticLink Hybrid engine**

This repository turns the original [Sogitech](https://github.com/ljrseals-dev/Sogitech) Cmajor instrument (Sogitec 4X-style 32-harmonic additive + Behringer-style VA + PowerSag + SOL-DPKVA dual-pole filter) into a full **codec** family:

1. **Pure DSP Analysis–Resynthesis Codec** – classic analysis front-end drives the hybrid engine
2. **Parameter / Feature Codec** – low-bitrate control stream of the exact parameters the synth already exposes
3. **Hybrid Neural Codec** – small neural controller (or residual network) that predicts parameters or residual; DSP remains the real-time decoder
4. **Brute Genius Search Space** – exhaustive combinatorial search over analysis methods, quantisation, residual strategies, and neural variants

Designed for high sample rates (≥ 96 kHz), experimental / Amorph-style sound design, and offline or real-time use.

---

## Quick Start

```bash
# Clone / unzip
cd sogitech-codec

# Python side (analysis + neural + Brute Genius)
python -m venv .venv
source .venv/bin/activate   # or Windows equivalent
pip install -e ".[dev]"

# Run pure DSP analysis-resynthesis demo
python scripts/demo_dsp_codec.py --input examples/demo.wav --out out_dsp.wav

# Run parameter codec (encode → decode)
python scripts/demo_param_codec.py --input examples/demo.wav --out out_param.wav

# Neural controller inference (weights optional / placeholder)
python scripts/demo_neural_codec.py --input examples/demo.wav --out out_neural.wav

# Brute Genius plan (dry-run cost estimate)
python -m python.brute.plan --space spaces/sogitech_codec_space.json
```

Cmajor decoder lives in `cmajor/`. Open any `.cmaj` file in Cmajor Player or the VS Code extension.

---

## Repository Layout

```
sogitech-codec/
├── README.md
├── LICENSE
├── pyproject.toml
├── cmajor/
│   ├── sogitech_codec_processor.cmaj      # main real-time decoder / processor
│   ├── sogitech_opticlink_hybrid.cmaj     # original hybrid engine (kept for reference)
│   └── sogitech_param_decoder.cmaj       # pure parameter-driven decoder
├── python/
│   ├── analysis/                         # pure-DSP analysis front-ends
│   │   ├── pitch.py
│   │   ├── spectral.py
│   │   ├── envelope.py
│   │   └── residual.py
│   ├── neural/                           # neural controller / residual nets
│   │   ├── controller.py
│   │   ├── residual.py
│   │   └── losses.py
│   └── brute/                            # Brute Genius integration
│       ├── space.py
│       ├── objectives.py
│       └── plan.py
├── weights/                              # optional ONNX / PyTorch checkpoints
├── examples/
├── docs/
│   ├── ARCHITECTURE.md
│   ├── PARAMETER_CODEC.md
│   ├── NEURAL_CODEC.md
│   └── BRUTE_GENIUS.md
└── scripts/
    ├── demo_dsp_codec.py
    ├── demo_param_codec.py
    ├── demo_neural_codec.py
    └── export_cmajor_params.py
```

---

## 1. Pure DSP Analysis–Resynthesis Codec

Analysis extracts the same controls the original instrument already understands:

- f0 / MIDI note + velocity / pressure proxies
- 32-harmonic amplitude envelope (or simplified additiveBlend + levels)
- spectral centroid / tilt → filter cutoffs
- RMS / load estimate → PowerSag
- noise residual energy

These drive `sogitech_codec_processor.cmaj` (or the lighter `sogitech_param_decoder.cmaj`).  
Fully deterministic, no weights, high sample-rate friendly.

## 2. Parameter / Feature Codec

The exposed parameter vector itself becomes the low-bitrate code:

```
osc1Wave, osc2Wave, osc1Level, osc2Level, subLevel, noiseLevel,
additiveBlend, powerSagDepth, filterLP, filterHP, filterRes,
keyTrack, velMod, glideTime, + optional residual energy
```

Encode side quantises / packs the vector (and optional residual).  
Decode side is pure Cmajor. Extremely low bitrate, morphable, ideal for live control or storage of “Amorph” gestures.

## 3. Hybrid Neural Codec

- **Controller** – small network maps audio features (or raw waveform) → parameter vector that drives the Cmajor decoder.
- **Residual** – optional network that predicts a residual signal mixed after the DSP chain.
- Weights live in `weights/` (ONNX preferred for deployment). Training code and loss functions are under `python/neural/`.

The DSP engine stays the authoritative real-time decoder; the neural parts only supply controls or a residual.

## 4. Brute Genius Search Space

See `docs/BRUTE_GENIUS.md` and `python/brute/`.

Search dimensions include:

- pitch tracker family
- envelope / harmonic estimator
- residual strategy
- quantisation bit allocation
- neural architecture variants (if enabled)
- objective combinations (reconstruction, spectral, perceptual, PowerSag fidelity, latency, CPU)

Brute Genius enumerates the Cartesian product, scores every combination, and returns a ranked list. Use `brute_genius_plan` first; respect cost ceilings.

---

## Origins & License

- Original hybrid engine derived from [ljrseals-dev/Sogitech](https://github.com/ljrseals-dev/Sogitech) (itself combining Sogitec 4X / IRCAM additive heritage + Behringer-style VA + Airwindows-inspired PowerSag + SOL-DPKVA filter).
- MIT License — free to use, modify, and distribute.

**Created with Grok assistance — November 2026**
