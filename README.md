# Sogitec OpticLink Hybrid

**Hybrid DSP Engine for Amorph / Experimental Artists**

A complete, self-contained Cmajor instrument that combines:

- **Sogitec 4X** (1981 IRCAM, Giuseppe Di Giugno) — 32-harmonic additive partial synthesis
- **Behringer Vintage**-style virtual analog (2 oscillators + sub + noise, hard sync, PWM, selectable waveforms)
- **Airwindows-inspired PowerSag** — dynamic voltage sag / breathing analog character under load
- **SOL-DPKVA** (Sogitec OpticLink DualPole KeyVel Adaptive Filter) — pristine fiber-optic quality double-pole (12 dB/oct) High-Pass + Low-Pass filter with real-time MIDI note + velocity/pressure automation

Designed for high sample rates (96 kHz and above) with transparent digital clarity while retaining organic, responsive behavior.

---

## Features

- Two main oscillators (sine / saw / square / triangle) with hard sync and PWM
- Sub-oscillator + noise generator with color control
- 32-harmonic additive layer (Sogitec / IRCAM style)
- Dual-pole State-Variable Filter (HP + LP) with key tracking and velocity/pressure response
- PowerSag for load-dependent analog-style compression and breathing
- Exposed parameters for live performance and sound design
- Ready for Cmajor Player, VS Code Cmajor extension, or export to plugins / WebAudio

---

## File Structure

```
sogitec-opticlink-hybrid/
├── README.md
├── sogitec-opticlink-hybrid.cmaj     ← Main processor (start here)
└── LICENSE
```

---

## How to Use

### Option 1: Cmajor Player / VS Code
1. Open `sogitec-opticlink-hybrid.cmaj` in Cmajor Player or VS Code with the Cmajor extension.
2. Play and tweak the input parameters live.
3. Connect real MIDI note + velocity for full KeyVel filter response.

### Option 2: Export
You can compile this processor to:
- VST3 / AU / AAX (via Cmajor toolchain)
- WebAudio / WASM
- Standalone

---

## Parameter Overview

| Parameter          | Description                                      | Range     |
|--------------------|--------------------------------------------------|-----------|
| osc1Wave / osc2Wave| Waveform select (0=sine, 1=saw, 2=square, 3=tri)| 0–3      |
| osc1Level / osc2Level | Oscillator levels                             | 0–1      |
| subLevel           | Sub-oscillator level                             | 0–1      |
| noiseLevel         | Noise amount                                     | 0–1      |
| additiveBlend      | Mix between VA and 32-harmonic additive          | 0–1      |
| powerSagDepth      | Amount of voltage sag / breathing                | 0–1      |
| filterLP / filterHP| Base Low-Pass and High-Pass cutoffs              | 0–1      |
| filterRes          | Resonance                                        | 0–1      |
| keyTrack           | Key tracking strength for the filter             | 0–1      |
| velMod             | Velocity / pressure modulation depth             | 0–1      |

---

## Origins

This hybrid was created by combining code from:

- https://github.com/TheRealla/sogitec4x-vintage
- https://github.com/TheRealla/sogitec-opticlink-dpkva

It is intended as a clean, artist-ready instrument for experimental and ambient sound design (Amorph-style textures, evolving pads, drones, and responsive spectral filtering).

---

## License

MIT License — free to use, modify, and distribute.

---

**Created with Grok assistance — November 2026**
