# Sogitech Codec Architecture

## Overview

The codec is built around a single real-time decoder (the hybrid engine originally written in Cmajor) that can be driven in three complementary ways:

1. **Pure DSP analysis front-end** → continuous parameter stream → Cmajor decoder
2. **Parameter / feature codec** → quantised low-bitrate stream → Cmajor decoder
3. **Neural controller / residual** → predicted parameters or residual → Cmajor decoder

In all cases the Cmajor processor remains the authoritative, deterministic, high-sample-rate sound engine.

## Signal Flow (generic)

```
Audio In
   │
   ├─► Analysis (pitch, spectral, envelope, residual)
   │         │
   │         ▼
   │   Parameter Vector  ──────────────────────────────┐
   │         │                                          │
   │         ├─ (optional) Quantisation / packing       │
   │         │                                          │
   │         ├─ (optional) Neural controller refinement │
   │         │                                          │
   │         ▼                                          │
   │   Cmajor Decoder (sogitech_param_decoder /         │
   │                   sogitech_codec_processor)        │
   │         │                                          │
   │         ├─ Additive 32-harmonic layer              │
   │         ├─ VA oscillators + sub + noise            │
   │         ├─ PowerSag                                │
   │         └─ SOL-DPKVA dual-pole filter              │
   │         │                                          │
   │         ▼                                          │
   └─► Residual mix ◄───────────────────────────────────┘
         │
         ▼
      Audio Out
```

## Key Design Decisions

- **Decoder stays DSP** – keeps latency, CPU, and determinism predictable.
- **Analysis is replaceable** – any pitch tracker, envelope, or residual method can be swapped; Brute Genius explores the combinations.
- **Neural parts are optional** – they only supply controls or a residual; they never replace the real-time oscillator/filter core.
- **High sample rates** – the original hybrid was designed for ≥ 96 kHz; the codec inherits that preference.

## File Roles

| Component | Location | Role |
|-----------|----------|------|
| Original hybrid | `cmajor/sogitech_opticlink_hybrid.cmaj` | Reference generator |
| Parameter decoder | `cmajor/sogitech_param_decoder.cmaj` | Pure control-driven decoder |
| Full codec processor | `cmajor/sogitech_codec_processor.cmaj` | Decoder + residual / dry-wet |
| Analysis | `python/analysis/` | f0, spectral, envelope, residual |
| Neural | `python/neural/` | Controller + residual nets |
| Brute Genius | `python/brute/` | Search space + objectives |
