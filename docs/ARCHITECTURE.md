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

## Vintage home-studio topology and preservation rules

The original OpticLink reference is a vintage-inspired, constrained signal
path. Its historical context includes the 1981 IRCAM Sogitec 4X concept, a
vintage VA oscillator layer, the additive harmonic layer, PowerSag, and the
SOL-DPKVA dual-pole filter. The home-studio wiring and available signal paths
are part of the design provenance, not evidence that the instrument was
intended to be a modern polyphonic workstation.

Accordingly:

- Preserve the original monophonic voice policy unless the owner explicitly
  authorises a separate polyphonic instrument variant. Do not treat the lack
  of a voice allocator as an accidental omission.
- Preserve the documented order and character of the oscillator, additive,
  sag, filter, and residual paths. Do not silently replace the vintage
  topology with a generic modern synth architecture.
- Keep MIDI channel and pitch matching, sustain release, panic handling,
  bounded parameter validation, and click-free gate transitions correct.
  These are reliability fixes; they do not require changing the instrument's
  historical voice count or signal path.
- Keep modern host conveniences in adapters around the core. A future
  multichannel or polyphonic mode should be an explicitly separate extension,
  not a silent rewrite of the legacy path.
- Treat the original `sogitech_opticlink_hybrid.cmaj` as a historical
  reference generator, not a verified hardware schematic or bit-exact model
  of a particular vintage unit. Its fixed 96 kHz rate and demo A4 MIDI values
  are reference-code limitations; production patches must use the host
  sample rate and incoming MIDI events.

This separation lets the project retain the sound and workflow constraints
that motivated it while making the software implementation dependable.

## File Roles

| Component | Location | Role |
|-----------|----------|------|
| Original hybrid | `cmajor/sogitech_opticlink_hybrid.cmaj` | Reference generator |
| Parameter decoder | `cmajor/sogitech_param_decoder.cmaj` | Pure control-driven decoder |
| Full codec processor | `cmajor/sogitech_codec_processor.cmaj` | Decoder + residual / dry-wet |
| Analysis | `python/analysis/` | f0, spectral, envelope, residual |
| Neural | `python/neural/` | Controller + residual nets |
| Brute Genius | `python/brute/` | Search space + objectives |
