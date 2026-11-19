# Parameter / Feature Codec

The 18 continuous parameters that already exist on the hybrid engine become the code itself.

## Parameter Vector (canonical order)

```
osc1Wave, osc2Wave, osc1Level, osc2Level,
subLevel, noiseLevel, additiveBlend, powerSagDepth,
filterLP, filterHP, filterRes, keyTrack, velMod,
glideTime, midiNote, velocity, pressure, residualGain
```

## Encoding

1. Analysis produces a time-varying instance of the vector.
2. Each dimension is quantised to a chosen bit depth (default 8).
3. Frames are packed into a JSON or binary stream together with timestamps.

Typical bit-rate (8-bit, 50 frames/s):

- 18 params × 8 bits × 50 Hz ≈ 7.2 kbit/s (before entropy coding)

This is low enough for live control, gesture storage, or transmission of “Amorph” textures.

## Decoding

Feed the de-quantised vector into `sogitech_param_decoder.cmaj` (or the full processor). The engine produces audio at the host sample rate with all the original hybrid character (additive + VA + PowerSag + dual-pole filter).
