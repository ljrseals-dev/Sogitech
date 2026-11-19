# Brute Genius search space

Sogitech exposes a bounded Cartesian search space in `python/brute/space.py`.
Version 2 adds explicit codec mode, pitch/envelope analysis, residual policy,
filter mapping, quantisation layout, neural controller size, sample rate and
latency budget. The full Cartesian upper bound is intentionally capped at
50,000 combinations.

## Run a local plan

```bash
python -m python.brute.plan --max-cost-usd 5
python -m python.brute.plan --space path/to/custom_space.json
```

The planner is local and read-only. It estimates a cost from a configurable
per-combination assumption; it does not call Brute Genius, provision workers,
or launch jobs. Conditional constraints are reported but are not applied as
pruning, so the displayed cardinality is a conservative upper bound.

## Search dimensions

| Axis | Choices |
|---|---|
| codec_mode | DSP, parameter |
| pitch_tracker | autocorrelation, librosa pYIN |
| envelope_method | RMS, peak, Hilbert |
| residual_strategy | none, high-pass residual, harmonic subtraction |
| filter_mapping | linear/log centroid, spectral rolloff |
| quantisation_bits | 4, 8, 12 |
| quantisation_layout | uniform, perceptual |
| neural_controller | none, MLP 64×128, MLP 64×256 |
| objective_primary | spectral L1, multi-resolution STFT, perceptual proxy |
| sample_rate | 48 kHz, 96 kHz |
| latency_budget_ms | 5, 20 |

## Evaluation protocol

Candidates should use source-grouped train/validation/test splits to prevent
near-identical crops from leaking across splits. Keep the random seed fixed per
candidate and repeat each measurement three times. Record spectral and
parameter errors alongside residual-energy error, measured latency, CPU load,
and peak level. Do not compare candidates using an estimated latency or cost
as though it were a measured result.

Neural variants are experimental and require a trained checkpoint. The local
planner does not claim that external models or Brute Genius workers are present.
