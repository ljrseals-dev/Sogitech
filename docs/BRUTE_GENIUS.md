# Brute Genius search space

Sogitech exposes a bounded Cartesian search space in `python/brute/space.py`.
Version 2 adds explicit codec mode, pitch/envelope analysis, residual policy,
filter mapping, quantisation layout, neural controller size, sample rate and
latency budget. The full Cartesian upper bound is capped at 50,000 combinations.

## Run a local plan

```bash
python -m python.brute.plan --max-cost-usd 5
python -m python.brute.plan --space path/to/custom_space.json
```

The local planner now enumerates candidates and applies the declared
conditional constraints before reporting the feasible count. Its cost remains
an illustrative per-candidate estimate; it does not call Brute Genius,
provision workers, execute DSP evaluations, or launch jobs. The Cartesian
upper bound is still checked against the configured safety ceiling before
enumeration.

For the default v2 space, the unconstrained Cartesian upper bound is 23,328.
The current conditional rule restricts DSP mode to the non-neural controller,
leaving 15,552 feasible candidates. This is an arithmetic expectation from
the declared dimensions, not a claim that 15,552 codec evaluations have run.

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

## Candidate enumeration contract

`python.brute.candidates.iter_candidates` yields candidate dictionaries in
stable dimension/value declaration order. `count_candidates` applies the
same filtering logic and returns the number of feasible candidates. Invalid
constraint references, empty axes, and search spaces exceeding the declared
combination ceiling fail closed.

These functions enumerate configurations only. A future evaluator must
separately bind each candidate to fixed source-grouped fixtures, deterministic
seeds, real measured metrics, and repeated runs. It must not rank candidates
using placeholder scores or estimated latency.

## Evaluation protocol

Candidates should use source-grouped train/validation/test splits to prevent
near-identical crops from leaking across splits. Keep the random seed fixed per
candidate and repeat each measurement three times. Record spectral and
parameter errors alongside residual-energy error, measured latency, CPU load,
and peak level. Do not compare candidates using an estimated latency or cost
as though it were a measured result.

Neural variants are experimental and require a trained checkpoint. The local
planner does not claim that external models or Brute Genius workers are present.
