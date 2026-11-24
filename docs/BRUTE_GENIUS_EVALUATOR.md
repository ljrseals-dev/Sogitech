# Local Brute Genius evaluator

**Status: first executable reference-path evaluator; not a production search runner.**

Run against a WAV fixture:

```bash
python -m python.brute.evaluate \
  --input examples/demo.wav \
  --output out/brute-results.jsonl \
  --source-group performer-a/session-03 \
  --split validation \
  --bits 8 \
  --sample-rate 48000 \
  --seed 7 \
  --repetitions 3 \
  --latency-budget-ms 20
```

Each JSONL record includes a canonical candidate ID, candidate settings,
SHA-256 of the exact input file, source-group and split labels, fixed seed,
repetition count, correctness result, per-run raw metrics, and aggregate mean
and population standard deviation. Output is append-only; use a new results
file or deliberately manage its contents between experiments.

## Implemented search axes

This evaluator deliberately supports only the parameter codec path:

- `codec_mode=parameter`
- `pitch_tracker=autocorr` (explicitly selected; no silent pYIN fallback)
- `envelope_method=rms`
- `residual_strategy=none`
- `filter_mapping=centroid_linear`
- `quantisation_bits=4|8|12`
- `quantisation_layout=uniform`
- `neural_controller=none`
- `objective_primary=multi_res_stft`
- `sample_rate=48000|96000`
- `latency_budget_ms=5|20` (recorded target, not an automatic pass/fail gate)

Other declared Brute Genius choices are not silently substituted. The evaluator
does not train a neural controller or use random/unverified checkpoints.

## Metrics and interpretation

- `spectral_l1`: mean absolute magnitude-spectrum difference at 1024-point STFT.
- `multi_res_stft`: mean magnitude difference across 512, 1024, and 2048-point STFTs.
- `parameter_l1`: mean absolute parameter error before/after uniform quantisation.
- `residual_energy_error`: absolute difference between input RMS and residual RMS.
- `peak_dbfs`: rendered peak level.
- `elapsed_ms`: measured local analysis/render wall time for each repetition.

The reference renderer is deterministic for a fixed input, seed, and environment.
The current evaluator records wall time but does not claim CPU utilisation,
real-time deadline compliance, perceptual quality, or Cmajor parity. A PASS
means only that output shape/samples and computed metrics passed basic finite
value checks. It is not a production-safety or audio-quality certification.

## Dataset discipline

Use stable `--source-group` values for performer/session/source identity and
keep related takes/crops in the same split. Run the evaluator separately for
train, validation, and test fixtures; do not rank candidates on training
fixtures. This first version records the split label but does not enforce
cross-file group exclusivity. That requires a manifest-driven corpus runner.

## Next gates

1. Add a fixture manifest with verified group exclusivity and train/validation/test hashes.
2. Add a corpus runner that aggregates by source group, not by WAV crop.
3. Establish baseline quality and latency distributions before ranking.
4. Compile and host-test the Cmajor decoder, then add a parity gate against
   the Python reference path.
5. Only then enable residual strategies, alternative filter mappings, or
   neural candidates that have an actual trained checkpoint and evaluation path.
