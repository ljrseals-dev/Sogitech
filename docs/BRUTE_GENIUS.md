# Brute Genius Integration

The search space lives in `python/brute/space.py`.

## Dimensions (default)

| Dimension | Values |
|-----------|--------|
| pitch_tracker | librosa_pyin, autocorr, crepe_tiny, rmvpe |
| envelope_method | rms, peak, harmonic_32 |
| residual_strategy | none, highpass_3k, harmonic_subtract, neural_residual |
| filter_mapping | centroid_linear, centroid_log, formant_proxy |
| quantisation_bits | 4, 6, 8, 12 |
| neural_controller | none, mlp_64_128, mlp_128_256 |
| objective_primary | spectral_l1, multi_res_stft, param_l1, perceptual_proxy |
| sample_rate | 48000, 96000 |
| latency_budget_ms | 5, 10, 20, 40 |

## Usage with Brute Genius MCP

1. Call `brute_genius_plan` with the space (or a reduced subset).
2. Respect the returned cost ceiling.
3. Launch only if acceptable.
4. Score every combination with the objectives in `python/brute/objectives.py`.
5. Rank and inspect the top configurations.

A local dry-run planner is available:

```bash
python -m python.brute.plan --max-cost-usd 2.0
```

This prints cardinality and a rough cost estimate without spending money.
