# Hybrid Neural Codec Path

## Controller

`python/neural/controller.py` defines a small MLP that maps a fixed-size feature vector (mel statistics, f0, RMS, etc.) onto the 18-dimensional parameter vector.

- Training target: parameters that, when fed to the Cmajor decoder, minimise a spectral or perceptual loss against the original audio.
- Inference: produce a parameter stream (or a single static vector for sustained textures) that drives the decoder.

Weights are optional. Without a checkpoint the controller returns a random but valid vector so the rest of the pipeline can be exercised.

## Residual Network

An optional second network predicts a residual waveform that is mixed after the DSP chain (`residualGain` control). This lets the neural path correct for limitations of the pure additive/VA model while still keeping the core engine deterministic and real-time.

## Training Sketch

1. Run analysis on a corpus → target parameter trajectories.
2. Render the decoder with those trajectories → reconstructed audio.
3. Train controller so that predicted parameters, when rendered, minimise multi-resolution STFT loss (+ optional parameter L1 regularisation).
4. Optionally train residual net on the difference signal.

The Cmajor side never needs to change; only the control stream and residual gain change.
