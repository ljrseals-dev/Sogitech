# Sogitech Codec

**Hybrid Analysis–Resynthesis / Parameter / Neural Codec built on the Sogitec OpticLink Hybrid engine.**

The repository contains a Cmajor hybrid synthesiser/decoder, Python analysis
front-ends, parameter quantisation, an optional neural parameter controller,
and a bounded Brute Genius search-space description.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"

# DSP analysis/resynthesis demo
python scripts/demo_dsp_codec.py --input examples/demo.wav --out out/dsp.wav

# Quantised parameter stream -> decoded audio + JSON sidecar
python scripts/demo_param_codec.py --input examples/demo.wav --out out/parameter.wav --bits 8

# Train a parameter controller from at least two WAV files
python -m pip install -e ".[neural]"
python scripts/train_controller.py --data path/to/wav_dataset --out weights/controller.pt

# Neural controller -> bounded parameters -> rendered audio
python scripts/demo_neural_codec.py --input examples/demo.wav --weights weights/controller.pt --out out/neural.wav

# Local Brute Genius cost/cardinality plan only (does not launch remote work)
python -m python.brute.plan --max-cost-usd 5
```

The parameter and neural demos render audible WAV output using the deterministic
offline reference renderer in `python/audio/render.py`; they no longer write
silent placeholder files. The renderer is a Python reference implementation,
not a bit-exact replacement for the real-time Cmajor DSP. The Cmajor processor
remains the authoritative real-time decoder.

## Architecture

- **DSP analysis/resynthesis:** estimates pitch, spectral shape and envelope,
  then renders a hybrid oscillator approximation.
- **Parameter codec:** quantises the 18-field control stream, writes a JSON
  sidecar, dequantises it, and renders audio from the decoded values.
- **Neural controller:** supervised pseudo-label training maps a fixed 64-value
  feature vector to bounded synth parameters. It is an experimental timbre
  controller, not a trained waveform codec or lossless compressor.
- **Cmajor decoder:** implements the hybrid VA/additive oscillator, PowerSag,
  dual-pole filter and optional residual mix. Both decoder processors accept
  raw `std::midi::Message` events for note on/off, pitch bend, channel/poly
  pressure, CC 1/7/64/71/74 and panic handling. Velocity-zero note-on is treated as note-off; note releases are channel/pitch matched. MIDI handling is currently
  monophonic: a matching note-off releases the active note.

## Cmajor patches and MIDI

See [docs/CMAJOR_MIDI.md](docs/CMAJOR_MIDI.md) for the patch manifests, MIDI
mapping, and the explicit monophonic limitations.

After installing the Cmajor command-line toolchain, run either patch:

```bash
cmaj play cmajor/sogitech_codec.cmajorpatch
cmaj play cmajor/sogitech_param_decoder.cmajorpatch
```

## Training data

Use a directory of representative WAV files, with source recordings grouped
by performer/session when splitting datasets. The training script derives
deterministic DSP pseudo-labels from each file; it does not require paired
ground-truth synth parameters. A checkpoint is required for neural inference:
the demo deliberately refuses to substitute random weights.

## Brute Genius

See [docs/BRUTE_GENIUS.md](docs/BRUTE_GENIUS.md). The local planner reports a
conservative Cartesian upper bound and an illustrative cost estimate. It does
not launch Brute Genius jobs or provision external workers.

## Validation

Python CI covers Python 3.10–3.12. The Cmajor patch sources still require
compilation with the installed `cmaj` toolchain before host validation.

## License

MIT. See [LICENSE](LICENSE).
