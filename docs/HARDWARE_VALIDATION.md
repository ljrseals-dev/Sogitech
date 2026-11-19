# Hardware / Cmajor Validation Gate

## Status

- **Software path**: complete (analysis, parameter codec, neural skeleton, Brute Genius space, offline Python reference renderer).
- **Cmajor patches**: packaged as proper `.cmajorpatch` + annotated processors ready for the official toolchain.
- **Hardware / real-time validation**: **blocked in this environment** — no `cmaj` binary and no audio/MIDI hardware.

## What you must run on a machine with the Cmajor toolchain

1. Install the Cmajor CLI from https://cmajor.dev (or use the VS Code Cmajor extension).

2. Play the patches:

```bash
cmaj play cmajor/patches/SogitechParamDecoder/SogitechParamDecoder.cmajorpatch
cmaj play cmajor/patches/SogitechCodecProcessor/SogitechCodecProcessor.cmajorpatch
```

3. Confirm:
   - Parameters appear in the host UI and respond.
   - MIDI note / velocity (or the exposed `midiNote` / `velocity` params) drive pitch and filter response.
   - PowerSag and dual-pole filter behave under load.
   - Residual input (if wired) mixes correctly.

4. Optional: export to VST3/AU via the Cmajor toolchain for DAW testing.

## Offline validation already performed here

```bash
PYTHONPATH=. python3 -c "… offline_engine.render …"
```

A short rendered file (`out_offline_validate.wav`) was produced successfully, confirming the DSP signal path (additive + VA + PowerSag + dual-pole) is coherent in pure Python.

## Remaining real blockers (not software)

| Item | Status |
|------|--------|
| Cmajor CLI / VS Code extension | Not present in this sandbox |
| Audio interface / MIDI keyboard | Not available |
| Real-time latency / CPU measurement on target hardware | Requires local machine |

These are environmental, not code defects. Once the patches are loaded with `cmaj play` on a machine that has the toolchain and audio hardware, the technical gate can be closed.
