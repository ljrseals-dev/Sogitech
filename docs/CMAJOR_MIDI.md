# Cmajor patches and MIDI contract

Two loadable patch manifests are provided:

- `cmajor/sogitech_codec.cmajorpatch` — full codec processor
- `cmajor/sogitech_param_decoder.cmajorpatch` — parameter-driven decoder

Install the Cmajor command-line tools, then run either patch with:

```bash
cmaj play cmajor/sogitech_codec.cmajorpatch
cmaj play cmajor/sogitech_param_decoder.cmajorpatch
```

## MIDI event contract

Both processors expose `input event std::midi::Message midiIn`. The event
handler consumes the standard Cmajor three-byte MIDI message type. Supported
messages:

| MIDI input | Behaviour |
|---|---|
| Note On | Sets active pitch and velocity; opens the voice |
| Matching Note Off | Releases the active monophonic voice |
| Pitch wheel | Applies a ±2-semitone bend |
| Channel pressure / poly aftertouch | Updates pressure expression |
| CC 1 | Updates pressure/modulation |
| CC 7 | Master output level |
| CC 71 | Filter resonance |
| CC 74 | Filter cutoff |
| CC 120 / CC 123 | Immediate voice release |

The current voice policy is intentionally **monophonic**. A note-off only
releases the currently active pitch; there is no polyphonic voice allocator,
per-channel MPE state, sustain-pedal latch, or overlapping-note stack yet.
The event callback only updates processor state; sample generation remains in
the normal audio loop. The sample rate is read from `processor.frequency`,
not hard-coded to 96 kHz.

## Validation status

The Python CI workflow tests the offline renderer and controller bounds.
Cmajor source compilation must still be checked with the installed `cmaj`
toolchain before treating either patch as host-validated.
