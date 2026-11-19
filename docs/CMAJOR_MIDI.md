# Cmajor patches and MIDI contract

Two loadable patch manifests are provided. Their manifests point to canonical `.cmajor` source files (the original `.cmaj` paths are retained as compatibility copies):

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
| Note On (velocity above zero) | Sets active pitch, channel and velocity; opens the voice |
| Note On (velocity zero) | Treated as Note Off, per MIDI convention |
| Matching Note Off | Releases the active monophonic voice |
| Pitch wheel | Applies a ±2-semitone bend |
| Channel pressure / poly aftertouch | Updates pressure expression on the active channel; poly aftertouch also matches the active pitch |
| CC 1 | Updates pressure/modulation |
| CC 7 | Master output level |
| CC 71 | Filter resonance |
| CC 74 | Filter cutoff |
| CC 64 | Channel-scoped sustain pedal; defers matching note-off until pedal release |
| CC 120 | Immediate all-sound-off on that channel (hard gate reset) |
| CC 123 | All-notes-off on that channel (smoothed release) |

The current voice policy is intentionally **monophonic**, preserving the
vintage home-studio instrument topology and its original constrained signal
routing. This is a design choice, not an unfinished polyphony feature. A
note-off only releases the currently active pitch; there is no polyphonic
voice allocator, per-channel MPE state, or overlapping-note stack. Any future
polyphonic or multichannel expansion should be a separate mode that leaves
the legacy path unchanged. Note-off matching includes MIDI channel and pitch; pitch, pressure and controller events are scoped to the active channel while a note is held. Sustain state is tracked against its MIDI channel; panic messages clear only the sustain state on their own channel. Note-on velocity zero is treated as note-off; note-off matching includes MIDI channel and pitch.
The event callback only updates processor state; sample generation remains in
the normal audio loop. The note gate uses a 5 ms attack and 20 ms release to reduce clicks. Power sag follows a load-dependent target with a 150 ms recovery time, so attenuation recovers after loud passages. The sample rate is read from `processor.frequency`, not hard-coded to 96 kHz.

## Validation status

The Python CI workflow tests the offline renderer and controller bounds.
The Cmajor compile test covers both patch manifests. Once the Cmajor CLI is
installed, run `cmaj test cmajor/tests/sogitech_patches.cmajtest` from the
repository root. Compilation must pass before host playback validation.
