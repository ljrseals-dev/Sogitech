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
| CC 120 | Immediate all-sound-off (hard gate reset) |
| CC 123 | All-notes-off (smoothed release) |

The current voice policy is intentionally **monophonic**. A note-off only
releases the currently active pitch; there is no polyphonic voice allocator,
per-channel MPE state, or overlapping-note stack yet. Note-off matching includes MIDI channel and pitch; pitch, pressure and controller events are scoped to the active channel while a note is held. Sustain state is tracked against its MIDI channel and is cleared by panic events. Note-on velocity zero is treated as note-off; note-off matching includes MIDI channel and pitch.
The event callback only updates processor state; sample generation remains in
the normal audio loop. The note gate uses a 5 ms attack and 20 ms release to reduce clicks. The sample rate is read from `processor.frequency`,
not hard-coded to 96 kHz.

## Validation status

The Python CI workflow tests the offline renderer and controller bounds.
Cmajor source compilation must still be checked with the installed `cmaj`
toolchain before treating either patch as host-validated.
