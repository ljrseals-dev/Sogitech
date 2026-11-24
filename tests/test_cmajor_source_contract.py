from pathlib import Path

import pytest


@pytest.mark.parametrize("name", [
    "sogitech_codec_processor.cmajor",
    "sogitech_param_decoder.cmajor",
])
def test_cmajor_voice_uses_host_pitch_when_no_midi_note_is_active(name):
    source = (Path(__file__).parents[1] / "cmajor" / name).read_text(encoding="utf-8")
    assert "let sourceNote = noteActive ? float (activeMidiNote) : midiNote;" in source
    assert "let sourceNote = activeMidiNote >= 0 ? float (activeMidiNote) : midiNote;" not in source


@pytest.mark.parametrize("name", [
    "sogitech_codec_processor.cmajor",
    "sogitech_param_decoder.cmajor",
])
def test_cmajor_triangle_is_bipolar_and_additive_stack_is_bandlimited(name):
    source = (Path(__file__).parents[1] / "cmajor" / name).read_text(encoding="utf-8")
    assert "1.0 - (2.0 / pi) * acos (clamp (sin (phase), -1.0, 1.0))" in source
    assert "floor (sampleRate * 0.45 / max (fundamental, 1.0))" in source
