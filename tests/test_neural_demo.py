import numpy as np

from scripts.demo_neural_codec import build_neural_frames


def test_neural_frames_gate_unvoiced_audio_and_use_predicted_note():
    params = {"midiNote": 60.0, "velocity": 0.8, "residualGain": 0.9}
    frames = build_neural_frames(params, np.array([440.0, 0.0, 220.0]))
    assert frames[0]["midiNote"] == 69.0
    assert frames[0]["velocity"] == 0.8
    assert frames[1]["midiNote"] == 60.0
    assert frames[1]["velocity"] == 0.0
    assert frames[2]["midiNote"] == 57.0
    assert all(frame["residualGain"] == 0.0 for frame in frames)


def test_neural_frames_reject_invalid_pitch_track():
    import pytest
    with pytest.raises(ValueError):
        build_neural_frames({"midiNote": 60.0, "velocity": 0.8},
                            np.array([440.0, np.nan]))
