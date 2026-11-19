import numpy as np
import pytest

from python.audio import render_parameter_stream

def _frames():
    base = {
        "osc1Wave": 0, "osc2Wave": 1, "osc1Level": 0.7,
        "osc2Level": 0.2, "subLevel": 0.1, "noiseLevel": 0.0,
        "additiveBlend": 0.2, "powerSagDepth": 0.1, "filterLP": 0.8,
        "filterHP": 0.02, "filterRes": 0.2, "midiNote": 69,
        "velocity": 0.8, "pressure": 0.0, "residualGain": 0.0,
    }
    other = dict(base, midiNote=72, filterLP=0.65)
    return [base, other]

def test_parameter_stream_produces_finite_audible_audio():
    audio = render_parameter_stream(_frames(), [0.0, 0.08], 4096, 48000, seed=11)
    assert audio.dtype == np.float32
    assert audio.shape == (4096,)
    assert np.all(np.isfinite(audio))
    assert float(np.max(np.abs(audio))) > 1e-3
    assert float(np.max(np.abs(audio))) <= 0.981

def test_renderer_is_deterministic_for_a_fixed_seed():
    a = render_parameter_stream(_frames(), [0.0, 0.08], 2048, 48000, seed=99)
    b = render_parameter_stream(_frames(), [0.0, 0.08], 2048, 48000, seed=99)
    np.testing.assert_array_equal(a, b)

def test_renderer_rejects_non_monotonic_times():
    with pytest.raises(ValueError, match="strictly increasing"):
        render_parameter_stream(_frames(), [0.1, 0.0], 2048, 48000)

def test_renderer_rejects_invalid_sample_rate():
    with pytest.raises(ValueError, match="sample_rate"):
        render_parameter_stream(_frames(), [0.0, 0.1], 2048, 1000)
