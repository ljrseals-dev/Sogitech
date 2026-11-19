import numpy as np
import pytest

from python.audio import render_parameter_stream
from python.analysis import estimate_f0, spectral_features
from python.brute.space import build_space, estimate_cardinality
from python.neural.controller import PARAM_NAMES, PARAM_RANGES, ParameterController, load_controller

def _frames():
    base = {
        "osc1Wave": 0, "osc2Wave": 1, "osc1Level": 0.7,
        "osc2Level": 0.2, "subLevel": 0.1, "noiseLevel": 0.0,
        "additiveBlend": 0.2, "powerSagDepth": 0.1, "filterLP": 0.8,
        "filterHP": 0.02, "filterRes": 0.2, "midiNote": 69,
        "velocity": 0.8, "pressure": 0.0, "residualGain": 0.0,
    }
    return [base, dict(base, midiNote=72, filterLP=0.65)]

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

def test_short_analysis_clips_are_supported():
    audio = np.zeros(128, dtype=np.float32)
    times, f0 = estimate_f0(audio, 48000)
    features = spectral_features(audio, 48000)
    assert len(times) >= 1
    assert len(f0) == len(times)
    assert len(features["times"]) >= 1

def test_controller_outputs_stay_inside_parameter_ranges():
    controller = ParameterController()
    params = controller.predict_dict(np.zeros(64, dtype=np.float32))
    assert set(params) == set(PARAM_NAMES)
    for key, value in params.items():
        low, high = PARAM_RANGES[key]
        assert low <= value <= high

def test_neural_inference_requires_a_checkpoint():
    with pytest.raises(ValueError, match="trained weights are required"):
        load_controller()

def test_brute_genius_space_is_expanded_and_bounded():
    space = build_space()
    assert "quantisation_layout" in space["dimensions"]
    assert "codec_mode" in space["dimensions"]
    assert estimate_cardinality(space) <= 50000

from python.audio.render import _oscillator

def test_saw_oscillator_wraps_phase_and_remains_bounded():
    phase = np.linspace(0.0, 200.0 * np.pi, 10001)
    saw = _oscillator(phase, np.ones_like(phase))
    assert np.all(np.isfinite(saw))
    assert float(np.min(saw)) >= -1.0
    assert float(np.max(saw)) <= 1.0

def test_renderer_rejects_non_finite_residual():
    frames = _frames()
    with pytest.raises(ValueError, match="finite samples"):
        render_parameter_stream(frames, [0.0, 0.08], 2048, 48000,
                                residual=np.full(2048, np.nan))

def test_renderer_rejects_non_integer_sample_count():
    with pytest.raises(ValueError, match="sample_count"):
        render_parameter_stream(_frames(), [0.0, 0.08], 12.5, 48000)

from python.analysis.envelope import harmonic_envelope, rms_envelope
from python.analysis.residual import residual_energy
from python.analysis.spectral import estimate_filter_params
from python.brute.space import estimate_cardinality

def test_envelope_and_residual_analysis_reject_invalid_sample_rates():
    audio = np.ones(2048, dtype=np.float32)
    with pytest.raises(ValueError, match="sample rate"):
        rms_envelope(audio, 0)
    with pytest.raises(ValueError, match="sample rate"):
        residual_energy(audio, 0)

def test_short_envelope_and_residual_frames_are_finite():
    audio = np.ones(64, dtype=np.float32) * 0.1
    _, rms = rms_envelope(audio, 48000)
    _, residual = residual_energy(audio, 4000)
    assert len(rms) == 1 and np.all(np.isfinite(rms))
    assert len(residual) == 1 and np.all(np.isfinite(residual))

def test_harmonic_envelope_rejects_mismatched_analysis_frames():
    with pytest.raises(ValueError, match="equal length"):
        harmonic_envelope(np.ones(2048), 48000, np.array([440.0]), np.array([0.0, 0.1]))

def test_filter_mapping_rejects_non_finite_features():
    with pytest.raises(ValueError, match="finite values"):
        estimate_filter_params({
            "centroid": np.array([1000.0, np.nan]),
            "flatness": np.array([0.1, 0.2]),
        }, 48000)

def test_brute_cardinality_rejects_non_list_dimensions():
    with pytest.raises(ValueError, match="non-empty list"):
        estimate_cardinality({"dimensions": {"pitch": "autocorr"}})

def test_controller_rejects_non_finite_feature_vector():
    controller = ParameterController()
    features = np.zeros(64, dtype=np.float32)
    features[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        controller.predict_dict(features)

def test_custom_brute_space_cannot_override_hard_candidate_ceiling():
    from python.brute.plan import main
    from unittest.mock import patch
    import json
    from pathlib import Path
    with patch("sys.argv", ["plan", "--max-cost-usd", "1000000000"]):
        with pytest.raises(SystemExit):
            main()
