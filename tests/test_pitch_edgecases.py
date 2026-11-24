import numpy as np
import pytest

from python.analysis.pitch import estimate_f0, midi_from_f0


def test_pitch_range_includes_exact_low_frequency_lag():
    sr = 8000
    frequency = 100.0
    t = np.arange(4096) / sr
    audio = np.sin(2 * np.pi * frequency * t)
    _, f0 = estimate_f0(audio, sr, frame_length=2048, hop_length=512,
                        fmin=100.0, fmax=500.0, method="autocorr")
    voiced = f0[f0 > 0]
    assert voiced.size
    assert np.median(voiced) == pytest.approx(frequency, abs=1.0)


@pytest.mark.parametrize("kwargs", [
    {"sr": 8000, "fmin": 50.0, "fmax": 4000.0},
    {"sr": 8000, "fmin": 0.0, "fmax": 1000.0},
    {"sr": 8000, "fmin": 1000.0, "fmax": 100.0},
    {"sr": 8000.5, "fmin": 50.0, "fmax": 1000.0},
])
def test_pitch_rejects_invalid_rate_or_range(kwargs):
    with pytest.raises(ValueError):
        estimate_f0(np.ones(2048), method="autocorr", **kwargs)


def test_midi_conversion_validates_frequency_and_default():
    result = midi_from_f0(np.array([0.0, 440.0]))
    np.testing.assert_allclose(result, [69.0, 69.0])
    with pytest.raises(ValueError):
        midi_from_f0(np.array([-1.0]))
    with pytest.raises(ValueError):
        midi_from_f0(np.array([np.nan]))
    with pytest.raises(ValueError):
        midi_from_f0(np.array([0.0]), default_midi=128.0)
