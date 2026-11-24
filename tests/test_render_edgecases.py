import numpy as np
import pytest

from python.audio.render import _oscillator, render_parameter_stream


def test_triangle_wave_is_bipolar_and_periodic():
    phase = np.linspace(0.0, 2.0 * np.pi, 1001)
    triangle = _oscillator(phase, np.full_like(phase, 3.0))
    assert triangle.min() == pytest.approx(-1.0, abs=1e-6)
    assert triangle.max() == pytest.approx(1.0, abs=1e-6)
    assert triangle[0] == pytest.approx(triangle[-1], abs=1e-6)


@pytest.mark.parametrize("kwargs", [
    {"sample_rate": 48000.5},
    {"sample_rate": True},
    {"sample_count": True},
    {"sample_count": 0},
    {"seed": 1.5},
])
def test_renderer_rejects_invalid_scalar_arguments(kwargs):
    args = {"frames": [{"midiNote": 60.0}], "times": [0.0],
            "sample_count": 1024, "sample_rate": 48000, "seed": 7}
    args.update(kwargs)
    with pytest.raises(ValueError):
        render_parameter_stream(**args)


def test_renderer_is_finite_and_repeatable_for_fixed_seed():
    frames = [{"midiNote": 69.0, "osc1Wave": 1.0, "osc2Wave": 2.0}]
    first = render_parameter_stream(frames, [0.0], 4096, 48000, seed=11)
    second = render_parameter_stream(frames, [0.0], 4096, 48000, seed=11)
    assert first.dtype == np.float32
    assert np.all(np.isfinite(first))
    np.testing.assert_array_equal(first, second)
