"""Tests for deterministic, constrained Brute Genius candidate enumeration."""
import pytest

from python.brute.candidates import count_candidates, iter_candidates
from python.brute.space import SOGITECH_CODEC_SPACE, estimate_cardinality


def test_default_space_prunes_conditional_combinations_deterministically():
    assert estimate_cardinality(SOGITECH_CODEC_SPACE) == 23328
    first = list(iter_candidates())
    second = list(iter_candidates())
    assert first == second
    assert len(first) == 15552
    assert count_candidates() == 15552
    assert all(
        candidate["neural_controller"] == "none"
        for candidate in first
        if candidate["codec_mode"] == "dsp"
    )


def test_parameter_mode_keeps_all_declared_controller_sizes():
    candidates = list(iter_candidates())
    neural_values = {
        candidate["neural_controller"]
        for candidate in candidates
        if candidate["codec_mode"] == "parameter"
    }
    assert neural_values == {"none", "mlp_64_128", "mlp_64_256"}


def test_custom_conditional_constraint_is_applied():
    space = {
        "dimensions": {"mode": ["fast", "safe"], "width": [4, 8]},
        "constraints": [{"if": {"mode": "fast"}, "then": {"width": [8]}}],
    }
    assert list(iter_candidates(space)) == [
        {"mode": "fast", "width": 8},
        {"mode": "safe", "width": 4},
        {"mode": "safe", "width": 8},
    ]


def test_unknown_constraint_dimension_is_rejected():
    space = {
        "dimensions": {"mode": ["fast"]},
        "constraints": [{"if": {"missing": "x"}, "then": {"mode": ["fast"]}}],
    }
    with pytest.raises(ValueError, match="unknown dimensions"):
        list(iter_candidates(space))


def test_cartesian_upper_bound_respects_ceiling():
    space = {
        "dimensions": {"mode": ["fast", "safe"]},
        "constraints": [{"max_combinations": 1}],
    }
    with pytest.raises(ValueError, match="exceeds combination ceiling"):
        list(iter_candidates(space))
