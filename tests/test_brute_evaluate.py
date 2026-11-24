"""Tests for the deterministic local parameter-codec evaluator."""
import numpy as np
import pytest

from python.brute.evaluate import (
    _dequantise, _quantise, _spectral_metrics, evaluate_candidate,
)


@pytest.mark.parametrize("bits", [4, 8, 12])
def test_quantizer_is_monotone_and_bounded(bits):
    values = np.linspace(0.0, 1.0, 101)
    codes = [_quantise(value, bits, 0.0, 1.0) for value in values]
    assert codes == sorted(codes)
    assert min(codes) == 0
    assert max(codes) == (1 << bits) - 1
    decoded = [_dequantise(code, bits, 0.0, 1.0) for code in codes]
    assert decoded == sorted(decoded)


def test_spectral_metric_detects_injected_sample_fault():
    sr = 48000
    t = np.arange(4096) / sr
    reference = np.sin(2 * np.pi * 440 * t)
    clean = reference.copy()
    corrupted = reference.copy()
    corrupted[1200] += 0.75
    clean_score = _spectral_metrics(reference, clean, sr)["multi_res_stft"]
    corrupted_score = _spectral_metrics(reference, corrupted, sr)["multi_res_stft"]
    assert clean_score == pytest.approx(0.0, abs=1e-12)
    assert corrupted_score > clean_score


def test_candidate_record_is_repeatable_and_has_provenance():
    sr = 48000
    t = np.arange(4096) / sr
    audio = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float64)
    first = evaluate_candidate(audio, "a" * 64, "performer-session-1",
                               "validation", sr, 8, 19, 2, 20)
    second = evaluate_candidate(audio, "a" * 64, "performer-session-1",
                                "validation", sr, 8, 19, 2, 20)
    assert first["candidate_id"] == second["candidate_id"]
    assert first["candidate"]["quantisation_bits"] == 8
    assert first["candidate"]["pitch_tracker"] == "autocorr"
    assert first["source_sha256"] == "a" * 64
    assert first["split"] == "validation"
    assert first["correctness"] == "PASS"
    assert first["metrics"]["multi_res_stft"]["mean"] == pytest.approx(
        second["metrics"]["multi_res_stft"]["mean"], abs=1e-12
    )
    assert "cmajor_parity_not_checked" in first["limitations"]
