"""Unit tests for the objective output analysers.

The windowed error metrics (MSE / MAE / spectral energy) are pure NumPy and
fully deterministic, so they are checked against golden values computed from
the pinned implementation on a small mock input. The perceptual metric is
checked through invariants plus a golden value, because its constant-Q backend
has two interchangeable implementations (essentia when available, librosa
otherwise).

The subprocess/metric installers that need external programs or networks
(PEAQ, PESQ, PLCMOS, MUSHRA) are intentionally out of scope for unit tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from plctestbench.file_wrapper import SimpleCalculatorData
from plctestbench.output_analyser import (
    MAECalculator,
    MSECalculator,
    OutputAnalyser,
    PerceptualCalculator,
    SimpleCalculator,
    SpectralEnergyCalculator,
    normalise,
)
from plctestbench.perceptual_metric import PerceptualMetric
from plctestbench.settings import (
    MAECalculatorSettings,
    MSECalculatorSettings,
    PerceptualCalculatorSettings,
    SpectralEnergyCalculatorSettings,
)

from test._helpers import AudioStub, DataStub, attach

N = 8
HOP = 4

#: Mock input: a short, deterministic mono waveform.
MOCK_ORIGINAL = np.array(
    [
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
        0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0,
    ],
    dtype=np.float32,
)
MOCK_RECONSTRUCTED = (MOCK_ORIGINAL * 0.9 + 0.01).astype(np.float32)

MSE_GOLDEN = np.array([1.1901139e-05, 1.0329546e-06, 4.6557202e-06])
MAE_GOLDEN = np.array(
    [0.0027472474612295628, 0.0007672745850868523, 0.0016483509680256248]
)
SE_GOLDEN = np.array(
    [
        [4.83031617e-04, 7.72116255e-05, 8.28707982e-07, 3.55491032e-08,
         1.11022296e-16, 3.55491032e-08, 8.28707982e-07, 7.72116255e-05],
        [3.76774587e-05, 1.91857953e-06, 3.62281594e-06, 4.14322869e-07,
         2.07161435e-07, 4.14322869e-07, 3.62281594e-06, 1.91857953e-06],
        [1.73891909e-04, 2.76860155e-05, 8.28762211e-07, 3.55491032e-08,
         1.90484161e-17, 3.55491032e-08, 8.28762211e-07, 2.76860155e-05],
    ]
)


def _nodes():
    return AudioStub(MOCK_ORIGINAL), AudioStub(MOCK_RECONSTRUCTED)


# --------------------------------------------------------------------------- #
# normalise
# --------------------------------------------------------------------------- #


def test_normalise_scales_the_peak_to_the_amplitude():
    np.testing.assert_allclose(normalise(np.array([1.0, 2.0, 4.0])), [0.25, 0.5, 1.0])


def test_normalise_respects_the_amplitude_scale():
    np.testing.assert_allclose(
        normalise(np.array([1.0, 2.0, 4.0]), amp_scale=2.0), [0.5, 1.0, 2.0]
    )


# --------------------------------------------------------------------------- #
# SimpleCalculator framing
# --------------------------------------------------------------------------- #


def test_simple_calculator_returns_one_window_per_hop():
    settings = attach(MSECalculatorSettings(N=N, hop=HOP))
    original, reconstructed = _nodes()
    x_rw, x_ew = SimpleCalculator(settings).run(original, reconstructed)

    assert x_rw.shape == (3, N)
    assert x_ew.shape == (3, N)


def test_simple_calculator_applies_a_hanning_window():
    settings = attach(MSECalculatorSettings(N=N, hop=HOP))
    original, reconstructed = _nodes()
    x_rw, _ = SimpleCalculator(settings).run(original, reconstructed)

    window = np.hanning(N + 1)[:-1]
    np.testing.assert_allclose(x_rw[0], window * normalise(MOCK_ORIGINAL)[0:N])


def test_simple_calculator_broadcasts_the_window_over_channels():
    settings = attach(MSECalculatorSettings(N=N, hop=HOP))
    stereo = np.stack((MOCK_ORIGINAL, MOCK_ORIGINAL), axis=1)
    x_rw, _ = SimpleCalculator(settings).run(AudioStub(stereo), AudioStub(stereo))

    assert x_rw.shape == (3, N, 2)


# --------------------------------------------------------------------------- #
# Error metrics
# --------------------------------------------------------------------------- #


def test_mse_matches_golden():
    settings = attach(MSECalculatorSettings(N=N, hop=HOP))
    original, reconstructed = _nodes()
    error = MSECalculator(settings).run(original, reconstructed).get_error()
    np.testing.assert_allclose(error, MSE_GOLDEN, rtol=1e-4, atol=1e-12)


def test_mae_matches_golden():
    settings = attach(MAECalculatorSettings(N=N, hop=HOP))
    original, reconstructed = _nodes()
    error = MAECalculator(settings).run(original, reconstructed).get_error()
    np.testing.assert_allclose(error, MAE_GOLDEN, rtol=1e-4, atol=1e-12)


def test_spectral_energy_matches_golden():
    settings = attach(SpectralEnergyCalculatorSettings(N=N, hop=HOP))
    original, reconstructed = _nodes()
    error = SpectralEnergyCalculator(settings).run(original, reconstructed).get_error()
    np.testing.assert_allclose(error, SE_GOLDEN, rtol=1e-4, atol=1e-12)


def test_mse_of_identical_signals_is_zero():
    settings = attach(MSECalculatorSettings(N=N, hop=HOP))
    original, _ = _nodes()
    error = MSECalculator(settings).run(original, original).get_error()
    np.testing.assert_allclose(error, np.zeros(3), atol=1e-12)


def test_mae_of_identical_signals_is_zero():
    settings = attach(MAECalculatorSettings(N=N, hop=HOP))
    original, _ = _nodes()
    error = MAECalculator(settings).run(original, original).get_error()
    np.testing.assert_allclose(error, np.zeros(3), atol=1e-12)


def test_spectral_energy_of_identical_signals_is_zero():
    settings = attach(SpectralEnergyCalculatorSettings(N=N, hop=HOP))
    original, _ = _nodes()
    error = SpectralEnergyCalculator(settings).run(original, original).get_error()
    np.testing.assert_allclose(error, np.zeros((3, N)), atol=1e-12)


def test_metric_results_are_wrapped_in_simple_calculator_data():
    settings = attach(MSECalculatorSettings(N=N, hop=HOP))
    original, reconstructed = _nodes()
    result = MSECalculator(settings).run(original, reconstructed)

    assert isinstance(result, SimpleCalculatorData)
    assert len(result) == 3
    assert isinstance(hash(result), int)


def test_output_analyser_base_class_constructs_with_primed_settings():
    assert OutputAnalyser(attach(MSECalculatorSettings())) is not None


# --------------------------------------------------------------------------- #
# Perceptual metric
# --------------------------------------------------------------------------- #

PERCEPTUAL_FS = 48000
PERCEPTUAL_SAMPLES = 2400
#: Golden metric for a full-scale packet zeroed in an otherwise clean sine.
PERCEPTUAL_CALCULATOR_GOLDEN = 0.0026366591919213533


def _perceptual_metric():
    return PerceptualMetric(
        min_frequency=32.7,
        max_frequency=20000,
        bins_per_octave=12,
        n_bins=100,
        minimum_window=128,
        input_size=PERCEPTUAL_SAMPLES,
        fs=PERCEPTUAL_FS,
        intorno_length=50,
    )


def test_perceptual_metric_spectrogram_exposes_both_signals():
    metric = _perceptual_metric()
    original = np.sin(2 * np.pi * 440 * np.arange(PERCEPTUAL_SAMPLES) / PERCEPTUAL_FS)
    spectrogram = metric.spectrogram(original, original.copy())

    assert set(spectrogram) == {"original", "reconstructed"}
    assert spectrogram["original"].shape == spectrogram["reconstructed"].shape


def test_perceptual_metric_is_zero_for_identical_signals():
    metric = _perceptual_metric()
    original = np.sin(2 * np.pi * 440 * np.arange(PERCEPTUAL_SAMPLES) / PERCEPTUAL_FS)
    assert metric(metric.spectrogram(original, original.copy())) == 0.0


def test_perceptual_metric_is_non_negative_and_finite():
    metric = _perceptual_metric()
    original = np.sin(2 * np.pi * 440 * np.arange(PERCEPTUAL_SAMPLES) / PERCEPTUAL_FS)
    degraded = original.copy()
    degraded[1000:1400] = 0.0
    value = metric(metric.spectrogram(original, degraded))

    assert np.isfinite(value)
    assert value >= 0.0


def test_perceptual_calculator_locates_the_degraded_packet():
    packet_size = 64
    settings = attach(
        PerceptualCalculatorSettings(intorno_length=50),
        packet_size=packet_size,
        fs=PERCEPTUAL_FS,
    )
    track = (0.5 * np.sin(2 * np.pi * 440 * np.arange(packet_size * 80) / PERCEPTUAL_FS))
    track = track.astype(np.float32).reshape(-1, 1)
    reconstructed = track.copy()
    reconstructed[packet_size * 40 : packet_size * 41] = 0.0
    lost_samples = np.array([packet_size * 40])

    metric = PerceptualCalculator(settings).run(
        AudioStub(track),
        AudioStub(reconstructed),
        DataStub(lost_samples),
        "test",
    ).get_error()

    assert metric.shape == (81,)
    assert np.count_nonzero(metric) == 1
    assert metric[40] > 0.0
    np.testing.assert_allclose(
        metric[40], PERCEPTUAL_CALCULATOR_GOLDEN, rtol=1e-2, atol=1e-4
    )


@pytest.mark.parametrize("drop_index", [39, 41])
def test_perceptual_calculator_is_selective_about_the_dropped_packet(drop_index):
    packet_size = 64
    settings = attach(
        PerceptualCalculatorSettings(intorno_length=50),
        packet_size=packet_size,
        fs=PERCEPTUAL_FS,
    )
    track = (0.5 * np.sin(2 * np.pi * 440 * np.arange(packet_size * 80) / PERCEPTUAL_FS))
    track = track.astype(np.float32).reshape(-1, 1)
    reconstructed = track.copy()
    reconstructed[packet_size * drop_index : packet_size * (drop_index + 1)] = 0.0

    metric = PerceptualCalculator(settings).run(
        AudioStub(track),
        AudioStub(reconstructed),
        DataStub(np.array([packet_size * drop_index])),
        "test",
    ).get_error()

    assert np.count_nonzero(metric) == 1
    assert metric[drop_index] > 0.0