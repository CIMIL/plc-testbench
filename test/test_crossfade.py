"""Unit tests for the crossfade implementations.

The window shapes are asserted against closed-form golden values, while the
``Crossfade`` state machine (``start``/``ongoing``/``idx`` bookkeeping and the
short-buffer padding path) is asserted against golden outputs for mock inputs.
"""

from __future__ import annotations

import numpy as np
import pytest

from plctestbench.crossfade import (
    Crossfade,
    MultibandCrossfade,
    power_crossfade,
    sinusoidal_crossfade,
)
from plctestbench.settings import (
    CrossfadeFunction,
    CrossfadeSettings,
    CrossfadeType,
    ManualCrossfadeSettings,
    NoCrossfadeSettings,
    ZerosPLCSettings,
)

from test._helpers import attach

# --------------------------------------------------------------------------- #
# Window generators
# --------------------------------------------------------------------------- #


def test_power_crossfade_window_matches_golden():
    settings = CrossfadeSettings(
        length=4,
        function=CrossfadeFunction.power,
        exponent=2.0,
        type=CrossfadeType.power,
    )
    np.testing.assert_allclose(
        power_crossfade(settings, 4),
        np.array([0.0, 1.0 / 9.0, 4.0 / 9.0, 1.0]),
        atol=1e-6,
    )


def test_power_crossfade_linear_exponent_one():
    settings = CrossfadeSettings(
        length=4,
        function=CrossfadeFunction.power,
        exponent=1.0,
        type=CrossfadeType.power,
    )
    np.testing.assert_allclose(
        power_crossfade(settings, 4), np.linspace(0.0, 1.0, 4), atol=1e-6
    )


def test_sinusoidal_crossfade_window_matches_golden():
    settings = CrossfadeSettings(
        length=4,
        function=CrossfadeFunction.sinusoidal,
        exponent=1.0,
        type=CrossfadeType.power,
    )
    np.testing.assert_allclose(
        sinusoidal_crossfade(settings, 4),
        np.array([0.0, 0.5, np.sqrt(3.0) / 2.0, 1.0]),
        atol=1e-6,
    )


# --------------------------------------------------------------------------- #
# Crossfade
# --------------------------------------------------------------------------- #


def _crossfade_settings(**kwargs):
    """A PLC-level settings object carrying ``fs`` plus its crossfade settings."""
    settings = attach(
        ZerosPLCSettings(), packet_size=4, fs=1000, context_length=0.5, **kwargs
    )
    return settings, settings.get("crossfade")[0]


def test_crossfade_complementary_window_for_amplitude_type():
    _, crossfade_settings = _crossfade_settings()
    manual = ManualCrossfadeSettings(length=4, type=CrossfadeType.amplitude)
    settings = attach(ZerosPLCSettings(), packet_size=4, fs=1000, context_length=0.5)
    crossfade = Crossfade(settings, manual)

    np.testing.assert_allclose(
        crossfade.crossfade_buffer_a, np.array([0.0, 1 / 3, 2 / 3, 1.0]), atol=1e-6
    )
    np.testing.assert_allclose(
        crossfade.crossfade_buffer_b, np.array([1.0, 2 / 3, 1 / 3, 0.0]), atol=1e-6
    )


def test_crossfade_complementary_window_for_power_type():
    settings = attach(ZerosPLCSettings(), packet_size=4, fs=1000, context_length=0.5)
    manual = ManualCrossfadeSettings(
        length=4, exponent=1.0, type=CrossfadeType.power
    )
    crossfade = Crossfade(settings, manual)

    window_a = np.linspace(0.0, 1.0, 4)
    np.testing.assert_allclose(
        crossfade.crossfade_buffer_b, np.sqrt(1.0 - window_a**2), atol=1e-6
    )


def test_crossfade_is_transparent_before_start():
    settings = attach(ZerosPLCSettings(), packet_size=4, fs=1000, context_length=0.5)
    crossfade = Crossfade(settings, ManualCrossfadeSettings(length=4))
    buffer = np.full((4, 1), 5.0)

    assert not crossfade.ongoing()
    np.testing.assert_array_equal(crossfade(np.ones((4, 1)), buffer), buffer)


def test_crossfade_blends_prediction_and_buffer_matching_golden():
    settings = attach(ZerosPLCSettings(), packet_size=4, fs=1000, context_length=0.5)
    crossfade = Crossfade(
        settings, ManualCrossfadeSettings(length=4, type=CrossfadeType.amplitude)
    )
    crossfade.start()
    assert crossfade.ongoing()

    output = crossfade(np.ones((4, 1)), np.zeros((4, 1)))

    np.testing.assert_allclose(
        output, np.array([[1.0], [2 / 3], [1 / 3], [0.0]]), atol=1e-6
    )
    assert crossfade.idx == 4
    assert not crossfade.ongoing()


def test_crossfade_defaults_buffer_to_zeros_when_omitted():
    settings = attach(ZerosPLCSettings(), packet_size=4, fs=1000, context_length=0.5)
    crossfade = Crossfade(
        settings, ManualCrossfadeSettings(length=4, type=CrossfadeType.amplitude)
    )
    crossfade.start()

    output = crossfade(np.ones((4, 1)))
    np.testing.assert_allclose(
        output, np.array([[1.0], [2 / 3], [1 / 3], [0.0]]), atol=1e-6
    )


def test_crossfade_pads_windows_when_buffer_outlasts_the_window():
    settings = attach(ZerosPLCSettings(), packet_size=4, fs=1000, context_length=0.5)
    crossfade = Crossfade(
        settings, ManualCrossfadeSettings(length=10, type=CrossfadeType.amplitude)
    )
    crossfade.start()

    output = crossfade(np.ones((12, 1)), np.zeros((12, 1)))

    assert output.shape == (12, 1)
    # The extra samples are padded with b=0 / a=1, so their contribution is 0.
    np.testing.assert_allclose(output[10:], np.zeros((2, 1)), atol=1e-6)
    expected = 1.0 - np.linspace(0.0, 1.0, 10)
    np.testing.assert_allclose(output[:10, 0], expected, atol=1e-6)


def test_crossfade_returns_buffer_once_the_window_is_exhausted():
    settings = attach(ZerosPLCSettings(), packet_size=4, fs=1000, context_length=0.5)
    crossfade = Crossfade(
        settings, ManualCrossfadeSettings(length=4, type=CrossfadeType.amplitude)
    )
    crossfade.start()
    crossfade(np.ones((4, 1)), np.zeros((4, 1)))

    buffer = np.full((4, 1), 2.0)
    assert not crossfade.ongoing()
    np.testing.assert_array_equal(crossfade(np.ones((4, 1)), buffer), buffer)


def test_crossfade_supports_exactly_power_and_sinusoidal_windows():
    assert {function.name for function in CrossfadeFunction} == {"power", "sinusoidal"}


# --------------------------------------------------------------------------- #
# MultibandCrossfade
# --------------------------------------------------------------------------- #


def _multiband_settings(crossfade_settings):
    return attach(
        ZerosPLCSettings(crossfade_frequencies=[1000], crossfade=crossfade_settings),
        packet_size=4,
        fs=16000,
        context_length=0.5,
    )


def test_multiband_requires_one_crossfade_per_band():
    # PLCSettings-validated settings always build one crossfade per band, so a
    # hand-made, mismatched list is the only way to reach the guard.
    settings = attach(
        ZerosPLCSettings(crossfade_frequencies=[1000]),
        packet_size=4,
        fs=16000,
        context_length=0.5,
    )
    with pytest.raises(AssertionError):
        MultibandCrossfade(settings, [NoCrossfadeSettings()])


def test_multiband_builds_one_crossfade_per_band():
    settings = _multiband_settings(
        [ManualCrossfadeSettings(length=1), ManualCrossfadeSettings(length=1)]
    )
    multiband = MultibandCrossfade(settings, settings.get("crossfade"))
    assert len(multiband.crossfades) == 2


def test_multiband_start_and_ongoing_delegate_to_bands():
    settings = _multiband_settings(
        [ManualCrossfadeSettings(length=1), ManualCrossfadeSettings(length=1)]
    )
    multiband = MultibandCrossfade(settings, settings.get("crossfade"))
    assert not multiband.ongoing()

    multiband.start()
    assert multiband.ongoing()


def test_multiband_output_shape_and_finiteness():
    settings = _multiband_settings(
        [ManualCrossfadeSettings(length=1), ManualCrossfadeSettings(length=1)]
    )
    multiband = MultibandCrossfade(settings, settings.get("crossfade"))
    multiband.start()

    prediction = np.ones((16, 1))
    buffer = np.zeros((16, 1))
    output = multiband(prediction, buffer)

    assert output.shape == prediction.shape
    assert np.isfinite(output).all()


def test_multiband_zero_length_crossfade_is_not_ongoing():
    settings = _multiband_settings([NoCrossfadeSettings(), NoCrossfadeSettings()])
    multiband = MultibandCrossfade(settings, settings.get("crossfade"))
    multiband.start()
    assert not multiband.ongoing()