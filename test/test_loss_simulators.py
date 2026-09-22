"""Unit tests for the packet loss simulators.

Every simulator is exercised through its ``run`` entry point with a seeded
mock input; the expected outputs are golden values captured from the pinned
implementation (see the module docstrings in ``plctestbench.loss_simulator``
for the intended semantics).
"""

from __future__ import annotations

import numpy as np
import pytest

from plctestbench.loss_simulator import (
    BinomialPLS,
    CustomMaskPLS,
    GilbertElliotPLS,
    MetronomePLS,
    PacketLossSimulator,
)
from plctestbench.settings import (
    BinomialPLSSettings,
    CustomMaskPLSSettings,
    GilbertElliotPLSSettings,
    MetronomePLSSettings,
)

from test._helpers import attach


def _run(simulator_class, simulator_settings, num_samples: int) -> np.ndarray:
    attach(simulator_settings)
    return simulator_class(simulator_settings).run(num_samples, "test")


# --------------------------------------------------------------------------- #
# Base class contract
# --------------------------------------------------------------------------- #


def test_base_simulator_tick_must_be_implemented():
    settings = attach(BinomialPLSSettings(seed=1, packet_size=4, per=0.5))
    base = PacketLossSimulator(settings)
    with pytest.raises(NotImplementedError):
        base.tick()


def test_base_simulator_run_holds_the_packet_decision_for_the_whole_packet():
    """A loss decision taken at a packet boundary applies to every sample."""
    settings = attach(CustomMaskPLSSettings(seed=1, packet_size=3, mask="10"))
    lost = PacketLossSimulator.run(CustomMaskPLS(settings), 9, "test")
    np.testing.assert_array_equal(lost, np.array([0, 1, 2]))


def test_string_representation_contains_class_and_seed():
    settings = attach(BinomialPLSSettings(seed=7, packet_size=4, per=0.5))
    assert str(BinomialPLS(settings)) == "BinomialPLS_s7"


# --------------------------------------------------------------------------- #
# BinomialPLS
# --------------------------------------------------------------------------- #


def test_binomial_matches_golden_output():
    settings = BinomialPLSSettings(seed=1, packet_size=10, per=0.5)
    lost = _run(BinomialPLS, settings, 50)
    np.testing.assert_array_equal(lost, np.array(list(range(10)) + list(range(20, 50))))


def test_binomial_is_reproducible_for_the_same_seed():
    first = _run(BinomialPLS, BinomialPLSSettings(seed=1, packet_size=10, per=0.5), 50)
    second = _run(BinomialPLS, BinomialPLSSettings(seed=1, packet_size=10, per=0.5), 50)
    np.testing.assert_array_equal(first, second)


def test_binomial_boundary_probabilities():
    assert _run(BinomialPLS, BinomialPLSSettings(seed=1, packet_size=1, per=0.0), 32).size == 0
    assert _run(BinomialPLS, BinomialPLSSettings(seed=1, packet_size=1, per=1.0), 32).size == 32


# --------------------------------------------------------------------------- #
# MetronomePLS
# --------------------------------------------------------------------------- #


def test_metronome_matches_golden_output():
    settings = MetronomePLSSettings(
        seed=1, packet_size=1, period=10, duration=3, offset=0
    )
    lost = _run(MetronomePLS, settings, 20)
    np.testing.assert_array_equal(lost, np.array([0, 1, 9, 10, 11, 19]))


def test_metronome_offset_shifts_the_burst():
    settings = MetronomePLSSettings(
        seed=1, packet_size=1, period=10, duration=3, offset=4
    )
    lost = _run(MetronomePLS, settings, 24)
    np.testing.assert_array_equal(lost, np.array([3, 4, 5, 13, 14, 15, 23]))


def test_metronome_is_packet_sized():
    settings = MetronomePLSSettings(
        seed=1, packet_size=2, period=3, duration=1, offset=0
    )
    lost = _run(MetronomePLS, settings, 24)
    np.testing.assert_array_equal(lost, np.array([4, 5, 10, 11, 16, 17, 22, 23]))


# --------------------------------------------------------------------------- #
# GilbertElliotPLS
# --------------------------------------------------------------------------- #


def test_gilbert_elliot_matches_golden_output():
    lost = _run(GilbertElliotPLS, GilbertElliotPLSSettings(seed=1, packet_size=1), 20)
    np.testing.assert_array_equal(lost, np.array([1, 2, 3]))


def test_gilbert_elliot_is_reproducible_for_the_same_seed():
    first = _run(GilbertElliotPLS, GilbertElliotPLSSettings(seed=3, packet_size=1), 100)
    second = _run(GilbertElliotPLS, GilbertElliotPLSSettings(seed=3, packet_size=1), 100)
    np.testing.assert_array_equal(first, second)


def test_gilbert_elliot_always_lost_configuration():
    """k=0 (GOOD always loses) and p=0 (never leave GOOD): every packet lost."""
    settings = GilbertElliotPLSSettings(
        seed=1, packet_size=1, p=0.0, r=1.0, h=0.5, k=0.0
    )
    lost = _run(GilbertElliotPLS, settings, 16)
    np.testing.assert_array_equal(lost, np.arange(16))


def test_gilbert_elliot_never_lost_configuration():
    """k=1 (GOOD never loses) and p=0 (never leave GOOD): no packet lost."""
    settings = GilbertElliotPLSSettings(
        seed=1, packet_size=1, p=0.0, r=1.0, h=0.5, k=1.0
    )
    lost = _run(GilbertElliotPLS, settings, 16)
    assert lost.size == 0


# --------------------------------------------------------------------------- #
# CustomMaskPLS
# --------------------------------------------------------------------------- #


def test_custom_mask_matches_golden_output():
    settings = CustomMaskPLSSettings(seed=1, packet_size=1, mask="101")
    lost = _run(CustomMaskPLS, settings, 6)
    np.testing.assert_array_equal(lost, np.array([0, 2]))


def test_custom_mask_invert_flips_the_meaning():
    settings = CustomMaskPLSSettings(seed=1, packet_size=1, mask="101", invert=True)
    lost = _run(CustomMaskPLS, settings, 3)
    np.testing.assert_array_equal(lost, np.array([1]))


def test_custom_mask_stops_dropping_past_the_end_of_the_mask():
    settings = CustomMaskPLSSettings(seed=1, packet_size=1, mask="1")
    lost = _run(CustomMaskPLS, settings, 5)
    np.testing.assert_array_equal(lost, np.array([0]))


def test_custom_mask_ignores_non_binary_characters():
    settings = CustomMaskPLSSettings(seed=1, packet_size=1, mask="1x1")
    lost = _run(CustomMaskPLS, settings, 3)
    np.testing.assert_array_equal(lost, np.array([0, 2]))