"""Unit tests for the PLC algorithms.

Each algorithm is driven on a short, seeded mock input with a known loss
pattern. Expected outputs are either derived analytically (``ZerosPLC``
zeroes the lost packets, ``LastPacketPLC`` repeats the previous packet) or
captured as golden values from the pinned implementation (``LowCostPLC``,
``AdvancedPLC``).

``BurgPLC`` and ``ExternalPLC`` depend on native packages that are not
available on all platforms; those paths are skipped (or asserted to raise a
descriptive ``ImportError``) rather than silently passing.
"""

from __future__ import annotations

import numpy as np
import pytest

from plctestbench.plc_algorithm import (
    AdvancedPLC,
    BurgPLC,
    ExternalPLC,
    LastPacketPLC,
    LowCostPLC,
    ZerosPLC,
)
from plctestbench.settings import (
    AdvancedPLCSettings,
    BurgPLCSettings,
    ClipStrategy,
    ExternalPLCSettings,
    LastPacketPLCSettings,
    LowCostPLCSettings,
    ManualCrossfadeSettings,
    StereoImageType,
    ZerosPLCSettings,
)

from test._helpers import FS, attach, seeded_track

PACKET_SIZE = 4

#: Mock input: a 12-sample mono ramp split into 3 packets of 4 samples.
MOCK_TRACK = (np.arange(12, dtype=np.float32) / 16.0).reshape(12, 1)
#: Packet 1 (samples 4..7) is lost.
MOCK_LOST = np.array([4, 5, 6, 7])

LAST_PACKET_GOLDEN = np.array(
    [0.0, 0.0625, 0.125, 0.1875, 0.0, 0.0625, 0.125, 0.1875, 0.5, 0.5625, 0.625, 0.6875]
)
LAST_PACKET_MIRROR_X_GOLDEN = np.array(
    [0.0, 0.0625, 0.125, 0.1875, 0.1875, 0.125, 0.0625, 0.0, 0.5, 0.5625, 0.625, 0.6875]
)
LAST_PACKET_MIRROR_XY_GOLDEN = np.array(
    [0.0, 0.0625, 0.125, 0.1875, 0.1875, 0.25, 0.3125, 0.375, 0.5, 0.5625, 0.625, 0.6875]
)


def _plc_settings(settings, **extra):
    return attach(settings, packet_size=PACKET_SIZE, fs=FS, context_length=0.5, **extra)


# --------------------------------------------------------------------------- #
# ZerosPLC
# --------------------------------------------------------------------------- #


def test_zeros_plc_matches_analytical_expected_output():
    plc = ZerosPLC(_plc_settings(ZerosPLCSettings()))
    output = plc.run(MOCK_TRACK, MOCK_LOST, "test")

    expected = LAST_PACKET_GOLDEN.copy()  # same valid samples...
    expected[4:8] = 0.0  # ...with the lost packet zeroed
    np.testing.assert_allclose(output, expected.reshape(12, 1), atol=1e-6)


def test_zeros_plc_zeroes_exactly_the_lost_packet():
    plc = ZerosPLC(_plc_settings(ZerosPLCSettings()))
    output = plc.run(MOCK_TRACK, MOCK_LOST, "test")

    np.testing.assert_array_equal(output[4:8], np.zeros((4, 1)))
    assert np.count_nonzero(output[4:8]) == 0


def test_zeros_plc_preserves_valid_packets_bit_for_bit():
    plc = ZerosPLC(_plc_settings(ZerosPLCSettings()))
    output = plc.run(MOCK_TRACK, MOCK_LOST, "test")

    np.testing.assert_array_equal(output[:4], MOCK_TRACK[:4])
    np.testing.assert_array_equal(output[8:], MOCK_TRACK[8:])


def test_plc_output_keeps_the_original_length_when_not_packet_aligned():
    track = np.arange(10, dtype=np.float32).reshape(10, 1) / 16.0
    plc = ZerosPLC(_plc_settings(ZerosPLCSettings()))
    output = plc.run(track, MOCK_LOST, "test")

    assert output.shape == track.shape


def test_plc_output_is_float32():
    plc = ZerosPLC(_plc_settings(ZerosPLCSettings()))
    assert plc.run(MOCK_TRACK, MOCK_LOST, "test").dtype == np.float32


def test_plc_returns_the_whole_track_when_nothing_is_lost():
    plc = ZerosPLC(_plc_settings(ZerosPLCSettings()))
    output = plc.run(MOCK_TRACK, np.array([1000]), "test")
    np.testing.assert_allclose(output, MOCK_TRACK, atol=1e-6)


def test_fade_in_longer_than_the_packet_raises():
    settings = ZerosPLCSettings(fade_in=[ManualCrossfadeSettings(length=10)])
    attach(settings, packet_size=4, fs=FS, context_length=0.5)
    with pytest.raises(ValueError):
        ZerosPLC(settings)


# --------------------------------------------------------------------------- #
# LastPacketPLC
# --------------------------------------------------------------------------- #


def test_last_packet_plc_repeats_the_previous_packet():
    plc = LastPacketPLC(_plc_settings(LastPacketPLCSettings()))
    output = plc.run(MOCK_TRACK, MOCK_LOST, "test")
    np.testing.assert_allclose(output, LAST_PACKET_GOLDEN.reshape(12, 1), atol=1e-6)


def test_last_packet_plc_mirror_x_flips_the_packet():
    plc = LastPacketPLC(_plc_settings(LastPacketPLCSettings(mirror_x=True)))
    output = plc.run(MOCK_TRACK, MOCK_LOST, "test")
    np.testing.assert_allclose(
        output, LAST_PACKET_MIRROR_X_GOLDEN.reshape(12, 1), atol=1e-6
    )


def test_last_packet_plc_mirror_x_and_y_flip_around_the_first_sample():
    plc = LastPacketPLC(
        _plc_settings(LastPacketPLCSettings(mirror_x=True, mirror_y=True))
    )
    output = plc.run(MOCK_TRACK, MOCK_LOST, "test")
    np.testing.assert_allclose(
        output, LAST_PACKET_MIRROR_XY_GOLDEN.reshape(12, 1), atol=1e-6
    )


def test_last_packet_plc_is_deterministic():
    settings = LastPacketPLCSettings(mirror_x=True, mirror_y=True)
    first = LastPacketPLC(_plc_settings(settings)).run(MOCK_TRACK, MOCK_LOST, "test")
    second = LastPacketPLC(_plc_settings(settings)).run(MOCK_TRACK, MOCK_LOST, "test")
    np.testing.assert_array_equal(first, second)


#: Mock input whose mirrored packet exceeds the [-1, 1] range (sample 5 -> -2.7).
CLIPPING_MOCK_TRACK = np.array(
    [[0.9], [-0.9], [0.9], [-0.9], [0.1], [0.1], [0.1], [0.1]], dtype=np.float32
)


def _clipped_prediction(strategy: ClipStrategy) -> np.ndarray:
    settings = _plc_settings(
        LastPacketPLCSettings(mirror_x=True, mirror_y=True, clip_strategy=strategy)
    )
    output = LastPacketPLC(settings).run(CLIPPING_MOCK_TRACK, MOCK_LOST, "test")
    return output[4:8, 0]


def test_last_packet_plc_subtract_strategy_shifts_the_excess():
    """``subtract`` removes the excess of the first out-of-range sample from the
    remainder of the packet, so the waveform shape is preserved."""
    np.testing.assert_allclose(
        _clipped_prediction(ClipStrategy.subtract),
        np.array([-0.9, -1.0, 0.8, -1.0]),
        atol=1e-6,
    )


def test_last_packet_plc_clip_strategy_limits_every_sample():
    np.testing.assert_allclose(
        _clipped_prediction(ClipStrategy.clip),
        np.array([-0.9, -1.0, -0.9, -1.0]),
        atol=1e-6,
    )


def test_last_packet_plc_clip_strategies_are_no_ops_when_in_range():
    for strategy in (ClipStrategy.subtract, ClipStrategy.clip):
        settings = _plc_settings(
            LastPacketPLCSettings(mirror_x=True, mirror_y=True, clip_strategy=strategy)
        )
        output = LastPacketPLC(settings).run(MOCK_TRACK, MOCK_LOST, "test")
        np.testing.assert_allclose(
            output, LAST_PACKET_MIRROR_XY_GOLDEN.reshape(12, 1), atol=1e-6
        )


# --------------------------------------------------------------------------- #
# LowCostPLC
# --------------------------------------------------------------------------- #

LOW_COST_PACKET_SIZE = 64
#: Golden output for the concealed packet (samples 192..223).
LOW_COST_GOLDEN = np.array(
    [
        0.306865, 0.553828, 0.772132, 0.725236, 0.791696, 0.717901, 0.930081,
        0.617420, 0.359468, -0.153983, 0.107850, -0.134867, 0.006366, -0.127169,
        0.135287, 0.115318, -0.041660, 0.079201, -0.218612, -0.298252, 0.087878,
        0.033335, 0.127006, 0.476629, 0.188896, -0.182564, 0.223403, -0.263181,
        -0.092317, -0.013648, 0.342669, -0.148951,
    ]
)


def _low_cost_input():
    settings = attach(LowCostPLCSettings(), packet_size=LOW_COST_PACKET_SIZE, fs=FS,
                      context_length=100)
    track = seeded_track(LOW_COST_PACKET_SIZE * 6, seed=0, amplitude=0.2)
    return settings, track


def test_low_cost_plc_matches_golden_concealment():
    settings, track = _low_cost_input()
    output = LowCostPLC(settings).run(track, np.array([192, 193]), "test")

    assert output.shape == track.shape
    np.testing.assert_allclose(
        output[192:224, 0], LOW_COST_GOLDEN, rtol=1e-4, atol=1e-6
    )


def test_low_cost_plc_passes_through_packets_before_any_loss():
    settings, track = _low_cost_input()
    output = LowCostPLC(settings).run(track, np.array([192, 193]), "test")

    np.testing.assert_allclose(output[:64], track[:64], rtol=1e-6, atol=1e-6)


def test_low_cost_plc_is_deterministic():
    settings, track = _low_cost_input()
    first = LowCostPLC(settings).run(track, np.array([192, 193]), "test")
    second = LowCostPLC(settings).run(track, np.array([192, 193]), "test")
    np.testing.assert_array_equal(first, second)


# --------------------------------------------------------------------------- #
# AdvancedPLC
# --------------------------------------------------------------------------- #


def _linked_advanced_settings(band_settings=None):
    band = attach(
        band_settings if band_settings is not None else LastPacketPLCSettings(),
        packet_size=PACKET_SIZE,
        fs=FS,
        context_length=0.5,
    )
    advanced = AdvancedPLCSettings(
        band_settings={"linked": [band]}, frequencies={"linked": []}
    )
    advanced.add("packet_size", PACKET_SIZE)
    advanced.add("fs", FS)
    return attach(advanced)


def test_advanced_plc_single_band_matches_the_band_algorithm():
    output = AdvancedPLC(_linked_advanced_settings()).run(MOCK_TRACK, MOCK_LOST, "test")
    np.testing.assert_allclose(output, LAST_PACKET_GOLDEN.reshape(12, 1), atol=1e-6)


def test_advanced_plc_multiband_preserves_shape_and_is_finite():
    band_settings = [
        _plc_settings(LastPacketPLCSettings()),
        _plc_settings(LastPacketPLCSettings(mirror_x=True)),
    ]
    multiband = AdvancedPLCSettings(
        band_settings={"linked": band_settings},
        frequencies={"linked": [1000]},
    )
    multiband.add("packet_size", PACKET_SIZE)
    multiband.add("fs", FS)
    attach(multiband)

    output = AdvancedPLC(multiband).run(MOCK_TRACK, MOCK_LOST, "test")

    assert output.shape == MOCK_TRACK.shape
    assert output.dtype == np.float32
    assert np.isfinite(output).all()


def test_advanced_plc_is_deterministic():
    first = AdvancedPLC(_linked_advanced_settings()).run(MOCK_TRACK, MOCK_LOST, "test")
    second = AdvancedPLC(_linked_advanced_settings()).run(MOCK_TRACK, MOCK_LOST, "test")
    np.testing.assert_array_equal(first, second)


def test_advanced_plc_mid_side_round_trip_preserves_amplitude():
    """Mid/side encoding followed by decoding must be gain-transparent.

    This guards the ``MidSideCodec.decode`` scaling: the previous ``* 2``
    doubled the reconstructed track in mid/side mode.
    """
    mid_band = _plc_settings(ZerosPLCSettings())
    side_band = _plc_settings(ZerosPLCSettings())
    settings = AdvancedPLCSettings(
        band_settings={"mid": [mid_band], "side": [side_band]},
        frequencies={"mid": [], "side": []},
        stereo_image_processing=StereoImageType.mid_side,
        channel_link=False,
    )
    settings.add("packet_size", PACKET_SIZE)
    settings.add("fs", FS)
    attach(settings)

    stereo = (np.arange(16, dtype=np.float32) / 16.0).reshape(8, 2)
    output = AdvancedPLC(settings).run(stereo, MOCK_LOST, "test")

    expected = stereo.copy()
    expected[4:8] = 0.0  # ZerosPLC on the lost packet
    np.testing.assert_allclose(output, expected, atol=1e-6)


# --------------------------------------------------------------------------- #
# Platform-gated algorithms
# --------------------------------------------------------------------------- #


def _burg_or_external_settings(settings_class):
    settings = settings_class()
    return attach(settings, packet_size=LOW_COST_PACKET_SIZE, fs=FS, context_length=100)


def test_burg_plc_reports_missing_native_dependency():
    from plctestbench import plc_algorithm

    settings = _burg_or_external_settings(BurgPLCSettings)
    if plc_algorithm.BurgBasic is None:
        with pytest.raises(ImportError):
            BurgPLC(settings)
    else:  # pragma: no cover - only on platforms where burg is installed
        assert BurgPLC(settings) is not None


def test_external_plc_reports_missing_native_dependency():
    from plctestbench import plc_algorithm

    settings = _burg_or_external_settings(ExternalPLCSettings)
    if plc_algorithm.BasePlcTemplate is None:
        with pytest.raises(ImportError):
            ExternalPLC(settings)
    else:  # pragma: no cover - only when the C++ template is installed
        assert ExternalPLC(settings) is not None