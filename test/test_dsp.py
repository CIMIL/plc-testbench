"""Unit tests for the DSP building blocks.

Covers the Linkwitz-Riley filters/crossover, the shared ``utils`` helpers and
the internal stages of the low-cost concealment algorithm. Golden values come
from a deterministic window (a sum of three sines at 440/1200/2100 Hz sampled
at 16 kHz) so that every stage of the concealment pipeline is reproducible.
"""

from __future__ import annotations

import numpy as np
import pytest

from plctestbench.filters import LinkwitzRileyCrossover, LinkwitzRileyFilter
from plctestbench.low_cost_concealment import LowCostConcealment
from plctestbench.utils import (
    ObjectFactory,
    compute_hash,
    extract_intorni,
    fade_in,
    fade_out,
    force_2d,
    force_single_loss_per_stimulus,
    is_loud_enough,
    leading_silence,
    recursive_split_audio,
    trailing_silence,
)

from test._helpers import AudioStub

FS = 16000
PACKET_SIZE = 64
WIN_SIZE = 200


# --------------------------------------------------------------------------- #
# Linkwitz-Riley filters
# --------------------------------------------------------------------------- #


def test_lowpass_passes_dc_and_highpass_rejects_it():
    dc = np.ones(4000)
    low = LinkwitzRileyFilter(4, 1000, FS, "low").filter(dc)
    high = LinkwitzRileyFilter(4, 1000, FS, "high").filter(dc)

    np.testing.assert_allclose(low[-10:, 0], np.ones(10), atol=1e-6)
    np.testing.assert_allclose(high[-10:, 0], np.zeros(10), atol=1e-6)


def _split_bands(signal):
    """Split ``signal`` with a fixed 1 kHz Linkwitz-Riley crossover."""
    return LinkwitzRileyCrossover(4, 1000, FS).split(signal)


@pytest.mark.parametrize(
    "frequency, louder_band",
    [(100.0, 0), (6000.0, 1)],
)
def test_crossover_splits_by_frequency(frequency, louder_band):
    time = np.arange(8000) / FS
    tone = np.sin(2 * np.pi * frequency * time)
    low, high = _split_bands(tone)

    energies = [np.sqrt(np.mean(low**2)), np.sqrt(np.mean(high**2))]
    assert energies[louder_band] > 10 * energies[1 - louder_band]


def test_crossover_returns_both_bands_with_the_input_shape():
    tone = np.sin(2 * np.pi * 440 * np.arange(1000) / FS)
    low, high = _split_bands(tone)

    assert low.shape == high.shape == (1000, 1)


def test_recursive_split_audio_produces_one_band_per_crossover_plus_one():
    tone = np.sin(2 * np.pi * 440 * np.arange(2000) / FS)
    crossovers = [
        LinkwitzRileyCrossover(4, 1000, FS),
        LinkwitzRileyCrossover(4, 3000, FS),
    ]
    bands = recursive_split_audio(tone, crossovers)

    assert len(bands) == len(crossovers) + 1
    assert all(band.shape == (2000, 1) for band in bands)


# --------------------------------------------------------------------------- #
# utils
# --------------------------------------------------------------------------- #


def test_force_2d_promotes_mono_audio():
    assert force_2d(np.zeros(5)).shape == (5, 1)
    assert force_2d(np.zeros((5, 2))).shape == (5, 2)


def test_extract_intorni_centers_the_window_on_the_loss():
    audio = np.arange(200, dtype=np.float32).reshape(200, 1)
    packet_idxs, intorni = extract_intorni(
        AudioStub(audio), [100, 101, 102], intorno_size=20, fs=1000, packet_size=10
    )

    assert packet_idxs == [10]
    assert len(intorni) == 1
    np.testing.assert_array_equal(intorni[0][:, 0], np.arange(91, 111))


def test_force_single_loss_per_stimulus_keeps_close_packets_together():
    selected = force_single_loss_per_stimulus([10, 11], 1000, 50, 10)
    assert selected == [10, 11]


def test_force_single_loss_per_stimulus_discards_packets_that_are_too_close():
    selected = force_single_loss_per_stimulus([0, 10, 20, 30], 1000, 50, 10)
    assert selected == [0]


def test_fade_in_ramps_the_start_of_the_buffer_in_place():
    audio = np.ones(100, dtype=np.float32)
    fade_in(audio, fs=1000, fade_in_time=10)

    np.testing.assert_allclose(audio[:10], np.linspace(0, 1, 10), atol=1e-6)
    np.testing.assert_allclose(audio[10:], np.ones(90), atol=1e-6)


def test_fade_out_ramps_the_end_of_the_buffer_in_place():
    audio = np.ones(100, dtype=np.float32)
    fade_out(audio, fs=1000, fade_out_time=10)

    np.testing.assert_allclose(audio[-10:], np.linspace(1, 0, 10), atol=1e-6)
    np.testing.assert_allclose(audio[:-10], np.ones(90), atol=1e-6)


def test_leading_and_trailing_silence_are_prepended_and_appended():
    audio = np.ones((20, 1), dtype=np.float32)

    padded_leading = leading_silence(audio, fs=1000, silence_time=10)
    padded_trailing = trailing_silence(audio, fs=1000, silence_time=10)

    assert padded_leading.shape == (30, 1)
    assert padded_trailing.shape == (30, 1)
    np.testing.assert_array_equal(padded_leading[:10], np.zeros((10, 1)))
    np.testing.assert_array_equal(padded_trailing[-10:], np.zeros((10, 1)))


def test_is_loud_enough_compares_against_the_reference():
    reference = np.full((100, 1), 0.5)
    assert is_loud_enough(np.full((100, 1), 0.5), reference, -10)
    assert not is_loud_enough(np.full((100, 1), 0.0001), reference, -10)


def test_compute_hash_is_deterministic_and_order_sensitive():
    assert compute_hash("abc") == compute_hash("abc")
    assert compute_hash("abc") != compute_hash("cba")


def test_object_factory_registers_and_creates_builders():
    factory = ObjectFactory()
    factory.register_builder("double", lambda value: value * 2)

    assert factory.create("double", 21) == 42
    with pytest.raises(ValueError):
        factory.create("missing")


# --------------------------------------------------------------------------- #
# LowCostConcealment internals
# --------------------------------------------------------------------------- #


def _lcc() -> LowCostConcealment:
    lcc = LowCostConcealment(
        max_frequency=4800,
        f_min=80,
        beta=1,
        n_m=2,
        fade_in_length=10,
        fade_out_length=0.5,
        extraction_length=2,
    )
    lcc.prepare_to_play(FS, PACKET_SIZE, 1)
    return lcc


def _window_buffer() -> np.ndarray:
    time = np.arange(WIN_SIZE) / FS
    signal = (
        0.5 * np.sin(2 * np.pi * 440 * time)
        + 0.3 * np.sin(2 * np.pi * 1200 * time)
        + 0.1 * np.sin(2 * np.pi * 2100 * time)
    )
    return signal.reshape(-1, 1)


def test_prepare_to_play_sets_window_and_lower_bound():
    lcc = _lcc()
    assert lcc._win_size == WIN_SIZE
    assert lcc._lower_bound == FS / (2 * 4800)
    assert lcc._window.shape == (WIN_SIZE, 1)


def test_pre_process_matches_golden():
    lcc = _lcc()
    pre_processed = lcc.pre_process(_window_buffer())
    np.testing.assert_allclose(
        pre_processed[:12, 0],
        np.array(
            [
                1.105228e-17, 9.516448e-02, 1.856216e-01, 2.680518e-01,
                3.397676e-01, 3.988897e-01, 4.443822e-01, 4.759480e-01,
                4.938213e-01, 4.985142e-01, 4.905818e-01, 4.704593e-01,
            ]
        ),
        rtol=1e-3,
        atol=1e-6,
    )


def test_zero_crossing_detect_matches_golden():
    lcc = _lcc()
    zero_crossings = lcc.zero_crossing_detect(lcc.pre_process(_window_buffer())[:, 0])
    np.testing.assert_array_equal(
        zero_crossings[:12], np.array([19, 38, 55, 73, 90, 108, 127, 146, 165, 183])
    )


def test_extract_returns_a_period_and_matches_golden():
    lcc = _lcc()
    buffer = _window_buffer()
    zero_crossings = lcc.zero_crossing_detect(lcc.pre_process(buffer)[:, 0])
    extracted = lcc.extract(buffer[:, 0], zero_crossings)

    assert len(extracted) == 164
    np.testing.assert_allclose(
        extracted[:8],
        np.array(
            [
                -0.381015, -0.425550, -0.470085, -0.614103,
                -0.654368, -0.626579, -0.575993, -0.528414,
            ]
        ),
        rtol=1e-3,
        atol=1e-6,
    )


def test_align_extends_the_period_to_the_extraction_length():
    lcc = _lcc()
    buffer = _window_buffer()
    zero_crossings = lcc.zero_crossing_detect(lcc.pre_process(buffer)[:, 0])
    extracted = lcc.extract(buffer[:, 0], zero_crossings)
    aligned = lcc.align(buffer[:, 0], extracted)

    assert len(aligned) == lcc._extraction_length * PACKET_SIZE
    np.testing.assert_allclose(
        aligned[:8],
        np.array(
            [
                -0.309767, -0.224627, -0.127422, -0.076552,
                -0.093083, -0.158718, -0.237316, -0.303776,
            ]
        ),
        rtol=1e-3,
        atol=1e-6,
    )


def test_extrapolate_and_fade_in_matches_golden():
    lcc = _lcc()
    buffer = _window_buffer()
    zero_crossings = lcc.zero_crossing_detect(lcc.pre_process(buffer)[:, 0])
    aligned = lcc.align(
        buffer[:, 0], lcc.extract(buffer[:, 0], zero_crossings)
    )
    faded = lcc.extrapolate_and_fade_in(buffer[:, 0], aligned)

    np.testing.assert_allclose(
        faded[:10],
        np.array(
            [
                0.073853, 0.127320, 0.181669, 0.217159, 0.209356,
                0.149019, 0.046715, -0.081364, -0.233544, -0.429289,
            ]
        ),
        rtol=1e-3,
        atol=1e-6,
    )


def test_fade_out_matches_golden():
    lcc = _lcc()
    buffer = _window_buffer()
    zero_crossings = lcc.zero_crossing_detect(lcc.pre_process(buffer)[:, 0])
    aligned = lcc.align(buffer[:, 0], lcc.extract(buffer[:, 0], zero_crossings))

    next_block = np.linspace(-0.4, 0.4, PACKET_SIZE)
    faded_out = lcc.fade_out(next_block.copy(), aligned[PACKET_SIZE : 2 * PACKET_SIZE])

    np.testing.assert_allclose(
        faded_out[:10],
        np.array(
            [
                0.345428, 0.176998, 0.053227, 0.017012, 0.071521,
                0.175693, 0.264791, 0.283642, 0.212975, 0.074531,
            ]
        ),
        rtol=1e-3,
        atol=1e-6,
    )


def test_process_returns_zeros_when_no_period_can_be_extracted():
    lcc = _lcc()  # the window starts as silence, so there are no zero crossings
    buffer = np.ones((PACKET_SIZE, 1))
    output = lcc.process(buffer, is_valid=False)

    np.testing.assert_array_equal(output, np.zeros((PACKET_SIZE, 1)))


def test_process_passes_valid_audio_through_unchanged():
    lcc = _lcc()
    buffer = np.full((PACKET_SIZE, 1), 0.3)
    output = lcc.process(buffer, is_valid=True)

    np.testing.assert_allclose(output, buffer)


def test_process_concealment_matches_golden():
    lcc = _lcc()
    track = (np.random.RandomState(0).randn(PACKET_SIZE * 6, 1) * 0.2).astype(np.float32)
    for packet in range(3):
        lcc.process(track[packet * PACKET_SIZE : (packet + 1) * PACKET_SIZE], True)

    concealed = lcc.process(
        track[3 * PACKET_SIZE : 4 * PACKET_SIZE], is_valid=False
    )

    assert concealed.shape == (PACKET_SIZE, 1)
    np.testing.assert_allclose(
        concealed[:10, 0],
        np.array(
            [
                0.306865, 0.553828, 0.772132, 0.725236, 0.791696,
                0.717901, 0.930081, 0.617420, 0.359468, -0.153983,
            ]
        ),
        rtol=1e-3,
        atol=1e-6,
    )