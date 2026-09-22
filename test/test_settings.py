"""Unit tests for the settings objects.

The settings classes are the package's validation layer: they are what stops an
invalid experiment configuration before any audio is processed. These tests
cover the validation bounds, the dictionary round-trip, cloning/inheritance and
the derived-settings modifiers.
"""

from __future__ import annotations

from typing import cast

import pytest

from plctestbench.settings import (
    AdvancedPLCSettings,
    BinomialPLSSettings,
    CrossfadeFunction,
    CrossfadeSettings,
    CrossfadeType,
    CustomMaskPLSSettings,
    GilbertElliotPLSSettings,
    LastPacketPLCSettings,
    LinearCrossfadeSettings,
    LowCostPLCSettings,
    NoCrossfadeSettings,
    Settings,
    ZerosPLCSettings,
)

from test._helpers import attach


# --------------------------------------------------------------------------- #
# Validation bounds
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("per", [-0.1, 1.1])
def test_binomial_rejects_out_of_range_per(per):
    with pytest.raises(AssertionError):
        BinomialPLSSettings(seed=1, packet_size=4, per=per)


def test_binomial_rejects_non_positive_packet_size():
    with pytest.raises(AssertionError):
        BinomialPLSSettings(seed=1, packet_size=0, per=0.5)


def test_binomial_rejects_non_numeric_per():
    # ``per`` is annotated as float; the cast models a caller passing garbage
    # that only the runtime validation can catch.
    with pytest.raises(AssertionError):
        BinomialPLSSettings(seed=1, packet_size=4, per=cast(float, "half"))


@pytest.mark.parametrize("field", ["p", "r", "h", "k"])
def test_gilbert_elliot_rejects_out_of_range_probabilities(field):
    kwargs = {"seed": 1, "packet_size": 4, "p": 0.1, "r": 0.1, "h": 0.1, "k": 0.1}
    kwargs[field] = 1.5
    with pytest.raises(AssertionError):
        GilbertElliotPLSSettings(**kwargs)


def test_custom_mask_rejects_non_positive_packet_size():
    with pytest.raises(AssertionError):
        CustomMaskPLSSettings(seed=1, packet_size=0, mask="1")


def test_crossfade_rejects_negative_length():
    with pytest.raises(AssertionError):
        CrossfadeSettings(
            length=-1.0,
            function=CrossfadeFunction.power,
            exponent=1.0,
            type=CrossfadeType.power,
        )


def test_crossfade_rejects_negative_exponent():
    with pytest.raises(AssertionError):
        CrossfadeSettings(
            length=10.0,
            function=CrossfadeFunction.power,
            exponent=-1.0,
            type=CrossfadeType.power,
        )


# --------------------------------------------------------------------------- #
# PLCSettings frequency / band validation
# --------------------------------------------------------------------------- #


def test_plc_settings_defaults_to_one_no_crossfade_band():
    settings = ZerosPLCSettings()
    assert len(settings.get("crossfade")) == 1
    assert isinstance(settings.get("crossfade")[0], NoCrossfadeSettings)


def test_plc_settings_builds_one_band_per_frequency_plus_one():
    settings = ZerosPLCSettings(crossfade_frequencies=[1000, 3000])
    assert len(settings.get("crossfade")) == 3


def test_plc_settings_rejects_unsorted_frequencies():
    with pytest.raises(AssertionError):
        ZerosPLCSettings(crossfade_frequencies=[3000, 1000])


def test_plc_settings_rejects_frequencies_above_the_audio_band():
    with pytest.raises(AssertionError):
        ZerosPLCSettings(crossfade_frequencies=[21000])


def test_plc_settings_rejects_negative_frequencies():
    with pytest.raises(AssertionError):
        ZerosPLCSettings(crossfade_frequencies=[-10])


def test_plc_settings_rejects_a_band_count_mismatch():
    with pytest.raises(AssertionError):
        ZerosPLCSettings(
            crossfade=[NoCrossfadeSettings()], crossfade_frequencies=[1000]
        )


def test_set_crossfade_frequencies_expands_the_band_list():
    base = ZerosPLCSettings(crossfade_frequencies=[1000])
    expanded = base.set_crossfade_frequencies([1000, 3000])

    assert len(expanded.get("crossfade")) == 3
    assert base.get("crossfade_frequencies") == [1000]  # original untouched


def test_low_cost_rejects_max_frequency_below_min_frequency():
    with pytest.raises(AssertionError):
        LowCostPLCSettings(max_frequency=50, f_min=80)


def test_low_cost_rejects_builtin_fade_in_with_general_fade_in():
    with pytest.raises(AssertionError):
        LowCostPLCSettings(
            fade_in=[LinearCrossfadeSettings(length=10)], fade_in_length=10
        )


def test_low_cost_accepts_the_declared_float_fade_out_length():
    settings = LowCostPLCSettings()
    assert settings.get("fade_out_length") == 0.5


def test_advanced_plc_rejects_unknown_channel_keys():
    with pytest.raises(AssertionError):
        AdvancedPLCSettings(band_settings={"bogus": [LastPacketPLCSettings()]})


def test_advanced_plc_requires_one_frequency_less_than_the_bands():
    with pytest.raises(AssertionError):
        AdvancedPLCSettings(
            band_settings={"linked": [LastPacketPLCSettings()]},
            frequencies={"linked": [1000]},
        )


# --------------------------------------------------------------------------- #
# Generic Settings behaviour
# --------------------------------------------------------------------------- #


def test_get_missing_key_raises():
    with pytest.raises(KeyError):
        Settings().get("missing")


def test_add_duplicate_key_raises():
    settings = Settings()
    settings.add("value", 1)
    with pytest.raises(KeyError):
        settings.add("value", 2)


def test_dictionary_round_trip_is_stable():
    original = attach(ZerosPLCSettings(crossfade_frequencies=[1000]))
    as_dict = original.to_dict()

    restored = Settings(as_dict)
    assert restored.to_dict() == as_dict


def test_hash_is_invariant_to_key_insertion_order():
    first = Settings({"alpha": 1, "beta": 2})
    second = Settings({"beta": 2, "alpha": 1})
    assert hash(first) == hash(second)


def test_clone_is_independent_of_the_original():
    original = ZerosPLCSettings(crossfade_frequencies=[1000])
    clone = original.clone()
    clone.settings["crossover_order"] = 99

    assert original.get("crossover_order") != 99
    assert clone.get("crossover_order") == 99


def test_inherit_from_copies_parent_values_and_binds_the_parent_hash():
    parent = attach(BinomialPLSSettings(seed=5, packet_size=8, per=0.5))
    child = attach(ZerosPLCSettings())
    child.inherit_from(parent)

    assert child.get("per") == 0.5
    assert child.get("seed") == 5
    assert child.get("packet_size") == 8
    assert getattr(child, "parent") == str(hash(parent))


def test_inherit_from_is_deterministic_across_siblings():
    parent = attach(BinomialPLSSettings(seed=5, packet_size=8, per=0.5))
    first = attach(ZerosPLCSettings())
    second = attach(ZerosPLCSettings())
    first.inherit_from(parent)
    second.inherit_from(parent)

    assert hash(first) == hash(second)


def test_progress_monitor_is_stored_and_retrieved():
    settings = Settings()
    sentinel = object()
    settings.set_progress_monitor(sentinel)
    assert settings.get_progress_monitor() is sentinel