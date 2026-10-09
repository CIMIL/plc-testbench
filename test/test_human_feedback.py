import numpy as np
import pytest
from plctestbench.human_listening import NoEligibleExcerptsError, select_excerpts


def select(indices, *, samples=10000):
    return select_excerpts(np.ones(samples), indices, track_id='track', mask_id='mask', sample_rate=1000,
                          packet_size=1, stimulus_length=3000, trials_per_track=20,
                          single_loss=True, seed=1, threshold_db=-10)


def test_dense_bursts_explain_single_loss_constraint_without_changing_settings():
    with pytest.raises(NoEligibleExcerptsError) as raised:
        select(np.arange(1000, 9000, 500))
    assert 'multiple loss bursts' in raised.value.user_message
    assert 'single_loss_per_stimulus' in raised.value.guidance


def test_no_events_and_short_track_have_distinct_guidance():
    with pytest.raises(NoEligibleExcerptsError) as raised:
        select([])
    assert 'no packet-loss events' in raised.value.user_message
    with pytest.raises(NoEligibleExcerptsError) as raised:
        select([500], samples=1000)
    assert 'No complete 3000 ms' in raised.value.user_message
    assert 'longer track' in raised.value.guidance
