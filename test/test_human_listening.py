"""Isolated MUSHRA preparation/import tests: no platform or web server."""
from dataclasses import replace
from typing import cast

from anytree import Node
import numpy as np
import pytest
import soundfile as sf
from ruamel.yaml import YAML

from plctestbench.data_manager import DataManager
from plctestbench.file_wrapper import AudioFile, DataFile
from plctestbench.human_listening import (
    HumanCalculatorData, assemble_study, loss_events, prepare_condition,
)
from plctestbench.models import RunStatus
from plctestbench.output_analyser import HumanCalculator
from plctestbench.settings import HumanCalculatorSettings
from test._helpers import attach


def material(tmp_path, *, channels=1, sample_rate=8000):
    time = np.arange(sample_rate * 12) / sample_rate
    audio = (0.2 * np.sin(2 * np.pi * 440 * time)).astype(np.float32)
    if channels == 2:
        audio = np.stack((audio, audio * 0.5), axis=1)
    original = AudioFile(audio, str(tmp_path / "original.wav"), samplerate=sample_rate)
    restored = AudioFile(audio * 0.95, str(tmp_path / "plc-a.wav"), samplerate=sample_rate)
    indices = np.concatenate((np.arange(sample_rate * 3, sample_rate * 3 + 80), np.arange(sample_rate * 8, sample_rate * 8 + 80)))
    return original, restored, indices


def settings(**kwargs):
    # Existing fixtures explicitly exercise loss-centered excerpt mode.
    options = {"use_full_track": False, "single_loss_per_stimulus": True, **kwargs}
    return attach(HumanCalculatorSettings(**options), fs=8000, packet_size=80)


def bundle(tmp_path, **kwargs):
    original, restored, indices = material(tmp_path)
    return prepare_condition(original, restored, indices, settings(**kwargs))


def test_preparation_is_nonblocking_persistent_and_does_not_use_global_rng(tmp_path):
    original, restored, indices = material(tmp_path)
    original_before = original.get_data().copy()
    restored_before = restored.get_data().copy()
    rng_before = cast(tuple, np.random.get_state())
    calculator = HumanCalculator(settings())
    result = calculator.run(original, restored, DataFile(indices, str(tmp_path / "loss.pickle")))
    rng_after = cast(tuple, np.random.get_state())
    assert isinstance(result, HumanCalculatorData)
    assert result.status == "prepared"
    assert calculator.is_persistent()
    assert len(result.excerpts) == 2
    assert "using 2" in result.warnings[0]
    assert result.to_dict()["excerpts"][0]["mean"] is None
    assert rng_before[0] == rng_after[0]
    np.testing.assert_array_equal(rng_before[1], rng_after[1])
    assert rng_before[2:] == rng_after[2:]
    np.testing.assert_array_equal(original.get_data(), original_before)
    np.testing.assert_array_equal(restored.get_data(), restored_before)


@pytest.mark.parametrize("channels", [1, 2])
def test_equal_material_shapes_resampling_and_pickle_hash(tmp_path, channels):
    original, restored, indices = material(tmp_path, channels=channels)
    result = prepare_condition(original, restored, indices, settings())
    for excerpt in result.excerpts:
        assert excerpt.reference.shape == excerpt.reconstructed.shape == excerpt.anchor.shape
        assert len(excerpt.reference) == 144000
        assert excerpt.definition.loss_end_sample - excerpt.definition.loss_start_sample == 80
        assert excerpt.reference[0].sum() == 0
    data = DataFile(result, str(tmp_path / "result.pickle"))
    reloaded = DataFile(path=str(tmp_path / "result.pickle"))
    assert hash(reloaded.get_data()) == hash(result)
    assert hash(reloaded) == hash(data)
    assert result.to_dict() == cast(HumanCalculatorData, reloaded.get_data()).to_dict()


def test_full_track_is_default_and_single_loss_is_not():
    defaults = HumanCalculatorSettings()
    assert defaults.get("use_full_track") is True
    assert defaults.get("single_loss_per_stimulus") is False


def test_excerpt_mode_allows_multiple_losses_by_default(tmp_path):
    original, restored, _ = material(tmp_path)
    config = attach(HumanCalculatorSettings(use_full_track=False), fs=8000, packet_size=80)
    result = prepare_condition(original, restored, [24000, 24100], config)
    assert result.excerpts
    assert all(excerpt.definition.end_sample - excerpt.definition.start_sample == 24000 for excerpt in result.excerpts)


@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("indices", [[], [0, 1, 100, 101, 95999]])
def test_full_track_bypasses_excerpt_selection_and_preserves_audio(tmp_path, channels, indices):
    from plctestbench.human_listening import _fade_boundaries, _listening_audio

    original, restored, _ = material(tmp_path, channels=channels)
    before = original.get_data().copy(), restored.get_data().copy()
    # Even a 12-second excerpt, strict single-loss constraint and 0 dB
    # eligibility threshold must not affect whole-track preparation.
    config = attach(HumanCalculatorSettings(
        stimulus_length=12000, single_loss_per_stimulus=True,
        loudness_threshold_db=0, trials_per_track=99, iterations=2,
    ), fs=8000, packet_size=80)
    result = prepare_condition(original, restored, indices, config)
    assert len(result.excerpts) == 1
    assert result.warnings == ()
    excerpt = result.excerpts[0]
    assert (excerpt.definition.start_sample, excerpt.definition.end_sample) == (0, 96000)
    assert (excerpt.definition.loss_start_sample, excerpt.definition.loss_end_sample) == (
        (0, 96000) if indices else (0, 0)
    )
    assert len(excerpt.reference) == 576000
    assert excerpt.reference.shape == excerpt.reconstructed.shape == excerpt.anchor.shape
    np.testing.assert_array_equal(excerpt.reference, _fade_boundaries(_listening_audio(before[0], 8000)))
    np.testing.assert_array_equal(excerpt.reconstructed, _fade_boundaries(_listening_audio(before[1], 8000)))
    np.testing.assert_array_equal(original.get_data(), before[0])
    np.testing.assert_array_equal(restored.get_data(), before[1])
    repeated = prepare_condition(original, restored, indices, config)
    assert hash(result) == hash(repeated)
    study = assemble_study([result, replace(result, condition_id="condition-b")], "full-track")
    assert len(study.definition.trials) == 2
    assert {trial.repetition for trial in study.definition.trials} == {0, 1}


def test_full_track_accepts_short_loss_free_audio_and_exports_all_samples(tmp_path):
    original, restored, _ = material(tmp_path)
    original.data = original.get_data()[:4000]
    restored.data = restored.get_data()[:4000]
    config = attach(HumanCalculatorSettings(), fs=8000, packet_size=80)
    result = prepare_condition(original, restored, [], config)
    study = assemble_study([result], "short-track")
    path = study.export(tmp_path / "full-track-export")
    exported = YAML(typ="safe").load(path)
    trial = exported["pages"][-1]
    for relative in (trial["reference"], *trial["stimuli"].values()):
        audio, sample_rate = sf.read(path.parent / relative)
        assert sample_rate == 48000
        assert len(audio) == 24000


def test_old_persisted_settings_keep_excerpt_mode(tmp_path):
    original, restored, indices = material(tmp_path)
    config = settings()
    del config.get_all()["use_full_track"]
    assert len(prepare_condition(original, restored, indices, config).excerpts) == 2


def test_loss_events_normalize_legacy_floats_and_keep_next_burst_start():
    assert loss_events([11.0, 10.0, 11.0, 20.0, 21.0], 100) == [(10, 12), (20, 22)]
    assert loss_events([], 100) == []


@pytest.mark.parametrize("indices", [[1.5], [-1], [100], [float("nan")], [[1]], [True], ["1"]])
def test_invalid_loss_indices_are_rejected(indices):
    with pytest.raises(ValueError):
        loss_events(indices, 100)


def test_empty_boundary_and_neighboring_events_fail_clearly(tmp_path):
    original, restored, _ = material(tmp_path)
    for indices in ([], [0], [95999], [24000, 24100]):
        with pytest.raises(ValueError, match="no eligible"):
            prepare_condition(original, restored, indices, settings())


def test_multiple_losses_allowed_when_explicit(tmp_path):
    original, restored, _ = material(tmp_path)
    result = prepare_condition(original, restored, [24000, 24100], settings(single_loss_per_stimulus=False))
    assert result.excerpts


@pytest.mark.parametrize("kwargs", [
    {"stimulus_length": 999}, {"stimulus_length": 12001},
    {"trials_per_track": 0}, {"iterations": 0}, {"choose_seed": -1},
    {"choose_seed": True}, {"single_loss_per_stimulus": "true"},
    {"use_full_track": "true"}, {"use_full_track": 1},
    {"test_type": "splat"}, {"reference": "external.wav"},
    {"anchor": "anchor.wav"}, {"loudness_threshold_db": float("nan")},
    {"loudness_threshold_db": 1}, {"pages": 0},
    {"pages": 2, "stimuli_per_page": 10, "trials_per_track": 1},
])
def test_settings_validation(kwargs):
    with pytest.raises(ValueError):
        HumanCalculatorSettings(**kwargs)


def test_legacy_settings_migrate_without_reinterpreting_splat():
    assert HumanCalculatorSettings(pages=3).get("trials_per_track") == 30
    assert HumanCalculatorSettings(stimuli_per_page=4).get("trials_per_track") == 8
    assert HumanCalculatorSettings(reference=None, anchor=None).get("test_type") == "mushra"


def test_assembly_export_and_finalize_preserve_zero_and_repetitions(tmp_path):
    first = bundle(tmp_path, iterations=2)
    second = replace(first, condition_id="condition-b")
    study = assemble_study([first, second], "study-1")
    assert len(study.definition.trials) == 4
    assert {condition.role for condition in study.definition.trials[0].conditions} == {"plc", "anchor", "hidden_reference"}
    config_path = study.export(tmp_path / "export")
    config = YAML(typ="safe").load(config_path)
    assert config["testId"] == "study-1"
    volume, *trials = config["pages"]
    assert volume["type"] == "volume"
    assert volume["defaultVolume"] == 0.8
    assert volume["stimulus"] == trials[0]["reference"]
    assert all(page["type"] == "mushra" for page in trials)
    assert "remoteService" not in config  # adapter must supply collection
    assert not trials[0]["createAnchor35"]
    for page in trials:
        for path in (page["reference"], *page["stimuli"].values()):
            audio, rate = sf.read(config_path.parent / path)
            assert rate == 48000 and len(audio) == 144000
    trial = study.definition.trials[0]
    rows = [{"session_id": "participant-1", "trial_id": trial.id, "condition_id": first.condition_id, "score": 0}]
    finalized = study.finalize(rows)
    assert all(result.status == "finalized" for result in finalized)
    assert finalized[0].responses[0].score == 0
    means = [excerpt["mean"] for excerpt in finalized[0].to_dict()["excerpts"]]
    assert 0 in means and None in means
    assert finalized[1].to_dict()["excerpts"][0]["mean"] is None
    with pytest.raises(FileExistsError):
        study.export(config_path.parent)


@pytest.mark.parametrize("change", [
    {"score": -1}, {"score": 101}, {"score": float("nan")}, {"score": True},
    {"score": "50"}, {"answered": False}, {"answered": 1},
    {"condition_id": "unknown"}, {"trial_id": "unknown"}, {"session_id": ""},
])
def test_response_validation(tmp_path, change):
    first = bundle(tmp_path)
    study = assemble_study([first], "study-1")
    row = {"session_id": "p1", "trial_id": study.definition.trials[0].id, "condition_id": first.condition_id, "score": 50}
    with pytest.raises(ValueError):
        study.import_responses([{**row, **change}])
    with pytest.raises(ValueError, match="Duplicate"):
        study.import_responses([row, row])


def test_assembly_rejects_mismatched_excerpts_and_duplicate_conditions(tmp_path):
    first = bundle(tmp_path)
    with pytest.raises(ValueError, match="Duplicate"):
        assemble_study([first, first], "study")
    changed = replace(first, condition_id="b", excerpts=tuple(reversed(first.excerpts)))
    with pytest.raises(ValueError, match="identical"):
        assemble_study([first, changed], "study")
    too_many = [replace(first, condition_id=f"c{index}") for index in range(11)]
    with pytest.raises(ValueError, match="1–10"):
        assemble_study(too_many, "study")


def test_anchor_has_no_filter_delay_and_attenuates_high_frequencies():
    from plctestbench.human_listening import _anchor
    impulse = np.zeros(48000, dtype=np.float32)
    impulse[24000] = 1
    assert np.argmax(_anchor(impulse)) == 24000
    time = np.arange(48000) / 48000
    low = _anchor(np.sin(2 * np.pi * 1000 * time).astype(np.float32))[1000:-1000]
    high = _anchor(np.sin(2 * np.pi * 10000 * time).astype(np.float32))[1000:-1000]
    assert np.sqrt(np.mean(high ** 2)) < np.sqrt(np.mean(low ** 2)) * 0.01


def test_run_status_does_not_transiently_fail_after_success(monkeypatch):
    manager = object.__new__(DataManager)
    root = Node("root")
    monkeypatch.setattr(root, "run", lambda: None, raising=False)
    manager.root_nodes = [root]
    manager.progress_monitor = lambda caller: lambda iterable, **kwargs: iterable
    statuses = []
    monkeypatch.setattr(manager, "_set_run_status", statuses.append)
    manager.run_testbench()
    assert statuses == [RunStatus.RUNNNING, RunStatus.COMPLETED]
    def fail():
        raise RuntimeError("failed computation")
    monkeypatch.setattr(root, "run", fail)
    statuses.clear()
    with pytest.raises(RuntimeError):
        manager.run_testbench()
    assert statuses == [RunStatus.RUNNNING, RunStatus.FAILED]


def test_pending_human_nodes_do_not_stop_other_automated_nodes(tmp_path, monkeypatch):
    from plctestbench.node import OutputAnalysisNode
    from plctestbench.settings import Settings

    prepared = bundle(tmp_path)
    pending = OutputAnalysisNode(file=DataFile(prepared, str(tmp_path / "human.pickle")), settings=Settings())
    automated = Node("automated")
    calls = []
    monkeypatch.setattr(pending, "run", lambda: calls.append("human"))
    monkeypatch.setattr(automated, "run", lambda: calls.append("automated"), raising=False)
    manager = object.__new__(DataManager)
    manager.root_nodes = [pending, automated]
    manager.progress_monitor = lambda caller: lambda iterable, **kwargs: iterable
    statuses = []
    monkeypatch.setattr(manager, "_set_run_status", statuses.append)
    manager.run_testbench()
    assert calls == ["human", "automated"]
    assert statuses == [RunStatus.RUNNNING, RunStatus.AWAITING_INPUT]
    pending.file = DataFile(replace(prepared, status="finalized"), str(tmp_path / "human.pickle"))
    statuses.clear()
    manager.run_testbench()
    assert statuses == [RunStatus.RUNNNING, RunStatus.COMPLETED]


def test_many_tracks_share_condition_identity_but_not_excerpt_identity(tmp_path):
    track_a = tmp_path / "track-a"
    track_b = tmp_path / "track-b"
    track_a.mkdir()
    track_b.mkdir()
    first = bundle(track_a)
    second = bundle(track_b)
    assert first.condition_id == second.condition_id
    assert first.excerpts[0].definition.id != second.excerpts[0].definition.id
    study = assemble_study([first, second], "multi-track")
    assert len(study.definition.trials) == 4


def test_preparation_rejects_silent_nonfinite_and_mismatched_audio(tmp_path):
    original, restored, indices = material(tmp_path)
    source = original.get_data().copy()
    original.data = np.zeros_like(source)
    with pytest.raises(ValueError, match="no eligible"):
        prepare_condition(original, restored, indices, settings())
    original.data = source.copy()
    original.data[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        prepare_condition(original, restored, indices, settings())
    original.data = source
    restored.data = restored.get_data()[:-1]
    with pytest.raises(ValueError, match="equal length"):
        prepare_condition(original, restored, indices, settings())


def test_shared_excerpt_helper_keeps_every_burst_and_skips_boundary_slices():
    from plctestbench.utils import extract_intorni
    from test._helpers import AudioStub

    audio = np.arange(200, dtype=np.float32)
    packets, excerpts = extract_intorni(AudioStub(audio), [1, 2, 50, 51, 100, 101, 199], 20, 1000, 10)
    assert packets == [5, 10]
    np.testing.assert_array_equal(excerpts[0], audio[40:60])
    np.testing.assert_array_equal(excerpts[1], audio[90:110])


def test_short_fades_and_mono_padding_are_safe():
    from plctestbench.utils import fade_in, fade_out, leading_silence, trailing_silence, is_loud_enough

    audio = np.ones(5, dtype=np.float32)
    fade_in(audio, 1000, 100)
    fade_out(audio, 1000, 100)
    assert np.all(np.isfinite(audio))
    before = audio.copy()
    fade_out(audio, 1000, 0)
    np.testing.assert_array_equal(audio, before)
    assert leading_silence(audio, 1000, 10).shape == (15,)
    assert trailing_silence(audio, 1000, 10).shape == (15,)
    assert not is_loud_enough(np.zeros(5), np.zeros(5))
    assert not is_loud_enough(np.array([np.nan]), np.ones(5))


def test_human_calculator_docstrings_describe_the_public_workflow():
    assert "AWAITING_INPUT" in (HumanCalculator.__doc__ or "")
    assert "trials_per_track" in (HumanCalculator.__init__.__doc__ or "")
    assert "loudness_threshold_db" in (HumanCalculator.__init__.__doc__ or "")
    assert "prepared" in (HumanCalculator.run.__doc__ or "")


def test_standalone_finalization_updates_cache_without_repreparing(tmp_path, monkeypatch):
    from tinydb import TinyDB
    from tinydb.storages import MemoryStorage
    from plctestbench.database_manager import TinyDBDatabaseManager
    from plctestbench.node import OutputAnalysisNode

    prepared = bundle(tmp_path)
    memory = TinyDB(storage=MemoryStorage)
    database = object.__new__(TinyDBDatabaseManager)
    monkeypatch.setattr(database, "get_database", lambda: memory)
    node = OutputAnalysisNode(
        file=DataFile(prepared, str(tmp_path / "prepared.pickle")),
        worker=HumanCalculator, settings=settings(), database=database,
    )
    node._save_to_database()
    study = assemble_study([prepared], "standalone-study")
    finalized = study.finalize([])[0]
    destination = str(tmp_path / "finalized.pickle")
    node.finalize_human_result(finalized, destination)
    assert (tmp_path / "prepared.pickle").exists()
    assert len(memory.table("OutputAnalysisNode")) == 1
    monkeypatch.setattr(node, "_run", lambda: pytest.fail("Finalization must not invalidate the prepared cache"))
    node.run()
    assert cast(HumanCalculatorData, node.get_data()).status == "finalized"
    assert node.file.get_path() == destination
    memory.close()
