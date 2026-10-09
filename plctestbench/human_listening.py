"""Portable human-listening artifacts, independent of any web server or platform.

Preparation is not a numerical result. A caller assembles condition bundles,
exports a study, collects responses, and explicitly finalizes it later.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from math import gcd
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import soundfile as sf
from ruamel.yaml import YAML
from scipy.signal import convolve, firwin, resample_poly

from .file_wrapper import OutputAnalysis

SCHEMA_VERSION = 1
LISTENING_SAMPLE_RATE = 48000


class NoEligibleExcerptsError(ValueError):
    """Preparation failure with safe, actionable listener-study guidance.

    Attributes:
        user_message: Plain-text explanation without private filesystem paths.
        guidance: Suggested changes; settings are never changed automatically.
    """

    def __init__(self, track_id: str, user_message: str, guidance: str):
        """Retain the legacy ValueError contract and attach user-facing guidance."""
        super().__init__(f"Track {track_id}: no eligible complete loss-centered excerpts")
        self.user_message = user_message
        self.guidance = guidance


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class ExcerptDefinition:
    id: str
    track_id: str
    mask_id: str
    start_sample: int
    end_sample: int  # exclusive; source sample rate
    loss_start_sample: int
    loss_end_sample: int  # exclusive
    packet_size: int
    source_sample_rate: int


@dataclass(frozen=True)
class ConditionDefinition:
    id: str
    role: Literal["plc", "hidden_reference", "anchor"]


@dataclass(frozen=True)
class TrialDefinition:
    id: str
    excerpt: ExcerptDefinition
    repetition: int
    conditions: tuple[ConditionDefinition, ...]


@dataclass(frozen=True)
class ListeningTestDefinition:
    id: str
    trials: tuple[TrialDefinition, ...]
    schema_version: int = SCHEMA_VERSION
    test_type: str = "mushra"
    scale_min: int = 0
    scale_max: int = 100
    higher_is_better: bool = True


@dataclass(frozen=True)
class HumanResponse:
    session_id: str
    trial_id: str
    condition_id: str
    score: float
    answered: bool = True


@dataclass(frozen=True)
class PreparedExcerpt:
    definition: ExcerptDefinition
    reference: np.ndarray
    reconstructed: np.ndarray
    anchor: np.ndarray


@dataclass(frozen=True)
class HumanCalculatorData(OutputAnalysis):
    """Persistable single-condition bundle with explicit preparation state."""

    condition_id: str
    excerpts: tuple[PreparedExcerpt, ...]
    iterations: int = 1
    warnings: tuple[str, ...] = ()
    status: Literal["prepared", "finalized"] = "prepared"
    study_id: str | None = None
    responses: tuple[HumanResponse, ...] = ()
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict:
        """JSON-safe metadata/results; WAV data is exported separately."""
        summaries = []
        for excerpt in self.excerpts:
            scores = [
                response.score for response in self.responses
                if response.condition_id == self.condition_id
                and response.trial_id in {
                    trial_id(excerpt.definition.id, repetition)
                    for repetition in range(self.iterations)
                }
            ]
            summaries.append({
                "excerpt": asdict(excerpt.definition),
                "response_count": len(scores),
                "mean": float(np.mean(scores)) if scores else None,
                "std": float(np.std(scores, ddof=1)) if len(scores) > 1 else None,
            })
        return {
            "kind": "human-listening", "schema_version": self.schema_version,
            "test_type": "mushra", "scale_min": 0, "scale_max": 100,
            "higher_is_better": True, "status": self.status,
            "condition_id": self.condition_id, "study_id": self.study_id,
            "iterations": self.iterations, "warnings": list(self.warnings),
            "excerpts": summaries,
            "responses": [asdict(response) for response in self.responses],
        }

    def __hash__(self) -> int:
        digest = hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode())
        for excerpt in self.excerpts:
            for audio in (excerpt.reference, excerpt.reconstructed, excerpt.anchor):
                digest.update(str(audio.shape).encode())
                digest.update(np.asarray(audio, dtype="<f4").tobytes())
        # Keep the result within a signed machine integer for FileWrapper hashes.
        return int.from_bytes(digest.digest()[:8], "big") >> 1


def loss_events(indices: object, sample_count: int) -> list[tuple[int, int]]:
    """Normalize persisted integer/float indices into complete contiguous bursts."""
    values = np.asarray(indices)
    if values.ndim != 1 or values.dtype.kind not in "iuf":
        raise ValueError("Lost-sample indices must be a one-dimensional integer array")
    if not np.all(np.isfinite(values)) or np.any(values != np.floor(values)):
        raise ValueError("Lost-sample indices must be finite integers")
    if np.any(values < 0) or np.any(values >= sample_count):
        raise ValueError("Lost-sample indices are outside the audio bounds")
    values = np.unique(values.astype(np.int64))
    if values.size == 0:
        return []
    groups = np.split(values, np.flatnonzero(np.diff(values) > 1) + 1)
    return [(int(group[0]), int(group[-1]) + 1) for group in groups]


def select_excerpts(
    original: np.ndarray, indices: object, *, track_id: str, mask_id: str,
    sample_rate: int, packet_size: int, stimulus_length: int,
    trials_per_track: int, single_loss: bool, seed: int, threshold_db: float,
) -> tuple[tuple[ExcerptDefinition, ...], tuple[str, ...]]:
    """Select once from the reference/mask, never from a PLC condition's output."""
    events = loss_events(indices, len(original))
    length = round(sample_rate * stimulus_length / 1000)
    reference_rms = float(np.sqrt(np.mean(np.square(original, dtype=np.float64))))
    candidates = []
    seen = set()
    in_bounds = multiple_losses = quiet = 0
    for loss_start, loss_end in events:
        center = (loss_start + loss_end - 1) // 2
        start = center - length // 2
        end = start + length
        if start < 0 or end > len(original) or loss_start < start or loss_end > end:
            continue
        in_bounds += 1
        if single_loss and sum(a < end and b > start for a, b in events) != 1:
            multiple_losses += 1
            continue
        if (start, end) in seen:
            continue
        clip_rms = float(np.sqrt(np.mean(np.square(original[start:end], dtype=np.float64))))
        if reference_rms <= 0 or clip_rms <= reference_rms * 10 ** (threshold_db / 20):
            quiet += 1
            continue
        seen.add((start, end))
        excerpt_id = "excerpt-" + _digest([track_id, mask_id, start, end])[:24]
        candidates.append(ExcerptDefinition(
            excerpt_id, track_id, mask_id, start, end, loss_start, loss_end,
            packet_size, sample_rate,
        ))
    if not candidates:
        if not events:
            message, guidance = "A selected track has no packet-loss events to rate.", "Choose a loss simulator configuration that produces packet loss."
        elif not in_bounds:
            message, guidance = f"No complete {stimulus_length} ms loss-centered excerpt fits in a selected track.", "Use a longer track, a shorter excerpt, or move loss events away from track boundaries."
        elif multiple_losses == in_bounds:
            message, guidance = f"Every complete {stimulus_length} ms excerpt contains multiple loss bursts, but exactly one burst per excerpt is required.", "Disable HumanCalculator single_loss_per_stimulus, shorten stimulus_length, or space simulator loss bursts farther apart."
        elif quiet == in_bounds:
            message, guidance = "All complete loss-centered excerpts are too quiet relative to the track.", "Choose non-silent material or lower loudness_threshold_db."
        else:
            message, guidance = "No excerpts satisfy the loss-selection and relative-loudness requirements.", "Review single_loss_per_stimulus and loss spacing, choose non-silent material, or lower loudness_threshold_db."
        raise NoEligibleExcerptsError(track_id, message, guidance)
    count = min(trials_per_track, len(candidates))
    selected = np.random.default_rng(seed).choice(len(candidates), count, replace=False)
    warnings = () if count == trials_per_track else (
        f"Track {track_id}: requested {trials_per_track} excerpts; using {count} eligible excerpts",
    )
    return tuple(candidates[int(index)] for index in selected), warnings


def _listening_audio(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    divisor = gcd(sample_rate, LISTENING_SAMPLE_RATE)
    clip = np.array(audio, dtype=np.float32, copy=True)
    if sample_rate != LISTENING_SAMPLE_RATE:
        clip = resample_poly(clip, LISTENING_SAMPLE_RATE // divisor, sample_rate // divisor, axis=0)
    return np.ascontiguousarray(clip, dtype=np.float32)


def _fade_boundaries(audio: np.ndarray) -> np.ndarray:
    clip = np.array(audio, dtype=np.float32, copy=True)
    fade_length = min(480, len(clip) // 2)
    if fade_length:
        window = np.linspace(0, 1, fade_length, dtype=np.float32)
        if clip.ndim == 2:
            window = window[:, None]
        clip[:fade_length] *= window
        clip[-fade_length:] *= window[::-1]
    return np.ascontiguousarray(clip, dtype=np.float32)


def _anchor(reference: np.ndarray) -> np.ndarray:
    taps = firwin(513, 3500, fs=LISTENING_SAMPLE_RATE, window="hann")
    padding = ((256, 256),) if reference.ndim == 1 else ((256, 256), (0, 0))
    padded = np.pad(reference, padding, mode="reflect")
    kernel = taps if reference.ndim == 1 else taps[:, None]
    return np.ascontiguousarray(convolve(padded, kernel, mode="valid"), dtype=np.float32)


def prepare_condition(original_file, reconstructed_file, indices, settings, condition_id: str = "") -> HumanCalculatorData:
    original = np.asarray(original_file.get_data())
    reconstructed = np.asarray(reconstructed_file.get_data())
    sample_rate = int(original_file.get_samplerate())
    if original.shape != reconstructed.shape or sample_rate != reconstructed_file.get_samplerate():
        raise ValueError("Reference and PLC audio must have equal length, sample rate, and channel layout")
    if original.ndim not in (1, 2) or (original.ndim == 2 and original.shape[1] not in (1, 2)):
        raise ValueError("Human listening tests support mono or stereo audio only")
    if not len(original) or not np.all(np.isfinite(original)) or not np.all(np.isfinite(reconstructed)):
        raise ValueError("Human listening tests require nonempty, finite audio")
    if sample_rate <= 0 or settings.get("fs") != sample_rate:
        raise ValueError("The configured sample rate must match the source audio")
    packet_size = settings.get("packet_size")
    if isinstance(packet_size, bool) or not isinstance(packet_size, int) or packet_size < 1:
        raise ValueError("packet_size must be a positive integer")
    track_id = "track-" + _digest(str(original_file.get_path()))[:24]
    events = loss_events(indices, len(original))
    mask_id = "mask-" + _digest([track_id, packet_size, events])[:24]
    condition_id = condition_id or "condition-" + _digest(Path(reconstructed_file.get_path()).stem)[:24]
    if condition_id in {"reference", "anchor35"}:
        raise ValueError("PLC condition ID is reserved for a reference or anchor")
    excerpts, warnings = select_excerpts(
        original, indices, track_id=track_id, mask_id=mask_id,
        sample_rate=sample_rate, packet_size=packet_size,
        stimulus_length=settings.get("stimulus_length"),
        trials_per_track=settings.get("trials_per_track"),
        single_loss=settings.get("single_loss_per_stimulus"),
        seed=settings.get("choose_seed"), threshold_db=settings.get("loudness_threshold_db"),
    )
    material = []
    for excerpt in excerpts:
        reference = _listening_audio(original[excerpt.start_sample:excerpt.end_sample], sample_rate)
        restored = _listening_audio(reconstructed[excerpt.start_sample:excerpt.end_sample], sample_rate)
        material.append(PreparedExcerpt(
            excerpt, _fade_boundaries(reference), _fade_boundaries(restored),
            _fade_boundaries(_anchor(reference)),
        ))
    return HumanCalculatorData(condition_id, tuple(material), settings.get("iterations"), warnings)


def trial_id(excerpt_id: str, repetition: int) -> str:
    return f"{excerpt_id}-r{repetition}"


@dataclass(frozen=True)
class ListeningStudy:
    definition: ListeningTestDefinition
    bundles: tuple[HumanCalculatorData, ...]

    def export(self, destination: str | Path) -> Path:
        """Write a self-contained YAML/WAV bundle, refusing accidental overwrites."""
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=False)
        audio_directory = destination / "audio"
        audio_directory.mkdir()
        material: dict[str, dict[str, PreparedExcerpt]] = {}
        for bundle in self.bundles:
            for excerpt in bundle.excerpts:
                material.setdefault(excerpt.definition.id, {})[bundle.condition_id] = excerpt
        pages = []
        for trial in self.definition.trials:
            excerpts = material[trial.excerpt.id]
            first = next(iter(excerpts.values()))
            prefix = _digest(trial.excerpt.id)[:24]
            reference_path = f"audio/{prefix}-reference.wav"
            anchor_path = f"audio/{prefix}-anchor.wav"
            for relative, audio in ((reference_path, first.reference), (anchor_path, first.anchor)):
                if not (destination / relative).exists():
                    sf.write(destination / relative, audio, LISTENING_SAMPLE_RATE, subtype="FLOAT")
            stimuli = {"anchor35": anchor_path}
            for condition_id, excerpt in excerpts.items():
                relative = f"audio/{prefix}-{_digest(condition_id)[:24]}.wav"
                if not (destination / relative).exists():
                    sf.write(destination / relative, excerpt.reconstructed, LISTENING_SAMPLE_RATE, subtype="FLOAT")
                stimuli[condition_id] = relative
            pages.append({
                "type": "mushra", "id": trial.id, "name": "Audio quality",
                "content": "Rate audio quality from 0 (bad) to 100 (excellent).",
                "reference": reference_path, "stimuli": stimuli,
                "createAnchor35": False, "createAnchor70": False,
                "showWaveform": False, "enableLooping": True,
                "randomize": True, "switchBack": False,
            })
        config = {
            "testname": "MUSHRA listening test", "testId": self.definition.id,
            "bufferSize": 2048, "stopOnErrors": True,
            "showButtonPreviousPage": True,
            # Collection must be explicitly configured by the hosting adapter.
            "pages": pages,
        }
        with (destination / "config.yaml").open("w") as file:
            YAML().dump(config, file)
        (destination / "definition.json").write_text(json.dumps(asdict(self.definition), indent=2))
        return destination / "config.yaml"

    def import_responses(self, records: Iterable[dict]) -> tuple[HumanResponse, ...]:
        """Import already completed sessions; never infer consent/completion here."""
        trials = {trial.id: trial for trial in self.definition.trials}
        responses = []
        seen = set()
        for record in records:
            if set(record) - {"session_id", "trial_id", "condition_id", "score", "answered"}:
                raise ValueError("Unknown response fields")
            try:
                response = HumanResponse(**record)
            except TypeError as error:
                raise ValueError("Invalid response fields") from error
            if not all(isinstance(value, str) and value for value in (
                response.session_id, response.trial_id, response.condition_id,
            )):
                raise ValueError("Response identifiers must be nonempty strings")
            if response.trial_id not in trials or response.condition_id not in {
                condition.id for condition in trials[response.trial_id].conditions
            }:
                raise ValueError("Response does not belong to this study")
            if not isinstance(response.answered, bool) or not response.answered or isinstance(response.score, bool) or not isinstance(response.score, (int, float)):
                raise ValueError("Imported responses must contain an explicitly answered numerical score")
            if not np.isfinite(response.score) or not 0 <= response.score <= 100:
                raise ValueError("MUSHRA scores must be finite and between 0 and 100")
            key = (response.session_id, response.trial_id, response.condition_id)
            if key in seen:
                raise ValueError("Duplicate response identity")
            seen.add(key)
            responses.append(response)
        return tuple(responses)

    def finalize(self, records: Iterable[dict]) -> tuple[HumanCalculatorData, ...]:
        responses = self.import_responses(records)
        return tuple(replace(
            bundle, status="finalized", study_id=self.definition.id,
            responses=tuple(
                response for response in responses
                if response.trial_id in {
                    trial_id(excerpt.definition.id, repetition)
                    for excerpt in bundle.excerpts for repetition in range(bundle.iterations)
                }
            ),
        ) for bundle in self.bundles)


def assemble_study(bundles: Iterable[HumanCalculatorData], study_id: str) -> ListeningStudy:
    """Combine explicitly grouped track/PLC bundles; callers group by simulator."""
    bundles = tuple(bundles)
    if not study_id or not bundles:
        raise ValueError("A study requires an explicit ID and at least one condition bundle")
    if any(bundle.schema_version != SCHEMA_VERSION or bundle.status != "prepared" for bundle in bundles):
        raise ValueError("Study assembly requires prepared version-1 bundles")
    iterations = {bundle.iterations for bundle in bundles}
    if len(iterations) != 1:
        raise ValueError("Study bundles have different repetition counts")
    tracks: dict[str, dict[str, HumanCalculatorData]] = {}
    for bundle in bundles:
        if not bundle.excerpts:
            raise ValueError("A condition bundle has no excerpts")
        track = bundle.excerpts[0].definition.track_id
        if track in tracks and bundle.condition_id in tracks[track]:
            raise ValueError("Duplicate condition bundle for track")
        if any(excerpt.definition.track_id != track for excerpt in bundle.excerpts):
            raise ValueError("A condition bundle must contain a single track")
        tracks.setdefault(track, {})[bundle.condition_id] = bundle
    expected_conditions = set(next(iter(tracks.values())))
    if not 1 <= len(expected_conditions) <= 10 or expected_conditions & {"reference", "anchor35"}:
        raise ValueError("MUSHRA requires 1–10 PLC conditions plus hidden reference and anchor")
    trials = []
    for conditions in tracks.values():
        if set(conditions) != expected_conditions:
            raise ValueError("Every track must contain the same PLC conditions")
        first = next(iter(conditions.values()))
        for bundle in conditions.values():
            if [excerpt.definition for excerpt in bundle.excerpts] != [excerpt.definition for excerpt in first.excerpts]:
                raise ValueError("PLC conditions must use identical track/mask excerpt bounds")
            for expected, actual in zip(first.excerpts, bundle.excerpts):
                if not np.array_equal(expected.reference, actual.reference) or not np.array_equal(expected.anchor, actual.anchor):
                    raise ValueError("PLC conditions must use identical reference and anchor material")
        roles = tuple(ConditionDefinition(condition, "plc") for condition in sorted(conditions)) + (
            ConditionDefinition("reference", "hidden_reference"), ConditionDefinition("anchor35", "anchor"),
        )
        for excerpt in first.excerpts:
            for repetition in range(first.iterations):
                trials.append(TrialDefinition(trial_id(excerpt.definition.id, repetition), excerpt.definition, repetition, roles))
    return ListeningStudy(ListeningTestDefinition(study_id, tuple(trials)), bundles)
