"""Shared, dependency-light helpers for the plctestbench unit test suite.

Nothing in here performs I/O or network access: every worker is driven with an
in-memory, seeded mock input and a silent progress monitor so that the tests
stay fast and deterministic.
"""

from __future__ import annotations

import numpy as np

from plctestbench.file_wrapper import AudioFile, DataFile


#: Default sampling rate used by the mock inputs. It is a real-world value so
#: that millisecond-based settings (crossfade/fade-in lengths, context length)
#: translate to sensible sample counts.
FS = 16000


class _NullBar:
    """Minimal stand-in for a tqdm bar.

    It is callable (like the tqdm class) so it can act as the per-worker bar
    factory: calling it with an iterable yields that iterable's items, while
    calling it with only ``total=`` yields an empty, non-consuming bar. Both
    forms expose the control methods the package relies on.
    """

    def __init__(self, iterable=None):
        self._iterable = [] if iterable is None else iterable

    def __call__(self, iterable=None, total=None, desc=None, **kwargs) -> _NullBar:
        return _NullBar(iterable)

    def __iter__(self):
        yield from self._iterable

    def update(self, n=1):
        return None

    def set_description(self, description=None, refresh=False):
        return None

    def close(self):
        return None


class SilentProgressMonitor:
    """Drop-in replacement for ``plctestbench.utils.progress_monitor``.

    ``Worker`` calls ``settings.get_progress_monitor()(self)`` and then uses
    the result as a tqdm-like factory. Mirroring that protocol keeps the tests
    output-free and free of timing overhead.
    """

    def __call__(self, caller=None) -> _NullBar:
        return _NullBar()


#: Shared instance; assign it to any settings object under test.
silent_progress_monitor = SilentProgressMonitor()


def attach(settings, **extra):
    """Attach the silent progress monitor and any missing setting values.

    ``settings.add`` raises on duplicate keys, so values that the settings
    class already defines (e.g. ``BurgPLCSettings.context_length``) are left
    untouched.
    """
    settings.set_progress_monitor(silent_progress_monitor)
    for key, value in extra.items():
        if key not in settings.get_all():
            settings.add(key, value)
    return settings


def seeded_track(
    length: int,
    channels: int = 1,
    seed: int = 0,
    amplitude: float = 0.2,
    dtype=np.float32,
) -> np.ndarray:
    """Deterministic white-noise audio buffer shaped ``(length, channels)``."""
    rng = np.random.RandomState(seed)
    return (rng.randn(length, channels) * amplitude).astype(dtype)


def seeded_sine(
    length: int,
    frequency: float = 440.0,
    fs: int = FS,
    channels: int = 1,
    amplitude: float = 0.5,
    seed: int = 0,
) -> np.ndarray:
    """Deterministic sine + tiny seeded noise, shaped ``(length, channels)``."""
    rng = np.random.RandomState(seed)
    time = np.arange(length) / float(fs)
    mono = amplitude * np.sin(2 * np.pi * frequency * time)
    mono = mono + 1e-4 * rng.standard_normal(length)
    return np.repeat(mono[:, np.newaxis], channels, axis=1).astype(np.float32)


class AudioStub(AudioFile):
    """In-memory ``AudioFile`` test double.

The output analysers only consume ``get_data`` (and ``get_samplerate`` /
    ``get_channels``), so overriding ``__init__`` avoids the file I/O that the
    real ``AudioFile`` performs, while staying a genuine ``AudioFile`` subtype
    for type checkers and ``isinstance`` checks.
    """

    def __init__(self, data, samplerate: int = FS):  # noqa: D107 - see class docstring
        self.data = np.asarray(data)
        self.samplerate = samplerate
        self.channels = 1 if self.data.ndim == 1 else self.data.shape[1]

    def get_data(self) -> np.ndarray:
        return self.data

    def get_samplerate(self) -> int:
        return self.samplerate

    def get_channels(self) -> int:
        return self.channels

    def get_path(self) -> str:  # pragma: no cover - never used by unit tests
        raise AssertionError("AudioStub has no path: this must not be reached")


class DataStub(DataFile):
    """In-memory ``DataFile`` test double (used for loss-index payloads)."""

    def __init__(self, data):  # noqa: D107 - see class docstring
        self.data = np.asarray(data)

    def get_data(self) -> np.ndarray:
        return self.data


__all__ = [
    "FS",
    "AudioStub",
    "DataStub",
    "SilentProgressMonitor",
    "attach",
    "seeded_sine",
    "seeded_track",
    "silent_progress_monitor",
]