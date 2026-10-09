import hashlib
import math
import sys
from collections.abc import Callable
from pathlib import Path
from time import sleep

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent


def _is_notebook() -> bool:
    """
    This function returns True if the code is running in a Jupyter notebook.
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


# Conditional import of tqdm.
# The progress_monitor alias is used to display progress bars
# and it can be overridden by the user before running the testbench.
if _is_notebook():
    from tqdm.notebook import tqdm

    DUMMY_BAR_SLEEP = 0
else:
    from tqdm import tqdm

    DUMMY_BAR_SLEEP = 0.1
progress_monitor = lambda caller: tqdm


def get_class(class_name: str):
    """
    This function returns the class with the given name.
    """
    for module_name, module in sys.modules.items():
        if module_name.startswith("plctestbench"):
            if hasattr(module, class_name):
                return getattr(module, class_name)

    cls_ = get_class_plugin(class_name)
    if cls_ is not None:
        return cls_

    raise ValueError(f"The class {class_name} does not exist.")


def get_class_plugin(class_name: str):

    import importlib.util
    import os
    from pathlib import Path

    is_settings_class: bool = class_name.endswith("Settings")

    module_name = (
        class_name.replace("Settings", "") if is_settings_class else class_name
    )

    plugins_directory = os.getenv("PLUGINS_DIRECTORY")
    if not plugins_directory:
        raise ValueError(
            f"The class {class_name} does not exist and no plugins directory configured."
        )

    plugin_file_path = Path(plugins_directory) / f"{module_name}Algorithm.py"

    if not plugin_file_path.exists():
        raise ValueError(
            f"The class {class_name} does not exist and plugin file {plugin_file_path} not found."
        )

    # Load the plugin module
    spec = importlib.util.spec_from_file_location(
        f"{module_name}Algorithm", plugin_file_path
    )
    plugin_module = importlib.util.module_from_spec(spec)
    sys.modules[f"{module_name}Algorithm"] = plugin_module
    spec.loader.exec_module(plugin_module)

    # Get the class from the plugin module
    if hasattr(plugin_module, class_name):
        return getattr(plugin_module, class_name)

    return None


def compute_hash(obj):
    """
    This function returns the hash of the given object.
    """
    # Legacy artifact IDs must remain stable; never use this for security tokens.
    # pi-lens-ignore: python-weak-hash
    return int.from_bytes(hashlib.md5(str(obj).encode("utf-8"), usedforsecurity=False).digest()[:7], "little")


def escape_email(email):
    """
    This function escapes the given email address.
    """
    return email.replace("@", "_at_").replace(".", "_dot_")


def dummy_progress_bar(worker, desc: str | None = None):
    """
    This function is used to create a dummy progress bar.
    """
    desc = desc if desc is not None else str(worker)
    for _ in worker.progress_monitor(range(10), desc=desc):
        sleep(DUMMY_BAR_SLEEP)


def recursive_split_audio(
    audio: np.ndarray, xovers: list, bands: list | None = None
) -> list:
    if bands is None:
        bands = []
    lp_audio, hp_audio = xovers[0].split(audio)
    bands.append(lp_audio)
    if len(xovers) == 1:
        bands.append(hp_audio)
        return bands
    return recursive_split_audio(hp_audio, xovers[1:], bands)


def force_2d(arr) -> np.ndarray:
    if arr.ndim == 1:
        arr = np.expand_dims(arr, axis=-1)
    return arr


def prepare_progress_monitor(progress_monitor) -> Callable:
    def composite_progress_monitor(iterable, desc):
        for item in iterable:
            progress_monitor.update(1)
            yield item

    return composite_progress_monitor


def relative_to_root(path):
    project_path = PROJECT_ROOT.joinpath(path)
    if project_path.exists():
        return project_path

    # Wheels install bundled resources inside ``plctestbench/`` rather than
    # beside it in site-packages. Keep source-checkout paths as the default,
    # but fall back to the package directory for installed model files.
    package_path = PACKAGE_ROOT.joinpath(path)
    return package_path if package_path.exists() else project_path


def extract_intorni(
    audio_file, lost_samples_idxs, intorno_size, fs, packet_size, unique=False
):
    # Import lazily: file_wrapper imports this module during initialization.
    from .human_listening import loss_events

    intorno_samples = round(intorno_size * fs / 1000)
    if intorno_samples < 1 or packet_size < 1:
        raise ValueError("Excerpt and packet sizes must be positive")
    audio_data = audio_file.get_data()
    try:
        boundary_indexes = loss_events(lost_samples_idxs, len(audio_data))
    except ValueError as error:
        raise ValueError(f"Lost-sample indices must be integers within audio bounds: {error}") from error
    intorni = []
    packet_idxs = []
    for start_idx, end_idx in boundary_indexes:
        center_idx = (start_idx + end_idx - 1) // 2
        start_sample = center_idx - intorno_samples // 2
        end_sample = start_sample + intorno_samples
        # Do not wrap negative NumPy slices or return truncated boundary clips.
        if start_sample < 0 or end_sample > len(audio_data):
            continue
        intorno = audio_data[start_sample:end_sample]
        if unique:
            intorni.append(intorno.copy())
        else:
            intorni.append(intorno)
        packet_idxs.append(center_idx // packet_size)
    return packet_idxs, intorni


def force_single_loss_per_stimulus(lost_samples_idxs, fs, spacing, samples_per_packet):

    # TODO: Allow for consecutive lost packets. This solution will discard any subsequent lost packets.
    selected_idxs = []
    prev_packet_idx = -1

    for idx in lost_samples_idxs:
        packet_idx = idx // samples_per_packet
        if (
            prev_packet_idx == -1
            or (packet_idx - prev_packet_idx)
            >= spacing / 1000 * fs / samples_per_packet
            or packet_idx == prev_packet_idx
        ):
            selected_idxs.append(idx)
            prev_packet_idx = packet_idx

    return selected_idxs


def fade_in(audio, fs, fade_in_time) -> None:
    fade_in_samples = min(len(audio), max(0, math.trunc(fade_in_time * fs / 1000)))
    if fade_in_samples == 0:
        return
    fade_in_window = np.linspace(0, 1, fade_in_samples, dtype=audio.dtype)
    if audio.ndim > 1:
        fade_in_window = fade_in_window[:, np.newaxis]
    audio[:fade_in_samples] *= fade_in_window


def fade_out(audio, fs, fade_out_time) -> None:
    fade_out_samples = min(len(audio), max(0, math.trunc(fade_out_time * fs / 1000)))
    if fade_out_samples == 0:
        return
    fade_out_window = np.linspace(1, 0, fade_out_samples, dtype=audio.dtype)
    if audio.ndim > 1:
        fade_out_window = fade_out_window[:, np.newaxis]
    audio[-fade_out_samples:] *= fade_out_window


def leading_silence(audio, fs, silence_time) -> np.ndarray:
    silence_samples = math.trunc(silence_time * fs / 1000)
    silence = np.zeros((silence_samples, *audio.shape[1:]), dtype=audio.dtype)
    return np.concatenate((silence, audio), axis=0)


def trailing_silence(audio, fs, silence_time) -> np.ndarray:
    silence_samples = math.trunc(silence_time * fs / 1000)
    silence = np.zeros((silence_samples, *audio.shape[1:]), dtype=audio.dtype)
    return np.concatenate((audio, silence), axis=0)


def is_loud_enough(audio_data, audio_reference, threshold=-30) -> bool:
    """
    Determines if the average loudness of the audio data is above a specified threshold.

    Args:
        audio_data (numpy.ndarray): Audio data in the shape of [samples, channels].
        audio_reference (numpy.ndarray): Audio data in the shape of [samples, channels].
        threshold (float): Loudness threshold in dB (default is -30 dB).

    Returns:
        bool: True if the average loudness is above the threshold, False otherwise.
    """
    if not np.size(audio_data) or not np.size(audio_reference):
        return False
    if not np.all(np.isfinite(audio_data)) or not np.all(np.isfinite(audio_reference)) or not np.isfinite(threshold):
        return False
    rms = np.sqrt(np.mean(np.square(audio_data, dtype=np.float64)))
    rms_reference = np.sqrt(np.mean(np.square(audio_reference, dtype=np.float64)))
    return bool(rms_reference > 0 and rms > rms_reference * 10 ** (threshold / 20))


class ObjectFactory:
    def __init__(self):
        self._builders = {}

    def register_builder(self, key: str, builder: object):
        self._builders[key] = builder

    def create(self, key: str, *args, **kwargs):
        builder = self._builders.get(key)
        if not builder:
            raise ValueError(key)
        return builder(*args, **kwargs)
