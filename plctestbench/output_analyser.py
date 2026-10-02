import subprocess
from contextlib import contextmanager
from math import gcd
from pathlib import Path

import numpy as np
import numpy.random as npr
from pesqc2 import pesq
from scipy.signal import resample_poly

from .file_wrapper import AudioFile, DataFile, PEAQData, SimpleCalculatorData
from .listening_tests import ListeningTest
from .perceptual_metric import PerceptualMetric
from .plcmos import PLCMOSEstimator
from .settings import (
    HumanCalculatorSettings,
    MAECalculatorSettings,
    MSECalculatorSettings,
    PEAQCalculatorSettings,
    PEAQMode,
    PerceptualCalculatorSettings,
    PESQCalculatorSettings,
    PESQMode,
    PLCMOSCalculatorSettings,
    PLCMOSModel,
    Settings,
    SpectralEnergyCalculatorSettings,
    WindowedPEAQCalculatorSettings,
)
from .utils import (
    dummy_progress_bar,
    extract_intorni,
    force_single_loss_per_stimulus,
    is_loud_enough,
)
from .worker import Worker


def normalise(x, amp_scale=1.0):
    return amp_scale * x / np.amax(np.abs(x))


@contextmanager
def _temporary_audio_files(
    original: AudioFile,
    reconstructed: AudioFile,
    suffix: str,
    original_data: np.ndarray,
    reconstructed_data: np.ndarray,
):
    """Create paired temporary audio files and remove them on every exit path."""
    original_path = original.get_path()
    reconstructed_path = reconstructed.get_path()
    temporary_original_path = original_path[:-4] + suffix + original_path[-4:]
    temporary_reconstructed_path = reconstructed_path[:-4] + suffix + reconstructed_path[-4:]

    try:
        yield (
            AudioFile.from_audio_file(original, new_data=original_data, new_path=temporary_original_path),
            AudioFile.from_audio_file(
                reconstructed,
                new_data=reconstructed_data,
                new_path=temporary_reconstructed_path,
            ),
        )
    finally:
        Path(temporary_original_path).unlink(missing_ok=True)
        Path(temporary_reconstructed_path).unlink(missing_ok=True)


@contextmanager
def _normalised_audio_files(original: AudioFile, reconstructed: AudioFile):
    """Create normalized PEAQ inputs and always remove their temporary files."""
    with _temporary_audio_files(
        original,
        reconstructed,
        "_norm",
        normalise(original.get_data()),
        normalise(reconstructed.get_data()),
    ) as normalized_files:
        yield normalized_files


def _as_channel_matrix(audio: np.ndarray) -> np.ndarray:
    """Return audio in ``(samples, channels)`` form."""
    return audio[:, np.newaxis] if audio.ndim == 1 else audio


def _resample_channel_matrix(
    audio: np.ndarray, original_samplerate: float, target_samplerate: int = 16000
) -> np.ndarray:
    """Resample audio without importing librosa's optional legacy dependencies."""
    channels = _as_channel_matrix(np.asarray(audio))
    original_samplerate = int(original_samplerate)
    if original_samplerate == target_samplerate:
        return channels

    divisor = gcd(original_samplerate, target_samplerate)
    return resample_poly(
        channels,
        target_samplerate // divisor,
        original_samplerate // divisor,
        axis=0,
    )


class OutputAnalyser(Worker):
    """Base class for output analysers.

    Attributes:
        settings (Settings): Configuration used by the analyser.
        persistent (bool): Whether results produced by the analyser persist.
        progress_monitor: Progress-monitor factory bound to this worker.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize state shared by output analysers.

        Concrete analysers document their supported configuration parameters.
        """
        super().__init__(settings)


class SimpleCalculator(OutputAnalyser):
    """Base class for calculators that compare windowed audio frames.

    Attributes:
        settings (Settings): Windowing and normalization configuration.
    """

    def run(
        self,
        original_track_node: AudioFile,
        reconstructed_track_node: AudioFile,
    ) -> SimpleCalculatorData:
        """Normalize, window, and frame two audio tracks.

        Args:
            original_track_node: Original reference track.
            reconstructed_track_node: Reconstructed track under test.

        Returns:
            (tuple[np.ndarray, np.ndarray]): The windowed reference frames and
                reconstructed frames.
        """
        amp_scale = self.settings.get("amp_scale")
        N = self.settings.get("N")
        hop = self.settings.get("hop")
        original_track = original_track_node.get_data()
        reconstructed_track = reconstructed_track_node.get_data()

        x_r = normalise(original_track, amp_scale)
        x_e = normalise(reconstructed_track, amp_scale)

        num_samples = len(x_r)

        w = np.hanning(N + 1)[:-1]
        if x_r.ndim > 1:
            w = np.transpose(np.tile(w, (np.shape(x_r)[1], 1)))

        x_rw = np.array(
            [np.multiply(w, x_r[i : i + N]) for i in range(0, num_samples - N, hop)]
        )
        x_ew = np.array(
            [np.multiply(w, x_e[i : i + N]) for i in range(0, num_samples - N, hop)]
        )

        return x_rw, x_ew


class MSECalculator(SimpleCalculator):
    """Calculate the windowed mean squared error between two audio tracks.

    Attributes:
        settings (MSECalculatorSettings): Windowing and normalization
            configuration.
    """

    def __init__(self, settings: MSECalculatorSettings) -> None:
        """Initialize a mean squared error calculator.

        Other Parameters:
            N (int): Analysis-window length in samples. Defaults to ``1024``.
            hop (int | None): Number of samples between adjacent windows.
                Defaults to half of ``N`` when ``None``.
            amp_scale (float): Scale applied when normalizing each track.
                Defaults to ``1.0``.
        """
        super().__init__(settings)

    def run(
        self,
        original_track_node: AudioFile,
        reconstructed_track_node: AudioFile,
        lost_samples_idxs: DataFile = None,
        id: str = "",
    ):
        """Calculate the mean squared error for each pair of frames.

        Args:
            original_track_node: Original reference track.
            reconstructed_track_node: Reconstructed track under test.
            lost_samples_idxs: Unused loss-index data.
            id: Identifier shown by the progress monitor.

        Returns:
            (SimpleCalculatorData): Windowed mean squared error values.
        """
        x_rw, x_ew = super().run(original_track_node, reconstructed_track_node)
        error = [
            np.mean((x_rw[n] - x_ew[n]) ** 2, 0)
            for n in self.progress_monitor(range(len(x_rw)), desc=f"{str(self)}|{id}")
        ]
        return SimpleCalculatorData(error)


class MAECalculator(SimpleCalculator):
    """Calculate the windowed mean absolute error between two audio tracks.

    Attributes:
        settings (MAECalculatorSettings): Windowing and normalization
            configuration.
    """

    def __init__(self, settings: MAECalculatorSettings) -> None:
        """Initialize a mean absolute error calculator.

        Other Parameters:
            N (int): Analysis-window length in samples. Defaults to ``1024``.
            hop (int | None): Number of samples between adjacent windows.
                Defaults to half of ``N`` when ``None``.
            amp_scale (float): Scale applied when normalizing each track.
                Defaults to ``1.0``.
        """
        super().__init__(settings)

    def run(
        self,
        original_track_node: AudioFile,
        reconstructed_track_node: AudioFile,
        lost_samples_idxs: DataFile = None,
        id: str = "",
    ):
        """Calculate the mean absolute error for each pair of frames.

        Args:
            original_track_node: Original reference track.
            reconstructed_track_node: Reconstructed track under test.
            lost_samples_idxs: Unused loss-index data.
            id: Identifier shown by the progress monitor.

        Returns:
            (SimpleCalculatorData): Windowed mean absolute error values.
        """
        x_rw, x_ew = super().run(original_track_node, reconstructed_track_node)
        error = [
            np.mean(np.abs((x_rw[n] - x_ew[n])), 0)
            for n in self.progress_monitor(range(len(x_rw)), desc=f"{str(self)}|{id}")
        ]
        return SimpleCalculatorData(error)


class SpectralEnergyCalculator(OutputAnalyser):
    """Calculate the squared error between the magnitude spectra.

    Attributes:
        settings (SpectralEnergyCalculatorSettings): Spectral-analysis
            configuration.
    """

    def __init__(self, settings: SpectralEnergyCalculatorSettings) -> None:
        """Initialize a spectral-energy calculator.

        Other Parameters:
            N (int): DFT window length in samples. Defaults to ``1024``.
            hop (int | None): Number of samples between adjacent windows.
                Defaults to half of ``N`` when ``None``.
            amp_scale (float): Scale applied when normalizing each track.
                Defaults to ``1.0``.
        """
        super().__init__(settings)

    def run(
        self,
        original_track_node: AudioFile,
        reconstructed_track_node: AudioFile,
        lost_samples_idxs: DataFile = None,
        id: str = "",
    ):
        """Calculate short-time spectral differences between two tracks.

        Args:
            original_track_node: Original reference track.
            reconstructed_track_node: Reconstructed track under test.
            lost_samples_idxs: Unused loss-index data.
            id: Identifier shown by the progress monitor.

        Returns:
            (SimpleCalculatorData): The spectral-energy difference for each
                analysis frame.
        """
        amp_scale = self.settings.get("amp_scale")
        N = self.settings.get("N")
        hop = self.settings.get("hop")
        original_track = original_track_node.get_data()
        reconstructed_track = reconstructed_track_node.get_data()

        w = np.hanning(N + 1)[:-1]

        x_r = normalise(original_track, amp_scale)
        x_e = normalise(reconstructed_track, amp_scale)

        num_samples = len(x_r)

        if x_r.ndim > 1:
            w = w[:, None]

        x_rk = []
        x_ek = []
        for sample in self.progress_monitor(
            range(0, num_samples - N, hop), desc=f"{str(self)}|{id}"
        ):
            x_r_win = w * x_r[sample : sample + N]
            x_e_win = w * x_e[sample : sample + N]
            x_rk.append(np.fft.fft(x_r_win, axis=0))
            x_ek.append(np.fft.fft(x_e_win, axis=0))

        x_2rk = np.abs(np.array(x_rk)) ** 2
        x_2ek = np.abs(np.array(x_ek)) ** 2

        se = np.array(x_2rk - 2 * np.sqrt(x_2rk * x_2ek) + x_2ek)

        return SimpleCalculatorData(se)


class PEAQCalculator(OutputAnalyser):
    """Calculate whole-track PEAQ objective quality metrics.

    The underlying PEAQ implementation is
    [GstPEAQ](https://github.com/HSU-ANT/gstpeaq).

    Attributes:
        settings (PEAQCalculatorSettings): PEAQ processing configuration.
    """

    def __init__(self, settings: PEAQCalculatorSettings) -> None:
        """Initialize a PEAQ calculator.

        Other Parameters:
            peaq_mode (str): PEAQ processing mode: ``"basic"`` or
                ``"advanced"``. Defaults to ``"basic"``.
        """
        super().__init__(settings)

    def run(
        self,
        original_track_node: AudioFile,
        reconstructed_track_node: AudioFile,
        lost_samples_idxs: DataFile = None,
        id: str = "",
    ) -> PEAQData:
        """Calculate whole-track PEAQ metrics.

        Returns:
            (PEAQData | None): The objective difference grade and distortion
                index, or ``None`` when GstPEAQ returns invalid output.
        """
        peaq_mode: PEAQMode = self.settings.get("peaq_mode")
        if peaq_mode == PEAQMode.basic:
            mode_flag = "--basic"
        elif peaq_mode == PEAQMode.advanced:
            mode_flag = "--advanced"
        else:
            mode_flag = "--basic"
        with _normalised_audio_files(original_track_node, reconstructed_track_node) as (
            original_track_norm_file,
            reconstructed_track_norm_file,
        ):
            completed_process = subprocess.run(
                [
                    "peaq",
                    mode_flag,
                    "--gst-plugin-path",
                    "/usr/lib/gstreamer-1.0/",
                    original_track_norm_file.get_path(),
                    reconstructed_track_norm_file.get_path(),
                ],
                capture_output=True,
                text=True,
                check=False,
            )

        peaq_output = completed_process.stdout

        dummy_progress_bar(self, desc=f"{str(self)}|{id}")

        peaq_odg_text = "Objective Difference Grade: "
        peaq_di_text = "Distortion Index: "
        if peaq_odg_text in peaq_output and peaq_di_text in peaq_output:
            peaq_odg, peaq_di = peaq_output.split("\n", 1)
            _, peaq_odg = peaq_odg.split(peaq_odg_text)
            _, peaq_di = peaq_di.split(peaq_di_text)
            try:
                return PEAQData(float(peaq_odg), float(peaq_di))
            except ValueError:
                print("The peaq program returned invalid metric values:")
                print(completed_process.stdout)
                return None

        print("The peaq program exited with the following errors:")
        print(completed_process.stdout)


class WindowedPEAQCalculator(OutputAnalyser):
    """Calculate packet-aligned PEAQ scores around individual losses.

    The underlying PEAQ implementation is
    [GstPEAQ](https://github.com/HSU-ANT/gstpeaq).

    Attributes:
        fs (int): Audio sample rate in hertz.
        packet_size (int): Number of audio samples in each packet.
        intorno_length (int): Length of each loss-centered analysis region in
            milliseconds.
        mode_flag (str): Command-line flag selecting the GstPEAQ mode.
    """

    def __init__(self, settings: WindowedPEAQCalculatorSettings) -> None:
        """Initialize a windowed PEAQ calculator.

        Other Parameters:
            peaq_mode (str): PEAQ processing mode: ``"basic"`` or
                ``"advanced"``. Defaults to ``"basic"``.
            intorno_length (int): Duration of each loss-centered analysis
                region in milliseconds. Defaults to ``300``.
        """
        super().__init__(settings)
        self.fs = self.settings.get("fs")
        self.packet_size = self.settings.get("packet_size")
        self.intorno_length = self.settings.get("intorno_length")
        peaq_mode = self.settings.get("peaq_mode")
        self.mode_flag = "--advanced" if peaq_mode == PEAQMode.advanced else "--basic"

    def run(
        self,
        original_track_node: AudioFile,
        reconstructed_track_node: AudioFile,
        lost_samples_idxs_data: DataFile = None,
        id: str = "",
    ) -> SimpleCalculatorData:
        """Calculate PEAQ scores around packet-loss events.

        Returns:
            (SimpleCalculatorData): Packet-aligned DI and ODG values.
        """
        with _normalised_audio_files(original_track_node, reconstructed_track_node) as (
            original_track_norm_file,
            reconstructed_track_norm_file,
        ):
            lost_samples_idxs = lost_samples_idxs_data.get_data()
            intorni_original = extract_intorni(
                original_track_norm_file,
                lost_samples_idxs,
                self.intorno_length,
                self.fs,
                self.packet_size,
            )
            intorni_reconstructed = extract_intorni(
                reconstructed_track_norm_file,
                lost_samples_idxs,
                self.intorno_length,
                self.fs,
                self.packet_size,
            )

            metric = np.zeros((len(original_track_node.get_data()) // self.packet_size, 2))
            for idx, intorno_original, intorno_reconstructed in self.progress_monitor(
                zip(intorni_original[0], intorni_original[1], intorni_reconstructed[1]),
                total=len(intorni_original[1]),
                desc=f"{str(self)}|{id}",
            ):
                with _temporary_audio_files(
                    original_track_norm_file,
                    reconstructed_track_norm_file,
                    "_chunk",
                    intorno_original,
                    intorno_reconstructed,
                ) as (original_intorno_file, reconstructed_intorno_file):
                    completed_process = subprocess.run(
                        [
                            "peaq",
                            self.mode_flag,
                            "--gst-plugin-path",
                            "/usr/lib/gstreamer-1.0/",
                            original_intorno_file.get_path(),
                            reconstructed_intorno_file.get_path(),
                        ],
                        capture_output=True,
                        text=True,
                        check=False,
                    )

                peaq_output = completed_process.stdout

                peaq_odg_text = "Objective Difference Grade: "
                peaq_di_text = "Distortion Index: "
                if peaq_odg_text in peaq_output and peaq_di_text in peaq_output:
                    peaq_odg, peaq_di = peaq_output.split("\n", 1)
                    _, peaq_odg = peaq_odg.split(peaq_odg_text)
                    _, peaq_di = peaq_di.split(peaq_di_text)
                    try:
                        metric[idx] = [float(peaq_di), float(peaq_odg)]
                    except ValueError:
                        print("The peaq program returned invalid metric values:")
                        print(completed_process.stdout)
                else:
                    print("The peaq program exited with the following errors:")
                    print(completed_process.stdout)

        return SimpleCalculatorData(metric)


class PerceptualCalculator(OutputAnalyser):
    """Calculate the perceived audibility of packet-loss glitches.

    This objective metric uses a constant-Q time-frequency representation to
    compare original and reconstructed audio near packet-loss events. It is
    the official implementation of the method described in
    [this work](https://aes2.org/publications/elibrary-page/?id=23028).

    Processing pipeline:

    - Extract loss-centered regions from the original and reconstructed audio.
    - Compute a constant-Q spectrogram for each pair of regions.
    - Estimate the perceptual audibility of each glitch.
    - Store the results in a packet-aligned metric vector.

    Attributes:
        fs (int): Audio sample rate in hertz.
        packet_size (int): Number of audio samples in each packet.
        intorno_length (int): Length of each loss-centered analysis region in
            milliseconds.
        min_frequency (float): Lowest analyzed frequency in hertz.
        max_frequency (float): Highest analyzed frequency in hertz.
        bins_per_octave (int): Number of constant-Q bins per octave.
        n_bins (int): Total number of constant-Q frequency bins.
        minimum_window (int): Minimum transform-window length in samples.
    """

    def __init__(self, settings: PerceptualCalculatorSettings) -> None:
        """Initialize a perceptual glitch-audibility calculator.

        Other Parameters:
            intorno_length (int): Duration of each loss-centered analysis
                region in milliseconds. Defaults to ``300``.
            linear_mag (bool): Whether to use linear magnitudes. Defaults to
                ``False``.
            min_frequency (float): Lowest analyzed frequency in hertz.
                Defaults to ``32.7``.
            max_frequency (float): Highest analyzed frequency in hertz.
                Defaults to ``20000``.
            bins_per_octave (int): Number of constant-Q bins per octave.
                Defaults to ``12``.
            n_bins (int): Total number of constant-Q frequency bins. Defaults
                to ``100``.
            minimum_window (int): Minimum transform-window length in samples.
                Defaults to ``128``.
        """
        super().__init__(settings)
        self.fs = self.settings.get("fs")
        self.packet_size = self.settings.get("packet_size")
        self.intorno_length = self.settings.get("intorno_length")
        self.min_frequency = self.settings.get("min_frequency")
        self.max_frequency = self.settings.get("max_frequency")
        self.bins_per_octave = self.settings.get("bins_per_octave")
        self.n_bins = self.settings.get("n_bins")
        self.minimum_window = self.settings.get("minimum_window")

    def run(
        self,
        original_track_node: AudioFile,
        reconstructed_track_node: AudioFile,
        lost_samples_idxs_data: DataFile = None,
        id: str = "",
    ) -> SimpleCalculatorData:
        """Calculate perceptual scores around packet-loss events.

        Returns:
            (SimpleCalculatorData): Packet-aligned perceptual metric values.
        """
        lost_samples_idxs = lost_samples_idxs_data.get_data()
        intorni_original = extract_intorni(
            original_track_node,
            lost_samples_idxs,
            self.intorno_length,
            self.fs,
            self.packet_size,
        )
        intorni_reconstructed = extract_intorni(
            reconstructed_track_node,
            lost_samples_idxs,
            self.intorno_length,
            self.fs,
            self.packet_size,
        )

        pm = PerceptualMetric(
            self.min_frequency,
            self.max_frequency,
            self.bins_per_octave,
            self.n_bins,
            self.minimum_window,
            len(intorni_original[1][0][:, 0]),
            self.fs,
            self.intorno_length,
        )

        spectrograms = [
            {"idx": idx, **pm.spectrogram(original[:, 0], reconstructed[:, 0])}
            for idx, original, reconstructed in self.progress_monitor(
                zip(intorni_original[0], intorni_original[1], intorni_reconstructed[1]),
                total=len(intorni_original[1]),
                desc=f"{str(self)}|{id}",
            )
        ]

        metric = np.zeros((len(original_track_node.get_data()) // self.packet_size) + 1)

        for spectrogram in spectrograms:
            perc_metric = pm(spectrogram)
            metric[spectrogram["idx"]] = perc_metric

        return SimpleCalculatorData(metric)


class HumanCalculator(OutputAnalyser):
    """Generate and run a human listening test for packet-loss events.

    Attributes:
        fs (int): Audio sample rate in hertz.
        packet_size (int): Number of audio samples in each packet.
        stimulus_length (int): Stimulus duration in milliseconds.
        single_loss (bool): Whether each stimulus contains only one loss.
        stimuli_per_page (int): Number of stimuli shown on each page.
        pages (int): Number of listening-test pages.
        stimuli_number (int): Total number of stimuli in the test.
        choose_seed (int): Seed used when selecting stimuli.
        persistent (bool): Whether the listening-test result persists.
        listening_test (ListeningTest): Listening test created by
            [`run()`][plctestbench.output_analyser.HumanCalculator.run].
    """

    def __init__(self, settings: HumanCalculatorSettings) -> None:
        """Initialize a human listening-test calculator.

        Other Parameters:
            stimulus_length (int): Stimulus duration in milliseconds. Defaults
                to ``3000``.
            single_loss_per_stimulus (bool): Whether each stimulus may contain
                only one loss event. Defaults to ``True``.
            stimuli_per_page (int): Number of stimuli shown on each page.
                Defaults to ``10``.
            pages (int): Number of listening-test pages. Defaults to ``2``.
            iterations (int): Number of test iterations. Defaults to ``1``.
            choose_seed (int): Seed used to select stimuli. Defaults to ``1``.
            reference (str | None): Optional reference-track path. Defaults to
                ``None``.
            anchor (str | None): Optional anchor-track path. Defaults to
                ``None``.
        """
        super().__init__(settings)
        self.fs = self.settings.get("fs")
        self.packet_size = self.settings.get("packet_size")
        self.stimulus_length = self.settings.get("stimulus_length")
        self.single_loss = self.settings.get("single_loss_per_stimulus")
        self.stimuli_per_page = self.settings.get("stimuli_per_page")
        self.pages = self.settings.get("pages")
        self.stimuli_number = self.stimuli_per_page * self.pages
        self.choose_seed = self.settings.get("choose_seed")
        self.persistent = False

    def run(
        self,
        original_track_node: AudioFile,
        reconstructed_track_node: AudioFile,
        lost_samples_idxs_data: DataFile = None,
        id: str = "",
    ):
        """Run the listening test and collect its scores.

        Returns:
            (SimpleCalculatorData): Packet-aligned mean listening-test scores.
        """

        def transpose(matrix):
            return [
                [matrix[j][i] for j in range(len(matrix))]
                for i in range(len(matrix[0]))
            ]

        self.listening_test = ListeningTest(self.settings)

        if self.single_loss:
            lost_samples_idxs = force_single_loss_per_stimulus(
                lost_samples_idxs_data.get_data(),
                self.fs,
                self.stimulus_length / 2,
                self.packet_size,
            )
        else:
            lost_samples_idxs = lost_samples_idxs_data.get_data()
        intorni_original = extract_intorni(
            original_track_node,
            lost_samples_idxs,
            self.stimulus_length,
            self.fs,
            self.packet_size,
            unique=True,
        )
        intorni_reconstructed = extract_intorni(
            reconstructed_track_node,
            lost_samples_idxs,
            self.stimulus_length,
            self.fs,
            self.packet_size,
            unique=True,
        )

        intorni_original_loud = []
        intorni_reconstructed_loud = []
        for idx in range(len(intorni_original[1])):
            if is_loud_enough(
                intorni_original[1][idx], original_track_node.get_data(), -10
            ):
                intorni_original_loud.append(
                    [intorno[idx] for intorno in intorni_original]
                )
                intorni_reconstructed_loud.append(
                    [intorno[idx] for intorno in intorni_reconstructed]
                )

        intorni_original_loud = transpose(intorni_original_loud)
        intorni_reconstructed_loud = transpose(intorni_reconstructed_loud)

        if self.stimuli_number > len(intorni_original_loud[1]):
            error_message = f"The number of stimuli requested ({self.stimuli_number}) is greater than the number of stimuli available ({len(intorni_original_loud)}). Increase the total length of available audio."
            discarded_packets_close = (
                len(lost_samples_idxs_data.get_data()) - len(lost_samples_idxs)
            ) // self.packet_size
            if discarded_packets_close > 0:
                error_message += f" {discarded_packets_close} stimulus were discarded because too close to each other."
            discarded_packet_loud = len(intorni_original[1]) - len(
                intorni_original_loud[1]
            )
            if discarded_packet_loud > 0:
                error_message += f" {discarded_packet_loud} stimulus were discarded because too quiet."
            raise ValueError(error_message)

        npr.seed(self.choose_seed)
        stimuli_idxs = npr.choice(
            range(len(intorni_original_loud[1])), self.stimuli_number, replace=False
        )
        stimuli_original = transpose(
            [
                [intorno[idx] for intorno in intorni_original_loud]
                for idx in stimuli_idxs
            ]
        )
        stimuli_reconstructed = transpose(
            [
                [intorno[idx] for intorno in intorni_reconstructed_loud]
                for idx in stimuli_idxs
            ]
        )

        self.listening_test.set_references(
            list(zip(stimuli_original[0], stimuli_original[1])), original_track_node
        )
        self.listening_test.set_stimuli(
            list(zip(stimuli_reconstructed[0], stimuli_reconstructed[1])),
            original_track_node,
        )
        self.listening_test.set_indexes(stimuli_original[0])
        self.listening_test.generate_config()
        results = self.listening_test.get_results()

        metric = np.zeros(len(original_track_node.get_data()) // self.packet_size)
        for idx, mean, _ in self.progress_monitor(results, desc=f"{str(self)}|{id}"):
            try:
                metric[int(idx.split("-")[-1])] = mean
            except (IndexError, TypeError, ValueError):
                print(f"Skipping listening-test result with invalid index: {idx!r}")

        return SimpleCalculatorData(metric)


class PLCMOSCalculator(OutputAnalyser):
    """Estimate perceptual quality with PLCMOS.

    The implementation is taken from the
    [official repository](https://github.com/microsoft/PLC-Challenge/tree/main/PLCMOS).

    Attributes:
        settings (PLCMOSCalculatorSettings): PLCMOS model configuration.
    """

    def __init__(self, settings: PLCMOSCalculatorSettings) -> None:
        """Initialize a PLCMOS calculator.

        Other Parameters:
            plcmos_model (str): PLCMOS model version. Supported values are
                ``"0"``, ``"0alpha"``, ``"2-val"``, and ``"2"``. Defaults
                to ``"2"``.
            request_intrusive (bool): Whether model version ``"0"`` should use
                the original track as an intrusive reference. Defaults to
                ``True``.
        """
        super().__init__(settings)

    def run(
        self,
        original_track_node: AudioFile,
        reconstructed_track_node: AudioFile,
        lost_samples_idxs: DataFile = None,
        id: str = "",
    ) -> SimpleCalculatorData:
        """Calculate the channel-averaged PLCMOS score.

        Returns:
            (SimpleCalculatorData): The channel-averaged PLCMOS score.
        """
        plcmos_model: PLCMOSModel = self.settings.get("plcmos_model")
        request_intrusive: bool = self.settings.get("request_intrusive")

        plcmos = PLCMOSEstimator(model_version=plcmos_model.value)

        # Use intrusive model
        is_intrusive = plcmos_model == PLCMOSModel.plcmos_0alpha or (
            plcmos_model == PLCMOSModel.plcmos_0 and request_intrusive
        )

        reconstructed_track_node_resampled = _resample_channel_matrix(
            reconstructed_track_node.get_data(),
            reconstructed_track_node.get_samplerate(),
        )
        original_track_node_resampled = _resample_channel_matrix(
            original_track_node.get_data(),
            original_track_node.get_samplerate(),
        )
        if (
            reconstructed_track_node_resampled.shape[1]
            != original_track_node_resampled.shape[1]
        ):
            raise ValueError(
                "Original and reconstructed tracks must have the same channels"
            )

        score = 0
        channel_count = original_track_node_resampled.shape[1]
        for channel_idx in range(channel_count):
            score += plcmos.run(
                reconstructed_track_node_resampled[:, channel_idx],
                16000,
                original_track_node_resampled[:, channel_idx] if is_intrusive else None,
            )

        dummy_progress_bar(self, desc=f"{str(self)}|{id}")

        score /= channel_count
        return SimpleCalculatorData(score)


class PESQCalculator(OutputAnalyser):
    """Calculate a whole-track PESQ quality score.

    The implementation is taken from `pesqc2`, which implements wideband PESQ
    according to the P.862.2 with Corrigendum 2 specification.

    Attributes:
        settings (PESQCalculatorSettings): PESQ bandwidth configuration.
    """

    def __init__(self, settings: PESQCalculatorSettings) -> None:
        """Initialize a PESQ calculator.

        Other Parameters:
            pesq_mode (PESQMode): PESQ bandwidth mode. Use ``PESQMode.wb``
                for wideband or ``PESQMode.nb`` for narrowband. Defaults to
                ``PESQMode.wb``.
        """
        super().__init__(settings)

    def run(
        self,
        original_track_node: AudioFile,
        reconstructed_track_node: AudioFile,
        lost_samples_idxs: DataFile = None,
        id: str = "",
    ) -> SimpleCalculatorData:
        """Calculate the PESQ score after resampling and downmixing.

        Returns:
            (SimpleCalculatorData): The whole-track PESQ score.
        """
        pesq_mode: PESQMode = self.settings.get("pesq_mode")

        # Resample and downmix to mono after normalizing mono and multichannel
        # inputs to the same samples-by-channels representation.
        reconstructed_track_node_resampled = _resample_channel_matrix(
            reconstructed_track_node.get_data(),
            reconstructed_track_node.get_samplerate(),
        ).mean(axis=1)
        original_track_node_resampled = _resample_channel_matrix(
            original_track_node.get_data(),
            original_track_node.get_samplerate(),
        ).mean(axis=1)

        score = pesq(
            16000,
            original_track_node_resampled,
            reconstructed_track_node_resampled,
            mode=pesq_mode.value,
        )

        dummy_progress_bar(self, desc=f"{str(self)}|{id}")

        return SimpleCalculatorData(score)
