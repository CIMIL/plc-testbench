import logging
from math import ceil

import librosa
import numpy as np

try:
    from burg_plc import BurgBasic
except ImportError:
    BurgBasic = None
    logging.warning("Burg PLC unavailable")
try:
    from cpp_plc_template import BasePlcTemplate  # type: ignore[import-not-found]
except ImportError:
    BasePlcTemplate = None
    logging.warning("External PLC unavailable")
from plctestbench.worker import Worker

from .crossfade import Crossfade, MultibandCrossfade
from .filters import LinkwitzRileyCrossover
from .low_cost_concealment import LowCostConcealment
from .parcnet import PARCnet
from .verma import VermaNet
from .settings import (
    PARCnetPLCSettings,
    BurgPLCSettings,
    ClipStrategy,
    VermaPLCSettings,
    ExternalPLCSettings,
    LastPacketPLCSettings,
    LowCostPLCSettings,
    PLCSettings,
    Settings,
    StereoImageType,
    ZerosPLCSettings,
)
from .spatial import CodecMode, MidSideCodec
from .utils import force_2d, prepare_progress_monitor, recursive_split_audio


class PLCAlgorithm(Worker):
    """Base class for packet-loss concealment algorithms.

    Attributes:
        packet_size (int): Number of samples processed per packet.
        crossfade_settings: Configuration for the active crossfade processor.
        crossfade_class: Crossfade implementation selected for the configured
            frequency bands.
        crossfade: Processor that blends predictions into valid audio.
        fade_in (Crossfade): Processor that blends context into predictions.
        algorithm_context_length (int): Number of context samples retained by
            the algorithm.
        n_channels (int): Number of channels in the current track.
        context (np.ndarray): Most recent samples available to the predictor.
    """

    def __init__(self, settings: PLCSettings) -> None:
        """Initialize common packet-loss concealment state.

        Other Parameters:
            crossfade (list[CrossfadeSettings] | CrossfadeSettings): Crossfade
                configuration applied after a loss.
            fade_in (list[CrossfadeSettings] | CrossfadeSettings): Fade-in
                configuration applied to predicted packets.
            crossfade_frequencies (list[int] | None): Frequencies separating
                independently crossfaded bands.
            crossover_order (int | None): Order of the crossover filters.
        """
        super().__init__(settings)
        self.packet_size = self.settings.get("packet_size")
        self.crossfade_settings = self.settings.get("crossfade")
        if len(self.crossfade_settings) > 1:
            self.crossfade_class = MultibandCrossfade
        else:
            self.crossfade_class = Crossfade
            self.crossfade_settings = self.crossfade_settings[0]
        self.crossfade = self.crossfade_class(self.settings, self.crossfade_settings)
        fade_in_settings = self.settings.get("fade_in")[0]
        if fade_in_settings.get("length") > self.packet_size:
            raise ValueError("fade in length cannot be longer than the packet size")
        self.fade_in = Crossfade(self.settings, fade_in_settings)
        try:
            self.algorithm_context_length = int(
                self.settings.get("context_length") * self.settings.get("fs") / 1000
            )
        except Exception:
            self.algorithm_context_length = self.packet_size

    def run(self, original_track: np.ndarray, lost_samples_idx: np.ndarray, id):
        """ """

        def zero_pad(original_track: np.ndarray):
            """ """
            rounding_difference = self.packet_size - track_length % self.packet_size
            npad = ((0, rounding_difference), (0, 0))
            return np.pad(original_track, npad, "constant")

        original_track = force_2d(original_track)
        self.n_channels = np.shape(original_track)[1]
        lost_packets_idx = lost_samples_idx[:: self.packet_size] / self.packet_size
        track_length = len(original_track)
        n_packets = ceil(track_length / self.packet_size)
        original_track = zero_pad(original_track)
        reconstructed_track = np.zeros(np.shape(original_track), np.float32)
        self._prepare_to_play()

        j = 0
        for i in self.progress_monitor(range(n_packets), desc=f"{str(self)}|{id}"):
            if i > lost_packets_idx[j] and j < len(lost_packets_idx) - 1:
                j += 1
            start_idx = i * self.packet_size
            end_idx = (i + 1) * self.packet_size
            buffer = original_track[start_idx:end_idx]
            is_valid = not i == lost_packets_idx[j]
            reconstructed_buffer = self._tick(buffer, is_valid)
            reconstructed_track[start_idx:end_idx] = reconstructed_buffer

        return reconstructed_track[:track_length]

    def _prepare_to_play(self):
        """
        Not all the PLC algorithms need to prepare to play.
        This function will be executed when the PLC algorithm
        doesn't override it (because it doesn't need it) so
        this function does nothing.
        """
        self.context = np.zeros((self.algorithm_context_length, self.n_channels))

    def _tick(self, buffer: np.ndarray, is_valid: bool) -> np.ndarray:
        """
        This function is called for every buffer. It manages the creation
        of the predicted buffer and the crossfading with the following audio.
        """
        output_buffer = self._a_priori(buffer, is_valid)
        if is_valid:
            output_buffer = self._crossfade(output_buffer)
        else:
            output_buffer = self._predict(output_buffer)
            output_buffer = self._fade_in(output_buffer)
            self.crossfade.start()
        output_buffer = self._a_posteriori(output_buffer, is_valid)
        return output_buffer

    def _a_priori(self, buffer: np.ndarray, is_valid: bool) -> np.ndarray:
        """
        This function is called for every buffer.
        """
        return buffer

    def _a_posteriori(self, buffer: np.ndarray, is_valid: bool) -> np.ndarray:
        """
        This function is called for every buffer.
        """
        self.context = np.roll(self.context, -self.packet_size, axis=1)
        self.context[-self.packet_size :, :] = buffer
        return buffer

    def _fade_in(self, buffer: np.ndarray) -> np.ndarray:
        """
        This function is called for every buffer.
        """
        self.fade_in.start()
        output_buffer = self.fade_in(self.context[-self.packet_size :], buffer)
        return output_buffer

    def _predict(self, buffer: np.ndarray) -> np.ndarray:
        """
        This function is called for every buffer.
        """
        return buffer

    def _crossfade(self, buffer: np.ndarray) -> np.ndarray:
        """
        This function is called for every buffer.
        """
        output_buffer = buffer
        if self.crossfade.ongoing():
            prediction = self._predict(buffer)
            output_buffer = self.crossfade(prediction, buffer)
        return output_buffer


class AdvancedPLC(PLCAlgorithm):
    """Apply independently configured PLC algorithms to channels and bands.

    Attributes:
        plc_algorithms (dict): PLC algorithm chain for each channel group.
        frequencies (dict[str, list[int]]): Crossover frequencies by channel.
        crossover_order (int): Order of each crossover filter.
        stereo_image_processing (StereoImageType): Stereo processing mode.
        channel_link (bool): Whether channels are processed together.
        fs (int): Audio sample rate in hertz.
        crossovers (dict): Crossover filters for each channel group.
        mid_side (bool): Whether mid/side processing is enabled.
        mid_side_codec (MidSideCodec): Stereo mid/side converter.
    """

    def get_worker(self, worker_settings, settings):
        class_name = type(worker_settings).__name__.replace("Settings", "")
        worker_settings.set_progress_monitor(settings.get_progress_monitor())
        return globals()[class_name](worker_settings)

    def __init__(self, settings: Settings) -> None:
        """Initialize a multichannel, multiband PLC pipeline.

        Other Parameters:
            band_settings (dict[str, list[PLCSettings]]): PLC algorithms for
                each channel group and frequency band.
            frequencies (dict[str, list[int]]): Crossover frequencies for each
                channel group.
            order (int): Crossover-filter order. Defaults to ``4``.
            stereo_image_processing (StereoImageType): Stereo processing mode.
            channel_link (bool): Whether channels are processed together.
                Defaults to ``True``.
        """
        Worker.__init__(self, settings)
        all_plc_settings = self.settings.get("settings")
        self.plc_algorithms = {
            channel: [
                self.get_worker(worker_settings, settings)
                for worker_settings in settings_list
            ]
            for channel, settings_list in all_plc_settings.items()
        }
        self.frequencies = self.settings.get("frequencies")
        self.crossover_order = self.settings.get("order")
        self.stereo_image_processing = self.settings.get("stereo_image_processing")
        self.channel_link = self.settings.get("channel_link")
        self.fs = self.settings.get("fs")
        self.crossovers = {
            channel: [
                LinkwitzRileyCrossover(self.crossover_order, freq, self.fs)
                for freq in frequency_list
            ]
            for channel, frequency_list in self.frequencies.items()
        }
        self.mid_side = (
            True if self.stereo_image_processing == StereoImageType.mid_side else False
        )
        self.mid_side_codec = MidSideCodec()

    def run(self, original_track: np.ndarray, lost_samples_idx: np.ndarray, id):
        """ """
        original_track = force_2d(original_track)
        original_track = (
            self.mid_side_codec(original_track) if self.mid_side else original_track
        )
        processed_track = {}
        if self.channel_link:
            processed_track["linked"] = original_track
        else:
            for idx, channel in enumerate(self.frequencies.keys()):
                processed_track[channel] = force_2d(original_track[:, idx])
        for channel, crossovers in self.crossovers.items():
            if len(crossovers) == 0:
                processed_track[channel] = [processed_track[channel]]
                continue
            processed_track[channel] = recursive_split_audio(
                processed_track[channel], crossovers, []
            )

        reconstructed_track = np.zeros(np.shape(original_track), np.float32)
        reconstructed_track_bands = {
            channel: np.zeros(np.shape(track_bands[0]), np.float32)
            for channel, track_bands in processed_track.items()
        }
        n_packets = ceil(len(original_track) / self.settings.get("packet_size"))
        number_of_iterations = (
            sum(
                [len(plc_algorithms) for plc_algorithms in self.plc_algorithms.values()]
            )
            * n_packets
        )
        progress_monitor = self.progress_monitor(
            total=number_of_iterations, desc=f"{str(self)}|{id}"
        )
        composite_progress_monitor = prepare_progress_monitor(progress_monitor)

        for channel, plc_algorithms in self.plc_algorithms.items():
            for idx, plc_algorithm in enumerate(plc_algorithms):
                plc_algorithm.set_progress_monitor(composite_progress_monitor)
                progress_monitor.set_description(
                    f"{self} - {channel} - {plc_algorithm}"
                )
                reconstructed_track_bands[channel] += plc_algorithm.run(
                    processed_track[channel][idx], lost_samples_idx, id
                )
        progress_monitor.set_description(f"{self}")
        progress_monitor.close()
        if not self.channel_link:
            reconstructed_track = np.concatenate(
                list(reconstructed_track_bands.values()), axis=-1
            )
        else:
            reconstructed_track = reconstructed_track_bands["linked"]
        if self.mid_side:
            reconstructed_track = self.mid_side_codec(
                reconstructed_track, type=CodecMode.DECODE
            )

        return reconstructed_track


class ZerosPLC(PLCAlgorithm):
    """Replace every lost packet with zeros.

    Attributes:
        settings (ZerosPLCSettings): Zero-filling and crossfade configuration.
    """

    def __init__(self, settings: ZerosPLCSettings) -> None:
        """Initialize a zero-filling PLC algorithm.

        Other Parameters:
            crossfade (list[CrossfadeSettings] | CrossfadeSettings): Crossfade
                configuration applied after a loss.
            fade_in (list[CrossfadeSettings] | CrossfadeSettings): Fade-in
                configuration applied to predicted packets.
            crossfade_frequencies (list[int] | None): Frequencies separating
                independently crossfaded bands.
            crossover_order (int | None): Order of the crossover filters.
        """
        super().__init__(settings)

    def _predict(self, buffer: np.ndarray):
        """ """
        return np.zeros(np.shape(buffer))


class LastPacketPLC(PLCAlgorithm):
    """Conceal losses by repeating or mirroring the previous packet.

    Mirroring sample values can move them outside the normalized ``[-1, 1]``
    range. The two clipping strategies handle that overflow differently:

    - ``ClipStrategy.clip`` clamps every sample independently to ``-1`` or
      ``1``. This guarantees bounded output, but large peaks become flat.
    - ``ClipStrategy.subtract`` finds the first out-of-range sample and shifts
      it and the rest of the packet by that sample's excess. The first
      overflow lands on ``-1`` or ``1`` while relative differences in the
      remaining waveform are preserved.

    Attributes:
        mirror_x (bool): Whether to reverse the packet in time.
        mirror_y (bool): Whether to mirror sample values around the first
            sample when time reversal is enabled.
        clip_strategy (ClipStrategy): Strategy used for out-of-range samples.
    """

    def __init__(self, settings: LastPacketPLCSettings) -> None:
        """Initialize a previous-packet concealment algorithm.

        Other Parameters:
            mirror_x (bool): Whether to reverse the packet in time. Defaults to
                ``False``.
            mirror_y (bool): Whether to mirror sample values. Defaults to
                ``False``.
            clip_strategy (ClipStrategy): Either clamp samples with
                ``ClipStrategy.clip`` or shift the overflowing waveform tail
                with ``ClipStrategy.subtract``. Defaults to
                ``ClipStrategy.subtract``.
        """
        super().__init__(settings)
        self.mirror_x = settings.get("mirror_x")
        self.mirror_y = settings.get("mirror_y")
        self.clip_strategy = settings.get("clip_strategy")

    def _predict(self, buffer: np.ndarray):
        """ """

        def _flip_in_place(buffer: np.ndarray):
            """ """
            return -(buffer - buffer[0]) + buffer[0]

        reconstructed_buffer = self.context[-self.packet_size :]
        if self.mirror_x:
            reconstructed_buffer = np.flip(reconstructed_buffer, axis=0)
            if self.mirror_y:
                for channel in range(self.n_channels):
                    reconstructed_buffer[:, channel] = _flip_in_place(
                        reconstructed_buffer[:, channel]
                    )
                    reconstructed_buffer[:, channel] = self._clip(
                        reconstructed_buffer[:, channel]
                    )
        return reconstructed_buffer

    def _clip(self, buffer: np.ndarray) -> np.ndarray:
        """Apply the configured overflow strategy to a mirrored packet.

        ``ClipStrategy.clip`` hard-limits every sample to ``[-1, 1]``.
        ``ClipStrategy.subtract`` instead shifts the packet tail by the excess
        of its first out-of-range sample. The latter preserves relative sample
        differences but does not independently clamp every subsequent sample.
        """
        if self.clip_strategy == ClipStrategy.clip:
            return np.clip(buffer, -1.0, 1.0)

        out_of_range = np.flatnonzero(np.abs(buffer) > 1.0)
        if out_of_range.size > 0:
            index = out_of_range[0]
            buffer[index:] -= buffer[index] - np.sign(buffer[index])
        return buffer


class LowCostPLC(PLCAlgorithm):
    """Implement low-cost concealment (LCC).

    LCC preprocesses recent valid audio to expose its periodic structure,
    detects suitable zero crossings, and extracts recent waveform periods.
    It phase-aligns and repeats those periods to replace a lost packet. This
    avoids the model fitting and autocorrelation cost of an autoregressive
    predictor and is therefore well suited to low-delay processing.

    The algorithm performs its own transitions: it fades from a short linear
    extrapolation into the repeated waveform and fades the concealment tail
    into the next valid packet. The generic
    [`PLCAlgorithm`][plctestbench.plc_algorithm.PLCAlgorithm] fade-in and
    crossfade should therefore remain disabled when LCC's built-in fade
    lengths are nonzero; otherwise the same transition would be applied twice.
    The settings validator rejects a generic fade-in and a single-band generic
    crossfade when the corresponding built-in transition is enabled.

    This implementation follows Marco Fink and Udo Zölzer,
    [*Low-delay error concealment with low computational overhead for audio
    over IP applications*](https://dafx.de/paper-archive/2014/dafx14_marco_fink_low_delay_error_concealme.pdf).

    Attributes:
        lcc (LowCostConcealment): Low-cost concealment processor.
        samplerate (int): Audio sample rate in hertz.
    """

    def __init__(self, settings: LowCostPLCSettings) -> None:
        """Initialize the low-cost concealment processor.

        Other Parameters:
            max_frequency (float): Highest processed frequency in hertz.
                Defaults to ``4800``.
            f_min (int): Lowest processed frequency in hertz. Defaults to
                ``80``.
            beta (float): LCC tuning coefficient. Defaults to ``1``.
            n_m (int): Number of modeled components. Defaults to ``2``.
            fade_in_length (int): Fade-in length. Defaults to ``10``.
            fade_out_length (float): Fade-out length. Defaults to ``0.5``.
            extraction_length (int): Context extraction length. Defaults to
                ``2``.
        """
        super().__init__(settings)
        self.lcc = LowCostConcealment(
            settings.get("max_frequency"),
            settings.get("f_min"),
            settings.get("beta"),
            settings.get("n_m"),
            settings.get("fade_in_length"),
            settings.get("fade_out_length"),
            settings.get("extraction_length"),
        )
        self.samplerate = settings.get("fs")

    def _prepare_to_play(self):
        super()._prepare_to_play()
        self.lcc.prepare_to_play(self.samplerate, self.packet_size, self.n_channels)

    def _a_priori(self, buffer: np.ndarray, is_valid: bool):
        """ """
        return self.lcc.process(buffer, is_valid)


class BurgPLC(PLCAlgorithm):
    """Predict lost packets with a Burg autoregressive model. Implementation
    taken from [here](https://github.com/matteosacchetto/burg-implementation-experiments).

    An autoregressive (AR) model represents each sample as a weighted sum of a
    fixed number of preceding samples. After fitting those weights on recent
    valid context, the model recursively predicts the samples of a missing
    packet. This works best while the signal remains locally stationary.

    Burg's parameter-estimation method determines the AR coefficients by
    recursively minimizing both forward and backward prediction errors. It
    updates one reflection coefficient at each model order, avoids explicitly
    forming an autocorrelation matrix, and produces a stable all-pole model.

    Attributes:
        order (int): Order of the autoregressive model.
        previous_valid (bool): Whether the previous packet was received.
        coefficients (np.ndarray): Current autoregressive coefficients.
        burg: Platform-specific Burg predictor.
    """

    def __init__(self, settings: BurgPLCSettings) -> None:
        """Initialize the Burg autoregressive predictor.

        Other Parameters:
            context_length (int): Context duration in milliseconds. Defaults
                to ``100``.
            order (int): Autoregressive model order. Defaults to ``1``.
        """
        super().__init__(settings)
        self.order = settings.get("order")
        self.previous_valid = False
        self.coefficients = np.zeros(self.order)
        context_length_samples = round(
            self.algorithm_context_length / 1000 * self.settings.get("fs")
        )
        if BurgBasic is None:
            raise ImportError("Burg PLC is not available on this platform.")
        self.burg = BurgBasic(context_length_samples)

    def _predict(self, buffer: np.ndarray):
        """ """
        reconstructed_buffer = np.zeros(np.shape(buffer), np.float32)
        n_channels = np.shape(buffer)[1]
        for n_channel in range(n_channels):
            context = self.context[:, n_channel]
            if self.previous_valid:
                self.coefficients, _ = self.burg.fit(context, self.order)
            reconstructed_buffer[:, n_channel] = self.burg.predict(
                context, self.coefficients, self.packet_size
            )
        return reconstructed_buffer

    def _a_posteriori(self, buffer: np.ndarray, is_valid: bool) -> np.ndarray:
        """ """
        super()._a_posteriori(buffer, is_valid)
        self.previous_valid = is_valid
        return buffer


class ExternalPLC(PLCAlgorithm):
    """Delegate concealment to the optional external PLC implementation.

    Attributes:
        bpt: Configured external ``BasePlcTemplate`` processor.
    """

    def __init__(self, settings: ExternalPLCSettings) -> None:
        """Initialize the optional external PLC processor.

        Other Parameters:
            crossfade (list[CrossfadeSettings] | CrossfadeSettings): Crossfade
                configuration applied after a loss.
            fade_in (list[CrossfadeSettings] | CrossfadeSettings): Fade-in
                configuration applied to predicted packets.
            crossfade_frequencies (list[int] | None): Frequencies separating
                independently crossfaded bands.
            crossover_order (int | None): Order of the crossover filters.
        """
        super().__init__(settings)
        if BasePlcTemplate is None:
            raise ImportError("External PLC is not available on this platform.")
        self.bpt = BasePlcTemplate()
        self.bpt.prepare_to_play(self.settings.get("fs"), self.packet_size)

    def _tick(self, buffer: np.ndarray, is_valid: bool):
        """ """
        buffer = np.transpose(buffer)
        reconstructed_buffer = np.zeros(np.shape(buffer), np.float32)
        self.bpt.process(buffer, reconstructed_buffer, is_valid)
        reconstructed_buffer = np.transpose(reconstructed_buffer)
        return reconstructed_buffer


class VermaPLC(PLCAlgorithm):
    """Predict lost audio with the Verma neural-network model. Reference paper:
    [*A Deep Learning Approach for
    Low-Latency Packet Loss Concealment of Audio Signals in Networked Music
    Performance Applications*](https://ieeexplore.ieee.org/document/9210988).

    The method converts recent valid audio into a mel spectrogram and supplies
    that time-frequency context, together with the immediately preceding audio
    packet, to a neural synthesis model. A convolutional encoder summarizes
    the spectral context and a fully connected synthesis network predicts the
    missing waveform directly. This implementation runs an exported ONNX model
    independently for each channel and uses the newest quarter of the
    configured context—the paper's two-second full-resolution branch with the
    default eight-second context.

    Attributes:
        model (VermaNet): Neural-network inference wrapper.
        fs_dl (int): Sample rate expected by the model.
        context_length (int): Configured context duration in milliseconds.
        context_length_samples (int): Context duration converted to samples.
        hop_size (int): Spectrogram hop size in samples.
        window_length (int): Spectrogram window length in samples.
        lower_edge_hertz (float): Lowest mel-spectrogram frequency.
        upper_edge_hertz (float): Highest mel-spectrogram frequency.
        num_mel_bins (int): Number of mel-frequency bins.
        sample_rate (int): Input audio sample rate in hertz.
    """

    def __init__(self, settings: VermaPLCSettings) -> None:
        """Initialize the Verma neural predictor.

        Other Parameters:
            model_path (str): Path to the ONNX model.
            fs_dl (int): Sample rate expected by the model. Defaults to
                ``16000``.
            context_length (int): Context duration in milliseconds. Defaults
                to ``8000``.
            hop_size (int): Spectrogram hop size. Defaults to ``160``.
            window_length (int): Spectrogram window length. Defaults to
                ``480``.
            lower_edge_hertz (float): Lowest mel frequency. Defaults to
                ``40.0``.
            upper_edge_hertz (float): Highest mel frequency. Defaults to
                ``7600.0``.
            num_mel_bins (int): Number of mel bins. Defaults to ``100``.
        """
        super().__init__(settings)
        self.model = VermaNet(settings.get("model_path"))
        self.fs_dl = settings.get("fs_dl")
        self.context_length = settings.get("context_length")
        self.context_length_samples = settings.get("context_length_samples")
        self.hop_size = settings.get("hop_size")
        self.window_length = settings.get("window_length")
        self.lower_edge_hertz = settings.get("lower_edge_hertz")
        self.upper_edge_hertz = settings.get("upper_edge_hertz")
        self.num_mel_bins = settings.get("num_mel_bins")
        self.sample_rate = settings.get("fs")

    def _predict(self, buffer: np.ndarray):
        """ """
        return self._predict_reconstructed_buffer(buffer)

    def _compute_spectrogram(self, context, fs):
        return librosa.feature.melspectrogram(
            y=np.pad(context, (0, self.window_length - self.hop_size)),
            sr=fs,
            n_fft=self.window_length,
            hop_length=self.hop_size,
            win_length=self.window_length,
            center=False,
            n_mels=self.num_mel_bins,
            fmin=self.lower_edge_hertz,
            fmax=self.upper_edge_hertz,
        )

    def _predict_reconstructed_buffer(self, buffer):
        reconstructed_buffer = np.zeros(np.shape(buffer.T))
        context = librosa.resample(
            self.context.T, orig_sr=self.sample_rate, target_sr=self.fs_dl
        ).T
        left_pad_size = self.context.shape[0] - context.shape[0]
        context = np.pad(
            context, pad_width=((left_pad_size, 0), (0, 0)), mode="constant"
        )
        for channel_index in range(np.shape(buffer)[1]):
            spectrogram_2s = self._compute_spectrogram(
                context[-round(self.context_length_samples / 4) :, channel_index],
                self.fs_dl,
            )
            spectrograms = spectrogram_2s[np.newaxis, ..., np.newaxis]
            last_packet = np.expand_dims(
                context[-self.packet_size :, channel_index], axis=0
            )
            reconstructed_buffer[channel_index, :] = self.model(
                (spectrograms, last_packet)
            )
        return reconstructed_buffer.T


class PARCnetPLC(PLCAlgorithm):
    """Predict lost packets with the PARCnet hybrid AR/neural model.
    Taken from the [official PARCnet implementation](https://github.com/polimi-ispl/PARCnet). Reference paper:
    [*Hybrid Packet Loss Concealment for Real-Time Networked Music
    Applications*](https://ieeexplore.ieee.org/abstract/document/10360264).

    PARCnet combines two parallel predictors. An online linear AR branch fits
    recent valid samples using autocorrelation and Levinson--Durbin recursion,
    while a causal neural branch predicts the residual that the linear model
    cannot explain. Their outputs are added to form the concealed packet. The
    neural contribution is faded in to reduce boundary discontinuities, and
    this implementation predicts an extra tail that is crossfaded into the
    next valid packet. For burst losses, it also fades the next prediction into
    the previous prediction tail.

    Attributes:
        dl_model_path (str): Path to the neural-network model.
        ar_order (int): Order of the autoregressive predictor.
        ar_fade_dim (int): Configured autoregressive fade length; retained for
            model-configuration compatibility.
        ar_diagonal_load (float): Diagonal loading applied to AR fitting.
        dl_fs (int): Sample rate expected by the neural model.
        extra_packet_dim (int): Number of extra prediction samples.
        nn_fade_dim (int): Neural-network crossfade length in samples.
        context_length_blocks (int): Number of context packets.
        context_length_samples (int): Total context length in samples.
        model (PARCnet): PARCnet inference wrapper.
        extra_pred_buffer (np.ndarray): Prediction tail blended into the next
            valid packet.
        is_burst (bool): Whether the current loss belongs to a burst.
    """

    def __init__(self, settings: PARCnetPLCSettings) -> None:
        """Initialize the PARCnet predictor.

        Other Parameters:
            dl_model_path (str): Path to the neural-network model.
            dl_fs (int): Model sample rate. Defaults to ``44100``.
            extra_packet_dim (int): Extra prediction length. Defaults to
                ``256``.
            ar_order (int): Autoregressive model order. Defaults to ``256``.
            ar_fade_dim (int): Autoregressive fade length retained for model
                configuration compatibility. Defaults to ``8``.
            ar_diagonal_load (float): AR diagonal loading. Defaults to
                ``0.001``.
            context_length_blocks (int): Number of context packets. Defaults
                to ``8``.
            nn_fade_dim (int): Neural-network fade length. Defaults to ``64``.
        """
        super().__init__(settings)

        self.dl_model_path = settings.get("dl_model_path")
        self.ar_order = settings.get("ar_order")
        self.ar_fade_dim = settings.get("ar_fade_dim")
        self.ar_diagonal_load = settings.get("ar_diagonal_load")
        self.dl_fs = settings.get("dl_fs")
        self.extra_packet_dim = settings.get("extra_packet_dim")
        self.nn_fade_dim = settings.get("nn_fade_dim")
        self.context_length_blocks = self.settings.get("context_length_blocks")

        self.context_length_samples = self.context_length_blocks * self.packet_size

        self.algorithm_context_length = self.context_length_samples

        self.model = PARCnet(
            self.dl_model_path,
            self.packet_size,
            self.extra_packet_dim,
            self.ar_order,
            self.ar_diagonal_load,
            self.context_length_blocks,
            self.context_length_blocks,
            self.nn_fade_dim,
        )

    def _prepare_to_play(self):
        self.extra_pred_buffer = np.zeros((self.extra_packet_dim, self.n_channels))
        self.is_burst = False
        return super()._prepare_to_play()

    def _a_priori(self, buffer, is_valid):
        if self.is_burst:
            buffer[: self.extra_packet_dim, :] *= self.model.fade_in[:, np.newaxis]
            buffer[: self.extra_packet_dim, :] += self.extra_pred_buffer
        return super()._a_priori(buffer, is_valid)

    def _predict(self, buffer: np.ndarray):
        """ """
        reconstructed_buffer = np.zeros(np.shape(buffer.T))
        for channel_idx in range(self.n_channels):
            prediction, extra_pred = self.model(
                self.context.T[:, channel_idx], self.is_burst
            )

            self.extra_pred_buffer[:, channel_idx] = extra_pred
            reconstructed_buffer[channel_idx, :] = prediction
        self.is_burst = True
        return reconstructed_buffer.T

    def _a_posteriori(self, buffer, is_valid):
        if is_valid:
            self.is_burst = False

        return super()._a_posteriori(buffer, is_valid)
