import numpy as np
import onnxruntime as ort


class VermaNet:
    """
    ONNX Runtime wrapper around the Verma PLC model.

    The exported model has two float32 inputs:

        * the mel spectrogram, shaped ``(batch, num_mel_bins, frames, 1)``
        * the last packet, shaped ``(batch, packet_size)``

    and a single float32 output, shaped ``(batch, packet_size)``.

    Calling the object mirrors the previous Keras call convention::

        model((spectrograms, last_packet))
    """

    def __init__(self, model_path: str):
        self.model_path = str(model_path)
        self.session = ort.InferenceSession(
            self.model_path, providers=["CPUExecutionProvider"]
        )
        inputs = self.session.get_inputs()
        if len(inputs) != 2:
            raise ValueError(
                f"Verma model '{self.model_path}' must expose exactly 2 inputs, "
                f"found {len(inputs)}."
            )
        self.spectrogram_input_name = inputs[0].name
        self.packet_input_name = inputs[1].name
        self.spectrogram_shape = list(inputs[0].shape)
        self.packet_shape = list(inputs[1].shape)

    def _validate(self, spectrogram: np.ndarray, last_packet: np.ndarray) -> None:
        for axis, (found, expected) in enumerate(
            zip(spectrogram.shape, self.spectrogram_shape)
        ):
            if isinstance(expected, int) and found != expected:
                raise ValueError(
                    f"Verma mel spectrogram axis {axis} is {found} but the ONNX model "
                    f"expects {expected}. Check the mel settings (fs_dl, context_length, "
                    f"hop_size, window_length, num_mel_bins) against the exported model."
                )
        found = last_packet.shape[-1]
        expected = self.packet_shape[-1]
        if isinstance(expected, int) and found != expected:
            raise ValueError(
                f"Verma last-packet length is {found} but the ONNX model expects "
                f"{expected}. Set the packet loss simulator packet_size to the value "
                f"the model was exported with."
            )

    def __call__(self, inputs) -> np.ndarray:
        spectrograms, last_packet = inputs
        spectrogram = np.ascontiguousarray(spectrograms, dtype=np.float32)
        packet = np.ascontiguousarray(last_packet, dtype=np.float32)
        self._validate(spectrogram, packet)
        result = self.session.run(
            None,
            {
                self.spectrogram_input_name: spectrogram,
                self.packet_input_name: packet,
            },
        )[0]
        return np.asarray(result, dtype=np.float32)
