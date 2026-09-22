"""Unit tests for the ONNX-backed deep-learning PLC algorithms.

``VermaPLC`` and ``PARCnetPLC`` are driven end-to-end on a deterministic
stereo mock track, asserting shape/dtype/finiteness and that repeated runs
are bit-identical (no hidden randomness).

Note: the golden-output parity tests against the original TensorFlow/Keras and
TorchScript models were removed together with their fixtures
(``test/fixtures/*.npz``). If those fixtures are recovered, parity checks can
be re-added on top of the run contracts below.
"""

from __future__ import annotations

import sys

import numpy as np

from plctestbench.plc_algorithm import PARCnetPLC, VermaPLC
from plctestbench.settings import PARCnetPLCSettings, VermaPLCSettings

from test._helpers import attach, seeded_sine

VERMA_PACKET_SIZE = 128
PARCNET_PACKET_SIZE = 256
DL_FS = 16000


def _verma_settings():
    return attach(VermaPLCSettings(), packet_size=VERMA_PACKET_SIZE, fs=DL_FS)


def _parcnet_settings():
    return attach(PARCnetPLCSettings(), packet_size=PARCNET_PACKET_SIZE, fs=DL_FS)


# --------------------------------------------------------------------------- #
# Seeded end-to-end run contracts
# --------------------------------------------------------------------------- #


def test_verma_plc_reconstructs_the_mock_track_shape():
    track = seeded_sine(DL_FS, frequency=440.0, channels=2, seed=0)
    lost_samples = np.array([128, 1024, 4096], dtype=np.int64)

    reconstructed = VermaPLC(_verma_settings()).run(track, lost_samples, "test")

    assert reconstructed.shape == track.shape
    assert reconstructed.dtype == np.float32
    assert np.isfinite(reconstructed).all()


def test_verma_plc_is_deterministic():
    track = seeded_sine(DL_FS, frequency=440.0, channels=2, seed=0)
    lost_samples = np.array([128, 1024, 4096], dtype=np.int64)

    first = VermaPLC(_verma_settings()).run(track, lost_samples, "test")
    second = VermaPLC(_verma_settings()).run(track, lost_samples, "test")

    np.testing.assert_array_equal(first, second)


def test_parcnet_plc_reconstructs_the_mock_track_shape():
    track = seeded_sine(DL_FS, frequency=440.0, channels=2, seed=0)
    lost_samples = np.array([256, 2048, 8192], dtype=np.int64)

    reconstructed = PARCnetPLC(_parcnet_settings()).run(track, lost_samples, "test")

    assert reconstructed.shape == track.shape
    assert reconstructed.dtype == np.float32
    assert np.isfinite(reconstructed).all()


def test_parcnet_plc_is_deterministic():
    track = seeded_sine(DL_FS, frequency=440.0, channels=2, seed=0)
    lost_samples = np.array([256, 2048, 8192], dtype=np.int64)

    first = PARCnetPLC(_parcnet_settings()).run(track, lost_samples, "test")
    second = PARCnetPLC(_parcnet_settings()).run(track, lost_samples, "test")

    np.testing.assert_array_equal(first, second)


def test_dl_algorithms_do_not_pull_in_torch_or_tensorflow():
    import plctestbench.parcnet  # noqa: F401
    import plctestbench.plc_algorithm  # noqa: F401

    assert "torch" not in sys.modules
    assert "tensorflow" not in sys.modules