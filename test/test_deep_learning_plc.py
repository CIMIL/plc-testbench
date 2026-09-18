"""Unit tests for the ONNX-backed deep-learning PLC algorithms.

Two layers of checking:

* **Numerical parity** - ``VermaNet`` and ``PARCnet`` are compared against
  golden outputs in ``test/fixtures`` that were captured from the original
  TensorFlow/Keras and TorchScript models before they were removed. This is
  what proves the ONNX conversion is faithful.
* **Seeded run contracts** - both PLC algorithms are driven end-to-end on a
  deterministic stereo mock track, asserting shape/dtype/finiteness and that
  repeated runs are bit-identical (no hidden randomness).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from plctestbench.plc_algorithm import PARCnetPLC, VermaPLC
from plctestbench.settings import PARCnetPLCSettings, VermaPLCSettings
from plctestbench.verma import VermaNet

from test._helpers import attach, seeded_sine

FIXTURES = Path(__file__).resolve().parent / "fixtures"

# Verma converts exactly (max observed |diff| ~ 7e-7); PARCnet has 4 Resize ops
# whose nearest/bilinear rounding differs slightly between PyTorch and ONNX
# Runtime (max observed |diff| ~ 2e-4 on the consumed 512-sample slice).
VERMA_ATOL, VERMA_RTOL = 1e-5, 1e-4
PARCNET_ATOL, PARCNET_RTOL = 5e-4, 1e-3

VERMA_PACKET_SIZE = 128
PARCNET_PACKET_SIZE = 256
DL_FS = 16000


def _verma_settings():
    return attach(
        VermaPLCSettings(), packet_size=VERMA_PACKET_SIZE, fs=DL_FS
    )


def _parcnet_settings():
    return attach(
        PARCnetPLCSettings(), packet_size=PARCNET_PACKET_SIZE, fs=DL_FS
    )


def _require(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"model fixture not available: {path}")


# --------------------------------------------------------------------------- #
# Numerical parity against the pre-conversion golden outputs
# --------------------------------------------------------------------------- #


def test_verma_net_matches_keras_golden():
    fixture = FIXTURES / "verma_golden.npz"
    _require(fixture)
    data = np.load(fixture)

    model = VermaNet(_verma_settings().get("model_path"))
    output = model((data["spectrogram"], data["last_packet"]))

    assert output.shape == data["output"].shape
    np.testing.assert_allclose(
        output, data["output"], atol=VERMA_ATOL, rtol=VERMA_RTOL
    )


def test_parcnet_model_matches_torchscript_golden():
    fixture = FIXTURES / "parcnet_golden.npz"
    _require(fixture)
    data = np.load(fixture)

    model = PARCnetPLC(_parcnet_settings()).model
    raw = model.session.run(None, {model.nn_input_name: data["nn_context"]})[0]
    output = np.squeeze(np.asarray(raw)[..., -model.pred_dim :])

    assert output.shape == data["output"].shape
    np.testing.assert_allclose(
        output, data["output"], atol=PARCNET_ATOL, rtol=PARCNET_RTOL
    )


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