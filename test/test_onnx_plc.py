"""Basic mock/parity checks for the ONNX-backed deep-learning PLC models.

Run it directly (no pytest required)::

    python -m test.test_onnx_plc

The ``test_*`` functions are plain asserts and are also runnable with pytest.

The golden fixtures in ``test/fixtures`` were captured from the original
TensorFlow/Keras (Verma) and TorchScript (PARCnet) models before they were
removed, so these checks verify that the ONNX conversion is numerically
faithful.
"""

import sys
from pathlib import Path

import numpy as np

from plctestbench.plc_algorithm import PARCnetPLC, VermaPLC
from plctestbench.plcmos import PLCMOSEstimator
from plctestbench.settings import PARCnetPLCSettings, VermaPLCSettings
from plctestbench.utils import progress_monitor
from plctestbench.verma import VermaNet

FIXTURES = Path(__file__).resolve().parent / "fixtures"

# Verma converts exactly (max observed |diff| ~ 7e-7); PARCnet has 4 Resize
# ops whose nearest/bilinear rounding differs slightly between PyTorch and
# ONNX Runtime (max observed |diff| ~ 2e-4 on the consumed 512-sample slice).
VERMA_ATOL = 1e-5
VERMA_RTOL = 1e-4
PARCNET_ATOL = 5e-4
PARCNET_RTOL = 1e-3

VERMA_PACKET_SIZE = 128
PARCNET_PACKET_SIZE = 256


def _synthetic_track(fs: int = 16000, seconds: float = 1.0, channels: int = 2):
    """Deterministic stereo sine + low-level noise, shaped (samples, channels)."""
    rng = np.random.RandomState(0)
    samples = int(fs * seconds)
    time = np.arange(samples) / fs
    mono = 0.5 * np.sin(2 * np.pi * 440.0 * time) + 1e-3 * rng.standard_normal(samples)
    return np.repeat(mono.astype(np.float32)[:, None], channels, axis=1)


def _verma_settings():
    settings = VermaPLCSettings()
    settings.set_progress_monitor(progress_monitor)
    settings.add("packet_size", VERMA_PACKET_SIZE)
    settings.add("fs", 16000)
    return settings


def _parcnet_settings():
    settings = PARCnetPLCSettings()
    settings.set_progress_monitor(progress_monitor)
    settings.add("packet_size", PARCNET_PACKET_SIZE)
    settings.add("fs", 16000)
    return settings


def test_verma_onnx_matches_golden():
    data = np.load(FIXTURES / "verma_golden.npz")
    model = VermaNet(_verma_settings().get("model_path"))
    output = model((data["spectrogram"], data["last_packet"]))
    assert output.shape == data["output"].shape, (output.shape, data["output"].shape)
    np.testing.assert_allclose(output, data["output"], atol=VERMA_ATOL, rtol=VERMA_RTOL)


def test_parcnet_onnx_matches_golden():
    data = np.load(FIXTURES / "parcnet_golden.npz")
    model = PARCnetPLC(_parcnet_settings()).model
    raw = model.session.run(None, {model.nn_input_name: data["nn_context"]})[0]
    output = np.squeeze(np.asarray(raw)[..., -model.pred_dim :])
    assert output.shape == data["output"].shape, (output.shape, data["output"].shape)
    np.testing.assert_allclose(output, data["output"], atol=PARCNET_ATOL, rtol=PARCNET_RTOL)


def test_verma_plc_smoke():
    track = _synthetic_track()
    lost_samples_idx = np.array([128, 1024, 4096], dtype=np.int64)
    plc = VermaPLC(_verma_settings())
    reconstructed = plc.run(track, lost_samples_idx, "smoke")
    assert reconstructed.shape == track.shape, (reconstructed.shape, track.shape)
    assert reconstructed.dtype == np.float32, reconstructed.dtype
    assert np.isfinite(reconstructed).all()


def test_parcnet_plc_smoke():
    track = _synthetic_track()
    lost_samples_idx = np.array([256, 2048, 8192], dtype=np.int64)
    plc = PARCnetPLC(_parcnet_settings())
    reconstructed = plc.run(track, lost_samples_idx, "smoke")
    assert reconstructed.shape == track.shape, (reconstructed.shape, track.shape)
    assert reconstructed.dtype == np.float32, reconstructed.dtype
    assert np.isfinite(reconstructed).all()


def test_plcmos_still_runs():
    estimator = PLCMOSEstimator()
    mos = estimator.run(_synthetic_track(fs=16000, channels=1)[:, 0], 16000)
    assert isinstance(mos, float)
    assert np.isfinite(mos)


def test_no_torch_or_tensorflow_imported():
    # Importing the PLC modules must not pull in the removed frameworks.
    import plctestbench.parcnet  # noqa: F401
    import plctestbench.plc_algorithm  # noqa: F401

    assert "torch" not in sys.modules, "torch was imported at runtime"
    assert "tensorflow" not in sys.modules, "tensorflow was imported at runtime"


def main() -> int:
    checks = [
        test_verma_onnx_matches_golden,
        test_parcnet_onnx_matches_golden,
        test_verma_plc_smoke,
        test_parcnet_plc_smoke,
        test_plcmos_still_runs,
        test_no_torch_or_tensorflow_imported,
    ]
    failures = 0
    for check in checks:
        try:
            check()
        except Exception as error:  # noqa: BLE001 - report every failure
            failures += 1
            print(f"FAIL {check.__name__}: {type(error).__name__}: {error}")
        else:
            print(f"PASS {check.__name__}")
    print(f"{len(checks) - failures}/{len(checks)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
