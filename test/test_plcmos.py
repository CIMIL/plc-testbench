from pathlib import Path

import pytest

from plctestbench.plcmos import PLCMOSEstimator


@pytest.mark.parametrize(
    ("model_version", "expected_files"),
    [
        ("0alpha", ["plcmos_v0.onnx"]),
        ("0", ["plcmos_v1_intrusive.onnx", "plcmos_v1_nonintrusive.onnx"]),
        ("2-val", ["plcmos_v2_val.onnx"]),
        ("2", ["plcmos_v2.onnx"]),
    ],
)
def test_plcmos_models_resolve_from_the_distribution(monkeypatch, model_version, expected_files):
    loaded_paths = []

    class FakeSession:
        def __init__(self, path):
            loaded_paths.append(Path(path))

    monkeypatch.setattr("plctestbench.plcmos.ort.InferenceSession", FakeSession)

    PLCMOSEstimator(model_version=model_version)

    assert [path.name for path in loaded_paths] == expected_files
    assert all(path.is_file() for path in loaded_paths)
