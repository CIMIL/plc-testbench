"""Unit tests for the mid/side spatial codec."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from plctestbench.spatial import CodecMode, MidSideCodec

MOCK_STEREO = np.array([[0.4, 0.2], [0.1, -0.3]], dtype=np.float64)


def test_encode_matches_analytical_expected_output():
    encoded = MidSideCodec.encode(MOCK_STEREO)
    # mid = (L + R) / 2 ; side = (L - R) / 2
    np.testing.assert_allclose(
        encoded, np.array([[0.3, 0.1], [-0.1, 0.2]]), atol=1e-12
    )


def test_decode_matches_analytical_expected_output():
    mid_side = np.array([[0.3, 0.1], [-0.1, 0.2]], dtype=np.float64)
    decoded = MidSideCodec.decode(mid_side)
    # left = mid + side ; right = mid - side
    np.testing.assert_allclose(decoded, MOCK_STEREO, atol=1e-12)


def test_call_dispatches_on_the_codec_mode():
    codec = MidSideCodec()
    np.testing.assert_allclose(
        codec(MOCK_STEREO, CodecMode.ENCODE), MidSideCodec.encode(MOCK_STEREO)
    )
    mid_side = MidSideCodec.encode(MOCK_STEREO)
    np.testing.assert_allclose(
        codec(mid_side, CodecMode.DECODE), MidSideCodec.decode(mid_side)
    )


def test_call_defaults_to_encode():
    codec = MidSideCodec()
    np.testing.assert_allclose(codec(MOCK_STEREO), MidSideCodec.encode(MOCK_STEREO))


def test_unsupported_codec_mode_raises():
    codec = MidSideCodec()
    with pytest.raises(ValueError):
        codec(MOCK_STEREO, cast(CodecMode, "not-a-mode"))


def test_round_trip_is_identity():
    """``decode`` is the exact inverse of ``encode``: ``decode(encode(x)) == x``."""
    codec = MidSideCodec()
    restored = codec(codec(MOCK_STEREO, CodecMode.ENCODE), CodecMode.DECODE)
    np.testing.assert_allclose(restored, MOCK_STEREO, atol=1e-12)


def test_encode_preserves_shape_and_dtype():
    encoded = MidSideCodec.encode(MOCK_STEREO)
    assert encoded.shape == MOCK_STEREO.shape
    assert encoded.dtype == MOCK_STEREO.dtype