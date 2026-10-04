import io

import numpy as np
import pytest

from vgrep3d.field.colmap_io import Camera, get_intrinsics, qvec_to_rotmat, read_next_bytes


def test_identity_quaternion_has_identity_rotation() -> None:
    np.testing.assert_allclose(qvec_to_rotmat(np.array([1.0, 0.0, 0.0, 0.0])), np.eye(3))


def test_simple_pinhole_intrinsics_share_focal_length() -> None:
    camera = Camera(1, "SIMPLE_PINHOLE", 640, 480, np.array([500.0, 320.0, 240.0]))
    np.testing.assert_allclose(
        get_intrinsics(camera),
        np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]),
    )


def test_binary_reader_reports_truncated_input() -> None:
    with pytest.raises(EOFError, match="Expected 8 bytes"):
        read_next_bytes(io.BytesIO(b"short"), 8, "Q")
