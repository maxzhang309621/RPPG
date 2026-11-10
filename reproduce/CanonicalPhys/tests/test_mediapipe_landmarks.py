# tests/test_mediapipe_landmarks.py
"""Unit tests for utils/mediapipe_landmarks.py helpers."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from utils.mediapipe_landmarks import (
    DLIB68_FROM_MP468,
    project_landmarks_to_crop,
    matrix_to_yaw_pitch_roll,
)


def test_index_map_size_and_range():
    assert len(DLIB68_FROM_MP468) == 68
    assert min(DLIB68_FROM_MP468) >= 0
    assert max(DLIB68_FROM_MP468) < 478
    # Indices must be unique (no double-mapping of mesh vertices)
    assert len(set(DLIB68_FROM_MP468)) == 68


def test_project_corner_to_corner():
    # box covers [10, 20] -> [10+100, 20+100] in raw frame; resize to 72.
    landmarks = np.array([[10.0, 20.0],     # top-left of box -> (0, 0)
                          [109.0, 119.0]],  # bottom-right -> (~71, ~71)
                         dtype=np.float32)
    out = project_landmarks_to_crop(landmarks, (10, 20, 100, 100), out_size=72)
    assert out.dtype == np.int16
    assert tuple(out[0]) == (0, 0)
    # The right-edge case projects to floor((99/100)*72)=71, after clip
    assert out[1, 0] == 71
    assert out[1, 1] == 71


def test_project_clips_out_of_box():
    # A landmark outside the box should clip to the edge, not error.
    landmarks = np.array([[-5.0, 1000.0]], dtype=np.float32)
    out = project_landmarks_to_crop(landmarks, (0, 0, 100, 100), out_size=72)
    assert out[0, 0] == 0
    assert out[0, 1] == 71


def test_yaw_pitch_roll_identity():
    R = np.eye(3)
    ypr = matrix_to_yaw_pitch_roll(R)
    np.testing.assert_allclose(ypr, [0.0, 0.0, 0.0], atol=1e-5)


def test_yaw_pitch_roll_pure_yaw_30():
    # 30-degree rotation about Y axis.
    a = np.deg2rad(30.0)
    Ry = np.array([[ np.cos(a), 0.0, np.sin(a)],
                   [ 0.0,       1.0, 0.0      ],
                   [-np.sin(a), 0.0, np.cos(a)]])
    yaw, pitch, roll = matrix_to_yaw_pitch_roll(Ry)
    assert abs(yaw - 30.0) < 1e-3
    assert abs(pitch) < 1e-3
    assert abs(roll) < 1e-3


def test_yaw_pitch_roll_pure_pitch_20():
    a = np.deg2rad(20.0)
    Rx = np.array([[1.0, 0.0,        0.0       ],
                   [0.0, np.cos(a), -np.sin(a)],
                   [0.0, np.sin(a),  np.cos(a)]])
    yaw, pitch, roll = matrix_to_yaw_pitch_roll(Rx)
    assert abs(yaw) < 1e-3
    assert abs(pitch - 20.0) < 1e-3
    assert abs(roll) < 1e-3


def test_yaw_pitch_roll_pure_roll_10():
    a = np.deg2rad(10.0)
    Rz = np.array([[np.cos(a), -np.sin(a), 0.0],
                   [np.sin(a),  np.cos(a), 0.0],
                   [0.0,        0.0,       1.0]])
    yaw, pitch, roll = matrix_to_yaw_pitch_roll(Rz)
    assert abs(yaw) < 1e-3
    assert abs(pitch) < 1e-3
    assert abs(roll - 10.0) < 1e-3


def test_yaw_pitch_roll_gimbal_lock_pitch_90():
    """Pure +90 deg pitch hits the gimbal-lock branch.

    At pitch = 90 deg, the standard Y-X-Z decomposition is degenerate
    (yaw and roll are coupled). The implementation forces roll = 0 and
    folds the rotation into yaw via the fallback formula. We assert that
    pitch comes out as 90 and that the function returns finite values.
    """
    # Rx(90 deg)
    Rx90 = np.array([[1.0, 0.0,  0.0],
                     [0.0, 0.0, -1.0],
                     [0.0, 1.0,  0.0]], dtype=np.float64)
    yaw, pitch, roll = matrix_to_yaw_pitch_roll(Rx90)
    assert abs(pitch - 90.0) < 1e-3, f"pitch={pitch}"
    assert abs(roll) < 1e-3, f"roll={roll} (gimbal lock should force 0)"
    # yaw should be finite (not NaN)
    assert np.isfinite(yaw), f"yaw={yaw}"


def test_yaw_pitch_roll_accepts_4x4():
    M = np.eye(4)
    ypr = matrix_to_yaw_pitch_roll(M)
    np.testing.assert_allclose(ypr, [0.0, 0.0, 0.0], atol=1e-5)


def _run_all():
    failed = []
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except Exception as e:
                print(f"FAIL  {name}: {e}")
                failed.append(name)
    if failed:
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    _run_all()
