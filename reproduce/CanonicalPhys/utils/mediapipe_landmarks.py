# utils/mediapipe_landmarks.py
"""Pure helpers for the MediaPipe-based landmark/pose pipeline.

Contents:
    DLIB68_FROM_MP468  -- index list mapping dlib's 68 anatomical points
                          to indices in MediaPipe's 478-point face mesh.
    project_landmarks_to_crop  -- map raw-frame (x, y) to 72x72 LMDB frame.
    matrix_to_yaw_pitch_roll   -- decompose a 3x3 rotation matrix to degrees.

These helpers have NO MediaPipe runtime dependency so they can be unit tested
without loading the .task model.
"""

import numpy as np


# All "left/right" labels refer to the subject's anatomical side. The visual
# sanity check in Task 5 (extraction smoke test) is the authoritative
# verification that these indices land on the correct features.
# Mapping from dlib's 68-point indices to MediaPipe Face Mesh (478 vertices).
# Anatomical correspondences are approximate; the visual sanity check in
# Task 5 must confirm the chosen points actually land on the right features.
DLIB68_FROM_MP468 = [
    # Jaw line, dlib idx 0..16 (17 points)
    127, 234, 93, 132, 58, 172, 136, 150, 152, 149, 176, 148, 377, 400, 378, 379, 365,
    # Subject-right eyebrow (image left), dlib idx 17..21 (5 points)
    70, 63, 105, 66, 107,
    # Subject-left eyebrow (image right), dlib idx 22..26 (5 points)
    336, 296, 334, 293, 300,
    # Nose bridge, dlib idx 27..30 (4 points)
    168, 6, 197, 195,
    # Nose bottom, dlib idx 31..35 (5 points)
    98, 97, 2, 326, 327,
    # Subject-right eye (image left), dlib idx 36..41 (6 points)
    33, 160, 158, 133, 153, 144,
    # Subject-left eye (image right), dlib idx 42..47 (6 points)
    362, 385, 387, 263, 373, 380,
    # Outer mouth, dlib idx 48..59 (12 points)
    61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181,
    # Inner mouth, dlib idx 60..67 (8 points)
    78, 81, 13, 311, 308, 402, 14, 178,
]
assert len(DLIB68_FROM_MP468) == 68, \
    f"DLIB68_FROM_MP468 has {len(DLIB68_FROM_MP468)} indices, expected 68"
assert max(DLIB68_FROM_MP468) < 478
assert min(DLIB68_FROM_MP468) >= 0


def project_landmarks_to_crop(landmarks_xy, box_xywh, out_size):
    """Map (x, y) landmarks from raw-frame coords to crop+resized coords.

    Args:
        landmarks_xy: (K, 2) ndarray of (x, y) in raw-frame pixels.
        box_xywh: (4,) ndarray-like (x, y, w, h) RetinaFace crop box in raw frame.
        out_size: int side length of the resized crop (e.g. 72).

    Returns:
        (K, 2) ndarray of int16 coordinates clipped to [0, out_size-1].
    """
    landmarks_xy = np.asarray(landmarks_xy, dtype=np.float32)
    bx, by, bw, bh = box_xywh
    if bw <= 0 or bh <= 0:
        raise ValueError(f"invalid crop box {box_xywh}")
    sx = out_size / bw
    sy = out_size / bh
    out = np.empty_like(landmarks_xy)
    out[:, 0] = (landmarks_xy[:, 0] - bx) * sx
    out[:, 1] = (landmarks_xy[:, 1] - by) * sy
    np.clip(out, 0, out_size - 1, out=out)
    return out.astype(np.int16)


def matrix_to_yaw_pitch_roll(matrix):
    """Decompose a 4x4 (or 3x3) rotation matrix to (yaw, pitch, roll) in degrees.

    Convention: yaw is rotation about the Y (vertical) axis (head turn left/right),
    pitch about X (head nod), roll about Z (head tilt). Matches the column order
    that build_pose_splits.py expects (column 0 = yaw).

    Decomposition assumes intrinsic Tait-Bryan Y-X-Z ordering, which matches
    MediaPipe's facial transformation matrix convention (Y-up, Z-out-of-screen).

    Args:
        matrix: (3, 3) or (4, 4) ndarray.

    Returns:
        (yaw_deg, pitch_deg, roll_deg) numpy float32 array of length 3.
    """
    R = np.asarray(matrix, dtype=np.float64)
    if R.shape == (4, 4):
        R = R[:3, :3]
    if R.shape != (3, 3):
        raise ValueError(f"matrix must be 3x3 or 4x4, got {R.shape}")

    # Y-X-Z intrinsic Tait-Bryan
    #   R = Ry(yaw) @ Rx(pitch) @ Rz(roll)
    # Standard derivation:
    #   pitch = arcsin(-R[1, 2])
    #   yaw   = arctan2(R[0, 2], R[2, 2])
    #   roll  = arctan2(R[1, 0], R[1, 1])
    sin_pitch = -R[1, 2]
    sin_pitch = np.clip(sin_pitch, -1.0, 1.0)
    pitch = np.arcsin(sin_pitch)

    cos_pitch = np.cos(pitch)
    if abs(cos_pitch) > 1e-6:
        yaw = np.arctan2(R[0, 2], R[2, 2])
        roll = np.arctan2(R[1, 0], R[1, 1])
    else:
        # Gimbal lock: pitch ~= +/- 90 deg, roll arbitrary, set to 0.
        yaw = np.arctan2(-R[2, 0], R[0, 0])
        roll = 0.0

    return np.array([np.degrees(yaw), np.degrees(pitch), np.degrees(roll)],
                    dtype=np.float32)
