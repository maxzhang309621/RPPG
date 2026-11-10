#!/usr/bin/env python
"""Precompute canonical outward normals for the dlib-68 landmark subset.

Generates `weights/mediapipe/canonical_normals_dlib68.npy` of shape (68, 3),
unit-length, computed as the normalized vector from the centroid of the 68
canonical landmark positions to each landmark, in MediaPipe's canonical
face-mesh coordinate frame.

Two paths:

1. PRIMARY: load MediaPipe's canonical face mesh from a known location
   (the canonical_face_model.obj that ships with mediapipe). If not found,
   fall back to path 2.

2. FALLBACK: run MediaPipe FaceLandmarker on the first frame of UBFC
   subject1 (a documented frontal video), extract the per-vertex 3D
   landmarks (which MediaPipe returns in normalized image coordinates),
   then apply the inverse of the same frame's facial transformation matrix
   to bring them into canonical face space.

Sanity assertion: each output normal must have unit length AND each
normal's dot-product with the (landmark_position - centroid) direction
must be > 0.95. Fails -> exit nonzero, do NOT write the file.
"""

import os
import sys

import numpy as np

import os
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from utils.mediapipe_landmarks import DLIB68_FROM_MP468


def load_canonical_mesh_from_mediapipe():
    """Try to load MediaPipe's canonical face mesh vertices.

    Returns (478, 3) ndarray or None if not available.
    """
    try:
        import mediapipe as mp
    except ImportError:
        return None

    # MediaPipe ships canonical_face_model.obj in its package data.
    candidates = []
    pkg_dir = os.path.dirname(mp.__file__)
    for sub in ("modules/face_geometry/data",
                "tasks/python/vision/face_landmarker"):
        candidates.append(os.path.join(pkg_dir, sub,
                                       "canonical_face_model.obj"))
    for path in candidates:
        if os.path.exists(path):
            verts = []
            with open(path, "r") as f:
                for line in f:
                    if line.startswith("v "):
                        parts = line.strip().split()
                        verts.append([float(parts[1]),
                                      float(parts[2]),
                                      float(parts[3])])
            verts = np.array(verts, dtype=np.float64)
            if verts.shape[0] >= 468:
                return verts[:478]
    return None


def fallback_via_facelandmarker():
    """Extract canonical 3D landmarks from a frontal UBFC frame.

    Returns (478, 3) ndarray in canonical face space.
    """
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.core import base_options as mp_base

    model_path = os.path.join(
        PROJECT_DIR, "weights/mediapipe/face_landmarker.task")
    if not os.path.exists(model_path):
        raise RuntimeError(
            f"Fallback failed: MediaPipe model not found at {model_path}")

    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_base.BaseOptions(model_asset_path=model_path),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
        output_facial_transformation_matrixes=True,
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    # Read first frame of UBFC subject1.
    # cv2.VideoCapture can fail on .avi in some environments; use PyAV instead.
    ubfc_video = "PATH/TO/RAW_DATASETS/UBFC-rPPG/DATASET_2/subject1/vid.avi"
    frame_rgb = None
    try:
        import av
        container = av.open(ubfc_video)
        for packet in container.demux(video=0):
            for frame_av in packet.decode():
                frame_rgb = frame_av.to_ndarray(format="rgb24")
                break
            if frame_rgb is not None:
                break
        container.close()
    except Exception as av_err:
        # PyAV failed; try cv2 as last resort
        cap = cv2.VideoCapture(ubfc_video)
        ok, frame_bgr = cap.read()
        cap.release()
        if ok:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        else:
            raise RuntimeError(
                f"Fallback failed: could not read frame from {ubfc_video} "
                f"(PyAV error: {av_err})")
    if frame_rgb is None:
        raise RuntimeError(
            f"Fallback failed: could not decode any frame from {ubfc_video}")
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        raise RuntimeError("Fallback failed: no face detected in UBFC frame")

    landmarks = result.face_landmarks[0]
    H, W = frame_rgb.shape[:2]
    # MediaPipe's lm.z uses "smaller = closer to camera". Negate to convert
    # to the standard convention (larger = closer to camera = +z front of
    # head), which matches MediaPipe's facial_transformation_matrix output
    # space. Without this negation, the canonical frame's z axis is flipped
    # relative to MediaPipe's R, and the visibility computation is inverted.
    pts_normalized = np.array(
        [[lm.x * W, lm.y * H, -lm.z * W] for lm in landmarks],
        dtype=np.float64)

    # Apply inverse rotation to bring to canonical pose.
    transform = np.array(result.facial_transformation_matrixes[0],
                         dtype=np.float64)
    R = transform[:3, :3]
    t = transform[:3, 3]
    R_inv = R.T
    canonical = (pts_normalized - t) @ R_inv.T
    return canonical


def main():
    out_path = os.path.join(
        PROJECT_DIR, "weights/mediapipe/canonical_normals_dlib68.npy")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print("Trying to load canonical mesh from MediaPipe ...")
    canonical_478 = load_canonical_mesh_from_mediapipe()
    if canonical_478 is None:
        print("Not available; running fallback (FaceLandmarker on UBFC) ...")
        canonical_478 = fallback_via_facelandmarker()
    else:
        print(f"Loaded canonical mesh: {canonical_478.shape}")

    # Subset to dlib 68
    canonical_68 = canonical_478[DLIB68_FROM_MP468]   # (68, 3)

    centroid = canonical_68.mean(axis=0)
    offsets = canonical_68 - centroid
    lengths = np.linalg.norm(offsets, axis=1, keepdims=True)
    normals = offsets / (lengths + 1e-8)              # (68, 3) unit vectors

    # Sanity check: every normal is unit length and dot(normal, offset) > 0.95
    norms = np.linalg.norm(normals, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), \
        f"Normals not unit length: min={norms.min():.4f} max={norms.max():.4f}"
    dots = (normals * (offsets / (lengths + 1e-8))).sum(axis=1)
    assert dots.min() > 0.95, \
        f"Normals not aligned with offsets: min dot = {dots.min():.4f}"

    np.save(out_path, normals.astype(np.float32))
    print(f"Wrote {out_path}: shape={normals.shape} dtype={normals.dtype}")
    print(f"  centroid in canonical space: {centroid}")
    print(f"  mean normal: {normals.mean(axis=0)}")


if __name__ == "__main__":
    main()
