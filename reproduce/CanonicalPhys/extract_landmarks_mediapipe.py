#!/usr/bin/env python
"""Extract MediaPipe FaceLandmarker landmarks + transformation matrices.

Reads raw videos via the same readers used by preprocess_lmdb.py, runs
MediaPipe FaceLandmarker on each raw frame, projects the 68-point dlib
subset back into the 72x72 LMDB coordinate frame using the same RetinaFace
crop box that preprocess_lmdb.py used, and writes per-subject sidecar
arrays:

    datasets/{DS}/landmarks/{sid}.npy        (T, 68, 2)  int16   72x72 coords
    datasets/{DS}/transforms/{sid}.npy       (T, 4, 4)   float32 raw MP matrix
    datasets/{DS}/valid_mask/{sid}.npy       (T,)        bool    detection ok

Usage:
    python extract_landmarks_mediapipe.py --data_config configs/datasets/ubfc.yaml

Notes:
    * The 72x72 LMDB face frames themselves are NOT regenerated.
    * Existing dlib outputs in datasets/{DS}/landmarks/ are overwritten.
      Move them to archive/dlib_landmarks/ before running this script
      (see Task 7).
"""

import argparse
import logging
import os
import sys

import numpy as np
from tqdm import tqdm

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from utils.config import load_config
from utils.mediapipe_landmarks import (
    DLIB68_FROM_MP468,
    project_landmarks_to_crop,
)

# Reuse the existing raw-data readers and face detector.
from preprocess_lmdb import (
    DATASET_READERS,
    face_detection_rf,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = (
    ""
    "weights/mediapipe/face_landmarker.task"
)
OUT_SIZE = 72  # must match preprocess_lmdb.py CROP_FACE.RESIZE


def build_landmarker(model_path):
    """Create a FaceLandmarker in IMAGE running mode with transformation
    matrices enabled.
    """
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=True,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


def recompute_crop_box(frames, config_preprocess):
    """Recompute the same RetinaFace crop box that preprocess_lmdb.py used.

    Mirrors the relevant control flow of preprocess_lmdb.crop_face_resize:
    static (first-frame) detection vs dynamic (per-N-frame) detection,
    optional median over detections, optional larger box.

    Returns: ndarray (4,) [x, y, w, h] int.
    """
    crop_cfg = config_preprocess.CROP_FACE
    use_larger = crop_cfg.USE_LARGE_FACE_BOX
    larger_coef = crop_cfg.LARGE_BOX_COEF
    do_dynamic = getattr(crop_cfg.DETECTION, 'DO_DYNAMIC_DETECTION', False)
    det_freq = getattr(crop_cfg.DETECTION, 'DYNAMIC_DETECTION_FREQUENCY', 32)
    use_median = getattr(crop_cfg.DETECTION, 'USE_MEDIAN_FACE_BOX', False)
    rf_path = crop_cfg.RETINAFACE_MODEL_PATH

    if do_dynamic:
        from math import ceil
        num_det = ceil(frames.shape[0] / det_freq)
    else:
        num_det = 1

    boxes = []
    for idx in range(num_det):
        frame_idx = det_freq * idx if do_dynamic else 0
        boxes.append(face_detection_rf(
            frames[frame_idx], rf_path, use_larger, larger_coef))
    boxes = np.array(boxes, dtype=int)

    if use_median:
        return np.median(boxes, axis=0).astype(int)
    return boxes[0]


def extract_for_frame(landmarker, frame_rgb):
    """Run FaceLandmarker on a single raw RGB frame.

    Returns:
        landmarks_xy: (478, 2) float32 in raw-frame pixel coords, or None.
        transform: (4, 4) float32, or None.
    """
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=np.ascontiguousarray(frame_rgb))
    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None, None

    H, W = frame_rgb.shape[:2]
    pts = result.face_landmarks[0]
    coords = np.array([[lm.x * W, lm.y * H] for lm in pts], dtype=np.float32)

    if (result.facial_transformation_matrixes
            and len(result.facial_transformation_matrixes) > 0):
        transform = np.asarray(
            result.facial_transformation_matrixes[0], dtype=np.float32)
    else:
        transform = np.eye(4, dtype=np.float32)

    return coords, transform


def process_subject(sid, frames, model_path, config_preprocess):
    """Run MediaPipe over all frames of one subject.

    Returns:
        landmarks: (T, 68, 2) int16 in 72x72 coords
        transforms: (T, 4, 4) float32
        valid: (T,) bool
    """
    nframes = frames.shape[0]
    landmarks_out = np.zeros((nframes, 68, 2), dtype=np.int16)
    transforms_out = np.tile(np.eye(4, dtype=np.float32), (nframes, 1, 1))
    valid_out = np.zeros(nframes, dtype=bool)

    box = recompute_crop_box(frames, config_preprocess)

    landmarker = build_landmarker(model_path)
    try:
        last_landmarks = None
        last_transform = None
        for t in range(nframes):
            coords_raw, transform = extract_for_frame(landmarker, frames[t])
            if coords_raw is None:
                # Carry forward last known good frame; mark invalid.
                if last_landmarks is not None:
                    landmarks_out[t] = last_landmarks
                    transforms_out[t] = last_transform
                continue

            # Pick 68-point dlib subset, project to 72x72.
            subset = coords_raw[DLIB68_FROM_MP468]
            projected = project_landmarks_to_crop(subset, box, OUT_SIZE)
            landmarks_out[t] = projected
            transforms_out[t] = transform
            valid_out[t] = True
            last_landmarks = projected
            last_transform = transform
    finally:
        landmarker.close()

    return landmarks_out, transforms_out, valid_out


def main():
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe FaceLandmarker landmarks/poses")
    parser.add_argument("--data_config", required=True,
                        help="Path to dataset config YAML")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH,
                        help="Path to face_landmarker.task")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most this many subjects (smoke test)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)])

    if not os.path.exists(args.model_path):
        logger.error(f"Model not found: {args.model_path}")
        sys.exit(1)

    config = load_config(args.data_config)
    data_cfg = config.TRAIN.DATA
    dataset_name = data_cfg.DATASET
    data_path = data_cfg.DATA_PATH
    config_preprocess = data_cfg.PREPROCESS

    lmdb_path = data_cfg.LMDB_PATH
    project_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(lmdb_path):
        lmdb_path = os.path.join(project_dir, lmdb_path)
    ds_dir = os.path.dirname(lmdb_path)

    landmarks_dir = os.path.join(ds_dir, "landmarks")
    transforms_dir = os.path.join(ds_dir, "transforms")
    valid_mask_dir = os.path.join(ds_dir, "valid_mask")
    for d in (landmarks_dir, transforms_dir, valid_mask_dir):
        os.makedirs(d, exist_ok=True)

    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Raw data: {data_path}")
    logger.info(f"LMDB: {lmdb_path}")
    logger.info(f"Outputs: {landmarks_dir} | {transforms_dir} | {valid_mask_dir}")
    logger.info(f"Model: {args.model_path}")

    reader_fn = DATASET_READERS[dataset_name]
    if dataset_name in ("OBF", "MMPD"):
        subjects_iter = reader_fn(data_path, target_fs=data_cfg.FS)
    else:
        subjects_iter = reader_fn(data_path)

    n_done = 0
    for sid, frames, _bvps in subjects_iter:
        out_lm = os.path.join(landmarks_dir, f"{sid}.npy")
        out_tf = os.path.join(transforms_dir, f"{sid}.npy")
        out_vm = os.path.join(valid_mask_dir, f"{sid}.npy")

        # Idempotent skip if all three sidecars exist with the right T.
        if (os.path.exists(out_lm)
                and os.path.exists(out_tf)
                and os.path.exists(out_vm)):
            existing = np.load(out_lm, mmap_mode="r")
            if existing.shape == (frames.shape[0], 68, 2):
                logger.info(f"  {sid}: already extracted, skipping")
                n_done += 1
                if args.limit is not None and n_done >= args.limit:
                    break
                continue

        logger.info(f"Processing {sid}: {frames.shape[0]} frames")
        landmarks, transforms, valid = process_subject(
            sid, frames, args.model_path, config_preprocess)
        np.save(out_lm, landmarks)
        np.save(out_tf, transforms)
        np.save(out_vm, valid)
        invalid_pct = 100.0 * (1.0 - valid.mean())
        logger.info(f"  {sid}: saved, invalid={invalid_pct:.1f}%")

        n_done += 1
        if args.limit is not None and n_done >= args.limit:
            logger.info(f"Reached --limit {args.limit}, stopping early.")
            break

    logger.info(f"Done. {n_done} subjects processed.")


if __name__ == "__main__":
    main()
