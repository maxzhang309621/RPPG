#!/usr/bin/env python
"""Compute (yaw, pitch, roll) per frame from MediaPipe transformation matrices.

Reads datasets/{DS}/transforms/{sid}.npy and datasets/{DS}/valid_mask/{sid}.npy
written by extract_landmarks_mediapipe.py, decomposes each 4x4 matrix to
[yaw, pitch, roll] in degrees, carries forward the last valid pose on
invalid frames (or zero if no valid frame yet), and saves
    datasets/{DS}/poses/{sid}.npy   (T, 3)  float32

Usage:
    python compute_poses_mediapipe.py --data_config configs/datasets/ubfc.yaml
"""

import argparse
import json
import logging
import os
import sys

import lmdb
import numpy as np

from utils.config import load_config
from utils.mediapipe_landmarks import matrix_to_yaw_pitch_roll

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Compute head pose from MediaPipe transformation matrices")
    parser.add_argument("--data_config", required=True)
    parser.add_argument("--smooth_window", type=int, default=5,
                        help="Median smoothing window size (0 = off)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)])

    config = load_config(args.data_config)
    data_cfg = config.TRAIN.DATA
    dataset_name = data_cfg.DATASET
    lmdb_path = data_cfg.LMDB_PATH
    project_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(lmdb_path):
        lmdb_path = os.path.join(project_dir, lmdb_path)
    ds_dir = os.path.dirname(lmdb_path)

    transforms_dir = os.path.join(ds_dir, "transforms")
    valid_mask_dir = os.path.join(ds_dir, "valid_mask")
    poses_dir = os.path.join(ds_dir, "poses")
    os.makedirs(poses_dir, exist_ok=True)

    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    with env.begin() as txn:
        subjects = json.loads(txn.get(b"__subjects__"))
    env.close()

    logger.info(f"Dataset {dataset_name}: {len(subjects)} subjects")
    for sid in subjects:
        tf_path = os.path.join(transforms_dir, f"{sid}.npy")
        vm_path = os.path.join(valid_mask_dir, f"{sid}.npy")
        if not (os.path.exists(tf_path) and os.path.exists(vm_path)):
            logger.warning(f"  {sid}: missing transforms or valid_mask, skipping")
            continue

        transforms = np.load(tf_path)  # (T, 4, 4)
        valid = np.load(vm_path)        # (T,)
        T = transforms.shape[0]

        poses = np.zeros((T, 3), dtype=np.float32)
        last_pose = np.zeros(3, dtype=np.float32)
        for t in range(T):
            if valid[t]:
                last_pose = matrix_to_yaw_pitch_roll(transforms[t])
            poses[t] = last_pose

        if args.smooth_window and args.smooth_window > 1:
            poses = _median_smooth(poses, args.smooth_window)

        out_path = os.path.join(poses_dir, f"{sid}.npy")
        np.save(out_path, poses)

        yaw_med = float(np.median(np.abs(poses[valid, 0]))) if valid.any() else float("nan")
        logger.info(f"  {sid}: T={T}  median|yaw|={yaw_med:.1f}  "
                    f"valid={valid.mean():.3f}")

    logger.info("Done.")


def _median_smooth(poses, window):
    """Per-axis temporal median smoothing with reflective padding."""
    half = window // 2
    out = poses.copy()
    for axis in range(poses.shape[1]):
        col = np.pad(poses[:, axis], half, mode="edge")
        for i in range(poses.shape[0]):
            out[i, axis] = np.median(col[i:i + window])
    return out


if __name__ == "__main__":
    main()
