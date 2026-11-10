#!/usr/bin/env python
"""Build pose-stratified evaluation splits from head pose data.

Loads per-subject pose .npy files, chunks them using the model's chunk_length,
computes the dominant yaw per chunk, and bins into pose categories.

Usage:
    python build_pose_splits.py --data_config configs/datasets/ubfc.yaml \
        --model_config configs/models/factorizephys.yaml
"""

import argparse
import json
import logging
import os
import sys

import lmdb
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.config import load_config, deep_merge

logger = logging.getLogger(__name__)

# Pose bins (absolute yaw in degrees)
POSE_BINS = [
    ("frontal", 0, 15),
    ("mild", 15, 30),
    ("moderate", 30, 45),
    ("large", 45, 180),
]

# Special bin for chunks where MediaPipe could not detect a face on too
# many frames; the yaw value is unreliable so we exclude these from the
# pose-stratified buckets.
UNKNOWN_BIN = "unknown"


def bin_yaw(yaw_abs):
    """Assign absolute yaw angle to a pose bin."""
    for name, lo, hi in POSE_BINS:
        if lo <= yaw_abs < hi:
            return name
    return "large"


def main():
    parser = argparse.ArgumentParser(
        description="Build pose-stratified evaluation splits")
    parser.add_argument("--data_config", required=True,
                        help="Path to dataset config YAML")
    parser.add_argument("--model_config", required=True,
                        help="Path to model config YAML (for chunk_length)")
    parser.add_argument("--invalid_threshold", type=float, default=0.25,
                        help="Chunks with this fraction of invalid frames "
                             "or more are binned as 'unknown'.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)])

    data_config = load_config(args.data_config)
    model_config = load_config(args.model_config)
    config = deep_merge(data_config, model_config)

    if hasattr(config, "TRAIN"):
        data_cfg = config.TRAIN.DATA
    elif hasattr(config, "DATA"):
        data_cfg = config.DATA
    else:
        data_cfg = config

    dataset_name = data_cfg.DATASET
    lmdb_path = data_cfg.LMDB_PATH
    chunk_length = int(data_cfg.PREPROCESS.CHUNK_LENGTH)

    project_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(lmdb_path):
        lmdb_path = os.path.join(project_dir, lmdb_path)

    ds_dir = os.path.dirname(lmdb_path)
    poses_dir = os.path.join(ds_dir, "poses")
    valid_dir = os.path.join(ds_dir, "valid_mask")

    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Chunk length: {chunk_length}")
    logger.info(f"Poses dir: {poses_dir}")

    # Get subject list and frame counts from LMDB
    env = lmdb.open(lmdb_path, readonly=True, lock=False,
                    readahead=False, meminit=False)
    with env.begin() as txn:
        subjects = json.loads(txn.get(b"__subjects__"))
        nframes_meta = json.loads(txn.get(b"__nframes__"))
    env.close()

    # Build pose splits
    splits = {}
    distribution = {"frontal": 0, "mild": 0, "moderate": 0, "large": 0,
                    "unknown": 0}
    all_chunk_yaws = []
    missing = 0

    for sid in subjects:
        pose_path = os.path.join(poses_dir, f"{sid}.npy")
        if not os.path.exists(pose_path):
            logger.warning(f"Missing pose file: {pose_path}")
            missing += 1
            continue

        poses = np.load(pose_path)  # (T, 3) [yaw, pitch, roll]

        valid_path = os.path.join(valid_dir, f"{sid}.npy")
        if os.path.exists(valid_path):
            valid = np.load(valid_path)
        else:
            valid = np.ones(poses.shape[0], dtype=bool)

        nframes = nframes_meta[sid]
        n_chunks = nframes // chunk_length

        splits[sid] = {}
        for c in range(n_chunks):
            start = c * chunk_length
            end = start + chunk_length

            chunk_valid = valid[start:end]
            invalid_frac = float(1.0 - chunk_valid.mean())

            chunk_yaws = np.abs(poses[start:end, 0])  # absolute yaw
            yaw_median = float(np.median(chunk_yaws))
            yaw_max = float(np.max(chunk_yaws))
            yaw_std = float(np.std(chunk_yaws))

            if invalid_frac >= args.invalid_threshold:
                pose_bin = "unknown"
            else:
                pose_bin = bin_yaw(yaw_median)

            splits[sid][str(c)] = {
                "yaw_median": round(yaw_median, 2),
                "yaw_max": round(yaw_max, 2),
                "yaw_std": round(yaw_std, 2),
                "invalid_frac": round(invalid_frac, 3),
                "bin": pose_bin,
            }
            distribution[pose_bin] += 1
            if pose_bin != "unknown":
                all_chunk_yaws.append(yaw_median)

    total_chunks = sum(distribution.values())
    output = {
        "dataset": dataset_name,
        "chunk_length": chunk_length,
        "total_chunks": total_chunks,
        "distribution": distribution,
        "subjects": splits,
    }

    # Save JSON
    out_path = os.path.join(ds_dir, f"pose_splits_cl{chunk_length}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Saved pose splits to {out_path}")

    # Print summary
    logger.info(f"Dataset: {dataset_name}, Chunk length: {chunk_length}")
    logger.info(f"Total chunks: {total_chunks}, Missing subjects: {missing}")
    for bin_name, count in distribution.items():
        pct = 100 * count / total_chunks if total_chunks > 0 else 0
        logger.info(f"  {bin_name:>10s}: {count:6d} chunks ({pct:5.1f}%)")

    # Generate histogram
    if all_chunk_yaws:
        all_chunk_yaws = np.array(all_chunk_yaws)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Histogram of median yaw per chunk
        ax = axes[0]
        ax.hist(all_chunk_yaws, bins=50, edgecolor='black', alpha=0.7)
        for _, lo, hi in POSE_BINS:
            if lo > 0:
                ax.axvline(lo, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Median |Yaw| per chunk (degrees)')
        ax.set_ylabel('Number of chunks')
        ax.set_title(f'{dataset_name} — Yaw Distribution (CL={chunk_length})')

        # Bar chart of bins
        ax = axes[1]
        bins_names = [b[0] for b in POSE_BINS] + ["unknown"]
        bins_counts = [distribution[b] for b in bins_names]
        bars = ax.bar(bins_names, bins_counts, edgecolor='black', alpha=0.7)
        for bar, count in zip(bars, bins_counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom', fontsize=10)
        ax.set_xlabel('Pose Bin')
        ax.set_ylabel('Number of chunks')
        ax.set_title(f'{dataset_name} — Pose Bin Distribution')

        fig.tight_layout()
        fig_path = os.path.join(ds_dir, f"pose_distribution_cl{chunk_length}.png")
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        logger.info(f"Saved distribution plot to {fig_path}")


if __name__ == "__main__":
    main()
