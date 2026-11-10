"""Precompute per-frame canonicalization homographies for all subjects.

For each subject in each dataset, loads the (T, 68, 2) landmarks,
extracts the 4 anchor points (dlib indices 36, 45, 48, 54), computes
the 3×3 homography that maps them to canonical positions, and saves
the result as datasets/{DS}/homographies/{sid}.npy of shape (T, 3, 3).

The degenerate-frame fallback logic matches canonical_warp.py exactly,
so training-time warp only needs grid_sample — no lstsq.

Usage:
    python scripts/precompute_homographies.py --dataset UBFC_rPPG
    python scripts/precompute_homographies.py --dataset PURE
    python scripts/precompute_homographies.py --dataset OBF
    python scripts/precompute_homographies.py --dataset MMPD
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.canonical_warp import (
    compute_homography_4pt,
    ANCHOR_INDICES,
    CANONICAL_ANCHORS,
)


DATASET_DIRS = {
    'UBFC_rPPG': 'datasets/UBFC_rPPG',
    'PURE': 'datasets/PURE',
    'OBF': 'datasets/OBF',
    'MMPD': 'datasets/MMPD',
}


def precompute_for_subject(landmarks_path, out_path):
    """Compute per-frame homographies for one subject.

    Args:
        landmarks_path: path to (T, 68, 2) int16 landmarks .npy
        out_path: path to write (T, 3, 3) float32 homographies .npy
    """
    lm_raw = np.load(landmarks_path)  # (T, 68, 2) int16
    T = lm_raw.shape[0]

    # Normalize to [0, 1] (same as lmdb_loader: divide by 71)
    lm = torch.from_numpy(lm_raw.astype(np.float32)) / 71.0  # (T, 68, 2)

    # Extract anchor landmarks
    idx = torch.tensor(ANCHOR_INDICES, dtype=torch.long)
    src = lm[:, idx, :]  # (T, 4, 2)
    dst = CANONICAL_ANCHORS.unsqueeze(0).expand(T, 4, 2)

    # Compute homographies in chunks to avoid GPU OOM on large subjects
    chunk_size = 4096
    H_all = []
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        H_chunk = compute_homography_4pt(
            src[start:end], dst[start:end])  # (chunk, 3, 3)
        H_all.append(H_chunk.detach().cpu())

    H = torch.cat(H_all, dim=0).numpy()  # (T, 3, 3) float32
    np.save(out_path, H)
    return T


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True,
                   choices=list(DATASET_DIRS.keys()))
    p.add_argument('--project_dir', default='.')
    args = p.parse_args()

    ds_dir = os.path.join(args.project_dir, DATASET_DIRS[args.dataset])
    lm_dir = os.path.join(ds_dir, 'landmarks')
    out_dir = os.path.join(ds_dir, 'homographies')
    os.makedirs(out_dir, exist_ok=True)

    subjects = sorted([
        f.replace('.npy', '') for f in os.listdir(lm_dir)
        if f.endswith('.npy')])

    print(f"Dataset: {args.dataset}")
    print(f"  Landmarks dir: {lm_dir} ({len(subjects)} subjects)")
    print(f"  Output dir:    {out_dir}")

    t0 = time.time()
    total_frames = 0
    for i, sid in enumerate(subjects):
        lm_path = os.path.join(lm_dir, f"{sid}.npy")
        h_path = os.path.join(out_dir, f"{sid}.npy")
        nframes = precompute_for_subject(lm_path, h_path)
        total_frames += nframes
        if (i + 1) % 50 == 0 or (i + 1) == len(subjects):
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(subjects)}] {sid}: {nframes} frames "
                  f"({elapsed:.1f}s elapsed, {total_frames} total frames)")

    elapsed = time.time() - t0
    print(f"\nDone: {len(subjects)} subjects, {total_frames} frames "
          f"in {elapsed:.1f}s ({total_frames / elapsed:.0f} frames/s)")


if __name__ == '__main__':
    main()
