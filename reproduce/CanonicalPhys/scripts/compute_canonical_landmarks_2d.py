"""Compute canonical 2D positions for all 68 dlib-subset landmarks.

For each subject, apply the per-frame canonicalization homography to
the observed landmarks. In canonical space, landmarks under various
poses collapse near their canonical-frontal positions. Average across
frames to get stable canonical positions.

Uses UBFC training frames (frontal, clean). Saves to:
    weights/mediapipe/canonical_landmarks_2d.npy  (68, 2) float32

Usage:
    python scripts/compute_canonical_landmarks_2d.py
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.canonical_warp import (
    compute_homography_4pt,
    ANCHOR_INDICES,
    CANONICAL_ANCHORS,
)


def apply_homography_to_points(points, H):
    """Apply a 3x3 homography to (N, 2) points.

    Args:
        points: (N, 2) float
        H: (3, 3) float
    Returns:
        (N, 2) transformed points
    """
    hom = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    out = hom @ H.T
    return out[:, :2] / (out[:, 2:3] + 1e-8)


def main():
    ds_dir = 'datasets/UBFC_rPPG'
    lm_dir = os.path.join(ds_dir, 'landmarks')
    valid_dir = os.path.join(ds_dir, 'valid_mask')

    all_canonical = []

    subjects = sorted([f.replace('.npy', '')
                       for f in os.listdir(lm_dir) if f.endswith('.npy')])
    for sid in subjects:
        lm = np.load(os.path.join(lm_dir, f'{sid}.npy')).astype(
            np.float32) / 71.0  # (T, 68, 2) normalized
        valid = np.load(os.path.join(valid_dir, f'{sid}.npy'))  # (T,) bool

        for t in range(lm.shape[0]):
            if not valid[t]:
                continue
            src = lm[t, ANCHOR_INDICES, :]           # (4, 2)
            dst = CANONICAL_ANCHORS.numpy()           # (4, 2)
            # compute_homography_4pt works per-frame; we call with N=1
            H = compute_homography_4pt(
                torch.from_numpy(src).unsqueeze(0),
                torch.from_numpy(dst).unsqueeze(0)
            ).squeeze(0).numpy()                      # (3, 3)

            # Apply H to all 68 landmarks → their canonical positions
            canonical_lm = apply_homography_to_points(lm[t], H)  # (68, 2)
            all_canonical.append(canonical_lm)

    all_canonical = np.array(all_canonical)   # (N, 68, 2)
    mean_canonical = all_canonical.mean(axis=0)   # (68, 2)

    # Verify: anchor landmarks should be close to CANONICAL_ANCHORS
    for i, idx in enumerate(ANCHOR_INDICES):
        actual = mean_canonical[idx]
        expected = CANONICAL_ANCHORS.numpy()[i]
        diff = np.linalg.norm(actual - expected)
        print(f"  Anchor {idx}: actual {actual} vs expected {expected} "
              f"(diff={diff:.6f})")

    out_path = 'weights/mediapipe/canonical_landmarks_2d.npy'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, mean_canonical.astype(np.float32))

    print(f"\nSaved {mean_canonical.shape} canonical 2D landmarks to {out_path}")
    print(f"Based on {len(all_canonical)} valid UBFC training frames "
          f"from {len(subjects)} subjects")


if __name__ == '__main__':
    main()
