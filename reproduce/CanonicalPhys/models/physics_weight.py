"""Physics-informed per-pixel weighting for canonical-space rPPG.

Motivation
----------
Once frames are in canonical face coordinates, every pixel corresponds
to a fixed anatomical location. The MediaPipe 4x4 facial transform
matrix provides the 3x3 rotation R between the face model frame and
the camera frame. Combined with precomputed canonical surface normals,
this lets us compute, *per pixel*, the angle between the skin surface
normal and the camera viewing direction — the key quantity in the
dichromatic reflection model:

    - Diffuse reflectance (carries the BVP signal) is proportional to
      cos(theta_normal, light) in Lambertian skin.
    - Specular reflectance (rPPG noise) peaks at grazing viewing
      angles.

This module produces a per-pixel diffuse weighting map that:
    - Upweights pixels whose surface normal faces the camera (central
      forehead / cheeks, strong BVP signal, weak specular).
    - Downweights grazing-angle pixels (weak signal, more noise).
    - Implicitly suppresses off-face pixels (their assigned landmark's
      normal is unreliable, so the sigmoid drops).

Unlike v2's region-level visibility mask at 13x13, this is a 72x72
per-pixel map derived from *canonical surface geometry*. Unlike
SkinMap, there are no learnable parameters — the weighting is fully
determined by physics and geometry.

Implementation
--------------
At build time (once), we assign each canonical pixel to its nearest
landmark and store the mapping. At runtime per frame:
    1. Fetch the canonical 3D normal for each pixel's assigned landmark.
    2. Rotate with per-frame R from the transform.
    3. Take the z-component (camera-axis alignment) and pass through
       sigmoid(SHARPNESS * z + BIAS) -> (B, T, H, W) weight.
The weight is applied multiplicatively to the canonical frame.
"""

import os

import numpy as np
import torch
import torch.nn as nn


def build_pixel_to_landmark_assignment(canonical_landmarks_2d, spatial_size=72):
    """Assign each pixel in a spatial_size x spatial_size canonical image
    to its nearest canonical landmark.

    Args:
        canonical_landmarks_2d: (K, 2) float32 in [0, 1] — canonical
            2D landmark positions.
        spatial_size: output image side length in pixels.

    Returns:
        assignment: (H, W) int64, values in [0, K) — nearest landmark
            index per pixel.
    """
    K = canonical_landmarks_2d.shape[0]
    # Build a (H, W, 2) grid of pixel centers in [0, 1]
    xs = np.linspace(0.0, 1.0, spatial_size, dtype=np.float32)
    ys = np.linspace(0.0, 1.0, spatial_size, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing='ij')   # (H, W) each
    pixels = np.stack([grid_x, grid_y], axis=-1)           # (H, W, 2)

    # Squared distance from each pixel to each landmark
    # pixels: (H, W, 1, 2), landmarks: (1, 1, K, 2)
    diff = pixels[:, :, None, :] - canonical_landmarks_2d[None, None, :, :]
    dist_sq = (diff ** 2).sum(axis=-1)                     # (H, W, K)
    assignment = np.argmin(dist_sq, axis=-1).astype(np.int64)  # (H, W)
    return assignment


class PhysicsWeight(nn.Module):
    """Per-pixel diffuse weighting from canonical surface geometry.

    Forward signature:
        weight = module(transforms)
        transforms: (B, T, 4, 4) MediaPipe facial transformation matrices.
        weight:     (B, T, H, W) in (0, 1), per-pixel diffuse weight.

    Hyperparameters
    ---------------
    sharpness: sigmoid slope on cos(theta). Higher = sharper
        visible/occluded transition.
    bias: added before sigmoid. Shifts the midpoint; positive bias means
        pixels slightly off-camera still get >0.5 weight.

    Notes
    -----
    The canonical normals (weights/mediapipe/canonical_normals_dlib68.npy)
    and canonical 2D landmark positions
    (weights/mediapipe/canonical_landmarks_2d.npy) are both fixed buffers.
    No learnable parameters.
    """

    def __init__(self,
                 canonical_normals_path='weights/mediapipe/canonical_normals_dlib68.npy',
                 canonical_landmarks_2d_path='weights/mediapipe/canonical_landmarks_2d.npy',
                 spatial_size=72,
                 sharpness=8.0,
                 bias=0.55):  # matches v2 VisibilityMaskAttn default
        super().__init__()
        self.spatial_size = int(spatial_size)
        self.sharpness = float(sharpness)
        self.bias = float(bias)

        n_canon = np.load(canonical_normals_path).astype(np.float32)  # (68, 3)
        lm_canon_2d = np.load(canonical_landmarks_2d_path).astype(
            np.float32)                                               # (68, 2)
        assert n_canon.shape[0] == lm_canon_2d.shape[0], \
            f"normals K={n_canon.shape[0]} vs landmarks K={lm_canon_2d.shape[0]}"

        assignment = build_pixel_to_landmark_assignment(
            lm_canon_2d, spatial_size=self.spatial_size)              # (H, W)

        # (H, W, 3) canonical normal at each pixel, via the assignment
        pixel_normals = n_canon[assignment]                           # (H, W, 3)

        self.register_buffer('pixel_normals_canonical',
                             torch.from_numpy(pixel_normals))         # (H, W, 3)

    def forward(self, transforms):
        """
        Args:
            transforms: (B, T, 4, 4) facial transformation matrices.

        Returns:
            weight: (B, T, H, W) per-pixel diffuse weight in (0, 1).
        """
        B, T = transforms.shape[:2]
        H = W = self.spatial_size

        # Match dtype to the normals buffer so einsum doesn't complain
        R = transforms[..., :3, :3].to(self.pixel_normals_canonical.dtype)

        # pixel_normals_canonical: (H, W, 3)
        # We want: camera_normal[b, t, h, w] = R[b, t] @ pixel_normal[h, w]
        # Reshape for batched matmul:
        #   n_c: (H*W, 3) -> (1, 1, H*W, 3)
        #   R:   (B, T, 3, 3) -> (B, T, 3, 3)
        # Result: (B, T, H*W, 3)
        n_c = self.pixel_normals_canonical.view(H * W, 3)            # (HW, 3)
        # Use einsum for clarity: bt[i,j] * hj -> bthi; here R is 3x3, n is 3
        cam_normals = torch.einsum('btij,hj->bthi', R, n_c)          # (B,T,HW,3)

        # z-component of camera-space normal (axis toward the camera)
        cos_theta = cam_normals[..., 2]                              # (B, T, H*W)
        cos_theta = cos_theta.view(B, T, H, W)

        weight = torch.sigmoid(self.sharpness * (cos_theta + self.bias))
        return weight
