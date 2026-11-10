"""Differentiable face canonicalization via 4-point homography.

Takes raw video frames + facial landmarks and warps each frame so that a
fixed set of anatomical landmarks (outer eye corners, mouth corners)
lands at fixed canonical image positions. The backbone sees a
pose-invariant representation regardless of actual head orientation.

Key design choices:
    - 4-point 2D homography (not 3D rotation). We don't need 3D
      landmarks; the 2D homography corrects the dominant rigid in-plane
      / quasi-planar component of head pose and is exact for small
      out-of-plane rotations.
    - Closed-form DLT via torch.linalg.solve. Fully differentiable wrt
      landmarks and frames (gradients flow through the warp), though in
      practice only frame gradients matter for training.
    - Invalid frames (face detection failure) pass through unchanged.

No learnable parameters in this module. The warp is fully deterministic
given the landmarks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# dlib-68 indices for the anchor landmarks
LEFT_EYE_OUTER = 36
RIGHT_EYE_OUTER = 45
LEFT_MOUTH_CORNER = 48
RIGHT_MOUTH_CORNER = 54

ANCHOR_INDICES = [LEFT_EYE_OUTER, RIGHT_EYE_OUTER,
                  LEFT_MOUTH_CORNER, RIGHT_MOUTH_CORNER]

# Canonical positions in [0, 1] image coordinates. Hand-designed to
# approximate a frontal face in a 72x72 crop. Symmetric left/right,
# eyes above mouth, forms a non-degenerate quadrilateral.
CANONICAL_ANCHORS = torch.tensor([
    [0.30, 0.40],  # left eye outer
    [0.70, 0.40],  # right eye outer
    [0.38, 0.72],  # left mouth corner
    [0.62, 0.72],  # right mouth corner
], dtype=torch.float32)


def compute_homography_4pt(src, dst):
    """Compute per-sample 3x3 homography from 4-point correspondences.

    Uses Direct Linear Transform (DLT) with normalization-free 8-equation
    linear system (fixes h33=1). Robust to degenerate inputs: frames
    where the source points are coincident, collinear, or otherwise
    produce a rank-deficient system fall back to the identity
    homography.

    Args:
        src: (N, 4, 2) source points.
        dst: (N, 4, 2) target points.

    Returns:
        H: (N, 3, 3) homography that maps src -> dst.
    """
    assert src.shape[-1] == 2 and dst.shape[-1] == 2
    assert src.shape[-2] == 4 and dst.shape[-2] == 4
    N = src.shape[0]
    device = src.device
    dtype = src.dtype

    x = src[..., 0]  # (N, 4)
    y = src[..., 1]
    u = dst[..., 0]
    v = dst[..., 1]

    zeros = torch.zeros_like(x)
    ones = torch.ones_like(x)

    row_u = torch.stack([x, y, ones, zeros, zeros, zeros,
                         -u * x, -u * y], dim=-1)      # (N, 4, 8)
    row_v = torch.stack([zeros, zeros, zeros, x, y, ones,
                         -v * x, -v * y], dim=-1)      # (N, 4, 8)

    # Interleave: (N, 8, 8)
    A = torch.stack([row_u, row_v], dim=-2).reshape(N, 8, 8)
    b = torch.stack([u, v], dim=-1).reshape(N, 8)

    # Solve Ah = b with lstsq. lstsq returns the least-squares solution
    # and is batched, so rank-deficient frames get a (junk) solution
    # rather than raising. We filter those out afterwards.
    sol = torch.linalg.lstsq(A, b.unsqueeze(-1)).solution  # (N, 8, 1)
    h = sol.squeeze(-1)                                    # (N, 8)

    # Any NaN/Inf in h indicates a failed solve; fall back to identity.
    identity_h = torch.tensor(
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        device=device, dtype=dtype)
    bad_solve = ~torch.isfinite(h).all(dim=-1, keepdim=True)  # (N, 1)

    # Also detect degenerate src upfront (all 4 points coincident or
    # collinear). Use variance of src points across the 4 corners as a
    # simple detector.
    src_var = src.var(dim=-2, unbiased=False).sum(dim=-1, keepdim=True)  # (N, 1)
    degenerate = src_var < 1e-6

    use_identity = bad_solve | degenerate
    h = torch.where(use_identity, identity_h.expand_as(h), h)

    # Assemble 3x3 homography (h33 = 1)
    H = torch.cat(
        [h, torch.ones(N, 1, device=device, dtype=dtype)],
        dim=-1).reshape(N, 3, 3)
    return H


def warp_by_homography(frames, H, out_size=None, padding_mode='zeros'):
    """Apply a batched 3x3 homography to a stack of frames.

    Args:
        frames: (N, C, H, W) frame tensor. N = B*T is typical.
        H: (N, 3, 3) homography mapping source (frames) coordinates to
           canonical output coordinates. We invert internally to get the
           canonical->source sampling grid expected by grid_sample.
        out_size: (Hout, Wout) output size. Defaults to the input size.
        padding_mode: passed to F.grid_sample ('zeros' | 'border' |
                       'reflection').

    Returns:
        warped: (N, C, Hout, Wout) warped frames.
    """
    N, C, H_in, W_in = frames.shape
    if out_size is None:
        Hout, Wout = H_in, W_in
    else:
        Hout, Wout = out_size

    device = frames.device
    dtype = frames.dtype

    # Build a (Hout, Wout) grid of canonical coords in [0, 1] space.
    ys = torch.linspace(0.0, 1.0, Hout, device=device, dtype=dtype)
    xs = torch.linspace(0.0, 1.0, Wout, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    # (Hout, Wout, 3) homogeneous canonical coords
    canon = torch.stack(
        [grid_x, grid_y, torch.ones_like(grid_x)], dim=-1)
    canon = canon.view(1, Hout * Wout, 3).expand(N, -1, -1)  # (N, HW, 3)

    # To sample, we need the inverse homography: canonical -> source.
    # Guard against singular H (happens when observed landmarks are
    # collinear / coincident, e.g. MediaPipe failure cases). For such
    # frames we fall back to identity, which yields an unwarped
    # passthrough instead of NaNs or a crash.
    det = torch.linalg.det(H).abs()
    safe = (det > 1e-6).view(-1, 1, 1)
    eye3 = torch.eye(3, device=device, dtype=dtype).expand_as(H)
    H_safe = torch.where(safe, H, eye3)
    H_inv = torch.linalg.inv(H_safe)  # (N, 3, 3)

    # Apply: src_h = H_inv @ canon^T  ->  src_h: (N, 3, HW)
    src_h = torch.bmm(H_inv, canon.transpose(1, 2))
    # Perspective divide
    w = src_h[:, 2:3, :]
    src = src_h[:, :2, :] / (w + 1e-8)           # (N, 2, HW)
    src = src.transpose(1, 2).reshape(N, Hout, Wout, 2)  # (N, H, W, 2)

    # Convert [0, 1] source coords to grid_sample's [-1, 1]
    src_norm = src * 2.0 - 1.0

    warped = F.grid_sample(
        frames, src_norm, mode='bilinear',
        padding_mode=padding_mode, align_corners=True)
    return warped


class CanonicalWarp(nn.Module):
    """Differentiable face canonicalization module.

    Forward:
        frames:     (B, C, T, H, W) raw video frames.
        landmarks:  (B, T, 68, 2) normalized landmarks in [0, 1].
        valid_mask: (B, T) optional, 1 for valid frames, 0 for invalid
                    (face detection failed — pass frame through).

    Returns:
        warped:     (B, C, T, H, W) canonicalized frames.
    """

    def __init__(self, anchor_indices=None, canonical_anchors=None,
                 padding_mode='zeros'):
        super().__init__()
        if anchor_indices is None:
            anchor_indices = ANCHOR_INDICES
        if canonical_anchors is None:
            canonical_anchors = CANONICAL_ANCHORS
        assert len(anchor_indices) == 4
        assert canonical_anchors.shape == (4, 2)
        self.register_buffer(
            'anchor_idx',
            torch.tensor(anchor_indices, dtype=torch.long))
        self.register_buffer(
            'canonical_anchors', canonical_anchors.clone())
        self.padding_mode = padding_mode

    def forward(self, frames, landmarks, valid_mask=None, homographies=None):
        B, C, T, H, W = frames.shape

        N = B * T

        if homographies is not None:
            # Use precomputed homographies (fast path — no lstsq needed).
            H_mat = homographies.reshape(N, 3, 3).to(frames.dtype)
        else:
            # Compute on-the-fly from landmarks (slow path — lstsq per frame).
            assert landmarks.shape[:2] == (B, T), \
                f"landmarks shape {tuple(landmarks.shape)} mismatch with {(B, T)}"
            src = landmarks.index_select(2, self.anchor_idx)
            dst = self.canonical_anchors.view(1, 1, 4, 2).expand(B, T, 4, 2)
            src_flat = src.reshape(N, 4, 2).to(frames.dtype)
            dst_flat = dst.reshape(N, 4, 2).to(frames.dtype)
            H_mat = compute_homography_4pt(src_flat, dst_flat)  # (N, 3, 3)

        # Reshape frames to (N, C, H, W) for warping
        # Input is (B, C, T, H, W); we want each T slice as a separate sample
        frames_flat = frames.permute(0, 2, 1, 3, 4).reshape(N, C, H, W)

        warped_flat = warp_by_homography(
            frames_flat, H_mat, out_size=(H, W),
            padding_mode=self.padding_mode)          # (N, C, H, W)

        # Restore to (B, C, T, H, W)
        warped = warped_flat.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)

        # For invalid frames, pass the original through unchanged
        if valid_mask is not None:
            vm = valid_mask.to(frames.dtype).view(B, 1, T, 1, 1)
            warped = vm * warped + (1.0 - vm) * frames

        return warped
