"""POS knowledge distillation loss for canonical-space rPPG.

Applies the POS method (Wang et al., 2017) to fixed canonical-space
forehead pixels to produce a pseudo-BVP signal, then computes
NegPearson loss between the model's predicted BVP and the pseudo-BVP.

This is only possible in canonical space: the forehead ROI sits at
fixed pixel coordinates for every subject and every frame, so the
classical method receives geometrically stable input.  No prior rPPG
method uses classical methods as auxiliary teachers for deep methods.

The pseudo-BVP is detached — POS is the teacher, gradients flow only
through the model's predicted BVP back into the backbone.

Reference implementation adapted from rPPG-Toolbox:
    unsupervised_methods/methods/POS_WANG.py
"""

import math

import torch
import torch.nn as nn


def _build_forehead_mask(spatial_size=72, cx=0.50, cy=0.20, sigma=0.10):
    """Build a Gaussian soft mask for the canonical forehead ROI.

    Returns:
        mask: (H, W) float32 in [0, 1].
    """
    ys = torch.linspace(0.0, 1.0, spatial_size)
    xs = torch.linspace(0.0, 1.0, spatial_size)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    d2 = (grid_x - cx) ** 2 + (grid_y - cy) ** 2
    return torch.exp(-d2 / (2 * sigma ** 2)).float()


def extract_roi_rgb(canonical_frames, roi_mask):
    """Extract ROI-averaged RGB trace from canonical frames.

    Args:
        canonical_frames: (B, C, T, H, W) canonical-space video (C=3 RGB).
        roi_mask: (H, W) soft mask.

    Returns:
        rgb: (B, T, 3) spatially-averaged RGB per frame.
    """
    # (B, C, T, H, W) * (H, W) -> weighted sum over H, W
    mask = roi_mask.to(canonical_frames.dtype).to(canonical_frames.device)
    # Expand mask: (1, 1, 1, H, W)
    m = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    weighted = (canonical_frames * m).sum(dim=(-1, -2))   # (B, C, T)
    den = mask.sum() + 1e-6
    rgb = weighted / den                                   # (B, C, T)
    return rgb.permute(0, 2, 1)                            # (B, T, 3)


def pos_projection(rgb, fps=30.0, win_sec=1.6):
    """Windowed POS projection (Wang et al., 2017) in PyTorch.

    Args:
        rgb: (B, T, 3) spatially-averaged RGB trace.
        fps: sampling rate in Hz.
        win_sec: POS window length in seconds.

    Returns:
        pulse: (B, T) pseudo-BVP signal.
    """
    B, T, _ = rgb.shape
    device = rgb.device
    dtype = rgb.dtype

    win_len = math.ceil(win_sec * fps)
    pulse = torch.zeros(B, T, device=device, dtype=dtype)

    # POS projection matrix: [[0, 1, -1], [-2, 1, 1]]
    P = torch.tensor([[0.0, 1.0, -1.0],
                       [-2.0, 1.0, 1.0]], device=device, dtype=dtype)

    for n in range(win_len, T + 1):
        m = n - win_len
        window_rgb = rgb[:, m:n, :]                        # (B, L, 3)
        win_mean = window_rgb.mean(dim=1, keepdim=True)    # (B, 1, 3)

        # Skip degenerate windows (all-black from warp padding)
        safe = (win_mean.abs().sum(dim=-1, keepdim=True) > 1e-6).float()

        # Temporal normalization: Cn = RGB / mean(RGB)
        Cn = window_rgb / (win_mean + 1e-8)               # (B, L, 3)

        # Project: S = P @ Cn^T -> (B, 2, L)
        S = torch.matmul(P, Cn.transpose(1, 2))           # (B, 2, L)
        S0 = S[:, 0, :]                                   # (B, L)
        S1 = S[:, 1, :]                                   # (B, L)

        # Combine: h = S0 + (std(S0) / std(S1)) * S1
        std0 = S0.std(dim=-1, keepdim=True) + 1e-8        # (B, 1)
        std1 = S1.std(dim=-1, keepdim=True) + 1e-8        # (B, 1)
        h = S0 + (std0 / std1) * S1                       # (B, L)

        # Zero-mean
        h = h - h.mean(dim=-1, keepdim=True)

        # Mask out degenerate windows
        h = h * safe.squeeze(-1)

        # Overlap-add
        pulse[:, m:n] = pulse[:, m:n] + h

    return pulse


class POSDistillationLoss(nn.Module):
    """POS knowledge distillation loss for canonical-space rPPG.

    Forward signature:
        loss = module(canonical_frames, predicted_bvp)

    Args:
        fps: video frame rate (default 30.0).
        win_sec: POS window length in seconds (default 1.6).
        spatial_size: canonical frame spatial resolution (default 72).
    """

    def __init__(self, fps=30.0, win_sec=1.6, spatial_size=72):
        super().__init__()
        self.fps = float(fps)
        self.win_sec = float(win_sec)
        mask = _build_forehead_mask(spatial_size)
        self.register_buffer('roi_mask', mask, persistent=False)

    def forward(self, canonical_frames, predicted_bvp):
        """
        Args:
            canonical_frames: (B, C, T_plus_1, H, W) canonical-space
                video BEFORE temporal diff and physics weighting.
                T_plus_1 = T + 1 (the extra appended frame).
            predicted_bvp: (B, T) model's BVP prediction.

        Returns:
            loss: scalar NegPearson between predicted BVP and POS
                pseudo-BVP.
        """
        B, T = predicted_bvp.shape

        # Extract forehead RGB from canonical frames
        rgb = extract_roi_rgb(canonical_frames, self.roi_mask)  # (B, T+1, 3)

        # POS projection -> pseudo-BVP
        pseudo_bvp = pos_projection(rgb, fps=self.fps,
                                    win_sec=self.win_sec)       # (B, T+1)
        # Slice to T frames to match predicted BVP
        pseudo_bvp = pseudo_bvp[:, :T]                          # (B, T)

        # Detach: POS is the teacher
        pseudo_bvp = pseudo_bvp.detach()

        # Normalize both signals
        pred = predicted_bvp - predicted_bvp.mean(dim=-1, keepdim=True)
        pred = pred / (pred.std(dim=-1, keepdim=True) + 1e-8)

        pseudo = pseudo_bvp - pseudo_bvp.mean(dim=-1, keepdim=True)
        pseudo = pseudo / (pseudo.std(dim=-1, keepdim=True) + 1e-8)

        # NegPearson: 1 - pearson_correlation
        cos = (pred * pseudo).mean(dim=-1)                      # (B,)
        loss = (1.0 - cos).mean()
        return loss
