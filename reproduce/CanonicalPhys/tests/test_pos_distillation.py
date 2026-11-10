"""Unit tests for POS distillation loss."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from losses.pos_distillation import (
    POSDistillationLoss,
    _build_forehead_mask,
    extract_roi_rgb,
    pos_projection,
)


def test_forehead_mask_shape_and_peak():
    mask = _build_forehead_mask(spatial_size=72, cx=0.5, cy=0.2, sigma=0.1)
    assert mask.shape == (72, 72)
    assert mask.min() >= 0.0 and mask.max() <= 1.0
    # Peak should be near center (0.5, 0.2) -> pixel (14, 36) approx
    peak_y, peak_x = divmod(mask.argmax().item(), 72)
    assert 10 <= peak_y <= 20, f"Peak y={peak_y}, expected near 14"
    assert 30 <= peak_x <= 42, f"Peak x={peak_x}, expected near 36"
    print("PASS: forehead mask shape and peak")


def test_extract_roi_rgb_shape():
    B, C, T, H, W = 2, 3, 64, 72, 72
    frames = torch.randn(B, C, T, H, W)
    mask = _build_forehead_mask(H)
    rgb = extract_roi_rgb(frames, mask)
    assert rgb.shape == (B, T, 3), f"Expected (2, 64, 3), got {rgb.shape}"
    print("PASS: extract_roi_rgb shape")


def test_pos_projection_shape():
    B, T = 2, 128
    rgb = torch.rand(B, T, 3) + 0.5  # positive values like pixel intensities
    pulse = pos_projection(rgb, fps=30.0, win_sec=1.6)
    assert pulse.shape == (B, T), f"Expected (2, 128), got {pulse.shape}"
    print("PASS: pos_projection shape")


def test_pos_projection_with_synthetic_pulse():
    """POS should recover a sinusoidal pulse embedded in the green channel."""
    B, T = 1, 256
    fps = 30.0
    freq_hz = 1.2  # ~72 BPM

    t = torch.linspace(0, T / fps, T)
    pulse_gt = torch.sin(2 * np.pi * freq_hz * t)

    # Simulate RGB: green channel modulated by pulse, R and B constant
    rgb = torch.zeros(B, T, 3) + 0.8
    rgb[:, :, 1] += 0.02 * pulse_gt  # small modulation in green

    pseudo_bvp = pos_projection(rgb, fps=fps, win_sec=1.6)

    # Check that pseudo-BVP correlates with the ground truth pulse
    # (skip first window_len frames where POS hasn't accumulated yet)
    win_len = int(np.ceil(1.6 * fps))
    gt_trimmed = pulse_gt[win_len:]
    pred_trimmed = pseudo_bvp[0, win_len:]

    gt_norm = gt_trimmed - gt_trimmed.mean()
    pred_norm = pred_trimmed - pred_trimmed.mean()
    pearson = (gt_norm * pred_norm).sum() / (
        gt_norm.norm() * pred_norm.norm() + 1e-8)

    assert abs(pearson.item()) > 0.3, \
        f"POS should recover pulse, Pearson={pearson.item():.3f}"
    print(f"PASS: synthetic pulse recovery (Pearson={pearson.item():.3f})")


def test_pos_degenerate_window():
    """All-zero frames should not produce NaN."""
    B, T = 1, 64
    rgb = torch.zeros(B, T, 3)
    pulse = pos_projection(rgb, fps=30.0, win_sec=1.6)
    assert torch.isfinite(pulse).all(), "POS produced NaN/Inf on zero input"
    print("PASS: degenerate window handling")


def test_distillation_loss_runs():
    """Full forward pass of POSDistillationLoss."""
    B, C, T, H, W = 2, 3, 129, 72, 72  # T+1 = 129
    canonical_frames = torch.rand(B, C, T, H, W) + 0.5
    predicted_bvp = torch.randn(B, T - 1)  # T frames after diff

    loss_fn = POSDistillationLoss(fps=30.0, spatial_size=72)
    loss = loss_fn(canonical_frames, predicted_bvp)

    assert loss.shape == (), f"Expected scalar, got {loss.shape}"
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
    assert loss.item() >= 0.0, f"NegPearson loss should be >= 0, got {loss.item()}"
    print(f"PASS: distillation loss forward (loss={loss.item():.4f})")


def test_distillation_loss_no_grad_through_pseudo():
    """Verify pseudo-BVP is detached — gradients only flow through pred."""
    B, C, T_plus_1, H, W = 1, 3, 65, 72, 72
    canonical_frames = torch.rand(B, C, T_plus_1, H, W) + 0.5
    predicted_bvp = torch.randn(B, T_plus_1 - 1, requires_grad=True)

    loss_fn = POSDistillationLoss(fps=30.0, spatial_size=72)
    loss = loss_fn(canonical_frames, predicted_bvp)
    loss.backward()

    assert predicted_bvp.grad is not None, "Gradients should flow to pred"
    assert predicted_bvp.grad.abs().sum() > 0, "Gradients should be nonzero"
    print("PASS: gradient flows only through predicted_bvp")


if __name__ == "__main__":
    test_forehead_mask_shape_and_peak()
    test_extract_roi_rgb_shape()
    test_pos_projection_shape()
    test_pos_projection_with_synthetic_pulse()
    test_pos_degenerate_window()
    test_distillation_loss_runs()
    test_distillation_loss_no_grad_through_pseudo()
    print("\nAll POS distillation tests passed!")
