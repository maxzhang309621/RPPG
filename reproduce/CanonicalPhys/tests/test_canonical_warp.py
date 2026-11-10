"""Unit tests for canonical_warp (homography + canonicalization)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models.canonical_warp import (
    compute_homography_4pt,
    warp_by_homography,
    CanonicalWarp,
    CANONICAL_ANCHORS,
    ANCHOR_INDICES,
)


def test_homography_identity():
    """src == dst -> H is identity."""
    src = torch.tensor([[[0.1, 0.2], [0.9, 0.2], [0.2, 0.8], [0.8, 0.8]]])
    H = compute_homography_4pt(src, src)
    torch.testing.assert_close(H, torch.eye(3).unsqueeze(0), atol=1e-5, rtol=1e-5)
    print("PASS: homography src==dst -> identity")


def test_homography_maps_points():
    """The computed H actually maps each src point to its dst."""
    src = torch.tensor([[[0.1, 0.2], [0.9, 0.1], [0.15, 0.7], [0.85, 0.85]]])
    dst = torch.tensor([[[0.30, 0.40], [0.70, 0.40], [0.38, 0.72], [0.62, 0.72]]])
    H = compute_homography_4pt(src, dst)  # (1, 3, 3)

    # Apply H to src in homogeneous form
    src_h = torch.cat([src, torch.ones(1, 4, 1)], dim=-1)  # (1, 4, 3)
    mapped_h = torch.bmm(H, src_h.transpose(1, 2)).transpose(1, 2)  # (1, 4, 3)
    mapped = mapped_h[..., :2] / mapped_h[..., 2:3]

    torch.testing.assert_close(mapped, dst, atol=1e-5, rtol=1e-5)
    print("PASS: homography H @ src == dst")


def test_warp_identity_passes_through():
    """Warping with identity homography leaves the frame unchanged."""
    N, C, H, W = 1, 3, 16, 16
    frames = torch.randn(N, C, H, W)
    H_id = torch.eye(3).unsqueeze(0)
    warped = warp_by_homography(frames, H_id, padding_mode='border')
    # With align_corners=True the grid samples input pixels exactly, so
    # expect near-perfect equality.
    torch.testing.assert_close(warped, frames, atol=1e-4, rtol=1e-4)
    print("PASS: identity homography -> identity warp")


def test_canonical_warp_identity_frontal():
    """When observed anchors already match canonical anchors, warp is identity."""
    B, C, T, H, W = 1, 3, 2, 72, 72
    frames = torch.randn(B, C, T, H, W)
    # Start with "random" landmarks, but overwrite anchor landmarks to
    # match the canonical positions exactly.
    landmarks = torch.rand(B, T, 68, 2)
    for idx, (cx, cy) in zip(ANCHOR_INDICES, CANONICAL_ANCHORS):
        landmarks[:, :, idx, 0] = cx
        landmarks[:, :, idx, 1] = cy

    module = CanonicalWarp()
    warped = module(frames, landmarks)
    torch.testing.assert_close(warped, frames, atol=1e-4, rtol=1e-4)
    print("PASS: frontal (anchors=canonical) -> identity warp")


def test_canonical_warp_shifts_misaligned_face():
    """When anchors are shifted, the warp produces a different output."""
    B, C, T, H, W = 1, 3, 2, 72, 72
    frames = torch.randn(B, C, T, H, W)
    landmarks = torch.rand(B, T, 68, 2)
    # Put anchors at non-canonical positions (shifted right and down)
    for idx, (cx, cy) in zip(ANCHOR_INDICES, CANONICAL_ANCHORS):
        landmarks[:, :, idx, 0] = cx + 0.1
        landmarks[:, :, idx, 1] = cy + 0.1

    module = CanonicalWarp()
    warped = module(frames, landmarks)

    # The warp should be noticeably different from the input
    diff = (warped - frames).abs().mean()
    assert diff > 0.05, f"Expected visible warp, got mean diff {diff:.4f}"
    print(f"PASS: shifted anchors -> non-trivial warp (mean diff {diff:.4f})")


def test_canonical_warp_valid_mask():
    """Invalid frames pass through unchanged."""
    B, C, T, H, W = 1, 3, 2, 72, 72
    frames = torch.randn(B, C, T, H, W)
    landmarks = torch.rand(B, T, 68, 2)
    # Non-canonical anchors (so the warp is meaningful)
    for idx, (cx, cy) in zip(ANCHOR_INDICES, CANONICAL_ANCHORS):
        landmarks[:, :, idx, 0] = cx + 0.15
        landmarks[:, :, idx, 1] = cy - 0.05

    module = CanonicalWarp()
    # First frame invalid, second valid
    valid_mask = torch.tensor([[0.0, 1.0]])
    warped = module(frames, landmarks, valid_mask=valid_mask)

    # Frame 0: should be identical to input
    torch.testing.assert_close(
        warped[:, :, 0], frames[:, :, 0], atol=1e-5, rtol=1e-5)
    # Frame 1: should NOT match input (non-trivial warp)
    diff1 = (warped[:, :, 1] - frames[:, :, 1]).abs().mean()
    assert diff1 > 0.05, f"Valid-frame warp should differ, mean diff {diff1:.4f}"
    print("PASS: invalid frames passthrough, valid frames warped")


def test_canonical_warp_gradient_flow():
    """Gradients flow through the warp back to the input frames."""
    B, C, T, H, W = 1, 3, 2, 72, 72
    frames = torch.randn(B, C, T, H, W, requires_grad=True)
    landmarks = torch.rand(B, T, 68, 2)

    module = CanonicalWarp()
    warped = module(frames, landmarks)
    loss = warped.mean()
    loss.backward()

    assert frames.grad is not None
    assert frames.grad.abs().sum() > 0
    print("PASS: gradient flows through the warp")


def test_degenerate_landmarks_fall_back_to_identity():
    """Degenerate anchor landmarks (all coincident) should give identity H,
    and CanonicalWarp should pass the frame through unchanged without
    raising a linalg error."""
    # Direct test of the homography solver with coincident src points.
    src_bad = torch.tensor([[[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]])
    dst_ok = torch.tensor([[[0.3, 0.4], [0.7, 0.4], [0.38, 0.72], [0.62, 0.72]]])
    H = compute_homography_4pt(src_bad, dst_ok)
    torch.testing.assert_close(H, torch.eye(3).unsqueeze(0), atol=1e-5, rtol=1e-5)

    # Full CanonicalWarp with one degenerate frame in a batch
    B, C, T, H_px, W_px = 1, 3, 3, 72, 72
    frames = torch.randn(B, C, T, H_px, W_px)
    landmarks = torch.rand(B, T, 68, 2)
    # Frame 1: set all 4 anchor landmarks to the same point
    for idx in ANCHOR_INDICES:
        landmarks[0, 1, idx] = torch.tensor([0.5, 0.5])
    module = CanonicalWarp()
    warped = module(frames, landmarks)
    # Frame 1 should pass through unchanged (identity warp)
    torch.testing.assert_close(
        warped[:, :, 1], frames[:, :, 1], atol=1e-4, rtol=1e-4)
    print("PASS: degenerate landmarks -> identity fallback (no crash)")


def test_singular_homography_in_warp_falls_back():
    """Even if a caller passes a rank-deficient H to warp_by_homography,
    the function must not crash."""
    N, C, H, W = 2, 3, 16, 16
    frames = torch.randn(N, C, H, W)
    # One good (identity), one rank-2 (second row is zero).
    H_good = torch.eye(3).unsqueeze(0)
    H_bad = torch.tensor([[[1.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0]]])
    H_batch = torch.cat([H_good, H_bad], dim=0)
    warped = warp_by_homography(frames, H_batch, padding_mode='border')
    assert warped.shape == frames.shape
    # Good sample unchanged
    torch.testing.assert_close(warped[0], frames[0], atol=1e-4, rtol=1e-4)
    # Bad sample also unchanged (identity fallback)
    torch.testing.assert_close(warped[1], frames[1], atol=1e-4, rtol=1e-4)
    print("PASS: singular H in warp_by_homography falls back to identity")


def test_canonical_warp_shapes_and_dtypes():
    """Canonicalization preserves (B, C, T, H, W) shape and dtype."""
    B, C, T, H, W = 2, 3, 8, 72, 72
    frames = torch.randn(B, C, T, H, W, dtype=torch.float32)
    landmarks = torch.rand(B, T, 68, 2)

    module = CanonicalWarp()
    warped = module(frames, landmarks)
    assert warped.shape == (B, C, T, H, W), f"Got {warped.shape}"
    assert warped.dtype == torch.float32
    print("PASS: shape + dtype preserved")


if __name__ == "__main__":
    print("=" * 60)
    print("CanonicalWarp Unit Tests")
    print("=" * 60)
    tests = [
        ("Homography identity", test_homography_identity),
        ("Homography maps points", test_homography_maps_points),
        ("Warp identity passthrough", test_warp_identity_passes_through),
        ("Canonical warp frontal", test_canonical_warp_identity_frontal),
        ("Canonical warp misaligned", test_canonical_warp_shifts_misaligned_face),
        ("Canonical warp valid mask", test_canonical_warp_valid_mask),
        ("Canonical warp gradient", test_canonical_warp_gradient_flow),
        ("Degenerate landmarks", test_degenerate_landmarks_fall_back_to_identity),
        ("Singular H in warp", test_singular_homography_in_warp_falls_back),
        ("Canonical warp shapes", test_canonical_warp_shapes_and_dtypes),
    ]
    passed, failed = 0, 0
    for name, fn in tests:
        print(f"--- {name} ---")
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    if failed:
        sys.exit(1)
