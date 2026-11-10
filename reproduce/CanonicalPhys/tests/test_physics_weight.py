"""Unit tests for PhysicsWeight."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from models.physics_weight import (
    PhysicsWeight,
    build_pixel_to_landmark_assignment,
)


def test_assignment_shape_and_range():
    K = 68
    canon = np.random.rand(K, 2).astype(np.float32)
    assign = build_pixel_to_landmark_assignment(canon, spatial_size=72)
    assert assign.shape == (72, 72)
    assert assign.min() >= 0 and assign.max() < K
    print("PASS: assignment shape and range")


def test_assignment_nearest():
    # Two landmarks at (0, 0) and (1, 1). Pixel (0,0) should assign to lm 0.
    canon = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    assign = build_pixel_to_landmark_assignment(canon, spatial_size=10)
    assert assign[0, 0] == 0, f"Expected 0, got {assign[0, 0]}"
    assert assign[-1, -1] == 1, f"Expected 1, got {assign[-1, -1]}"
    print("PASS: nearest-landmark assignment")


def test_physics_weight_identity_output_shape():
    """Forward with identity transform gives the right shape."""
    mod = PhysicsWeight()
    B, T = 2, 4
    I = torch.eye(4).view(1, 1, 4, 4).expand(B, T, 4, 4).contiguous()
    w = mod(I)
    assert w.shape == (B, T, 72, 72), f"Got {w.shape}"
    assert (w > 0).all() and (w < 1).all()
    print(f"PASS: identity shape={w.shape}, in (0,1)")


def test_physics_weight_yaw_45_contralateral_asymmetric():
    """45° yaw: one side's normals rotate away from the camera.
    The left-right weight difference should be substantial."""
    mod = PhysicsWeight()
    B, T = 1, 1
    theta = float(np.pi / 4)
    c, s = np.cos(theta).astype(np.float32), np.sin(theta).astype(np.float32)
    R_yaw = torch.tensor([[c, 0.0, s, 0.0],
                          [0.0, 1.0, 0.0, 0.0],
                          [-s, 0.0, c, 0.0],
                          [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
    transforms = R_yaw.view(1, 1, 4, 4).expand(B, T, 4, 4).contiguous()
    w = mod(transforms)             # (B, T, 72, 72)
    # Left half vs right half mean weight
    left_mean = w[:, :, :, :36].mean().item()
    right_mean = w[:, :, :, 36:].mean().item()
    diff = abs(left_mean - right_mean)
    assert diff > 0.1, \
        f"Yaw 45° should give asymmetric weights, got diff {diff:.4f}"
    print(f"PASS: yaw 45° -> L={left_mean:.3f}, R={right_mean:.3f} (diff {diff:.3f})")


def test_physics_weight_forehead_stronger_than_jaw():
    """Under identity pose, central pixels (forehead-aligned) should
    have higher weight than peripheral pixels (jaw-aligned)."""
    mod = PhysicsWeight()
    I = torch.eye(4).view(1, 1, 4, 4).contiguous()
    w = mod(I)  # (1, 1, 72, 72)
    # Center region (forehead/face) vs corners
    center = w[0, 0, 20:52, 20:52].mean().item()
    corners = torch.cat([
        w[0, 0, :10, :10].flatten(),
        w[0, 0, :10, -10:].flatten(),
        w[0, 0, -10:, :10].flatten(),
        w[0, 0, -10:, -10:].flatten(),
    ]).mean().item()
    assert center > corners, f"Center {center:.3f} should exceed corners {corners:.3f}"
    print(f"PASS: center {center:.3f} > corners {corners:.3f}")


def test_physics_weight_no_params():
    """PhysicsWeight must have zero trainable parameters."""
    mod = PhysicsWeight()
    n = sum(p.numel() for p in mod.parameters() if p.requires_grad)
    assert n == 0, f"Expected 0 trainable params, got {n}"
    print("PASS: zero trainable parameters")


if __name__ == "__main__":
    print("=" * 60)
    print("PhysicsWeight Unit Tests")
    print("=" * 60)
    tests = [
        ("Assignment shape/range", test_assignment_shape_and_range),
        ("Nearest assignment", test_assignment_nearest),
        ("Identity output shape", test_physics_weight_identity_output_shape),
        ("Yaw 45° asymmetric", test_physics_weight_yaw_45_contralateral_asymmetric),
        ("Center > corners", test_physics_weight_forehead_stronger_than_jaw),
        ("Zero params", test_physics_weight_no_params),
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
