"""Unit tests for models/visibility_mask_attn.py."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from models.visibility_mask_attn import compute_visibility


# Three canonical normals: front-facing, subject-right, subject-left.
N_CANON_TEST = torch.tensor([
    [0.0, 0.0, 1.0],   # 0: front-facing landmark (e.g. nose tip)
    [1.0, 0.0, 0.0],   # 1: subject-right cheek
    [-1.0, 0.0, 0.0],  # 2: subject-left cheek
], dtype=torch.float32)


def _yaw_matrix(deg):
    """Standard rotation about the y-axis matching MediaPipe's convention.

    Positive yaw means the subject turns their head to subject's right
    (their right cheek rotates away from the camera, their left cheek
    rotates into view). This is the convention used by MediaPipe's
    facial_transformation_matrix and decoded by
    utils/mediapipe_landmarks.matrix_to_yaw_pitch_roll: R[0,2] = +sin(yaw),
    R[2,0] = -sin(yaw). Verified directly against real MMPD transforms.
    """
    a = np.deg2rad(deg)
    return torch.tensor([
        [ np.cos(a), 0.0, np.sin(a)],
        [ 0.0,       1.0, 0.0      ],
        [-np.sin(a), 0.0, np.cos(a)],
    ], dtype=torch.float32)


def _pitch_matrix(deg):
    a = np.deg2rad(deg)
    return torch.tensor([
        [1.0, 0.0,        0.0      ],
        [0.0, np.cos(a), -np.sin(a)],
        [0.0, np.sin(a),  np.cos(a)],
    ], dtype=torch.float32)


def test_visibility_identity_rotation():
    """A pure front-facing landmark (z=1) at identity rotation reads ~1."""
    R = torch.eye(3).view(1, 1, 3, 3)
    n_canon = torch.tensor([[0.0, 0.0, 1.0]] * 5, dtype=torch.float32)
    # Use bias=0 for this test so we measure the formula at the boundary.
    v = compute_visibility(R, n_canon, sharpness=8.0, bias=0.0)
    assert v.shape == (1, 1, 5)
    assert (v > 0.99).all(), f"expected all visibilities ~1, got {v}"


def test_visibility_yaw_plus_45_with_bias():
    """At yaw +45 deg (subject turns to subject's RIGHT, exposing left cheek):
    - subject-right cheek (canonical normal +x) rotates to -z, becomes LESS visible
    - subject-left cheek (-x) rotates to +z, becomes MORE visible
    - front-facing landmark stays high
    """
    R = _yaw_matrix(45).view(1, 1, 3, 3)
    v = compute_visibility(R, N_CANON_TEST, sharpness=8.0, bias=0.55)  # (1, 1, 3)
    v0, v1, v2 = v[0, 0].tolist()
    # Front-facing landmark stays highly visible (>0.99)
    assert v0 > 0.99, f"front visibility dropped: {v0}"
    # Subject-right (+x): rotates to z = -sin(45) = -0.707
    # sigmoid(8 * (-0.707 + 0.55)) = sigmoid(-1.26) ~= 0.221 -> invisible
    assert v1 < 0.3, f"subject-right should be occluded at yaw +45: {v1}"
    # Subject-left (-x): rotates to z = +sin(45) = +0.707
    # sigmoid(8 * (0.707 + 0.55)) = sigmoid(10.06) ~= 0.99996 -> visible
    assert v2 > 0.95, f"subject-left should be visible at yaw +45: {v2}"


def test_visibility_yaw_minus_45_with_bias():
    """Mirror of test_visibility_yaw_plus_45_with_bias.

    At yaw -45 deg (subject turns to subject's LEFT, exposing right cheek):
    - subject-right (+x) rotates toward +z, becomes MORE visible
    - subject-left (-x) rotates toward -z, becomes LESS visible
    """
    R = _yaw_matrix(-45).view(1, 1, 3, 3)
    v = compute_visibility(R, N_CANON_TEST, sharpness=8.0, bias=0.55)
    v0, v1, v2 = v[0, 0].tolist()
    assert v0 > 0.99
    assert v1 > 0.95, f"subject-right should be visible at yaw -45: {v1}"
    assert v2 < 0.3, f"subject-left should be occluded at yaw -45: {v2}"


def test_visibility_real_canonical_normals_frontal():
    """End-to-end check: load the actual canonical normals from disk and a
    real MediaPipe transform from a near-frontal MMPD frame, verify that
    the nose tip (dlib 30) is read as visible (>0.95) and forehead brow
    landmarks (17-26) average above 0.85.
    """
    project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    n_path = os.path.join(project, 'weights/mediapipe/canonical_normals_dlib68.npy')
    tf_path = os.path.join(project, 'datasets/MMPD/transforms/p16_0.npy')
    if not os.path.exists(n_path) or not os.path.exists(tf_path):
        print(f"  SKIP: missing data file ({n_path} or {tf_path})")
        return
    n = torch.from_numpy(np.load(n_path))                  # (68, 3)
    tf = np.load(tf_path)
    R = torch.from_numpy(tf[:1, :3, :3]).unsqueeze(0)      # (1, 1, 3, 3)
    v = compute_visibility(R, n, sharpness=8.0, bias=0.55)  # (1, 1, 68)
    nose_v = v[0, 0, 30].item()
    brow_v = v[0, 0, 17:27].mean().item()
    assert nose_v > 0.95, f"nose tip visibility too low at frontal: {nose_v}"
    assert brow_v > 0.85, f"brow visibility too low at frontal: {brow_v}"
    print(f"  nose tip visibility: {nose_v:.4f}")
    print(f"  brow mean visibility: {brow_v:.4f}")


from models.visibility_mask_attn import (
    signed_distance_to_convex_polygon,
    soft_rasterize_polygon,
)


def test_signed_distance_inside_outside():
    """Point inside the polygon -> negative; point outside -> positive."""
    # Square with vertices ordered such that the polygon interior is on the
    # right of each edge in image-y-down coordinates: top-left -> top-right
    # -> bottom-right -> bottom-left.
    poly = torch.tensor([
        [0.3, 0.3],
        [0.7, 0.3],
        [0.7, 0.7],
        [0.3, 0.7],
    ], dtype=torch.float32).view(1, 1, 4, 2)
    # Grid: (H=3, W=3, 2) at center (0.5, 0.5), corners (0,0), (1,1), edges
    grid = torch.tensor([
        [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]],
        [[0.0, 0.5], [0.5, 0.5], [1.0, 0.5]],
        [[0.0, 1.0], [0.5, 1.0], [1.0, 1.0]],
    ], dtype=torch.float32)
    sdf = signed_distance_to_convex_polygon(grid, poly)  # (1, 1, 3, 3)
    assert sdf.shape == (1, 1, 3, 3)
    # Center point is inside -> negative
    assert sdf[0, 0, 1, 1] < 0, f"center should be inside, got {sdf[0,0,1,1]}"
    # Corners are outside -> positive
    assert sdf[0, 0, 0, 0] > 0
    assert sdf[0, 0, 2, 2] > 0


def test_soft_rasterize_mass_concentrates_inside():
    """A square polygon at known location -> soft mask is brighter inside than outside."""
    poly = torch.tensor([
        [0.3, 0.3],
        [0.7, 0.3],
        [0.7, 0.7],
        [0.3, 0.7],
    ], dtype=torch.float32).view(1, 1, 4, 2)
    H = W = 13
    cells = torch.linspace(0, 1, H)
    grid_y, grid_x = torch.meshgrid(cells, cells, indexing='ij')
    grid = torch.stack([grid_x, grid_y], dim=-1)         # (H, W, 2)
    mask = soft_rasterize_polygon(poly, grid)            # (1, 1, H, W)
    assert mask.shape == (1, 1, H, W)
    # Inside center should be near 1.0
    assert mask[0, 0, H // 2, W // 2] > 0.9, mask[0, 0, H // 2, W // 2]
    # Outside corner should be near 0.0
    assert mask[0, 0, 0, 0] < 0.1, mask[0, 0, 0, 0]


from models.visibility_mask_attn import (
    LEFT_CHEEK_IDX,
    RIGHT_CHEEK_IDX,
    build_forehead_polygon,
    build_cheek_polygon,
)


def _make_frontal_landmarks():
    """Create a synthetic dlib-68 layout in normalized [0, 1] coords.

    Jaw points 0..16 trace the bottom; brow points 17..26 trace just above
    the eyes. Other points are placed approximately so the layout is
    self-consistent for forehead/cheek tests.
    """
    lm = torch.zeros(1, 1, 68, 2)
    # Jaw 0..16: arc along y=0.85, x evenly spaced from 0.15 to 0.85
    jaw_x = torch.linspace(0.15, 0.85, 17)
    lm[0, 0, 0:17, 0] = jaw_x
    lm[0, 0, 0:17, 1] = 0.85
    # Brows 17..21 (subject-right eyebrow): y=0.30, x from 0.20 to 0.40
    lm[0, 0, 17:22, 0] = torch.linspace(0.20, 0.40, 5)
    lm[0, 0, 17:22, 1] = 0.30
    # Brows 22..26 (subject-left eyebrow): y=0.30, x from 0.60 to 0.80
    lm[0, 0, 22:27, 0] = torch.linspace(0.60, 0.80, 5)
    lm[0, 0, 22:27, 1] = 0.30
    # Nose 27..30: bridge running down centerline
    lm[0, 0, 27:31, 0] = 0.5
    lm[0, 0, 27:31, 1] = torch.linspace(0.35, 0.55, 4)
    # Nose 31..35: bottom of nose y~0.60
    lm[0, 0, 31:36, 0] = torch.linspace(0.42, 0.58, 5)
    lm[0, 0, 31:36, 1] = 0.60
    # Eyes 36..47 placed below the brows
    lm[0, 0, 36:42, 0] = torch.linspace(0.22, 0.38, 6)
    lm[0, 0, 36:42, 1] = 0.40
    lm[0, 0, 42:48, 0] = torch.linspace(0.62, 0.78, 6)
    lm[0, 0, 42:48, 1] = 0.40
    # Mouth 48..67 around y~0.75
    lm[0, 0, 48:68, 0] = 0.5
    lm[0, 0, 48:68, 1] = 0.75
    return lm


def test_forehead_polygon_above_brows():
    lm = _make_frontal_landmarks()
    poly = build_forehead_polygon(lm)   # (1, 1, 4, 2)
    # The new convex 4-vertex rectangle.
    assert poly.shape == (1, 1, 4, 2), \
        f"expected (1, 1, 4, 2), got {poly.shape}"
    # Top of forehead should be ABOVE the brow line (smaller y)
    brow_y = 0.30
    top_y = poly[0, 0, :, 1].min().item()
    assert top_y < brow_y, f"forehead top {top_y} not above brow {brow_y}"
    # And bounded below by 0
    assert top_y >= 0.0
    # Bottom of forehead should be at the brow line (brow_y_min for synthetic
    # landmarks where all brow points share y=0.30)
    bot_y = poly[0, 0, :, 1].max().item()
    assert abs(bot_y - 0.30) < 1e-5, f"forehead bottom {bot_y} not at brow line"


def test_cheek_polygons_have_expected_size():
    lm = _make_frontal_landmarks()
    left = build_cheek_polygon(lm, LEFT_CHEEK_IDX)
    right = build_cheek_polygon(lm, RIGHT_CHEEK_IDX)
    assert left.shape == (1, 1, len(LEFT_CHEEK_IDX), 2)
    assert right.shape == (1, 1, len(RIGHT_CHEEK_IDX), 2)
    # Subject-left cheek (image right) is on the right side of the
    # synthetic frontal frame: mean x > 0.5.
    assert left[0, 0, :, 0].mean().item() > 0.5
    assert right[0, 0, :, 0].mean().item() < 0.5


from models.visibility_mask_attn import VisibilityMaskAttn


def test_module_has_exactly_4_trainable_params():
    m = VisibilityMaskAttn(
        canonical_normals_path="weights/mediapipe/canonical_normals_dlib68.npy")
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert n == 4, f"expected 4 trainable scalars, got {n}"
    # And the canonical normals are a non-trainable buffer.
    assert hasattr(m, "n_canon")
    assert m.n_canon.shape == (68, 3)
    assert not m.n_canon.requires_grad


def test_forward_shape_and_gradient():
    m = VisibilityMaskAttn(
        canonical_normals_path="weights/mediapipe/canonical_normals_dlib68.npy")
    B, T, C, H, W = 2, 4, 16, 13, 13
    voxel = torch.randn(B, C, T, H, W)  # (B, C, T, H, W)
    voxel.requires_grad_(True)
    landmarks = torch.rand(B, T, 68, 2)
    R = torch.eye(3).view(1, 1, 3, 3).expand(B, T, 3, 3).contiguous()
    transforms = torch.zeros(B, T, 4, 4)
    transforms[..., :3, :3] = R
    transforms[..., 3, 3] = 1
    out = m(voxel, landmarks, transforms)
    assert out.shape == voxel.shape
    out.sum().backward()
    # Gradient flows into the 4 module parameters
    assert m.region_logits.grad is not None
    assert m.alpha_logit.grad is not None


def test_forward_valid_mask_collapses_to_identity():
    m = VisibilityMaskAttn(
        canonical_normals_path="weights/mediapipe/canonical_normals_dlib68.npy")
    B, T = 1, 4
    voxel = torch.randn(B, 16, T, 13, 13)
    landmarks = torch.rand(B, T, 68, 2)
    transforms = torch.eye(4).view(1, 1, 4, 4).expand(B, T, 4, 4).contiguous()
    valid_mask = torch.zeros(B, T)   # ALL frames invalid
    out = m(voxel, landmarks, transforms, valid_mask=valid_mask)
    # Mask is all-ones on invalid frames -> alpha + (1-alpha)*1 = 1 -> output == voxel
    torch.testing.assert_close(out, voxel)


def test_mask_dropout_zero_is_identity():
    """mask_dropout=0.0 produces identical output to the default constructor."""
    m_default = VisibilityMaskAttn(
        canonical_normals_path="weights/mediapipe/canonical_normals_dlib68.npy")
    m_nodrop = VisibilityMaskAttn(
        canonical_normals_path="weights/mediapipe/canonical_normals_dlib68.npy",
        mask_dropout=0.0)
    # Copy weights so outputs are comparable.
    m_nodrop.load_state_dict(m_default.state_dict())
    B, T, C, H, W = 1, 4, 16, 13, 13
    voxel = torch.randn(B, C, T, H, W)
    landmarks = torch.rand(B, T, 68, 2)
    transforms = torch.eye(4).view(1, 1, 4, 4).expand(B, T, 4, 4).contiguous()
    m_default.eval()
    m_nodrop.eval()
    out_default = m_default(voxel, landmarks, transforms)
    out_nodrop = m_nodrop(voxel, landmarks, transforms)
    torch.testing.assert_close(out_default, out_nodrop)


def test_mask_dropout_one_zeros_mask():
    """mask_dropout=1.0 during training zeros the entire skin mask,
    so output = voxel * alpha (floor only, no spatial modulation)."""
    m = VisibilityMaskAttn(
        canonical_normals_path="weights/mediapipe/canonical_normals_dlib68.npy",
        mask_dropout=1.0)
    m.train()
    B, T, C, H, W = 2, 4, 16, 13, 13
    voxel = torch.randn(B, C, T, H, W)
    landmarks = torch.rand(B, T, 68, 2)
    transforms = torch.eye(4).view(1, 1, 4, 4).expand(B, T, 4, 4).contiguous()
    out = m(voxel, landmarks, transforms)
    alpha = torch.sigmoid(m.alpha_logit)
    # With mask all-zero: output = voxel * (alpha + (1-alpha)*0) = voxel * alpha
    expected = voxel * alpha
    torch.testing.assert_close(out, expected)


def test_mask_dropout_disabled_at_eval():
    """Even with mask_dropout=0.9, eval mode produces deterministic output
    identical to mask_dropout=0.0."""
    m_drop = VisibilityMaskAttn(
        canonical_normals_path="weights/mediapipe/canonical_normals_dlib68.npy",
        mask_dropout=0.9)
    m_nodrop = VisibilityMaskAttn(
        canonical_normals_path="weights/mediapipe/canonical_normals_dlib68.npy",
        mask_dropout=0.0)
    m_nodrop.load_state_dict(m_drop.state_dict())
    B, T, C, H, W = 1, 4, 16, 13, 13
    voxel = torch.randn(B, C, T, H, W)
    landmarks = torch.rand(B, T, 68, 2)
    transforms = torch.eye(4).view(1, 1, 4, 4).expand(B, T, 4, 4).contiguous()
    m_drop.eval()
    m_nodrop.eval()
    out_drop = m_drop(voxel, landmarks, transforms)
    out_nodrop = m_nodrop(voxel, landmarks, transforms)
    torch.testing.assert_close(out_drop, out_nodrop)


def test_signed_distance_auto_winding_ccw_polygon():
    """A CCW polygon (in image-y-down) should still give correct interior/exterior signs.

    Walks the same square as test_signed_distance_inside_outside but in
    REVERSED order (CCW in image coords). The auto-winding-detection in
    signed_distance_to_convex_polygon should flip the normals so the
    interior is still detected as negative SDF.
    """
    poly_ccw = torch.tensor([
        [0.3, 0.3],   # top-left
        [0.3, 0.7],   # bottom-left
        [0.7, 0.7],   # bottom-right
        [0.7, 0.3],   # top-right
    ], dtype=torch.float32).view(1, 1, 4, 2)   # CCW (reversed of test_signed_distance_inside_outside)
    grid = torch.tensor([
        [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]],
        [[0.0, 0.5], [0.5, 0.5], [1.0, 0.5]],
        [[0.0, 1.0], [0.5, 1.0], [1.0, 1.0]],
    ], dtype=torch.float32)
    sdf = signed_distance_to_convex_polygon(grid, poly_ccw)
    assert sdf.shape == (1, 1, 3, 3)
    # Center is inside -> negative
    assert sdf[0, 0, 1, 1] < 0, f"CCW center should be inside, got {sdf[0,0,1,1]}"
    # Corners are outside -> positive
    assert sdf[0, 0, 0, 0] > 0
    assert sdf[0, 0, 2, 2] > 0


def _run_all():
    failed = []
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except Exception as e:
                print(f"FAIL  {name}: {e}")
                failed.append(name)
    if failed:
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    _run_all()
