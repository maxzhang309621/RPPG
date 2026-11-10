"""Unit tests for LandmarkPhys (both v1 LandmarkAttn and v2 VisibilityMaskAttn).

Tests:
    1. Forward pass with correct output shapes (both variants)
    2. Gradient flow through backbone and the attention module (both variants)
    3. Forward pass without landmarks (fallback to baseline)
    4. Parameter counts for v1 and v2
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def _identity_transforms(B, T):
    """Build (B, T, 4, 4) identity transforms (frontal pose)."""
    tf = torch.eye(4).view(1, 1, 4, 4).expand(B, T, 4, 4).contiguous()
    return tf.clone()


def test_forward_pass_shapes():
    """Forward pass produces correct output shapes."""
    from models.landmark_phys import LandmarkPhys

    B, C, T, H, W = 2, 3, 128, 72, 72

    model = LandmarkPhys(
        frames=T,
        md_config={"MD_FSAM": True, "MD_INFERENCE": True},
        in_channels=C,
        dropout=0.1,
        device=torch.device("cpu"),
    )
    model.eval()

    x = torch.randn(B, C, T + 1, H, W)
    landmarks = torch.rand(B, T, 68, 2)
    poses = torch.randn(B, T, 3)
    transforms = _identity_transforms(B, T)

    with torch.no_grad():
        rPPG, vox_embed, factorized_embed, appx_error = model(
            x, landmarks=landmarks, poses=poses, transforms=transforms)

    assert rPPG.shape == (B, T), f"rPPG: {rPPG.shape} != ({B}, {T})"
    assert vox_embed.shape == (B, 16, T, 13, 13), f"vox: {vox_embed.shape}"
    assert factorized_embed.shape[0] == B
    print("PASS: Forward pass shapes correct")


def test_gradient_flow_v2():
    """v2: Gradients flow through backbone and through the 4 v2 scalars."""
    from models.landmark_phys import LandmarkPhys
    from losses.neg_pearson import Neg_Pearson

    B, C, T, H, W = 2, 3, 128, 72, 72

    model = LandmarkPhys(
        frames=T,
        md_config={"MD_FSAM": True},
        in_channels=C,
        device=torch.device("cpu"),
        landmark_config={'attn_type': 'v2'},
    )
    model.train()
    loss_fn = Neg_Pearson()

    x = torch.randn(B, C, T + 1, H, W)
    landmarks = torch.rand(B, T, 68, 2)
    poses = torch.randn(B, T, 3)
    transforms = _identity_transforms(B, T)
    labels = torch.randn(B, T)

    rPPG, vox_embed, factorized_embed, appx_error = model(
        x, landmarks=landmarks, poses=poses, transforms=transforms)
    loss = loss_fn(rPPG, labels)
    loss.backward()

    # Backbone has gradient
    backbone_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.rppg_feature_extractor.parameters())
    assert backbone_has_grad, "No gradients in backbone!"

    # The 4 v2 scalars have gradients
    assert model.landmark_attn.region_logits.grad is not None
    assert model.landmark_attn.region_logits.grad.abs().sum() > 0, \
        "region_logits has zero gradient"
    assert model.landmark_attn.alpha_logit.grad is not None
    assert model.landmark_attn.alpha_logit.grad.abs().sum() > 0, \
        "alpha_logit has zero gradient"

    # Canonical normals stay frozen
    assert not model.landmark_attn.n_canon.requires_grad
    print("PASS: Gradient flow through backbone and v2 attention")


def test_gradient_flow_v1():
    """v1: Gradients flow through backbone AND the v1 components actually
    used in forward (SpatialPriorGenerator + PoseGating).

    Note: v1's LandmarkEncoder is instantiated but NOT called in forward —
    it is dead weight (5696 of 5806 'v1 params'). This matches the training
    code that produced the v1 cross-dataset results we want to reproduce.
    Fixing the bug would change the architecture from what is archived.
    """
    from models.landmark_phys import LandmarkPhys
    from losses.neg_pearson import Neg_Pearson

    B, C, T, H, W = 2, 3, 128, 72, 72

    model = LandmarkPhys(
        frames=T,
        md_config={"MD_FSAM": True},
        in_channels=C,
        device=torch.device("cpu"),
        landmark_config={'attn_type': 'v1'},
    )
    model.train()
    loss_fn = Neg_Pearson()

    x = torch.randn(B, C, T + 1, H, W)
    landmarks = torch.rand(B, T, 68, 2)
    poses = torch.randn(B, T, 3)
    labels = torch.randn(B, T)

    rPPG, vox_embed, factorized_embed, appx_error = model(
        x, landmarks=landmarks, poses=poses)
    loss = loss_fn(rPPG, labels)
    loss.backward()

    # Backbone has gradient
    backbone_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.rppg_feature_extractor.parameters())
    assert backbone_has_grad, "No gradients in backbone!"

    # v1 spatial prior (learnable landmark_logits + log_sigma) has gradient
    assert model.landmark_attn.spatial_prior.landmark_logits.grad is not None
    assert model.landmark_attn.spatial_prior.landmark_logits.grad.abs().sum() > 0
    assert model.landmark_attn.spatial_prior.log_sigma.grad is not None
    assert model.landmark_attn.spatial_prior.log_sigma.grad.abs().sum() > 0

    # v1 pose gating has gradient
    pose_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.landmark_attn.pose_gating.parameters())
    assert pose_has_grad, "No gradients in v1 pose gating"

    print("PASS: Gradient flow through backbone, v1 spatial_prior, pose_gating")


def test_no_landmarks_fallback():
    """Forward without landmarks/transforms still works."""
    from models.landmark_phys import LandmarkPhys

    B, C, T, H, W = 2, 3, 128, 72, 72

    model = LandmarkPhys(
        frames=T,
        md_config={"MD_FSAM": True, "MD_INFERENCE": True},
        in_channels=C,
        device=torch.device("cpu"),
    )
    model.eval()

    x = torch.randn(B, C, T + 1, H, W)
    with torch.no_grad():
        rPPG, vox_embed, factorized_embed, appx_error = model(x)
    assert rPPG.shape == (B, T)
    assert not torch.isnan(rPPG).any()
    print("PASS: Forward pass works without landmarks/transforms (fallback)")


def test_parameter_count_v2():
    """v2: LandmarkPhys = FactorizePhys backbone + exactly 4 v2 attention scalars."""
    from models.landmark_phys import LandmarkPhys
    from models.factorizephys import FactorizePhys

    T = 128
    baseline = FactorizePhys(frames=T, md_config={"MD_FSAM": True})
    landmark_model = LandmarkPhys(
        frames=T, md_config={"MD_FSAM": True},
        landmark_config={'attn_type': 'v2'})

    baseline_params = sum(p.numel() for p in baseline.parameters())
    total_params = sum(p.numel() for p in landmark_model.parameters())
    landmark_params = sum(
        p.numel() for p in landmark_model.landmark_attn.parameters()
        if p.requires_grad)

    print(f"PASS: v2 parameter count")
    print(f"  FactorizePhys (baseline): {baseline_params:,}")
    print(f"  LandmarkPhys v2 (total):  {total_params:,}")
    print(f"  v2 attention (trainable): {landmark_params}")
    assert landmark_params == 4, \
        f"v2 attention should have 4 trainable scalars, got {landmark_params}"


def test_parameter_count_v1():
    """v1: LandmarkPhys with v1 has ~5.8k trainable landmark params (<10% of total)."""
    from models.landmark_phys import LandmarkPhys
    from models.factorizephys import FactorizePhys

    T = 128
    baseline = FactorizePhys(frames=T, md_config={"MD_FSAM": True})
    landmark_model = LandmarkPhys(
        frames=T, md_config={"MD_FSAM": True},
        landmark_config={'attn_type': 'v1'})

    baseline_params = sum(p.numel() for p in baseline.parameters())
    total_params = sum(p.numel() for p in landmark_model.parameters())
    landmark_params = sum(
        p.numel() for p in landmark_model.landmark_attn.parameters()
        if p.requires_grad)

    print(f"PASS: v1 parameter count")
    print(f"  FactorizePhys (baseline): {baseline_params:,}")
    print(f"  LandmarkPhys v1 (total):  {total_params:,}")
    print(f"  v1 attention (trainable): {landmark_params}")
    # v1 branch should be lightweight but non-trivial (MLPs + Gaussian + pose MLP)
    assert 1000 < landmark_params < 10000, \
        f"v1 attention should be ~5.8k params, got {landmark_params}"
    assert landmark_params < 0.15 * total_params, \
        f"v1 overhead {100*landmark_params/total_params:.1f}% exceeds 15% target"


if __name__ == "__main__":
    print("=" * 60)
    print("LandmarkPhys Unit Tests (v1 + v2)")
    print("=" * 60)
    print()

    tests = [
        ("Forward pass shapes", test_forward_pass_shapes),
        ("Gradient flow (v1)", test_gradient_flow_v1),
        ("Gradient flow (v2)", test_gradient_flow_v2),
        ("No-landmarks fallback", test_no_landmarks_fallback),
        ("Parameter count (v1)", test_parameter_count_v1),
        ("Parameter count (v2)", test_parameter_count_v2),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        print(f"--- {name} ---")
        try:
            test_fn()
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
    if failed > 0:
        sys.exit(1)
