"""CanonicalPhys: Pose-Canonicalized FactorizePhys for rPPG.

Uses a differentiable 4-point homography on facial landmarks to warp
each raw frame to a canonical frontal-face coordinate system before the
rPPG backbone. The backbone always sees pose-invariant inputs, so its
features cannot depend on head orientation.

**Three warp modes** controlled by ``canonical_config``:

1. Per-frame warp (``stabilize_warp=False``, ``adaptive_stabilization=False``):
   each frame uses its own homography. Accurate per frame but creates
   warp-interpolation artifacts in the temporal diff signal.

2. Fully stabilized (``stabilize_warp=True``, default): all frames in a
   chunk share the temporal-median homography H_ref. No diff artifacts
   but can be inaccurate for high-motion frames.

3. Adaptive (``adaptive_stabilization=True``, recommended): per-chunk
   decision — use H_ref when within-chunk landmark motion is low (most
   chunks), fall back to per-frame H when motion exceeds
   ``motion_threshold``. Combines the strengths of both.

No learnable parameters in the canonicalization module. The backbone is
unchanged FactorizePhys.
"""

import torch
import torch.nn as nn

from models.factorizephys import (
    rPPG_FeatureExtractor,
    BVP_Head,
    _default_md_config,
)
from models.canonical_warp import CanonicalWarp, ANCHOR_INDICES
from models.physics_weight import PhysicsWeight


class CanonicalPhys(nn.Module):
    """Pose-canonicalized rPPG model.

    Forward inputs match LandmarkPhys for trainer compatibility.
    The `poses` and `transforms` kwargs are accepted but unused — v3 only
    needs the 2D landmarks to compute the canonicalization homography.

    Args:
        frames: Number of output frames (T). Input should be T+1 frames.
        md_config: FSAM configuration dict.
        in_channels: Number of input channels.
        dropout: Dropout rate for backbone Dropout3d layers.
        device: torch.device for FSAM internal buffers.
        canonical_config: Optional dict of canonicalization settings.
            Keys: 'padding_mode' (default 'zeros').
    """

    def __init__(self, frames=128, md_config=None, in_channels=3,
                 dropout=0.1, device=torch.device("cpu"),
                 canonical_config=None):
        super().__init__()

        if md_config is None:
            md_config = {}
        for key in _default_md_config:
            if key not in md_config:
                md_config[key] = _default_md_config[key]

        if canonical_config is None:
            canonical_config = {}

        self.in_channels = in_channels
        self.use_fsam = md_config["MD_FSAM"]
        self.md_infer = md_config["MD_INFERENCE"]
        self.stabilize_warp = canonical_config.get('stabilize_warp', True)
        self.adaptive_stabilization = canonical_config.get(
            'adaptive_stabilization', False)
        self.motion_threshold = float(
            canonical_config.get('motion_threshold', 0.08))
        self.use_physics_weight = canonical_config.get(
            'physics_weight', False)

        # Register anchor indices as a buffer for motion computation
        self.register_buffer(
            '_anchor_idx',
            torch.tensor(ANCHOR_INDICES, dtype=torch.long),
            persistent=False)

        self.canonical_warp = CanonicalWarp(
            padding_mode=canonical_config.get('padding_mode', 'zeros'))

        if self.use_physics_weight:
            self.physics_weight = PhysicsWeight(
                canonical_normals_path=canonical_config.get(
                    'canonical_normals_path',
                    'weights/mediapipe/canonical_normals_dlib68.npy'),
                canonical_landmarks_2d_path=canonical_config.get(
                    'canonical_landmarks_2d_path',
                    'weights/mediapipe/canonical_landmarks_2d.npy'),
                spatial_size=72,
                sharpness=float(canonical_config.get(
                    'physics_sharpness', 8.0)),
                bias=float(canonical_config.get(
                    'physics_bias', 0.55)),
            )

        # FactorizePhys backbone (unchanged)
        self.norm = nn.InstanceNorm3d(self.in_channels)
        self.rppg_feature_extractor = rPPG_FeatureExtractor(
            self.in_channels, dropout_rate=dropout)
        self.rppg_head = BVP_Head(
            md_config, device=device, dropout_rate=dropout)

    def forward(self, x, landmarks=None, poses=None, valid_mask=None,
                transforms=None, homographies=None):
        """Forward pass.

        Args:
            x: (B, C, T+1, H, W) input video tensor.
            landmarks: (B, T, 68, 2) normalized landmark coordinates in
                [0, 1]. If None, the canonicalization is skipped and
                the model reduces to plain FactorizePhys.
            poses: Unused. Accepted for API compatibility.
            valid_mask: (B, T) float in {0, 1}. Invalid frames pass
                through without warping.
            transforms: (B, T, 4, 4) MediaPipe facial transforms. Used
                by the physics-weight module when enabled.
            homographies: (B, T, 3, 3) precomputed homographies (faster
                than recomputing from landmarks).

        Returns:
            (rPPG, voxel_embeddings[, factorized_embeddings, appx_error])
            matching FactorizePhys's signature.
        """
        B, C, T_plus_1, H, W = x.shape

        self._canonical_frames = None

        # Canonicalize raw pixels before the backbone.
        if landmarks is not None:
            # Pad landmarks (and valid_mask) from T to T+1 by repeating
            # the last slice; the trainer appends a duplicate last frame
            # so frame T+1 has the same content as frame T.
            T = landmarks.shape[1]
            assert T + 1 == T_plus_1, \
                f"Landmark T={T} must match frames T+1={T_plus_1}"
            landmarks_padded = torch.cat(
                [landmarks, landmarks[:, -1:]], dim=1)  # (B, T+1, 68, 2)

            if valid_mask is not None:
                valid_padded = torch.cat(
                    [valid_mask, valid_mask[:, -1:]], dim=1)  # (B, T+1)
            else:
                valid_padded = None

            # Pad homographies if precomputed
            hom_padded = None
            if homographies is not None:
                hom_padded = torch.cat(
                    [homographies, homographies[:, -1:]], dim=1)

                if self.adaptive_stabilization or self.stabilize_warp:
                    # Compute H_ref = temporal median of per-frame H
                    H_ref = hom_padded.median(dim=1, keepdim=True).values
                    # Normalize so h33 = 1 (homography scale convention)
                    H_ref = H_ref / (H_ref[:, :, 2:3, 2:3] + 1e-8)
                    H_ref_expanded = H_ref.expand_as(hom_padded)

                    if self.adaptive_stabilization:
                        # Per-chunk motion score from anchor landmark std.
                        # anchors: (B, T+1, 4, 2) → std across time
                        anchors = landmarks_padded.index_select(
                            2, self._anchor_idx)              # (B, T+1, 4, 2)
                        motion = anchors.std(dim=1).mean(
                            dim=(-1, -2))                      # (B,)
                        use_stable = (motion < self.motion_threshold).view(
                            -1, 1, 1, 1)                       # (B, 1, 1, 1)
                        # Select per-sample: H_ref if low motion, else per-frame
                        hom_padded = torch.where(
                            use_stable, H_ref_expanded, hom_padded)
                    else:
                        # Always stabilized (legacy mode)
                        hom_padded = H_ref_expanded

            x = self.canonical_warp(x, landmarks_padded,
                                    valid_mask=valid_padded,
                                    homographies=hom_padded)

            # Save canonical frames before physics weight / diff for
            # POS distillation loss (Direction 3).
            if self.training:
                self._canonical_frames = x.detach()

            # Physics-informed per-pixel weighting in canonical space.
            # Multiplies canonical pixels by their diffuse-reflection
            # confidence derived from per-frame surface orientation.
            if self.use_physics_weight and transforms is not None:
                # Pad transforms to T+1 to match frames (same duplicate-last
                # convention as landmarks/valid_mask above).
                tf_padded = torch.cat(
                    [transforms, transforms[:, -1:]], dim=1)  # (B, T+1, 4, 4)
                # weight: (B, T+1, 72, 72); broadcast across channel dim
                weight = self.physics_weight(tf_padded).unsqueeze(1)  # (B, 1, T+1, H, W)
                x = x * weight

        # Standard FactorizePhys forward from here on.
        length = x.shape[2]
        x = torch.diff(x, dim=2)                               # (B, C, T, H, W)
        x = self.norm(x[:, :self.in_channels, :, :, :])
        voxel_embeddings = self.rppg_feature_extractor(x)      # (B, 16, T, 13, 13)

        if (self.md_infer or self.training) and self.use_fsam:
            rPPG, factorized_embeddings, appx_error = self.rppg_head(
                voxel_embeddings, B, length - 1)
            return rPPG, voxel_embeddings, factorized_embeddings, appx_error
        else:
            rPPG = self.rppg_head(voxel_embeddings, B, length - 1)
            return rPPG, voxel_embeddings
