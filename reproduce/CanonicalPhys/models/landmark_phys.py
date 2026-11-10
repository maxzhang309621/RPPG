"""LandmarkPhys: Landmark-Guided Attention for Robust rPPG.

Integrates a landmark-based attention branch with the FactorizePhys 3D-CNN
backbone. Two attention variants are supported via ``landmark_config['attn_type']``:

    - ``'v1'`` (default): LandmarkAttn — Gaussian spatial prior + pose-conditioned
      gating. Aligns with the paper's stated contribution (pose-robust rPPG).
    - ``'v2'``: VisibilityMaskAttn — deterministic 3D visibility mask from
      MediaPipe facial transforms. Kept for ablation / comparison.

Architecture:
    Raw Frames -> Diff -> InstanceNorm -> 3D-CNN Backbone -> voxel_embeddings
                                                              |
    Landmarks (+poses|+transforms) -> LandmarkAttn|VisibilityMaskAttn
                                                              v
                                              voxel_embeddings_enhanced
                                                              |
                                                       BVP_Head (FSAM) -> rPPG

The landmark branch is complementary to FSAM: it provides geometric priors
while FSAM provides data-driven factorization-based attention.
"""

import torch
import torch.nn as nn

from models.factorizephys import (
    rPPG_FeatureExtractor,
    BVP_Head,
    _default_md_config,
)
from models.landmark_attn import LandmarkAttn
from models.visibility_mask_attn import VisibilityMaskAttn


class LandmarkPhys(nn.Module):
    """LandmarkPhys model.

    Reuses the FactorizePhys backbone (rPPG_FeatureExtractor + BVP_Head)
    and inserts a LandmarkAttn module between feature extraction and
    the BVP prediction head.

    Args:
        frames: Number of output frames (T). Input should be T+1 frames.
        md_config: Dict of FSAM / matrix decomposition configuration.
        in_channels: Number of input channels (default 3 for RGB).
        dropout: Dropout rate for Dropout3d layers.
        device: torch.device for FSAM internal buffers.
        landmark_config: Dict of landmark branch configuration.
    """

    def __init__(self, frames=128, md_config=None, in_channels=3,
                 dropout=0.1, device=torch.device("cpu"),
                 landmark_config=None):
        super().__init__()

        if md_config is None:
            md_config = {}
        for key in _default_md_config:
            if key not in md_config:
                md_config[key] = _default_md_config[key]

        if landmark_config is None:
            landmark_config = {}

        self.in_channels = in_channels
        self.use_fsam = md_config["MD_FSAM"]
        self.md_infer = md_config["MD_INFERENCE"]

        # Attention variant: 'v1' (LandmarkAttn) or 'v2' (VisibilityMaskAttn).
        # Default 'v1' — aligns with the paper's pose-robust contribution and
        # is the best-performing variant on cross-dataset benchmarks.
        self.attn_type = landmark_config.get('attn_type', 'v1')

        # Shared backbone components (same as FactorizePhys)
        self.norm = nn.InstanceNorm3d(self.in_channels)
        self.rppg_feature_extractor = rPPG_FeatureExtractor(
            self.in_channels, dropout_rate=dropout)
        self.rppg_head = BVP_Head(
            md_config, device=device, dropout_rate=dropout)

        if self.attn_type == 'v1':
            # LandmarkAttn: Gaussian spatial prior + pose gating (v1)
            self.landmark_attn = LandmarkAttn(
                num_landmarks=landmark_config.get('num_landmarks', 68),
                landmark_dim=landmark_config.get('landmark_dim', 2),
                spatial_size=landmark_config.get('spatial_size', 13),
                embed_channels=landmark_config.get('embed_channels', 16),
                hidden_dim=landmark_config.get('hidden_dim', 32),
                pose_dim=landmark_config.get('pose_dim', 3),
                pose_hidden=landmark_config.get('pose_hidden', 8),
                sigma_init=landmark_config.get('sigma_init', 0.1),
                temporal_kernel=landmark_config.get('temporal_kernel', 3),
                use_spatial_prior=landmark_config.get('use_spatial_prior', True),
                use_pose_gating=landmark_config.get('use_pose_gating', True),
                use_temporal_conv=landmark_config.get('use_temporal_conv', True),
                landmark_noise_std=landmark_config.get('landmark_noise_std', 0.0),
                landmark_dropout_rate=landmark_config.get('landmark_dropout_rate', 0.0),
            )
        elif self.attn_type == 'v2':
            # VisibilityMaskAttn: deterministic 3D visibility mask (v2)
            self.landmark_attn = VisibilityMaskAttn(
                canonical_normals_path=landmark_config.get(
                    'canonical_normals_path',
                    'weights/mediapipe/canonical_normals_dlib68.npy'),
                spatial_size=landmark_config.get('spatial_size', 13),
                alpha_init=landmark_config.get('alpha_init', 0.3),
                visibility_sharpness=landmark_config.get(
                    'visibility_sharpness', 8.0),
                visibility_bias=landmark_config.get('visibility_bias', 0.55),
                raster_temp=landmark_config.get('raster_temp', 50.0),
                mask_dropout=landmark_config.get('mask_dropout', 0.0),
            )
        else:
            raise ValueError(
                f"Unknown attn_type: {self.attn_type!r}. Expected 'v1' or 'v2'.")

    def forward(self, x, landmarks=None, poses=None, valid_mask=None,
                transforms=None, homographies=None):
        """Forward pass.

        Args:
            x: (B, C, T+1, H, W) input video tensor.
            landmarks: (B, T, K, 2) normalized landmark coordinates, or None.
            poses: (B, T, 3) head pose angles. Kept in the API for backward
                compatibility but not consumed by VisibilityMaskAttn (the v2
                module derives pose-dependence from `transforms` instead).
            valid_mask: (B, T) float in {0, 1}, or None. Invalid frames see
                backbone-only behaviour (skin mask collapses to all-ones).
            transforms: (B, T, 4, 4) MediaPipe facial transformation matrix.
                Required for the visibility-aware attention to activate.

        Returns:
            If FSAM active (training or md_infer):
                (rPPG, voxel_embeddings, factorized_embeddings, appx_error)
            Else:
                (rPPG, voxel_embeddings)
            where rPPG has shape (B, T).
        """
        [batch, channel, length, width, height] = x.shape

        # Temporal differencing: T+1 -> T
        x = torch.diff(x, dim=2)

        # Instance normalization
        x = self.norm(x[:, :self.in_channels, :, :, :])

        # Feature extraction -> voxel embeddings (B, 16, T, 13, 13)
        voxel_embeddings = self.rppg_feature_extractor(x)

        # Apply landmark-guided attention.
        # v1 uses poses, v2 uses transforms. Landmarks are T frames; after
        # torch.diff the backbone temporal dim matches.
        if landmarks is not None:
            if self.attn_type == 'v1' and poses is not None:
                voxel_embeddings = self.landmark_attn(
                    voxel_embeddings, landmarks, poses,
                    valid_mask=valid_mask)
            elif self.attn_type == 'v2' and transforms is not None:
                voxel_embeddings = self.landmark_attn(
                    voxel_embeddings, landmarks, transforms,
                    valid_mask=valid_mask)

        # BVP prediction head (includes FSAM)
        if (self.md_infer or self.training) and self.use_fsam:
            rPPG, factorized_embeddings, appx_error = self.rppg_head(
                voxel_embeddings, batch, length - 1)
            return rPPG, voxel_embeddings, factorized_embeddings, appx_error
        else:
            rPPG = self.rppg_head(voxel_embeddings, batch, length - 1)
            return rPPG, voxel_embeddings
