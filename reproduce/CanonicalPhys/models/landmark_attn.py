"""Landmark-Guided Attention Module (LandmarkAttn).

Provides geometry-aware spatial attention for rPPG feature enhancement.
Landmarks encode explicit knowledge about visible skin regions and head pose;
this module transforms them into spatial-temporal attention priors that
modulate backbone voxel embeddings.

Components:
    LandmarkEncoder: MLP + TemporalConv encodes per-frame landmarks.
    SpatialPriorGenerator: Gaussian-weighted soft spatial mask from landmarks.
    PoseGating: Pose-conditioned adaptive fusion strength.
    LandmarkAttn: Combines all three to modulate backbone features.
"""

import torch
import torch.nn as nn


class LandmarkEncoder(nn.Module):
    """Encodes facial landmark sequences into temporal features.

    Architecture (Option A from plan): per-frame MLP + temporal 1D convolution.

    Args:
        num_landmarks: Number of landmarks per frame (default 68 for dlib).
        landmark_dim: Coordinate dimension per landmark (default 2 for x,y).
        hidden_dim: MLP hidden layer size.
        out_dim: Output feature dimension per frame.
        temporal_kernel: Kernel size for temporal convolution.
    """

    def __init__(self, num_landmarks=68, landmark_dim=2, hidden_dim=32,
                 out_dim=16, temporal_kernel=3, use_temporal_conv=True):
        super().__init__()
        self.use_temporal_conv = use_temporal_conv
        in_dim = num_landmarks * landmark_dim  # 68*2 = 136
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(inplace=True),
        )
        if use_temporal_conv:
            pad = temporal_kernel // 2
            self.temporal_conv = nn.Conv1d(
                out_dim, out_dim, kernel_size=temporal_kernel, padding=pad)

    def forward(self, landmarks):
        """
        Args:
            landmarks: (B, T, K, D) normalized landmark coordinates in [0,1].

        Returns:
            features: (B, T, out_dim) temporal landmark features.
        """
        B, T, K, D = landmarks.shape
        x = landmarks.reshape(B, T, K * D)  # (B, T, 136)
        x = self.mlp(x)                     # (B, T, out_dim)
        if self.use_temporal_conv:
            x = x.permute(0, 2, 1)          # (B, out_dim, T)
            x = self.temporal_conv(x)        # (B, out_dim, T)
            x = x.permute(0, 2, 1)          # (B, T, out_dim)
        return x


class SpatialPriorGenerator(nn.Module):
    """Generates soft spatial attention maps from landmark positions.

    For each frame, creates a Gaussian-weighted skin probability map:
        S(x,y) = sum_k w_k * exp(-||[x,y] - l_k||^2 / (2 * sigma^2))
    where w_k are learned per-landmark weights and sigma is a learned bandwidth.

    The map naturally adapts to pose: under side-face, occluded landmarks
    shift position, and the map concentrates on visible skin regions.

    Args:
        num_landmarks: Number of landmarks (68).
        spatial_size: Output spatial resolution (must match voxel embeddings).
        sigma_init: Initial Gaussian bandwidth in normalized [0,1] space.
    """

    def __init__(self, num_landmarks=68, spatial_size=13, sigma_init=0.1):
        super().__init__()
        self.spatial_size = spatial_size

        # Learnable per-landmark importance weights (unnormalized logits)
        self.landmark_logits = nn.Parameter(torch.zeros(num_landmarks))

        # Learnable bandwidth (log-space for positivity)
        self.log_sigma = nn.Parameter(torch.tensor(sigma_init).log())

        # Pre-compute spatial grid coordinates (normalized to [0,1])
        grid = torch.linspace(0, 1, spatial_size)
        grid_y, grid_x = torch.meshgrid(grid, grid, indexing='ij')
        # (H, W, 2) grid of (x, y) coordinates
        self.register_buffer(
            'grid', torch.stack([grid_x, grid_y], dim=-1))

    def forward(self, landmarks):
        """
        Args:
            landmarks: (B, T, K, 2) normalized coordinates in [0,1].

        Returns:
            spatial_prior: (B, T, H, W) soft attention map.
        """
        B, T, K, _ = landmarks.shape
        H = W = self.spatial_size
        sigma = self.log_sigma.exp()

        # Expand for broadcasting:
        #   landmarks: (B, T, K, 1, 1, 2)
        #   grid:      (1, 1, 1, H, W, 2)
        lm = landmarks.unsqueeze(-2).unsqueeze(-2)  # (B, T, K, 1, 1, 2)
        g = self.grid.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, H, W, 2)

        # Squared distance: (B, T, K, H, W)
        dist_sq = ((g - lm) ** 2).sum(dim=-1)

        # Gaussian: (B, T, K, H, W)
        gaussians = torch.exp(-dist_sq / (2 * sigma ** 2))

        # Weighted sum over landmarks
        weights = self.landmark_logits.softmax(dim=0)  # (K,)
        weights = weights.view(1, 1, K, 1, 1)
        spatial_prior = (weights * gaussians).sum(dim=2)  # (B, T, H, W)

        return spatial_prior


class PoseGating(nn.Module):
    """Pose-conditioned adaptive gating.

    Computes gamma in [0,1] that controls how much landmark spatial priors
    influence backbone features. Intuition: for frontal faces gamma -> 0
    (backbone works well alone); for side faces gamma -> 1 (landmark
    guidance is critical).

    Args:
        pose_dim: Input pose dimension (3 for yaw/pitch/roll).
        hidden_dim: Hidden layer size.
    """

    def __init__(self, pose_dim=3, hidden_dim=8):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, poses):
        """
        Args:
            poses: (B, T, 3) head pose angles [yaw, pitch, roll] in degrees.

        Returns:
            gamma: (B, T, 1) gating value in [0, 1].
        """
        return self.gate(poses)


class LandmarkAttn(nn.Module):
    """Landmark-Guided Attention Module.

    Combines landmark-derived spatial priors with pose-conditioned gating
    to modulate backbone voxel embeddings:

        S = SpatialPriorGenerator(landmarks)        # (B, T, H, W)
        gamma = PoseGating(poses)                    # (B, T, 1)
        epsilon_out = epsilon * (1 + gamma * S)      # soft modulation

    The formulation ensures that the prior only enhances, never fully
    suppresses features (additive modulation starting from 1).

    Ablation flags (all default True for the full model):
        use_spatial_prior: If False, uses uniform spatial map (no landmark geometry).
        use_pose_gating: If False, uses fixed gamma=0.5 (no pose adaptation).
        use_temporal_conv: If False, skips temporal conv in encoder (per-frame only).
        landmark_noise_std: Additive Gaussian noise std on landmarks (0 = off).
        landmark_dropout_rate: Fraction of landmarks zeroed per frame (0 = off).
    """

    def __init__(self, num_landmarks=68, landmark_dim=2, spatial_size=13,
                 embed_channels=16, hidden_dim=32, pose_dim=3,
                 pose_hidden=8, sigma_init=0.1, temporal_kernel=3,
                 use_spatial_prior=True, use_pose_gating=True,
                 use_temporal_conv=True,
                 landmark_noise_std=0.0, landmark_dropout_rate=0.0):
        super().__init__()

        self.use_spatial_prior = use_spatial_prior
        self.use_pose_gating = use_pose_gating
        self.use_temporal_conv = use_temporal_conv
        self.landmark_noise_std = landmark_noise_std
        self.landmark_dropout_rate = landmark_dropout_rate

        self.encoder = LandmarkEncoder(
            num_landmarks=num_landmarks,
            landmark_dim=landmark_dim,
            hidden_dim=hidden_dim,
            out_dim=embed_channels,
            temporal_kernel=temporal_kernel,
            use_temporal_conv=use_temporal_conv,
        )

        if use_spatial_prior:
            self.spatial_prior = SpatialPriorGenerator(
                num_landmarks=num_landmarks,
                spatial_size=spatial_size,
                sigma_init=sigma_init,
            )

        if use_pose_gating:
            self.pose_gating = PoseGating(
                pose_dim=pose_dim,
                hidden_dim=pose_hidden,
            )

    def forward(self, voxel_embeddings, landmarks, poses, valid_mask=None):
        """Apply landmark-guided attention to backbone features.

        Args:
            voxel_embeddings: (B, C, T, H, W) backbone features.
            landmarks: (B, T, K, 2) normalized landmark coordinates.
            poses: (B, T, 3) head pose angles.
            valid_mask: (B, T) float in {0, 1}. When provided, gamma is
                multiplied by the mask so invalid frames see fusion
                epsilon * (1 + 0) = epsilon (backbone-only behaviour).
                If None (default), all frames are treated as valid.

        Returns:
            enhanced: (B, C, T, H, W) landmark-guided features.
        """
        # Landmark noise injection (for robustness ablation, Task 5.4)
        if self.landmark_noise_std > 0:
            noise = torch.randn_like(landmarks) * self.landmark_noise_std
            landmarks = (landmarks + noise).clamp(0.0, 1.0)

        # Landmark dropout (for robustness ablation, Task 5.4)
        if self.landmark_dropout_rate > 0 and self.training:
            mask = torch.rand(
                landmarks.shape[0], landmarks.shape[1], landmarks.shape[2], 1,
                device=landmarks.device) > self.landmark_dropout_rate
            landmarks = landmarks * mask.float()

        # Spatial prior: (B, T, H, W) or uniform 1.0
        if self.use_spatial_prior:
            S = self.spatial_prior(landmarks)
        else:
            B, T = landmarks.shape[:2]
            H = W = voxel_embeddings.shape[-1]
            S = torch.ones(B, T, H, W, device=landmarks.device)

        # Pose gating: (B, T, 1) or fixed 0.5
        if self.use_pose_gating:
            gamma = self.pose_gating(poses)
        else:
            B, T = poses.shape[:2]
            gamma = torch.full((B, T, 1), 0.5, device=poses.device)

        # Mask invalid frames so the prior collapses to identity there
        if valid_mask is not None:
            # valid_mask: (B, T) -> (B, T, 1)
            gamma = gamma * valid_mask.unsqueeze(-1).to(gamma.dtype)

        # Modulate: epsilon * (1 + gamma * S)
        gamma = gamma.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)  # (B, 1, T, 1, 1)
        S = S.unsqueeze(1)                                           # (B, 1, T, H, W)

        enhanced = voxel_embeddings * (1.0 + gamma * S)

        return enhanced
