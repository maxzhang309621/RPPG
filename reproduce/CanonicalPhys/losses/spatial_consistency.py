"""Cross-spatial temporal consistency loss for canonical-space rPPG.

Motivation
----------
In canonical face coordinates, the feature at spatial position
(0.5, 0.3) is forehead for EVERY subject and EVERY frame. A given
person has ONE heartbeat, so BVP signals extracted from different
canonical skin ROIs (forehead, left cheek, right cheek) must share the
same phase and frequency, differing only in amplitude.

This loss enforces that constraint on the BACKBONE'S spatial feature
maps. For each ROI r we compute a scalar-per-frame signal s_r[t] by
masking the (B, C, T, Hf, Wf) voxel features with a soft canonical
ROI mask, spatially averaging, and reducing the channel dim. After
temporal normalization we apply pairwise cosine-distance:

    L_consistency = Σ_{i<j} (1 − cosine_similarity(s_i, s_j))

This is a self-supervised auxiliary signal — it does not require
ground-truth BVP. Only expressible in canonical space (you need fixed
coordinates for "the forehead"). No prior rPPG method uses this.

Operating on backbone features (not raw pixels) is what makes the
gradient flow back into learnable parameters: the backbone learns to
produce spatially-consistent BVP features across canonical ROIs.
"""

import torch
import torch.nn as nn


def _normalize_temporal(signal):
    """Zero-mean, unit-std per-sample along the temporal axis.

    Args:
        signal: (B, T) tensor.
    Returns:
        (B, T) normalized.
    """
    mu = signal.mean(dim=-1, keepdim=True)
    std = signal.std(dim=-1, keepdim=True) + 1e-6
    return (signal - mu) / std


def roi_temporal_signal(canonical_frames, roi_mask, channel=1):
    """Extract a temporal signal from a canonical ROI.

    Args:
        canonical_frames: (B, C, T, H, W) canonical-space video.
        roi_mask: (H, W) binary or soft mask of the ROI (0/1 float).
        channel: which channel to use (default 1 = green, strongest rPPG).

    Returns:
        signal: (B, T) mean-intensity series over the ROI.
    """
    # (B, T, H, W) for the selected channel
    g = canonical_frames[:, channel]                           # (B, T, H, W)
    m = roi_mask.to(g.dtype)                                   # (H, W)
    # weighted mean over H, W
    num = (g * m).sum(dim=(-1, -2))                            # (B, T)
    den = m.sum() + 1e-6
    return num / den


def build_canonical_roi_masks(spatial_size=72):
    """Build three canonical skin ROI masks: forehead, left cheek, right
    cheek. Positions are hand-chosen to match our canonical anchor
    layout (eyes at y=0.40, mouth corners at y=0.72).

    Returns:
        masks: (3, H, W) float32 soft masks (Gaussian blobs in [0, 1]).
    """
    H = W = spatial_size
    # ROI centers in normalized [0, 1] coords and Gaussian sigmas
    rois = [
        ('forehead',    0.50, 0.20, 0.10),
        ('left_cheek',  0.28, 0.58, 0.07),
        ('right_cheek', 0.72, 0.58, 0.07),
    ]
    ys = torch.linspace(0.0, 1.0, H)
    xs = torch.linspace(0.0, 1.0, W)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    masks = []
    for _, cx, cy, sigma in rois:
        d2 = (grid_x - cx) ** 2 + (grid_y - cy) ** 2
        m = torch.exp(-d2 / (2 * sigma ** 2))
        masks.append(m)
    return torch.stack(masks, dim=0).float()


class SpatialConsistencyLoss(nn.Module):
    """Cross-ROI temporal consistency loss on backbone feature maps.

    Args:
        feature_size: backbone feature-map side length (default 13 for
            FactorizePhys after 3D-CNN downsampling of a 72x72 input).
        temporal_diff: if True, apply `diff` along time first to
            emphasize rPPG frequencies and suppress DC/slow drifts.
    """

    def __init__(self, feature_size=13, temporal_diff=True):
        super().__init__()
        self.temporal_diff = bool(temporal_diff)
        masks = build_canonical_roi_masks(feature_size)      # (3, H, W)
        self.register_buffer('roi_masks', masks, persistent=False)

    def forward(self, voxel_embeddings, valid_mask=None):
        """
        Args:
            voxel_embeddings: (B, C, T, Hf, Wf) backbone features.
                The spatial axes are assumed to be in canonical
                coordinates (CanonicalPhys canonicalizes pixels, so the
                backbone's spatial axes inherit canonical semantics).
            valid_mask: (B, T) unused for now (kept for API parity).

        Returns:
            scalar loss = mean over pairs of (1 − cosine_similarity).
        """
        B, C, T, Hf, Wf = voxel_embeddings.shape
        assert self.roi_masks.shape[-1] == Wf, \
            f"ROI mask size {self.roi_masks.shape[-1]} != feature size {Wf}"

        signals = []
        for k in range(self.roi_masks.shape[0]):
            mask = self.roi_masks[k].to(voxel_embeddings.dtype)   # (Hf, Wf)
            # Weighted spatial mean: (B, C, T)
            num = (voxel_embeddings * mask).sum(dim=(-1, -2))
            den = mask.sum() + 1e-6
            roi_feat = num / den                                   # (B, C, T)
            # Reduce channel dim via L2 norm → (B, T)
            s = roi_feat.pow(2).sum(dim=1).clamp_min(1e-12).sqrt()
            if self.temporal_diff:
                s = s[:, 1:] - s[:, :-1]                           # (B, T-1)
            s = _normalize_temporal(s)
            signals.append(s)

        K = len(signals)
        loss = 0.0
        pairs = 0
        for i in range(K):
            for j in range(i + 1, K):
                cos = (signals[i] * signals[j]).mean(dim=-1)       # (B,)
                loss = loss + (1.0 - cos).mean()
                pairs += 1
        return loss / max(pairs, 1)
